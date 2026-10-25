"""Replacement experiment: base runs, checkpoint branching, baselines.

Commands
  python experiment.py ping    --models models.json
  python experiment.py base    --out runs/main --n 15 --rounds 60
  python experiment.py branch  --out runs/main --rounds 60 --arms A0,A1,A5,A6
  python experiment.py base    --out runs/b0 --n 5 --rounds 60 --prefix competitive
  python experiment.py smoke   (offline test with a scripted mock agent)

Every command can be re-run: finished games are skipped, so a crash only costs
the games that were in flight.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from agents import build_prompt, call_with_retries, make_client
from market import Market, benchmarks, collusion_index, profit_index

# ------------------------------------------------------------------ arms
# Seller A is the incumbent, seller B is the one replaced.
# LLM calls are stateless, so an "agent" here is exactly: model weights + notes.
# A "fresh instance of the same model" therefore means: same model, notes wiped.
ARMS = {
    "A0": {"desc": "No swap (control)",
           "A": ("M", True), "B": ("M", True), "history": True},
    "A1": {"desc": "B replaced by a fresh instance of the same model",
           "A": ("M", True), "B": ("M", False), "history": True},
    "A2": {"desc": "B replaced by another family, inheriting B's notes (handover)",
           "A": ("M", True), "B": ("P", True), "history": True},
    "A3": {"desc": "B replaced by a same-family model of another size",
           "A": ("M", True), "B": ("Mprime", False), "history": True},
    "A4": {"desc": "B replaced by another model family",
           "A": ("M", True), "B": ("P", False), "history": True},
    "A5": {"desc": "Both agents' notes wiped, public history kept",
           "A": ("M", False), "B": ("M", False), "history": True},
    "A6": {"desc": "Clean slate: both notes and the history wiped",
           "A": ("M", False), "B": ("M", False), "history": False},
    "A7": {"desc": "Retention limit: both notes wiped, only the last 3 rounds ever shown",
           "A": ("M", False), "B": ("M", False), "history": True, "window": 3},
    "A8": {"desc": "Rival prices hidden: both notes wiped, record shows own outcomes only",
           "A": ("M", False), "B": ("M", False), "history": True, "hide_rival": True},
}


# ------------------------------------------------------------------ helpers

def load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def seed_for(*parts) -> int:
    """Deterministic per-call seed. It ignores the arm, so all arms branching from
    one base run share random numbers round by round (common random numbers)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def visible_history(state: dict, i: int, window: int) -> list[dict]:
    rows = [h for h in state["history"] if h["round"] >= state["visible_from"]]
    rows = rows[-window:] if window > 0 else []
    j = 1 - i
    return [{"round": h["round"], "own_price": h["prices"][i], "comp_price": h["prices"][j],
             "own_qty": h["qty"][i], "own_profit": h["profit"][i]} for h in rows]


def new_state(run_id: str, seed: int, prefix: str, market: Market) -> dict:
    return {
        "run_id": run_id, "seed": seed, "prefix": prefix,
        "market": market.to_dict(), "bench": benchmarks(market),
        "round": 1, "visible_from": 1,
        "sellers": [{"name": "A", "model": "M", "notes": {}},
                    {"name": "B", "model": "M", "notes": {}}],
        "history": [], "arm": None, "rep": 0, "swap_round": None,
    }


def apply_arm(base: dict, arm: str, rep: int) -> dict:
    spec = ARMS[arm]
    s = copy.deepcopy(base)
    s["arm"], s["rep"], s["swap_round"] = arm, rep, s["round"]
    for idx, key in ((0, "A"), (1, "B")):
        model, keep_notes = spec[key]
        s["sellers"][idx]["model"] = model
        if not keep_notes:
            s["sellers"][idx]["notes"] = {}
    if not spec["history"]:
        s["visible_from"] = s["round"]
    s["window"] = spec.get("window")
    s["hide_rival"] = spec.get("hide_rival", False)
    return s


# ------------------------------------------------------------------ the game

async def play(state: dict, n_rounds: int, clients: dict, cfg, sem, log_path: Path) -> dict:
    market = Market.from_dict(state["market"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        for _ in range(n_rounds):
            t = state["round"]
            calls, prompts = [], []
            for i, seller in enumerate(state["sellers"]):
                hist = visible_history(state, i, state.get("window") or cfg.history)
                prompt = build_prompt(state["prefix"], t, market.cost(i), hist, seller["notes"],
                                      hide_rival=state.get("hide_rival", False))
                prompts.append(prompt)
                ctx = {"alpha": market.alpha, "cost": market.cost(i), "history": hist}
                seed = seed_for(state["run_id"], state["rep"], t, i)
                calls.append(call_with_retries(clients[seller["model"]], prompt, ctx, seed=seed,
                                               temperature=cfg.temperature,
                                               max_tokens=cfg.max_tokens, sem=sem))
            results = await asyncio.gather(*calls)

            prices, ok = [], []
            for i, (raw, parsed, _) in enumerate(results):
                if parsed.price is None:  # fall back to last own price, else Nash
                    prev = state["history"][-1]["prices"][i] if state["history"] else \
                        state["bench"]["p_nash"][i]
                    prices.append(prev)
                    ok.append(False)
                else:
                    lo, hi = 0.0, market.cost(i) + 10 * market.alpha
                    prices.append(min(max(parsed.price, lo), hi))
                    ok.append(True)
                    state["sellers"][i]["notes"] = {"plans": parsed.plans,
                                                    "insights": parsed.insights}
            qty, profit = market.quantities(prices), market.profits(prices)
            state["history"].append({
                "round": t, "prices": prices, "qty": qty, "profit": profit, "parse_ok": ok,
                "models": [s["model"] for s in state["sellers"]],
                "delta": profit_index(profit, state["bench"]),
                "delta_price": collusion_index(prices, state["bench"]),
            })
            for i, (raw, parsed, n_try) in enumerate(results):
                log.write(json.dumps({
                    "run_id": state["run_id"], "arm": state["arm"], "rep": state["rep"],
                    "round": t, "seller": state["sellers"][i]["name"],
                    "model": state["sellers"][i]["model"], "price": prices[i],
                    "parse_ok": ok[i], "attempts": n_try, "response": raw,
                    "prompt": prompts[i],
                }) + "\n")
            log.flush()
            state["round"] += 1
    return state


# ------------------------------------------------------------------ commands

def load_models(path: str) -> dict:
    return json.loads(Path(path).read_text())


def build_clients(model_specs: dict, keys) -> dict:
    missing = [k for k in keys if k not in model_specs]
    if missing:
        raise SystemExit(f"models file has no entry for {missing}")
    return {k: make_client(model_specs[k]) for k in keys}


def record_args(out: Path, cmd: str, args) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run_args.jsonl").open("a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "cmd": cmd,
                            **{k: v for k, v in vars(args).items() if k != "func"}}) + "\n")


def pre_swap_delta(state: dict, window: int) -> float:
    rows = state["history"][-window:]
    return sum(profit_index(h["profit"], state["bench"]) for h in rows) / len(rows)


async def cmd_base(args) -> None:
    out = Path(args.out)
    record_args(out, "base", args)
    specs = load_models(args.models)
    clients = build_clients(specs, ["M"])
    sem = asyncio.Semaphore(args.concurrency)
    alphas = [float(a) for a in args.alphas.split(",")]

    jobs = []
    for k in range(args.n):
        seed = args.seed0 + k
        run_id = f"{out.name}-s{seed}"
        ck = out / "checkpoints" / f"{run_id}.json"
        if ck.exists():
            continue
        market = Market(alpha=alphas[k % len(alphas)])
        state = new_state(run_id, seed, args.prefix, market)

        async def job(state=state, ck=ck):
            done = await play(state, args.rounds, clients, args, sem,
                              out / "logs" / f"{state['run_id']}__base.jsonl")
            write_json(ck, done)
            d = pre_swap_delta(done, args.window)
            print(f"  {done['run_id']}  alpha={done['market']['alpha']}  "
                  f"Δ(last {args.window}) = {d:+.2f}  {'PASS' if d >= args.threshold else 'fail'}",
                  flush=True)
        jobs.append(job())
    print(f"Running {len(jobs)} base games of {args.rounds} rounds "
          f"({len(jobs) * args.rounds * 2} model calls)...", flush=True)
    t0 = time.time()
    await asyncio.gather(*jobs)
    print(f"Done in {time.time() - t0:.0f}s. Checkpoints in {out / 'checkpoints'}")


async def cmd_branch(args) -> None:
    out = Path(args.out)
    record_args(out, "branch", args)
    arms = args.arms.split(",")
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"Unknown arm {a}. Choose from {list(ARMS)}")
    needed = sorted({ARMS[a][s][0] for a in arms for s in ("A", "B")})
    clients = build_clients(load_models(args.models), needed)
    sem = asyncio.Semaphore(args.concurrency)

    bases = [json.loads(p.read_text()) for p in sorted((out / "checkpoints").glob("*.json"))]
    if not bases:
        raise SystemExit(f"No checkpoints in {out / 'checkpoints'}. Run `base` first.")
    included = [b for b in bases if args.all or pre_swap_delta(b, args.window) >= args.threshold]
    print(f"{len(included)}/{len(bases)} base runs pass the inclusion rule "
          f"(mean Δ over the last {args.window} rounds ≥ {args.threshold}).")

    jobs = []
    for b in included:
        for arm in arms:
            for rep in range(args.reps):
                dst = out / "branches" / f"{b['run_id']}__{arm}__r{rep}.json"
                if dst.exists():
                    continue
                state = apply_arm(b, arm, rep)

                async def job(state=state, dst=dst):
                    done = await play(state, args.rounds, clients, args, sem,
                                      out / "logs" / f"{state['run_id']}__{state['arm']}"
                                                     f"__r{state['rep']}.jsonl")
                    write_json(dst, done)
                    post = done["history"][-args.window:]
                    d = sum(h["delta"] for h in post) / len(post)
                    print(f"  {done['run_id']} {done['arm']} r{done['rep']}  "
                          f"Δ(last {args.window}) = {d:+.2f}", flush=True)
                jobs.append(job())
    print(f"Running {len(jobs)} branches of {args.rounds} rounds "
          f"({len(jobs) * args.rounds * 2} model calls), models needed: {needed}", flush=True)
    t0 = time.time()
    await asyncio.gather(*jobs)
    print(f"Done in {time.time() - t0:.0f}s. Branches in {out / 'branches'}")


async def cmd_ping(args) -> None:
    specs = load_models(args.models)
    keys = args.keys.split(",") if args.keys else [k for k in specs if not k.startswith("_")]
    for k in keys:
        client = make_client(specs[k])
        print(f"{k}: {specs[k].get('model', specs[k].get('provider'))} ... "
              "(first call can take several minutes while the GPU starts)", flush=True)
        t0 = time.time()
        sem = asyncio.Semaphore(1)
        raw, parsed, n = await call_with_retries(
            client, "Reply with: PLANS: test\nINSIGHTS: test\nPRICE: 1.50", {"alpha": 1,
            "cost": 1, "history": []}, seed=1, temperature=0.0, max_tokens=50, sem=sem)
        print(f"   {time.time() - t0:.1f}s, parsed price={parsed.price}, reply={raw[:120]!r}")


def cmd_smoke(args) -> None:
    import shutil
    import analyze
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    ns = dict(models="models_mock.json", concurrency=64, history=30, temperature=0.7,
              max_tokens=400, threshold=0.5, window=20, prefix="longrun")
    asyncio.run(cmd_base(argparse.Namespace(out=str(out), n=6, rounds=60, seed0=0,
                                            alphas="1,3.2,10", **ns)))
    asyncio.run(cmd_branch(argparse.Namespace(out=str(out), rounds=60, reps=1, all=False,
                                              arms=",".join(ARMS), **ns)))
    analyze.main(["--out", str(out)])


def main(argv=None) -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--models", default="models.json")
        sp.add_argument("--out", default="runs/main")
        sp.add_argument("--concurrency", type=int, default=128, help="max requests in flight")
        sp.add_argument("--history", type=int, default=30, help="rounds of history shown")
        sp.add_argument("--temperature", type=float, default=0.7)
        sp.add_argument("--max-tokens", type=int, default=400)
        sp.add_argument("--threshold", type=float, default=0.5, help="collusion threshold on Δ")
        sp.add_argument("--window", type=int, default=20, help="rounds averaged for Δ")

    sb = sub.add_parser("base", help="run fresh games up to the swap round")
    common(sb)
    sb.add_argument("--n", type=int, default=10)
    sb.add_argument("--rounds", type=int, default=60, help="rounds before the swap (T)")
    sb.add_argument("--seed0", type=int, default=0)
    sb.add_argument("--alphas", default="1,3.2,10", help="price scales, cycled across seeds")
    sb.add_argument("--prefix", default="p1", choices=["p1", "longrun", "competitive"])

    sr = sub.add_parser("branch", help="fork each passing checkpoint into arms")
    common(sr)
    sr.add_argument("--rounds", type=int, default=60, help="rounds after the swap")
    sr.add_argument("--arms", default="A0,A1,A5,A6")
    sr.add_argument("--reps", type=int, default=1)
    sr.add_argument("--all", action="store_true", help="branch every base run, ignore inclusion")

    sp = sub.add_parser("ping", help="check each model endpoint answers")
    sp.add_argument("--models", default="models.json")
    sp.add_argument("--keys", default="")

    ss = sub.add_parser("smoke", help="offline end-to-end test with a mock agent")
    ss.add_argument("--out", default="runs/smoke")

    args = p.parse_args(argv)
    if args.cmd == "base":
        asyncio.run(cmd_base(args))
    elif args.cmd == "branch":
        asyncio.run(cmd_branch(args))
    elif args.cmd == "ping":
        asyncio.run(cmd_ping(args))
    elif args.cmd == "smoke":
        cmd_smoke(args)


if __name__ == "__main__":
    sys.exit(main())
