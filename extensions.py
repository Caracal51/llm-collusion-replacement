"""Two follow-up experiments (written during the sprint, Sat 24 Oct evening).

  python extensions.py seed    --out runs/seeded --reps 3 --rounds 60
  python extensions.py transplant --main runs/main --b0 runs/b0 --rounds 60
  python extensions.py deviate --main runs/main --rounds 30
  python extensions.py report  --seeded runs/seeded --main runs/main

seed:    counterfactual records. Fresh agents (no notes) get a fabricated 30-round history
         with both sellers at a chosen price level (price war, Nash, monopoly, the high level
         our agents settle at, or no record). Same seeds across levels, so levels are paired.
         Question: does the record alone set where prices go?
transplant: real records. Fresh agents (prompt p1, no notes) inherit the real last 30 rounds of
         every game that did NOT get arm A5 already: the 22 non-collusive main games and the 5
         competitive-prompt B0 games (price wars). Together with A5 on the 14 collusive games this
         covers real records from price war to overpricing. Question: does the final price track
         the recorded price?
deviate: punishment test. From each collusive base run (the A0 state), seller B is forced to
         the Nash price for one round, then both play on. Paired with A0 (same seeds).
         Question: does seller A punish the cut and do prices return (tacit collusion), or not?
Everything is resumable and nothing in experiment.py or analyze.py changes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import statistics as st
import time
from pathlib import Path

from agents import make_client
from analyze import _load, bootstrap_ci, signflip_p
from experiment import apply_arm, load_dotenv, load_models, play, pre_swap_delta, write_json, new_state
from market import Market, collusion_index, profit_index

LEVELS = {  # price index of the fabricated record (0 = Nash, 1 = monopoly)
    "war": -1.0, "nash": 0.0, "monopoly": 1.0, "high": 1.4, "none": None,
}
SEED_ROUNDS = 30


# ------------------------------------------------------------------ helpers

class Cfg:
    def __init__(self, a):
        self.history, self.temperature, self.max_tokens = a.history, a.temperature, a.max_tokens


def level_prices(bench: dict, idx: float) -> list[float]:
    return [bench["p_nash"][i] + idx * (bench["p_monopoly"][i] - bench["p_nash"][i]) for i in range(2)]


def seeded_state(level: str, alpha: float, rep: int, prefix: str) -> dict:
    m = Market(alpha=alpha)
    run_id = f"seed-a{alpha:g}-r{rep}"            # shared across levels: paired by (alpha, rep)
    s = new_state(run_id, rep, prefix, m)
    s["arm"], s["rep"], s["swap_round"] = f"S-{level}", rep, SEED_ROUNDS + 1
    s["round"] = SEED_ROUNDS + 1
    s["visible_from"] = SEED_ROUNDS + 1 if LEVELS[level] is None else 1
    if LEVELS[level] is not None:
        base = level_prices(s["bench"], LEVELS[level])
        rng = random.Random(f"{run_id}-{level}")
        for t in range(1, SEED_ROUNDS + 1):
            p = [base[i] * (1 + rng.uniform(-0.005, 0.005)) for i in range(2)]
            q, pi = m.quantities(p), m.profits(p)
            s["history"].append({"round": t, "prices": p, "qty": q, "profit": pi,
                                 "parse_ok": [True, True], "models": ["record", "record"],
                                 "delta": profit_index(pi, s["bench"]),
                                 "delta_price": collusion_index(p, s["bench"])})
    return s


class ForcedPrice:
    """Wraps a client: on one round it returns a fixed price and echoes the seller's notes."""

    def __init__(self, inner, forced_round: int, price: float):
        self.inner, self.forced_round, self.price = inner, forced_round, price

    async def complete(self, prompt, ctx, **kw):
        m = re.search(r"This is round (\d+)\.", prompt)
        if m and int(m.group(1)) == self.forced_round:
            plans = re.search(r"^PLANS: (.*)$", prompt, re.M)
            ins = re.search(r"^INSIGHTS: (.*)$", prompt, re.M)
            return (f"PLANS: {plans.group(1) if plans else ''}\n"
                    f"INSIGHTS: {ins.group(1) if ins else ''}\nPRICE: {self.price:.4f}")
        return await self.inner.complete(prompt, ctx, **kw)


# ------------------------------------------------------------------ commands

async def cmd_seed(a):
    out = Path(a.out)
    clients = {"M": make_client(load_models(a.models)["M"])}
    sem = asyncio.Semaphore(a.concurrency)
    jobs = []
    for level in a.levels.split(","):
        for alpha in [float(x) for x in a.alphas.split(",")]:
            for rep in range(a.reps):
                dst = out / "games" / f"S-{level}__a{alpha:g}__r{rep}.json"
                if dst.exists():
                    continue
                s = seeded_state(level, alpha, rep, a.prefix)

                async def job(s=s, dst=dst):
                    done = await play(s, a.rounds, clients, Cfg(a), sem,
                                      out / "logs" / f"{dst.stem}.jsonl")
                    write_json(dst, done)
                    post = done["history"][-20:]
                    print(f"  {dst.stem}  price idx {st.mean(h['delta_price'] for h in post):+.2f}"
                          f"  profit idx {st.mean(h['delta'] for h in post):+.2f}", flush=True)
                jobs.append(job())
    print(f"Running {len(jobs)} seeded games of {a.rounds} rounds ({len(jobs) * a.rounds * 2} calls)")
    t0 = time.time()
    await asyncio.gather(*jobs)
    print(f"Done in {time.time() - t0:.0f}s.")


async def cmd_transplant(a):
    out = Path(a.out)
    clients = {"M": make_client(load_models(a.models)["M"])}
    sem = asyncio.Semaphore(a.concurrency)
    donors = [json.loads(p.read_text()) for p in sorted((Path(a.main) / "checkpoints").glob("*.json"))]
    donors = [d for d in donors if pre_swap_delta(d, 20) < a.threshold]       # A5 covers the rest
    donors += [json.loads(p.read_text()) for p in sorted((Path(a.b0) / "checkpoints").glob("*.json"))]
    jobs = []
    for d in donors:
        dst = out / "games" / f"{d['run_id']}__T.json"
        if dst.exists():
            continue
        s = apply_arm(d, "A5", 0)              # both agents new, notes wiped, record kept
        s["arm"], s["donor_prefix"], s["prefix"] = "T", d["prefix"], a.prefix

        async def job(s=s, dst=dst):
            done = await play(s, a.rounds, clients, Cfg(a), sem, out / "logs" / f"{dst.stem}.jsonl")
            write_json(dst, done)
            print(f"  {dst.stem} done", flush=True)
        jobs.append(job())
    print(f"Running {len(jobs)} transplant games of {a.rounds} rounds ({len(jobs) * a.rounds * 2} calls)")
    t0 = time.time()
    await asyncio.gather(*jobs)
    print(f"Done in {time.time() - t0:.0f}s.")


async def cmd_deviate(a):
    main = Path(a.main)
    shared = make_client(load_models(a.models)["M"])
    sem = asyncio.Semaphore(a.concurrency)
    bases = [json.loads(p.read_text()) for p in sorted((main / "checkpoints").glob("*.json"))]
    bases = [b for b in bases if pre_swap_delta(b, 20) >= a.threshold]
    print(f"{len(bases)} collusive base runs")
    jobs = []
    for b in bases:
        dst = main / "deviation" / f"{b['run_id']}__D1.json"
        if dst.exists():
            continue
        s = apply_arm(b, "A0", 0)                  # same seeds as arm A0: paired comparison
        s["arm"] = "D1"
        s["sellers"][1]["model"] = "D"
        clients = {"M": shared, "D": ForcedPrice(shared, s["swap_round"], s["bench"]["p_nash"][1])}

        async def job(s=s, dst=dst, clients=clients):
            done = await play(s, a.rounds, clients, Cfg(a), sem,
                              main / "deviation" / "logs" / f"{dst.stem}.jsonl")
            write_json(dst, done)
            print(f"  {dst.stem} done", flush=True)
        jobs.append(job())
    print(f"Running {len(jobs)} deviation games of {a.rounds} rounds")
    t0 = time.time()
    await asyncio.gather(*jobs)
    print(f"Done in {time.time() - t0:.0f}s.")


# ------------------------------------------------------------------ report

def cmd_report(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = ["# Follow-up experiments", ""]

    # ---- data-quality gate: any failed calls (the code reuses the last price after 4 failures)
    bad = []
    for folder in [Path(a.seeded) / "games", Path(a.transplant) / "games", Path(a.main) / "deviation",
                   Path(a.main) / "branches", Path(a.main) / "checkpoints"]:
        for f in sorted(folder.glob("*.json")) if folder.exists() else []:
            g = json.loads(f.read_text())
            n = sum(not ok for h in g["history"] if h.get("models", [""])[0] != "record"
                    for ok in h["parse_ok"])
            if n:
                bad.append(f"{folder.name}/{f.name}: {n} failed calls")
    lines += ["## Data quality", "",
              "All games clean: no failed or unparseable calls." if not bad else
              f"**{len(bad)} games had failed calls; exclude or rerun them:**"] + [f"- {b}" for b in bad] + [""]
    figs = Path(a.main) / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    # ---- counterfactual records
    seeded = Path(a.seeded)
    games = [_load(f) for f in sorted((seeded / "games").glob("*.json"))] if seeded.exists() else []
    if games:
        order = [l for l in LEVELS if any(g["arm"] == f"S-{l}" for g in games)]

        def final(g, key):
            post = [h for h in g["history"] if h["round"] >= g["swap_round"]][-20:]
            return st.mean(h[key] for h in post)

        def first(g):
            h = next(h for h in g["history"] if h["round"] == g["swap_round"])
            return h["delta_price"]
        lines += ["## Counterfactual records: fresh agents given a fabricated 30-round history", "",
                  "| Record level (price index) | Games | First-round price index | Final price index "
                  "[95% CI] | Final profit index |", "| --- | --- | --- | --- | --- |"]
        fin = {}
        for l in order:
            gs = [g for g in games if g["arm"] == f"S-{l}"]
            fin[l] = {g["run_id"]: final(g, "delta_price") for g in gs}
            v = list(fin[l].values())
            lo, hi = bootstrap_ci(v)
            lab = "no record" if LEVELS[l] is None else f"{l} ({LEVELS[l]:+.1f})"
            lines.append(f"| {lab} | {len(gs)} | {st.mean(first(g) for g in gs):+.2f} | "
                         f"{st.mean(v):+.2f} [{lo:+.2f}, {hi:+.2f}] | "
                         f"{st.mean(final(g, 'delta') for g in gs):+.2f} |")
        lines += ["", "| Paired contrast (final price index) | Pairs | Mean [95% CI] | Sign-flip p |",
                  "| --- | --- | --- | --- |"]
        pairs = [("high", "nash"), ("monopoly", "nash"), ("nash", "war"), ("high", "none"),
                 ("nash", "none"), ("war", "none")]
        for x, y in pairs:
            if x in fin and y in fin:
                k = sorted(set(fin[x]) & set(fin[y]))
                d = [fin[x][i] - fin[y][i] for i in k]
                lo, hi = bootstrap_ci(d)
                lines.append(f"| {x} − {y} | {len(k)} | {st.mean(d):+.2f} [{lo:+.2f}, {hi:+.2f}] | "
                             f"{signflip_p(d):.3f} |")
        xs = [(LEVELS[l], v) for l in order if LEVELS[l] is not None for v in fin[l].values()]
        if len(xs) > 2:
            mx, my = st.mean(x for x, _ in xs), st.mean(y for _, y in xs)
            slope = sum((x - mx) * (y - my) for x, y in xs) / sum((x - mx) ** 2 for x, _ in xs)
            lines += ["", f"Slope of final price index on the recorded price index: {slope:.2f} "
                      "(1 = prices stay exactly where the record put them; 0 = the record has no lasting effect).", ""]

        fig, ax = plt.subplots(figsize=(9, 4.2))
        colors = {"war": "#2a6fdb", "nash": "#1aa39a", "monopoly": "#e08a1e", "high": "#d6457a",
                  "none": "#4a4a4a"}
        for l in order:
            gs = [g for g in games if g["arm"] == f"S-{l}"]
            series = [[h["delta_price"] for h in g["history"]] for g in gs]
            n = min(len(s) for s in series)
            mean = [st.mean(s[t] for s in series) for t in range(n)]
            rounds = list(range(1, n + 1)) if LEVELS[l] is not None else \
                list(range(SEED_ROUNDS + 1, SEED_ROUNDS + 1 + n))
            lab = "no record" if LEVELS[l] is None else f"record at {l}"
            ax.plot(rounds, mean, color=colors[l], lw=2, label=lab)
        ax.axvspan(0.5, SEED_ROUNDS + 0.5, color="#888", alpha=0.08)
        ax.text(SEED_ROUNDS / 2, ax.get_ylim()[1], "fabricated record", ha="center", va="top",
                fontsize=8, color="#555")
        ax.axhline(0, color="#555", lw=0.8, ls=":")
        ax.axhline(1, color="#555", lw=0.8, ls=":")
        ax.set_xlabel("round (fresh agents start at round 31)")
        ax.set_ylabel("price index (0 = Nash, 1 = monopoly)")
        ax.set_title("Fresh agents price wherever the record says", loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        fig.tight_layout()
        fig.savefig(figs / "fig4_counterfactual_records.png", dpi=200)
        plt.close(fig)

    # ---- real-record transplants (plus A5 on the collusive games)
    tdir = Path(a.transplant) / "games"
    tgames = [_load(f) for f in sorted(tdir.glob("*.json"))] if tdir.exists() else []
    a5 = [_load(f) for f in (Path(a.main) / "branches").glob("*__A5__r0.json")]
    if tgames:
        pts = []
        for g in tgames + a5:
            sw = g["swap_round"]
            rec = [h for h in g["history"] if h["round"] < sw][-30:]
            post = [h for h in g["history"] if h["round"] >= sw]
            src = ("B0 (competitive prompt)" if g.get("donor_prefix") == "competitive" else
                   "collusive (A5)" if g["arm"] == "A5" else "non-collusive")
            pts.append((st.mean(h["delta_price"] for h in rec), st.mean(h["delta_price"] for h in post[-20:]),
                        post[0]["delta_price"], src))
        mx, my = st.mean(p[0] for p in pts), st.mean(p[1] for p in pts)
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
        r = slope * math.sqrt(sxx / sum((p[1] - my) ** 2 for p in pts))
        rng = random.Random(0)
        boots = []
        for _ in range(5000):
            b = [rng.choice(pts) for _ in pts]
            bx, by = st.mean(p[0] for p in b), st.mean(p[1] for p in b)
            bxx = sum((p[0] - bx) ** 2 for p in b)
            if bxx:
                boots.append(sum((p[0] - bx) * (p[1] - by) for p in b) / bxx)
        boots.sort()
        lines += ["## Real records transplanted into fresh agents", "",
                  f"{len(pts)} real records (last 30 rounds of a real game) given to two fresh agents "
                  "(prompt p1, no notes), 60 rounds each.", "",
                  "| Record source | Games | Recorded price index | First-round price index | Final price index |",
                  "| --- | --- | --- | --- | --- |"]
        for src in ["B0 (competitive prompt)", "non-collusive", "collusive (A5)"]:
            q = [p for p in pts if p[3] == src]
            if q:
                lines.append(f"| {src} | {len(q)} | {st.mean(p[0] for p in q):+.2f} | "
                             f"{st.mean(p[2] for p in q):+.2f} | {st.mean(p[1] for p in q):+.2f} |")
        lines += ["", f"Final price index on recorded price index: slope {slope:.2f} "
                  f"[{boots[int(0.025 * len(boots))]:.2f}, {boots[int(0.975 * len(boots)) - 1]:.2f}], "
                  f"r = {r:.2f}, n = {len(pts)}. A slope of 1 means prices stay where the record left them.", ""]
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
        cols = {"B0 (competitive prompt)": "#2a6fdb", "non-collusive": "#e08a1e", "collusive (A5)": "#d6457a"}
        for src, c in cols.items():
            q = [p for p in pts if p[3] == src]
            ax.scatter([p[0] for p in q], [p[1] for p in q], s=28, color=c, alpha=0.8, lw=0, label=src)
        lo_ = min(min(p[0] for p in pts), min(p[1] for p in pts)) - 0.2
        hi_ = max(max(p[0] for p in pts), max(p[1] for p in pts)) + 0.2
        ax.plot([lo_, hi_], [lo_, hi_], color="#888", lw=1, ls="--", label="final = recorded")
        ax.plot([lo_, hi_], [my + slope * (lo_ - mx), my + slope * (hi_ - mx)], color="#222", lw=1.5,
                label=f"fit, slope {slope:.2f}")
        for v, t in ((0, "Nash"), (1, "monopoly")):
            ax.axvline(v, color="#bbb", lw=0.8, ls=":")
            ax.axhline(v, color="#bbb", lw=0.8, ls=":")
        ax.set_xlabel("price index of the inherited real record")
        ax.set_ylabel("price index of fresh agents, last 20 rounds")
        ax.set_title("Fresh agents keep the prices they inherit", loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        fig.tight_layout()
        fig.savefig(figs / "fig5_real_record_transplants.png", dpi=200)
        plt.close(fig)

    # ---- deviation test
    dev_dir = Path(a.main) / "deviation"
    devs = [_load(f) for f in sorted(dev_dir.glob("*.json"))] if dev_dir.exists() else []
    a0 = {s["run_id"]: s for s in (_load(f) for f in (Path(a.main) / "branches").glob("*__A0__r0.json"))}
    if devs:
        lines += ["## Deviation test: seller B forced to the Nash price for one round", "",
                  "Prices relative to each seller's last price before the deviation; paired with A0.", ""]
        rows = []
        for d in devs:
            c = a0.get(d["run_id"])
            if not c:
                continue
            sw = d["swap_round"]
            pre = next(h for h in d["history"] if h["round"] == sw - 1)["prices"]

            def rel(s, k, i):
                h = next((h for h in s["history"] if h["round"] == sw + k), None)
                return None if h is None else h["prices"][i] / pre[i] - 1
            rows.append({k: (rel(d, k, 0), rel(c, k, 0), rel(d, k, 1), rel(c, k, 1))
                         for k in (0, 1, 2, 3, 5, 10, 20, 29)})
        lines += ["| Rounds after the cut | A's price change (deviation) | A's change (no deviation) | "
                  "B's price change (deviation) | A cut ≥ 2% (punishment) |",
                  "| --- | --- | --- | --- | --- |"]
        for k in (1, 2, 3, 5, 10, 20, 29):
            r = [x[k] for x in rows if x[k][0] is not None]
            if not r:
                continue
            lines.append(f"| {k} | {st.mean(x[0] for x in r):+.1%} | {st.mean(x[1] for x in r):+.1%} | "
                         f"{st.mean(x[2] for x in r):+.1%} | {sum(x[0] <= -0.02 for x in r)}/{len(r)} |")
        d1 = [x[1][0] - x[1][1] for x in rows if x[1][0] is not None]
        lo, hi = bootstrap_ci(d1)
        lines += ["", f"A's response one round after the cut, deviation − control: {st.mean(d1):+.1%} "
                  f"[{lo:+.1%}, {hi:+.1%}], sign-flip p = {signflip_p(d1):.3f}.",
                  "Punishment (tacit collusion) predicts A cuts its price after B's cut, then both return.", ""]

    # ---- responsiveness in the original games (free)
    bases = [_load(f) for f in sorted((Path(a.main) / "checkpoints").glob("*.json"))]
    xs, ys = [], []
    for b in bases:
        p = [h["prices"] for h in b["history"]]
        for t in range(2, len(p)):
            for i in (0, 1):
                xs.append((p[t - 1][1 - i] - p[t - 2][1 - i]) / b["market"]["alpha"])
                ys.append((p[t][i] - p[t - 1][i]) / b["market"]["alpha"])
    if xs:
        mx, my = st.mean(xs), st.mean(ys)
        sx = math.sqrt(sum((x - mx) ** 2 for x in xs)); sy = math.sqrt(sum((y - my) ** 2 for y in ys))
        r = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else float("nan")
        lines += ["## Responsiveness in the base games", "",
                  f"Correlation between a seller's price change and the rival's price change one round "
                  f"earlier, pooled over {len(bases)} base games: r = {r:+.2f} (n = {len(xs)}).", ""]

    out = Path(a.main) / "results" / "followups.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))


def main(argv=None):
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("seed", "transplant", "deviate"):
        sp = sub.add_parser(name)
        sp.add_argument("--models", default="models.json")
        sp.add_argument("--concurrency", type=int, default=128)
        sp.add_argument("--history", type=int, default=30)
        sp.add_argument("--temperature", type=float, default=0.7)
        sp.add_argument("--max-tokens", type=int, default=400)
        sp.add_argument("--rounds", type=int, default=60 if name == "seed" else 30)
    sub.choices["seed"].add_argument("--out", default="runs/seeded")
    sub.choices["seed"].add_argument("--reps", type=int, default=3)
    sub.choices["seed"].add_argument("--alphas", default="1,3.2,10")
    sub.choices["seed"].add_argument("--levels", default="war,nash,monopoly,high,none")
    sub.choices["seed"].add_argument("--prefix", default="p1")
    sub.choices["transplant"].add_argument("--out", default="runs/transplant")
    sub.choices["transplant"].add_argument("--main", default="runs/main")
    sub.choices["transplant"].add_argument("--b0", default="runs/b0")
    sub.choices["transplant"].add_argument("--prefix", default="p1")
    sub.choices["transplant"].add_argument("--threshold", type=float, default=0.5)
    sub.choices["deviate"].add_argument("--main", default="runs/main")
    sub.choices["deviate"].add_argument("--threshold", type=float, default=0.5)
    sr = sub.add_parser("report")
    sr.add_argument("--seeded", default="runs/seeded")
    sr.add_argument("--main", default="runs/main")
    sr.add_argument("--transplant", default="runs/transplant")
    a = p.parse_args(argv)
    if a.cmd == "seed":
        asyncio.run(cmd_seed(a))
    elif a.cmd == "transplant":
        asyncio.run(cmd_transplant(a))
    elif a.cmd == "deviate":
        asyncio.run(cmd_deviate(a))
    else:
        cmd_report(a)


if __name__ == "__main__":
    main()
