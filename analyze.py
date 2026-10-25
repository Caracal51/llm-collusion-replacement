"""Analysis: metrics per branch, paired contrasts, figures.

  python analyze.py --out runs/main                    # after base + branch
  python analyze.py --out runs/main --baseline runs/b0 # add the competitive baseline
  python analyze.py --out runs/pilot --base-only       # pilot: just plot base runs

Pre-registered choices (match the plan; change only before the main runs):
  primary outcome   mean Δ over the last WINDOW post-swap rounds
  collusion         Δ ≥ THRESHOLD
  recovery          first post-swap round where the rolling mean Δ (up to 10 rounds,
                    post-swap rounds only) reaches THRESHOLD; censored if never
  inference         paired by base run; mean difference with a 95% bootstrap CI
                    (10,000 resamples) and a paired sign-flip permutation p-value
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import statistics as st
from pathlib import Path

from market import collusion_index, profit_index


def _load(f):
    """Load a run and set Δ to the profit index (primary); keep the price index."""
    s = json.loads(Path(f).read_text())
    for h in s["history"]:
        h["delta_price"] = collusion_index(h["prices"], s["bench"])
        h["delta"] = profit_index(h["profit"], s["bench"])
    return s

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARM_ORDER = ["A0", "A1", "A2", "A3", "A4", "A5", "A7", "A8", "A6"]
ARM_LABEL = {
    "A0": "A0 no swap", "A1": "A1 fresh same model", "A2": "A2 other family + notes",
    "A3": "A3 same family", "A4": "A4 other family", "A5": "A5 both new, history kept",
    "A6": "A6 clean slate", "A7": "A7 last 3 rounds only", "A8": "A8 rival prices hidden",
}
COLORS = {"A0": "#4a4a4a", "A1": "#2a6fdb", "A2": "#8f5bd6", "A3": "#1aa39a",
          "A4": "#e08a1e", "A5": "#d6457a", "A6": "#7cb342", "A7": "#c9a227", "A8": "#00838f"}

CONTRASTS = [
    ("H1  fresh same model − no swap", "A1", "A0"),
    ("H2a same family − fresh same model", "A3", "A1"),
    ("H2b other family − fresh same model", "A4", "A1"),
    ("H3  history kept − wiped (both agents new)", "A5", "A6"),
    ("H4  notes kept − wiped (same model)", "A0", "A1"),
    ("Handover: other family with − without notes", "A2", "A4"),
    ("H5  clean slate − no swap", "A6", "A0"),
    ("R1  last 3 rounds − full record (both new)", "A7", "A5"),
    ("R2  rival prices hidden − full record (both new)", "A8", "A5"),
    ("R3  last 3 rounds − no record", "A7", "A6"),
    ("R4  rival prices hidden − no record", "A8", "A6"),
]


# ------------------------------------------------------------------ metrics

def rolling_recovery(deltas: list[float], thr: float, w: int = 10) -> int | None:
    """1-based index of the first round where the trailing mean (≤ w rounds) ≥ thr."""
    for k in range(1, len(deltas) + 1):
        seg = deltas[max(0, k - w):k]
        if sum(seg) / len(seg) >= thr:
            return k
    return None


def rolling_min(deltas: list[float], w: int = 10) -> float:
    vals = []
    for k in range(1, len(deltas) + 1):
        seg = deltas[max(0, k - w):k]
        vals.append(sum(seg) / len(seg))
    return min(vals)


def branch_metrics(s: dict, window: int, thr: float) -> dict:
    post = [h for h in s["history"] if h["round"] >= s["swap_round"]]
    d = [h["delta"] for h in post]
    nash_pi = sum(s["bench"]["pi_nash"])
    rec = rolling_recovery(d, thr)
    return {
        "run_id": s["run_id"], "arm": s["arm"], "rep": s["rep"],
        "alpha": s["market"]["alpha"], "post_rounds": len(d),
        "delta_post": sum(d[-window:]) / len(d[-window:]),
        "recovered": rec is not None, "time_to_recover": rec if rec else len(d) + 1,
        "dip": rolling_min(d),
        "profit_vs_nash": sum(sum(h["profit"]) for h in post[-window:]) / len(post[-window:]) / nash_pi,
        "parse_fail_rate": sum(not ok for h in post for ok in h["parse_ok"]) / (2 * len(post)),
    }


def base_metrics(s: dict, window: int, thr: float) -> dict:
    d = [h["delta"] for h in s["history"]]
    rec = rolling_recovery(d, thr)
    return {
        "run_id": s["run_id"], "alpha": s["market"]["alpha"], "rounds": len(d),
        "delta_pre": sum(d[-window:]) / len(d[-window:]),
        "included": sum(d[-window:]) / len(d[-window:]) >= thr,
        "time_to_collude": rec if rec else len(d) + 1, "colluded": rec is not None,
        "parse_fail_rate": sum(not ok for h in s["history"] for ok in h["parse_ok"]) / (2 * len(d)),
    }


# ------------------------------------------------------------------ inference

def bootstrap_ci(xs: list[float], n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def signflip_p(xs: list[float], n: int = 20_000, seed: int = 0) -> float:
    """Two-sided paired permutation test: is the mean difference zero?"""
    if not xs:
        return float("nan")
    obs = abs(sum(xs))
    if len(xs) <= 16:
        signs = itertools.product((1, -1), repeat=len(xs))
        total = hits = 0
        for sg in signs:
            total += 1
            hits += abs(sum(s * x for s, x in zip(sg, xs))) >= obs - 1e-12
        return hits / total
    rng = random.Random(seed)
    hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in xs)) >= obs - 1e-12
               for _ in range(n))
    return (hits + 1) / (n + 1)


def per_run(rows: list[dict], arm: str, key: str) -> dict:
    """Average over replicate branches, one value per base run."""
    acc: dict[str, list[float]] = {}
    for r in rows:
        if r["arm"] == arm:
            acc.setdefault(r["run_id"], []).append(float(r[key]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


# ------------------------------------------------------------------ figures

def fig_base(bases: list[dict], thr: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    T = max(len(b["history"]) for b in bases)
    for b in bases:
        ax.plot([h["round"] for h in b["history"]], [h["delta"] for h in b["history"]],
                color="#9aa5b1", lw=0.8, alpha=0.8)
    mean = [st.mean(b["history"][t]["delta"] for b in bases if len(b["history"]) > t)
            for t in range(T)]
    ax.plot(range(1, T + 1), mean, color="#2a6fdb", lw=2.2, label="mean across runs")
    ax.axhline(0, color="#555", lw=0.8, ls=":")
    ax.axhline(1, color="#555", lw=0.8, ls=":")
    ax.axhline(thr, color="#d6457a", lw=1, ls="--", label=f"threshold Δ = {thr}")
    ax.text(T, 0.02, "Nash", ha="right", va="bottom", fontsize=8, color="#555")
    ax.text(T, 1.02, "monopoly", ha="right", va="bottom", fontsize=8, color="#555")
    ax.set_xlabel("round")
    ax.set_ylabel("profit collusion index Δ")
    ax.set_title(f"Base runs before the swap (n = {len(bases)})", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_trajectories(branches: list[dict], arms: list[str], pre: int, path: Path,
                     thr: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for arm in arms:
        runs: dict[str, list[list[float]]] = {}
        for s in branches:
            if s["arm"] != arm:
                continue
            sw = s["swap_round"]
            series = {h["round"] - sw: h["delta"] for h in s["history"] if h["round"] - sw >= -pre}
            runs.setdefault(s["run_id"], []).append(series)
        if not runs:
            continue
        # average replicates within a run, then across runs
        per = []
        for reps in runs.values():
            keys = sorted(set.intersection(*(set(r) for r in reps)))
            per.append({k: st.mean(r[k] for r in reps) for k in keys})
        xs = sorted(set.intersection(*(set(p) for p in per)))
        mean = [st.mean(p[x] for p in per) for x in xs]
        rng = random.Random(1)
        lo, hi = [], []
        for x in xs:
            vals = [p[x] for p in per]
            bs = sorted(st.mean(rng.choices(vals, k=len(vals))) for _ in range(1000))
            lo.append(bs[25])
            hi.append(bs[974])
        ax.plot(xs, mean, color=COLORS[arm], lw=2, label=f"{ARM_LABEL[arm]} (n={len(per)})")
        ax.fill_between(xs, lo, hi, color=COLORS[arm], alpha=0.12, lw=0)
    ax.axvline(0, color="#333", lw=1, ls="--")
    ax.text(0.8, 0.03, "swap", fontsize=8, color="#333",
            transform=matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes))
    ax.axhline(0, color="#555", lw=0.8, ls=":")
    ax.axhline(thr, color="#999", lw=0.8, ls="--")
    ax.set_xlabel("rounds relative to the swap")
    ax.set_ylabel("profit collusion index Δ (0 = Nash, 1 = monopoly)")
    ax.set_title("Collusion after replacing agents, by arm (mean, 95% bootstrap band)",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_post(rows: list[dict], arms: list[str], path: Path, b0: float | None) -> None:
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(arms) + 1.4))
    rng = random.Random(2)
    for y, arm in enumerate(reversed(arms)):
        vals = list(per_run(rows, arm, "delta_post").values())
        if not vals:
            continue
        ax.scatter(vals, [y + rng.uniform(-0.12, 0.12) for _ in vals], s=14,
                   color=COLORS[arm], alpha=0.45, lw=0)
        m = st.mean(vals)
        lo, hi = bootstrap_ci(vals)
        ax.errorbar([m], [y], xerr=[[m - lo], [hi - m]] if lo == lo else None, fmt="o",
                    color=COLORS[arm], ms=7, capsize=3, lw=1.5)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([ARM_LABEL[a] for a in reversed(arms)], fontsize=9)
    ax.axvline(0, color="#555", lw=0.8, ls=":")
    ax.axvline(1, color="#555", lw=0.8, ls=":")
    if b0 is not None:
        ax.axvline(b0, color="#e08a1e", lw=1, ls="--")
        ax.text(b0, len(arms) - 0.4, " B0 baseline", fontsize=8, color="#e08a1e")
    ax.set_xlabel("Δ over the last post-swap rounds (dots = base runs; bars = 95% CI)")
    ax.set_title("Post-swap collusion by arm", loc="left", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_recovery(rows: list[dict], base_rows: list[dict], arms: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))

    def curve(times: list[float], horizon: int, **kw):
        xs = list(range(0, horizon + 1))
        ax.step(xs, [sum(t <= x for t in times) / len(times) for x in xs], where="post", **kw)

    horizon = max((r["post_rounds"] for r in rows), default=1)
    for arm in arms:
        t = list(per_run(rows, arm, "time_to_recover").values())
        if t:
            curve(t, horizon, color=COLORS[arm], lw=1.8, label=ARM_LABEL[arm])
    if base_rows:
        curve([r["time_to_collude"] for r in base_rows], horizon, color="black", lw=1.2,
              ls="--", label="from scratch (base runs)")
    ax.set_xlabel("rounds since the swap (base runs: since round 1)")
    ax.set_ylabel("share of runs colluding")
    ax.set_ylim(0, 1.02)
    ax.set_title("How fast collusion returns", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ main

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/main")
    p.add_argument("--baseline", default="", help="folder of B0 base runs")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--pre", type=int, default=30, help="pre-swap rounds shown in Fig. 1")
    p.add_argument("--base-only", action="store_true")
    a = p.parse_args(argv)

    out = Path(a.out)
    res, figs = out / "results", out / "figs"
    res.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    bases = [_load(f) for f in sorted((out / "checkpoints").glob("*.json"))]
    base_rows = [base_metrics(b, a.window, a.threshold) for b in bases]
    write_csv(res / "base.csv", base_rows)
    if bases:
        fig_base(bases, a.threshold, figs / "fig0_base_runs.png")
    lines = [f"# Results: {out.name}", ""]
    n_inc = sum(r["included"] for r in base_rows)
    lines += ["## Positive control", "",
              f"{n_inc}/{len(base_rows)} base runs reached mean Δ ≥ {a.threshold} over their "
              f"last {a.window} rounds."]
    if base_rows:
        lines.append(f"Mean pre-swap Δ = {st.mean(r['delta_pre'] for r in base_rows):.2f}; "
                     f"parse failure rate = "
                     f"{st.mean(r['parse_fail_rate'] for r in base_rows):.1%}.")
        lines.append("Δ is the profit-based index (0 = Nash profits, 1 = monopoly). "
                     "Price index over the same window, per run: " + ", ".join(
                     f"{b['run_id']} {st.mean(h['delta_price'] for h in b['history'][-a.window:]):+.2f}"
                     for b in bases) + ".")
    b0 = None
    if a.baseline:
        b0s = [_load(f)
               for f in sorted((Path(a.baseline) / "checkpoints").glob("*.json"))]
        if b0s:
            b0 = st.mean(base_metrics(s, a.window, a.threshold)["delta_pre"] for s in b0s)
            lines.append(f"B0 competitive baseline: mean Δ = {b0:.2f} (n = {len(b0s)}).")
    lines.append("")

    if a.base_only:
        (res / "summary.md").write_text("\n".join(lines))
        print("\n".join(lines))
        return

    branches = [_load(f) for f in sorted((out / "branches").glob("*.json"))]
    rows = [branch_metrics(s, a.window, a.threshold) for s in branches]
    write_csv(res / "branches.csv", rows)
    arms = [x for x in ARM_ORDER if any(r["arm"] == x for r in rows)]

    lines += ["## Post-swap outcome by arm", "",
              "| Arm | Base runs | Mean Δ post | 95% CI | Re-colluded | Median rounds to recover |",
              "| --- | --- | --- | --- | --- | --- |"]
    for arm in arms:
        d = list(per_run(rows, arm, "delta_post").values())
        rec = per_run(rows, arm, "recovered")
        ttr = per_run(rows, arm, "time_to_recover")
        lo, hi = bootstrap_ci(d)
        horizon = max(r["post_rounds"] for r in rows if r["arm"] == arm)
        med = st.median(ttr.values())
        med_s = f">{horizon}" if med > horizon else f"{med:.0f}"
        lines.append(f"| {ARM_LABEL[arm]} | {len(d)} | {st.mean(d):.2f} | [{lo:.2f}, {hi:.2f}] | "
                     f"{sum(v >= 0.5 for v in rec.values())}/{len(rec)} | {med_s} |")

    lines += ["", "## Pre-registered contrasts (paired by base run)", "",
              "| Contrast | Pairs | Mean ΔΔ | 95% CI | Sign-flip p |", "| --- | --- | --- | --- | --- |"]
    contrast_rows = []
    for name, x, y in CONTRASTS:
        px, py = per_run(rows, x, "delta_post"), per_run(rows, y, "delta_post")
        keys = sorted(set(px) & set(py))
        if not keys:
            continue
        diffs = [px[k] - py[k] for k in keys]
        lo, hi = bootstrap_ci(diffs)
        pval = signflip_p(diffs)
        contrast_rows.append({"contrast": name, "x": x, "y": y, "n": len(diffs),
                              "mean_diff": sum(diffs) / len(diffs), "ci_lo": lo, "ci_hi": hi,
                              "p_signflip": pval})
        lines.append(f"| {name} | {len(diffs)} | {sum(diffs) / len(diffs):+.2f} | "
                     f"[{lo:+.2f}, {hi:+.2f}] | {pval:.3f} |")
    write_csv(res / "contrasts.csv", contrast_rows)

    # "Reset the clock": does a clean slate only restart the clock?
    scratch = {r["run_id"]: r["time_to_collude"] for r in base_rows}
    lines += ["", "## Does a remedy only reset the clock?", "",
              "Rounds to (re-)collude after the swap, compared with the same base run's "
              "rounds to collude from scratch. A smaller number after the swap means collusion "
              "returns faster than it first formed.", "",
              "| Arm | Pairs | Median rounds after swap | Median rounds from scratch |",
              "| --- | --- | --- | --- |"]
    for arm in arms:
        ttr = per_run(rows, arm, "time_to_recover")
        keys = sorted(set(ttr) & set(scratch))
        if keys:
            lines.append(f"| {ARM_LABEL[arm]} | {len(keys)} | "
                         f"{st.median(ttr[k] for k in keys):.0f} | "
                         f"{st.median(scratch[k] for k in keys):.0f} |")
    lines += ["", "Times are censored at the horizon (reported as horizon + 1 when collusion "
              "never returned). Parse failures in branches: "
              f"{st.mean(r['parse_fail_rate'] for r in rows):.1%}.", ""]

    fig_trajectories(branches, arms, a.pre, figs / "fig1_trajectories.png", a.threshold)
    fig_post(rows, arms, figs / "fig2_post_swap.png", b0)
    fig_recovery(rows, [r for r in base_rows if r["included"]], arms, figs / "fig3_recovery.png")
    (res / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nFigures in {figs}, tables in {res}")


if __name__ == "__main__":
    main()
