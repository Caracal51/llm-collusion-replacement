"""Final figures for the paper (Sun 25 Oct). Reads the downloaded results; writes paper_figs/*.png.

  python paper_figs.py --runs path/to/runs --out paper_figs
"""
from __future__ import annotations

import argparse
import random
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from analyze import _load

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "font.family": "DejaVu Sans"})
INK, MUTED, GRID = "#222222", "#6b6b6b", "#d9d9d9"
ARM = {  # colour, label
    "A0": ("#4a4a4a", "No swap (A0)"),
    "A1": ("#2a6fdb", "Fresh Qwen replaces B (A1)"),
    "A4": ("#e08a1e", "Phi-4 replaces B (A4)"),
    "A5": ("#c2185b", "Both agents new, record kept (A5)"),
    "A6": ("#2e9d4f", "Both new, record wiped (A6)"),
}


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def mean_band(series_list, rng):
    xs = sorted(set.intersection(*(set(s) for s in series_list)))
    mean, lo, hi = [], [], []
    for x in xs:
        v = [s[x] for s in series_list]
        mean.append(st.mean(v))
        bs = sorted(st.mean(rng.choices(v, k=len(v))) for _ in range(1000))
        lo.append(bs[25]); hi.append(bs[974])
    return xs, mean, lo, hi


def fig_design(out):
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec=MUTED, bold=False, fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", linespacing=1.25)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1, shrinkA=0, shrinkB=0))

    box(1, 15, 17, 10, "36 games\nQwen2.5-14B × 2\n100 rounds", bold=False)
    box(24, 15, 17, 10, "14 collusive\ncheckpoints\n(profit index ≥ 0.5)", fc="#fbe9ef", ec="#c2185b")
    arrow(18.6, 20, 23.4, 20)
    groups = [
        (33, "Replace an agent", "A1 fresh Qwen · A4 Phi-4\nA2 Phi-4 + old notes"),
        (24, "Replace both agents", "A5 notes wiped, record kept"),
        (15, "Restrict the record", "A7 last 3 rounds · A8 rival hidden"),
        (6, "Wipe the record", "A6 notes and record wiped"),
    ]
    for y, head, body in groups:
        box(50, y - 3.2, 49, 7.2, f"{head}:  {body}", fs=7.3)
        arrow(41.6, 20, 49.4, y + 0.4)
    ax.text(50, 39.2, "Branch each checkpoint at round 101 (60 rounds; same seeds across arms)",
            fontsize=7.6, color=MUTED, va="top")
    ax.text(1, 6.5, "Follow-ups: fresh agents given 27 other\nreal records or fabricated records;\n"
            "a forced one-round price cut",
            fontsize=7.3, color=MUTED, va="center")
    fig.tight_layout(pad=0.2)
    fig.savefig(out / "fig1_design.png", dpi=300)
    plt.close(fig)


def fig_main(runs, out):
    br = [_load(f) for f in sorted((runs / "main" / "branches").glob("*.json"))]
    rng = random.Random(1)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    for ax, key, title, ylab in [
        (axes[0], "delta", "(a) Seller profits", "profit index (0 = Nash, 1 = monopoly)"),
        (axes[1], "delta_price", "(b) Prices", "price index (0 = Nash, 1 = monopoly)")]:
        for arm, (col, lab) in ARM.items():
            ser = []
            for s in br:
                if s["arm"] != arm:
                    continue
                sw = s["swap_round"]
                ser.append({h["round"] - sw: h[key] for h in s["history"] if h["round"] - sw >= -20})
            xs, m, lo, hi = mean_band(ser, rng)
            ax.plot(xs, m, color=col, lw=1.7, label=lab)
            ax.fill_between(xs, lo, hi, color=col, alpha=0.12, lw=0)
        ax.axvline(0, color=INK, lw=0.8, ls="--")
        for v in (0, 1):
            ax.axhline(v, color=GRID, lw=0.8, zorder=0)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("rounds relative to the swap")
        ax.set_ylabel(ylab)
        clean(ax)
    axes[0].text(0.8, axes[0].get_ylim()[0] + 0.05, "swap", fontsize=7, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out / "fig2_main.png", dpi=300)
    plt.close(fig)


def fig_record(runs, out):
    seeded = [_load(f) for f in sorted((runs / "seeded" / "games").glob("*.json"))]
    trans = [_load(f) for f in sorted((runs / "transplant" / "games").glob("*.json"))]
    a5 = [_load(f) for f in (runs / "main" / "branches").glob("*__A5__r0.json")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    levels = [("high", "#c2185b", "high (1.4)"), ("monopoly", "#e08a1e", "monopoly (1)"),
              ("nash", "#2a6fdb", "Nash (0)"), ("war", "#5e35b1", "price war (−1)"),
              ("none", "#4a4a4a", "no record")]
    for lvl, col, lab in levels:
        gs = [g for g in seeded if g["arm"] == f"S-{lvl}"]
        n = min(len(g["history"]) for g in gs)
        m = [st.mean(g["history"][t]["delta_price"] for g in gs) for t in range(n)]
        rounds = list(range(1, n + 1)) if lvl != "none" else list(range(31, 31 + n))
        ax.plot(rounds, m, color=col, lw=1.7, label=lab)
        ax.text(rounds[-1] + 1.5, m[-1], lab, color=col, fontsize=7, va="center")
    ax.set_xlim(0, 118)
    ax.axvspan(0.5, 30.5, color="#9e9e9e", alpha=0.12, lw=0)
    ax.text(15.5, 2.55, "fabricated\nrecord", ha="center", va="top", fontsize=7, color=MUTED)
    for v in (0, 1):
        ax.axhline(v, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("round (fresh agents take over at round 31)")
    ax.set_ylabel("price index (0 = Nash, 1 = monopoly)")
    ax.set_title("(a) Fabricated records, 6 games each", loc="left", color=INK)
    ax.set_xticks([0, 30, 60, 90])
    clean(ax)

    ax = axes[1]
    pts = []
    for g in trans + a5:
        sw = g["swap_round"]
        rec = [h for h in g["history"] if h["round"] < sw][-30:]
        post = [h for h in g["history"] if h["round"] >= sw][-20:]
        src = "b0" if g.get("donor_prefix") == "competitive" else ("a5" if g["arm"] == "A5" else "nc")
        pts.append((st.mean(h["delta_price"] for h in rec), st.mean(h["delta_price"] for h in post), src))
    style = {"b0": ("#5e35b1", "price war (B0)"), "nc": ("#e08a1e", "non-collusive"),
             "a5": ("#c2185b", "collusive")}
    for src, (col, lab) in style.items():
        q = [p for p in pts if p[2] == src]
        ax.scatter([p[0] for p in q], [p[1] for p in q], s=16, color=col, alpha=0.85, lw=0, label=lab)
    mx, my = st.mean(p[0] for p in pts), st.mean(p[1] for p in pts)
    slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / sum((p[0] - mx) ** 2 for p in pts)
    lo_, hi_ = -1.6, 4.5
    ax.plot([lo_, hi_], [lo_, hi_], color=MUTED, lw=0.9, ls="--")
    ax.plot([lo_, hi_], [my + slope * (lo_ - mx), my + slope * (hi_ - mx)], color=INK, lw=1.3)
    ax.text(3.2, 4.2, "final = recorded", fontsize=7, color=MUTED, rotation=38)
    ax.text(2.0, 0.6, f"fit: slope {slope:.2f}", fontsize=7.5, color=INK)
    for v in (0, 1):
        ax.axhline(v, color=GRID, lw=0.8, zorder=0)
        ax.axvline(v, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("price index of the inherited real record")
    ax.set_ylabel("price index, last 20 rounds")
    ax.set_title(f"(b) Real records, n = {len(pts)}", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    clean(ax)
    fig.tight_layout()
    fig.savefig(out / "fig3_record.png", dpi=300)
    plt.close(fig)


def fig_deviation(runs, out):
    devs = {s["run_id"]: s for s in (_load(f) for f in (runs / "main" / "deviation").glob("*.json"))}
    ctl = {s["run_id"]: s for s in (_load(f) for f in (runs / "main" / "branches").glob("*__A0__r0.json"))}
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for seller, col, lab in [(0, "#2a6fdb", "seller A (not forced)"), (1, "#c2185b", "seller B (forced)")]:
        ks = list(range(-3, 11))
        vals = []
        for rid, d in devs.items():
            sw = d["swap_round"]
            h = {x["round"]: x for x in d["history"]}
            pre = h[sw - 1]["prices"][seller]
            vals.append([h[sw + k]["prices"][seller] / pre - 1 for k in ks])
        m = [100 * st.mean(v[i] for v in vals) for i in range(len(ks))]
        ax.plot(ks, m, color=col, lw=1.7, marker="o", ms=3, label=lab)
    ax.axvline(0, color=INK, lw=0.8, ls="--")
    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("rounds relative to the forced cut")
    ax.set_ylabel("price change vs before (%)")
    ax.set_title("B cut to the Nash price for one round", loc="left", color=INK, fontsize=9)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    clean(ax)
    fig.tight_layout()
    fig.savefig(out / "figA_deviation.png", dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--out", default="paper_figs")
    a = p.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_design(out)
    fig_main(runs, out)
    fig_record(runs, out)
    fig_deviation(runs, out)
    print("wrote", sorted(x.name for x in out.glob("*.png")))


if __name__ == "__main__":
    main()
