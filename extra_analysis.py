"""Extra analyses for the report (written during the sprint, after the main runs).

  python extra_analysis.py --out runs/main

1. Robustness: every contrast recomputed with the price index instead of the profit index.
2. Anchoring: does a newcomer's first post-swap price sit at the last price on the record?
3. Words: what the newcomers' first notes say (keyword rates + example quotes).
Writes results/extra.md and results/quotes.md.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

import math

from analyze import CONTRASTS, ARM_LABEL, ARM_ORDER, _load, bootstrap_ci, signflip_p
from market import Market


def consumer_surplus(m: Market, p) -> float:
    """Logit consumer surplus (Anderson, de Palma and Thisse 1992), in money units."""
    x = [(m.a[i] - p[i] / m.alpha) / m.mu for i in range(2)] + [m.a0 / m.mu]
    top = max(x)
    return m.beta * m.mu * m.alpha * (top + math.log(sum(math.exp(v - top) for v in x)))


def harm_index(s: dict, prices) -> float:
    """Consumer-harm index: 0 = consumers as well off as at Nash, 1 = as badly off as at monopoly."""
    m, b = Market.from_dict(s["market"]), s["bench"]
    cn, cm = consumer_surplus(m, b["p_nash"]), consumer_surplus(m, b["p_monopoly"])
    return (cn - consumer_surplus(m, prices)) / (cn - cm)

KEYWORDS = {
    "competitor / rival": r"\bcompetitor|\brival|\bother (seller|firm)",
    "maintain / keep / stable": r"\bmaintain|\bkeep\b|\bkeeping\b|\bstable|\bsteady|\bhold\b|\bconsisten",
    "match / align / similar": r"\bmatch|\balign|\bsimilar price|\bin line with|\bmirror",
    "raise / increase": r"\braise|\bincreas|\bhigher price|\bgradually (raise|increase)",
    "undercut / lower": r"\bundercut|\blower (my |our |the )?price|\bdecreas|\bprice war|\bcut (my |our |the )?price",
    "history / past rounds": r"\bhistor|\bpast rounds|\bprevious rounds|\blast \d+ rounds|\bconsistently",
}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/main")
    p.add_argument("--window", type=int, default=20)
    a = p.parse_args(argv)
    out = Path(a.out)
    branches = [_load(f) for f in sorted((out / "branches").glob("*.json"))]
    lines = ["# Extra analyses", ""]

    # 1. price-index robustness ---------------------------------------------------------
    def post_mean(s, key):
        post = [h for h in s["history"] if h["round"] >= s["swap_round"]][-a.window:]
        return sum(h[key] for h in post) / len(post)

    per = {}
    for s in branches:
        per.setdefault(s["arm"], {})[s["run_id"]] = (post_mean(s, "delta"), post_mean(s, "delta_price"))
    lines += ["## 1. Robustness: same contrasts on the price index", "",
              "| Contrast | Pairs | Profit index ΔΔ (p) | Price index ΔΔ [95% CI] (p) |",
              "| --- | --- | --- | --- |"]
    for name, x, y in CONTRASTS:
        if x not in per or y not in per:
            continue
        keys = sorted(set(per[x]) & set(per[y]))
        dp = [per[x][k][0] - per[y][k][0] for k in keys]
        dq = [per[x][k][1] - per[y][k][1] for k in keys]
        lo, hi = bootstrap_ci(dq)
        lines.append(f"| {name.strip()} | {len(keys)} | {st.mean(dp):+.2f} ({signflip_p(dp):.3f}) | "
                     f"{st.mean(dq):+.2f} [{lo:+.2f}, {hi:+.2f}] ({signflip_p(dq):.3f}) |")
    lines += ["", "Price index: 0 = Nash price, 1 = monopoly price; values above 1 are overpricing.", ""]

    # 2. anchoring -----------------------------------------------------------------------
    logs = {}
    for f in (out / "logs").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("arm"):
                logs.setdefault((r["run_id"], r["arm"]), []).append(r)
    lines += ["## 2. Anchoring: first post-swap price vs the last price on the record", "",
              "Share of replaced sellers whose first price is within 2% of their seat's last price "
              "before the swap, and the median absolute change.", "",
              "| Arm | Seller(s) checked | Within 2% | Median |change| |", "| --- | --- | --- | --- |"]
    which = {"A1": ["B"], "A2": ["B"], "A4": ["B"], "A5": ["A", "B"], "A7": ["A", "B"],
             "A8": ["A", "B"], "A6": ["A", "B"]}
    for arm in [x for x in ARM_ORDER if x in which]:
        rel = []
        for s in branches:
            if s["arm"] != arm:
                continue
            sw = s["swap_round"]
            before = next(h for h in s["history"] if h["round"] == sw - 1)
            after = next(h for h in s["history"] if h["round"] == sw)
            for i, name in enumerate("AB"):
                if name in which[arm]:
                    rel.append(abs(after["prices"][i] - before["prices"][i]) / before["prices"][i])
        if rel:
            lines.append(f"| {ARM_LABEL[arm]} | {', '.join(which[arm])} | "
                         f"{sum(r <= 0.02 for r in rel)}/{len(rel)} | {st.median(rel):.1%} |")
    lines += ["", "A6 is the comparison: its agents see no record, so any match is coincidence.", ""]

    # 3. words ----------------------------------------------------------------------------
    lines += ["## 3. What newcomers write in their first post-swap notes", "",
              "Share of first post-swap replies (replaced sellers) mentioning each theme.", ""]
    head = "| Theme | " + " | ".join(ARM_LABEL[x] for x in ["A1", "A4", "A5", "A8", "A6"]) + " |"
    lines += [head, "| --- " * 6 + "|"]
    firsts = {}
    for (run_id, arm), rows in logs.items():
        sw = min(r["round"] for r in rows)
        for r in rows:
            if r["round"] == sw and r["seller"] in which.get(arm, []):
                firsts.setdefault(arm, []).append(r)
    for theme, pat in KEYWORDS.items():
        cells = []
        for arm in ["A1", "A4", "A5", "A8", "A6"]:
            rs = firsts.get(arm, [])
            n = sum(bool(re.search(pat, r["response"], re.I)) for r in rs)
            cells.append(f"{n}/{len(rs)}")
        lines.append(f"| {theme} | " + " | ".join(cells) + " |")
    lines += ["", "Keyword matching is a rough guide, not a judge; read the quotes.", ""]
    # 4. three outcome measures side by side + dynamics ----------------------------------
    def post(s):
        return [h for h in s["history"] if h["round"] >= s["swap_round"]][-a.window:]
    lines += ["## 4. Sellers vs consumers: three measures by arm (last 20 post-swap rounds)", "",
              "| Arm | Profit index | Price index | Consumer-harm index [95% CI] | Runs priced above monopoly |",
              "| --- | --- | --- | --- | --- |"]
    harm = {}
    for arm in ARM_ORDER:
        rs = [s for s in branches if s["arm"] == arm]
        if not rs:
            continue
        P = [st.mean(h["delta"] for h in post(s)) for s in rs]
        Q = [st.mean(h["delta_price"] for h in post(s)) for s in rs]
        C = [st.mean(harm_index(s, h["prices"]) for h in post(s)) for s in rs]
        harm[arm] = {s["run_id"]: c for s, c in zip(rs, C)}
        lo, hi = bootstrap_ci(C)
        lines.append(f"| {ARM_LABEL[arm]} | {st.mean(P):+.2f} | {st.mean(Q):+.2f} | "
                     f"{st.mean(C):+.2f} [{lo:+.2f}, {hi:+.2f}] | {sum(q > 1 for q in Q)}/{len(Q)} |")
    lines += ["", "Consumer-harm index: 0 = consumers as well off as at Nash prices, 1 = as badly off as at "
              "monopoly prices; above 1 = worse than monopoly (overpricing).", "",
              "| Consumer-harm contrast | Pairs | Mean [95% CI] | Sign-flip p |", "| --- | --- | --- | --- |"]
    for name, x, y in CONTRASTS:
        if x in harm and y in harm:
            k = sorted(set(harm[x]) & set(harm[y]))
            d = [harm[x][i] - harm[y][i] for i in k]
            lo, hi = bootstrap_ci(d)
            lines.append(f"| {name.strip()} | {len(k)} | {st.mean(d):+.2f} [{lo:+.2f}, {hi:+.2f}] | "
                         f"{signflip_p(d):.3f} |")
    bases = [_load(f) for f in sorted((out / "checkpoints").glob("*.json"))]

    def blocks(hs, key, size=10):
        return [st.mean(h[key] for h in hs[i:i + size]) for i in range(0, len(hs), size)]
    lines += ["", "## 5. Price dynamics (mean price index by 10-round block)", ""]
    rows = [("All 36 base games, from round 1", [blocks(b["history"], "delta_price") for b in bases])]
    for arm in ["A0", "A5", "A6"]:
        rows.append((f"{ARM_LABEL[arm]}, from the swap",
                     [blocks([h for h in s["history"] if h["round"] >= s["swap_round"]], "delta_price")
                      for s in branches if s["arm"] == arm]))
    for label, B in rows:
        lines.append(f"- {label}: " + ", ".join(f"{st.mean(c):+.2f}" for c in zip(*B)))
    lines += ["", "Fresh agents (round 1 and A6) open far above the monopoly price and drift down; "
              "agents with a record start at the recorded price.", ""]
    (out / "results" / "extra.md").write_text("\n".join(lines))
    print("\n".join(lines))

    # quotes file: every first reply for the key arms, for picking examples by hand
    q = ["# First post-swap replies (replaced sellers)", ""]
    for arm in ["A1", "A4", "A5", "A8", "A6"]:
        q += [f"## {ARM_LABEL[arm]}", ""]
        for r in sorted(firsts.get(arm, []), key=lambda r: (r["run_id"], r["seller"])):
            q += [f"**{r['run_id']} seller {r['seller']} ({r['model']}), round {r['round']}, "
                  f"price {r['price']:.2f}**", "", "> " + r["response"].replace("\n", "\n> "), ""]
    (out / "results" / "quotes.md").write_text("\n".join(q))


if __name__ == "__main__":
    main()
