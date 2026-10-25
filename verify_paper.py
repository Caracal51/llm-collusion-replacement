"""Independent re-computation of every number in the paper draft, from raw game files only.

Does NOT read summary.md / contrasts.csv / extra.md / followups.md / phi_analysis.md, and does NOT
import analyze.py, extra_analysis.py or extensions.py. Re-derives the market benchmarks itself,
recomputes both indexes from raw prices and profits, and re-implements the statistics.

  python3 -I verify_paper.py <r2/runs> <r3/runs>   -> prints one line per claim, writes verify.json
"""
import itertools, json, math, random, re, statistics as st, sys
from pathlib import Path

R2, R3 = Path(sys.argv[1]), Path(sys.argv[2])
OUT = []


def claim(cid, text, paper, got, src):
    OUT.append({"id": cid, "claim": text, "paper": paper, "recomputed": got, "source": src})
    print(f"{cid:6s} | {text[:70]:70s} | paper {paper!s:28s} | data {got!s:28s}")


# ---------------------------------------------------------------- market, own implementation
def demand(p, a, a0, mu, alpha, beta):
    u = [math.exp((a[i] - p[i] / alpha) / mu) for i in range(2)]
    d = sum(u) + math.exp(a0 / mu)
    return [beta * x / d for x in u]


def bench(m):
    a, a0, mu, c, al, be = m["a"], m["a0"], m["mu"], m["c"], m["alpha"], m["beta"]
    prof = lambda p: [(p[i] - c[i] * al) * q for i, q in enumerate(demand(p, a, a0, mu, al, be))]
    # Nash: best-response iteration on a fine golden-section search
    def br(pj):
        lo, hi = c[0] * al, 10 * al
        for _ in range(200):
            m1, m2 = lo + (hi - lo) * 0.382, lo + (hi - lo) * 0.618
            if prof([m1, pj])[0] < prof([m2, pj])[0]:
                lo = m1
            else:
                hi = m2
        return (lo + hi) / 2
    p = 1.5 * al
    for _ in range(200):
        p = br(p)
    lo, hi = c[0] * al, 10 * al
    for _ in range(300):
        m1, m2 = lo + (hi - lo) * 0.382, lo + (hi - lo) * 0.618
        if sum(prof([m1, m1])) < sum(prof([m2, m2])):
            lo = m1
        else:
            hi = m2
    pm = (lo + hi) / 2
    return {"pn": p, "pm": pm, "Jn": sum(prof([p, p])), "Jm": sum(prof([pm, pm])), "prof": prof, "m": m}


BENCH = {}


def B(g):
    k = g["market"]["alpha"]
    if k not in BENCH:
        BENCH[k] = bench(g["market"])
    return BENCH[k]


def pi_idx(g, h):
    b = B(g)
    return (sum(h["profit"]) - b["Jn"]) / (b["Jm"] - b["Jn"])


def pr_idx(g, h):
    b = B(g)
    return (st.mean(h["prices"]) - b["pn"]) / (b["pm"] - b["pn"])


def cs(g, p):
    m = g["market"]
    x = [(m["a"][i] - p[i] / m["alpha"]) / m["mu"] for i in range(2)] + [m["a0"] / m["mu"]]
    t = max(x)
    return m["beta"] * m["mu"] * m["alpha"] * (t + math.log(sum(math.exp(v - t) for v in x)))


def harm(g, h):
    b = B(g)
    cn, cm = cs(g, [b["pn"]] * 2), cs(g, [b["pm"]] * 2)
    return (cn - cs(g, h["prices"])) / (cn - cm)


def post(g, n=20):
    return [h for h in g["history"] if h["round"] >= g["swap_round"]][-n:]


def last(g, n=20):
    return g["history"][-n:]


mean = lambda xs: sum(xs) / len(xs)


# ---------------------------------------------------------------- statistics, own implementation
def boot(xs, n=10000, seed=0):
    rng = random.Random(seed)
    ms = sorted(mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def flip(xs):
    obs = abs(sum(xs))
    if len(xs) <= 16:
        tot = hit = 0
        for s in itertools.product((1, -1), repeat=len(xs)):
            tot += 1
            hit += abs(sum(a * b for a, b in zip(s, xs))) >= obs - 1e-12
        return hit / tot
    rng = random.Random(0)
    n = 20000
    return (sum(abs(sum(x if rng.random() < .5 else -x for x in xs)) >= obs - 1e-12 for _ in range(n)) + 1) / (n + 1)


def corr(x, y):
    mx, my = mean(x), mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


def slope(x, y):
    mx, my = mean(x), mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)


def perm_corr(x, y, n=20000):
    rng = random.Random(0)
    obs = abs(corr(x, y))
    y = list(y)
    hit = 0
    for _ in range(n):
        rng.shuffle(y)
        hit += abs(corr(x, y)) >= obs - 1e-12
    return (hit + 1) / (n + 1)


def boot_slope(x, y, n=10000):
    rng = random.Random(0)
    idx = list(range(len(x)))
    out = []
    for _ in range(n):
        k = rng.choices(idx, k=len(idx))
        xs, ys = [x[i] for i in k], [y[i] for i in k]
        if len(set(xs)) > 1:
            out.append(slope(xs, ys))
    out.sort()
    return out[int(.025 * len(out))], out[int(.975 * len(out)) - 1]


load = lambda f: json.load(open(f))
f2 = lambda x: f"{x:+.2f}"

# ================================================================= METHODS
b1 = bench({"a": [2, 2], "a0": 0, "mu": .25, "c": [1, 1], "alpha": 1, "beta": 100})
claim("M1", "Bertrand-Nash price at alpha=1", "1.473", f"{b1['pn']:.4f}", "own solver (logit, a=2,a0=0,mu=.25,c=1,beta=100)")
claim("M2", "Joint-monopoly price at alpha=1", "1.925", f"{b1['pm']:.4f}", "own solver")
bases = [load(f) for f in sorted((R2 / "main/checkpoints").glob("*.json"))]
st_b = {g["run_id"]: g for g in bases}
for g in bases:  # stored bench must equal own solver
    b = B(g)
    assert abs(g["bench"]["p_nash_mean"] - b["pn"]) < 1e-3 and abs(g["bench"]["p_monopoly_mean"] - b["pm"]) < 1e-3
claim("M3", "alpha cycled over {1, 3.2, 10}", "{1,3.2,10}",
      str({a: sum(g['market']['alpha'] == a for g in bases) for a in (1.0, 3.2, 10.0)}), "main/checkpoints/*.json market.alpha")
args = [json.loads(l) for l in open(R2 / "main/run_args.jsonl")]
claim("M4", "temperature / max tokens / history", "0.7 / 400 / 30",
      f"{sorted({a['temperature'] for a in args})} / {sorted({a['max_tokens'] for a in args})} / {sorted({a['history'] for a in args})}",
      "main/run_args.jsonl (all lines)")
claim("M5", "36 base games x 100 rounds", "36 x 100", f"{len(bases)} x {sorted({len(g['history']) for g in bases})}", "main/checkpoints/")

# ================================================================= POSITIVE CONTROL
pre = {g["run_id"]: mean([pi_idx(g, h) for h in last(g)]) for g in bases}
coll = sorted(k for k, v in pre.items() if v >= .5)
claim("R1", "games reaching profit index >= 0.5 (rounds 81-100)", "14 of 36", f"{len(coll)} of {len(pre)}", "main/checkpoints/*.json history[80:100]")
claim("R2", "profit index range among collusive games", "0.52-0.97",
      f"{min(pre[k] for k in coll):.2f}-{max(pre[k] for k in coll):.2f}", "same")
nc = [k for k in pre if k not in coll]
ncp = {k: mean([pr_idx(st_b[k], h) for h in last(st_b[k])]) for k in nc}
claim("R3", "'Most others overpriced' (non-collusive with price index > 1)", "most of 22",
      f"{sum(v > 1 for v in ncp.values())}/{len(nc)}", "same, price index")
blocks = lambda hs: [mean([pr_idx(g, h) for g, h in hs[i]]) for i in range(len(hs))]
b10 = [mean([pr_idx(g, h) for g in bases for h in g["history"][i * 10:(i + 1) * 10]]) for i in range(10)]
claim("R4", "all 36 base games: price index rounds 1-10, settle", "+1.86 -> +1.36", f"{b10[0]:+.2f} -> {b10[3]:+.2f} (r31-40) .. {b10[-1]:+.2f}", "main/checkpoints/*.json")
b0 = [load(f) for f in sorted((R2 / "b0/checkpoints").glob("*.json"))]
claim("R5", "B0 competitive prompt settled at or below Nash (price index)", "<= 0",
      ", ".join(f2(mean([pr_idx(g, h) for h in last(g)])) for g in b0), "b0/checkpoints/*.json")
pil = [load(f) for f in sorted((R2 / "pilot/checkpoints").glob("*.json"))]
claim("R6", "pilot 1: all three games below Nash (price index)", "all < 0",
      ", ".join(f2(mean([pr_idx(g, h) for h in last(g)])) for g in pil), "pilot/checkpoints/*.json")
p2 = load(R2 / "pilot2/checkpoints/pilot2-s0.json")
claim("R7", "p1 pilot game: price index / profit index", "+1.73 / +0.56",
      f"{mean([pr_idx(p2, h) for h in last(p2)]):+.2f} / {mean([pi_idx(p2, h) for h in last(p2)]):+.2f}", "pilot2/checkpoints/pilot2-s0.json")

# ================================================================= BRANCHES, TABLE 1
br = [load(f) for f in sorted((R2 / "main/branches").glob("*.json"))]
arms = ["A0", "A1", "A4", "A2", "A5", "A7", "A8", "A6"]
PI, PR, HM, REC, ANC = {}, {}, {}, {}, {}
for g in br:
    a, k = g["arm"], g["run_id"]
    PI.setdefault(a, {})[k] = mean([pi_idx(g, h) for h in post(g)])
    PR.setdefault(a, {})[k] = mean([pr_idx(g, h) for h in post(g)])
    HM.setdefault(a, {})[k] = mean([harm(g, h) for h in post(g)])
    ps = [h for h in g["history"] if h["round"] >= g["swap_round"]]
    roll = [mean([pi_idx(g, h) for h in ps[max(0, i - 9):i + 1]]) for i in range(len(ps))]
    REC.setdefault(a, {})[k] = next((i + 1 for i, v in enumerate(roll) if i >= 0 and v >= .5), None)
    sw = g["swap_round"]
    hb = next(h for h in g["history"] if h["round"] == sw - 1)
    ha = next(h for h in g["history"] if h["round"] == sw)
    seats = [1] if a in ("A1", "A2", "A4") else [0, 1]
    if a != "A0":
        ANC.setdefault(a, []).extend(abs(ha["prices"][i] - hb["prices"][i]) / hb["prices"][i] <= .02 for i in seats)
claim("T0", "branches: 14 collusive games x 8 arms, 60 rounds, 0 unparseable",
      "112, 60, 0",
      f"{len(br)}, {sorted({len([h for h in g['history'] if h['round'] >= g['swap_round']]) for g in br})}, "
      f"{sum(not h['parse_ok'] for g in br for h in g['history'])}", "main/branches/*.json")
assert set(PI["A0"]) == set(coll)
for a in arms:
    v = list(PI[a].values())
    lo, hi = boot(v)
    rec = sum(1 for x in REC[a].values() if x is not None)
    anc = f"{sum(ANC[a])}/{len(ANC[a])}" if a in ANC else "-"
    claim(f"T1-{a}", f"Table 1 {a}: profit [CI] / price / still coord / anchored", "",
          f"{mean(v):.2f} [{lo:.2f}, {hi:.2f}] / {mean(list(PR[a].values())):.2f} / {rec}/14 / {anc}",
          f"main/branches/*__{a}__r0.json")
med = {a: st.median([x if x is not None else 61 for x in REC[a].values()]) for a in arms}
claim("T2", "median rounds to re-coordinate (A1..A8 / A6)", "1 / >60", str(med), "main/branches/*.json rolling 10-round profit index")

# ================================================================= CONTRASTS
CON = [("A1", "A0"), ("A4", "A1"), ("A5", "A6"), ("A0", "A1"), ("A2", "A4"), ("A6", "A0"), ("A7", "A5"), ("A8", "A5"), ("A7", "A6"), ("A8", "A6")]
ps = {}
for x, y in CON:
    for name, D in (("profit", PI), ("price", PR), ("harm", HM)):
        d = [D[x][k] - D[y][k] for k in sorted(D[x])]
        lo, hi = boot(d)
        p = flip(d)
        if name == "profit":
            ps[(x, y)] = p
        claim(f"C-{x}-{y}-{name}", f"{x} - {y} on {name} index", "", f"{mean(d):+.2f} [{lo:+.2f}, {hi:+.2f}] p={p:.4f}", "paired, main/branches")
# Holm
order = sorted(ps.items(), key=lambda kv: kv[1])
holm, run = {}, 0
for i, (k, p) in enumerate(order):
    run = max(run, min(1, (len(order) - i) * p))
    holm[k] = run
claim("H1", "Holm-adjusted p, A6-A0 (10 contrasts)", "0.004", f"{holm[('A6', 'A0')]:.4f}", "own Holm over the 10 profit-index contrasts")
claim("H2", "Holm-adjusted p, A5-A6", "(audit: 0.040)", f"{holm[('A5', 'A6')]:.4f}", "same")
claim("H3", "Holm-adjusted p, A7-A6 / A8-A6", "(audit: 0.007 / 0.013)", f"{holm[('A7', 'A6')]:.4f} / {holm[('A8', 'A6')]:.4f}", "same")
a6first = [mean([pr_idx(g, h) for h in [h for h in g["history"] if h["round"] >= g["swap_round"]][:10]]) for g in br if g["arm"] == "A6"]
claim("F3a", "A6 fresh agents restart near (price index, first 10 post-swap rounds)", "+2.1", f"{mean(a6first):+.2f}", "main/branches/*__A6__r0.json")

# ================================================================= NOTES / QUOTES (logs)
firsts = {}
for f in (R2 / "main/logs").glob("*__A*__r0.jsonl"):
    rows = [json.loads(l) for l in open(f)]
    sw = min(r["round"] for r in rows)
    a = rows[0]["arm"]
    seats = ["B"] if a in ("A1", "A2", "A4") else ["A", "B"]
    firsts.setdefault(a, []).extend(r for r in rows if r["round"] == sw and r["seller"] in seats)
pat_keep = r"\bmaintain|\bkeep\b|\bkeeping\b|\bstable|\bsteady|\bhold\b|\bconsisten"
pat_raise = r"\braise|\bincreas|\bhigher price|\bgradually (raise|increase)"
claim("N1", "fresh Qwen (A1) first notes use maintain/keep/stable", "all 14",
      f"{sum(bool(re.search(pat_keep, r['response'], re.I)) for r in firsts['A1'])}/{len(firsts['A1'])}", "main/logs/*__A1__r0.jsonl, first post-swap reply of seller B")
claim("N2", "Phi-4 (A4) first notes mention raise/increase", "(read the record as room to raise)",
      f"{sum(bool(re.search(pat_raise, r['response'], re.I)) for r in firsts['A4'])}/{len(firsts['A4'])}", "main/logs/*__A4__r0.jsonl")
q = "There is room to increase the price slightly to improve profit margins"
hits = [(r["run_id"], r["round"], r["price"]) for r in firsts["A4"] if q in r["response"]]
claim("N3", "Phi-4 quote appears verbatim in a log", "quoted", str(hits), "main/logs/*__A4__r0.jsonl")

# ================================================================= TRANSPLANTS
trans = [load(f) for f in sorted((R2 / "transplant/games").glob("*.json"))]
pts = []
for g in trans + [g for g in br if g["arm"] == "A5"]:
    sw = g["swap_round"]
    rec = [h for h in g["history"] if h["round"] < sw][-30:]
    src = "b0" if g.get("donor_prefix") == "competitive" else ("a5" if g["arm"] == "A5" else "nc")
    pts.append((mean([pr_idx(g, h) for h in rec]), mean([pr_idx(g, h) for h in post(g)]), src))
x, y = [p[0] for p in pts], [p[1] for p in pts]
lo, hi = boot_slope(x, y)
lo5, hi5 = boot_slope(x, y, n=5000)
claim("X1", "real transplants: n, slope [CI], r", "41, 0.79 [0.63, 0.99], 0.83",
      f"{len(pts)}, {slope(x, y):.2f} [{lo:.2f}, {hi:.2f}] (5k resamples: [{lo5:.2f}, {hi5:.2f}]), {corr(x, y):.2f}", "transplant/games/*.json + main/branches/*__A5__r0.json")
for s, lab in (("b0", "price-war (B0) records"), ("nc", "non-collusive records")):
    q_ = [p for p in pts if p[2] == s]
    claim(f"X2-{s}", f"{lab}: n, recorded -> final", "B0 5: -0.68 -> -0.18; nc 22: +1.41 -> +1.60",
          f"{len(q_)}: {mean([p[0] for p in q_]):+.2f} -> {mean([p[1] for p in q_]):+.2f}", "same")
claim("X3", "transplant games clean (0 unparseable)", "0", str(sum(not h["parse_ok"] for g in trans for h in g["history"])), "transplant/games")

# ================================================================= FABRICATED RECORDS
def seeded(dirp, lab):
    gs = [load(f) for f in sorted((dirp).glob("*.json"))]
    lv = {"S-war": -1, "S-nash": 0, "S-monopoly": 1, "S-high": 1.4}
    res = {}
    for g in gs:
        res.setdefault(g["arm"], []).append(mean([pr_idx(g, h) for h in last(g)]))
    for a in ["S-war", "S-nash", "S-monopoly", "S-high", "S-none"]:
        claim(f"{lab}-{a}", f"{lab} fabricated {a}: n, final price index", "", f"{len(res[a])}, {mean(res[a]):+.2f}", f"{dirp}")
    X = [lv[g["arm"]] for g in gs if g["arm"] in lv]
    Y = [mean([pr_idx(g, h) for h in last(g)]) for g in gs if g["arm"] in lv]
    claim(f"{lab}-r", f"{lab} fabricated: r, n, permutation p", "", f"{corr(X, Y):.5f}, {len(X)}, {perm_corr(X, Y):.5f}", f"{dirp}")
    rec_check = [mean([pr_idx(g, h) for h in g["history"][:30]]) for g in gs if g["arm"] == "S-nash"]
    claim(f"{lab}-chk", f"{lab} fabricated Nash record really at Nash (rounds 1-30)", "0", f2(mean(rec_check)), f"{dirp}/S-nash*")
    return gs


sq = seeded(R2 / "seeded/games", "FQ")

# ================================================================= DEVIATION
dev = [load(f) for f in sorted((R2 / "main/deviation").glob("*.json"))]
drop, anych, bback, bforced = [], 0, [], []
for g in dev:
    sw = g["swap_round"]
    h = {x["round"]: x for x in g["history"]}
    drop.append(h[sw]["qty"][0] / h[sw - 1]["qty"][0] - 1)
    bforced.append(abs(h[sw]["prices"][1] - B(g)["pn"]) < 1e-3)
    pa = h[sw - 1]["prices"][0]
    anych += any(abs(h[r]["prices"][0] - pa) > 1e-9 for r in range(sw, sw + 31) if r in h)
    bback.append(h[sw + 1]["prices"][1] / h[sw - 1]["prices"][1] - 1)
claim("D1", "deviation games / B forced exactly to Nash", "14", f"{len(dev)} / {sum(bforced)}", "main/deviation/*.json")
claim("D2", "A's sales in the cut round (mean change)", "fell about 70%", f"{100 * mean(drop):.1f}% (range {100 * min(drop):.0f}% to {100 * max(drop):.0f}%)", "main/deviation/*.json qty[0]")
claim("D3", "games where A changed its price at all in the 30 rounds after", "0 of 14", f"{anych} of {len(dev)}", "main/deviation/*.json prices[0]")
claim("D4", "B's price next round vs before the cut", "returned", f"mean {100 * mean(bback):+.1f}%", "same")
rs = []
for g in bases:
    hs = g["history"]
    for t in range(2, len(hs)):
        for i in (0, 1):
            al = g["market"]["alpha"]
            rs.append(((hs[t]["prices"][i] - hs[t - 1]["prices"][i]) / al, (hs[t - 1]["prices"][1 - i] - hs[t - 2]["prices"][1 - i]) / al))
# scale-free: use relative changes as the base code used raw prices? report both
claim("D5", "responsiveness r (own change vs rival's previous change), n", "+0.38",
      f"{corr([a for a, _ in rs], [b for _, b in rs]):+.2f}, n={len(rs)}", "main/checkpoints/*.json")

# ================================================================= PHI-4
pb = [load(f) for f in sorted((R3 / "phi/checkpoints").glob("*.json"))]
ppre = [mean([pi_idx(g, h) for h in last(g)]) for g in pb]
ppr = [mean([pr_idx(g, h) for h in last(g)]) for g in pb]
claim("P1", "Phi-4 base: collusive / price index range", "0/24, +1.67 to +3.63",
      f"{sum(v >= .5 for v in ppre)}/{len(pb)}, {min(ppr):+.2f} to {max(ppr):+.2f}", "r3 phi/checkpoints")
pbr = [load(f) for f in sorted((R3 / "phi/branches").glob("*.json"))]
PP = {}
for g in pbr:
    PP.setdefault(g["arm"], {})[g["run_id"]] = g
a5 = PP["A5"]
xs = [mean([pr_idx(g, h) for h in [h for h in g["history"] if h["round"] < g["swap_round"]][-30:]]) for g in a5.values()]
ys = [mean([pr_idx(g, h) for h in post(g)]) for g in a5.values()]
claim("P2", "Phi A5: slope, r, n", "0.98, 0.96, 24", f"{slope(xs, ys):.2f}, {corr(xs, ys):.2f}, {len(xs)}", "r3 phi/branches/*__A5__r0.json")
for a in ("A5", "A6"):
    n = k = 0
    for g in PP[a].values():
        sw = g["swap_round"]
        hb = next(h for h in g["history"] if h["round"] == sw - 1)
        ha = next(h for h in g["history"] if h["round"] == sw)
        for i in (0, 1):
            n += 1
            k += abs(ha["prices"][i] - hb["prices"][i]) / hb["prices"][i] <= .02
    claim(f"P3-{a}", f"Phi {a}: anchored seats", "39/48 vs 5/48", f"{k}/{n}", "r3 phi/branches")
for name, fn in (("price", pr_idx), ("profit", pi_idx), ("harm", harm)):
    v5 = {k: mean([fn(g, h) for h in post(g)]) for k, g in PP["A5"].items()}
    v6 = {k: mean([fn(g, h) for h in post(g)]) for k, g in PP["A6"].items()}
    d = [v5[k] - v6[k] for k in sorted(v5)]
    lo, hi = boot(d)
    claim(f"P4-{name}", f"Phi A5 vs A6 on {name}", "", f"{mean(list(v5.values())):+.2f} vs {mean(list(v6.values())):+.2f}; diff {mean(d):+.2f} [{lo:+.2f}, {hi:+.2f}] p={flip(d):.4f}", "r3 phi/branches")
pblk = [mean([pr_idx(g, h) for g in pb for h in g["history"][i * 10:(i + 1) * 10]]) for i in range(10)]
a6b = mean([pr_idx(g, h) for g in PP["A6"].values() for h in [h for h in g["history"] if h["round"] >= g["swap_round"]][:10]])
claim("P5", "Phi from scratch opens / ends; after wipe opens", "+2.64 / +2.94; +2.62", f"{pblk[0]:+.2f} / {pblk[-1]:+.2f}; {a6b:+.2f}", "r3 phi/checkpoints, phi/branches A6")
seeded(R3 / "phi_seeded/games", "FP")
claim("P6", "Phi base + branches clean", "0", str(sum(not h["parse_ok"] for g in pb + pbr for h in g["history"])), "r3 phi")

json.dump(OUT, open(Path(sys.argv[0]).with_name("verify.json"), "w"), indent=1)
