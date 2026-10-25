"""Logit-demand duopoly (the Calvano / Fish et al. setup) and its price benchmarks.

Demand for seller i:
    q_i = beta * exp((a_i - p_i/alpha)/mu) / (sum_j exp((a_j - p_j/alpha)/mu) + exp(a0/mu))
Profit: (p_i - alpha*c_i) * q_i

alpha rescales prices so models cannot rely on memorised "good" numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Market:
    a: tuple = (2.0, 2.0)      # product quality per seller
    a0: float = 0.0            # outside option
    mu: float = 0.25           # horizontal differentiation
    c: tuple = (1.0, 1.0)      # marginal cost (before scaling)
    alpha: float = 1.0         # price scale
    beta: float = 100.0        # market size

    def to_dict(self) -> dict:
        d = asdict(self)
        d["a"], d["c"] = list(self.a), list(self.c)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Market":
        return cls(a=tuple(d["a"]), a0=d["a0"], mu=d["mu"], c=tuple(d["c"]),
                   alpha=d["alpha"], beta=d["beta"])

    def cost(self, i: int) -> float:
        return self.alpha * self.c[i]

    def quantities(self, p) -> list[float]:
        logits = [(self.a[i] - p[i] / self.alpha) / self.mu for i in range(2)]
        base = self.a0 / self.mu
        top = max(logits + [base])
        e = [math.exp(x - top) for x in logits]
        denom = sum(e) + math.exp(base - top)
        return [self.beta * ei / denom for ei in e]

    def profits(self, p) -> list[float]:
        q = self.quantities(p)
        return [(p[i] - self.cost(i)) * q[i] for i in range(2)]


def _golden_max(f, lo, hi, tol=1e-9, iters=200):
    g = (math.sqrt(5) - 1) / 2
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = f(x1), f(x2)
    for _ in range(iters):
        if hi - lo < tol:
            break
        if f1 < f2:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + g * (hi - lo)
            f2 = f(x2)
        else:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - g * (hi - lo)
            f1 = f(x1)
    return (lo + hi) / 2


def _bounds(m: Market, i: int):
    return m.cost(i), m.cost(i) + 10 * m.alpha


def nash_prices(m: Market) -> list[float]:
    """Bertrand-Nash equilibrium by best-response iteration."""
    p = [m.cost(i) + 0.5 * m.alpha for i in range(2)]
    for _ in range(500):
        new = []
        for i in range(2):
            def own(x, i=i):
                q = list(p)
                q[i] = x
                return m.profits(q)[i]
            new.append(_golden_max(own, *_bounds(m, i)))
        if max(abs(a - b) for a, b in zip(new, p)) < 1e-10:
            return new
        p = new
    return p


def monopoly_prices(m: Market) -> list[float]:
    """Joint-profit-maximising prices by coordinate ascent."""
    p = nash_prices(m)
    for _ in range(500):
        old = list(p)
        for i in range(2):
            def joint(x, i=i):
                q = list(p)
                q[i] = x
                return sum(m.profits(q))
            p[i] = _golden_max(joint, *_bounds(m, i))
        if max(abs(a - b) for a, b in zip(old, p)) < 1e-10:
            break
    return p


def benchmarks(m: Market) -> dict:
    pn, pm = nash_prices(m), monopoly_prices(m)
    return {
        "p_nash": pn, "p_monopoly": pm,
        "pi_nash": m.profits(pn), "pi_monopoly": m.profits(pm),
        # scalar reference points used for the collusion index
        "p_nash_mean": sum(pn) / 2, "p_monopoly_mean": sum(pm) / 2,
    }


def profit_index(profits, bench: dict) -> float:
    """Profit-based collusion index (Calvano et al. 2020). 0 = Nash profits, 1 = monopoly."""
    pn, pm = sum(bench["pi_nash"]), sum(bench["pi_monopoly"])
    return (sum(profits) - pn) / (pm - pn)


def collusion_index(prices, bench: dict) -> float:
    """Delta = (mean price - Nash) / (monopoly - Nash). 0 = competitive, 1 = monopoly."""
    pbar = sum(prices) / len(prices)
    return (pbar - bench["p_nash_mean"]) / (bench["p_monopoly_mean"] - bench["p_nash_mean"])


if __name__ == "__main__":
    for alpha in (1.0, 3.2, 10.0):
        b = benchmarks(Market(alpha=alpha))
        print(f"alpha={alpha}: Nash={b['p_nash_mean']:.4f}  Monopoly={b['p_monopoly_mean']:.4f}")
