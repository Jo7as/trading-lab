"""S0 pre-flight: can a signal at this horizon even pay the spread?

Before any strategy code: measure the *size of the move* an instrument produces
over the intended holding horizon and compare it with the round-trip cost. It is
a **necessary condition** — no directional signal can capture more than the
average absolute move — so if this fails, nothing built on top can work.

Two uses:

- ``horizon_gate(m1, minutes)`` — per-bar ``E|Δmid|`` vs the spread (this is the
  cheap, pre-code check; run it on the tick/1-minute data before writing code);
- ``trades_move_gate(trades, m1)`` — the post-hoc version on a real trade log.

Costs are taken from the data's own bid/ask. Two cost levels are reported: a
taker exit (both legs cross the spread) and a limit exit (only the entry crosses)
— the truth is between them, and the "Vary costs" Monte Carlo later scales them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_K = 3.0          # required E|move| / cost ratio
NEAR_K = 2.0             # below this: hopeless; between NEAR_K and K: marginal


@dataclass(frozen=True)
class GateResult:
    minutes: int
    move_pct: float               # median |mid move| over the horizon, in %
    move_p25_pct: float
    cost_taker_pct: float         # full spread (both legs taker)
    cost_limit_pct: float         # half spread (limit exit)
    ratio_taker: float
    ratio_limit: float

    def verdict(self, k: float = DEFAULT_K) -> str:
        r = self.ratio_taker
        if r >= k:
            return "PASS"
        if r >= NEAR_K:
            return "MARGINAL"
        return "FAIL"

    def __str__(self) -> str:
        return (f"{self.minutes:>3d}m  move {self.move_pct:6.3f}%  "
                f"cost {self.cost_taker_pct:6.3f}% (limit {self.cost_limit_pct:6.3f}%)  "
                f"ratio {self.ratio_taker:6.2f} (limit {self.ratio_limit:6.2f})  {self.verdict()}")


def mid(m1: pd.DataFrame) -> pd.Series:
    return (m1["close"] + m1["ask_close"]) / 2.0


def spread_pct(m1: pd.DataFrame) -> pd.Series:
    """Full spread as a fraction of the mid price, per bar."""
    return (m1["ask_close"] - m1["close"]) / mid(m1)


def horizon_moves(m1: pd.DataFrame, minutes: int) -> pd.Series:
    """|mid move| over ``minutes`` 1-minute bars, per bar (overlapping)."""
    m = mid(m1)
    return (m.shift(-minutes) - m).abs() / m


def horizon_gate(m1: pd.DataFrame, minutes: int) -> GateResult:
    mv = horizon_moves(m1, minutes).dropna()
    sp = spread_pct(m1).dropna()
    move = float(mv.median())
    move_p25 = float(mv.quantile(0.25))
    cost_taker = float(sp.median())
    cost_limit = cost_taker / 2.0
    return GateResult(
        minutes=minutes, move_pct=move * 100, move_p25_pct=move_p25 * 100,
        cost_taker_pct=cost_taker * 100, cost_limit_pct=cost_limit * 100,
        ratio_taker=move / cost_taker if cost_taker else float("inf"),
        ratio_limit=move / cost_limit if cost_limit else float("inf"),
    )


def trades_move_gate(trades: list[dict], m1: pd.DataFrame) -> dict:
    """Post-hoc: the same ratio, measured on trades that were actually taken."""
    if not trades:
        return {"trades": 0}
    t = pd.DataFrame(trades)
    entry = t["entry_price"].astype(float)
    exit_ = t["exit_price"].astype(float)
    move = ((exit_ - entry) / entry).abs()
    idx = m1.index.as_unit("ns").asi8
    pos = np.searchsorted(idx, (t["entry_time"].astype(float) * 1e9).astype("int64"), "left")
    sp = spread_pct(m1).to_numpy()[np.clip(pos, 0, len(idx) - 1)]
    taker = ~t["exit_reason"].astype(str).str.lower().str.contains("tp|target")
    cost = np.where(taker, sp, sp / 2.0)
    return {
        "trades": len(t),
        "move_pct_median": round(float(move.median()) * 100, 4),
        "cost_pct_median": round(float(np.median(cost)) * 100, 4),
        "ratio": round(float(move.median() / np.median(cost)), 2) if np.median(cost) else None,
        "ratio_p25": round(float(move.quantile(0.25) / np.median(cost)), 2),
    }
