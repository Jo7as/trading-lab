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
The per-leg rule behind those two levels is written down once below
(``leg_cost`` / ``round_trip_spread``) and every study should use it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_K = 3.0          # required E|move| / cost ratio
NEAR_K = 2.0             # below this: hopeless; between NEAR_K and K: marginal

# ── The cost convention (single source for the whole repo) ──────────────
#
# Everything is measured in the **mid** space: the move (R numerator, E|Δmid|) is
# mid-to-mid, and a leg costs half a spread when it crosses the book and nothing
# when it rests in it:
#
#   entry \ exit   limit / TP   market / stop / time
#   limit              0.0              0.5
#   market             0.5              1.0        (units of the FULL spread)
#
# So a resting entry *and* a resting exit are free in this model, and a both-legs
# market trade pays one full spread. Two consequences to keep in mind:
#
# * A limit strategy must report its **fill rate** — the spread it saves is paid
#   for in fills and adverse selection (`studies/h010_fill_rate.py` measures it).
# * If a study instead measures the move on the **bid** series, the same trade
#   needs a different charge (a long pays 1.0 spread for a market entry and 0 for
#   the exit, because both of its exits are already bid-side). Mixing the two
#   spaces is how a study ends up charging two spreads for one round trip, which
#   is what `run_h001`, `mfe_mae` and `first_passage_grid` used to do.
LEG_MARKET = 0.5
LEG_LIMIT = 0.0
_LIMIT_KINDS = ("limit", "tp", "target", "tp1", "tp2")
_MARKET_KINDS = ("market", "stop", "taker", "session", "time", "eod", "flat")


def leg_cost(kind: str) -> float:
    """Cost of one leg in units of the full spread (see the table above)."""
    k = str(kind).strip().lower()
    if k in _LIMIT_KINDS:
        return LEG_LIMIT
    if k in _MARKET_KINDS:
        return LEG_MARKET
    raise ValueError(f"unknown order type {kind!r}: add it to the convention table")


def round_trip_spread(entry: str = "market", exit_: str = "market") -> float:
    """Round-trip cost in units of the full spread, from the two leg types."""
    return leg_cost(entry) + leg_cost(exit_)


@dataclass(frozen=True)
class GateResult:
    minutes: int
    move_pct: float               # median |mid move| over the horizon, in %
    move_p25_pct: float
    cost_taker_pct: float         # full spread (both legs taker)
    cost_limit_pct: float         # half spread (limit exit)
    ratio_taker: float
    ratio_limit: float
    p_reach: float = float("nan")  # share of windows whose own |move| covers the spread

    def verdict(self, k: float = DEFAULT_K, *, limit: bool = False) -> str:
        """PASS/MARGINAL/FAIL for the given bar.

        ``limit=True`` judges the ratio with a resting exit (half spread). The
        default stays the conservative taker reading, so existing callers are
        unchanged; a limit-entry/exit strategy should ask for the limit verdict
        instead of being taxed as if it crossed the book twice (audit F3).
        """
        r = self.ratio_limit if limit else self.ratio_taker
        if r >= k:
            return "PASS"
        if r >= NEAR_K:
            return "MARGINAL"
        return "FAIL"

    def __str__(self) -> str:
        reach = "  n/a" if self.p_reach != self.p_reach else f"{self.p_reach:5.0%}"
        return (f"{self.minutes:>3d}m  move {self.move_pct:6.3f}%  "
                f"cost {self.cost_taker_pct:6.3f}% (limit {self.cost_limit_pct:6.3f}%)  "
                f"ratio {self.ratio_taker:6.2f} (limit {self.ratio_limit:6.2f})  "
                f"reach {reach}  {self.verdict()}")


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
    med_spread = float(sp.median())
    cost_taker = med_spread * round_trip_spread("market", "market")
    cost_limit = med_spread * round_trip_spread("market", "limit")
    # p_reach: the share of windows whose own |move| is at least that window's own
    # round-trip cost. The mean |move| says nothing about a *single* trade paying;
    # this does - if only 30 % of windows move far enough, an edge is unlikely.
    per_bar_cost = spread_pct(m1).reindex(mv.index) * round_trip_spread("market", "market")
    p_reach = float((mv >= per_bar_cost).mean()) if len(mv) else float("nan")
    return GateResult(
        minutes=minutes, move_pct=move * 100, move_p25_pct=move_p25 * 100,
        cost_taker_pct=cost_taker * 100, cost_limit_pct=cost_limit * 100,
        ratio_taker=move / cost_taker if cost_taker else float("inf"),
        ratio_limit=move / cost_limit if cost_limit else float("inf"),
        p_reach=p_reach,
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
    limit_exit = t["exit_reason"].astype(str).str.lower().str.contains("tp|target")
    per_trade = np.where(
        limit_exit.to_numpy(),
        round_trip_spread("market", "limit"),
        round_trip_spread("market", "market"),
    )
    cost = sp * per_trade
    return {
        "trades": len(t),
        "move_pct_median": round(float(move.median()) * 100, 4),
        "cost_pct_median": round(float(np.median(cost)) * 100, 4),
        "ratio": round(float(move.median() / np.median(cost)), 2) if np.median(cost) else None,
        "ratio_p25": round(float(move.quantile(0.25) / np.median(cost)), 2),
        "p_reach": round(float((move.to_numpy() >= cost).mean()), 3),
    }
