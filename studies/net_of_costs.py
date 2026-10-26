"""Net any strategy's trade log of the real spread — without touching its code.

The per-leg rule is the repo convention in :mod:`trading_lab.cost_gate`: a leg that
crosses the book costs half a spread, a resting leg (limit/TP) costs nothing, so a
round trip is 0.5 (limit exit), 1.0 (both market) or 0.0 (both resting) full
spreads. The spread is looked up from the 1-minute ask-bid at the trade's entry
minute. The slippage band scales that total by 1.0 / 1.5 / 2.0 — the margin on
which thin edges live or die.

``entry_leg`` matters: a breakout strategy enters at the close (market, half a
spread), a resting-limit strategy does not. Pass it per strategy instead of
assuming — the old version charged half a spread on *every* entry, which
over-charged the limit entry of ``silver_bullet`` by half a spread.

  .venv/bin/python studies/net_of_costs.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "studies")

sys.path.insert(0, ".")  # runnable as a script from the repo root

from pass_speed import load_cached  # noqa: E402

from trading_lab.cost_gate import round_trip_spread  # noqa: E402
from trading_lab.strategies.opening_range_breakout import (  # noqa: E402
    OpeningRangeBreakoutConfig, run_backtest as run_orb,
)
from trading_lab.strategies.silver_bullet import (  # noqa: E402
    SilverBulletConfig, run_backtest as sb,
)

LIMIT_HINTS = ("tp", "target")


def net_trades(trades: list[dict], m1: pd.DataFrame, label: str,
               *, entry_leg: str = "market") -> pd.DataFrame:
    """Net one trade log. ``entry_leg`` is "market" or "limit" (see the docstring)."""
    t = pd.DataFrame(trades)
    t["pnl_pct"] = t["pnl_pct"].astype(float)
    half = ((m1["ask_close"] - m1["close"]) / 2.0).to_numpy()
    idx = m1.index.as_unit("ns").asi8

    def half_at(sec: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(idx, (sec * 1e9).astype("int64"), side="left")
        return half[np.clip(pos, 0, len(idx) - 1)]

    h_e = half_at(t["entry_time"].astype(float).to_numpy())
    limit_x = t["exit_reason"].astype(str).str.lower().str.contains("|".join(LIMIT_HINTS))
    price = t["entry_price"].astype(float)
    size = t["size"].astype(float) if "size" in t else pd.Series(1.0, index=t.index)

    # full spread at the entry minute × the convention's round-trip multiplier
    mult = np.where(limit_x, round_trip_spread(entry_leg, "limit"),
                    round_trip_spread(entry_leg, "market"))
    taker_legs_cost = (h_e * 2.0 * mult) / price * size * 100                    # % account
    t["cost_0"] = taker_legs_cost
    t["cost_50"] = taker_legs_cost * 1.5
    t["cost_100"] = taker_legs_cost * 2.0
    t["year"] = pd.to_datetime(t["entry_time"], unit="s", utc=True).dt.year
    gross = t["pnl_pct"].sum()
    print(f"\n{label}: n={len(t)}  median half-spread {np.median(h_e):.4f}  "
          f"limit exits {limit_x.mean():.0%}")
    for k in ("cost_0", "cost_50", "cost_100"):
        tag = k.split("_")[1]
        print(f"  net (slippage {tag}% of half-spread/taker leg): "
              f"{gross - t[k].sum():+8.1f}%   (gross {gross:+.1f}%)")
    pos_years = sum(1 for _, g in t.groupby("year")
                    if g["pnl_pct"].sum() - g["cost_50"].sum() > 0)
    print(f"  positive years net (0.5×): {pos_years}/{t['year'].nunique()}")
    return t


def _rs(df, tf):
    return df.resample(tf).agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                "volume": "sum", "ask_open": "first", "ask_high": "max",
                                "ask_low": "min", "ask_close": "last"}).dropna()


def main() -> None:
    us100, xau = load_cached("US100"), load_cached("XAUUSD")
    net_trades(run_orb(us100, OpeningRangeBreakoutConfig())["trades"], us100,
               "opening_range_breakout / US100")
    # silver_bullet enters with a resting limit at the FVG, not at the close
    net_trades(sb(_rs(xau, "15min"), xau, SilverBulletConfig())["trades"], xau,
               "silver_bullet / XAUUSD (15m+1m)", entry_leg="limit")
    # note: orb_2r already models the spread itself (records spread + cost_r_est) — no netting needed
    print("\norb_2r: already cost-modelled in-engine (spread + cost_r_est) — not netted here")


if __name__ == "__main__":
    main()
