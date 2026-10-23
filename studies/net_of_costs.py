"""Net any strategy's trade log of the real spread — without touching its code.

For each trade: an entry is a taker (pays half the spread), a stop/time exit is a
taker (another half), a take-profit/limit exit pays nothing. The spread is looked
up from the 1-minute ask-bid at the trade's entry/exit minute. Slippage is added
as a band (0 / 0.5 / 1.0 × half-spread per taker leg), because that is the margin
on which thin edges live or die.

  .venv/bin/python studies/net_of_costs.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "studies")
from pass_speed import load_cached  # noqa: E402

from trading_lab.strategies.opening_range_breakout import (  # noqa: E402
    OpeningRangeBreakoutConfig, run_backtest as run_orb,
)
from trading_lab.strategies.silver_bullet import (  # noqa: E402
    SilverBulletConfig, run_backtest as sb,
)

LIMIT_HINTS = ("tp", "target")


def net_trades(trades: list[dict], m1: pd.DataFrame, label: str) -> pd.DataFrame:
    t = pd.DataFrame(trades)
    t["pnl_pct"] = t["pnl_pct"].astype(float)
    half = ((m1["ask_close"] - m1["close"]) / 2.0).to_numpy()
    idx = m1.index.as_unit("ns").asi8

    def half_at(sec: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(idx, (sec * 1e9).astype("int64"), side="left")
        return half[np.clip(pos, 0, len(idx) - 1)]

    h_e = half_at(t["entry_time"].astype(float).to_numpy())
    h_x = half_at(t["exit_time"].astype(float).to_numpy())
    limit_x = t["exit_reason"].astype(str).str.lower().str.contains("|".join(LIMIT_HINTS))
    price = t["entry_price"].astype(float)
    size = t["size"].astype(float) if "size" in t else pd.Series(1.0, index=t.index)

    taker_legs_cost = (h_e + np.where(limit_x, 0.0, h_x)) / price * size * 100   # % account
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
    net_trades(sb(_rs(xau, "15min"), xau, SilverBulletConfig())["trades"], xau,
               "silver_bullet / XAUUSD (15m+1m)")
    # note: orb_2r already models the spread itself (records spread + cost_r_est) — no netting needed
    print("\norb_2r: already cost-modelled in-engine (spread + cost_r_est) — not netted here")


if __name__ == "__main__":
    main()
