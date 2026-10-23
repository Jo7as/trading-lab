"""Gold Supertrend baseline: net of the ACTUAL spread at each trade's minutes.

Instead of one median spread, look up the real 1-minute ask-bid at every trade's
entry and exit minute and charge it with the correct order-type logic:
entry is always a taker (1 half-spread), an exit at a stop/time is a taker
(1 half-spread), a limit (take-profit) exit pays nothing. Also reports a slippage
band on stop exits and a per-year split after costs.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "studies")
from pass_speed import load_cached  # noqa: E402

from trading_lab.strategies.supertrend_volume import SupertrendVolumeConfig, run_backtest  # noqa: E402

BASE = dict(supertrend_period=7, supertrend_mult=1.5, volume_ma_period=20, volume_mult=1.0,
            ema_fast=3, ema_slow=3, atr_period=14, min_atr_pct=0.0, pullback_entry=False,
            account_size=10000.0, risk_per_trade=0.01, sl_atr_mult=1.5, tp1_r=2.0,
            tp1_close_pct=0.60, max_hold_hours=48, max_daily_loss=0.03, cooldown_hours=0,
            max_concurrent_positions=1, long_only=False)


def main() -> None:
    df = load_cached("XAUUSD")
    half = (df["ask_close"] - df["close"]) / 2.0          # half-spread in USD, per 1m bar
    idx_ns = df.index.as_unit("ns").asi8

    h = df.resample("1h").agg({"open": "first", "high": "max", "low": "min",
                               "close": "last", "volume": "sum", "ask_close": "last"}).dropna()
    t = pd.DataFrame(run_backtest(h, SupertrendVolumeConfig(**BASE))["trades"])
    entry_ns = (t["entry_time"].astype(float) * 1e9).astype("int64")
    exit_ns = (t["exit_time"].astype(float) * 1e9).astype("int64")

    def half_at(ns: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(idx_ns, ns, side="left").clip(0, len(idx_ns) - 1)
        return half.to_numpy()[pos]

    h_e, h_x = half_at(entry_ns), half_at(exit_ns)
    taker_x = ~t["exit_reason"].astype(str).str.contains("tp")   # 0 taker legs for a limit exit
    price = t["entry_price"].astype(float)
    size = t["size"].astype(float)

    base_cost = (h_e + np.where(taker_x, h_x, 0.0)) / price * size * 100      # % of account
    # slippage band: each taker leg may additionally slip by k × half-spread
    band = {}
    for k in (0.0, 0.5, 1.0):
        extra = (k * h_e + k * np.where(taker_x, h_x, 0.0)) / price * size * 100
        band[k] = float((base_cost + extra).sum())

    gross = float(t["pnl_pct"].sum())
    print(f"trades {len(t)}  taker exits {taker_x.mean():.0%}  median half-spread "
          f"{np.median(h_e):.3f} USD ({np.median(h_e)/price.median()*100:.4f}%)")
    print(f"gross {gross:+.1f}%")
    for k, c in band.items():
        print(f"  net (slippage {k:.1f}× half-spread on taker legs): {gross - c:+.1f}%   (cost {c:.1f}%)")

    t["year"] = pd.to_datetime(t["entry_time"], unit="s", utc=True).dt.year
    t["cost"] = base_cost
    print("\nper year (gross / net at slippage 0.5×):")
    for y, g in t.groupby("year"):
        net = g["pnl_pct"].sum() - g["cost"].sum() * 1.5
        print(f"  {y}: n={len(g):4d} gross {g['pnl_pct'].sum():+7.1f}%  net {net:+7.1f}%")


if __name__ == "__main__":
    main()
