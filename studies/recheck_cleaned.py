"""Step 1: re-run the headline results on CLEANED data (corrupt bars + roll window).

Prints, all on cleaned frames (rows dropped, not NaN-filled, so indicators stay
sane):

- the S0 cost gate per instrument/horizon (does the conclusion hold on clean data?),
- the gold Supertrend baseline gross/net,
- the ORB-fade expectancy per market,
- the pair-spread gross/net.

  .venv/bin/python studies/recheck_cleaned.py
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "studies")
from pass_speed import load_cached  # noqa: E402

from trading_lab.cost_gate import horizon_gate  # noqa: E402
from trading_lab.data_quality import clean  # noqa: E402
from trading_lab.strategies.orb_rejection_ifvg import OrbRejConfig, run_backtest as run_fade  # noqa: E402
from trading_lab.strategies.pair_spread import PairSpreadConfig, run_backtest as run_pair  # noqa: E402
from trading_lab.strategies.supertrend_volume import (  # noqa: E402
    SupertrendVolumeConfig, run_backtest as run_st,
)

TZ = {"US100": "America/New_York", "US30": "America/New_York", "SP500": "America/New_York",
      "GER40": "Europe/Berlin", "UK100": "Europe/London", "XAUUSD": "America/New_York",
      "XAGUSD": "America/New_York", "COPPER": "America/New_York", "BRENT": "America/New_York",
      "UST10Y": "America/New_York", "JP225": "Asia/Tokyo"}


def cleaned(asset: str) -> pd.DataFrame:
    df = load_cached(asset)
    return clean(df, tz=TZ.get(asset, "UTC")).dropna()


def main() -> None:
    assets = tuple(TZ)
    print("### S0 cost gate on CLEANED data (taker-exit ratio, k=3 → PASS)\n")
    for a in assets:
        try:
            df = cleaned(a)
        except Exception as e:  # noqa: BLE001
            print(f"{a}: {e}")
            continue
        row = [f"{a:7s}"]
        for h in (15, 60, 240):
            g = horizon_gate(df, h)
            row.append(f"{h}m {g.ratio_taker:5.2f} {g.verdict():8s}")
        print("  ".join(row))

    print("\n### gold Supertrend hf_ls01, cleaned, 1h")
    st = dict(supertrend_period=7, supertrend_mult=1.5, volume_ma_period=20, volume_mult=1.0,
              ema_fast=3, ema_slow=3, atr_period=14, min_atr_pct=0.0, pullback_entry=False,
              account_size=10000.0, risk_per_trade=0.01, sl_atr_mult=1.5, tp1_r=2.0,
              tp1_close_pct=0.60, max_hold_hours=48, max_daily_loss=0.03, cooldown_hours=0,
              max_concurrent_positions=1, long_only=False)
    x = cleaned("XAUUSD")
    h = x.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                              "volume": "sum", "ask_close": "last"}).dropna()
    t = pd.DataFrame(run_st(h, SupertrendVolumeConfig(**st))["trades"])
    half = ((x["ask_close"] - x["close"]) / 2.0).to_numpy()
    idx = x.index.as_unit("ns").asi8

    def half_at(sec):
        pos = np.searchsorted(idx, (sec * 1e9).astype("int64"), "left")
        return half[np.clip(pos, 0, len(idx) - 1)]

    he = half_at(t["entry_time"].astype(float).to_numpy())
    hx = half_at(t["exit_time"].astype(float).to_numpy())
    taker = ~t["exit_reason"].astype(str).str.lower().str.contains("tp")
    cost = (he + np.where(taker, hx, 0.0)) / t["entry_price"].astype(float) * t["size"] * 100
    g = float(t["pnl_pct"].astype(float).sum())
    t["year"] = pd.to_datetime(t["entry_time"], unit="s", utc=True).dt.year
    pos_years = sum(1 for _, gg in t.groupby("year")
                    if gg["pnl_pct"].astype(float).sum() - gg["cost"].sum() > 0) if "cost" in t else 0
    print(f"  trades {len(t)}  gross {g:+.1f}%  cost {cost.sum():.1f}%  "
          f"net(0x) {g-cost.sum():+.1f}%  net(0.5x slip) {g-1.5*cost.sum():+.1f}%")

    print("\n### ORB-fade (naive) on cleaned data, price-scaled stops, next-open")
    for a in ("US100", "SP500", "GER40"):
        df = cleaned(a)
        price = float(df["close"].median())
        r = run_fade(df, OrbRejConfig(asset=a, mode="naive", entry_mode="next_open",
                                      tick_size=round(0.25 * float((df["high"] - df["low"]).median()), 5),
                                      min_stop_points=round(2 * float((df["high"] - df["low"]).median()), 5),
                                      max_stop_points=round(10 * float((df["high"] - df["low"]).median()), 5)))
        rs = np.array([float(x["r"]) for x in r["trades"]])
        if not len(rs):
            print(f"  {a}: no trades")
            continue
        pf = rs[rs > 0].sum() / abs(rs[rs < 0].sum()) if (rs < 0).any() else float("inf")
        print(f"  {a:7s} n={len(rs):5d} exp {rs.mean():+.3f}R WR {100*(rs>0).mean():4.1f}% PF {pf:.2f}")

    print("\n### pair spreads on cleaned data, 15m")
    frames = {a: cleaned(a) for a in ("US100", "SP500", "XAUUSD", "XAGUSD")}
    for pair in (("US100", "SP500"), ("XAUUSD", "XAGUSD")):
        for label, cfg in (("real", PairSpreadConfig()), ("mid", PairSpreadConfig(mid_prices=True))):
            m = run_pair(frames, cfg, pair=pair)["metrics"]
            print(f"  {pair[0]}/{pair[1]:7s} {label:5s} n={m['total_trades']:5d} "
                  f"pnl {m['total_pnl_pct']:+8.1f}%  PF {m['profit_factor']}")


if __name__ == "__main__":
    main()
