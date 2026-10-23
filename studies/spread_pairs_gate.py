"""Spread pairs: does the reversion move beat the combined cost of BOTH legs?

Dollar-neutral framing: long 1 $ of A / short 1 $ of B → the tradable return is
Δ log(A/B). We measure, per pair and timeframe:

  σ_d      std of the deviation from its rolling mean (in log-return units)
  half-life  bars for a deviation to halve (AR(1))
  signals  share of bars with |z| > 2  (trade frequency)
  cost     round-trip cost per 1 $ notional = (half-spread A + half-spread B)
  gate     2·σ_d / cost   — NOTE: misleading on its own! σ_d is the deviation
                            *volatility*, not the realized reversion. The engine
                            test (studies/../pair_spread) showed gross PF ≈ 1.0
                            despite 2σ/cost of 8-54. Measure realized reversion.

  .venv/bin/python studies/spread_pairs_gate.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "studies")
from pass_speed import load_cached  # noqa: E402

PAIRS = [("US100", "SP500"), ("US30", "SP500"), ("US30", "US100"),
         ("GER40", "UK100"), ("XAUUSD", "XAGUSD"), ("BRENT", "WTI"),
         ("EURUSD", "GBPUSD")]
TIMEFRAMES = ("15min", "1h")
LOOKBACK = 60


def bar_frame(asset: str, tf: str) -> pd.DataFrame:
    df = load_cached(asset)
    return df.resample(tf).agg({"close": "last", "ask_close": "last"}).dropna()


def half_spread_pct(asset: str) -> float:
    df = load_cached(asset)
    price = float(df["close"].median())
    spread = float((df["ask_close"] - df["close"]).median())
    return 0.5 * spread / price if price else np.nan


def main() -> None:
    print(f"{'pair':16s} {'tf':6s} {'bars':>6s} {'σ_d%':>6s} {'hl_bars':>8s} "
          f"{'sig/day':>8s} {'cost%':>7s} {'2σ/cost':>8s} {'corr':>6s}")
    for a, b in PAIRS:
        try:
            ca, cb = half_spread_pct(a), half_spread_pct(b)
        except (ValueError, FileNotFoundError):
            print(f"{a}/{b}: no cache, skipped")
            continue
        cost = ca + cb
        for tf in TIMEFRAMES:
            fa, fb = bar_frame(a, tf), bar_frame(b, tf)
            j = fa.join(fb, how="inner", lsuffix="_a", rsuffix="_b").dropna()
            if len(j) < 500:
                print(f"{a}/{b:8s} {tf:6s} too few bars")
                continue
            lr = np.log(j["close_a"]) - np.log(j["close_b"])
            mu = lr.rolling(LOOKBACK).mean()
            sd = lr.rolling(LOOKBACK).std()
            z = (lr - mu) / sd
            d = lr - mu
            dd = d.dropna()
            phi = float(np.polyfit(dd[:-1], dd[1:], 1)[0]) if len(dd) > 100 else np.nan
            hl = np.log(0.5) / np.log(phi) if 0 < phi < 1 else np.nan
            sd_med = float(sd.median())
            sig = float((z.abs() > 2).mean())
            bars_per_day = {"15min": 96, "1h": 24}[tf]
            corr = float(j["close_a"].pct_change().corr(j["close_b"].pct_change()))
            print(f"{a}/{b:8s} {tf:6s} {len(j):6d} {sd_med*100:6.3f} {hl:8.1f} "
                  f"{sig*bars_per_day:8.2f} {cost*100:7.4f} {2*sd_med/cost:8.1f} {corr:6.2f}")


if __name__ == "__main__":
    main()
