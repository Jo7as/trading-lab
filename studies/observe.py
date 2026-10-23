#!/usr/bin/env python3
"""S1/S2b: observe recurring behaviour and test it before writing any strategy.

For an instrument and a timeframe it prints descriptive session statistics and
then the pre-registered event studies (forward return in the condition's
direction vs round-trip cost, placebo, dose-response). A condition that fails
here should be dropped, not tuned.

  .venv/bin/python studies/observe.py --asset US100 --timeframe 15min
  .venv/bin/python studies/observe.py --asset XAUUSD --timeframe 1h
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

from trading_lab.research import data_quality, prepare, run_all

CACHE = "artifacts/cache/m1"


def load_cached(asset: str, start: int = 2019, end: int = 2026) -> pd.DataFrame:
    sides = {}
    for side in ("bid", "ask"):
        files = [f for f in sorted(glob.glob(f"{CACHE}/{asset}_1m_{side}_*.pkl"))
                 if f.split(f"1m_{side}_")[1][:4].isdigit()
                 and start <= int(f.split(f"1m_{side}_")[1][:4]) <= end]
        if not files:
            raise FileNotFoundError(asset)
        s = pd.concat([pd.read_pickle(f) for f in files])
        sides[side] = s[~s.index.duplicated(keep="last")].sort_index()
    bid, ask = sides["bid"], sides["ask"]
    return bid.join(ask[["open", "high", "low", "close"]].add_prefix("ask_"), how="inner")


def describe(m1: pd.DataFrame, tz: str, timeframe: str) -> None:
    """S1: what the instrument does, by session hour (no strategy attached)."""
    df, mid = prepare(m1, timeframe, tz)
    loc = df.index.tz_convert(tz)
    ret = mid.pct_change()
    print(f"--- S1 observations ({timeframe}) ---")
    print(f"{'hour':>4s} {'bars':>7s} {'move%med':>9s} {'spread%med':>11s} "
          f"{'autocorr1':>10s} {'autocorr5':>10s}")
    for h in sorted(set(loc.hour)):
        m = loc.hour == h
        r = ret[m]
        sp = ((df['ask_close'] - df['close']) / mid)[m]
        if len(r) < 50:
            continue
        ac1 = r.autocorr(1)
        ac5 = r.autocorr(5)
        print(f"{h:>4d} {len(r):>7d} {r.abs().median()*100:9.4f} {sp.median()*100:11.4f} "
              f"{ac1:10.3f} {ac5:10.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", default="US100")
    ap.add_argument("--timeframe", default="15min")
    ap.add_argument("--tz", default="America/New_York")
    args = ap.parse_args()

    m1 = load_cached(args.asset)
    print(f"=== {args.asset}, {args.timeframe}, {m1.index.min():%Y-%m} … {m1.index.max():%Y-%m} ===")
    describe(m1, args.tz, args.timeframe)
    df, mid = prepare(m1, args.timeframe, args.tz)
    q = data_quality(df, mid)
    print(f"\n--- S1 data gate ---\n  bars {q['bars']:,}  median |move| {q['median_abs_move_pct']}%  "
          f"worst {q['worst_pct']}% = {q['worst_move_in_median_units']}x median  "
          f"outlier bars {q['outlier_bars']}")
    if q["outlier_bars"]:
        print("  -> outlier bars removed before the studies (else every mean is garbage)")
    print("\n--- S2b event studies (fwd return in the condition's direction) ---")
    for s in run_all(m1, timeframe=args.timeframe, tz=args.tz, clean=True):
        print(s.summary())
        print(f"    mechanism: {s.mechanism}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
