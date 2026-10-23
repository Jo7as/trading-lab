#!/usr/bin/env python3
"""S0 pre-flight cost gate across instruments and horizons.

Run this **before** writing any strategy code: it measures whether an instrument
produces a move over the intended horizon that is big enough to pay the spread.

  .venv/bin/python studies/cost_gate.py
  .venv/bin/python studies/cost_gate.py --assets US100,XAUUSD --horizons 15,60 --k 3
"""
from __future__ import annotations

import argparse
import glob

import pandas as pd

from trading_lab.cost_gate import DEFAULT_K, horizon_gate

CACHE = "artifacts/cache/m1"
DEFAULT_ASSETS = ("US100", "US30", "SP500", "GER40", "UK100", "JP225",
                  "XAUUSD", "XAGUSD", "COPPER", "BRENT", "UST10Y")
DEFAULT_HORIZONS = (5, 15, 30, 60, 240)


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
    out = bid.join(ask[["open", "high", "low", "close"]].add_prefix("ask_"), how="inner")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", default=",".join(DEFAULT_ASSETS))
    ap.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    ap.add_argument("--k", type=float, default=DEFAULT_K)
    args = ap.parse_args()
    horizons = [int(h) for h in args.horizons.split(",")]

    print(f"S0 cost gate — required ratio E|move|/cost ≥ {args.k:.1f} "
          f"(taker-exit costs; limit exits are half)\n")
    for asset in [a.strip() for a in args.assets.split(",") if a.strip()]:
        try:
            m1 = load_cached(asset)
        except FileNotFoundError:
            print(f"{asset}: no cache, skipped")
            continue
        price = float(m1["close"].median())
        print(f"{asset}  (median price {price:,.1f}, bars {len(m1):,})")
        for h in horizons:
            g = horizon_gate(m1, h)
            print(f"   {g}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
