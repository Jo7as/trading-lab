#!/usr/bin/env python3
"""Data audit (plan step 1) across the cached instruments.

  .venv/bin/python studies/data_audit.py
  .venv/bin/python studies/data_audit.py --asset XAUUSD --tz America/New_York
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from trading_lab.data_quality import audit

CACHE = "artifacts/cache/m1"
ASSETS = ("US100", "US30", "SP500", "GER40", "UK100", "JP225",
          "XAUUSD", "XAGUSD", "COPPER", "BRENT", "UST10Y")
TZ = {"US100": "America/New_York", "US30": "America/New_York", "SP500": "America/New_York",
      "GER40": "Europe/Berlin", "UK100": "Europe/London", "XAUUSD": "America/New_York",
      "XAGUSD": "America/New_York", "COPPER": "America/New_York",
      "BRENT": "America/New_York", "UST10Y": "America/New_York", "JP225": "Asia/Tokyo"}


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", default=None)
    ap.add_argument("--tz", default=None)
    ap.add_argument("--out", type=Path, default=Path("artifacts/data_audit.json"))
    args = ap.parse_args()

    assets = [args.asset] if args.asset else list(ASSETS)
    reports = []
    for a in assets:
        try:
            df = load_cached(a)
        except FileNotFoundError:
            print(f"{a}: no cache, skipped")
            continue
        rep = audit(df, a, tz=args.tz or TZ.get(a, "UTC"))
        print(rep.summary())
        for b in rep.worst_bars[:3]:
            print(f"        worst bar {b['ts']}  {b['ret_pct']:+.2f}%")
        us = {h: v for h, v in rep.spread_by_hour.items() if 9 <= h <= 16}
        asia = {h: v for h, v in rep.spread_by_hour.items() if 0 <= h <= 5}
        if us and asia:
            print(f"        spread (typo-free check): US hours median "
                  f"{sum(us.values())/len(us):.4f}%  Asia {sum(asia.values())/len(asia):.4f}%")
        reports.append(rep)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([r.__dict__ for r in reports], indent=2, default=str))
    print(f"\njson: {args.out}")
    suspect = [r.asset for r in reports if r.verdict() != "OK"]
    print(f"suspect: {suspect or '—'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
