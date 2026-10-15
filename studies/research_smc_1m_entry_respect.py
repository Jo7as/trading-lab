#!/usr/bin/env python3
"""Research: 1m entry OB(+FVG) rejection after 1m ChoCh (pullback fill quality).

Example:
  python studies/research_smc_1m_entry_respect.py --year 2020
  python studies/research_smc_1m_entry_respect.py --year 2020 --grid
"""
from __future__ import annotations

import sys as _sys, pathlib as _path
_sys.path.insert(0, str(_path.Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from trading_lab.adapters.dukascopy_adapter import DukascopyAdapter
from trading_lab.strategies.smc.poi_respect import (
    RespectParams,
    evaluate_respect_1m_after_choch,
    grid_search_1m_entry,
)


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default="EUR/USD")
    p.add_argument("--year", type=int, default=2020)
    p.add_argument("--grid", action="store_true")
    p.add_argument("--swing", type=int, default=5)
    args = p.parse_args()

    start = datetime(args.year, 1, 1, tzinfo=timezone.utc)
    end = datetime(args.year + 1, 1, 1, tzinfo=timezone.utc)
    adapter = DukascopyAdapter()

    print(f"Fetching {args.asset} {args.year} 1m...", flush=True)
    t0 = time.time()
    df_1m = await adapter.fetch_history(args.asset, "1m", start=start, end=end)
    print(f"  1m={len(df_1m)} in {time.time()-t0:.1f}s", flush=True)

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    tag = args.asset.replace("/", "")

    if args.grid:
        print("Grid search 1m entry POI params...", flush=True)
        t1 = time.time()
        reports = grid_search_1m_entry(df_1m, swing_length_1m=args.swing)
        print(f"  done in {time.time()-t1:.1f}s", flush=True)
        top = [r.to_dict() for r in reports[:12]]
        print("\n=== TOP 1m entry configs ===")
        for r in top[:8]:
            print(
                f"  rate={r['respect_rate']:.1%} touched={r['n_touched']} "
                f"pois={r['n_pois']} kinds={r['choch_location_counts']} "
                f"touch={r['touch_part_counts']} params={r['params']}"
            )
        path = out_dir / f"smc_1m_entry_respect_{tag}_{args.year}_grid.json"
        path.write_text(json.dumps({"top": top, "n": len(reports)}, indent=2))
        print(f"Wrote {path}")
        return

    params = RespectParams(
        ob_lookback=10,
        ob_range="wick",
        horizon_bars=30,
        min_reaction_atr=0.5,
        min_fvg_atr=0.0,
    )
    print("Evaluating default 1m entry respect...", flush=True)
    t1 = time.time()
    report = evaluate_respect_1m_after_choch(
        df_1m, params, swing_length_1m=args.swing
    )
    print(f"  done in {time.time()-t1:.1f}s", flush=True)
    d = report.to_dict()
    print("\n=== 1m entry POI respect (after ChoCh) ===")
    print(json.dumps(d, indent=2))
    path = out_dir / f"smc_1m_entry_respect_{tag}_{args.year}.json"
    path.write_text(json.dumps(d, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    asyncio.run(main())
