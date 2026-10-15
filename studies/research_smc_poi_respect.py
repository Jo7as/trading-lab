#!/usr/bin/env python3
"""Research: 15m OB+FVG rejection rate + where 1m ChoCh lands (OB vs FVG).

Example:
  python studies/research_smc_poi_respect.py --year 2020
  python studies/research_smc_poi_respect.py --year 2020 --grid
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
    evaluate_respect_15m,
    grid_search_respect,
)


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default="EUR/USD")
    p.add_argument("--year", type=int, default=2020)
    p.add_argument("--grid", action="store_true")
    p.add_argument("--with-1m", action="store_true", default=True)
    p.add_argument("--no-1m", action="store_true")
    args = p.parse_args()
    use_1m = args.with_1m and not args.no_1m

    start = datetime(args.year, 1, 1, tzinfo=timezone.utc)
    end = datetime(args.year + 1, 1, 1, tzinfo=timezone.utc)
    adapter = DukascopyAdapter()

    print(f"Fetching {args.asset} {args.year} 15m...", flush=True)
    t0 = time.time()
    df_15m = await adapter.fetch_history(args.asset, "15m", start=start, end=end)
    print(f"  15m={len(df_15m)} in {time.time()-t0:.1f}s", flush=True)

    df_1m = None
    if use_1m:
        print(f"Fetching {args.asset} {args.year} 1m (for ChoCh location)...", flush=True)
        t1 = time.time()
        df_1m = await adapter.fetch_history(args.asset, "1m", start=start, end=end)
        print(f"  1m={len(df_1m)} in {time.time()-t1:.1f}s", flush=True)

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    if args.grid:
        print("Running parameter grid...", flush=True)
        t2 = time.time()
        reports = grid_search_respect(df_15m, df_1m=df_1m)
        print(f"  grid done in {time.time()-t2:.1f}s", flush=True)
        rows = [r.to_dict() for r in reports[:15]]
        print("\n=== TOP configs by respect_rate ===")
        for r in rows[:8]:
            print(
                f"  rate={r['respect_rate']:.1%} touched={r['n_touched']} "
                f"pois={r['n_pois']} params={r['params']} "
                f"touch_parts={r['touch_part_counts']} "
                f"choch_loc={r['choch_location_counts']}"
            )
        path = out_dir / f"smc_poi_respect_{args.asset.replace('/','')}_{args.year}_grid.json"
        path.write_text(json.dumps({"top": rows, "n": len(reports)}, indent=2))
        print(f"Wrote {path}")
        return

    params = RespectParams()
    print("Evaluating default params...", flush=True)
    t2 = time.time()
    report = evaluate_respect_15m(df_15m, params, df_1m=df_1m)
    print(f"  done in {time.time()-t2:.1f}s", flush=True)
    d = report.to_dict()
    print("\n=== 15m POI respect ===")
    print(json.dumps(d, indent=2))
    path = out_dir / f"smc_poi_respect_{args.asset.replace('/','')}_{args.year}.json"
    path.write_text(json.dumps(d, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    asyncio.run(main())
