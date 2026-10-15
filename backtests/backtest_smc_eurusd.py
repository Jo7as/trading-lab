#!/usr/bin/env python3
"""SMC/ICT smoke research: EUR/USD (or other forex) multi-TF backtest.

Example:
  PYTHONPATH=. uv run python backtest_smc_eurusd.py --year 2020
  PYTHONPATH=. uv run python backtest_smc_eurusd.py --start 2020 --end 2022
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from trading_lab.adapters.dukascopy_adapter import DukascopyAdapter
from trading_lab.strategies.smc_ict import SmcIctConfig, run_backtest


async def fetch_year(adapter: DukascopyAdapter, asset: str, year: int):
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    print(f"Fetching {asset} {year} (4h/15m/1m)...", flush=True)
    t0 = time.time()
    df_4h = await adapter.fetch_history(asset, "4h", start=start, end=end)
    df_15m = await adapter.fetch_history(asset, "15m", start=start, end=end)
    df_1m = await adapter.fetch_history(asset, "1m", start=start, end=end)
    print(
        f"  4h={len(df_4h)} 15m={len(df_15m)} 1m={len(df_1m)} "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )
    return df_4h, df_15m, df_1m


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default="EUR/USD")
    p.add_argument("--start", type=int, default=2020)
    p.add_argument("--end", type=int, default=2020, help="inclusive year")
    p.add_argument("--sl-buffer", type=float, default=0.0008)
    args = p.parse_args()

    adapter = DukascopyAdapter()
    cfg = SmcIctConfig(
        asset=args.asset,
        asset_label=args.asset,
        sl_buffer=args.sl_buffer,
        cooldown_hours=0,
    )

    all_trades: list[dict] = []
    all_signals: list[dict] = []
    yearly: list[dict] = []

    for year in range(args.start, args.end + 1):
        df_4h, df_15m, df_1m = await fetch_year(adapter, args.asset, year)
        if df_1m.empty:
            print(f"  skip {year}: empty 1m")
            continue
        t0 = time.time()
        result = run_backtest(df_4h, df_15m, df_1m, cfg, verbose=True)
        elapsed = time.time() - t0
        m = result["metrics"]
        print(
            f"  {year}: trades={m.get('total_trades')} wr={m.get('win_rate')} "
            f"pnl={m.get('total_pnl_pct')}% elapsed={elapsed:.0f}s",
            flush=True,
        )
        yearly.append({"year": year, "metrics": m, "bars_1m": len(df_1m)})
        for s in result["signals"]:
            s = dict(s)
            s["year"] = year
            all_signals.append(s)
        for t in result["trades"]:
            t = dict(t)
            t["year"] = year
            all_trades.append(t)

    out = Path("artifacts") / f"smc_{args.asset.replace('/', '').lower()}_{args.start}_{args.end}.json"
    out.parent.mkdir(exist_ok=True)
    payload = {
        "asset": args.asset,
        "start": args.start,
        "end": args.end,
        "sl_buffer": args.sl_buffer,
        "yearly": yearly,
        "n_signals": len(all_signals),
        "n_trades": len(all_trades),
        "signals": all_signals,
        "trades": all_trades,
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {out} (signals={len(all_signals)} trades={len(all_trades)})")


if __name__ == "__main__":
    asyncio.run(main())
