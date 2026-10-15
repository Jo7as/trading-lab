#!/usr/bin/env python3
"""Opening Range Breakout research — Nasdaq-100 (Dukascopy E_NQ-100).

Example:
  python studies/research_orb_nq.py
  python studies/research_orb_nq.py --long-only --tp 1.0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from trading_lab.adapters.dukascopy_adapter import DukascopyAdapter
from trading_lab.strategies.opening_range_breakout import (
    OpeningRangeBreakoutConfig,
    run_backtest,
)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="NQ")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--orb-minutes", type=int, default=15)
    ap.add_argument("--tp", type=float, default=0.5, help="TP as multiple of OR width")
    ap.add_argument("--max-orb-pct", type=float, default=0.008)
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--entry-cutoff", default="11:00")
    args = ap.parse_args()

    cfg = OpeningRangeBreakoutConfig(
        asset=args.asset,
        asset_label=args.asset,
        orb_minutes=args.orb_minutes,
        tp_range_mult=args.tp,
        max_orb_pct=args.max_orb_pct,
        long_only=args.long_only,
        entry_cutoff=args.entry_cutoff,
        bar_minutes=5,
    )
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)

    ad = DukascopyAdapter()
    print(f"Fetching {args.asset} 5m {start.date()}→{end.date()} ...", flush=True)
    t0 = time.time()
    df = await ad.fetch_history(args.asset, "5m", start=start, end=end)
    print(f"Loaded 5m={len(df)} in {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    r = run_backtest(df, cfg, verbose=True)
    print(f"Backtest {time.time()-t0:.1f}s", flush=True)
    m = r["metrics"]
    trades = [t for t in r["trades"] if t.get("status") == "closed"]
    print(
        f"n={m.get('total_trades')} WR={m.get('win_rate')} PnL={m.get('total_pnl_pct')}% "
        f"DD={m.get('max_drawdown_pct')} PF={m.get('profit_factor')} Sharpe={m.get('sharpe')} "
        f"setups={m.get('setups')}",
        flush=True,
    )
    print("reasons:", dict(Counter(t.get("exit_reason") for t in trades)), flush=True)
    print(
        "dirs:",
        dict(Counter(t.get("direction") for t in trades)),
        flush=True,
    )

    out = (
        Path("artifacts")
        / f"orb_{args.asset.lower()}_{args.start[:4]}.json"
    )
    out.parent.mkdir(exist_ok=True)
    payload = {
        "asset": args.asset,
        "cfg": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
        "metrics": m,
        "n_trades": len(trades),
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
