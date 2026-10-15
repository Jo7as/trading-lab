#!/usr/bin/env python3
"""Trend Range Rejection (TRR) research — USDJPY / XAUUSD.

Example:
  python studies/research_trr.py --asset USD/JPY --start 2025-01-01
  python studies/research_trr.py --asset XAU/USD --start 2025-01-01
"""
from __future__ import annotations

import sys as _sys, pathlib as _path
_sys.path.insert(0, str(_path.Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from trading_lab.adapters.dukascopy_adapter import DukascopyAdapter
from trading_lab.score import trades_per_week
from trading_lab.strategies.trend_range_rejection import (
    TrendRangeRejectionConfig,
    run_backtest,
)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="USD/JPY")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--tp-mode", default="impulse_50", choices=["impulse_50", "fixed_rr"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    label = args.asset.replace("/", "")
    cfg = TrendRangeRejectionConfig(
        asset=args.asset,
        asset_label=label,
        risk_per_trade=args.risk,
        tp_mode=args.tp_mode,
    )
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)

    ad = DukascopyAdapter()
    print(f"Fetching {args.asset} 1m {start.date()}→{end.date()} ...", flush=True)
    t0 = time.time()
    df = await ad.fetch_history(args.asset, "1m", start=start, end=end)
    print(f"Loaded 1m={len(df)} in {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    r = run_backtest(df, cfg, verbose=args.verbose)
    print(f"Backtest {time.time()-t0:.1f}s", flush=True)
    m = r["metrics"]
    trades = [t for t in r["trades"] if t.get("status") == "closed"]
    days = max((df.index[-1] - df.index[0]).days, 1)
    tpw = trades_per_week(trades, days)
    print(
        f"n={m.get('total_trades')} WR={m.get('win_rate')} PnL={m.get('total_pnl_pct')}% "
        f"DD={m.get('max_drawdown_pct')} PF={m.get('profit_factor')} Sharpe={m.get('sharpe')} "
        f"tpw={tpw:.2f} setups={m.get('setups')}",
        flush=True,
    )
    print("reasons:", dict(Counter(t.get("exit_reason") for t in trades)), flush=True)
    print("dirs:", dict(Counter(t.get("direction") for t in trades)), flush=True)

    out = Path("artifacts") / f"trr_{label.lower()}_{args.start[:4]}.json"
    out.parent.mkdir(exist_ok=True)
    payload = {
        "asset": args.asset,
        "cfg": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
        "metrics": m,
        "n_trades": len(trades),
        "tpw": tpw,
        "reasons": dict(Counter(t.get("exit_reason") for t in trades)),
    }
    # tuples not JSON-serializable
    payload["cfg"]["timing_minutes"] = list(cfg.timing_minutes)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
