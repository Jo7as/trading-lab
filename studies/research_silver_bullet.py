#!/usr/bin/env python3
"""Silver Bullet backtest research (default XAU/USD 2024→now).

  python studies/research_silver_bullet.py
  python studies/research_silver_bullet.py --asset EUR/USD --sl-buffer 0.0001
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
from trading_lab.strategies.silver_bullet import SilverBulletConfig, run_backtest


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="XAU/USD")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--sl-buffer", type=float, default=None)
    ap.add_argument("--ny-am-only", action="store_true")
    ap.add_argument("--no-pd-zone", action="store_true")
    args = ap.parse_args()

    sl = args.sl_buffer
    if sl is None:
        sl = 0.50 if "XAU" in args.asset.upper() or "GOLD" in args.asset.upper() else 0.0001

    cfg = SilverBulletConfig(
        asset=args.asset,
        asset_label=args.asset,
        sl_buffer=sl,
        use_london=not args.ny_am_only,
        use_ny_am=True,
        use_ny_pm=not args.ny_am_only,
        require_pd_zone=not args.no_pd_zone,
    )
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)

    ad = DukascopyAdapter()
    print(f"Fetching {args.asset} 15m/1m {start.date()}→{end.date()} ...", flush=True)
    t0 = time.time()
    # Warm 15m a bit earlier for PD/PW
    warm = datetime(start.year - 0 if start.month > 6 else start.year - 1, 7, 1, tzinfo=timezone.utc)
    if start.month > 6:
        warm = datetime(start.year, 1, 1, tzinfo=timezone.utc)
    df_15m = await ad.fetch_history(args.asset, "15m", start=warm, end=end)
    df_1m = await ad.fetch_history(args.asset, "1m", start=start, end=end)
    print(f"Loaded 15m={len(df_15m)} 1m={len(df_1m)} in {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    r = run_backtest(df_15m, df_1m, cfg, verbose=True)
    print(f"Backtest {time.time()-t0:.0f}s", flush=True)
    m = r["metrics"]
    trades = [t for t in r["trades"] if t.get("status") == "closed"]
    print(
        f"n={m.get('total_trades')} WR={m.get('win_rate')} PnL={m.get('total_pnl_pct')}% "
        f"DD={m.get('max_drawdown_pct')} PF={m.get('profit_factor')} Sharpe={m.get('sharpe')} "
        f"setups={m.get('setups')}",
        flush=True,
    )
    print("reasons:", dict(Counter(t.get("exit_reason") for t in trades)), flush=True)
    by_win: Counter[str] = Counter()
    for t in trades:
        meta = t.get("meta") or {}
        by_win[str(meta.get("sb_window") or "?")] += 1
    print("by window:", dict(by_win), flush=True)

    out = Path("artifacts") / f"silver_bullet_{args.asset.replace('/', '').lower()}_{args.start[:4]}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {"asset": args.asset, "cfg": cfg.__dict__, "metrics": m, "n_trades": len(trades)},
            indent=2,
            default=str,
        )
    )
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
