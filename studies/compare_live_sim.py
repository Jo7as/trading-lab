#!/usr/bin/env python3
"""CLI: compare live signal trades vs simulation, or run long backtest."""
from __future__ import annotations

from trading_lab.env import env

import argparse
import asyncio
import os
from datetime import datetime
from pathlib import Path

from trading_lab.adapters.market_data import get_market_adapter
from trading_lab.backtest import print_metrics
from trading_lab.compare_live import load_strategy, run_compare
from trading_lab.strategies.supertrend_volume import run_backtest


def _default_state_dir() -> Path:
    env = env("LAB_STATE")
    if env:
        return Path(env)
    for candidate in (
        Path("live-state-template"),
        Path.home() / "trading-lab" / "state",
    ):
        if candidate.exists():
            return candidate
    return Path("live-state-template")


def _default_strategy(state_dir: Path) -> Path:
    env = env("LAB_STRATEGY_PATH")
    if env:
        return Path(env)
    for candidate in (state_dir / "hf_baseline.yaml", Path("hf_baseline.yaml")):
        if candidate.exists():
            return candidate
    return Path("hf_baseline.yaml")


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Live vs sim compare / long backtest")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Signal worker state (live_trades.jsonl)",
    )
    parser.add_argument(
        "--strategy",
        type=Path,
        default=None,
        help="hf_baseline.yaml path",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run long backtest instead of live compare",
    )
    parser.add_argument(
        "--data-source",
        default=None,
        choices=["yfinance", "dukascopy"],
        help="Market data provider (default: LAB_DATA_SOURCE or yfinance)",
    )
    parser.add_argument(
        "--period",
        default="730d",
        help="History window: 730d, max, or start date YYYY-MM-DD (dukascopy)",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Explicit start date YYYY-MM-DD (overrides --period start)",
    )
    parser.add_argument(
        "--interval",
        default="1h",
        help="Candle interval (1h = strategy native)",
    )
    parser.add_argument(
        "--compare-days",
        type=int,
        default=60,
        help="Replay window when no live trades yet",
    )
    args = parser.parse_args()

    state_dir = args.state_dir or _default_state_dir()
    strategy_path = args.strategy or _default_strategy(state_dir)

    if args.backtest:
        cfg = load_strategy(strategy_path)
        adapter = get_market_adapter(args.data_source)
        source = args.data_source or env("LAB_DATA_SOURCE", "yfinance")

        start_dt = (
            datetime.fromisoformat(args.start) if args.start else None
        )
        df = await adapter.fetch_history(
            cfg.asset,
            interval=args.interval,
            period=args.period,
            start=start_dt,
        )
        if df.empty:
            print(
                f"ERROR: no data for {cfg.asset} {args.interval} "
                f"source={source} period={args.period}"
            )
            return

        result = run_backtest(df, cfg, verbose=False)
        m = result["metrics"]
        closed = [t for t in result["trades"] if t.get("status") == "closed"]

        window = args.start or args.period
        print("=" * 72)
        print(
            f"BACKTEST — {cfg.asset_label} {args.interval} "
            f"({source}, {window})"
        )
        print(
            f"Data: {df.index[0]} → {df.index[-1]} "
            f"({len(df)} candles, {len(closed)} trades)"
        )
        print("=" * 72)
        print_metrics(m, label="hf_ls01")

        by_side = m.get("by_side", {})
        for side in ("long", "short"):
            s = by_side.get(side, {})
            if s.get("trades", 0):
                print(
                    f"  {side.upper()}: {s['trades']} trades | "
                    f"PnL {s['pnl_pct']:+.2f}% | WR {s['win_rate']:.1f}%"
                )

        if source == "yfinance" and args.interval == "1h":
            print()
            print("LIMIT: yfinance 1h max ~730 days. Use --data-source dukascopy for 2020+.")
        elif source == "dukascopy":
            print()
            print("NOTE: Dukascopy XAU/USD spot — slight price diff vs GC=F (signal bot).")
        return

    period = f"{max(args.compare_days + 90, 120)}d"
    report = await run_compare(state_dir, strategy_path, period=period)
    print(report)

    out_path = state_dir / "live_vs_sim_report.txt"
    out_path.write_text(report)
    print(f"Report saved: {out_path}")


if __name__ == "__main__":
    asyncio.run(_main())
