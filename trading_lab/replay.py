"""Replay/Backtest CLI — run MCTFB strategy on historical Yahoo Finance data.

Usage:
  uv run python -m trading_lab.replay --asset GC=F --period 730d
  uv run python -m trading_lab.replay --asset EURUSD=X --period 365d

Produces a full backtest report with trade log and metrics.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .adapters.historical_yf import HistoricalYFAdapter
from .backtest import backtest, print_metrics
from .config import MCTFBConfig, load_config_from_yaml, config_to_yaml
from .reflect import reflect_fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="MCTFB backtest / replay")
    parser.add_argument("--asset", default=None, help="Yahoo Finance ticker (e.g. GC=F, EURUSD=X)")
    parser.add_argument("--period", default="730d", help="History period (e.g. 365d, 730d)")
    parser.add_argument("--interval", default="1h", help="Candle interval (default 1h)")
    parser.add_argument("--strategy", default=None, help="Path to strategy.yaml")
    parser.add_argument("--reflect", action="store_true", help="Run reflection after backtest")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    args = parser.parse_args()

    # Load config
    state_dir = Path(__file__).parent.parent / "state"
    strategy_path = Path(args.strategy) if args.strategy else state_dir / "strategy.yaml"

    if strategy_path.exists():
        cfg = load_config_from_yaml(strategy_path)
    else:
        cfg = MCTFBConfig()

    if args.asset:
        cfg = MCTFBConfig(**{**cfg.__dict__, "asset": args.asset, "asset_label": args.asset})

    print("MCTFB Backtest")
    print(f"  asset:    {cfg.asset} ({cfg.asset_label})")
    print(f"  period:   {args.period}")
    print(f"  interval: {args.interval}")
    print(f"  config:   {config_to_yaml(cfg)}")

    # Load data
    adapter = HistoricalYFAdapter(period=args.period, interval=args.interval)

    async def run():
        df = await adapter.load(cfg.asset)
        if df.empty:
            print("ERROR: No data loaded. Check asset ticker or network.")
            sys.exit(1)
        return df

    df = asyncio.run(run())

    # Run backtest
    print(f"\nRunning backtest on {len(df)} candles...")
    result = backtest(df, cfg, verbose=args.verbose)

    # Print results
    print_metrics(result["metrics"])

    # Print trade log summary
    trades = result["trades"]
    if trades:
        print(f"\nFirst 5 trades:")
        for t in trades[:5]:
            print(f"  {t.get('direction')} entry={t.get('entry_price'):.2f} "
                  f"exit={t.get('exit_price')} reason={t.get('exit_reason')} "
                  f"pnl={t.get('pnl_pct')}%")

        print(f"\nLast 5 trades:")
        for t in trades[-5:]:
            print(f"  {t.get('direction')} entry={t.get('entry_price'):.2f} "
                  f"exit={t.get('exit_price')} reason={t.get('exit_reason')} "
                  f"pnl={t.get('pnl_pct')}%")

    # Save trades to file
    trades_path = state_dir / "trades.jsonl"
    with open(trades_path, "w") as f:
        for t in trades:
            f.write(json.dumps(t, default=str) + "\n")
    print(f"\nTrades saved to {trades_path}")

    # Optional reflection
    if args.reflect:
        print("\nRunning reflection cycle...")
        reflect_fallback()
        # Re-read updated strategy
        cfg = load_config_from_yaml(strategy_path)
        print(f"Updated strategy: {config_to_yaml(cfg)}")


if __name__ == "__main__":
    main()
