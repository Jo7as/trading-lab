#!/usr/bin/env python3
"""Sanity check for the z-score mean reversion: one backtest, no evolution.

Only answers: does it run, is the trade count plausible, anything obviously off.
"""
from __future__ import annotations

from trading_lab.env import env

import argparse
import asyncio
import logging
import os
from collections import Counter
from datetime import datetime, timezone

from trading_lab.backtest import print_metrics
from trading_lab.mr_common import load_bid_ask, period_days
from trading_lab.strategies.mean_reversion import MeanReversionConfig, run_backtest

ASSET_LABELS = {"GC=F": "XAU/USD", "XAU/USD": "XAU/USD", "USD/JPY": "USD/JPY"}


async def run(asset: str, start: str) -> None:
    cfg = MeanReversionConfig(asset=asset, asset_label=ASSET_LABELS.get(asset, asset))
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    print(f"Loading {asset} {cfg.timeframe} from {start} ...")
    df = await load_bid_ask(asset, cfg.timeframe, start_dt)
    if df.empty:
        print("ERROR: no data")
        return
    has_ask = "ask_close" in df.columns
    print(f"  {len(df)} bars {df.index[0]} → {df.index[-1]} | ask side: {has_ask}")
    if has_ask:
        spread = (df["ask_close"] - df["close"]) / df["close"]
        print(f"  median spread: {spread.median() * 1e4:.2f} bp | p95: {spread.quantile(0.95) * 1e4:.2f} bp")

    res = run_backtest(df, cfg)
    m = res["metrics"]
    print_metrics(m, label=f"MEAN REVERSION SANITY {asset}")

    sig = res["signals_df"]
    days = period_days(df)
    closed = [t for t in res["trades"] if t.get("status") == "closed"]
    risk = [t["meta"]["risk_pct"] for t in closed]
    r_mult = [t["pnl_pct"] / t["meta"]["risk_pct"] for t in closed if t["meta"]["risk_pct"] > 0]
    print(f"  Trades/week (calendar): {len(closed) / (days / 7):.2f}")
    print(f"  Tier trades:            {m['tier_trades']}")
    print(f"  Exit reasons:           {dict(Counter(t['exit_reason'] for t in closed))}")
    print(f"  Leverage-capped trades: {m['leverage_capped_trades']}")
    if risk:
        print(f"  Avg risk/trade:         {sum(risk) / len(risk):.3f}%")
        print(f"  Avg R/trade:            {sum(r_mult) / len(r_mult):+.3f}R")
    print("  Filter pass rates (share of bars):")
    print(f"    std_ok:     {sig['std_ok'].mean():.1%}")
    print(f"    adx < {cfg.adx_block_threshold:g}:   {(sig['adx'] < cfg.adx_block_threshold).mean():.1%}")
    print(f"    in_session: {sig['in_session'].mean():.1%}")
    print(f"    gap_ok:     {sig['gap_ok'].mean():.1%}")
    print(f"    |z| >= {cfg.z_entry_tier_b:g}:  {(sig['z'].abs() >= cfg.z_entry_tier_b).mean():.1%}")


def main() -> None:
    logging.getLogger("DUKASCRIPT").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Mean reversion sanity backtest")
    parser.add_argument("--asset", default="GC=F")
    parser.add_argument("--start", default=env("EVOLUTION_START", "2020-01-01"))
    args = parser.parse_args()
    asyncio.run(run(args.asset, args.start))


if __name__ == "__main__":
    main()
