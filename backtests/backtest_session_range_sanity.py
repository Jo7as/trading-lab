#!/usr/bin/env python3
"""Sanity check for the Session-Range-Fade: one backtest, no evolution.

Only answers: does it run, is the trade count plausible, anything obviously off.
"""
from __future__ import annotations
from trading_lab.env import env

import argparse
import asyncio
import dataclasses
import logging
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from trading_lab.backtest import print_metrics
from trading_lab.mr_common import load_bid_ask, period_days, resample_ohlc
from trading_lab.strategies.session_range_fade import SessionRangeFadeConfig, run_backtest


def _load_cfg(path: Path | None) -> SessionRangeFadeConfig:
    if path is None:
        return SessionRangeFadeConfig()
    raw = yaml.safe_load(path.read_text()) or {}
    valid = {f.name for f in dataclasses.fields(SessionRangeFadeConfig)}
    return SessionRangeFadeConfig(**{k: v for k, v in raw.items() if k in valid})


async def run(start: str, timeframe: str, config: Path | None) -> None:
    cfg = dataclasses.replace(_load_cfg(config), timeframe=timeframe)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    print(f"Loading {cfg.asset} 5min from {start} ...")
    df = await load_bid_ask(cfg.asset, "5min", start_dt)
    if df.empty:
        print("ERROR: no data")
        return
    if timeframe != "5min":
        df = resample_ohlc(df, timeframe)
    has_ask = "ask_close" in df.columns
    print(f"  {len(df)} {timeframe} bars {df.index[0]} → {df.index[-1]} | ask side: {has_ask}")

    res = run_backtest(df, cfg)
    m = res["metrics"]
    print_metrics(m, label=f"SESSION RANGE FADE SANITY {cfg.asset} {timeframe}")

    days = period_days(df)
    closed = [t for t in res["trades"] if t.get("status") == "closed"]
    print(f"  Days total / valid:     {m['days_total']} / {m['days_valid']}")
    print(f"  Skip reasons:           {m['skip_reasons']}")
    print(f"  Fade setups found:      {m['setups']}")
    print(f"  Trades/week (calendar): {len(closed) / (days / 7):.2f}")
    print(f"  Exit reasons:           {dict(Counter(t['exit_reason'] for t in closed))}")
    print(f"  Circuit-breaker skips:  {m['circuit_breaker_skips']}")
    print(f"  Leverage-capped trades: {m['leverage_capped_trades']}")
    if closed:
        r_mult = [t["pnl_pct"] / t["meta"]["risk_pct"] for t in closed if t["meta"]["risk_pct"] > 0]
        print(f"  Avg R/trade:            {sum(r_mult) / len(r_mult):+.3f}R")

    ranges = res["ranges"]
    ratio = (ranges["range_size"] / ranges["atr_ref"]).dropna()
    print(f"  Asia range / ATR({cfg.atr_period}, {cfg.atr_timeframe}) distribution:")
    for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99):
        print(f"    p{int(q * 100):02d}: {ratio.quantile(q):.2f}")
    print(
        f"    share inside [{cfg.range_min_atr_mult:g}, {cfg.range_max_atr_mult:g}]: "
        f"{((ratio >= cfg.range_min_atr_mult) & (ratio <= cfg.range_max_atr_mult)).mean():.1%}"
    )


def main() -> None:
    logging.getLogger("DUKASCRIPT").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Session-Range-Fade sanity backtest")
    parser.add_argument("--start", default=env("EVOLUTION_START", "2020-01-01"))
    parser.add_argument("--timeframe", default="15min", choices=["5min", "15min"])
    parser.add_argument("--config", type=Path, default=None, help="strategy YAML (default: dataclass defaults)")
    args = parser.parse_args()
    asyncio.run(run(args.start, args.timeframe, args.config))


if __name__ == "__main__":
    main()
