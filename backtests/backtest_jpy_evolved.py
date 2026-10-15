#!/usr/bin/env python3
"""Validate jpy_evolved_v48831 vs jpy_ls01 on Dukascopy USD/JPY 1h from 2020."""
from __future__ import annotations

import sys as _sys, pathlib as _path
_sys.path.insert(0, str(_path.Path(__file__).resolve().parent.parent))
from trading_lab.env import env

import argparse
import asyncio
import dataclasses
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from trading_lab.adapters.market_data import get_market_adapter
from trading_lab.backtest import print_metrics
from trading_lab.strategies.supertrend_volume import SupertrendVolumeConfig, run_backtest


def _load_cfg(path: Path) -> SupertrendVolumeConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    valid = {f.name for f in dataclasses.fields(SupertrendVolumeConfig)}
    return SupertrendVolumeConfig(**{k: v for k, v in raw.items() if k in valid})


def _summary(metrics: dict[str, Any], *, months: float) -> dict[str, Any]:
    return {
        "total_trades": metrics["total_trades"],
        "trades_per_week": metrics.get("trades_per_week", 0),
        "trades_per_month": round(metrics["total_trades"] / months, 1) if months else 0,
        "win_rate": metrics["win_rate"],
        "total_pnl_pct": metrics["total_pnl_pct"],
        "pnl_per_month": round(metrics["total_pnl_pct"] / months, 2) if months else 0,
        "sharpe": metrics["sharpe"],
        "max_drawdown_pct": metrics["max_drawdown_pct"],
        "profit_factor": metrics.get("profit_factor", 0),
        "by_side": metrics.get("by_side", {}),
    }


async def run_validation(
    *,
    start: str,
    interval: str,
    oos_split: float,
    data_source: str,
    output: Path,
) -> dict[str, Any]:
    adapter = get_market_adapter(data_source)
    start_dt = datetime.fromisoformat(start)
    print(f"Loading USD/JPY {interval} from {start} ({data_source})...")
    df = await adapter.fetch_history("USD/JPY", interval=interval, start=start_dt)
    if df.empty:
        raise RuntimeError("No USD/JPY data")

    split = int(len(df) * (1 - oos_split))
    df_oos = df.iloc[split:].copy()
    months = max((df.index[-1] - df.index[0]).days / 30.44, 1)
    oos_months = max(months * oos_split, 1)

    configs = {
        "jpy_v48831": _load_cfg(Path("jpy_evolved_v48831.yaml")),
        "jpy_ls01": _load_cfg(Path("jpy_baseline.yaml")),
    }

    results: dict[str, Any] = {}
    for label, cfg in configs.items():
        cfg = dataclasses.replace(cfg, asset="USD/JPY", asset_label="USD/JPY")
        full = run_backtest(df, cfg, verbose=False)["metrics"]
        oos = run_backtest(df_oos, cfg, verbose=False)["metrics"]
        results[label] = {
            "full": _summary(full, months=months),
            "oos": _summary(oos, months=oos_months),
        }
        print(f"\n{'='*60}")
        print(f"{label} — FULL ({df.index[0].date()} → {df.index[-1].date()})")
        print_metrics(full, label="FULL")
        print(f"\n{label} — OOS ({df_oos.index[0].date()} → {df_oos.index[-1].date()}, {oos_split:.0%})")
        print_metrics(oos, label="OOS")

    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset": "USD/JPY",
        "data_source": data_source,
        "interval": interval,
        "start": start,
        "end": str(df.index[-1]),
        "candles": len(df),
        "oos_split": oos_split,
        "oos_start": str(df_oos.index[0]),
        "configs": results,
        "comparison": {
            "oos_pnl_delta_pct": round(
                results["jpy_v48831"]["oos"]["total_pnl_pct"]
                - results["jpy_ls01"]["oos"]["total_pnl_pct"],
                2,
            ),
            "full_pnl_delta_pct": round(
                results["jpy_v48831"]["full"]["total_pnl_pct"]
                - results["jpy_ls01"]["full"]["total_pnl_pct"],
                2,
            ),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2))
    print(f"\nSaved: {output}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate JPY evolved vs baseline")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--oos-split", type=float, default=0.30)
    parser.add_argument(
        "--data-source",
        default=env("LAB_DATA_SOURCE", "dukascopy"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/jpy_evolved_v48831_validation.json"),
    )
    args = parser.parse_args()
    asyncio.run(
        run_validation(
            start=args.start,
            interval=args.interval,
            oos_split=args.oos_split,
            data_source=args.data_source,
            output=args.output,
        )
    )


if __name__ == "__main__":
    main()
