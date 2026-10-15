#!/usr/bin/env python3
"""Multi-asset hf_ls01 backtest for frequency research (Dukascopy)."""
from __future__ import annotations

from trading_lab.env import env

import argparse
import asyncio
import dataclasses
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from trading_lab.adapters.market_data import get_market_adapter
from trading_lab.strategies.supertrend_volume import SupertrendVolumeConfig, run_backtest

ASSET_LABELS: dict[str, str] = {
    "GC=F": "XAU/USD",
    "XAU/USD": "XAU/USD",
    "XAG/USD": "XAG/USD",
    "SI=F": "XAG/USD",
    "USD/JPY": "USD/JPY",
    "USDJPY=X": "USD/JPY",
    "GBP/USD": "GBP/USD",
    "GBPUSD=X": "GBP/USD",
    "EUR/USD": "EUR/USD",
    "EURUSD=X": "EUR/USD",
    "AUD/USD": "AUD/USD",
    "AUDUSD=X": "AUD/USD",
}


def _load_hf_config(path: Path) -> SupertrendVolumeConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    valid = {f.name for f in dataclasses.fields(SupertrendVolumeConfig)}
    return SupertrendVolumeConfig(**{k: v for k, v in raw.items() if k in valid})


def _baseline_path() -> Path:
    for candidate in (
        Path("hf_baseline.yaml"),
        Path(__file__).resolve().parent / "hf_baseline.yaml",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("hf_baseline.yaml not found")


def _row_metrics(
    asset: str,
    interval: str,
    full: dict[str, Any],
    oos: dict[str, Any],
    *,
    months: float,
    oos_months: float,
    candles: int,
) -> dict[str, Any]:
    by_side = full.get("by_side", {})
    long_s = by_side.get("long", {})
    short_s = by_side.get("short", {})
    return {
        "asset": asset,
        "asset_label": ASSET_LABELS.get(asset, asset),
        "interval": interval,
        "candles": candles,
        "full_trades": full["total_trades"],
        "full_tpw": round(full.get("trades_per_week", 0), 2),
        "full_tpm": round(full["total_trades"] / months, 1) if months else 0,
        "full_wr": round(full["win_rate"], 1),
        "full_pnl_pct": round(full["total_pnl_pct"], 2),
        "full_pnl_per_month": round(full["total_pnl_pct"] / months, 2) if months else 0,
        "full_sharpe": round(full["sharpe"], 2),
        "full_dd_pct": round(full["max_drawdown_pct"], 2),
        "full_pf": round(full["profit_factor"], 2),
        "oos_trades": oos["total_trades"],
        "oos_tpw": round(oos.get("trades_per_week", 0), 2),
        "oos_tpm": round(oos["total_trades"] / oos_months, 1) if oos_months else 0,
        "oos_wr": round(oos["win_rate"], 1),
        "oos_pnl_pct": round(oos["total_pnl_pct"], 2),
        "oos_pnl_per_month": round(oos["total_pnl_pct"] / oos_months, 2) if oos_months else 0,
        "oos_sharpe": round(oos["sharpe"], 2),
        "oos_dd_pct": round(oos["max_drawdown_pct"], 2),
        "long_pnl": round(float(long_s.get("pnl_pct", 0)), 2),
        "short_pnl": round(float(short_s.get("pnl_pct", 0)), 2),
        "long_trades": int(long_s.get("trades", 0)),
        "short_trades": int(short_s.get("trades", 0)),
    }


async def backtest_asset(
    asset: str,
    cfg: SupertrendVolumeConfig,
    *,
    data_source: str,
    start: str,
    interval: str,
    oos_split: float,
    long_only: bool | None = None,
) -> dict[str, Any] | None:
    adapter = get_market_adapter(data_source)
    start_dt = datetime.fromisoformat(start)
    print(f"  Loading {asset} {interval} from {start}...")
    df = await adapter.fetch_history(asset, interval=interval, start=start_dt)
    if df.empty:
        print(f"  SKIP {asset}: no data")
        return None

    split = int(len(df) * (1 - oos_split))
    df_oos = df.iloc[split:].copy()
    months = max((df.index[-1] - df.index[0]).days / 30.44, 1)
    oos_months = max(months * oos_split, 1)

    asset_cfg = dataclasses.replace(
        cfg,
        asset=asset,
        asset_label=ASSET_LABELS.get(asset, asset),
        long_only=long_only if long_only is not None else cfg.long_only,
    )
    full = run_backtest(df, asset_cfg, verbose=False)["metrics"]
    oos = run_backtest(df_oos, asset_cfg, verbose=False)["metrics"]
    row = _row_metrics(
        asset,
        interval,
        full,
        oos,
        months=months,
        oos_months=oos_months,
        candles=len(df),
    )
    print(
        f"  {asset}: {row['full_trades']} trades ({row['full_tpw']}/wk), "
        f"OOS {row['oos_pnl_pct']:+.1f}%, OOS {row['oos_tpw']}/wk"
    )
    return row


def _weekly_returns(df: pd.DataFrame) -> pd.Series:
    close = df["close"].astype(float)
    weekly = close.resample("W").last().pct_change().dropna()
    return weekly


async def compute_correlations(
    assets: list[str],
    *,
    data_source: str,
    start: str,
    interval: str,
) -> dict[str, Any]:
    adapter = get_market_adapter(data_source)
    start_dt = datetime.fromisoformat(start)
    series: dict[str, pd.Series] = {}
    for asset in assets:
        df = await adapter.fetch_history(asset, interval=interval, start=start_dt)
        if df.empty or len(df) < 100:
            continue
        series[asset] = _weekly_returns(df)

    if len(series) < 2:
        return {"pairs": [], "matrix": {}}

    combined = pd.DataFrame(series).dropna()
    if combined.empty:
        return {"pairs": [], "matrix": {}}

    corr = combined.corr()
    pairs: list[dict[str, Any]] = []
    keys = list(corr.columns)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            pairs.append({"a": a, "b": b, "corr": round(float(corr.loc[a, b]), 3)})

    matrix = {
        a: {b: round(float(corr.loc[a, b]), 3) for b in keys}
        for a in keys
    }
    return {"pairs": pairs, "matrix": matrix, "weeks": len(combined)}


async def run_study(
    *,
    assets: list[str],
    data_source: str,
    start: str,
    interval: str,
    oos_split: float,
    output: Path,
    long_only: bool | None = None,
) -> dict[str, Any]:
    cfg = _load_hf_config(_baseline_path())
    mode = "long_only" if long_only else f"long_only={cfg.long_only}"
    print(f"Config: hf_ls01 (st{cfg.supertrend_period}/{cfg.supertrend_mult}, {mode})")
    print(f"Data: {data_source}, interval={interval}, start={start}\n")

    rows: list[dict[str, Any]] = []
    for asset in assets:
        row = await backtest_asset(
            asset,
            cfg,
            data_source=data_source,
            start=start,
            interval=interval,
            oos_split=oos_split,
            long_only=long_only,
        )
        if row:
            rows.append(row)

    corr_assets = [a for a in assets if a in ("GC=F", "XAG/USD", "USD/JPY", "GBP/USD", "EUR/USD", "AUD/USD")]
    print("\nComputing weekly return correlations...")
    correlations = await compute_correlations(
        corr_assets,
        data_source=data_source,
        start=start,
        interval=interval,
    )

    gold = next((r for r in rows if r["asset"] in ("GC=F", "XAU/USD")), None)
    total_full_tpw = sum(r["full_tpw"] for r in rows)
    total_oos_tpw = sum(r["oos_tpw"] for r in rows)

    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": "hf_ls01",
        "long_only": long_only if long_only is not None else cfg.long_only,
        "data_source": data_source,
        "interval": interval,
        "start": start,
        "oos_split": oos_split,
        "assets": rows,
        "portfolio_estimate": {
            "sum_full_tpw": round(total_full_tpw, 2),
            "sum_oos_tpw": round(total_oos_tpw, 2),
            "gold_only_full_tpw": gold["full_tpw"] if gold else None,
            "gold_only_oos_tpw": gold["oos_tpw"] if gold else None,
        },
        "correlations": correlations,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2))
    print(f"\nSaved: {output}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-asset hf_ls01 backtest")
    parser.add_argument(
        "--assets",
        default="GC=F,XAG/USD,USD/JPY,GBP/USD,EUR/USD,AUD/USD",
    )
    parser.add_argument(
        "--data-source",
        default=env("LAB_DATA_SOURCE", "dukascopy"),
    )
    parser.add_argument("--start", default=env("EVOLUTION_START", "2020-01-01"))
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--oos-split", type=float, default=0.30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/multi_asset_backtest.json"),
    )
    parser.add_argument(
        "--long-only",
        action="store_true",
        help="Force long_only=True for all assets in this run",
    )
    args = parser.parse_args()
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    asyncio.run(
        run_study(
            assets=assets,
            data_source=args.data_source,
            start=args.start,
            interval=args.interval,
            oos_split=args.oos_split,
            output=args.output,
            long_only=True if args.long_only else None,
        )
    )


if __name__ == "__main__":
    main()
