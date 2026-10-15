#!/usr/bin/env python3
"""ADX Regime-Hybrid backtest for Forex majors + Gold (Dukascopy)."""
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
from trading_lab.strategies.regime_hybrid import RegimeHybridConfig, run_backtest
from trading_lab.strategies.supertrend_volume import SupertrendVolumeConfig, run_backtest as st_run_backtest

ASSET_LABELS: dict[str, str] = {
    "GC=F": "XAU/USD",
    "XAU/USD": "XAU/USD",
    "USD/JPY": "USD/JPY",
    "USDJPY=X": "USD/JPY",
    "GBP/USD": "GBP/USD",
    "GBPUSD=X": "GBP/USD",
    "EUR/USD": "EUR/USD",
    "EURUSD=X": "EUR/USD",
    "AUD/USD": "AUD/USD",
    "AUDUSD=X": "AUD/USD",
}

ST_BASELINE_LABELS: dict[str, str] = {
    "USD/JPY": "jpy_ls01",
    "GC=F": "hf_ls01",
}

GO_CRITERIA = {
    "oos_pnl_pct": 0.0,
    "oos_sharpe": 0.8,
    "oos_dd_pct": -10.0,
    "oos_tpw": 3.0,
}


def _load_st_baseline(asset: str) -> SupertrendVolumeConfig:
    if asset == "USD/JPY":
        path = Path("jpy_baseline.yaml")
    else:
        path = Path("hf_baseline.yaml")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    raw = yaml.safe_load(path.read_text()) or {}
    valid = {f.name for f in dataclasses.fields(SupertrendVolumeConfig)}
    cfg = SupertrendVolumeConfig(**{k: v for k, v in raw.items() if k in valid})
    return dataclasses.replace(
        cfg,
        asset=asset,
        asset_label=ASSET_LABELS.get(asset, asset),
    )


def _default_regime_cfg(asset: str) -> RegimeHybridConfig:
    return RegimeHybridConfig(
        asset=asset,
        asset_label=ASSET_LABELS.get(asset, asset),
    )


def _row_metrics(
    asset: str,
    interval: str,
    full: dict[str, Any],
    oos: dict[str, Any],
    *,
    months: float,
    oos_months: float,
    candles: int,
    regime_stats: dict[str, float],
    module_trades: dict[str, int],
    st_baseline_oos: dict[str, Any] | None,
) -> dict[str, Any]:
    go = bool(
        oos["total_pnl_pct"] > GO_CRITERIA["oos_pnl_pct"]
        and oos["sharpe"] >= GO_CRITERIA["oos_sharpe"]
        and oos["max_drawdown_pct"] >= GO_CRITERIA["oos_dd_pct"]
        and oos.get("trades_per_week", 0) >= GO_CRITERIA["oos_tpw"]
    )
    vs_st = None
    if st_baseline_oos:
        vs_st = round(oos["total_pnl_pct"] - st_baseline_oos["total_pnl_pct"], 2)

    return {
        "asset": asset,
        "asset_label": ASSET_LABELS.get(asset, asset),
        "interval": interval,
        "candles": candles,
        "regime_pct": regime_stats,
        "module_trades_full": module_trades,
        "full_trades": full["total_trades"],
        "full_tpw": round(full.get("trades_per_week", 0), 2),
        "full_pnl_pct": round(full["total_pnl_pct"], 2),
        "full_sharpe": round(full["sharpe"], 2),
        "full_dd_pct": round(full["max_drawdown_pct"], 2),
        "oos_trades": oos["total_trades"],
        "oos_tpw": round(oos.get("trades_per_week", 0), 2),
        "oos_pnl_pct": round(oos["total_pnl_pct"], 2),
        "oos_sharpe": round(oos["sharpe"], 2),
        "oos_dd_pct": round(oos["max_drawdown_pct"], 2),
        "oos_pf": round(oos.get("profit_factor", 0), 2),
        "st_baseline_oos_pnl_pct": (
            round(st_baseline_oos["total_pnl_pct"], 2) if st_baseline_oos else None
        ),
        "vs_st_oos_pnl_pct": vs_st,
        "go": go,
        "go_checks": {
            "oos_pnl_positive": bool(oos["total_pnl_pct"] > 0),
            "oos_sharpe_ge_0.8": bool(oos["sharpe"] >= 0.8),
            "oos_dd_le_10pct": bool(oos["max_drawdown_pct"] >= -10.0),
            "oos_tpw_ge_3": bool(oos.get("trades_per_week", 0) >= 3.0),
        },
    }


async def backtest_asset(
    asset: str,
    *,
    data_source: str,
    start: str,
    interval: str,
    oos_split: float,
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

    cfg = _default_regime_cfg(asset)
    full_res = run_backtest(df, cfg, verbose=False)
    oos_res = run_backtest(df_oos, cfg, verbose=False)

    st_baseline_oos = None
    st_cfg = _load_st_baseline(asset)
    st_baseline_oos = st_run_backtest(df_oos, st_cfg, verbose=False)["metrics"]

    row = _row_metrics(
        asset,
        interval,
        full_res["metrics"],
        oos_res["metrics"],
        months=months,
        oos_months=oos_months,
        candles=len(df),
        regime_stats=full_res.get("regime_stats", {}),
        module_trades=full_res.get("module_stats", {}),
        st_baseline_oos=st_baseline_oos,
    )
    print(
        f"  {asset}: {row['full_trades']} trades, OOS {row['oos_pnl_pct']:+.1f}% "
        f"(ST {row['st_baseline_oos_pnl_pct']}), GO={'YES' if row['go'] else 'NO'}"
    )
    return row


async def run_study(
    *,
    assets: list[str],
    data_source: str,
    start: str,
    interval: str,
    oos_split: float,
    output: Path,
) -> dict[str, Any]:
    print("Config: ADX Regime-Hybrid (ST trend ADX>=25, BB-MR ADX<=15)")
    print(f"Data: {data_source}, interval={interval}, start={start}\n")

    rows: list[dict[str, Any]] = []
    for asset in assets:
        row = await backtest_asset(
            asset,
            data_source=data_source,
            start=start,
            interval=interval,
            oos_split=oos_split,
        )
        if row:
            rows.append(row)

    forex_go = [r for r in rows if r["asset"] in ("EUR/USD", "GBP/USD", "AUD/USD") and r["go"]]
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": "adx_regime_hybrid",
        "data_source": data_source,
        "interval": interval,
        "start": start,
        "oos_split": oos_split,
        "go_criteria": GO_CRITERIA,
        "assets": rows,
        "summary": {
            "forex_winners": [r["asset"] for r in forex_go],
            "forex_go_count": len(forex_go),
            "all_go": [r["asset"] for r in rows if r["go"]],
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2))
    print(f"\nSaved: {output}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="ADX Regime-Hybrid forex backtest")
    parser.add_argument(
        "--assets",
        default="EUR/USD,GBP/USD,AUD/USD,USD/JPY,GC=F",
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
        default=Path("artifacts/regime_forex_backtest.json"),
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
        )
    )


if __name__ == "__main__":
    main()
