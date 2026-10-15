#!/usr/bin/env python3
"""BB Squeeze Breakout backtest for Forex majors + Gold (Dukascopy)."""
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
from trading_lab.strategies.bb_squeeze_breakout import BBSqueezeBreakoutConfig, run_backtest
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

GO_CRITERIA = {
    "oos_pnl_pct": 0.0,
    "oos_sharpe": 0.8,
    "oos_dd_pct": -10.0,
    "oos_tpw": 3.0,
}

FOREX_MAJORS = ("EUR/USD", "GBP/USD", "AUD/USD")


def _load_st_baseline(asset: str) -> SupertrendVolumeConfig:
    path = Path("jpy_baseline.yaml") if asset == "USD/JPY" else Path("hf_baseline.yaml")
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


def _load_regime_baselines() -> dict[str, float]:
    path = Path("artifacts/regime_forex_backtest.json")
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {
        row["asset"]: row["oos_pnl_pct"]
        for row in data.get("assets", [])
    }


def _default_squeeze_cfg(asset: str) -> BBSqueezeBreakoutConfig:
    return BBSqueezeBreakoutConfig(
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
    st_baseline_oos: dict[str, Any],
    regime_oos_pnl: float | None,
) -> dict[str, Any]:
    go = bool(
        oos["total_pnl_pct"] > GO_CRITERIA["oos_pnl_pct"]
        and oos["sharpe"] >= GO_CRITERIA["oos_sharpe"]
        and oos["max_drawdown_pct"] >= GO_CRITERIA["oos_dd_pct"]
        and oos.get("trades_per_week", 0) >= GO_CRITERIA["oos_tpw"]
    )
    st_pnl = st_baseline_oos["total_pnl_pct"]
    return {
        "asset": asset,
        "asset_label": ASSET_LABELS.get(asset, asset),
        "interval": interval,
        "candles": candles,
        "squeeze_pct_full": full.get("squeeze_pct", 0),
        "breakout_signals_full": full.get("breakout_signals", {}),
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
        "st_baseline_oos_pnl_pct": round(st_pnl, 2),
        "vs_st_oos_pnl_pct": round(oos["total_pnl_pct"] - st_pnl, 2),
        "regime_hybrid_oos_pnl_pct": regime_oos_pnl,
        "vs_regime_oos_pnl_pct": (
            round(oos["total_pnl_pct"] - regime_oos_pnl, 2)
            if regime_oos_pnl is not None else None
        ),
        "go": go,
        "go_checks": {
            "oos_pnl_positive": bool(oos["total_pnl_pct"] > 0),
            "oos_sharpe_ge_0.8": bool(oos["sharpe"] >= 0.8),
            "oos_dd_le_10pct": bool(oos["max_drawdown_pct"] >= -10.0),
            "oos_tpw_ge_3": bool(oos.get("trades_per_week", 0) >= 3.0),
        },
    }


def _weekly_returns(df: pd.DataFrame) -> pd.Series:
    return df["close"].astype(float).resample("W").last().pct_change().dropna()


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
    keys = list(corr.columns)
    pairs = [
        {"a": a, "b": b, "corr": round(float(corr.loc[a, b]), 3)}
        for i, a in enumerate(keys)
        for b in keys[i + 1:]
    ]
    matrix = {
        a: {b: round(float(corr.loc[a, b]), 3) for b in keys}
        for a in keys
    }
    return {"pairs": pairs, "matrix": matrix, "weeks": len(combined)}


async def backtest_asset(
    asset: str,
    *,
    data_source: str,
    start: str,
    interval: str,
    oos_split: float,
    regime_baselines: dict[str, float],
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

    cfg = _default_squeeze_cfg(asset)
    full_res = run_backtest(df, cfg, verbose=False)
    oos_res = run_backtest(df_oos, cfg, verbose=False)

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
        st_baseline_oos=st_baseline_oos,
        regime_oos_pnl=regime_baselines.get(asset),
    )
    print(
        f"  {asset}: {row['full_trades']} trades, squeeze {row['squeeze_pct_full']:.1f}%, "
        f"OOS {row['oos_pnl_pct']:+.1f}% (ST {row['st_baseline_oos_pnl_pct']}), "
        f"GO={'YES' if row['go'] else 'NO'}"
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
    regime_baselines = _load_regime_baselines()
    print("Config: BB Squeeze Breakout H1 (squeeze → band breakout)")
    print(f"Data: {data_source}, interval={interval}, start={start}\n")

    rows: list[dict[str, Any]] = []
    for asset in assets:
        row = await backtest_asset(
            asset,
            data_source=data_source,
            start=start,
            interval=interval,
            oos_split=oos_split,
            regime_baselines=regime_baselines,
        )
        if row:
            rows.append(row)

    forex_rows = [r for r in rows if r["asset"] in FOREX_MAJORS]
    forex_go = [r for r in forex_rows if r["go"]]
    best_forex = None
    if forex_rows:
        best_forex = max(forex_rows, key=lambda r: r["oos_pnl_pct"])

    corr_assets = [a for a in assets if a in ("GC=F", "USD/JPY", "EUR/USD", "GBP/USD", "AUD/USD")]
    print("\nComputing weekly return correlations...")
    correlations = await compute_correlations(
        corr_assets,
        data_source=data_source,
        start=start,
        interval=interval,
    )

    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": "bb_squeeze_breakout",
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
            "recommended_forex_leg": best_forex["asset"] if best_forex and best_forex["oos_pnl_pct"] > 0 else None,
            "recommended_forex_oos_pnl_pct": best_forex["oos_pnl_pct"] if best_forex else None,
        },
        "correlations": correlations,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2))
    print(f"\nSaved: {output}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="BB Squeeze Breakout forex backtest")
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
        default=Path("artifacts/squeeze_forex_backtest.json"),
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
