#!/usr/bin/env python3
"""Portfolio backtest: Gold + Silver (long only) + USD/JPY with hf_ls01."""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from trading_lab.adapters.market_data import get_market_adapter
from trading_lab.backtest import _compute_metrics, _compute_side_metrics
from trading_lab.score import trades_per_week
from trading_lab.strategies.supertrend_volume import SupertrendVolumeConfig, run_backtest

PORTFOLIO: list[tuple[str, bool]] = [
    ("GC=F", False),       # long+short
    ("XAG/USD", True),     # long only
    ("USD/JPY", False),    # long+short
]


def _load_hf_config(path: Path) -> SupertrendVolumeConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    valid = {f.name for f in dataclasses.fields(SupertrendVolumeConfig)}
    return SupertrendVolumeConfig(**{k: v for k, v in raw.items() if k in valid})


def _baseline_path() -> Path:
    for candidate in (Path("hf_baseline.yaml"), Path(__file__).resolve().parent / "hf_baseline.yaml"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("hf_baseline.yaml not found")


def _portfolio_metrics(
    trades: list[dict],
    *,
    period_days: float,
    cfg: SupertrendVolumeConfig,
) -> dict[str, Any]:
    if not trades:
        base = _compute_metrics([], [cfg.account_size], cfg)
        base["trades_per_week"] = 0.0
        return base
    closed = sorted(
        [t for t in trades if t.get("status") == "closed"],
        key=lambda t: t.get("exit_time") or 0,
    )
    equity = cfg.account_size
    equity_curve = [equity]
    for t in closed:
        pnl_pct = t.get("pnl_pct", 0.0) or 0.0
        equity += equity * (pnl_pct / 100.0) if False else cfg.account_size * (pnl_pct / 100.0)
        # fixed notional: each trade pnl is % of account_size, additive
        equity_curve.append(equity)

    # Rebuild with additive fixed sizing (matches run_backtest)
    equity = cfg.account_size
    equity_curve = [equity]
    cum_pnl_pct = 0.0
    for t in closed:
        cum_pnl_pct += t.get("pnl_pct", 0.0) or 0.0
        equity_curve.append(cfg.account_size * (1 + cum_pnl_pct / 100.0))

    metrics = _compute_metrics(closed, equity_curve, cfg)
    metrics["trades_per_week"] = round(trades_per_week(closed, period_days), 2)
    metrics["by_asset"] = {}
    for asset in {t.get("asset") for t in closed}:
        asset_trades = [t for t in closed if t.get("asset") == asset]
        pnls = [t.get("pnl_pct", 0.0) or 0.0 for t in asset_trades]
        metrics["by_asset"][asset] = {
            "trades": len(asset_trades),
            "pnl_pct": round(sum(pnls), 2),
            "win_rate": round(len([p for p in pnls if p > 0]) / len(pnls) * 100, 1) if pnls else 0,
        }
    return metrics


def _overlap_stats(trades: list[dict]) -> dict[str, Any]:
    """How often multiple assets had open positions simultaneously."""
    events: list[tuple[float, int, str]] = []
    for t in trades:
        if t.get("status") != "closed":
            continue
        et = t.get("entry_time")
        xt = t.get("exit_time")
        asset = t.get("asset", "?")
        if et is None or xt is None:
            continue
        events.append((float(et), 1, asset))
        events.append((float(xt), -1, asset))
    events.sort(key=lambda x: (x[0], -x[1]))

    open_assets: dict[str, int] = {}
    max_concurrent = 0
    samples_2plus = 0
    samples_3 = 0
    for _ts, delta, asset in events:
        if delta > 0:
            open_assets[asset] = open_assets.get(asset, 0) + 1
        else:
            open_assets[asset] = open_assets.get(asset, 0) - 1
            if open_assets[asset] <= 0:
                open_assets.pop(asset, None)
        n = len(open_assets)
        max_concurrent = max(max_concurrent, n)
        if n >= 2:
            samples_2plus += 1
        if n >= 3:
            samples_3 += 1
    return {
        "max_concurrent_assets": max_concurrent,
        "event_samples_2plus": samples_2plus,
        "event_samples_3": samples_3,
    }


async def run_portfolio(
    *,
    data_source: str,
    start: str,
    interval: str,
    oos_split: float,
    output: Path,
) -> dict[str, Any]:
    base_cfg = _load_hf_config(_baseline_path())
    adapter = get_market_adapter(data_source)
    start_dt = datetime.fromisoformat(start)

    all_trades: list[dict] = []
    per_asset: list[dict[str, Any]] = []

    print(f"Portfolio: Gold L+S + Silver long only + USD/JPY L+S")
    print(f"Config: hf_ls01, {interval}, {data_source}, from {start}\n")

    oos_cutoff_ts: float | None = None
    period_days = 0.0

    for asset, long_only in PORTFOLIO:
        df = await adapter.fetch_history(asset, interval=interval, start=start_dt)
        if df.empty:
            print(f"  SKIP {asset}: no data")
            continue

        if oos_cutoff_ts is None:
            split = int(len(df) * (1 - oos_split))
            oos_cutoff_ts = float(df.iloc[split].name.timestamp())
            period_days = max((df.index[-1] - df.index[0]).days, 1)

        cfg = dataclasses.replace(
            base_cfg,
            asset=asset,
            long_only=long_only,
        )
        result = run_backtest(df, cfg, verbose=False)
        m = result["metrics"]
        tagged = []
        for t in result["trades"]:
            if t.get("status") == "closed":
                td = dict(t)
                td["asset"] = asset
                tagged.append(td)
        all_trades.extend(tagged)

        per_asset.append({
            "asset": asset,
            "long_only": long_only,
            "full_trades": m["total_trades"],
            "full_tpw": m.get("trades_per_week", 0),
            "full_pnl_pct": round(m["total_pnl_pct"], 2),
            "full_sharpe": round(m["sharpe"], 2),
            "full_dd_pct": round(m["max_drawdown_pct"], 2),
        })
        print(
            f"  {asset} (long_only={long_only}): {m['total_trades']} trades, "
            f"{m['total_pnl_pct']:+.1f}% full, Sharpe {m['sharpe']:.2f}"
        )

    oos_trades = [
        t for t in all_trades
        if oos_cutoff_ts is not None and (t.get("entry_time") or 0) >= oos_cutoff_ts
    ]

    full_m = _portfolio_metrics(all_trades, period_days=period_days, cfg=base_cfg)
    oos_days = max(period_days * oos_split, 1)
    oos_m = _portfolio_metrics(oos_trades, period_days=oos_days, cfg=base_cfg)
    overlap = _overlap_stats(all_trades)

    # Naive sum of individual full pnl (independent $10k each — upper bound)
    naive_sum_pnl = sum(a["full_pnl_pct"] for a in per_asset)

    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "portfolio": "GC=F + XAG/USD(long_only) + USD/JPY",
        "config": "hf_ls01",
        "data_source": data_source,
        "interval": interval,
        "start": start,
        "oos_split": oos_split,
        "per_asset": per_asset,
        "portfolio_full": {
            "trades": full_m["total_trades"],
            "trades_per_week": full_m.get("trades_per_week", 0),
            "trades_per_month": round(full_m["total_trades"] / (period_days / 30.44), 1),
            "win_rate": round(full_m["win_rate"], 1),
            "total_pnl_pct": round(full_m["total_pnl_pct"], 2),
            "pnl_per_month": round(full_m["total_pnl_pct"] / (period_days / 30.44), 2),
            "sharpe": round(full_m["sharpe"], 2),
            "max_drawdown_pct": round(full_m["max_drawdown_pct"], 2),
            "profit_factor": round(full_m["profit_factor"], 2),
            "by_asset": full_m.get("by_asset", {}),
        },
        "portfolio_oos": {
            "trades": oos_m["total_trades"],
            "trades_per_week": oos_m.get("trades_per_week", 0),
            "trades_per_month": round(oos_m["total_trades"] / (oos_days / 30.44), 1),
            "win_rate": round(oos_m["win_rate"], 1),
            "total_pnl_pct": round(oos_m["total_pnl_pct"], 2),
            "pnl_per_month": round(oos_m["total_pnl_pct"] / (oos_days / 30.44), 2),
            "sharpe": round(oos_m["sharpe"], 2),
            "max_drawdown_pct": round(oos_m["max_drawdown_pct"], 2),
            "profit_factor": round(oos_m["profit_factor"], 2),
            "by_asset": oos_m.get("by_asset", {}),
        },
        "overlap": overlap,
        "naive_independent_sum_full_pnl_pct": round(naive_sum_pnl, 2),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2))

    print(f"\n--- Portfolio (shared $10k, 1% risk/trade, max 1 pos/asset) ---")
    print(f"  Full: {data['portfolio_full']['trades']} trades "
          f"({data['portfolio_full']['trades_per_week']}/wk), "
          f"PnL {data['portfolio_full']['total_pnl_pct']:+.1f}%, "
          f"DD {data['portfolio_full']['max_drawdown_pct']:.1f}%")
    print(f"  OOS:  {data['portfolio_oos']['trades']} trades "
          f"({data['portfolio_oos']['trades_per_week']}/wk), "
          f"PnL {data['portfolio_oos']['total_pnl_pct']:+.1f}%, "
          f"~{data['portfolio_oos']['pnl_per_month']:+.2f}%/mo")
    print(f"  Max concurrent assets: {overlap['max_concurrent_assets']}")
    print(f"\nSaved: {output}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio backtest Gold+Silver+JPY")
    parser.add_argument("--data-source", default="dukascopy")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--oos-split", type=float, default=0.30)
    parser.add_argument("--output", type=Path, default=Path("artifacts/portfolio_gold_xag_jpy.json"))
    args = parser.parse_args()
    asyncio.run(
        run_portfolio(
            data_source=args.data_source,
            start=args.start,
            interval=args.interval,
            oos_split=args.oos_split,
            output=args.output,
        )
    )


if __name__ == "__main__":
    main()
