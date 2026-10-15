#!/usr/bin/env python3
"""Monthly breakdown: Gold + USD/JPY portfolio (hf_ls01), last 12 months."""
from __future__ import annotations

import sys as _sys, pathlib as _path
_sys.path.insert(0, str(_path.Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import dataclasses
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from trading_lab.adapters.market_data import get_market_adapter
from trading_lab.backtest import _compute_metrics
from trading_lab.strategies.supertrend_volume import SupertrendVolumeConfig, run_backtest

ASSETS = [("GC=F", "Gold", False), ("USD/JPY", "JPY", False)]


def _load_hf_config(path: Path) -> SupertrendVolumeConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    valid = {f.name for f in dataclasses.fields(SupertrendVolumeConfig)}
    return SupertrendVolumeConfig(**{k: v for k, v in raw.items() if k in valid})


def _month_key(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")


def _month_metrics(
    trades: list[dict],
    *,
    account_size: float,
) -> dict[str, Any]:
    closed = sorted(
        [t for t in trades if t.get("status") == "closed"],
        key=lambda t: t.get("exit_time") or 0,
    )
    if not closed:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "pnl_pct": 0.0,
            "max_dd_pct": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "gold_trades": 0,
            "gold_pnl_pct": 0.0,
            "jpy_trades": 0,
            "jpy_pnl_pct": 0.0,
            "long_trades": 0,
            "short_trades": 0,
        }

    equity = account_size
    equity_curve = [equity]
    for t in closed:
        equity_curve.append(equity + account_size * ((t.get("pnl_pct") or 0) / 100))
        equity = equity_curve[-1]

    m = _compute_metrics(closed, equity_curve, SupertrendVolumeConfig(account_size=account_size))

    by_asset: dict[str, list[float]] = defaultdict(list)
    for t in closed:
        by_asset[t.get("asset", "?")].append(t.get("pnl_pct") or 0)

    long_n = sum(1 for t in closed if t.get("direction") == "long")
    short_n = sum(1 for t in closed if t.get("direction") == "short")

    return {
        "trades": len(closed),
        "win_rate": round(m["win_rate"], 1),
        "pnl_pct": round(m["total_pnl_pct"], 2),
        "max_dd_pct": round(m["max_drawdown_pct"], 2),
        "profit_factor": round(m["profit_factor"], 2),
        "sharpe": round(m["sharpe"], 2),
        "gold_trades": len(by_asset.get("GC=F", [])),
        "gold_pnl_pct": round(sum(by_asset.get("GC=F", [])), 2),
        "jpy_trades": len(by_asset.get("USD/JPY", [])),
        "jpy_pnl_pct": round(sum(by_asset.get("USD/JPY", [])), 2),
        "long_trades": long_n,
        "short_trades": short_n,
    }


async def run_monthly(
    *,
    data_source: str,
    months_back: int,
    interval: str,
    output: Path,
) -> dict[str, Any]:
    base_cfg = _load_hf_config(Path("hf_baseline.yaml"))
    adapter = get_market_adapter(data_source)

    # Fetch enough history: ~18 months buffer
    start_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    all_trades: list[dict] = []

    for asset, label, long_only in ASSETS:
        df = await adapter.fetch_history(asset, interval=interval, start=start_dt)
        if df.empty:
            continue
        cfg = dataclasses.replace(base_cfg, asset=asset, long_only=long_only)
        result = run_backtest(df, cfg, verbose=False)
        for t in result["trades"]:
            if t.get("status") != "closed":
                continue
            td = dict(t)
            td["asset"] = asset
            td["asset_label"] = label
            all_trades.append(td)

    if not all_trades:
        raise RuntimeError("No trades")

    # Determine month window from latest trade
    latest = max(t["exit_time"] for t in all_trades)
    end_dt = datetime.fromtimestamp(latest, tz=timezone.utc)
    start_month = end_dt.replace(day=1)
    month_keys: list[str] = []
    y, m = start_month.year, start_month.month
    for _ in range(months_back):
        month_keys.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    month_keys = sorted(month_keys)

    # Portfolio cumulative equity (for running DD across months)
    all_sorted = sorted(all_trades, key=lambda t: t["exit_time"])
    cum_equity = base_cfg.account_size
    running_peak = cum_equity
    portfolio_rows: list[dict[str, Any]] = []

    for mk in month_keys:
        month_trades = [t for t in all_sorted if _month_key(t["exit_time"]) == mk]
        mm = _month_metrics(month_trades, account_size=base_cfg.account_size)

        # Running portfolio equity through month
        month_start_equity = cum_equity
        for t in sorted(month_trades, key=lambda t: t["exit_time"]):
            cum_equity += base_cfg.account_size * ((t.get("pnl_pct") or 0) / 100)
            running_peak = max(running_peak, cum_equity)
        month_end_equity = cum_equity
        running_dd = (
            (cum_equity - running_peak) / running_peak * 100 if running_peak > 0 else 0
        )

        portfolio_rows.append({
            "month": mk,
            **mm,
            "equity_start": round(month_start_equity, 2),
            "equity_end": round(month_end_equity, 2),
            "running_dd_pct": round(running_dd, 2),
        })

    total_pnl = sum(r["pnl_pct"] for r in portfolio_rows)
    total_trades = sum(r["trades"] for r in portfolio_rows)
    winning_months = sum(1 for r in portfolio_rows if r["pnl_pct"] > 0)

    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "portfolio": "GC=F + USD/JPY",
        "config": "hf_ls01",
        "data_source": data_source,
        "interval": interval,
        "months": months_back,
        "window": f"{month_keys[0]} → {month_keys[-1]}",
        "account_size": base_cfg.account_size,
        "risk_per_trade": base_cfg.risk_per_trade,
        "summary": {
            "total_pnl_pct": round(total_pnl, 2),
            "avg_pnl_per_month": round(total_pnl / len(month_keys), 2) if month_keys else 0,
            "total_trades": total_trades,
            "avg_trades_per_month": round(total_trades / len(month_keys), 1) if month_keys else 0,
            "winning_months": winning_months,
            "losing_months": len(month_keys) - winning_months,
            "final_equity": round(cum_equity, 2),
        },
        "monthly": portfolio_rows,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2))
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-source", default="dukascopy")
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--output", type=Path, default=Path("artifacts/gold_jpy_monthly.json"))
    args = parser.parse_args()
    data = asyncio.run(
        run_monthly(
            data_source=args.data_source,
            months_back=args.months,
            interval=args.interval,
            output=args.output,
        )
    )

    print(f"Gold + JPY monthly ({data['window']}) — hf_ls01, {args.interval}")
    print(f"{'Monat':<8} {'Trades':>6} {'PnL%':>7} {'DD%':>7} {'WR%':>6} {'Gold':>7} {'JPY':>7} {'RunDD':>7}")
    print("-" * 62)
    for r in data["monthly"]:
        print(
            f"{r['month']:<8} {r['trades']:>6} {r['pnl_pct']:>+6.1f}% "
            f"{r['max_dd_pct']:>6.1f}% {r['win_rate']:>5.0f}% "
            f"{r['gold_pnl_pct']:>+6.1f}% {r['jpy_pnl_pct']:>+6.1f}% "
            f"{r['running_dd_pct']:>6.1f}%"
        )
    s = data["summary"]
    print("-" * 62)
    print(
        f"Gesamt: {s['total_pnl_pct']:+.1f}% über {args.months} Monate "
        f"(Ø {s['avg_pnl_per_month']:+.2f}%/Mo), {s['total_trades']} Trades, "
        f"{s['winning_months']} grüne / {s['losing_months']} rote Monate"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
