#!/usr/bin/env python3
"""Monthly breakdown for JPY evolved v48831 vs jpy_ls01 (Dukascopy from 2020)."""
from __future__ import annotations

import sys as _sys, pathlib as _path
_sys.path.insert(0, str(_path.Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from trading_lab.adapters.market_data import get_market_adapter
from trading_lab.backtest import _compute_metrics
from trading_lab.strategies.supertrend_volume import SupertrendVolumeConfig, run_backtest


def _load_cfg(path: Path) -> SupertrendVolumeConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    valid = {f.name for f in dataclasses.fields(SupertrendVolumeConfig)}
    return SupertrendVolumeConfig(**{k: v for k, v in raw.items() if k in valid})


def _month_key(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")


def _month_metrics(trades: list[dict], *, account_size: float) -> dict[str, Any]:
    closed = sorted(
        [t for t in trades if t.get("status") == "closed"],
        key=lambda t: t.get("exit_time") or 0,
    )
    if not closed:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "pnl_pct": 0.0,
            "max_dd_pct": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "long_pnl_pct": 0.0,
            "short_pnl_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
        }

    equity = account_size
    equity_curve = [equity]
    for t in closed:
        equity += account_size * ((t.get("pnl_pct") or 0) / 100)
        equity_curve.append(equity)

    m = _compute_metrics(closed, equity_curve, SupertrendVolumeConfig(account_size=account_size))
    long_t = [t for t in closed if t.get("direction") == "long"]
    short_t = [t for t in closed if t.get("direction") == "short"]
    long_pnls = [t.get("pnl_pct") or 0 for t in long_t]
    short_pnls = [t.get("pnl_pct") or 0 for t in short_t]

    return {
        "trades": len(closed),
        "wins": m["wins"],
        "losses": m["losses"],
        "win_rate": round(m["win_rate"], 1),
        "pnl_pct": round(m["total_pnl_pct"], 2),
        "max_dd_pct": round(m["max_drawdown_pct"], 2),
        "profit_factor": round(m["profit_factor"], 2),
        "sharpe": round(m["sharpe"], 2),
        "long_trades": len(long_t),
        "short_trades": len(short_t),
        "long_pnl_pct": round(sum(long_pnls), 2),
        "short_pnl_pct": round(sum(short_pnls), 2),
        "avg_win_pct": m.get("avg_win_pct", 0),
        "avg_loss_pct": m.get("avg_loss_pct", 0),
    }


def _build_monthly_rows(
    trades: list[dict],
    *,
    account_size: float,
) -> list[dict[str, Any]]:
    closed = [t for t in trades if t.get("status") == "closed" and t.get("exit_time")]
    if not closed:
        return []

    months = sorted({_month_key(t["exit_time"]) for t in closed})
    cum_equity = account_size
    running_peak = cum_equity
    rows: list[dict[str, Any]] = []

    for mk in months:
        month_trades = [t for t in closed if _month_key(t["exit_time"]) == mk]
        mm = _month_metrics(month_trades, account_size=account_size)
        month_start = cum_equity
        for t in sorted(month_trades, key=lambda x: x["exit_time"]):
            cum_equity += account_size * ((t.get("pnl_pct") or 0) / 100)
            running_peak = max(running_peak, cum_equity)
        running_dd = (cum_equity - running_peak) / running_peak * 100 if running_peak > 0 else 0
        rows.append({
            "month": mk,
            **mm,
            "equity_start": round(month_start, 2),
            "equity_end": round(cum_equity, 2),
            "running_dd_pct": round(running_dd, 2),
        })
    return rows


async def run_monthly(
    *,
    start: str,
    interval: str,
    data_source: str,
    output: Path,
) -> dict[str, Any]:
    adapter = get_market_adapter(data_source)
    start_dt = datetime.fromisoformat(start)
    df = await adapter.fetch_history("USD/JPY", interval=interval, start=start_dt)
    if df.empty:
        raise RuntimeError("No USD/JPY data")

    configs = {
        "jpy_v48831": _load_cfg(Path("jpy_evolved_v48831.yaml")),
        "jpy_ls01": _load_cfg(Path("jpy_baseline.yaml")),
    }

    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset": "USD/JPY",
        "data_source": data_source,
        "interval": interval,
        "start": start,
        "end": str(df.index[-1]),
        "candles": len(df),
        "configs": {},
    }

    for label, cfg in configs.items():
        cfg = dataclasses.replace(cfg, asset="USD/JPY", asset_label="USD/JPY")
        trades = run_backtest(df, cfg, verbose=False)["trades"]
        rows = _build_monthly_rows(trades, account_size=cfg.account_size)
        total_pnl = sum(r["pnl_pct"] for r in rows)
        total_trades = sum(r["trades"] for r in rows)
        winning = sum(1 for r in rows if r["pnl_pct"] > 0)
        result["configs"][label] = {
            "summary": {
                "months": len(rows),
                "window": f"{rows[0]['month']} → {rows[-1]['month']}" if rows else "",
                "total_pnl_pct": round(total_pnl, 2),
                "avg_pnl_per_month": round(total_pnl / len(rows), 2) if rows else 0,
                "total_trades": total_trades,
                "avg_trades_per_month": round(total_trades / len(rows), 1) if rows else 0,
                "winning_months": winning,
                "losing_months": len(rows) - winning,
                "final_equity": rows[-1]["equity_end"] if rows else cfg.account_size,
            },
            "monthly": rows,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--data-source", default="dukascopy")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/jpy_evolved_v48831_monthly.json"),
    )
    args = parser.parse_args()
    data = asyncio.run(
        run_monthly(
            start=args.start,
            interval=args.interval,
            data_source=args.data_source,
            output=args.output,
        )
    )

    for label in ("jpy_v48831", "jpy_ls01"):
        block = data["configs"][label]
        s = block["summary"]
        print(f"\n{'='*72}")
        print(f"{label} — {s['window']} ({s['months']} Monate)")
        print(f"{'Monat':<8} {'Tr':>4} {'W/L':>7} {'PnL%':>7} {'MoDD%':>7} {'RunDD%':>8} {'WR%':>5} {'PF':>5} {'Sharpe':>7}")
        print("-" * 72)
        for r in block["monthly"]:
            wl = f"{r['wins']}/{r['losses']}"
            print(
                f"{r['month']:<8} {r['trades']:>4} {wl:>7} {r['pnl_pct']:>+6.2f}% "
                f"{r['max_dd_pct']:>6.2f}% {r['running_dd_pct']:>7.2f}% "
                f"{r['win_rate']:>4.0f}% {r['profit_factor']:>5.2f} {r['sharpe']:>7.2f}"
            )
        print("-" * 72)
        print(
            f"Gesamt: {s['total_pnl_pct']:+.2f}% | Ø {s['avg_pnl_per_month']:+.2f}%/Mo | "
            f"{s['total_trades']} Trades | {s['winning_months']} grün / {s['losing_months']} rot | "
            f"Equity ${s['final_equity']:,.0f}"
        )

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
