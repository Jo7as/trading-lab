"""Compare live signal-bot trades vs historical simulation replay.

Reads ``live_trades.jsonl`` and optional ``position.json`` from the signal
worker state directory, replays the same hf_ls01 logic on yfinance 1h data,
and reports match/mismatch per trade.

Usage:
  uv run python compare_live_sim.py
  uv run python compare_live_sim.py --state-dir ./live-state-template
  uv run python compare_live_sim.py --backtest --period 730d
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .adapters.market_data import get_market_adapter
from .risk import Position, RiskManager
from .strategies.supertrend_volume import (
    SupertrendVolumeConfig,
    generate_signals,
    manage_position,
)


def load_strategy(path: Path) -> SupertrendVolumeConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    valid = {f.name for f in dataclasses.fields(SupertrendVolumeConfig)}
    return SupertrendVolumeConfig(**{k: v for k, v in raw.items() if k in valid})


def load_live_trades(state_dir: Path) -> list[dict[str, Any]]:
    path = state_dir / "live_trades.jsonl"
    if not path.exists():
        return []
    trades: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            trades.append(json.loads(line))
    return trades


def load_open_position(state_dir: Path) -> dict[str, Any] | None:
    path = state_dir / "position.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    pos = raw.get("position")
    return pos if pos and pos.get("status") == "open" else None


def _ts_str(ts: float | None) -> str:
    if ts is None:
        return "?"
    return pd.to_datetime(ts, unit="s", utc=True).strftime("%Y-%m-%d %H:%M UTC")


def replay_trades(
    df: pd.DataFrame,
    cfg: SupertrendVolumeConfig,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> list[dict[str, Any]]:
    """Bar-by-bar replay matching signal_loop entry/exit semantics."""
    df_ind = generate_signals(df, cfg)
    df_ind = df_ind.dropna(
        subset=["atr", "ema_slow_val", "supertrend", "st_dir"]
    ).copy()

    if start is not None:
        start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
    if end is not None:
        end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")

    rm = RiskManager(cfg, account_size=cfg.account_size, backtest_mode=True)
    open_pos: Position | None = None
    closed: list[dict[str, Any]] = []

    for i in range(len(df_ind)):
        row = df_ind.iloc[i]
        bar_ts = pd.Timestamp(row.name)
        if bar_ts.tzinfo is None:
            bar_ts = bar_ts.tz_localize("UTC")
        else:
            bar_ts = bar_ts.tz_convert("UTC")

        if end is not None and bar_ts > end:
            break

        t = float(bar_ts.timestamp())
        price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])

        if open_pos is not None and open_pos.status == "open":
            pending: list[dict] = []
            manage_position(
                open_pos, row, i, t, price, high, low, cfg, rm, pending
            )
            if open_pos.status == "closed":
                entry_bar = pd.to_datetime(
                    open_pos.entry_time, unit="s", utc=True
                )
                if start is None or entry_bar >= start:
                    closed.append(asdict(open_pos))
                open_pos = None

        if open_pos is None:
            allowed, _ = rm.can_trade(t)
            if allowed and float(row.get("atr", 0)) > 0:
                direction: str | None = None
                if row.get("long_signal", 0) > 0:
                    direction = "long"
                elif row.get("short_signal", 0) > 0 and not cfg.long_only:
                    direction = "short"
                if direction:
                    pos = rm.open_position(
                        price, float(row["atr"]), direction, t, i
                    )
                    if pos:
                        open_pos = pos

    if open_pos is not None and open_pos.status == "open":
        entry_bar = pd.to_datetime(open_pos.entry_time, unit="s", utc=True)
        if start is None or entry_bar >= start:
            closed.append({**asdict(open_pos), "status": "open"})

    return closed


def _match_key(trade: dict[str, Any]) -> tuple[str, str]:
    et = _ts_str(trade.get("entry_time"))
    return (et, str(trade.get("direction", "?")))


def compare_trade_lists(
    live: list[dict[str, Any]],
    sim: list[dict[str, Any]],
    *,
    entry_tol_pct: float = 0.3,
    pnl_tol_pct: float = 0.15,
) -> dict[str, Any]:
    """Pair live and simulated trades by entry time + direction."""
    live_closed = [t for t in live if t.get("status") == "closed"]
    sim_closed = [t for t in sim if t.get("status") == "closed"]

    live_by_key = {_match_key(t): t for t in live_closed}
    sim_by_key = {_match_key(t): t for t in sim_closed}

    all_keys = sorted(set(live_by_key) | set(sim_by_key))
    rows: list[dict[str, Any]] = []

    for key in all_keys:
        lt = live_by_key.get(key)
        st = sim_by_key.get(key)
        if lt and st:
            entry_live = float(lt.get("entry_price") or 0)
            entry_sim = float(st.get("entry_price") or 0)
            entry_diff = (
                abs(entry_live - entry_sim) / entry_live * 100
                if entry_live
                else 0.0
            )
            pnl_live = float(lt.get("pnl_pct") or 0)
            pnl_sim = float(st.get("pnl_pct") or 0)
            status = "match"
            notes: list[str] = []
            if entry_diff > entry_tol_pct:
                status = "mismatch"
                notes.append(f"entry Δ{entry_diff:.2f}%")
            if abs(pnl_live - pnl_sim) > pnl_tol_pct:
                status = "mismatch"
                notes.append(f"pnl Δ{pnl_live - pnl_sim:+.2f}%")
            if lt.get("exit_reason") != st.get("exit_reason"):
                notes.append(
                    f"exit {lt.get('exit_reason')} vs {st.get('exit_reason')}"
                )
                if status == "match" and notes:
                    status = "warn"
            rows.append(
                {
                    "key": key,
                    "status": status,
                    "direction": key[1],
                    "entry_time": key[0],
                    "live_entry": entry_live,
                    "sim_entry": entry_sim,
                    "live_pnl": pnl_live,
                    "sim_pnl": pnl_sim,
                    "live_exit": lt.get("exit_reason"),
                    "sim_exit": st.get("exit_reason"),
                    "notes": "; ".join(notes),
                }
            )
        elif lt:
            rows.append(
                {
                    "key": key,
                    "status": "live_only",
                    "direction": key[1],
                    "entry_time": key[0],
                    "live_entry": lt.get("entry_price"),
                    "sim_entry": None,
                    "live_pnl": lt.get("pnl_pct"),
                    "sim_pnl": None,
                    "live_exit": lt.get("exit_reason"),
                    "sim_exit": None,
                    "notes": "in live log, not in replay",
                }
            )
        else:
            assert st is not None
            rows.append(
                {
                    "key": key,
                    "status": "sim_only",
                    "direction": key[1],
                    "entry_time": key[0],
                    "live_entry": None,
                    "sim_entry": st.get("entry_price"),
                    "live_pnl": None,
                    "sim_pnl": st.get("pnl_pct"),
                    "live_exit": None,
                    "sim_exit": st.get("exit_reason"),
                    "notes": "in replay, not in live log",
                }
            )

    matched = sum(1 for r in rows if r["status"] == "match")
    warns = sum(1 for r in rows if r["status"] == "warn")
    mismatches = sum(1 for r in rows if r["status"] == "mismatch")

    return {
        "rows": rows,
        "live_count": len(live_closed),
        "sim_count": len(sim_closed),
        "matched": matched,
        "warn": warns,
        "mismatch": mismatches,
        "live_only": sum(1 for r in rows if r["status"] == "live_only"),
        "sim_only": sum(1 for r in rows if r["status"] == "sim_only"),
    }


def format_report(
    result: dict[str, Any],
    *,
    open_pos: dict[str, Any] | None = None,
) -> str:
    lines = [
        "=" * 72,
        "LIVE vs SIMULATION — hf_ls01",
        "=" * 72,
        f"Live trades (closed):  {result['live_count']}",
        f"Sim trades (closed):   {result['sim_count']}",
        f"Matched:               {result['matched']}",
        f"Warnings:              {result['warn']}",
        f"Mismatches:            {result['mismatch']}",
        f"Live only:             {result['live_only']}",
        f"Sim only:              {result['sim_only']}",
    ]
    if open_pos:
        lines.append(
            f"Open position:         {open_pos.get('direction')} "
            f"@ {open_pos.get('entry_price')} "
            f"(since {_ts_str(open_pos.get('entry_time'))})"
        )
    lines.append("")

    if not result["rows"]:
        lines.append("No trades to compare yet — run again after first live signals.")
        return "\n".join(lines)

    if result["live_count"] == 0 and result["sim_count"] > 0:
        lines.append(
            "Baseline mode: no live trades yet — sim column shows expected "
            "replay for the window."
        )
        lines.append("")
    for r in result["rows"]:
        flag = r["status"].upper()
        lines.append(f"[{flag}] {r['entry_time']} {r['direction'].upper()}")
        if r["live_entry"] is not None:
            lines.append(
                f"  live: entry={r['live_entry']:.2f} "
                f"pnl={r['live_pnl']:+.2f}% exit={r['live_exit']}"
            )
        if r["sim_entry"] is not None:
            lines.append(
                f"  sim:  entry={r['sim_entry']:.2f} "
                f"pnl={r['sim_pnl']:+.2f}% exit={r['sim_exit']}"
            )
        if r["notes"]:
            lines.append(f"  → {r['notes']}")
        lines.append("")

    return "\n".join(lines)


async def run_compare(
    state_dir: Path,
    strategy_path: Path,
    *,
    period: str = "120d",
    warmup_days: int = 90,
) -> str:
    cfg = load_strategy(strategy_path)
    live = load_live_trades(state_dir)
    open_pos = load_open_position(state_dir)

    adapter = get_market_adapter()
    df = await adapter.fetch_history(cfg.asset, interval="1h", period=period)
    if df.empty:
        return "ERROR: no yfinance data"

    if live:
        entry_times = [
            pd.to_datetime(t["entry_time"], unit="s", utc=True)
            for t in live
            if t.get("entry_time")
        ]
        start = min(entry_times) - pd.Timedelta(days=1)
    else:
        start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=60)

    end = pd.Timestamp.now(tz="UTC")
    # extend fetch window for indicator warmup
    _ = warmup_days

    sim = replay_trades(df, cfg, start=start, end=end)
    result = compare_trade_lists(live, sim)
    return format_report(result, open_pos=open_pos)

