#!/usr/bin/env python3
"""Attribute SMC exits (TP2 / TP1+BE / full SL) to setup features.

Runs 2024→now (or --start/--end) with the reference strategy.yaml and prints
which tags correlate with reaching TP2 vs dying at SL.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path

import yaml

from trading_lab.adapters.dukascopy_adapter import DukascopyAdapter
from trading_lab.strategies.smc_ict import SmcIctConfig, run_backtest


def _load_cfg(path: Path) -> SmcIctConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    valid = {f.name for f in fields(SmcIctConfig)}
    return SmcIctConfig(**{k: v for k, v in raw.items() if k in valid})


def _outcome(t: dict) -> str:
    """Bucket trade into path quality."""
    reason = t.get("exit_reason") or ""
    tp1 = bool(t.get("tp1_hit"))
    tp2 = bool(t.get("tp2_hit")) or reason == "tp2"
    if tp2:
        return "tp2"
    if tp1 and reason == "stop_loss":
        return "tp1_then_be"  # partial win, runner stopped at BE
    if reason == "stop_loss" and not tp1:
        return "full_sl"
    if reason == "time_stop":
        return "time_stop" if tp1 else "time_stop_no_tp1"
    return reason or "other"


def _tag(t: dict, key: str) -> str:
    meta = t.get("meta") or {}
    if isinstance(meta, dict) and key in meta and meta[key] is not None:
        return str(meta[key])
    return str(t.get(key, "na"))


def _slice_table(trades: list[dict], key: str) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        groups[_tag(t, key)].append(t)

    print(f"\n--- by {key} ---")
    print(
        f"{'bucket':<16} {'n':>5} {'tp2%':>6} {'tp1be%':>7} {'fullSL%':>8} "
        f"{'WR%':>6} {'avgPnL':>8} {'sumPnL':>8}"
    )
    rows = []
    for bucket, ts in groups.items():
        n = len(ts)
        if n == 0:
            continue
        outs = [_outcome(t) for t in ts]
        tp2 = 100.0 * sum(1 for o in outs if o == "tp2") / n
        tp1be = 100.0 * sum(1 for o in outs if o == "tp1_then_be") / n
        fsl = 100.0 * sum(1 for o in outs if o == "full_sl") / n
        pnls = [float(t.get("pnl_pct") or 0) for t in ts]
        wr = 100.0 * sum(1 for p in pnls if p > 0) / n
        rows.append(
            (
                n,
                bucket,
                tp2,
                tp1be,
                fsl,
                wr,
                sum(pnls) / n,
                sum(pnls),
            )
        )
    rows.sort(key=lambda r: (-r[2], -r[0]))  # tp2% then n
    for n, bucket, tp2, tp1be, fsl, wr, avg, total in rows:
        print(
            f"{bucket:<16} {n:5d} {tp2:5.1f}% {tp1be:6.1f}% {fsl:7.1f}% "
            f"{wr:5.1f}% {avg:+7.3f}% {total:+7.2f}%"
        )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("smc-state/strategy.yaml"))
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--enable-b", action="store_true", help="B premium/discount 15m_swing")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    if args.enable_b:
        cfg = replace(
            cfg, flow_b_premium_discount=True, flow_b_range_mode="15m_swing"
        )

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)
    print(f"Config B={cfg.flow_b_premium_discount}  period={start.date()}→{end.date()}")

    ad = DukascopyAdapter()
    df_4h = await ad.fetch_history(
        "EUR/USD", "4h", start=datetime(2023, 7, 1, tzinfo=timezone.utc), end=end
    )
    df_15m = await ad.fetch_history("EUR/USD", "15m", start=start, end=end)
    df_1m = await ad.fetch_history("EUR/USD", "1m", start=start, end=end)
    print(f"Loaded 4h={len(df_4h)} 15m={len(df_15m)} 1m={len(df_1m)}")

    r = run_backtest(df_4h, df_15m, df_1m, cfg, verbose=False)
    trades = [t for t in r["trades"] if t.get("status") == "closed"]
    print(f"Closed trades: {len(trades)}")

    # Overall outcome mix
    outs = [_outcome(t) for t in trades]
    from collections import Counter

    print("\n=== Outcome mix ===")
    for k, v in Counter(outs).most_common():
        print(f"  {k:<18} {v:4d}  ({100*v/len(trades):.1f}%)")

    for key in (
        "direction",
        "session",
        "liq_src",
        "liq_family",
        "sweep_kind",
        "pd_side",
        "weekday",
        "hour_ny",
    ):
        _slice_table(trades, key)

    # Sweep distance to entry in R buckets
    print("\n--- by sweep_to_entry_r (buckets) ---")
    buckets = {"0-0.5R": [], "0.5-1R": [], "1-2R": [], "2R+": [], "na": []}
    for t in trades:
        meta = t.get("meta") or {}
        d = meta.get("sweep_to_entry_r") if isinstance(meta, dict) else None
        if d is None:
            buckets["na"].append(t)
        elif d < 0.5:
            buckets["0-0.5R"].append(t)
        elif d < 1.0:
            buckets["0.5-1R"].append(t)
        elif d < 2.0:
            buckets["1-2R"].append(t)
        else:
            buckets["2R+"].append(t)
    # reuse printer via fake tag
    fake = []
    for b, ts in buckets.items():
        for t in ts:
            tt = dict(t)
            meta = dict(tt.get("meta") or {})
            meta["sweep_r_bucket"] = b
            tt["meta"] = meta
            fake.append(tt)
    _slice_table(fake, "sweep_r_bucket")

    print("\nDONE")


if __name__ == "__main__":
    asyncio.run(main())
