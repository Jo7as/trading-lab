#!/usr/bin/env python3
"""P1 ablation: killzone + 15m MSS (± displacement) × exit stacks.

Fixed flow base: Stage B + sell-side (current best filter stack).
Exit stacks: baseline OB+RR+BE vs P0 sweep+liquidity (strategy.yaml v04).

Example:
  python studies/research_smc_p1_ablation.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _path
_sys.path.insert(0, str(_path.Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import time
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


def _exit_base(cfg: SmcIctConfig, stack: str) -> SmcIctConfig:
    """Reset P1 gates; pin exit stack."""
    common = dict(
        flow_b_premium_discount=True,
        flow_b_range_mode="15m_swing",
        flow_sell_side_sweeps_only=True,
        flow_a_sweep_close_back=False,
        flow_c_require_15m_mss=False,
        flow_c_displacement=False,
        flow_c_align_fallback=True,
        flow_killzone="off",
        flow_ny_session_only=False,
    )
    if stack == "baseline":
        return replace(
            cfg,
            **common,
            sl_mode="ob",
            tp_mode="rr",
            move_sl_to_be_on_tp1=True,
            max_hold_bars=360,
            tp1_r=1.5,
            tp2_r=3.0,
        )
    # P0 / v04 liquidity exits
    return replace(
        cfg,
        **common,
        sl_mode="sweep",
        tp_mode="liquidity",
        move_sl_to_be_on_tp1=False,
        max_hold_bars=480,
        max_sl_dist=0.0012,
        max_irl_r=3.0,
        max_erl_r=6.0,
    )


def _p1_variants() -> list[tuple[str, dict]]:
    return [
        ("ctrl", {}),
        ("kz_ny", {"flow_killzone": "ny"}),
        ("kz_ny_am", {"flow_killzone": "ny_am"}),
        ("kz_london_ny", {"flow_killzone": "london_ny"}),
        ("mss", {"flow_c_require_15m_mss": True}),
        ("mss_disp", {"flow_c_require_15m_mss": True, "flow_c_displacement": True}),
        (
            "kz_ny_am+mss",
            {"flow_killzone": "ny_am", "flow_c_require_15m_mss": True},
        ),
        (
            "kz_ny+mss",
            {"flow_killzone": "ny", "flow_c_require_15m_mss": True},
        ),
    ]


def _summarize(label: str, r: dict) -> dict:
    m = r["metrics"]
    trades = [t for t in r["trades"] if t.get("status") == "closed"]
    reasons: dict[str, int] = {}
    for t in trades:
        k = str(t.get("exit_reason") or "?")
        reasons[k] = reasons.get(k, 0) + 1
    weeks = 130.0  # ~2024-01 → mid-2026 rough
    n = int(m.get("total_trades") or 0)
    row = {
        "label": label,
        "n": n,
        "wr": m.get("win_rate"),
        "pnl": m.get("total_pnl_pct"),
        "dd": m.get("max_drawdown_pct"),
        "pf": m.get("profit_factor"),
        "sharpe": m.get("sharpe"),
        "per_week": round(n / weeks, 2) if weeks else None,
        "reasons": reasons,
    }
    print(
        f"{label:<28} n={row['n']:4d} WR={row['wr']!s:>6} "
        f"PnL={row['pnl']!s:>6} DD={row['dd']!s:>6} "
        f"PF={row['pf']!s:>5} Sh={row['sharpe']!s:>6} "
        f"/w={row['per_week']}  {reasons}",
        flush=True,
    )
    return row


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("smc-state/strategy.yaml"))
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument(
        "--stacks",
        default="baseline,p0",
        help="Comma list: baseline,p0",
    )
    args = ap.parse_args()

    cfg0 = _load_cfg(args.config)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)

    ad = DukascopyAdapter()
    print(f"Fetching EUR/USD {start.date()}→{end.date()} ...", flush=True)
    t0 = time.time()
    df_4h = await ad.fetch_history(
        "EUR/USD", "4h", start=datetime(2023, 7, 1, tzinfo=timezone.utc), end=end
    )
    df_15m = await ad.fetch_history("EUR/USD", "15m", start=start, end=end)
    df_1m = await ad.fetch_history("EUR/USD", "1m", start=start, end=end)
    print(
        f"Loaded 4h={len(df_4h)} 15m={len(df_15m)} 1m={len(df_1m)} "
        f"in {time.time() - t0:.0f}s",
        flush=True,
    )

    stacks = [s.strip() for s in args.stacks.split(",") if s.strip()]
    rows: list[dict] = []
    print("-" * 100, flush=True)

    for stack in stacks:
        base = _exit_base(cfg0, stack)
        for p1_name, overrides in _p1_variants():
            cfg = replace(base, **overrides)
            label = f"{stack}|{p1_name}"
            t1 = time.time()
            r = run_backtest(df_4h, df_15m, df_1m, cfg, verbose=False)
            print(f"  [{label}] {time.time() - t1:.0f}s", flush=True)
            row = _summarize(label, r)
            row["stack"] = stack
            row["p1"] = p1_name
            rows.append(row)

    ranked = sorted(
        rows,
        key=lambda x: (-(float(x["pnl"] or -999)), -(float(x["pf"] or 0))),
    )
    print("\n=== Ranked by PnL ===", flush=True)
    for i, row in enumerate(ranked[:10], 1):
        print(
            f"{i:2d}. {row['label']:<28} PnL={row['pnl']} PF={row['pf']} "
            f"n={row['n']} WR={row['wr']} Sh={row['sharpe']}",
            flush=True,
        )

    out = Path("artifacts") / "smc_p1_ablation_2024.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "ranked": ranked}, indent=2, default=str))
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
