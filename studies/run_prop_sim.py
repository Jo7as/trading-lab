#!/usr/bin/env python3
"""Apply the prop-firm simulator to a strategy (reusable for any trade log).

Runs a named strategy over a period, converts its trades to a daily P&L stream
at a fixed dollar risk, and evaluates an FTMO or Lucid-style account with the
Monte-Carlo pass rate. To use it for a new strategy, add a loader to
``STRATEGIES`` that returns ``(trades, trading_days)`` — everything else is
strategy-agnostic.

  .venv/bin/python studies/run_prop_sim.py --firm ftmo --asset US100 --risk-pct 0.5
  .venv/bin/python studies/run_prop_sim.py --firm lucid --asset SP500 --runs 10000
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from trading_lab.minute_data import load_minute_bid_ask
from trading_lab.prop_sim import PropFirm, daily_pnl, evaluate, get_firm, risk_usd_for
from trading_lab.strategies.orb_2r import skipped_days
from trading_lab.strategies.orb_rejection_ifvg import OrbRejConfig, run_backtest

TradeLoader = Callable[[str, int, int], tuple[list[dict], list[date]]]


def _trading_days(start_year: int, end_year: int) -> list[date]:
    skip: set[date] = set()
    for y in range(start_year, end_year + 1):
        skip |= skipped_days("nyse", y)
    days = pd.bdate_range(f"{start_year}-01-01", f"{end_year}-12-31")
    return [d.date() for d in days if d.weekday() < 5 and d.date() not in skip]


def load_orb_fade(asset: str, start_year: int, end_year: int) -> tuple[list[dict], list[date]]:
    """The validated naive edge-fade (see docs / the ablation report)."""
    df = load_minute_bid_ask(asset, start_year, verbose=False)
    years = df.index.tz_convert("UTC").year
    sub = df[(years >= start_year) & (years <= end_year)]
    res = run_backtest(sub, OrbRejConfig(asset=asset, asset_label=asset, mode="naive"))
    return res["trades"], _trading_days(start_year, end_year)


STRATEGIES: dict[str, TradeLoader] = {"orb_fade": load_orb_fade}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default="orb_fade", choices=sorted(STRATEGIES))
    ap.add_argument("--asset", default="US100")
    ap.add_argument("--firm", default="ftmo", choices=["ftmo", "lucid"])
    ap.add_argument("--start-year", type=int, default=2024)
    ap.add_argument("--end-year", type=int, default=datetime.now(timezone.utc).year)
    ap.add_argument("--risk-pct", type=float, default=0.5, help="risk per trade, % of account")
    ap.add_argument("--runs", type=int, default=5000)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    firm: PropFirm = get_firm(args.firm)
    trades, days = STRATEGIES[args.strategy](args.asset, args.start_year, args.end_year)
    if not trades:
        print("no trades")
        return 1
    risk_usd = risk_usd_for(firm.account_size, args.risk_pct)
    stream = daily_pnl(trades, risk_usd=risk_usd, days=days)
    worst_r = min(float(t["r"]) for t in trades)

    print(f"{args.strategy} / {args.asset}  {args.start_year}–{args.end_year}")
    print(f"  firm: {firm.name}")
    print(f"  risk ${risk_usd:,.0f}/trade ({args.risk_pct}% of ${firm.account_size:,.0f}) | "
          f"daily limit ${firm.daily_limit_usd:,.0f} | total limit ${firm.total_limit_usd:,.0f}")
    print(f"  trades {len(trades)} | worst single trade ${worst_r * risk_usd:,.0f} "
          f"({'OK' if abs(worst_r) * risk_usd < firm.daily_limit_usd else 'BREACH'} vs daily limit)")
    print(f"  days {len(days)} | worst day ${stream.min():,.0f} | best day ${stream.max():,.0f} "
          f"| trading days used {int((stream != 0).sum())}")

    res = evaluate(stream, firm, runs=args.runs, block=args.block)
    print("  " + res.summary())

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "strategy": args.strategy, "asset": args.asset, "firm": firm.name,
            "risk_pct": args.risk_pct, "risk_usd": risk_usd,
            "worst_single_trade_usd": round(worst_r * risk_usd, 2),
            "worst_day_usd": round(float(stream.min()), 2),
            "pass_rate": res.pass_rate, "phase_pass_rate": res.phase_pass_rate,
            "median_days": res.median_days, "p95_days": res.p95_days,
            "fails": res.fails, "cost_per_funded": res.cost_per_funded,
        }, indent=2))
        print(f"  json: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
