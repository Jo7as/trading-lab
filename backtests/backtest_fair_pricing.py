#!/usr/bin/env python3
"""Backtest for the fair-pricing (JJ Simon) strategy on Dukascopy 1m bid+ask.

Runs the strategy's own engine through the shared pipeline: metrics from
``trading_lab.backtest``, the standard acceptance path, and a report under
``artifacts/``. News setups (C, and the 08:30 fair-price rule) need a local
economic calendar — pass ``--calendar`` (see ``trading_lab/calendar_events.py``
for the CSV/JSON schema); without it they stay inactive.

  uv run python backtests/backtest_fair_pricing.py --asset US100
  uv run python backtests/backtest_fair_pricing.py --asset US100 \\
      --calendar configs/calendar/us_red_folder.example.csv
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_lab.acceptance import run_acceptance
from trading_lab.backtest import print_metrics
from trading_lab.calendar_events import load_events
from trading_lab.minute_data import load_minute_bid_ask
from trading_lab.strategies.fair_pricing import FairPricingConfig, run_backtest

PERIODS: dict[str, tuple[int, int]] = {
    "dev_2016_2021": (2016, 2021),
    "test_2022_now": (2022, datetime.now(timezone.utc).year),
}


def _slice(df, a: int, b: int):
    years = df.index.tz_convert("UTC").year
    return df[(years >= a) & (years <= b)]


def _setup_counts(trades: list[dict]) -> dict[str, int]:
    return dict(Counter(t.get("meta", {}).get("setup", "?") for t in trades))


def _setup_stats(trades: list[dict]) -> dict[str, dict[str, float]]:
    """Per-setup trade count, win rate and average/summed R for interpretability."""
    out: dict[str, dict[str, float]] = {}
    for t in trades:
        setup = t.get("meta", {}).get("setup", "?")
        r = float(t.get("meta", {}).get("r") or 0.0)
        s = out.setdefault(setup, {"trades": 0.0, "wins": 0.0, "sum_r": 0.0})
        s["trades"] += 1
        s["wins"] += 1 if r > 0 else 0
        s["sum_r"] += r
    for s in out.values():
        n = s["trades"]
        s["win_rate"] = round(s["wins"] / n * 100, 1) if n else 0.0
        s["sum_r"] = round(s["sum_r"], 1)
        s["avg_r"] = round(s["sum_r"] / n, 3) if n else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", default="US100")
    ap.add_argument("--calendar", default=None, help="CSV/JSON economic calendar")
    ap.add_argument("--start-year", type=int, default=2016)
    ap.add_argument("--slippage", type=float, default=1.0)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    events = load_events(args.calendar) if args.calendar else None
    if args.calendar and not events:
        print(f"calendar {args.calendar} is empty — news setups inactive")

    cfg = FairPricingConfig(asset=args.asset, asset_label=args.asset, slippage=args.slippage)
    df = load_minute_bid_ask(args.asset, args.start_year, verbose=args.verbose)
    if df.empty:
        print(f"no data for {args.asset}")
        return 1

    report: dict[str, Any] = {
        "asset": args.asset,
        "calendar": args.calendar,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "periods": {},
    }
    for name, (a, b) in PERIODS.items():
        sub = _slice(df, a, b)
        if sub.empty:
            continue
        res = run_backtest(sub, cfg, events=events, verbose=args.verbose)
        trades = res["trades"]
        acc = run_acceptance(trades)
        print_metrics(res["metrics"], label=f"{args.asset} {name}")
        stats = _setup_stats(trades)
        for setup, s in sorted(stats.items()):
            print(f"  setup {setup}: {int(s['trades'])} trades | WR {s['win_rate']}% | "
                  f"avg {s['avg_r']:+.3f}R | sum {s['sum_r']:+.1f}R")
        print(acc.summary())
        report["periods"][name] = {
            "years": [a, b],
            "metrics": res["metrics"],
            "setups": _setup_counts(trades),
            "setup_stats": stats,
            "acceptance": {"passed": acc.passed, "significance": acc.significance.as_dict()},
        }

    out = args.output or Path("artifacts") / f"fair_pricing_{args.asset}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
