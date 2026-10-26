#!/usr/bin/env python3
"""Run the exit attribution over the repo's strategies on real data.

Answers "is our reward/risk actually doing anything?" by grouping each
strategy's trades by ``exit_reason``: how often the target fires, how much R
each exit group contributes, and how far the trades that never reached the
target actually got. See ``trading_lab/exit_attribution.py`` for the logic.

  .venv/bin/python studies/exit_attribution.py --strategy orb_2r --asset US100
  .venv/bin/python studies/exit_attribution.py --strategy all
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable

sys.path.insert(0, "."); sys.path.insert(0, "studies")

from trading_lab.exit_attribution import attribution, report  # noqa: E402
from trading_lab.minute_data import load_minute_bid_ask  # noqa: E402
from trading_lab.strategies.orb_2r import MARKETS, build_sessions, run_backtest  # noqa: E402


def load_orb_2r(asset: str, start_year: int, *, costs: float = 1.0) -> list[dict]:
    """The 15-minute opening-range breakout with its fixed R target."""
    cfg = MARKETS[asset]
    df = load_minute_bid_ask(asset, start_year, verbose=False)
    if df.empty:
        return []
    sessions = build_sessions(df, cfg, costs)
    return run_backtest(sessions, cfg)["trades"]


Loaders = dict[str, Callable[..., list[dict]]]
STRATEGIES: Loaders = {"orb_2r": load_orb_2r}
MARKET_CHOICES = sorted(MARKETS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default="orb_2r",
                    choices=sorted(STRATEGIES) + ["all"])
    ap.add_argument("--asset", default="US100", choices=MARKET_CHOICES + ["all"])
    ap.add_argument("--start-year", type=int, default=2016)
    ap.add_argument("--costs", type=float, default=1.0,
                    help="cost multiplier: 0 = gross, 2 = doubled costs")
    args = ap.parse_args()

    strategies = sorted(STRATEGIES) if args.strategy == "all" else [args.strategy]
    assets = MARKET_CHOICES if args.asset == "all" else [args.asset]
    for name in strategies:
        for asset in assets:
            try:
                trades = STRATEGIES[name](asset, args.start_year, costs=args.costs)
            except Exception as exc:                      # noqa: BLE001 - report, keep going
                print(f"### {name} {asset}: fehlgeschlagen ({type(exc).__name__}: {exc})\n")
                continue
            if not trades:
                print(f"### {name} {asset}: keine Trades\n")
                continue
            label = f"{name} {asset}" + ("" if args.costs == 1.0 else f" (Kosten ×{args.costs:g})")
            print(report(trades, label))
            a = attribution(trades)
            tgt = a.loc[a["is_target"], "share"].sum()
            tim = a.loc[a["is_time"], "share"].sum()
            print(f"    Ziel-Anteil {tgt*100:.0f} %   Zeit-Exit-Anteil {tim*100:.0f} %\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
