#!/usr/bin/env python3
"""Validation backtest for the ORB rejection fade (frozen v0) + baselines.

Runs the ablation (naive → touches → engulf → full) and the two mandatory
baselines on US100 and SP500 (Dukascopy CFDs, NQ/ES proxies), 1-minute RTH:

    breakout → classic ORB breakout (first close beyond an edge, stop at mid, 2R)
    random   → same touch bars, same stop/target distances, random direction

IS 2019–2023, OOS 2024–now. No parameter search. Report under ``artifacts/``.

  .venv/bin/python backtests/backtest_orb_rejection_ifvg.py --runs 20
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_lab.acceptance import run_acceptance
from trading_lab.minute_data import load_minute_bid_ask
from trading_lab.significance import percentile_of
from trading_lab.strategies.orb_rejection_ifvg import OrbRejConfig, prepare, run_backtest

PERIODS: dict[str, tuple[int, int]] = {
    "is_2019_2023": (2019, 2023),
    "oos_2024_now": (2024, datetime.now(timezone.utc).year),
}
FADE_MODES = ("naive", "touches", "engulf", "full")


def _slice(df: pd.DataFrame, a: int, b: int) -> pd.DataFrame:
    years = df.index.tz_convert("UTC").year
    return df[(years >= a) & (years <= b)]


def _stats(trades: list[dict]) -> dict[str, Any]:
    rs = [float(t["r"]) for t in trades]
    n = len(rs)
    return {
        "trades": n,
        "total_r": round(sum(rs), 1),
        "expectancy_r": round(sum(rs) / n, 3) if n else 0.0,
        "win_rate": round(100 * sum(1 for r in rs if r > 0) / n, 1) if n else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", default="US100,SP500")
    ap.add_argument("--start-year", type=int, default=2016)
    ap.add_argument("--runs", type=int, default=20, help="random-direction baseline runs")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    report: dict[str, Any] = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "assets": {}}
    for asset in [a.strip() for a in args.assets.split(",") if a.strip()]:
        cfg = OrbRejConfig(asset=asset, asset_label=asset)
        df = load_minute_bid_ask(asset, args.start_year, verbose=False)
        if df.empty:
            print(f"== {asset}: no data (skipped)")
            continue
        print(f"\n=================== {asset} ===================")
        asset_rep: dict[str, Any] = {"config": asdict(cfg), "periods": {}}
        for pname, (a, b) in PERIODS.items():
            sub = _slice(df, a, b)
            if sub.empty:
                continue
            period: dict[str, Any] = {}
            print(f"-- {asset} {pname}")
            prep = prepare(sub, cfg)
            for mode in FADE_MODES + ("breakout",):
                res = run_backtest(sub, replace(cfg, mode=mode), prepared=prep)
                s = _stats(res["trades"])
                period[mode] = s
                print(f"   {mode:8s} n={s['trades']:4d} WR={s['win_rate']:5.1f}% "
                      f"exp={s['expectancy_r']:+.3f}R totalR={s['total_r']:+.0f}")
                if mode == "naive":
                    acc = run_acceptance(res["trades"])
                    period["naive_acceptance"] = {"passed": acc.passed,
                                                  "significance": acc.significance.as_dict()}
                    print(f"            acceptance: {'PASS' if acc.passed else 'FAIL'} "
                          f"({acc.significance.summary()})")
                    naive_total = s["total_r"]
            sums = [_stats(run_backtest(sub, replace(cfg, mode="random", random_seed=sd),
                                        prepared=prep)["trades"])["total_r"]
                    for sd in range(args.runs)]
            period["random"] = {"mean_r": round(float(np.mean(sums)), 1),
                                "p95_r": round(float(np.percentile(sums, 95)), 1),
                                "runs": args.runs}
            period["naive_pct_vs_random"] = round(percentile_of(naive_total, sums), 1)
            print(f"   random   mean={period['random']['mean_r']:+.0f}R p95={period['random']['p95_r']:+.0f}R "
                  f"| naive percentile={period['naive_pct_vs_random']}")
            asset_rep["periods"][pname] = period
        report["assets"][asset] = asset_rep

    out = args.output or Path("artifacts") / "orb_rejection_ifvg_validation"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
