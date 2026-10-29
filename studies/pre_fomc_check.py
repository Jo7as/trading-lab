#!/usr/bin/env python3
"""S-009 pre-FOMC drift: reference, placebo and funnel (portfolio of 4 indices, literature track).

Spec: ``docs/specs/pre_fomc.yaml``. 1m CFD data, train 2019-2024; holdout only with a release.

  .venv/bin/python studies/pre_fomc_check.py [--holdout 2026-10-29]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from studies.satellite_check import pre_fomc as measure  # noqa: E402
from trading_lab import costs, rates, screening  # noqa: E402
from trading_lab.registry import HOLDOUT_START, load_cached  # noqa: E402
from trading_lab.strategies import pre_fomc as P  # noqa: E402

QUOTE = {"GER40": "EUR"}
OUT = Path("artifacts/pre_fomc")
PLACEBO_RUNS = 100


def markets(holdout: bool) -> list[P.Market]:
    r = rates.load_rates(("USD", "EUR")).ffill()
    out = []
    for a in P.INDICES:
        m1 = costs.adjust_frame(load_cached(a), a, "ftmo")
        if not holdout:
            m1 = m1[m1.index < HOLDOUT_START]
        q = r[QUOTE.get(a, "USD")] / 100
        daily = q.reindex(pd.date_range(q.index.min(), "2030-12-31", freq="D")).ffill()
        out.append(P.market(m1, a, -(daily + 0.025)))
    return out


def daily_r(trades: list[dict], start, end) -> pd.Series:
    cal = pd.bdate_range(start, end)
    s = pd.Series(0.0, index=cal)
    for t in trades:
        d = t["exit_time"].tz_convert(None).normalize()
        if d in s.index:
            s[d] += t["r"]
    return s


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", metavar="FREIGABE")
    a = ap.parse_args()
    hs = HOLDOUT_START.tz_localize(None)
    days = [d for d in P.statement_dates() if pd.Timestamp("2019-01-01") <= d < hs]
    mk = markets(False)
    res = P.run_backtest(mk, days)
    tr = res["trades"]
    df = pd.DataFrame(tr)
    print(f"Training 2019-2024: {len(days)} Sitzungen, {len(tr)} Positionen")
    print("Bewegung je Index (Mitte, bp):", df.groupby("asset")["move_bp"].agg(["count", "mean"]).round(1).to_dict("index"))
    ref = {x: measure(x) for x in P.INDICES}
    good = sum(r["fomc_bp"] > max(r["other_bp"], 0) for r in ref.values())
    d = daily_r(tr, "2019-01-01", hs - pd.Timedelta(days=1))
    yrs = df.assign(y=df["exit_time"].dt.year).groupby("y")["r"].sum()
    cand = pd.bdate_range("2019-01-01", hs - pd.Timedelta(days=2))
    pl = []
    for k in range(PLACEBO_RUNS):
        pdays = P.placebo_days(cand, days, np.random.default_rng(k))
        ptr = P.run_backtest(mk, pdays)["trades"]
        pl.append(np.mean([t["r_gross"] for t in ptr]))
    row = {"name": "pre_fomc:INDIZES#0", "strategy": "pre_fomc", "asset": "INDIZES", "params": {},
           "n": len(tr), "gross_mean": float(df["r_gross"].mean()), "net_mean": float(df["r"].mean()),
           "placebo_gross": float(np.mean(pl)), "placebo_edge": float(df["r_gross"].mean() - np.mean(pl)),
           "pos_years": int((yrs > 0).sum()), "n_years": int(len(yrs)), "neighbour_positive": None,
           "t_daily": float(screening.t_stat(d)), "n_trials": 1, "dsr": np.nan, "pbo": np.nan,
           "track": "literatur", "paper_base": True, "reference_ok": good >= 3, "holding_days": 1.0,
           "per_year_r": {int(k): round(float(v), 2) for k, v in yrs.items()}}
    print(f"Ø brutto {row['gross_mean']:+.3f} R, netto {row['net_mean']:+.3f} R je Position | Placebo {row['placebo_gross']:+.3f} R "
          f"-> Vorsprung {row['placebo_edge']:+.3f} R | t {row['t_daily']:.2f} | Jahre {row['pos_years']}/{row['n_years']} "
          f"{row['per_year_r']} | Referenz {good}/4")
    print(f"Summe {df['r'].sum():+.1f} R in 6 Jahren = {df['r'].sum() / 6:+.2f} R/Jahr "
          f"(~{df['r'].sum() / 6 * 0.5:+.2f} % p.a. bei 0,5 %/R)")
    if a.holdout:
        hd = [x for x in P.statement_dates() if hs <= x < pd.Timestamp.now().normalize()]
        htr = P.run_backtest(markets(True), hd)["trades"]
        hr = np.array([t["r"] for t in htr])
        row.update({"oos_n": len(htr), "oos_meetings": len(hd), "oos_mean_r": float(hr.mean()) if len(hr) else None,
                    "oos_sum_r": float(hr.sum()), "oos_pass": bool(len(hr) and hr.sum() > 0)})
        print(f"Holdout: {len(hd)} Sitzungen, {len(htr)} Positionen, Ø {row['oos_mean_r']:+.3f} R, Summe {hr.sum():+.2f} R")
    f = screening.run_funnel(pd.DataFrame([row]), up_to="S5" if a.holdout else "S4", holdout_freigabe=a.holdout)
    print("\n" + f.summary())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("funnel_holdout.json" if a.holdout else "funnel.json")).write_text(json.dumps(row, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
