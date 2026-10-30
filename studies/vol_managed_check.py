#!/usr/bin/env python3
"""S-010 volatility-managed equity premium: reference, placebo, funnel (literature track).

Spec: ``docs/specs/vol_managed_equity.yaml``. Data and costs as in S-007; swaps measured on FundedNext.

  .venv/bin/python studies/vol_managed_check.py [--holdout 2026-10-30]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import studies.tsmom_check as T  # noqa: E402
from trading_lab import screening  # noqa: E402
from trading_lab.registry import HOLDOUT_START  # noqa: E402
from trading_lab.strategies import tsmom_multi_asset as M  # noqa: E402
from trading_lab.strategies import vol_managed_equity as V  # noqa: E402

OUT = Path("artifacts/vol_managed")
PLACEBO_RUNS = 100
SWAPS = json.loads(Path("artifacts/broker_costs/fundednext_demo.json").read_text())["swaps_pa"]
LONG = {a: SWAPS[a][0] for a in V.INDICES if a in SWAPS}
START = "2006-01-01"                       # 2004 + 24 months warm-up


def sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std() * np.sqrt(252))


def summary(res: dict, start: str = START) -> dict:
    d = res["daily"].loc[start:]
    tr = [t for t in res["trades"] if t["entry_time"] >= pd.Timestamp(start)]
    yrs = d["net"].groupby(d.index.year).sum()
    e = d["net"].cumsum()
    return {"n": len(tr), "gross_mean": float(np.mean([t["r_gross"] for t in tr])),
            "net_mean": float(np.mean([t["r"] for t in tr])), "pos_years": int((yrs > 0).sum()),
            "n_years": int(len(yrs)), "per_year_pct": {int(k): round(100 * v, 1) for k, v in yrs.items()},
            "t_daily": float(screening.t_stat(d["net"] / res["unit"])), "sharpe_net": sharpe(d["net"]),
            "sharpe_gross": sharpe(d["gross"]), "vol": float(d["net"].std() * np.sqrt(252)),
            "max_dd": float((e - e.cummax()).min()), "pct_pa": float(100 * d["net"].sum() / len(yrs)),
            "holding_days": float(np.mean([(t["exit_time"] - t["entry_time"]).days for t in tr]))}


def placebo(p: M.Prepared, cfg: V.VolManagedConfig) -> float:
    months = pd.period_range(p.dates[0], p.dates[-1], freq="M")
    vals = []
    for s in range(PLACEBO_RUNS):
        flips = pd.Series(np.random.default_rng(s).choice([-1.0, 1.0], len(months)), index=months)
        r = V.run_backtest(p, cfg, LONG, flips=flips)
        vals.append(np.mean([t["r_gross"] for t in r["trades"] if t["entry_time"] >= pd.Timestamp(START)]))
    return float(np.mean(vals))


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", metavar="FREIGABE")
    a = ap.parse_args()
    p, _ = T.load_all(HOLDOUT_START)
    base = V.VolManagedConfig()
    runs = {g: V.run_backtest(p, replace(base, gewicht=g), LONG) for g in ("varianz", "vola", "konstant")}
    sums = {g: summary(r) for g, r in runs.items()}
    for g, s in sums.items():
        print(f"{g:9s} Sharpe brutto {s['sharpe_gross']:.2f}, netto {s['sharpe_net']:.2f}, t {s['t_daily']:.2f}, "
              f"{s['pct_pa']:+.1f} % p.a., Vola {100 * s['vol']:.1f} %, max DD {100 * s['max_dd']:.1f} %, "
              f"Jahre {s['pos_years']}/{s['n_years']}, {s['n']} Positionen, Ø {s['net_mean']:+.3f} R")
    print("Jahre varianz:", sums["varianz"]["per_year_pct"])
    print("Jahre konstant:", sums["konstant"]["per_year_pct"])
    formula = {a_: -(0.03 + 0.025) for a_ in V.INDICES}
    print(f"Robustheit Formel-Swaps (~-5,5 % p.a.): Sharpe netto "
          f"{sharpe(V.run_backtest(p, base, formula)['daily']['net'].loc[START:]):.2f}")
    s7 = M.run_backtest(p, M.TsmomConfig(swaps_pa=tuple((k, lo, sh) for k, (lo, sh) in SWAPS.items())))["daily"]["net"]
    d = runs["varianz"]["daily"]["net"]
    j = pd.concat([s7.rename("s7"), d.rename("v")], axis=1).loc[START:].dropna()
    z = j / j.std()
    print(f"Korrelation mit S-007 {j.corr().iloc[0, 1]:+.2f}, Korb gleiches Risiko Sharpe {sharpe((z.s7 + z.v) / 2):.2f} "
          f"(S-007 allein {sharpe(z.s7):.2f}, Zeitraum ab {START[:4]})")
    ref_ok = sums["varianz"]["sharpe_net"] >= sums["konstant"]["sharpe_net"]
    print(f"Referenz (Steuerung >= konstante Position): {ref_ok}")
    rows = []
    for g in ("varianz", "vola"):
        s = sums[g]
        pg = placebo(p, replace(base, gewicht=g))
        other = sums["vola" if g == "varianz" else "varianz"]
        rows.append({"name": f"vol_managed_equity:KORB#{len(rows)}", "strategy": "vol_managed_equity", "asset": "KORB",
                     "params": {"gewicht": g}, "n": s["n"], "gross_mean": s["gross_mean"], "net_mean": s["net_mean"],
                     "placebo_gross": pg, "placebo_edge": s["gross_mean"] - pg, "pos_years": s["pos_years"],
                     "n_years": s["n_years"], "neighbour_positive": float(other["net_mean"] > 0),
                     "t_daily": s["t_daily"], "n_trials": 2,
                     "dsr": screening.deflated_sharpe((runs[g]["daily"]["net"].loc[START:] / runs[g]["unit"]).to_numpy(), n_trials=2),
                     "pbo": np.nan, "track": "literatur", "paper_base": g == "varianz", "reference_ok": bool(ref_ok),
                     "holding_days": s["holding_days"], "sharpe_net": s["sharpe_net"]})
        print(f"{g}: Placebo {pg:+.3f} R, Vorsprung {s['gross_mean'] - pg:+.3f} R, DSR {rows[-1]['dsr']:.3f}")
    if a.holdout:
        full, _ = T.load_all(None)
        for row in rows:
            r = V.run_backtest(full, replace(base, **row["params"]), LONG)
            d2 = r["daily"].loc[HOLDOUT_START.tz_localize(None):]
            row.update({"oos_net_pct": float(100 * d2["net"].sum()), "oos_sharpe": sharpe(d2["net"]),
                        "oos_t_daily": float(screening.t_stat(d2["net"] / r["unit"])), "oos_pass": bool(d2["net"].sum() > 0)})
            print(f"Holdout {row['params']}: {row['oos_net_pct']:+.2f} %, Sharpe {row['oos_sharpe']:.2f}, t {row['oos_t_daily']:.2f}")
    f = screening.run_funnel(pd.DataFrame(rows), up_to="S5" if a.holdout else "S4", holdout_freigabe=a.holdout)
    print("\n" + f.summary())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("funnel_holdout.json" if a.holdout else "funnel.json")).write_text(json.dumps(rows, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
