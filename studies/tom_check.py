#!/usr/bin/env python3
"""S-008 Turn of the Month: reference check and funnel (portfolio of 6 indices, literature track).

Spec: ``docs/specs/turn_of_month.yaml``. Data, costs and rates as in S-007
(``studies/tsmom_check.py --fetch`` once).

  .venv/bin/python studies/tom_check.py --referenz
  .venv/bin/python studies/tom_check.py --trichter [--holdout 2026-10-29]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from studies.tsmom_check import TRAIN_START, load_all  # noqa: E402
from trading_lab import screening  # noqa: E402
from trading_lab.registry import HOLDOUT_START  # noqa: E402
from trading_lab.strategies import turn_of_month as TOM  # noqa: E402
from trading_lab.strategies.tsmom_multi_asset import daily_returns  # noqa: E402

OUT = Path("artifacts/tom")
PLACEBO_RUNS = 100


def summary(res: dict, start: str) -> dict:
    d = res["daily"].loc[start:]
    tr = [t for t in res["trades"] if t["entry_time"] >= pd.Timestamp(start)]
    yrs = d["net"].groupby(d.index.year).sum()
    mon = d["net"].groupby(d.index.to_period("M")).sum()
    return {"n": len(tr), "gross_mean": float(np.mean([t["r_gross"] for t in tr])),
            "net_mean": float(np.mean([t["r"] for t in tr])),
            "pos_years": int((yrs > 0).sum()), "n_years": int(len(yrs)),
            "per_year_pct": {int(k): round(100 * v, 2) for k, v in yrs.items()},
            "t_daily": float(screening.t_stat(d["net"] / res["unit"])),
            "sharpe_net": float(d["net"].mean() / d["net"].std() * np.sqrt(252)),
            "sharpe_gross": float(d["gross"].mean() / d["gross"].std() * np.sqrt(252)),
            "pos_months": float((mon[mon != 0] > 0).mean()),
            "holding_days": float(np.mean([(t["exit_time"] - t["entry_time"]).days for t in tr])),
            "pct_pa": float(100 * d["net"].sum() / max(len(yrs), 1))}


def window_vs_rest(p, start: str) -> dict:
    """Reference (McConnell/Xu): mean daily return inside the -1..+3 window vs the other days."""
    ret = daily_returns(p.close)
    out = {}
    for i, a in enumerate(p.assets):
        if a not in TOM.INDICES:
            continue
        inside = np.zeros(len(p.dates), bool)
        for e, x in TOM.tom_windows(p.dates, p.has_bar[:, i]):
            inside[e + 1:x + 1] = True
        ok = p.has_bar[:, i] & (p.dates >= pd.Timestamp(start))
        out[a] = (float(ret[ok & inside, i].mean() * 1e4), float(ret[ok & ~inside, i].mean() * 1e4))
    return out


def _placebo(args) -> float:
    p, seed = args
    rng = np.random.default_rng(seed)
    w = {a: TOM.random_windows(p.dates, p.has_bar[:, i], rng) for i, a in enumerate(p.assets) if a in TOM.INDICES}
    tr = [t for t in TOM.run_backtest(p, windows=w)["trades"] if t["entry_time"] >= pd.Timestamp(TRAIN_START)]
    return float(np.mean([t["r_gross"] for t in tr]))


def referenz() -> dict:
    p, _ = load_all(HOLDOUT_START)
    s = summary(TOM.run_backtest(p), TRAIN_START)
    wr = window_vs_rest(p, TRAIN_START)
    print("Referenz: Ø Tagesrendite im Fenster vs. übrige Tage (bp), 2004-2024")
    for a, (i, o) in wr.items():
        print(f"  {a:6s} Fenster {i:+6.2f}  Rest {o:+6.2f}  {'✓' if i > o and i > 0 else '✗'}")
    print(f"\nBasis: {s['n']} Positionen, Ø brutto {s['gross_mean']:+.3f} R, netto {s['net_mean']:+.3f} R, "
          f"Sharpe vor Kosten {s['sharpe_gross']:.2f} / netto {s['sharpe_net']:.2f}, t {s['t_daily']:.2f}, "
          f"Gewinnmonate {s['pos_months']:.0%}, Jahre {s['pos_years']}/{s['n_years']}, ~{s['pct_pa']:.2f} % p.a.")
    print("Jahre (netto %):", s["per_year_pct"])
    s["window_vs_rest"] = wr
    return s


def trichter(holdout: str | None) -> int:
    s = referenz()
    p, _ = load_all(HOLDOUT_START)
    with ProcessPoolExecutor(4) as ex:
        pg = float(np.mean(list(ex.map(_placebo, [(p, k) for k in range(PLACEBO_RUNS)]))))
    wr = s["window_vs_rest"]
    good = sum(i > o and i > 0 for i, o in wr.values())
    row = {"name": "turn_of_month:INDIZES#0", "strategy": "turn_of_month", "asset": "INDIZES", "params": {},
           "n": s["n"], "gross_mean": s["gross_mean"], "net_mean": s["net_mean"],
           "placebo_gross": pg, "placebo_edge": s["gross_mean"] - pg,
           "pos_years": s["pos_years"], "n_years": s["n_years"], "neighbour_positive": None,
           "t_daily": s["t_daily"], "n_trials": 1, "dsr": np.nan, "pbo": np.nan,
           "track": "literatur", "paper_base": True, "reference_ok": good >= 4,
           "holding_days": s["holding_days"], "sharpe_net": s["sharpe_net"]}
    print(f"\nPlacebo (Monatsmitte, 4 Tage): Ø brutto {pg:+.3f} R -> Vorsprung {row['placebo_edge']:+.3f} R; "
          f"Referenz {good}/6 Indizes; Haltedauer {s['holding_days']:.1f} Kalendertage")
    if holdout:
        full, _ = load_all(None)
        res = TOM.run_backtest(full)
        d = res["daily"].loc[HOLDOUT_START.tz_localize(None):]
        tr = [t for t in res["trades"] if t["entry_time"] >= HOLDOUT_START.tz_localize(None)]
        row.update({"oos_n": len(tr), "oos_net_pct": float(100 * d["net"].sum()),
                    "oos_mean_r": float(np.mean([t["r"] for t in tr])) if tr else None,
                    "oos_t_daily": float(screening.t_stat(d["net"] / res["unit"])),
                    "oos_pass": bool(d["net"].sum() > 0)})
        print(f"Holdout: {len(tr)} Positionen, netto {row['oos_net_pct']:+.2f} %, Ø {row['oos_mean_r']:+.3f} R, "
              f"t {row['oos_t_daily']:.2f}")
    f = screening.run_funnel(pd.DataFrame([row]), up_to="S5" if holdout else "S4", holdout_freigabe=holdout)
    print("\n" + f.summary())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("funnel_holdout.json" if holdout else "funnel.json")).write_text(json.dumps(row, indent=1, default=str))
    return 0


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--referenz", action="store_true")
    ap.add_argument("--trichter", action="store_true")
    ap.add_argument("--holdout", metavar="FREIGABE")
    a = ap.parse_args()
    if a.trichter:
        return trichter(a.holdout)
    if a.referenz:
        referenz()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
