#!/usr/bin/env python3
"""Measurement (no spec yet): does S-007 get better with the further markets FundedNext offers?

Same rules as S-007 (12m signal, no stop, 8 % vol, equal risk per asset class), same measured
FundedNext swaps where we have them, the formula otherwise. Train 2004-2024 only. The extra
markets have no 1m cache - their one-way cost is ASSUMED conservatively (indices 2 bp, FX
crosses 2 bp, SEK/NOK 3 bp, platinum/palladium 10 bp).

  .venv/bin/python studies/tsmom_universe_check.py [--fetch]
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from studies.tsmom_check import TRAIN_START, aqr_monthly  # noqa: E402
from trading_lab import long_daily, rates, screening  # noqa: E402
from trading_lab.registry import HOLDOUT_START  # noqa: E402
from trading_lab.strategies import tsmom_multi_asset as M  # noqa: E402

ASSUMED_BP = {"AUS200": 2.0, "FRA40": 2.0, "EU50": 2.0, "ESP35": 2.0, "HK50": 2.0, "XPTUSD": 10.0, "XPDUSD": 10.0,
              "USDSEK": 3.0, "USDNOK": 3.0}


def frames(names) -> dict:
    return {a: long_daily.clean_bad_ticks(long_daily.load_daily(a, fetch=True))[0] for a in names}


def run(names) -> dict:
    cost = json.loads(Path("artifacts/tsmom/costs.json").read_text())
    for a in long_daily.EXTRA_TICKERS:
        cost.setdefault(a, ASSUMED_BP.get(a, 2.0))
    r = rates.load_rates(("USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK")).ffill()
    p = M.prepare(frames(names), r, cost, start="2003-01-01", end=str(HOLDOUT_START.date()))
    sw = json.loads(Path("artifacts/broker_costs/fundednext_demo.json").read_text())["swaps_pa"]
    cfg = M.TsmomConfig(swaps_pa=tuple((a, lo, sh) for a, (lo, sh) in sw.items()))
    res = M.run_backtest(p, cfg)
    d = res["daily"].loc[TRAIN_START:]
    mon = d["net"].groupby(d.index.to_period("M")).sum()
    both = pd.concat([mon, aqr_monthly()], axis=1, join="inner").dropna()
    yrs = d["net"].groupby(d.index.year).sum()
    tr = pd.DataFrame(res["trades"])
    return {"n_markets": len(p.assets), "sharpe_gross": d["gross"].mean() / d["gross"].std() * np.sqrt(252),
            "sharpe_net": d["net"].mean() / d["net"].std() * np.sqrt(252), "t": screening.t_stat(d["net"] / res["unit"]),
            "pos_years": int((yrs > 0).sum()), "n_years": len(yrs), "aqr": both.corr().iloc[0, 1],
            "cost_pa": 100 * d["cost"].sum() / len(yrs), "fin_pa": 100 * d["fin"].sum() / len(yrs),
            "net_pa": 100 * d["net"].sum() / len(yrs), "by_group": tr.groupby("group")["r"].sum().round(1).to_dict(),
            "daily": d["net"]}


def main() -> int:
    warnings.filterwarnings("ignore")
    base = run(list(long_daily.TICKERS))
    wide = run(list(long_daily.ALL_TICKERS))
    print("S-007-Regeln, Training 2004-2024, gemessene FundedNext-Swaps wo vorhanden")
    for name, x in (("26 Märkte (S-007)", base), ("erweitert", wide)):
        print(f"  {name:18s} {x['n_markets']} Märkte | Sharpe brutto {x['sharpe_gross']:.2f}, netto {x['sharpe_net']:.2f} "
              f"(t {x['t']:.2f}) | Jahre {x['pos_years']}/{x['n_years']} | AQR {x['aqr']:.2f} | "
              f"netto {x['net_pa']:+.1f} % p.a. (Kosten {x['cost_pa']:.2f}, Swaps {x['fin_pa']:+.2f})")
        print(f"      je Klasse (Summe R): {x['by_group']}")
    print(f"  Korrelation der Tagesrenditen beider Varianten: {base['daily'].corr(wide['daily']):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
