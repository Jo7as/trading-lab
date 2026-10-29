#!/usr/bin/env python3
"""Quick check (no strategy): cross-sectional momentum across S-007's 26 markets.

Asness/Moskowitz/Pedersen (2013): inside each asset class, long the markets with the best
12-1 month return, short the worst (rank weights, demeaned within the class), vol-scaled,
monthly rebalance at month end. Classes with < 3 markets are ranked together with
"rohstoffe/metalle/anleihen/krypto" as one "real assets" class. Price returns, minus the
measured one-way costs on the monthly turnover; swaps left out (long-short books offset
most of them). Train 2004-2024; correlation with S-007 and the basket of both.

  .venv/bin/python studies/xsmom_check.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import studies.tsmom_check as T  # noqa: E402
from trading_lab import screening  # noqa: E402
from trading_lab.registry import HOLDOUT_START  # noqa: E402
from trading_lab.strategies import tsmom_multi_asset as M  # noqa: E402

CLASS = {"aktienindizes": "aktien", "fx_usd": "fx_usd", "fx_kreuz": "fx_kreuz", "metalle": "real",
         "rohstoffe": "real", "anleihen": "real", "krypto": "real"}


def main() -> int:
    warnings.filterwarnings("ignore")
    T.BROKER.update(json.loads(Path("artifacts/broker_costs/fundednext_demo.json").read_text()))
    p, _ = T.load_all(HOLDOUT_START)
    close = pd.DataFrame(p.close, index=p.dates, columns=p.assets)
    me = close.resample("ME").last()
    mret = me.pct_change()
    mom = me.shift(1) / me.shift(12) - 1                                 # 12-1 months, known at month end
    vol = close.pct_change().rolling(126, min_periods=60).std().resample("ME").last() * np.sqrt(252)
    cls = pd.Series({a: CLASS[M.GROUP_OF[a]] for a in p.assets})
    w = pd.DataFrame(0.0, index=me.index, columns=p.assets)
    for c in cls.unique():
        cols = cls[cls == c].index
        rk = mom[cols].rank(axis=1)
        dm = rk.sub(rk.mean(axis=1), axis=0)
        w[cols] = (dm / vol[cols]).fillna(0)
    # scale each month's book to ~8 % vol (uncorrelated proxy) and charge the turnover
    scale = 0.08 / np.sqrt(((w * vol.fillna(0)) ** 2).sum(axis=1)).replace(0, np.nan)
    w = w.mul(scale, axis=0).fillna(0)
    cost_bp = pd.Series(p.cost_bp, index=p.assets)
    gross = (w.shift(1) * mret).sum(axis=1)
    cost = (w.diff().abs() * cost_bp / 1e4).sum(axis=1)
    net = (gross - cost).loc[T.TRAIN_START:"2024-12-31"]
    s7 = M.run_backtest(p, T._cfg({"signal": "12m", "stop": "keiner"}))["daily"]["net"]
    s7m = s7.resample("ME").sum().loc[net.index.min():net.index.max()]
    sh = lambda x: float(x.mean() / x.std() * np.sqrt(12))  # noqa: E731
    yrs = net.groupby(net.index.year).sum()
    print("Querschnitts-Momentum (12-1 Monate, je Klasse), 26 Märkte, 2004-2024")
    print(f"  Sharpe brutto {sh(gross.loc[net.index])}, netto {sh(net):.2f} (t {screening.t_stat(net):.2f}), "
          f"Jahre {(yrs > 0).sum()}/{len(yrs)}, Kosten {100 * cost.loc[net.index].sum() / len(yrs):.2f} % p.a.")
    for c in cls.unique():
        cols = cls[cls == c].index
        g = (w[cols].shift(1) * mret[cols]).sum(axis=1).loc[net.index]
        print(f"     {c:9s} Sharpe {sh(g):+.2f}")
    j = pd.concat([net.rename("xs"), s7m.rename("s7")], axis=1).dropna()
    print(f"  Korrelation mit S-007 (monatlich): {j.corr().iloc[0, 1]:+.2f}")
    z = j / j.std()
    for wx in (0.25, 0.5):
        b = (1 - wx) * z["s7"] + wx * z["xs"]
        print(f"  Korb {100 * (1 - wx):.0f} % S-007 + {100 * wx:.0f} % Querschnitt (Risiko): Sharpe {sh(b):.2f} "
              f"vs. S-007 allein {sh(z['s7']):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
