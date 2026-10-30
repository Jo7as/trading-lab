#!/usr/bin/env python3
"""Quick checks (no strategies): multi-day building blocks and what they add to S-007. TRAIN only.

1. Correlation filter (Hurst/Ooi/Pedersen 2017): S-007 returns by the lagged average absolute pairwise
   correlation of its markets; scaled version (risk x expanding median corr / corr, clipped 0.5-1.5).
2. Vol-managed equity premium (Moreira/Muir 2017): long the 6 indices, equal risk, each sized by
   8 % / 21-day realised vol (cap 3x), weekly, 25 % buffer, measured FundedNext swaps + costs.
3. Index short-term reversal (liquidity provision, Nagel 2012; asymmetry Boyarchenko et al.): 3-day
   return < -2 daily sigmas (x sqrt 3) -> long at that close for 5 days; vs the unconditional 5-day
   return of the same index; net of 2 x cost and financing.

Long daily data 2004-2024, S-007 = base variant with measured swaps.

  .venv/bin/python studies/multiday_scan.py
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

IDX = ("SP500", "US100", "US30", "GER40", "UK100", "JP225")
START = T.TRAIN_START


def sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std() * np.sqrt(252))


def stats(x: pd.Series) -> str:
    y = x.groupby(x.index.year).sum()
    e = x.cumsum()
    return (f"Sharpe {sharpe(x):+.2f}, t {screening.t_stat(x):+.2f}, {100 * x.sum() / len(y):+.1f} % p.a., "
            f"max DD {100 * (e - e.cummax()).min():.1f} %, Jahre {int((y > 0).sum())}/{len(y)}")


def basket(s7: pd.Series, b: pd.Series, name: str) -> None:
    j = pd.concat([s7.rename("s7"), b.rename("b")], axis=1).dropna()
    z = j / j.std()
    print(f"   Korrelation mit S-007 {j.corr().iloc[0, 1]:+.2f} | Korb 50/50 Risiko: Sharpe "
          f"{sharpe((z['s7'] + z['b']) / 2):.2f} (S-007 allein {sharpe(z['s7']):.2f}) - {name}")


def corr_filter(p: M.Prepared, s7: pd.Series) -> None:
    ret = pd.DataFrame(M.daily_returns(p.close), index=p.dates, columns=p.assets)
    me = ret.resample("ME").last().index
    vals = {}
    for d in me:
        w = ret.loc[:d].tail(126)
        if len(w) < 100:
            continue
        c = w.loc[:, w.std() > 0].corr().abs().to_numpy()
        vals[d] = c[np.triu_indices_from(c, 1)].mean()
    c = pd.Series(vals).shift(1).reindex(s7.index, method="ffill")     # known before the month
    j = pd.concat([s7.rename("r"), c.rename("c")], axis=1).dropna()
    q = pd.qcut(j["c"], 5, labels=False)
    print("1) Korrelations-Filter: S-007-Sharpe je Quintil (niedrig -> hoch):",
          [round(sharpe(j["r"][q == i]), 2) for i in range(5)])
    w = (j["c"].expanding(252).median() / j["c"]).clip(0.5, 1.5).dropna()
    r = j["r"].reindex(w.index) * w
    r = r * j["r"].reindex(w.index).std() / r.std()
    print("   ungefiltert:", stats(j["r"].reindex(w.index)))
    print("   gefiltert  :", stats(r))


def vol_managed(p: M.Prepared, swaps: dict) -> pd.Series:
    cols = [a for a in IDX if a in p.assets]
    close = pd.DataFrame(p.close, index=p.dates, columns=p.assets)[cols]
    ret = close.pct_change().fillna(0)
    vol = ret.rolling(21, min_periods=21).std() * np.sqrt(252)
    tgt = (0.08 / len(cols) ** 0.5 / vol).clip(upper=3.0)              # uncorrelated proxy for 8 %
    cost = pd.Series(p.cost_bp, index=p.assets)[cols] / 1e4
    days = pd.Series(np.r_[0, np.diff(p.dates.values).astype("timedelta64[D]").astype(int)], index=p.dates)
    long_pa = pd.Series({a: swaps.get(a, [-0.018])[0] for a in cols})
    pos = pd.Series(0.0, index=cols)
    out = []
    for t in range(1, len(close)):
        d = close.index[t]
        pnl = float((pos * ret.iloc[t]).sum() + (pos * long_pa).sum() * days.iloc[t] / 365)
        if d.dayofweek == 2:
            target = tgt.iloc[t - 1].fillna(0)
            change = ((target - pos).abs() > 0.25 * pos.abs()) | (pos == 0)
            new = pos.where(~change, target)
            pnl -= float(((new - pos).abs() * cost).sum())
            pos = new
        out.append((d, pnl))
    return pd.Series(dict(out))


def reversal(p: M.Prepared, swaps: dict) -> pd.Series:
    cols = [a for a in IDX if a in p.assets]
    close = pd.DataFrame(p.close, index=p.dates, columns=p.assets)[cols]
    ret = close.pct_change().fillna(0)
    sd = ret.rolling(60, min_periods=60).std().shift(3)
    z = (close / close.shift(3) - 1) / (sd * np.sqrt(3))
    cost = pd.Series(p.cost_bp, index=p.assets)
    daily = pd.Series(0.0, index=close.index)
    ex, base, n = [], [], 0
    for a in cols:
        fwd = close[a].shift(-5) / close[a] - 1
        sig = z[a] < -2.0
        last = -10
        for i in np.flatnonzero(sig.to_numpy()):
            if i - last < 5 or i + 5 >= len(close):
                continue
            last = i
            size = 0.005 / len(cols) / sd[a].iloc[i]                    # 1 R = 0.5 %, shared budget
            net = size * (fwd.iloc[i] - 2 * cost[a] / 1e4 + swaps.get(a, [-0.018])[0] * 7 / 365)
            daily.iloc[i + 5] += net
            ex.append(fwd.iloc[i])
            n += 1
        base.append(fwd.dropna())
    print(f"3) Umkehr: {n} Ereignisse, Ø 5-Tage-Rendite {1e4 * np.mean(ex):+.1f} bp "
          f"vs. alle Tage {1e4 * pd.concat(base).mean():+.1f} bp")
    return daily


def main() -> int:
    warnings.filterwarnings("ignore")
    T.BROKER.update(json.loads(Path("artifacts/broker_costs/fundednext_demo.json").read_text()))
    swaps = T.BROKER["swaps_pa"]
    p, _ = T.load_all(HOLDOUT_START)
    s7 = M.run_backtest(p, T._cfg({"signal": "12m", "stop": "keiner"}))["daily"]["net"].loc[START:]
    print("S-007:", stats(s7))
    corr_filter(p, s7)
    vm = vol_managed(p, swaps).loc[START:]
    print("2) Vol-gesteuerte Aktienprämie:", stats(vm))
    basket(s7, vm, "Aktienprämie")
    rv = reversal(p, swaps).loc[START:]
    print("   Portfolio:", stats(rv))
    basket(s7, rv, "Umkehr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
