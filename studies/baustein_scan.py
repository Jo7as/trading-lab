#!/usr/bin/env python3
"""Quick checks for new basket building blocks - measurements, no strategies. TRAIN only.

Each idea has a forced-flow mechanism and a published source:

1. Options expiration (3rd Friday): index returns in the OpEx week vs other weeks, and the
   week after (Stivers/Sun-type expiration effects; dealer hedging). Long daily data 2004-2024.
2. Intraday momentum into the close (Gao et al. 2018, Baltussen et al. 2021): the return from
   the previous close to 15:30 NY predicts 15:30 -> 15:59. 1m CFD data 2019-2024.
3. FX fix (Evans 2018, "Forex trading and the WMR fix"): the move into 16:00 London reverses
   afterwards; strongest at month end. 1m CFD data 2019-2024.

Every line shows the effect next to the round-trip cost of the measured CFD (bp).

  .venv/bin/python studies/baustein_scan.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_lab import long_daily, screening  # noqa: E402
from trading_lab.registry import HOLDOUT_START, load_cached  # noqa: E402

NY, LDN = "America/New_York", "Europe/London"
COSTS = json.loads(Path("artifacts/tsmom/costs.json").read_text())
END = HOLDOUT_START.tz_localize(None)


def third_friday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year, month, 1)
    return d + pd.Timedelta(days=(4 - d.dayofweek) % 7 + 14)


def opex() -> None:
    print("1) Options-Verfall (3. Freitag): Ø Tagesrendite in bp, 2004-2024")
    for a in ("SP500", "US100", "US30", "GER40"):
        c = long_daily.load_daily(a)["close"]
        c = c[(c.index >= "2004-01-01") & (c.index < END)]
        r = c.pct_change().dropna()
        fr = {third_friday(y, m) for y in range(2004, 2025) for m in range(1, 13)}
        wk = pd.Series("rest", index=r.index)
        for f in fr:
            mon = f - pd.Timedelta(days=4)
            wk[(r.index >= mon) & (r.index <= f)] = "opex_woche"
            wk[(r.index > f) & (r.index <= f + pd.Timedelta(days=7))] = "woche_danach"
        g = r.groupby(wk)
        o, a_, rest = r[wk == "opex_woche"], r[wk == "woche_danach"], r[wk == "rest"]
        t = (o.mean() - rest.mean()) / np.sqrt(o.var() / len(o) + rest.var() / len(rest))
        print(f"   {a:6s} OpEx-Woche {1e4 * o.mean():+5.2f} | Woche danach {1e4 * a_.mean():+5.2f} | übrige {1e4 * rest.mean():+5.2f}"
              f" | t(OpEx-übrige) {t:+.2f} | Woche OpEx ~{5e4 * o.mean():+.1f} bp vs. Kosten ~{2 * COSTS[a] + 4:.1f} bp")
        del g


def close_momentum() -> None:
    print("\n2) Intraday-Momentum: Rendite Vortagesschluss -> 15:30 NY sagt 15:30 -> 15:59 voraus (2019-2024)")
    for a in ("SP500", "US100", "US30"):
        m1 = load_cached(a)
        m1 = m1[m1.index < HOLDOUT_START]
        mid = ((m1["close"] + m1["ask_close"]) / 2)
        mid.index = mid.index.tz_convert(NY).tz_localize(None)
        day = mid.index.normalize()
        at = lambda hh, mm: mid[(mid.index.hour == hh) & (mid.index.minute == mm)].groupby(day[(mid.index.hour == hh) & (mid.index.minute == mm)]).last()  # noqa: E731
        c16, c1530, c1559 = at(15, 59), at(15, 30), at(15, 59)
        prev = c16.shift(1)
        df = pd.concat({"r1": c1530 / prev - 1, "r2": c1559 / c1530 - 1}, axis=1).dropna()
        df = df[df.index.dayofweek < 5]
        sig = np.sign(df["r1"])
        trade = sig * df["r2"] * 1e4
        big = df["r1"].abs() > df["r1"].abs().quantile(0.8)
        print(f"   {a:6s} n={len(df)} | Korrelation {df.corr().iloc[0, 1]:+.3f} | mit Vorzeichen handeln Ø {trade.mean():+.2f} bp "
              f"(t {screening.t_stat(trade):+.2f}) | nur große Tage (Top 20 %) Ø {trade[big].mean():+.2f} bp "
              f"(t {screening.t_stat(trade[big]):+.2f}) | Kosten ~{2 * COSTS[a]:.1f} bp")


def fx_fix() -> None:
    print("\n3) FX-Fixing 16:00 London: Bewegung 15:30->16:00 und danach 16:00->16:30 (2019-2024)")
    for a in ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURGBP"):
        m1 = load_cached(a)
        m1 = m1[m1.index < HOLDOUT_START]
        mid = ((m1["close"] + m1["ask_close"]) / 2)
        mid.index = mid.index.tz_convert(LDN).tz_localize(None)
        day = mid.index.normalize()
        def at(hh, mm):  # noqa: E306
            m = (mid.index.hour == hh) & (mid.index.minute == mm)
            return mid[m].groupby(day[m]).last()
        df = pd.concat({"pre": at(16, 0) / at(15, 30) - 1, "post": at(16, 30) / at(16, 0) - 1}, axis=1).dropna()
        df = df[df.index.dayofweek < 5]
        me = df.index.to_series().groupby(df.index.to_period("M")).transform("max") == df.index.to_series()
        rev = -np.sign(df["pre"]) * df["post"] * 1e4                     # fade the move into the fix
        print(f"   {a:6s} alle Tage: Gegenbewegung Ø {rev.mean():+.2f} bp (t {screening.t_stat(rev):+.2f}) | "
              f"Monatsende (n={int(me.sum())}): Ø {rev[me].mean():+.2f} bp (t {screening.t_stat(rev[me]):+.2f}) | "
              f"Kosten ~{2 * COSTS[a]:.1f} bp")


def main() -> int:
    warnings.filterwarnings("ignore")
    opex()
    close_momentum()
    fx_fix()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
