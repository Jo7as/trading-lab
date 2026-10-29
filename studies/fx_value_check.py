#!/usr/bin/env python3
"""Quick check (no strategy): FX value - do real exchange rates mean-revert over years?

Asness/Moskowitz/Pedersen (2013, "Value and Momentum Everywhere"): currency value = minus the
5-year change of the real exchange rate (spot x foreign CPI / US CPI, against its average
4.5-5.5 years ago). Cheap currencies outperform; value is NEGATIVELY correlated with momentum
- the reason to look at it as a partner for S-007.

Data (FRED, free, cached in artifacts/cache/fred): daily spot rates (H.10) since 1971, CPI
(US CPIAUCSL, OECD MEI for the others - Japan ends 2021-06 and is carried forward; Germany for the euro, DEM (EXGEUS, monthly) before 1999). CPI is used
with a 2-month lag (publication delay). Monthly rebalance at month end, 7 currencies vs USD.

* time-series: long a currency when it is cheap (value > 0), short when expensive;
* cross-section: weights = demeaned value rank (dollar neutral).
Both scaled to ~8 % vol p.a. (equal risk per currency, ex-ante vol of the last 36 months).
Price return only - FundedNext FX swaps are ~0 (AUD pairs aside, docs/FINDINGS.md).
TRAIN only (to 2024); correlation with the S-007 base over 2004-2024.

  .venv/bin/python studies/fx_value_check.py
"""
from __future__ import annotations

import io
import json
import sys
import urllib.request
import warnings
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_lab import screening  # noqa: E402

CACHE = Path("artifacts/cache/fred")
SPOT = {  # FRED id, True if quoted as USD per foreign unit
    "EUR": ("DEXUSEU", True), "DEM": ("EXGEUS", False), "JPY": ("DEXJPUS", False), "GBP": ("DEXUSUK", True),
    "CHF": ("DEXSZUS", False), "CAD": ("DEXCAUS", False), "AUD": ("DEXUSAL", True), "NZD": ("DEXUSNZ", True),
}
CPI = {"USD": "CPIAUCSL", "EUR": "DEUCPIALLMINMEI", "JPY": "JPNCPIALLMINMEI", "GBP": "GBRCPIALLMINMEI",
       "CHF": "CHECPIALLMINMEI", "CAD": "CANCPIALLMINMEI", "AUD": "AUSCPIALLQINMEI", "NZD": "NZLCPIALLQINMEI"}
CCYS = ("EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD")
END = "2025-01-01"


def fred(sid: str) -> pd.Series:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{sid}.csv"
    if not f.exists():
        req = urllib.request.Request(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}",
                                     headers={"User-Agent": "Mozilla/5.0"})
        f.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    df = pd.read_csv(f)
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna()


def usd_per(ccy: str) -> pd.Series:
    """Month-end USD price of one unit of ``ccy``; EUR before 1999 from DEM."""
    def one(c):
        sid, direct = SPOT[c]
        s = fred(sid)
        return (s if direct else 1 / s).resample("ME").last()
    s = one(ccy)
    if ccy == "EUR":
        dem = one("DEM") * 1.95583
        s = s.combine_first(dem[dem.index < s.index.min()])
    return s


def main() -> int:
    warnings.filterwarnings("ignore")
    spot = pd.DataFrame({c: usd_per(c) for c in CCYS})
    cpi = pd.DataFrame({c: fred(CPI[c]).resample("ME").last().ffill() for c in CPI}).ffill()
    cpi = cpi.shift(2).reindex(spot.index).ffill()                      # publication lag, on the spot months
    rer = np.log(spot * cpi[list(CCYS)].div(cpi["USD"], axis=0))
    past = rer.rolling(13).mean().shift(54)                             # average 4.5-5.5 years ago
    value = -(rer - past)                                               # > 0: cheap
    ret = spot.pct_change().shift(-1)                                   # next month's return
    vol = spot.pct_change().rolling(36).std() * np.sqrt(12)
    ok = value.notna() & vol.notna() & ret.notna()
    w_ts = np.sign(value).where(ok, 0) / vol
    rank = value.where(ok).rank(axis=1)
    w_cs = (rank.sub(rank.mean(axis=1), axis=0)).fillna(0) / vol
    out = {}
    for name, w in (("Zeitreihe", w_ts), ("Querschnitt", w_cs)):
        w = w.fillna(0)
        raw = (w * ret.fillna(0)).sum(axis=1)
        # scale to ~8 % using the ex-ante vol of the raw book (equal risk, uncorrelated proxy)
        scale = 0.08 / np.sqrt(((w * vol.fillna(0)) ** 2).sum(axis=1)).replace(0, np.nan)
        r = (raw * scale).dropna()
        r = r[(r.index >= "1980-01-01") & (r.index < END)]
        out[name] = r
    # S-007 base monthly (net, measured swaps) for the correlation
    s7 = None
    try:
        sys.argv = ["x"]
        import studies.tsmom_check as T
        from trading_lab.registry import HOLDOUT_START
        from trading_lab.strategies import tsmom_multi_asset as M
        T.BROKER.update(json.loads(Path("artifacts/broker_costs/fundednext_demo.json").read_text()))
        p, _ = T.load_all(HOLDOUT_START)
        d = M.run_backtest(p, T._cfg({"signal": "12m", "stop": "keiner"}))["daily"]["net"]
        s7 = d.groupby(d.index.to_period("M")).sum()
    except Exception as e:  # noqa: BLE001
        print("S-007 nicht geladen:", e)
    print("FX-Value (7 Währungen gg. USD), monatlich, ~8 % Ziel-Vola, nur Kursrendite")
    for name, r in out.items():
        print(f"\n  {name}: Sharpe 1980-2024 {r.mean() / r.std() * np.sqrt(12):.2f} (t {screening.t_stat(r):.2f}, {len(r)} Monate)")
        for a, b in (("1980", "1990"), ("1990", "2000"), ("2000", "2010"), ("2010", "2020"), ("2020", "2025")):
            x = r[(r.index >= a) & (r.index < b)]
            print(f"     {a}er: Sharpe {x.mean() / x.std() * np.sqrt(12):+.2f}, {100 * x.sum() / max(len(x) / 12, 1):+.1f} % p.a.")
        x = r[r.index >= "2004-01-01"]
        print(f"     2004-2024: Sharpe {x.mean() / x.std() * np.sqrt(12):+.2f}, Jahre positiv "
              f"{(x.groupby(x.index.year).sum() > 0).sum()}/{x.index.year.nunique()}")
        if s7 is not None:
            j = pd.concat([x.to_period("M"), s7], axis=1, join="inner").dropna()
            print(f"     Korrelation mit S-007 (monatlich, {len(j)} Monate): {j.corr().iloc[0, 1]:+.2f}")
            if name == "Zeitreihe":
                k7 = j.iloc[:, 1] / j.iloc[:, 1].std()
                kv = j.iloc[:, 0] / j.iloc[:, 0].std()
                both = (k7 + kv) / 2
                print(f"     Korb S-007 + Value (gleiches Risiko): Sharpe {both.mean() / both.std() * np.sqrt(12):.2f} "
                      f"vs. S-007 allein {k7.mean() / k7.std() * np.sqrt(12):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
