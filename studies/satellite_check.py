#!/usr/bin/env python3
"""Quick checks BEFORE a spec: do two published effects still exist in our data? (no strategy)

1. Pre-FOMC drift (Lucca/Moench 2015): index return from 14:00 New York the day before a
   scheduled FOMC statement to 13:55 on statement day, vs. the same 24 h window on all
   other days. 1m CFD data, TRAIN only (2019-2024; statement at 14:00 NY). Dates:
   ``trading_lab/fomc_meetings.json`` (federalreserve.gov).
2. FX carry (G10, our 13 pairs): long the higher-yielding currency, vol-scaled, monthly
   rebalance, on long daily data 2004-2024; P&L = price + CFD swap (rate differential -
   1 % markup per side, the S-007 formula). Shows how much of the carry survives the markup.

  .venv/bin/python studies/satellite_check.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from studies.tsmom_check import TRAIN_START, load_all  # noqa: E402
from trading_lab import screening  # noqa: E402
from trading_lab.registry import HOLDOUT_START, load_cached  # noqa: E402
from trading_lab.strategies.tsmom_multi_asset import GROUP_OF, carry_rates, daily_returns, ewma_vol, TsmomConfig  # noqa: E402

NY = "America/New_York"


def fomc_dates(press_only: bool = False) -> list[pd.Timestamp]:
    m = json.loads(Path("trading_lab/fomc_meetings.json").read_text())["meetings"]
    return [pd.Timestamp(x["date"]) for x in m if (x["press"] or not press_only)]


def pre_fomc(asset: str) -> dict:
    m1 = load_cached(asset)
    m1 = m1[m1.index < HOLDOUT_START]
    mid = (m1["close"] + m1["ask_close"]) / 2
    ny = mid.copy()
    ny.index = mid.index.tz_convert(NY).tz_localize(None)
    days = pd.DatetimeIndex(sorted(set(ny.index.normalize())))
    days = days[days.dayofweek < 5]
    at = lambda d, hh, mm: ny.asof(d + pd.Timedelta(hours=hh, minutes=mm))  # noqa: E731
    rows = []
    for prev, day in zip(days[:-1], days[1:]):
        if (day - prev).days > 4:
            continue
        a, b = at(prev, 14, 0), at(day, 13, 55)
        if np.isfinite(a) and np.isfinite(b):
            rows.append((day, b / a - 1))
    s = pd.Series(dict(rows))
    s = s[s.index >= "2019-01-01"]
    fd = set(d for d in fomc_dates() if d < HOLDOUT_START.tz_localize(None))
    is_f = s.index.isin(list(fd))
    f, o = s[is_f], s[~is_f]
    return {"asset": asset, "n_fomc": int(is_f.sum()), "fomc_bp": float(f.mean() * 1e4),
            "other_bp": float(o.mean() * 1e4), "t_diff": float((f.mean() - o.mean()) / np.sqrt(f.var() / len(f) + o.var() / len(o))),
            "hit": float((f > 0).mean()), "sd_bp": float(f.std() * 1e4)}


def fx_carry() -> dict:
    p, _ = load_all(HOLDOUT_START)
    fx = [i for i, a in enumerate(p.assets) if GROUP_OF[a] in ("fx_usd", "fx_kreuz")]
    ret = daily_returns(p.close)[:, fx]
    vol = ewma_vol(ret, 60)
    cfg = TsmomConfig()
    lo = np.column_stack([carry_rates(p.assets[i], p.rates, cfg)[0].reindex(p.dates).ffill().to_numpy() for i in fx])
    sh = np.column_stack([carry_rates(p.assets[i], p.rates, cfg)[1].reindex(p.dates).ffill().to_numpy() for i in fx])
    diff = (lo - sh) / 2                                   # = rate differential (base - quote)
    days = np.r_[0, np.diff(p.dates.values).astype("timedelta64[D]").astype(int)]
    month = p.dates.to_period("M")
    pos = np.zeros(len(fx))
    price, fin, fin_nomark = np.zeros(len(p.dates)), np.zeros(len(p.dates)), np.zeros(len(p.dates))
    for t in range(1, len(p.dates)):
        price[t] = pos @ ret[t]
        car = np.where(pos > 0, lo[t], sh[t])
        fin[t] = np.nansum(pos * car) * days[t] / 365          # pos sign x (diff - markup)
        fin_nomark[t] = np.nansum(pos * diff[t]) * days[t] / 365
        if month[t] != month[t - 1]:                            # rebalance at the first day of a month
            ok = np.isfinite(vol[t - 1]) & np.isfinite(diff[t - 1]) & (vol[t - 1] > 0)
            w = np.where(ok, np.sign(np.nan_to_num(diff[t - 1])) / np.where(ok, vol[t - 1], 1), 0)
            # equal risk per pair, scaled to ~8 % p.a. as if uncorrelated (a measurement, not a spec)
            pos = np.where(ok, w * 0.08 / max(np.sqrt(np.sum((w * np.nan_to_num(vol[t - 1])) ** 2)), 1e-9), 0)
    d = pd.DataFrame({"price": price, "fin": fin, "fin_nomark": fin_nomark}, index=p.dates).loc[TRAIN_START:]
    net, gross = d["price"] + d["fin"], d["price"] + d["fin_nomark"]
    sh_ = lambda x: float(x.mean() / x.std() * np.sqrt(252))  # noqa: E731
    return {"years": int(d.index.year.nunique()), "price_pa": float(d["price"].sum() / d.index.year.nunique() * 100),
            "carry_pa_no_markup": float(d["fin_nomark"].sum() / d.index.year.nunique() * 100),
            "carry_pa_net": float(d["fin"].sum() / d.index.year.nunique() * 100),
            "sharpe_no_markup": sh_(gross), "sharpe_net": sh_(net), "t_net": float(screening.t_stat(net)),
            "per_year_net": {int(k): round(100 * v, 1) for k, v in net.groupby(net.index.year).sum().items()}}


def main() -> int:
    warnings.filterwarnings("ignore")
    print("1) Pre-FOMC-Drift, 14:00 NY Vortag -> 13:55 NY Entscheidungstag, 2019-2024 (Training)")
    for a in ("SP500", "US100", "US30", "GER40"):
        r = pre_fomc(a)
        print(f"   {a:6s} FOMC n={r['n_fomc']:2d}: Ø {r['fomc_bp']:+6.1f} bp (Streuung {r['sd_bp']:.0f} bp, "
              f"{r['hit']:.0%} positiv) | übrige Tage Ø {r['other_bp']:+5.1f} bp | Unterschied t {r['t_diff']:+.2f}")
    print("\n2) FX-Carry G10 (13 Paare), monatlich, Vola-skaliert ~8 %, 2004-2024 (Training)")
    c = fx_carry()
    print(f"   Kursbewegung {c['price_pa']:+.2f} % p.a. | Zinsertrag ohne Aufschlag {c['carry_pa_no_markup']:+.2f} % p.a. | "
          f"mit 1 % Aufschlag {c['carry_pa_net']:+.2f} % p.a.")
    print(f"   Sharpe ohne Aufschlag {c['sharpe_no_markup']:.2f}, mit Aufschlag {c['sharpe_net']:.2f} (t {c['t_net']:.2f})")
    print("   Jahre netto %:", c["per_year_net"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
