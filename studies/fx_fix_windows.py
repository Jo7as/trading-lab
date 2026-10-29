#!/usr/bin/env python3
"""Quick check (no strategy): the LONG fix windows of Krohn/Mueller/Whelan (JF 2024) on our data.

The USD rises into the Tokyo, ECB and London fixes and falls afterwards (docs/PAPERS.md).
Windows as in their Section V.E (local exchange times, DST-safe):

* ``vor_ezb``: USD long from 02:00 New York to the ECB fix 14:15 Frankfurt;
* ``nach_london``: foreign currency long from the London fix 16:00 London to 17:00 New York;
* ``vor_tokio`` / ``nach_tokio``: USD long 17:00 NY (previous day) -> Tokyo fix 9:55 Tokyo,
  then foreign currency long until 02:00 New York.

Per pair and window: mean return in bp on mid, and net of the real bid/ask of the entry and exit
minute + 0.5 bp slippage per side (ftmo profile). 1m Dukascopy data, TRAIN only (2019-2024).

  .venv/bin/python studies/fx_fix_windows.py
"""
from __future__ import annotations

import sys
import warnings

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_lab import costs, screening  # noqa: E402
from trading_lab.registry import HOLDOUT_START, load_cached  # noqa: E402

PAIRS = {"EURUSD": 1, "GBPUSD": 1, "AUDUSD": 1, "NZDUSD": 1, "USDCAD": -1, "USDCHF": -1, "USDJPY": -1}
SLIP_BP = 0.5


def at_local(days: pd.DatetimeIndex, tz: str, hh: int, mm: int, shift_days: int = 0) -> pd.DatetimeIndex:
    t = (days + pd.Timedelta(days=shift_days, hours=hh, minutes=mm)).tz_localize(tz, nonexistent="shift_forward",
                                                                               ambiguous=False)
    return t.tz_convert("UTC")


def window_returns(m1: pd.DataFrame, start: pd.DatetimeIndex, end: pd.DatetimeIndex, d_usd: int, sign: int
                   ) -> pd.DataFrame:
    """d_usd = +1: USD long. sign = +1 if the pair quotes the foreign currency in USD (EURUSD)."""
    idx = m1.index
    t = idx.as_unit("ns").asi8                                # the cache index is not in ns (repo trap)
    i0 = np.searchsorted(t, start.as_unit("ns").asi8)
    i1 = np.searchsorted(t, end.as_unit("ns").asi8)
    ok = (i0 < len(idx)) & (i1 < len(idx)) & (i1 > i0)
    i0, i1 = i0[ok], i1[ok]
    gap0 = (idx[i0] - start[ok]) < pd.Timedelta(minutes=10)
    gap1 = (idx[i1] - end[ok]) < pd.Timedelta(minutes=10)
    i0, i1 = i0[gap0 & gap1], i1[gap0 & gap1]
    d_pair = -d_usd * sign                                    # +1 = buy the pair
    bo, ao = m1["open"].to_numpy(), m1["ask_open"].to_numpy()
    mid0, mid1 = (bo[i0] + ao[i0]) / 2, (bo[i1] + ao[i1]) / 2
    ent = np.where(d_pair > 0, ao[i0], bo[i0]) * (1 + d_pair * SLIP_BP * 1e-4)
    ext = np.where(d_pair > 0, bo[i1], ao[i1]) * (1 - d_pair * SLIP_BP * 1e-4)
    return pd.DataFrame({"mid_bp": d_pair * (mid1 / mid0 - 1) * 1e4, "net_bp": d_pair * (ext / ent - 1) * 1e4},
                        index=idx[i0])


def main() -> int:
    warnings.filterwarnings("ignore")
    rows, port = [], {}
    for pair, sign in PAIRS.items():
        m1 = costs.adjust_frame(load_cached(pair), pair, "ftmo")
        m1 = m1[m1.index < HOLDOUT_START]
        days = pd.bdate_range("2019-01-02", "2024-12-31")
        wins = {
            "vor_ezb": (at_local(days, "America/New_York", 2, 0), at_local(days, "Europe/Berlin", 14, 15), +1),
            "nach_london": (at_local(days, "Europe/London", 16, 0), at_local(days, "America/New_York", 17, 0), -1),
            "vor_tokio": (at_local(days - pd.Timedelta(days=1), "America/New_York", 17, 0),
                          at_local(days, "Asia/Tokyo", 9, 55), +1),
            "nach_tokio": (at_local(days, "Asia/Tokyo", 9, 55), at_local(days, "America/New_York", 2, 0), -1),
        }
        for w, (s, e, d) in wins.items():
            r = window_returns(m1, s, e, d, sign)
            r = r[r.index.dayofweek < 5]
            port.setdefault(w, []).append(r["net_bp"].rename(pair))
            yrs = r["net_bp"].groupby(r.index.year).sum()
            rows.append({"paar": pair, "fenster": w, "n": len(r), "mitte_bp": r["mid_bp"].mean(),
                         "t_mitte": screening.t_stat(r["mid_bp"]), "netto_bp": r["net_bp"].mean(),
                         "t_netto": screening.t_stat(r["net_bp"]), "jahre_pos": f"{int((yrs > 0).sum())}/{len(yrs)}"})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("Krohn/Mueller/Whelan-Fenster, 2019-2024 (Training), bp je Tag")
    print(df.round(2).to_string(index=False))
    print("\nGleichgewichtetes Portfolio der 7 Paare je Fenster (netto):")
    for w, parts in port.items():
        p = pd.concat(parts, axis=1).mean(axis=1)
        yrs = p.groupby(p.index.year).sum()
        print(f"  {w:12s} Ø {p.mean():+.2f} bp/Tag, t {screening.t_stat(p):+.2f}, Jahre positiv "
              f"{int((yrs > 0).sum())}/{len(yrs)}, ~{p.mean() * 252 / 100:+.1f} % p.a. auf das Nominal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
