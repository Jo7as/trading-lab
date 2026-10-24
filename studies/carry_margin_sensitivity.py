"""Asymmetric broker margin + triple-rollover sensitivity for the FX carry basket.

H-009 assumed a *symmetric* haircut on the net differential. Reality:
(a) brokers pay less on the income side and charge more on the cost side, and the
carry basket is long high-yielders (income) and short low-yielders (also income,
because the short legs have negative carry) - so the *short-side* markup is what
costs us; (b) one weekday carries a triple swap (~+2/7 of the accrual).

Self-check: m_long = m_short = 30% must reproduce H-009 (Sharpe 1.59, MDD -8.9%).

  .venv/bin/python studies/carry_margin_sensitivity.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "."); sys.path.insert(0, "studies")
from trading_lab.rates import load_rates  # noqa: E402
import run_carry  # noqa: E402

K, VIX_THR = 4, 0.70
VARIANTS = [
    ("symmetric -30% (H-009)", 0.30, 0.30, False),
    ("symmetric -50%", 0.50, 0.50, False),
    ("asymmetric L10/S30", 0.10, 0.30, False),
    ("asymmetric L20/S50", 0.20, 0.50, False),
    ("asymmetric L20/S50 + 3x rollover", 0.20, 0.50, True),
    ("asymmetric L30/S70 (harsh)", 0.30, 0.70, True),
]


def build(daily: dict[str, pd.Series], rates: pd.DataFrame, m_l: float, m_s: float,
          triple: bool) -> pd.DataFrame:
    prices = pd.DataFrame(daily).sort_index().resample("1D").last().ffill()
    me = prices.resample("ME").last().index
    rows = []
    for a, b in zip(me[:-1], me[1:]):
        pa, pb = prices.loc[a], prices.loc[b]
        bb = b.tz_localize(None) if b.tzinfo else b
        c = {p: float((rates[p.split("/")[0]] - rates[p.split("/")[1]]).reindex([bb], method="ffill").iloc[0])
             for p in prices.columns}
        ranked = sorted(c, key=c.get)
        lon, sho = ranked[-K:], ranked[:K]
        pr = lambda p: float(pb[p] / pa[p]) - 1.0
        price = float(np.mean([pr(p) for p in lon]) - np.mean([pr(p) for p in sho]))
        days = (b - a).days
        # income on each side, each with its own markup; the short legs earn -carry
        inc_l = float(np.mean([c[p] for p in lon])) / 100.0 * (1 - m_l)
        inc_s = float(np.mean([-c[p] for p in sho])) / 100.0 * (1 - m_s)
        swap = (inc_l + inc_s) * days / 365.0
        if triple:                                   # one rollover day per week is 3x
            swap *= (days + 2.0 * round(days / 7.0)) / days
        rows.append({"date": b, "price": price, "swap": swap, "total": price + swap})
    return pd.DataFrame(rows).set_index("date")


def stats(r: np.ndarray, label: str) -> str:
    r = r[np.isfinite(r)]
    if len(r) < 12:
        return f"  {label:32s} too few"
    ann = float(np.prod(1 + r) ** (12 / len(r)) - 1); vol = float(np.std(r, ddof=1) * np.sqrt(12))
    seq = pd.Series(r).reset_index(drop=True)
    dd = float(((1 + seq).cumprod() / (1 + seq).cumprod().cummax() - 1).min())
    return (f"  {label:32s} n={len(r):3d} CAGR {ann*100:+6.1f}%  vol {vol*100:5.1f}%  "
            f"Sharpe {ann/vol if vol else 0:5.2f}  MDD {dd*100:+6.1f}%  pos {100*(seq>0).mean():3.0f}%")


def main() -> None:
    rates = load_rates()
    vix = yf.download("^VIX", start="2018-12-01", end="2026-10-01", interval="1d",
                      progress=False, auto_adjust=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vp = vix.rolling(250, min_periods=60).rank(pct=True)
    daily = {p: run_carry.daily(p, c) for p, c in run_carry.PAIRS.items()}

    print("### FX carry, 26 pairs, FRED rates, filter — broker margin sensitivity")
    for label, m_l, m_s, trip in VARIANTS:
        m = build(daily, rates, m_l, m_s, trip)
        idx = m.index.tz_localize(None) if m.index.tz else m.index
        act = (vp.reindex(idx, method="ffill") < VIX_THR).to_numpy()
        print(stats(m.loc[act, "total"].to_numpy(), label))
        if label.startswith("asymmetric L20/S50"):
            for yy, g in m.loc[act].groupby(m.loc[act].index.year):
                print(f"        {int(yy)}: {(1 + g['total']).prod() - 1:+.1%}")


if __name__ == "__main__":
    main()
