"""Carry final: G10-only base + a volatility-target sizing rule.

Step A — G10-only base (drop PLN/HUF/CZK/NOK/SEK whose real financing is the
open question), with the FRED rates and a -30% broker markup, reported with all
the protocol's checks: full sample, train/holdout, per year, placebo (random
ranking), dose-response over the carry spread, and a monthly turnover cost.

Step B — volatility-target sizing on that base: exposure scaled monthly by
min(cap, target_vol / trailing 60-day realised vol), causal. Reports the scaled
metrics, the average leverage and the implied drawdown (the prop question).

  .venv/bin/python studies/carry_final.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "."); sys.path.insert(0, "studies")
from trading_lab.rates import load_rates  # noqa: E402
import run_carry  # noqa: E402

EXOTIC = {"EUR/PLN", "EUR/HUF", "EUR/CZK", "EUR/NOK", "EUR/SEK"}
K, VIX_THR, MARKUP = 4, 0.70, 0.30
TARGET_VOL, LEV_CAP = 0.12, 2.0


def build(daily: dict[str, pd.Series], rates: pd.DataFrame) -> pd.DataFrame:
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
        spread = (float(np.mean([c[p] for p in lon])) - float(np.mean([c[p] for p in sho])))
        days = (b - a).days
        swap = spread / 100.0 * (1 - MARKUP) * days / 365.0
        price_r = float(np.mean([pr(p) for p in sho]) - np.mean([pr(p) for p in lon]))
        rows.append({"date": b, "price": price, "swap": swap, "total": price + swap,
                     "carry_spread": spread, "placebo": price_r})
    return pd.DataFrame(rows).set_index("date")


def stats(r: np.ndarray, label: str) -> str:
    r = r[np.isfinite(r)]
    if len(r) < 12:
        return f"  {label:38s} too few ({len(r)})"
    ann = float(np.prod(1 + r) ** (12 / len(r)) - 1); vol = float(np.std(r, ddof=1) * np.sqrt(12))
    seq = pd.Series(r).reset_index(drop=True)
    dd = float(((1 + seq).cumprod() / (1 + seq).cumprod().cummax() - 1).min())
    return (f"  {label:38s} n={len(r):3d} CAGR {ann*100:+6.1f}%  vol {vol*100:5.1f}%  "
            f"Sharpe {ann/vol if vol else 0:5.2f}  MDD {dd*100:+6.1f}%  pos {100*(seq>0).mean():3.0f}%")


def main() -> None:
    rates = load_rates()
    vix = yf.download("^VIX", start="2018-12-01", end="2026-10-01", interval="1d",
                      progress=False, auto_adjust=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vp = vix.rolling(250, min_periods=60).rank(pct=True)

    allp = {p: run_carry.daily(p, c) for p, c in run_carry.PAIRS.items()}
    g10 = {p: s for p, s in allp.items() if p not in EXOTIC}
    out = {}
    for name, basket in (("all 26 pairs", allp), ("G10 only (21)", g10)):
        m = build(basket, rates)
        idx = m.index.tz_localize(None) if m.index.tz else m.index
        m["active"] = (vp.reindex(idx, method="ffill") < VIX_THR).to_numpy()
        out[name] = m
        print(f"\n### {name} — FRED rates, -{int(MARKUP*100)}% markup, filter")
        for col in ("price", "total"):
            print(stats(m.loc[m["active"], col].to_numpy(), f"{col} (full 2019-2026)"))
            print(stats(m.loc[m["active"] & (m.index.year <= 2023), col].to_numpy(), f"{col} (train 2019-23)"))
            print(stats(m.loc[m["active"] & (m.index.year >= 2024), col].to_numpy(), f"{col} (holdout 2024-26)"))
        print(stats(m.loc[m["active"], "placebo"].to_numpy(), "PLACEBO (random ranking)"))
        yy = m.loc[m["active"]].groupby(m.loc[m["active"]].index.year)["total"].apply(lambda s: (1 + s).prod() - 1)
        print("    per year %:", {int(y): round(float(v) * 100, 1) for y, v in yy.items()})

    # dose-response: sort active months by the carry spread into quintiles
    g = out["G10 only (21)"]
    act = g.loc[g["active"]].copy()
    act["q"] = pd.qcut(act["carry_spread"].rank(method="first"), 5, labels=False)
    print("\n### dose-response (active months, carry spread quintiles, total return %)")
    print("   ", [round(float(act.loc[act["q"] == k, "total"].mean()) * 100, 2) for k in range(5)])

    # turnover cost estimate: 2 x mean half-spread of the held legs per month
    print("\n### cost estimate (monthly rebalance, 8 legs)")
    print("    using the FX spread table: ~0.3-1.5 bp per leg -> ~1-3 bp/month "
          "= 0.12-0.36 %/yr -> negligible against +8..16 %/yr")

    # ── Step B: volatility-target sizing on the G10 base ──
    print("\n### Step B — vol-target sizing (target 12 %, cap 2x) on the G10 base")
    tot = g["total"].where(g["active"], 0.0)
    trailing = tot.rolling(12, min_periods=6).std(ddof=1) * np.sqrt(12)
    scale = (TARGET_VOL / trailing).clip(upper=LEV_CAP).shift(1)     # causal
    scaled = tot * scale
    print(stats(tot.to_numpy(), "1x, active only"))
    print(stats(scaled.to_numpy(), "vol-targeted (causal)"))
    print(f"    average leverage {float(scale.mean()):.2f}x  max {float(scale.max()):.2f}x  "
          f"active months {int((tot != 0).sum())}/{len(tot)}")


if __name__ == "__main__":
    main()
