#!/usr/bin/env python3
"""Swap validation for the FX carry basket with the most realistic data available.

1. **Real rates**: monthly 3-month interbank rates from FRED (not hand-written
   annual averages) -> measured carry per pair and month.
2. **Broker markup sensitivity**: a haircut on the earned differential
   (0/30/50 %) - brokers pay the client less than the full differential.
3. **Roll diagnostic**: does the CFD price already contain a swap adjustment?
   We measure each pair's median 1-minute return around the 17:00 NY roll and
   compare it with the expected daily carry. ~0 means swap is charged separately
   (so it must be modelled); ≈ the differential means the data already pays it.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
import yfinance as yf

sys.path.insert(0, "."); sys.path.insert(0, "studies")
from trading_lab.carry import CarryConfig, monthly_returns, POLICY_RATES
from trading_lab.rates import coverage, load_rates
import run_carry

PAIRS = run_carry.PAIRS
EXOTIC = {"EUR/PLN", "EUR/HUF", "EUR/CZK", "EUR/NOK", "EUR/SEK"}


def carry_from_rates(rates: pd.DataFrame) -> dict[str, pd.Series]:
    """Monthly carry differential per pair, in % p.a., from the FRED table."""
    out = {}
    for p in PAIRS:
        base, quote = p.split("/")
        if base in rates.columns and quote in rates.columns:
            out[p] = (rates[base] - rates[quote]).ffill()
    return out


def portfolio(pairs_daily: dict[str, pd.Series], carry_monthly: dict[str, pd.Series],
              *, haircut: float, include_swap: bool, vix_pct: pd.Series) -> dict:
    prices = pd.DataFrame(pairs_daily).sort_index().resample("1D").last().ffill()
    me = prices.resample("ME").last().index
    rows = []
    for a, b in zip(me[:-1], me[1:]):
        pa, pb = prices.loc[a], prices.loc[b]
        bb = b.tz_localize(None) if getattr(b, "tzinfo", None) is not None else b
        c = {p: float(carry_monthly[p].reindex([bb], method="ffill").iloc[0]) for p in carry_monthly}
        ranked = sorted(c, key=c.get)
        lon, sho = ranked[-4:], ranked[:4]
        pr = lambda p: float(pb[p] / pa[p]) - 1.0
        price = np.mean([pr(p) for p in lon]) - np.mean([pr(p) for p in sho])
        days = (b - a).days
        swap = ((np.mean([c[p] for p in lon]) - np.mean([c[p] for p in sho])) / 100.0
                * (1 - haircut) * days / 365.0) if include_swap else 0.0
        rows.append({"date": b, "price": price, "swap": swap, "total": price + swap})
    m = pd.DataFrame(rows).set_index("date")
    v = vix_pct.reindex(m.index.tz_localize(None) if m.index.tz else m.index, method="ffill").to_numpy()
    m["active"] = v < 0.70
    return m


def stats(r: np.ndarray, label: str) -> str:
    r = r[np.isfinite(r)]
    if len(r) < 12:
        return f"  {label:34s} too few ({len(r)})"
    ann = float(np.prod(1 + r) ** (12 / len(r)) - 1); vol = float(np.std(r, ddof=1) * np.sqrt(12))
    seq = pd.Series(r); dd = float(((1 + seq).cumprod() / (1 + seq).cumprod().cummax() - 1).min())
    return (f"  {label:34s} n={len(r):3d} CAGR {ann*100:+6.1f}%  vol {vol*100:5.1f}%  "
            f"Sharpe {ann/vol if vol else 0:5.2f}  MDD {dd*100:+6.1f}%  pos {100*(seq>0).mean():3.0f}%")


def main() -> None:
    rates = load_rates()
    print("### FRED rate coverage (monthly 3m interbank, % p.a.)")
    print(coverage(rates).to_string())
    vix = yf.download("^VIX", start="2018-12-01", end="2026-10-01", interval="1d",
                      progress=False, auto_adjust=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix_pct = vix.rolling(250, min_periods=60).rank(pct=True)

    daily = {p: run_carry.daily(p, c) for p, c in PAIRS.items()}
    carry = carry_from_rates(rates)
    print(f"\n### carry portfolio with FRED rates ({len(carry)} pairs)")

    for label, hc in (("swap: full differential", 0.0), ("swap: -30% broker markup", 0.30),
                      ("swap: -50% broker markup", 0.50)):
        m = portfolio(daily, carry, haircut=hc, include_swap=True, vix_pct=vix_pct)
        print(stats(m.loc[m["active"], "price"].to_numpy(), f"price only (filter) [{label}]"))
        print(stats(m.loc[m["active"], "total"].to_numpy(), f"price+swap (filter) [{label}]"))
    g10 = {p: s for p, s in daily.items() if p not in EXOTIC}
    m = portfolio(g10, {p: c for p, c in carry.items() if p in g10}, haircut=0.3,
                  include_swap=True, vix_pct=vix_pct)
    print(stats(m.loc[m["active"], "total"].to_numpy(), "G10 only, -30% markup (filter)"))

    # roll diagnostic: does the 1m CFD price move at the 17:00 NY roll by the carry?
    print("\n### roll diagnostic (does the CFD price already pay the swap?)")
    print(f"{'pair':9s} {'median roll ret bp':>18s} {'expected daily bp':>18s} {'ratio':>7s}")
    from pass_speed import load_cached
    for p in ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD"):
        try:
            m1 = load_cached(p)
        except Exception:
            continue
        loc = m1.index.tz_convert("America/New_York")
        mins = loc.hour * 60 + loc.minute
        sel = mins.isin([16 * 60 + 58, 17 * 60 + 1]).to_numpy()
        sub = pd.DataFrame({"d": loc.date[sel], "m": mins.to_numpy()[sel],
                            "c": m1["close"].to_numpy()[sel]})
        piv = sub.pivot_table(index="d", columns="m", values="c")
        if piv.shape[1] < 2 or piv.empty:
            print(f"{p:9s} {'no roll bars':>18s}")
            continue
        roll = float((piv[17 * 60 + 1] / piv[16 * 60 + 58] - 1).median()) * 1e4
        base, quote = (p[:3], p[3:])
        exp = float((rates[base] - rates[quote]).median()) / 100.0 / 365.0 * 1e4 if base in rates else np.nan
        print(f"{p:9s} {roll:18.3f} {exp:18.3f} {(roll/exp if exp else np.nan):7.2f}")


if __name__ == "__main__":
    main()
