"""H-009 re-run with the *measured* Dukascopy swap markup instead of the guessed 30%.

H-009 charged ``differential x (1 - 30%)``. The broker's published swap table
(``data/duka_overnight_swaps_forex_2026-10-24.csv``, read by
``trading_lab.duka_swaps``) says the markup is a **flat fee per side**, not a
share of the differential: over the 26 carry pairs a median of 74 bp/yr for the
21 major-quoted legs and 168 bp/yr for the five exotic-quoted ones, at
differentials from -3.4 % to +4.6 % p.a. So the replacement is an absolute drag,

    swap = (differential - markup_long - markup_short) * days / 365

with each leg's markup taken from the table. The study sweeps the measured case,
the better *Advanced*/*Premium* terms, a doubled-markup stress case, and the
triple-rollover day.

  .venv/bin/python studies/carry_real_swaps.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "."); sys.path.insert(0, "studies")
from trading_lab.duka_swaps import carry_pct, markup_bp_yr, pip_size  # noqa: E402
from trading_lab.rates import load_rates  # noqa: E402
import run_carry  # noqa: E402

K, VIX_THR = 4, 0.70
PAIRS = list(run_carry.PAIRS)


def measured(prices: dict[str, float], policy: str = "regular") -> dict[str, float]:
    """``{pair: markup per side as a fraction}`` straight from the swap table."""
    return {p: markup_bp_yr(p, prices[p], pip_size(p.split("/")[1]), policy=policy) / 1e4
            for p in PAIRS}


def build(daily: dict[str, pd.Series], rates: pd.DataFrame, *,
          markup: dict[str, float] | None, triple: bool, placebo: bool = False,
          markup_scale: float = 1.0, scale_series: pd.Series | None = None,
          vix_pct: pd.Series | None = None) -> pd.DataFrame:
    """Monthly carry portfolio. ``active`` is decided at the START of the period.

    The calm filter must use information available when the position is opened
    (the month's first day), not the VIX at the month end - the end value is
    correlated with the month's own outcome and turns the filter into look-ahead.
    ``vix_end`` is kept only so the study can show the difference.
    """
    prices = pd.DataFrame(daily).sort_index().resample("1D").last().ffill()
    me = prices.resample("ME").last().index
    rows = []
    for a, b in zip(me[:-1], me[1:]):
        pa, pb = prices.loc[a], prices.loc[b]
        aa = a.tz_localize(None) if a.tzinfo else a
        bb = b.tz_localize(None) if b.tzinfo else b
        c = {p: float((rates[p.split("/")[0]] - rates[p.split("/")[1]])
                      .reindex([aa], method="ffill").iloc[0]) for p in prices.columns}
        ranked = sorted(c, key=c.get)
        if placebo:
            rng = np.random.default_rng(abs(hash((a.isoformat(), "placebo"))) % (2**32))
            order = list(rng.permutation(ranked))
            lon, sho = order[-K:], order[:K]
        else:
            lon, sho = ranked[-K:], ranked[:K]
        pr = lambda p: float(pb[p] / pa[p]) - 1.0
        price = float(np.mean([pr(p) for p in lon]) - np.mean([pr(p) for p in sho]))
        spread = float(np.mean([c[p] for p in lon]) - np.mean([c[p] for p in sho]))
        days = (b - a).days
        if markup is None:                       # H-009: 30 % of the differential
            swap = spread / 100.0 * 0.70 * days / 365.0
        else:                                    # measured: flat per-side fee
            scale = markup_scale
            if scale_series is not None:         # time-varying markup hypothesis
                scale *= float(scale_series.reindex([aa], method="ffill").iloc[0])
            fee = scale * float(np.mean([markup[p] for p in lon])
                                + np.mean([markup[p] for p in sho]))
            swap = (spread / 100.0 - fee) * days / 365.0
        if triple:
            swap *= (days + 2.0 * round(days / 7.0)) / days
        row = {"date": b, "price": price, "swap": swap, "total": price + swap,
               "spread": spread}
        if vix_pct is not None:
            v_start = float(vix_pct.reindex([aa], method="ffill").iloc[0])
            v_end = float(vix_pct.reindex([bb], method="ffill").iloc[0])
            row |= {"vix_start": v_start, "vix_end": v_end,
                    "active": v_start < VIX_THR}
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def stats(r: np.ndarray, label: str) -> str:
    r = r[np.isfinite(r)]
    if len(r) < 12:
        return f"  {label:42s} too few ({len(r)})"
    ann = float(np.prod(1 + r) ** (12 / len(r)) - 1); vol = float(np.std(r, ddof=1) * np.sqrt(12))
    seq = pd.Series(r).reset_index(drop=True)
    dd = float(((1 + seq).cumprod() / (1 + seq).cumprod().cummax() - 1).min())
    return (f"  {label:42s} n={len(r):3d} CAGR {ann*100:+6.1f}%  vol {vol*100:5.1f}%  "
            f"Sharpe {ann/vol if vol else 0:5.2f}  MDD {dd*100:+6.1f}%  pos {100*(seq>0).mean():3.0f}%")


def main() -> None:
    rates = load_rates()
    daily = {p: run_carry.daily(p, c) for p, c in run_carry.PAIRS.items()}
    last = {p: float(s.dropna().iloc[-1]) for p, s in daily.items()}

    print("### measured Dukascopy markup, per carry leg (Regular conditions)")
    print(f"    {'pair':9s} {'mid %/yr':>9s} {'FRED%':>7s} {'markup bp/yr':>13s}")
    for p in PAIRS:
        pv = pip_size(p.split("/")[1])
        mid = carry_pct(p, last[p], pv)
        fred = float((rates[p.split("/")[0]] - rates[p.split("/")[1]]).ffill().iloc[-1])
        print(f"    {p:9s} {mid:9.2f} {fred:7.2f} "
              f"{markup_bp_yr(p, last[p], pv):13.1f}")
    reg = measured(last, "regular")
    maj = [p for p in PAIRS if p.split("/")[1] not in {"PLN", "HUF", "CZK", "NOK", "SEK"}]
    exo = [p for p in PAIRS if p.split("/")[1] in {"PLN", "HUF", "CZK", "NOK", "SEK"}]
    print(f"    -> majors (n={len(maj)}) median {np.median([reg[p] for p in maj])*1e4:.0f} bp/yr, "
          f"exotics (n={len(exo)}) median {np.median([reg[p] for p in exo])*1e4:.0f} bp/yr per side\n")

    vix = yf.download("^VIX", start="2018-12-01", end="2026-10-01", interval="1d",
                      progress=False, auto_adjust=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vp = vix.rolling(250, min_periods=60).rank(pct=True)

    base = build(daily, rates, markup=None, triple=False, vix_pct=vp)
    act = base["active"].to_numpy()
    sp = base.loc[act, "spread"].mean()
    print("### THE BUG THAT KILLED H-009 - the calm filter must be read at period START")
    print(stats(base.loc[act, "total"].to_numpy(), "VIX at month START (correct)"))
    print(stats(base.loc[base['vix_end'] < VIX_THR, "total"].to_numpy(),
                "VIX at month END (look-ahead)"))
    print(stats(base.loc[base['vix_end'] >= VIX_THR, "total"].to_numpy(),
                "  control: VIX-end >= 70"))
    print(stats(base.loc[base['vix_start'] >= VIX_THR, "total"].to_numpy(),
                "  control: VIX-start >= 70"))
    print(stats(base["total"].to_numpy(), "no filter at all"))
    print(stats(base["price"].to_numpy(), "  price leg only (no financing)"))
    print("    -> the filter separates the groups by 30 pp at month end and by 0.1 at")
    print("       month start: there is no filter edge. H-009 is rejected.\n")

    print(f"### FX carry, 26 pairs, FRED rates, correct filter  "
          f"(mean carry spread over active months {sp:+.2f}%/yr)")
    print(f"    (H-009's -30 % therefore charged {0.30*sp*1e2:.0f} bp/yr per side)")

    cases: list[tuple[str, dict[str, float] | None, float, bool]] = [
        ("modelled -30% of differential (H-009)", None, 1.0, False),
        ("measured, Regular (the carry account)", reg, 1.0, False),
        ("measured, Regular + 3x rollover", reg, 1.0, True),
        ("measured, Advanced", measured(last, "advanced"), 1.0, False),
        ("measured, Premium", measured(last, "premium"), 1.0, False),
        ("measured, Regular x2 (stress)", reg, 2.0, True),
    ]
    for label, mk, scale, trip in cases:
        r = build(daily, rates, markup=mk, triple=trip, markup_scale=scale, vix_pct=vp)
        print(stats(r.loc[act, "total"].to_numpy(), label))

    print("\n### detailed: measured, Regular + 3x rollover")
    r = build(daily, rates, markup=reg, triple=True, vix_pct=vp)
    for lab, mask in (("full 2019-2026", np.ones(len(r), bool)),
                      ("train 2019-23", r.index.year <= 2023),
                      ("holdout 2024-26", r.index.year >= 2024)):
        print(stats(r.loc[act & mask, "total"].to_numpy(), lab))
    print(stats(r.loc[act, "price"].to_numpy(), "price only (no financing)"))
    yy = r.loc[act].groupby(r.loc[act].index.year)["total"].apply(lambda s: (1 + s).prod() - 1)
    print("    per year %:", {int(y): round(float(v) * 100, 1) for y, v in yy.items()})
    plc = build(daily, rates, markup=reg, triple=True, placebo=True, vix_pct=vp)
    print(stats(plc.loc[act, "total"].to_numpy(), "PLACEBO random ranking"))

    # ── the one assumption the snapshot cannot settle: is the markup stable? ──
    # The markup does not track the differential (corr +0.08), so the only
    # plausible time dependence is the rate level (2020-21 were zero-rate years)
    # or stress. Both are testable without a swap history.
    print("\n### is the markup time-stable? (the only thing a swap history would settle)")
    level = rates.mean(axis=1)                       # average rate level, % p.a.
    ref_lvl = float(level.dropna().iloc[-1])
    vix_m = vix.resample("ME").mean()
    ref_vix = float(vix_m.dropna().iloc[-1])
    print(f"    reference (snapshot month): avg rate level {ref_lvl:.2f}% p.a., VIX {ref_vix:.1f}")
    for label, ser in (("markup scales with the RATE LEVEL", level / ref_lvl),
                       ("markup scales with VIX", vix_m / ref_vix)):
        rr = build(daily, rates, markup=reg, triple=True, scale_series=ser, vix_pct=vp)
        print(stats(rr.loc[act, "total"].to_numpy(), f"{label} (x1 in 2026)"))
        for lab, mask in (("train 2019-23", rr.index.year <= 2023),
                          ("holdout 2024-26", rr.index.year >= 2024)):
            print(stats(rr.loc[act & mask, "total"].to_numpy(), f"    {lab}"))
    z = (level / ref_lvl).reindex(
        base.index.tz_localize(None) if base.index.tz else base.index, method="ffill")
    print(f"    scale range over the sample: rate level {z.min():.2f}x .. {z.max():.2f}x")
    print("    -> the markup was measured at TODAY's rate level; if it scaled with rates")
    print("       the 2020-21 zero-rate months were cheaper, not dearer, so the measured")
    print("       result is the conservative end of that axis.")


if __name__ == "__main__":
    main()
