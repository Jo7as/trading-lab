#!/usr/bin/env python3
"""Horizon scan: at which holding period does any measurable state carry signal?

All nine registered hypotheses (H-001..H-009) were intraday (1 h to session end),
where the S0 threshold says a directional signal must hit ~70 % to break even.
The multi-day horizons were never touched, and there the same threshold is ~52 %.
This is the *descriptive* pass that comes before a hypothesis: it measures, per
market and holding period, how much of the forward move a state condition can
explain - and how that compares with the round-trip cost.

It is **exploration, not a test**. Scanning many states x markets x horizons is
exactly the multiple-testing trap the protocol warns about, so nothing here can
be quoted as evidence; it only decides what is worth pre-registering as a card.
Anything that survives must be re-tested on untouched data via
``docs/PATTERN_RESEARCH.md``.

States used (all dimensionless, all computed from data up to the decision bar):
  Z1  volatility regime   - 20d realised vol, percentile of the last 250d
  Z5  gap / overnight     - (open - prev close) / sigma_day
  Z6  compression         - range / median range of the last 60d
  Z10 autocorrelation     - sign of the previous H-return (momentum vs reversion)

  .venv/bin/python studies/state_scan.py
  .venv/bin/python studies/state_scan.py --assets US100,GER40 --horizons 240,1440,7200
"""
from __future__ import annotations

import argparse
import sys
import warnings

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_lab.cost_gate import spread_pct  # noqa: E402
from trading_lab.minute_data import DEFAULT_CACHE, load_minute_bid_ask  # noqa: E402

ASSETS = ("US100", "US30", "SP500", "GER40", "XAUUSD")
# Holding period in TRADING DAYS, measured on daily bars. Sub-daily horizons are
# deliberately excluded: they are what H-001..H-007 already covered, and the S0
# screen says the threshold there is ~70 % hit rate versus ~52 % here.
HORIZONS = (1, 2, 5, 10)
MIN_BARS = 3000


def daily_bars(m1: pd.DataFrame) -> pd.DataFrame:
    """Daily OHLC + median spread fraction from the 1-minute bid/ask frame."""
    g = m1.groupby(m1.index.tz_convert("UTC").normalize())
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(),
        "spread": g.apply(lambda d: float(spread_pct(d).median())),
    })
    return out.dropna()


def states(d: pd.DataFrame) -> pd.DataFrame:
    """The Z1/Z5/Z6 state values, all as of the *close* of each day."""
    ret = d["close"].pct_change()
    sigma = ret.rolling(20).std()
    prev_c, prev_h, prev_l = d["close"].shift(1), d["high"].shift(1), d["low"].shift(1)
    rng = (d["high"] - d["low"])
    return pd.DataFrame({
        "Z1_vol_pct": sigma.rolling(250, min_periods=60).rank(pct=True),
        "Z5_gap": (d["open"] - prev_c) / (sigma * d["close"]),
        "Z6_range_ratio": rng / rng.rolling(60).median(),
        "Z2_dist_high": (d["close"] - prev_h) / (sigma * d["close"]),
        "Z2_dist_low": (d["close"] - prev_l) / (sigma * d["close"]),
    }, index=d.index)


def scan(asset: str, m1: pd.DataFrame) -> list[dict]:
    d = daily_bars(m1)
    if len(d) < 260:
        return []
    st = states(d)
    cost = float(d["spread"].median())
    rows = []
    for h in HORIZONS:
        # Non-overlapping windows only. With daily observations and an h-day
        # forward return the windows overlap by (h-1)/h, which inflates the
        # t-statistic by ~sqrt(h) - the classic overlapping-horizon trap. Taking
        # every h-th day makes the samples independent and the t honest; the
        # cost is n/h observations.
        sub = d.iloc[::h]
        fwd = sub["close"].shift(-1) / sub["close"] - 1.0     # exactly h days
        v_all = st.reindex(sub.index)
        ok0 = fwd.notna()
        if ok0.sum() < 60:
            continue
        move = float(fwd[ok0].abs().median())
        if not np.isfinite(move) or move == 0:
            continue
        ratio = move / cost                      # E|move| / round-trip spread
        base = float(fwd[ok0].mean())            # unconditional drift - the placebo
        for name in st.columns:
            v = v_all[name]
            ok = ok0 & v.notna()
            if ok.sum() < 40:
                continue
            hi, lo = v[ok].quantile(0.8), v[ok].quantile(0.2)
            for side, mask in (("high", ok & (v >= hi)), ("low", ok & (v <= lo))):
                r = fwd[mask]
                if len(r) < 20:
                    continue
                # excess over the unconditional drift, not over zero: in a rising
                # market every condition looks profitable against zero
                eff = float(r.mean()) - base
                sd = float(r.std(ddof=1))
                t = eff / (sd / np.sqrt(len(r))) if sd else 0.0
                rows.append({"asset": asset, "days": h, "state": name, "side": side,
                             "n": int(mask.sum()), "excess_pct": eff * 100,
                             "base_pct": base * 100,
                             "net_pct": (eff - cost) * 100, "t": t,
                             "ratio_move_cost": ratio,
                             "eff_over_cost": abs(eff) / cost if cost else np.nan})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", default=",".join(ASSETS))
    ap.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS))
    ap.add_argument("--start-year", type=int, default=2016)
    ap.add_argument("--min-t", type=float, default=3.0)
    args = ap.parse_args()
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    cached = {f.name.split("_1m_")[0] for f in DEFAULT_CACHE.glob("*_1m_bid_*.pkl")}

    warnings.filterwarnings("ignore")
    rows: list[dict] = []
    for a in assets:
        if a not in cached:
            print(f"  {a}: nicht im Cache", flush=True); continue
        m1 = load_minute_bid_ask(a, args.start_year, verbose=False)
        if m1 is None or len(m1) < MIN_BARS:
            continue
        r = scan(a, m1)
        rows += r
        print(f"  {a}: {len(m1):,} bars -> {len(r)} state/horizon cells", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        print("keine Ergebnisse"); return 1
    df["days"] = df["days"].astype(int)

    print("\n### Horizont-Scan (Exploration, KEIN Test)")
    print("    ratio = E|move|/Kosten. Bei ratio r braucht ein Richtungssignal ~(1+1/r)/2 Trefferquote.")
    print("    excess_pct = bedingte Rendite MINUS unbedingte Drift (Placebo), in %.")
    print("    net_pct = excess MINUS Kosten (>0 = schlaegt Drift UND Kosten).")
    print("    Nur nicht-ueberlappende Fenster (jeder h-te Tag), sonst sind die t-Werte ~sqrt(h) zu hoch.\n")
    pivot = df.pivot_table(index=["asset", "state", "side"], columns="days",
                           values="ratio_move_cost", aggfunc="mean")
    print("    E|move|/Kosten je Horizont:")
    print(pivot.round(2).to_string())

    print(f"\n    Zellen mit |t| >= {args.min_t} UND net_pct > 0 "
          f"(schlaegt Drift UND Kosten):")
    hits = df[(df["t"].abs() >= args.min_t) & (df["net_pct"] > 0)]
    if hits.empty:
        print("      keine")
    else:
        cols = ["asset", "days", "state", "side", "n", "excess_pct", "base_pct", "net_pct",
                "t", "ratio_move_cost"]
        print(hits.sort_values("eff_over_cost", ascending=False)[cols].round(3).to_string(index=False))
    sig = df[df["t"].abs() >= args.min_t]
    print(f"    Zellen mit |t| >= {args.min_t} (signifikant, egal ob nach Kosten): "
          f"{len(sig)} von {len(df)}")
    if not sig.empty:
        print(sig.sort_values("excess_pct", ascending=False)[
            ["asset", "days", "state", "side", "n", "excess_pct", "net_pct", "t"]
        ].head(12).round(3).to_string(index=False))
    print(f"    Verteilung excess_pct: Median {df['excess_pct'].median():+.3f}%  "
          f"p5 {df['excess_pct'].quantile(0.05):+.3f}%  p95 {df['excess_pct'].quantile(0.95):+.3f}%")
    print(f"\n    {len(df)} Zellen geprueft, {len(hits)} Treffer.")
    print("    ACHTUNG Multiple Testing: bei {0} Zellen sind {1:.1f} Treffer allein durch Zufall".format(
        len(df), len(df) * 0.05))
    print("    erwartet. Ein Treffer hier ist ein Kandidat fuer eine Karte, kein Beweis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
