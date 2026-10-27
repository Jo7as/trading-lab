#!/usr/bin/env python3
"""H-012: one fixed time-series-momentum rule across 26 markets (train only).

Pre-registered rule - written down BEFORE the first run, no parameter search:

  * Universe: every instrument in the m1 cache with 2019-2026 data except BTCUSD
    (crypto CFDs are expensive and often restricted at prop firms): 14 FX pairs,
    6 indices, gold, silver, Brent, WTI, copper, US 10y.
  * Daily bars: mid close at 17:00 New York (the FX day boundary).
  * Signal: sign of the 63-trading-day return (~3 months). Long if > 0, short if < 0.
    Decided on day t's close, traded at day t+1's close (signal lagged one day).
  * Size: inverse volatility - each market gets the same risk: weight =
    (TARGET_VOL / sqrt(N_active)) / sigma_i, sigma_i = 20-day std of daily returns,
    annualised. Portfolio target 10 % annual volatility (ex ante, correlations ignored).
  * Costs, per venue profile (``trading_lab.costs``, default ftmo):
      - trading: |change in weight| x (half the liquid-hours median spread + one
        commission side), in fractions of notional;
      - financing (overnight swaps / CFD funding): 2.5 % p.a. of |notional| on
        indices, commodities, metals and bonds, 1.0 % p.a. on FX - a conservative
        markup that ignores the interest-rate component (which can go either way).

The ONLY confirmatory variant is lookback 63. Lookbacks 21/126/252 are computed as
neighbours for S3 (is it a plateau?) and are NOT used for selection. n_trials = 1.

TRAIN ONLY (2019 .. 2024). The holdout (2025+) stays closed until released.

  .venv/bin/python studies/run_h012_trend_multi.py
  .venv/bin/python studies/run_h012_trend_multi.py --kosten ecn_raw
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_lab import costs, screening  # noqa: E402
from trading_lab.registry import HOLDOUT_START, load_cached  # noqa: E402

UNIVERSE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD", "EURGBP",
            "EURJPY", "GBPJPY", "AUDJPY", "EURCHF", "AUDNZD", "AUDCAD",
            "US100", "SP500", "US30", "GER40", "UK100", "JP225",
            "XAUUSD", "XAGUSD", "BRENT", "WTI", "COPPER", "UST10Y"]
PRIMARY_LOOKBACK = 63
NEIGHBOURS = (21, 126, 252)
VOL_WINDOW = 20
TARGET_VOL = 0.10
FIN_RATE = {"fx": 0.010, "index": 0.025, "metal": 0.025, "other": 0.025}
DAILY_CACHE = Path("artifacts/cache/daily_h012")   # own folder: cache/daily belongs to run_carry.py
OUT = Path("artifacts/h012_trend_multi.json")


def daily(asset: str) -> pd.DataFrame:
    """Daily mid close at 17:00 NY plus the liquid-hours median spread (fraction)."""
    DAILY_CACHE.mkdir(parents=True, exist_ok=True)
    f = DAILY_CACHE / f"{asset}.pkl"
    if f.exists():
        return pd.read_pickle(f)
    m1 = load_cached(asset, 2019)
    if m1.empty:
        return pd.DataFrame()
    mid = (m1["close"] + m1["ask_close"]) / 2
    ny = mid.index.tz_convert("America/New_York")
    key = (ny - pd.Timedelta(hours=17)).normalize().tz_localize(None) + pd.Timedelta(days=1)
    d = pd.DataFrame({"close": mid.groupby(key.to_numpy()).last()})
    d = d[d.index.dayofweek < 5]
    sp = (m1["ask_close"] - m1["close"]) / mid
    liquid = (m1.index.hour >= 13) & (m1.index.hour < 16)
    sp_l = sp[liquid]
    d["spread_frac"] = sp_l.groupby(sp_l.index.year).median().reindex(d.index.year).to_numpy()
    d.to_pickle(f)
    return d


def portfolio(closes: pd.DataFrame, spread: pd.DataFrame, lookback: int, profile: str
              ) -> tuple[pd.Series, pd.DataFrame, dict]:
    ret = closes.pct_change()
    sig = np.sign(closes / closes.shift(lookback) - 1.0)
    sigma = ret.rolling(VOL_WINDOW).std() * np.sqrt(252)
    n_active = sig.notna().sum(axis=1).clip(lower=1)
    w = sig * (TARGET_VOL / np.sqrt(n_active)).to_numpy()[:, None] / sigma
    w = w.shift(1)                                   # decided at t's close, held over t+1
    gross = (w * ret).fillna(0.0)
    # costs
    com = pd.DataFrame({a: [costs.commission_bp_side(a, float(p), profile) * 1e-4 for p in closes[a].ffill().bfill()]
                        for a in closes}, index=closes.index)
    turn = w.diff().abs().fillna(w.abs().fillna(0))
    trade_cost = (turn * (spread / 2 + com)).fillna(0.0)
    cal_days = pd.Series(closes.index, index=closes.index).diff().dt.days.fillna(1).clip(lower=1)
    fin = pd.DataFrame({a: FIN_RATE[costs.asset_class(a)] for a in closes}, index=closes.index)
    fin_cost = (w.abs() * fin.mul(cal_days, axis=0) / 365.0).fillna(0.0)
    net = gross - trade_cost - fin_cost
    info = {"brutto_pa": float(gross.sum(axis=1).mean() * 252), "handel_pa": float(trade_cost.sum(axis=1).mean() * 252),
            "finanzierung_pa": float(fin_cost.sum(axis=1).mean() * 252),
            "umschlag_pa": float(turn.sum(axis=1).mean() * 252)}
    return net.sum(axis=1), net, info


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    m = (1 + r).resample("ME").prod() - 1
    eq = (1 + r).cumprod()
    dd = float((1 - eq / eq.cummax()).max())
    yrs = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    vol = float(r.std() * np.sqrt(252))
    return {"pct_pro_monat_mittel": round(float(m.mean()) * 100, 3), "pct_pro_monat_median": round(float(m.median()) * 100, 3),
            "positive_monate": round(float((m > 0).mean()), 3), "rendite_pa_pct": round(float(r.mean() * 252) * 100, 2),
            "vol_pa_pct": round(vol * 100, 2), "sharpe": round(float(r.mean() * 252) / vol, 2) if vol else None,
            "max_dd_pct": round(dd * 100, 2), "t_taeglich": round(screening.t_stat(r), 2),
            "jahre": {int(k): round(float(v) * 100, 1) for k, v in yrs.items()},
            "pos_years": int((yrs > 0).sum()), "n_years": int(len(yrs))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kosten", default=costs.DEFAULT_PROFILE)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    frames = {}
    for a in UNIVERSE:
        d = daily(a)
        if d.empty:
            print(f"  {a}: keine Daten - übersprungen")
            continue
        frames[a] = d
        print(f"  {a:7s} {len(d)} Tage, Spread (liquide) {np.nanmedian(d['spread_frac']) * 1e4:.2f} bp", flush=True)
    closes = pd.DataFrame({a: d["close"] for a, d in frames.items()}).sort_index()
    spread = pd.DataFrame({a: d["spread_frac"] for a, d in frames.items()}).reindex(closes.index)
    train = closes.index < HOLDOUT_START.tz_localize(None)
    closes, spread = closes[train], spread[train]
    if args.kosten in ("ecn_raw", "cme_micro"):
        prof = costs.get(args.kosten)
        for a, v in prof.spread_override.items():
            if a in spread:
                spread[a] = v / closes[a]
    res = {}
    for lb in (PRIMARY_LOOKBACK,) + NEIGHBOURS:
        r, per, info = portfolio(closes, spread, lb, args.kosten)
        r = r[r.index >= closes.index[0] + pd.Timedelta(days=int(lb * 1.5) + 30)]   # after warm-up
        s = stats(r)
        s.update({k: round(v * 100, 2) for k, v in info.items()})
        s["beitrag_klasse_pa_pct"] = {c: round(float(per[[a for a in per if costs.asset_class(a) == c]].sum(axis=1).mean() * 252 * 100), 2)
                                      for c in ("fx", "index", "metal", "other")}
        res[lb] = s
        tag = "PRIMÄR" if lb == PRIMARY_LOOKBACK else "Nachbar"
        print(f"\n  [{tag}] Lookback {lb}: {s['pct_pro_monat_mittel']:+.2f} %/Monat, Sharpe {s['sharpe']}, "
              f"max DD {s['max_dd_pct']} %, t {s['t_taeglich']}, Jahre {s['pos_years']}/{s['n_years']} {s['jahre']}")
        print(f"     brutto {s['brutto_pa']:+.2f} %/J, Handel -{s['handel_pa']:.2f}, Finanzierung -{s['finanzierung_pa']:.2f}, "
              f"Klassen {s['beitrag_klasse_pa_pct']}")
    p = res[PRIMARY_LOOKBACK]
    nb = [res[k]["rendite_pa_pct"] > 0 for k in NEIGHBOURS]
    row = {"name": f"H-012 TSMOM {PRIMARY_LOOKBACK}d [{args.kosten}]", "n": int(train.sum()),
           "gross_mean": p["brutto_pa"], "net_mean": p["rendite_pa_pct"], "placebo_gross": 0.0,
           "placebo_edge": p["brutto_pa"], "pos_years": p["pos_years"], "n_years": p["n_years"],
           "neighbour_positive": float(np.mean(nb)), "t_daily": p["t_taeglich"], "n_trials": 1,
           "pbo": None, "oos_pass": None, "gross_bp": None, "cost_bp": None}
    f = screening.run_funnel(pd.DataFrame([row]), up_to="S4")
    print("\n" + f.summary())
    for s in f.stages:
        if len(s.dropped):
            print(f"  raus an {s.name}")
    OUT.write_text(json.dumps({"kosten": args.kosten, "ergebnisse": res, "funnel_row": row,
                               "stufen": [{"name": s.name, "kept": len(s.kept), "dropped": len(s.dropped)} for s in f.stages]},
                              indent=1, default=str))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
