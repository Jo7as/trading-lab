#!/usr/bin/env python3
"""H-010 dose-response and neighbourhood: is the pullback response real or a spike?

The base event study (`run_h010_pullback.py`) left two hints: responses that grow
with the holding horizon, and one instrument (GER40) above the significance line.
Both are exactly what a real mechanism produces - and exactly what noise produces
if you look at 32 cells. This separates the two by sweeping the two gates that
*define* the setup (trend strength, retrace depth) and by looking at the response
as a function of those two gates rather than at a single point.

A real retracement mechanism predicts: the deeper the retrace within an intact
trend, the stronger the continuation - so the response should be **monotone in
both**. A spike with no neighbourhood is noise.

Forward returns are measured **within the session day**: the frame is
session-filtered, so a plain shift would run the window across the overnight
boundary and book the gap as the setup's response. The window is capped at the
last bar of the same session day and samples that would cross are dropped (see
``forward_return_same_day``); the report prints how many were affected.

Costs follow the single convention in ``trading_lab.cost_gate``: a limit entry
with a time exit is ``round_trip_spread("limit", "time")`` = 0.5 x the full spread.

  .venv/bin/python studies/h010_variants.py
"""
from __future__ import annotations

import argparse
import sys
import warnings

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_lab.minute_data import DEFAULT_CACHE, load_minute_bid_ask  # noqa: E402
from trading_lab.cost_gate import round_trip_spread  # noqa: E402
from trading_lab.strategies.pullback import (  # noqa: E402
    PullbackConfig,
    session_frame,
    sigma_by_minute,
)

TRENDS = (0.25, 0.5, 0.75, 1.0, 1.5)
RETRACES = (0.15, 0.30, 0.50, 0.75)
HORIZON = 240          # the horizon with the strongest response in the base study


def forward_return_same_day(mid: np.ndarray, day: np.ndarray, horizon: int
                            ) -> tuple[np.ndarray, int]:
    """Forward return over ``horizon`` bars, dropped when it would leave the day.

    The frame is session-filtered, so a plain ``mid[i + horizon]`` runs the window
    into the *next* session across the overnight gap and books the gap as the
    setup's response. We require the bar at ``i + horizon`` to carry the same
    ``day`` label; samples whose window would cross the boundary are dropped
    (NaN). Returns the forward-return array and the number of dropped samples.
    """
    mid = np.asarray(mid, float)
    day = np.asarray(day)
    n = len(mid)
    fwd = np.full(n, np.nan)
    crossed = 0
    if horizon < n:
        same_day = day[horizon:] == day[:-horizon]
        crossed = int((~same_day).sum())
        fwd[:-horizon] = np.where(same_day, mid[horizon:] / mid[:-horizon] - 1.0, np.nan)
    return fwd, crossed


def prepare(m1: pd.DataFrame, cfg: PullbackConfig, horizon: int = HORIZON) -> dict:
    """Everything that does NOT depend on the two thresholds, computed once.

    sigma_by_minute is the expensive part (pivot + rolling over millions of rows),
    and it only depends on the session and the lookback - not on trend_min or
    pullback_min. Computing it once per (asset, session) is what makes the sweep
    affordable.
    """
    f = session_frame(m1, cfg)
    sig = sigma_by_minute(f, cfg)
    f = f.join(sig, on=["day", "dmin"])
    f = f[f["sigma"] > 0]
    mid = f["mid"].to_numpy()
    scale = (f["sigma"] * f["anchor"]).to_numpy()
    trend = (f["mid"] - f["anchor"]).to_numpy() / scale
    hi = f.groupby("day")["mid"].cummax().to_numpy()
    lo = f.groupby("day")["mid"].cummin().to_numpy()
    day = f["day"].to_numpy()
    fwd, crossed = forward_return_same_day(mid, day, horizon)
    return {"signals": f, "trend": trend, "retrace_long": (hi - mid) / scale,
            "retrace_short": (mid - lo) / scale, "fwd": fwd,
            "day": day, "crossed": crossed,
            "spread": float(((f["ask"] - f["bid"]) / f["mid"]).median()),
            "move": float(np.nanmean(np.abs(fwd)))}


def response(pre: dict, trend_min: float, pullback_min: float, pullback_max: float = 3.0
             ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """(signed forward return per trigger, day ids, mean |move|, spread)."""
    f = pre["signals"]
    day = pre["day"]
    state = ((pre["trend"] >= trend_min) & (pre["retrace_long"] >= pullback_min)
             & (pre["retrace_long"] <= pullback_max))
    state_s = ((pre["trend"] <= -trend_min) & (pre["retrace_short"] >= pullback_min)
               & (pre["retrace_short"] <= pullback_max))
    prev = pd.Series(state).groupby(f["day"]).shift(1, fill_value=False).to_numpy()
    prev_s = pd.Series(state_s).groupby(f["day"]).shift(1, fill_value=False).to_numpy()
    trig = np.where(state & ~prev, 1, np.where(state_s & ~prev_s, -1, 0))
    ok = (trig != 0) & np.isfinite(pre["fwd"])
    return trig[ok] * pre["fwd"][ok], day[ok], pre["move"], pre["spread"]


def t_block(r: np.ndarray, blk: np.ndarray, *, runs: int = 2000,
            seed: int = 7) -> float:
    """t of the mean from a day-level bootstrap.

    The cluster is the day, so the resampling unit is the day's *mean response*,
    not the individual trigger. Resampling triggers would treat the (heavily
    clustered) intraday triggers as independent and inflate t; collapsing to day
    means first is both correct and ~100x faster.
    """
    if len(r) < 50:
        return float("nan")
    day_mean = pd.Series(r).groupby(blk).mean().to_numpy()
    if len(day_mean) < 5:
        return float("nan")
    rng = np.random.default_rng(seed)
    n = len(day_mean)
    draws = rng.choice(day_mean, size=(runs, n), replace=True)
    se = draws.mean(axis=1).std(ddof=1)
    return float(r.mean() / se) if se else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", default="US100,SP500,GER40,XAUUSD,EURUSD,USDJPY")
    ap.add_argument("--sessions", default="ny,london")
    ap.add_argument("--start-year", type=int, default=2019)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    cached = {f.name.split("_1m_")[0] for f in DEFAULT_CACHE.glob("*_1m_bid_*.pkl")}

    frames = {a: load_minute_bid_ask(a, args.start_year, verbose=False)
              for a in assets if a in cached}
    print(f"### H-010 Dosis-Wirkung (Trend x Ruecksetzer), Horizont {HORIZON} Min")
    print("    Erwartung eines echten Mechanismus: Antwort steigt monoton in BEIDEN Achsen.")
    print(f"    net_lim = Antwort - {round_trip_spread('limit', 'time'):.1f} Spread "
          "(Limit-Einstieg + Zeit-Exit)")
    print("    Vorwaertsfenster endet spaetestens am letzten Bar desselben Session-Tages "
          "(cross-day-Samples verworfen).\n")
    allrows = []
    for session in sessions:
        for asset, m1 in frames.items():
            grid = []
            pre = {s: prepare(m1, PullbackConfig(asset=asset, session=s)) for s in [session]}
            pre = pre[session]
            for tmin in TRENDS:
                row = []
                for rmin in RETRACES:
                    r, blk, move, spread = response(pre, tmin, rmin)
                    if len(r) < 50:
                        row.append((np.nan, np.nan, 0)); continue
                    net = (r.mean() - round_trip_spread("limit", "time") * spread) * 1e4
                    row.append((r.mean() * 1e4, t_block(r, blk), len(r), net))
                    allrows.append({"session": session, "asset": asset, "trend": tmin,
                                    "retrace": rmin, "n": len(r),
                                    "mean_bp": r.mean() * 1e4, "net_lim_bp": net,
                                    "t": t_block(r, blk)})
                grid.append(row)
            print(f"    {session:6s} {asset:8s}  " + "".join(f"{r:>18.2f}" for r in RETRACES)
                  + f"   (Spalte = retrace_min; cross-day-Fenster verworfen: {pre['crossed']})")
            for tmin, row in zip(TRENDS, grid):
                cells = ""
                for c in row:
                    m = c[0] if np.isfinite(c[0]) else float("nan")
                    tt = c[1] if np.isfinite(c[1]) else float("nan")
                    mark = "*" if (np.isfinite(tt) and abs(tt) >= 2.5) else " "
                    cells += f"{m:8.2f}{tt:7.2f}{mark}" if np.isfinite(m) else f"{'-':>16s}"
                print(f"           trend>={tmin:<4.2f}      {cells}")
            print()

    df = pd.DataFrame(allrows)
    if df.empty:
        return 1
    print("    Monotonie-Test: Korrelation der Antwort mit trend_min und retrace_min")
    for (s, a), g in df.groupby(["session", "asset"]):
        if len(g) < 8:
            continue
        ct = np.corrcoef(g["trend"], g["mean_bp"])[0, 1]
        cr = np.corrcoef(g["retrace"], g["mean_bp"])[0, 1]
        n_pos = int((g["net_lim_bp"] > 0).sum())
        print(f"      {s:6s} {a:8s}  corr(trend) {ct:+.2f}   corr(retrace) {cr:+.2f}   "
              f"netto>0: {n_pos}/{len(g)}")
    good = df[(df["t"].abs() >= 2.5) & (df["net_lim_bp"] > 0)]
    print(f"\n    Zellen mit |t|>=2.5 und positivem Netto: {len(good)} von {len(df)}")
    if not good.empty:
        print(good.sort_values("net_lim_bp", ascending=False).round(3).to_string(index=False))
    print(f"    (Bei {len(df)} Zellen sind {len(df)*0.05:.1f} Treffer allein durch Zufall zu erwarten.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
