#!/usr/bin/env python3
"""H11-A quick measurement (no spec yet): how predictable is a big London / NY move?

Before any token is built, this sets the bar a token reading has to clear. For each
session day:

  y      session range (first 4 h after the open) / median of the same window over the
         previous 20 sessions  -> "big move" = top quartile of y (threshold from the
         whole discovery sample, the label is about *relative* size)
  x_pre  range of the 6 h before the open / its own 20-session median
  x_rel  range of the 6 h before the open / the session's 20-session median
  x_prev y of the previous session (persistence)
  x_reg  20-session median / 100-session median of the window range (vol trend)

Reported per (asset, session): AUC of each predictor for the big-move label, the
big-move rate in the bottom / top quintile of the predictor (base rate 25 %), AUC per
year, P(big) by weekday (scheduled news sits on fixed days), and the room the move
leaves over costs (median window range in full spreads).

Clock: exchange-local per session (docs/TIME_CONVENTIONS.md). Mid prices,
``data_audit.build_study_frame``. All normalisers use prior sessions only.
Data: discovery years only (default 2016-2021); 2022-2024 confirmation, 2025+ holdout.

  uv run python studies/h11_move_forecast_quick.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from studies.cost_gate import load_cached  # noqa: E402
from trading_lab.data_audit import build_study_frame  # noqa: E402

SESSIONS = {
    "london": {"tz": "Europe/London", "open": "08:00"},
    "ny": {"tz": "America/New_York", "open": "09:30"},
}
PLAN = {  # asset -> sessions measured
    "US100": ("ny",), "SP500": ("ny",), "US30": ("ny",),
    "GER40": ("london",), "UK100": ("london",), "XAUUSD": ("london", "ny"),
}
WINDOW_H, PRE_H = 4, 6
NORM, NORM_LONG = 20, 100
MIN_COVER = 0.8
PREDICTORS = ("x_pre", "x_rel", "x_prev", "x_reg")


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUC (ties averaged); 0.5 = no information."""
    ok = ~np.isnan(score)
    s, y = score[ok], label[ok].astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def session_table(m1: pd.DataFrame, tz: str, open_hhmm: str) -> pd.DataFrame:
    hi = ((m1["high"] + m1["ask_high"]) / 2).to_numpy()
    lo = ((m1["low"] + m1["ask_low"]) / 2).to_numpy()
    spr = (m1["ask_close"] - m1["close"]).to_numpy()
    idx = m1.index
    days = pd.DatetimeIndex(sorted(set(idx.tz_convert(tz).normalize().tz_localize(None))))
    days = days[days.dayofweek < 5]
    rows = []
    for day in days:
        t0 = pd.Timestamp(f"{day.date()} {open_hhmm}", tz=tz)
        a = int(idx.searchsorted((t0 - pd.Timedelta(hours=PRE_H)).tz_convert("UTC")))
        b = int(idx.searchsorted(t0.tz_convert("UTC")))
        c = int(idx.searchsorted((t0 + pd.Timedelta(hours=WINDOW_H)).tz_convert("UTC")))
        if c - b < MIN_COVER * 60 * WINDOW_H or b - a < MIN_COVER * 60 * PRE_H:
            continue
        rows.append({"day": day, "pre": hi[a:b].max() - lo[a:b].min(),
                     "win": hi[b:c].max() - lo[b:c].min(), "spread": float(np.median(spr[b:c]))})
    t = pd.DataFrame(rows).set_index("day")
    med = lambda s, n: s.rolling(n, min_periods=n).median().shift(1)  # noqa: E731 prior only
    t["y"] = t["win"] / med(t["win"], NORM)
    t["x_pre"] = t["pre"] / med(t["pre"], NORM)
    t["x_rel"] = t["pre"] / med(t["win"], NORM)
    t["x_prev"] = t["y"].shift(1)
    t["x_reg"] = med(t["win"], NORM) / med(t["win"], NORM_LONG)
    return t.dropna(subset=["y", "x_reg"])


def evaluate(t: pd.DataFrame, q: float) -> dict:
    big = (t["y"] >= q).to_numpy()
    out = {"n": len(t), "base": float(big.mean()),
           "room_spreads": float((t["win"] / t["spread"]).median()),
           "room_spreads_big": float((t["win"] / t["spread"])[big].median()),
           "weekday": t.assign(big=big).groupby(t.index.dayofweek)["big"].mean().round(3).to_dict()}
    for x in PREDICTORS:
        s = t[x].to_numpy()
        quint = pd.qcut(t[x], 5, labels=False, duplicates="drop").to_numpy()
        out[x] = {
            "auc": auc(s, big),
            "big_bottom_q": float(big[quint == 0].mean()),
            "big_top_q": float(big[quint == quint.max()].mean()),
            "auc_by_year": {int(yr): round(auc(g[x].to_numpy(), (g["y"] >= q).to_numpy()), 3)
                            for yr, g in t.groupby(t.index.year)},
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=2016)
    ap.add_argument("--end", type=int, default=2021)
    args = ap.parse_args()
    if args.end >= 2022:
        raise SystemExit("discovery only: 2022-2024 is confirmation, 2025+ is the holdout")

    report = {}
    for asset, sessions in PLAN.items():
        try:
            raw = load_cached(asset, args.start, args.end)
        except FileNotFoundError:
            print(f"{asset}: no cache")
            continue
        m1, _ = build_study_frame(raw, asset)
        for sess in sessions:
            t = session_table(m1, SESSIONS[sess]["tz"], SESSIONS[sess]["open"])
            q = float(t["y"].quantile(0.75))
            r = evaluate(t, q)
            report[f"{asset}:{sess}"] = r
            print(f"\n{asset} {sess}  n={r['n']}  big = y >= {q:.2f}  "
                  f"room {r['room_spreads']:.0f} spreads (big days {r['room_spreads_big']:.0f})")
            print(f"  P(big) by weekday Mo..Fr: {r['weekday']}")
            for x in PREDICTORS:
                e = r[x]
                print(f"  {x:7s} AUC {e['auc']:.3f}   P(big) bottom-q {e['big_bottom_q']:.0%} "
                      f"top-q {e['big_top_q']:.0%}   by year {e['auc_by_year']}")
    out = Path("artifacts/h11_move_forecast_quick.json")
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
