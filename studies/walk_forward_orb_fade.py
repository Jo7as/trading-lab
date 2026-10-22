#!/usr/bin/env python3
"""A5: walk-forward / rolling stability for the frozen ORB-fade rule.

Parameters are frozen (no re-tuning), so "walk-forward" here means: evaluate the
same rule out-of-sample across calendar years and rolling 12-month windows, per
market and for the inverse-vol portfolio, and look for any era or window where it
loses. Nothing is optimised on train windows.

  .venv/bin/python studies/walk_forward_orb_fade.py
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from trading_lab.strategies.orb_2r import skipped_days
from trading_lab.strategies.orb_rejection_ifvg import OrbRejConfig, run_backtest

CACHE = "artifacts/cache/m1"
START, END, RISK_PCT = 2019, 2026, 0.5
# session per market: tz, calendar, open, close, flat, tick
SPEC = {
    "US100": ("America/New_York", "nyse", "09:30", "16:00", "15:55", 0.25),
    "SP500": ("America/New_York", "nyse", "09:30", "16:00", "15:55", 0.25),
    "GER40": ("Europe/Berlin", "xetra", "09:00", "17:30", "17:25", 0.5),
    "XAUUSD": ("America/New_York", "nyse", "08:20", "17:00", "16:55", 0.05),
    "JP225": ("Asia/Tokyo", "tse", "09:00", "15:00", "14:55", 1.0),
}


def load_cached(asset: str) -> pd.DataFrame:
    sides = {}
    for side in ("bid", "ask"):
        files = [f for f in sorted(glob.glob(f"{CACHE}/{asset}_1m_{side}_*.pkl"))
                 if f.split(f"1m_{side}_")[1][:4].isdigit()
                 and START <= int(f.split(f"1m_{side}_")[1][:4]) <= END]
        s = pd.concat([pd.read_pickle(f) for f in files])
        sides[side] = s[~s.index.duplicated(keep="last")].sort_index()
    bid, ask = sides["bid"], sides["ask"]
    out = bid.join(ask[["open", "high", "low", "close"]].add_prefix("ask_"), how="inner")
    for c in ("open", "high", "low", "close"):
        out[f"ask_{c}"] = np.maximum(out[f"ask_{c}"], out[c])
    return out


def trades_for(asset: str) -> pd.DataFrame:
    tz, cal, op, cl, flat, tick = SPEC[asset]
    df = load_cached(asset)
    y = df.index.tz_convert("UTC").year
    sub = df[(y >= START) & (y <= END)]
    price = float(sub["close"].median())
    cfg = OrbRejConfig(
        asset=asset, asset_label=asset, tz=tz, calendar=cal, session_open=op, session_close=cl,
        flat_time=flat, entry_start=f"{int(op[:2]):02d}:{int(op[3:]) + 15:02d}",
        mode="naive", entry_mode="next_open", tick_size=tick,
        slippage_ticks_entry=1.0, slippage_ticks_stop=2.0,
        min_stop_points=round(0.0001 * price, 3), max_stop_points=round(0.0005 * price, 3))
    t = pd.DataFrame(run_backtest(sub, cfg)["trades"])[["day", "r"]]
    t["r"] = t["r"].astype(float)
    t["year"] = t["day"].str[:4].astype(int)
    return t


def main() -> None:
    skip = set().union(*[skipped_days("nyse", y) for y in range(START, END + 1)])
    cal = [d for d in pd.bdate_range(f"{START}-01-01", f"{END}-12-31") if d.date() not in skip]

    print("### per market × calendar year (expectancy R / PF)")
    frames = {}
    for a in SPEC:
        try:
            t = trades_for(a)
        except FileNotFoundError:
            print(f"  {a}: no cache, skipped")
            continue
        frames[a] = t
        row = []
        for y in range(START, END + 1):
            g = t[t.year == y]["r"]
            if not len(g):
                row.append(f"{y}:-")
                continue
            pf = g[g > 0].sum() / abs(g[g < 0].sum()) if (g < 0).any() else float("inf")
            row.append(f"{y}:{g.mean():+.2f}/{pf:.1f}")
        neg = sum(1 for y in range(START, END + 1)
                  if len(t[t.year == y]) and t[t.year == y]["r"].mean() < 0)
        print(f"  {a:7s} " + "  ".join(row) + f"   negative years: {neg}")

    # inverse-vol portfolio on the aligned calendar
    R = pd.DataFrame({a: frames[a].assign(d=pd.to_datetime(frames[a]["day"]))
                      .groupby("d")["r"].sum() for a in frames}).reindex(cal).fillna(0.0)

    # pre-committed selection: keep a market only if positive in 2019-2022 (n>=50),
    # then judge it on 2023-2026. No peeking at the test window.
    is_end, cutoff = 2022, 50
    sel = [a for a in frames if (n := len(g := frames[a][frames[a].year <= is_end]["r"])) >= cutoff
           and g.mean() > 0]
    dropped = [a for a in frames if a not in sel]
    print(f"\n### selection rule: positive mean R in 2019-{is_end} (n>={cutoff})")
    print(f"  kept: {sel}")
    print(f"  dropped: {dropped}")

    def book(markets, sub_years=None):
        r = R[markets]
        if sub_years:
            r = r[(r.index.year >= sub_years[0]) & (r.index.year <= sub_years[1])]
        vol = r.std(ddof=1).replace(0, np.nan)
        w = (1 / vol)
        w = (w / w.sum()).fillna(0.0)
        return (r @ w) * RISK_PCT, w

    port, w = book(sel)
    print(f"\n### portfolio (inverse vol {dict(w.round(2))})")
    print("  per year %:")
    yrs = (1 + port / 100).groupby(port.index.year).prod().sub(1) * 100
    print("   " + "  ".join(f"{y}:{v:+.1f}" for y, v in yrs.items()))
    oos, _ = book(sel, (2023, 2026))
    yrs_oos = len(set(oos.index.year))
    print(f"  OOS 2023-2026: total {(1 + oos / 100).prod() - 1:+.1%} | "
          f"compounded {((1 + oos / 100).prod())**(1/yrs_oos) - 1:+.1%}/yr | "
          f"linear-sum {oos.mean()*252:+.1f}%/yr | vol {oos.std(ddof=1)*np.sqrt(252):.1f}% | "
          f"Sharpe {oos.mean()/oos.std(ddof=1)*np.sqrt(252):.2f}")

    roll = 252
    rr = port.rolling(roll).apply(lambda x: (1 + x / 100).prod() - 1, raw=True) * 100
    rr = rr.dropna()
    dd = port.rolling(roll).apply(
        lambda x: float(((100 + np.cumsum(x)) - np.maximum.accumulate(100 + np.cumsum(x))).min()),
        raw=True).dropna()
    sharpe = (port.rolling(roll).mean() / port.rolling(roll).std()) * np.sqrt(252)
    print(f"\n  rolling {roll}-day windows ({len(rr)}):")
    print(f"    return %: min {rr.min():+.1f}  p5 {np.percentile(rr,5):+.1f}  median {rr.median():+.1f}  "
          f"max {rr.max():+.1f}   %positive {100*(rr>0).mean():.1f}")
    print(f"    Sharpe : min {sharpe.min():.2f}  p5 {np.percentile(sharpe.dropna(),5):.2f}  "
          f"median {sharpe.median():.2f}")
    print(f"    worst window drawdown: {dd.min():.1f}%")


if __name__ == "__main__":
    main()
