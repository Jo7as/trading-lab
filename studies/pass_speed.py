#!/usr/bin/env python3
"""Pass-speed surface: risk per trade × number of markets on one account.

Answers two questions: does raising the risk per trade make an evaluation pass
faster (and at what breach cost), and does running several markets on one
account speed it up by simply producing more trades per day?

  .venv/bin/python studies/pass_speed.py
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from trading_lab.prop_sim import FIRMS, evaluate, simulate_funded
from trading_lab.strategies.orb_2r import skipped_days
from trading_lab.strategies.orb_rejection_ifvg import OrbRejConfig, run_backtest

CACHE = "artifacts/cache/m1"
START, END = 2019, 2026
SPEC = {
    "US100": ("America/New_York", "nyse", "09:30", "16:00", "15:55", 0.25),
    "SP500": ("America/New_York", "nyse", "09:30", "16:00", "15:55", 0.25),
    "GER40": ("Europe/Berlin", "xetra", "09:00", "17:30", "17:25", 0.5),
}
RISKS = (0.25, 0.5, 0.75, 1.0, 1.5)


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


def daily_r(asset: str) -> pd.Series:
    tz, cal, op, cl, flat, tick = SPEC[asset]
    df = load_cached(asset)
    y = df.index.tz_convert("UTC").year
    sub = df[(y >= START) & (y <= END)]
    price = float(sub["close"].median())
    cfg = OrbRejConfig(asset=asset, asset_label=asset, tz=tz, calendar=cal, session_open=op,
                       session_close=cl, flat_time=flat,
                       entry_start=f"{int(op[:2]):02d}:{int(op[3:]) + 15:02d}",
                       mode="naive", entry_mode="next_open", tick_size=tick,
                       slippage_ticks_entry=1.0, slippage_ticks_stop=2.0,
                       min_stop_points=round(0.0001 * price, 3),
                       max_stop_points=round(0.0005 * price, 3))
    t = pd.DataFrame(run_backtest(sub, cfg)["trades"])
    return pd.Series(t["r"].astype(float).to_numpy(), index=pd.to_datetime(t["day"])).groupby(level=0).sum()


def main() -> None:
    skip = set().union(*[skipped_days("nyse", y) for y in range(START, END + 1)])
    cal = [d for d in pd.bdate_range(f"{START}-01-01", f"{END}-12-31") if d.date() not in skip]
    R = pd.DataFrame({a: daily_r(a) for a in SPEC}).reindex(cal).fillna(0.0)
    books = {"US100 only": R[["US100"]].sum(axis=1), "3 markets on one account": R.sum(axis=1)}
    print(f"{START}-{END} · markets {list(SPEC)} · one account, each market risks the % below per trade\n")

    hdr = f"{'firm':11s} {'risk':>5s} {'book':24s} {'pass%':>6s} {'median d':>9s} {'worst day $':>12s} {'limit $':>9s}"
    print(hdr)
    for firm_name in ("ftmo", "one_step", "lucid", "instant_25k"):
        firm = FIRMS[firm_name]
        for risk in RISKS:
            for book_name, daily_r_sum in books.items():
                stream = daily_r_sum.to_numpy() * risk / 100.0 * firm.account_size
                if firm.is_instant:
                    res = simulate_funded(stream, firm, runs=2000)
                    line = f"{firm_name:11s} {risk:5.2f} {book_name:24s} " \
                           f"{'surv ' + format(res.survival_rate*100, '.0f') + '%':>6s} " \
                           f"{'-':>9s} {stream.min():12,.0f} {firm.daily_limit_usd:9,.0f}"
                else:
                    res = evaluate(stream, firm, runs=2000)
                    line = f"{firm_name:11s} {risk:5.2f} {book_name:24s} " \
                           f"{res.pass_rate*100:5.0f}% {res.median_days:9.0f} " \
                           f"{stream.min():12,.0f} {firm.daily_limit_usd:9,.0f}"
                print(line)
        print()


if __name__ == "__main__":
    main()
