"""Completeness check: the (rejected) ORB-fade rule across all cached markets.

Engine is fixed (correct session slices). Stops and costs are scaled to each
instrument's median 1-minute range so the comparison is instrument-agnostic.

  .venv/bin/python studies/multimarket_check.py
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from trading_lab.strategies.orb_rejection_ifvg import OrbRejConfig, run_backtest

CACHE = "artifacts/cache/m1"
START, END = 2019, 2026
# asset: tz, calendar, open, close, flat  (session approximations for the CFDs)
SPEC = {
    "US100": ("America/New_York", "nyse", "09:30", "16:00", "15:55"),
    "SP500": ("America/New_York", "nyse", "09:30", "16:00", "15:55"),
    "US30": ("America/New_York", "nyse", "09:30", "16:00", "15:55"),
    "GER40": ("Europe/Berlin", "xetra", "09:00", "17:30", "17:25"),
    "UK100": ("Europe/London", None, "08:00", "16:30", "16:25"),
    "XAUUSD": ("America/New_York", "nyse", "08:20", "17:00", "16:55"),
    "XAGUSD": ("America/New_York", "nyse", "08:20", "17:00", "16:55"),
    "COPPER": ("America/New_York", "nyse", "08:00", "13:00", "12:55"),
    "BRENT": ("America/New_York", "nyse", "09:00", "14:30", "14:25"),
    "UST10Y": ("America/New_York", "nyse", "08:20", "15:00", "14:55"),
    "JP225": ("Asia/Tokyo", "tse", "09:00", "15:00", "14:55"),
}


def load_cached(asset: str) -> pd.DataFrame:
    sides = {}
    for side in ("bid", "ask"):
        files = [f for f in sorted(glob.glob(f"{CACHE}/{asset}_1m_{side}_*.pkl"))
                 if f.split(f"1m_{side}_")[1][:4].isdigit()
                 and START <= int(f.split(f"1m_{side}_")[1][:4]) <= END]
        if not files:
            raise FileNotFoundError(asset)
        s = pd.concat([pd.read_pickle(f) for f in files])
        sides[side] = s[~s.index.duplicated(keep="last")].sort_index()
    bid, ask = sides["bid"], sides["ask"]
    out = bid.join(ask[["open", "high", "low", "close"]].add_prefix("ask_"), how="inner")
    for c in ("open", "high", "low", "close"):
        out[f"ask_{c}"] = np.maximum(out[f"ask_{c}"], out[c])
    return out


def main() -> None:
    print(f"{'market':8s} {'n':>5s} {'exp R':>7s} {'WR':>6s} {'PF':>6s} {'negYears':>9s} {'1m range':>9s}")
    rets = {}
    for asset, (tz, cal, op, cl, flat) in SPEC.items():
        try:
            df = load_cached(asset)
        except FileNotFoundError:
            print(f"{asset:8s} (no cache)")
            continue
        y = df.index.tz_convert("UTC").year
        sub = df[(y >= START) & (y <= END)]
        if sub.empty:
            continue
        rng1 = float((sub["high"] - sub["low"]).median())
        cfg = OrbRejConfig(
            asset=asset, asset_label=asset, tz=tz, calendar=cal or "nyse",
            session_open=op, session_close=cl, flat_time=flat,
            entry_start=f"{int(op[:2]):02d}:{int(op[3:]) + 15:02d}", mode="naive",
            entry_mode="next_open", tick_size=round(0.25 * rng1, 5),
            slippage_ticks_entry=1.0, slippage_ticks_stop=2.0,
            min_stop_points=round(2 * rng1, 5), max_stop_points=round(10 * rng1, 5))
        tr = pd.DataFrame(run_backtest(sub, cfg)["trades"])
        if tr.empty:
            print(f"{asset:8s} no trades")
            continue
        tr["r"] = tr["r"].astype(float)
        tr["year"] = tr["day"].str[:4].astype(int)
        rs = tr["r"]
        pf = rs[rs > 0].sum() / abs(rs[rs < 0].sum()) if (rs < 0).any() else float("inf")
        neg = sum(1 for _, g in tr.groupby("year") if g["r"].mean() < 0)
        print(f"{asset:8s} {len(rs):5d} {rs.mean():+7.3f} {100*(rs>0).mean():5.1f}% {pf:6.2f} "
              f"{neg:9d} {rng1:9.4f}")
        rets[asset] = pd.Series(rs.to_numpy(), index=pd.to_datetime(tr["day"])).groupby(level=0).sum()

    if len(rets) > 1:
        M = pd.DataFrame(rets).fillna(0.0)
        corr = M.corr()
        mask = np.triu(np.ones(corr.shape), 1).astype(bool)
        print("\ndaily-R correlation — most negative pairs:")
        print(corr.where(mask).stack().sort_values().head(5).round(2).to_string())


if __name__ == "__main__":
    main()
