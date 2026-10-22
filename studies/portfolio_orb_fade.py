#!/usr/bin/env python3
"""Portfolio layer for the ORB fade: single markets vs equal-weight vs inverse-vol.

Same rule on US100, SP500 and GER40 (2019–2026, price-scaled stops, next-open
entry), combined with a capital split (weights sum to 1) and optionally scaled to
a volatility target. Reports the joint risk profile and the prop-firm outcome of
the portfolio, plus a block-bootstrap Monte Carlo of the *aligned* daily matrix
(so cross-market correlation is preserved).

  .venv/bin/python studies/portfolio_orb_fade.py
"""
from __future__ import annotations

import glob
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from trading_lab.prop_sim import FIRMS, evaluate, simulate_funded
from trading_lab.strategies.orb_rejection_ifvg import OrbRejConfig, run_backtest

CACHE = "artifacts/cache/m1"
SPEC = {
    "US100": dict(tz="America/New_York", calendar="nyse", open="09:30", close="16:00", flat="15:55", tick=0.25),
    "SP500": dict(tz="America/New_York", calendar="nyse", open="09:30", close="16:00", flat="15:55", tick=0.25),
    "GER40": dict(tz="Europe/Berlin", calendar="xetra", open="09:00", close="17:30", flat="17:25", tick=0.5),
}
START, END, RISK_PCT = 2019, 2026, 0.5


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
    df = load_cached(asset)
    y = df.index.tz_convert("UTC").year
    sub = df[(y >= START) & (y <= END)]
    s = SPEC[asset]
    price = float(sub["close"].median())
    cfg = OrbRejConfig(
        asset=asset, asset_label=asset, tz=s["tz"], calendar=s["calendar"],
        session_open=s["open"], session_close=s["close"], flat_time=s["flat"],
        entry_start=f"{int(s['open'][:2]):02d}:{int(s['open'][3:]) + 15:02d}",
        mode="naive", entry_mode="next_open", tick_size=s["tick"],
        slippage_ticks_entry=1.0, slippage_ticks_stop=2.0,
        min_stop_points=round(0.0001 * price, 3), max_stop_points=round(0.0005 * price, 3))
    t = run_backtest(sub, cfg)["trades"]
    ser = pd.Series([float(x["r"]) for x in t], index=pd.to_datetime([x["day"] for x in t]))
    return ser.groupby(level=0).sum()


def metrics(ret: pd.Series) -> dict:
    ann = float(ret.mean() * 252)
    vol = float(ret.std(ddof=1) * np.sqrt(252))
    eq = 100 + ret.cumsum()
    dd = float((eq - eq.cummax()).min())
    m = (1 + ret / 100.0).resample("ME").prod().sub(1) * 100     # ret is in percent
    neg = cur = worst = 0
    for x in m:
        cur = cur + 1 if x < 0 else 0
        worst = max(worst, cur)
    return dict(per_year=round(ann, 1), vol=round(vol, 1),
                sharpe=round(ann / vol, 2) if vol else 0.0, max_dd=round(dd, 1),
                months=len(m), pos_months=round(100 * (m > 0).mean(), 1),
                mean_month=round(float(m.mean()), 2), worst_month=round(float(m.min()), 2),
                neg_streak=worst)


def prop(stream_pct: pd.Series, account: float, name: str, target_frac_of_acct: float, instant: bool) -> str:
    stream = stream_pct / 100.0 * account
    firm = FIRMS[name]
    if instant:
        r = simulate_funded(stream.to_numpy(), firm, runs=3000)
        return f"survive {r.survival_rate*100:.0f}% | payout {r.payout_rate*100:.0f}% | mean ${r.mean_payout_usd:,.0f}"
    r = evaluate(stream.to_numpy(), firm, runs=3000)
    return f"pass {r.pass_rate*100:.0f}% | median {r.median_days:.0f}d"


def main() -> None:
    print(f"ORB fade portfolio {START}-{END} (risk {RISK_PCT}%/trade of each allocation)")
    from trading_lab.strategies.orb_2r import skipped_days
    skip = set().union(*[skipped_days("nyse", y) for y in range(START, END + 1)])
    cal = [d for d in pd.bdate_range(f"{START}-01-01", f"{END}-12-31") if d.date() not in skip]
    R = pd.DataFrame({a: daily_r(a) for a in SPEC}).reindex(cal).fillna(0.0)
    R.index = pd.to_datetime(R.index)
    print(f"days with >=1 trade: {int((R.abs().sum(axis=1) > 0).sum())} of {len(R)} trading days")
    print("daily-R correlation:\n" + R[R.abs().sum(axis=1) > 0].corr().round(2).to_string())

    vol = R.std(ddof=1).replace(0, np.nan)
    w_eq = pd.Series(1 / len(R.columns), index=R.columns)
    w_iv = (1 / vol)
    w_iv = (w_iv / w_iv.sum()).fillna(0.0)

    variants = {a: R[a] * RISK_PCT for a in R.columns}          # single market, 0.5%
    variants["equal_weight"] = (R @ w_eq) * RISK_PCT
    variants["inverse_vol"] = (R @ w_iv) * RISK_PCT
    # vol-target the inverse-vol book to 12% annual (cap 1.5x)
    iv = variants["inverse_vol"]
    scale = min(1.5, 12.0 / (iv.std(ddof=1) * np.sqrt(252))) if iv.std(ddof=1) else 1.0
    variants["inv_vol_12pct"] = iv * scale

    print(f"\nweights: equal {dict(w_eq.round(2))}  inverse-vol {dict(w_iv.round(2))}  "
          f"(12%-vol scale {scale:.2f}x)\n")
    head = f"{'variant':14s} {'%/yr':>7s} {'vol':>6s} {'Sharpe':>7s} {'maxDD':>7s} {'posMon':>7s} {'negStreak':>9s}"
    print(head)
    for name, ret in variants.items():
        m = metrics(ret)
        print(f"{name:14s} {m['per_year']:7.1f} {m['vol']:6.1f} {m['sharpe']:7.2f} "
              f"{m['max_dd']:7.1f} {m['pos_months']:6.1f}% {m['neg_streak']:9d}")

    print("\nprop on the vol-targeted inverse-vol portfolio (daily % → $ on the account):")
    for acct, name, inst in ((100_000.0, "ftmo", False), (25_000.0, "lucid", False),
                             (25_000.0, "instant_25k", True), (100_000.0, "instant_100k", True)):
        print(f"  {name:13s} ${acct:,.0f}  {prop(variants['inv_vol_12pct'], acct, name, 0.0, inst)}")

    # ── A6: block bootstrap of the aligned daily matrix ─────────────────
    print("\nMonte Carlo (block bootstrap of days, 20-day blocks, 3000 runs):")
    rng = np.random.default_rng(7)
    arr = variants["inv_vol_12pct"].to_numpy()
    n = len(arr)
    nb = int(np.ceil(252 / 20)) + 1
    year_rets, dds = [], []
    for _ in range(3000):
        starts = rng.integers(0, n, size=nb)
        path = np.concatenate([arr[s:s + 20] for s in starts])[:252]
        eq = np.cumsum(path)
        year_rets.append(eq[-1])
        dds.append((eq - np.maximum.accumulate(eq)).min())
    yr = np.array(year_rets)
    dd = np.array(dds)
    print(f"  annual return: mean {yr.mean():+.1f}%  p5 {np.percentile(yr,5):+.1f}%  "
          f"P(<0) {100*(yr<0).mean():.1f}%")
    print(f"  max drawdown : median {np.median(dd):.1f}%  p95 {np.percentile(dd,5):.1f}%  "
          f"P(DD>20%) {100*(dd<-20).mean():.1f}%  P(DD>30%) {100*(dd<-30).mean():.1f}%")


if __name__ == "__main__":
    main()
