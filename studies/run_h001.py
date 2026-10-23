"""Test of the registered card H-001 (noise-area progress at high volatility).

Condition (card): at the first hourly checkpoint whose |progress| >= 1.0 sigma_m
(progress = move from the gap-corrected session anchor, sigma_m = mean |same-minute
move| of the last 14 days), take the position in the direction of the progress and
hold to the session close. Variants pre-registered in the card: `basis` (all days)
and `nur_hohe_vola` (20-day realised vol in the top third of the last 250 days).

Direct clock computation on the audit-correct frame (data_audit.build_study_frame),
gates as in the protocol (median effect, sign-test z >= 3, effect >= 2x cost,
placebo, dose-response).

  .venv/bin/python studies/run_h001.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "studies")
from pass_speed import load_cached  # noqa: E402

from trading_lab.data_audit import SESSIONS, build_study_frame  # noqa: E402

CLOCK = {"US100": (570, 960), "US30": (570, 960), "SP500": (570, 960), "GER40": (540, 1050)}
CHECK_EVERY = 60


def matrices(df: pd.DataFrame, asset: str):
    tz = SESSIONS[asset]["tz"]
    loc = df.index.tz_convert(tz)
    mins = (loc.hour * 60 + loc.minute).to_numpy()
    mi = pd.MultiIndex.from_arrays([loc.date, mins], names=["d", "m"])
    close = pd.Series(df["close"].to_numpy(), index=mi).unstack("m")
    spread = pd.Series(((df["ask_close"] - df["close"]) / df["close"]).to_numpy(), index=mi).unstack("m")
    return close, spread, loc.date


def test(close: pd.DataFrame, spread: pd.DataFrame, open_m: int, close_m: int,
         sel: np.ndarray) -> dict:
    """First hourly checkpoint with |progress| >= 1 sigma_m, hold to the close."""
    col_of = {m: i for i, m in enumerate(close.columns)}
    C = close.to_numpy(float)
    S = spread.to_numpy(float)
    prev = pd.DataFrame(C).ffill(axis=1).iloc[:, -1].shift(1).to_numpy()
    o = close[open_m].to_numpy()
    upper = np.maximum(o, prev)[:, None]
    lower = np.minimum(o, prev)[:, None]
    anchor = np.where(C >= o[:, None], upper, lower)
    prog = (C - anchor) / anchor
    rel = np.abs(C / o[:, None] - 1.0)
    sigma = pd.DataFrame(rel).rolling(14, min_periods=8).mean().shift(1).to_numpy()

    last = col_of[close_m - 5]
    cps = [m for m in range(open_m + CHECK_EVERY, close_m - 5, CHECK_EVERY) if m in col_of]
    cols = [col_of[m] for m in cps]
    cond = (np.abs(prog[:, cols]) >= sigma[:, cols]) & (sigma[:, cols] > 0) & np.isfinite(prog[:, cols])
    has = cond.any(axis=1)
    days = np.flatnonzero(has & sel)
    pick = np.argmax(cond, axis=1)
    cp = np.array(cols)[pick[days]]
    prog_star = prog[days, cp]
    sigma_star = sigma[days, cp]
    px_cp, px_end = C[days, cp], C[days, last]
    sp_cp, sp_end = S[days, cp], S[days, last]
    d = np.sign(prog_star)
    move = d * (px_end - px_cp) / px_cp                 # signed forward return, in price %
    return {"n": len(days), "move": move, "cost": sp_cp + sp_end, "mag": np.abs(prog_star) / sigma_star, "d": d}


def report(label: str, r: dict, seed: int = 7) -> str:
    ok = np.isfinite(r["move"]) & np.isfinite(r["cost"]) & (r["move"] != 0)
    r = {"n": int(ok.sum()), "move": r["move"][ok], "cost": r["cost"][ok],
         "mag": r["mag"][ok], "d": r["d"][ok]}
    n = r["n"]
    if n < 30:
        return f"  {label:22s} n={n}: too few events"
    signed = r["move"]
    med = float(np.median(signed))
    n_pos = int((signed > 0).sum())
    z = (n_pos - n / 2) / np.sqrt(n / 4)
    cost = float(np.median(r["cost"])) * 100
    eff = abs(med) * 100 / cost if cost else float("inf")
    rng = np.random.default_rng(seed)
    placebo = float(np.mean(rng.choice(r["d"], n) * rng.permutation(np.abs(signed))))
    q = pd.qcut(pd.Series(r["mag"]).rank(method="first"), 5, labels=False)
    quint = [round(float(np.mean(signed[q == k])) * 100, 3) for k in range(5)]
    slope = np.polyfit(range(5), np.array(quint) * np.sign(med or 1), 1)[0]
    verdict = ("PASS" if (z >= 3 and eff >= 2 and abs(placebo) * 100 < 0.5 * abs(med * 100) and slope > 0)
               else "FAIL(sign)" if z < 3 else "FAIL(cost)" if eff < 2
               else "FAIL(dose)" if slope <= 0 else "FAIL(placebo)")
    return (f"  {label:22s} n={n:5d}  med {med*100:+.3f}%  z={z:5.2f}  hit {n_pos/n:4.0%}  "
            f"cost {cost:.4f}%  eff/cost {eff:5.1f}  placebo {placebo*100:+.3f}%  "
            f"quint {quint}  {verdict}")


def main() -> None:
    print("### H-001 (registered): noise-area progress at high volatility, hold to close")
    for asset, (open_m, close_m) in CLOCK.items():
        df, info = build_study_frame(load_cached(asset), asset)
        close, spread, _ = matrices(df, asset)
        o = close[open_m]
        ret = (close.ffill(axis=1).iloc[:, -1] / o - 1)
        vol20 = ret.rolling(20, min_periods=10).std()
        pct = vol20.rolling(250, min_periods=60).rank(pct=True)
        high = (pct >= 2 / 3).to_numpy()
        base = ret.notna().to_numpy()
        print(f"\n{asset}  days={int(base.sum())}  (dropped rows: {info['dropped']}, "
              f"thin days: {info['thin_days']})")
        print(report("basis (alle Tage)", test(close, spread, open_m, close_m, base)))
        print(report("nur_hohe_vola", test(close, spread, open_m, close_m, base & high)))


if __name__ == "__main__":
    main()
