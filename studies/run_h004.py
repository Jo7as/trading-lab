"""H-004: Asian-range breakout -> continuation into London/NY (EURUSD, USDJPY).

Direct clock computation on the audit-correct frame (1h bars), entry at the next
bar's open with adverse slippage, exit at the London/NY session end (17:00 NY) or
a time stop. Reports gross/net (real bid/ask), a random-breakout placebo, the
dose-response over breakout width, per-year stability — and the *goal gate*
(annualised net return in the 27-60 % band that 2-4 %/month requires).

  .venv/bin/python studies/run_h004.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "studies")
from pass_speed import load_cached  # noqa: E402

from trading_lab.data_audit import build_study_frame  # noqa: E402

PAIRS = ("EURUSD", "USDJPY")
SLIP = 0.00002
ASIA = (0, 180)          # 00:00-02:59 New York
LONDON = (180, 660)      # 03:00-10:59 NY entry window
NY_END = 1020            # 17:00 NY exit


def bars_for(asset: str) -> tuple[pd.DataFrame, pd.DatetimeIndex, np.ndarray]:
    df, _ = build_study_frame(load_cached(asset), asset)
    b = df.resample("60min").agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "ask_close": "last"}).dropna()
    loc = b.index.tz_convert("America/New_York")
    return b, loc, (loc.hour * 60 + loc.minute).to_numpy()


def run(asset: str, *, random_entry: bool = False, seed: int = 7) -> pd.DataFrame:
    b, loc, mins = bars_for(asset)
    mid = ((b["close"] + b["ask_close"]) / 2).to_numpy()
    half = ((b["ask_close"] - b["close"]) / 2).to_numpy()
    day = loc.date
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.unique(day):
        m = day == d
        asia = np.flatnonzero(m & (mins >= ASIA[0]) & (mins < ASIA[1]))
        lon = np.flatnonzero(m & (mins >= LONDON[0]) & (mins < LONDON[1]))
        end = np.flatnonzero(m & (mins >= NY_END - 60) & (mins <= NY_END))
        if len(asia) < 2 or len(lon) < 3 or len(end) < 1:
            continue
        hi, lo = float(mid[asia].max()), float(mid[asia].min())
        width = hi - lo
        i_end = int(end[-1])
        cand = None
        for i in lon:                                   # first close outside the range
            if mid[i] > hi:
                cand = (i, 1)
                break
            if mid[i] < lo:
                cand = (i, -1)
                break
        if random_entry:                                # placebo: random bar + direction
            i = int(lon[rng.integers(0, len(lon))])
            cand = (i, 1 if rng.random() < 0.5 else -1)
        if cand is None or cand[0] + 1 >= len(b):
            continue
        i_sig, direction = cand
        i_in = i_sig + 1                                # fill at the next open
        px_in = float(b["open"].iloc[i_in]) * (1 + SLIP * direction)
        i_out = min(i_end, i_in + 3)                    # hold to the session end (≤3 bars here)
        px_out = float(b["close"].iloc[i_out]) * (1 - SLIP * direction)
        move = direction * (px_out - px_in) / px_in
        cost = (half[i_in] + half[i_out]) / px_in
        rows.append({"day": str(d), "direction": direction, "move": move, "cost": cost,
                     "width_atr": width / float(b["high"].iloc[asia].max() - b["low"].iloc[asia].min()),
                     "year": pd.Timestamp(d).year})
    return pd.DataFrame(rows)


def report(label: str, t: pd.DataFrame, weeks: float) -> str:
    if len(t) < 30:
        return f"  {label:22s} n={len(t)}: too few"
    r = t["move"].to_numpy()
    med = float(np.median(r))
    z = (int((r > 0).sum()) - len(r) / 2) / np.sqrt(len(r) / 4)
    cost = float(t["cost"].median())
    eff = abs(med) / cost if cost else float("inf")
    q = pd.qcut(pd.Series(t["width_atr"]).rank(method="first"), 5, labels=False)
    quint = [round(float(np.mean(r[q == k])) * 100, 3) for k in range(5)]
    slope = np.polyfit(range(5), np.array(quint) * np.sign(med or 1), 1)[0]
    pos_years = sum(1 for _, g in t.groupby("year") if g["move"].sum() - g["cost"].sum() > 0)
    net = float((t["move"] - t["cost"]).sum())
    gross = float(t["move"].sum())
    verdict = ("PASS" if (z >= 3 and eff >= 2 and slope > 0) else
               "FAIL(sign)" if z < 3 else "FAIL(cost)" if eff < 2 else "FAIL(dose)")
    return (f"  {label:22s} n={len(t):5d} ({len(t)/weeks:4.1f}/w) gross {gross*100:+7.2f}% "
            f"net {net*100:+7.2f}%  med {med*100:+.3f}% z={z:5.2f} cost {cost*100:.4f}% "
            f"eff/cost {eff:4.1f} quint {quint} posY {pos_years}/{t['year'].nunique()}  {verdict}")


def main() -> None:
    print("### H-004 — Asian-range breakout -> London/NY continuation (1h, audit-correct)")
    for a in PAIRS:
        b, loc, _ = bars_for(a)
        weeks = (b.index[-1] - b.index[0]).days / 7
        t = run(a)
        print(f"\n{a}  (1h bars, {weeks:.0f} weeks)")
        print(report("basis (long+short)", t, weeks))
        print(report("long only", t[t["direction"] > 0], weeks))
        print(report("short only", t[t["direction"] < 0], weeks))
        print(report("placebo (random brk)", run(a, random_entry=True), weeks))
        net_yr = float((t["move"] - t["cost"]).sum()) / ((b.index[-1] - b.index[0]).days / 365.25)
        print(f"  goal gate: net {net_yr*100:+.2f}%/yr  (needed band: +27% to +60%/yr) -> "
              f"{'PASS' if net_yr >= 0.27 else 'FAIL'}")


if __name__ == "__main__":
    main()
