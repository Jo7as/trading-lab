#!/usr/bin/env python3
"""H11 quick measurement (no spec yet): "target first" after the NY opening impulse.

The idea from the user's chart (NQ 5m, 29 Oct 2026): the first 30 minutes of New York
draw a range 0 / 0.5 / 1, an untouched 1h swing high (or low) sits beyond it as the
target, and two *different* paths are expected to lead to the same target:

  deep     price retraces to the 0.5 of the range first  -> buy limit at 0.5, stop 0
  shallow  price breaks the range extreme first ("15m continuation")
                                                       -> buy stop at 1, stop 0.5

The question is not "how often is the target hit" but whether the **real** 1h level is
hit more often than a level at the same distance would be. So every group is compared
with a shuffle placebo: the target distances (in range units) are permuted across the
days of the same group, keeping the day type and the path but breaking the link to
the actual level. A real "liquidity magnet" beats its own shuffle.

Space and clock (docs/TIME_CONVENTIONS.md): mid prices, America/New_York, range
09:30-09:59, entries 10:00-11:59, flat 15:55. Stop wins ties, and the target is only
checked from the bar after the entry (the entry bar's high may predate the fill).
Cost: spread at the entry bar x ``round_trip_spread(entry, exit)`` (limit/stop entry,
tp/stop/time exit). Frame: ``data_audit.build_study_frame``.

Data: 2016-2021 only (discovery). 2022-2024 stays untouched for confirmation; the
holdout (2025+) is never opened here.

  uv run python studies/h11_target_first_quick.py
  uv run python studies/h11_target_first_quick.py --asset US100 --shuffles 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from studies.cost_gate import load_cached  # noqa: E402
from trading_lab.cost_gate import round_trip_spread  # noqa: E402
from trading_lab.data_audit import build_study_frame  # noqa: E402

TZ = "America/New_York"
OR_START, OR_END = "09:30", "10:00"
ENTRY_END = "12:00"
FLAT = "15:55"
PIVOT_SIDE = 2          # 1h fractal: higher than 2 bars on each side
MIN_OR_BARS = 25


@dataclass
class Trade:
    day: str
    year: int
    direction: str      # "bull" (low before high in the range) / "bear"
    path: str           # "deep" / "shallow"
    entry: float        # all prices in the bull-oriented space (bear is mirrored)
    stop: float
    target: float
    rng: float          # range size H - L
    extreme: float      # H (bull-oriented)
    entry_i: int        # position of the entry bar in the study frame
    spread: float       # full spread at the entry bar, price units
    outcome: str        # "tp" / "stop" / "time"
    r_gross: float
    cost_r: float
    r_net: float


# ---------------------------------------------------------------------------
# core (pure functions on arrays, bull-oriented; tested in tests/test_h11_target_first.py)
# ---------------------------------------------------------------------------

def first_passage(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, entry_i: int,
                  entry: float, stop: float, target: float) -> tuple[str, float]:
    """Walk the path from the entry bar: stop counts from the entry bar, the target from
    the next one; stop wins a tie. Returns (outcome, exit price); time exit at the last
    close."""
    n = len(hi)
    for j in range(entry_i, n):
        if lo[j] <= stop:
            return "stop", stop
        if j > entry_i and hi[j] >= target:
            return "tp", target
    return "time", float(cl[n - 1])


def classify_path(hi: np.ndarray, lo: np.ndarray, op: np.ndarray, H: float, L: float
                  ) -> tuple[str, int, float, float] | None:
    """First event in the entry window: touch of the 0.5 (deep) or break of H (shallow).

    Returns (path, bar index, entry price, stop) or None (no event, or both in one bar).
    A gapped bar fills at its open when that is better than the order price."""
    half = (H + L) / 2.0
    for j in range(len(hi)):
        touch, brk = lo[j] <= half, hi[j] > H
        if touch and brk:
            return None
        if touch:
            return "deep", j, min(half, op[j]), L
        if brk:
            return "shallow", j, max(H, op[j]), half
    return None


def hourly_pivot_highs(hi: np.ndarray, side: int = PIVOT_SIDE) -> list[int]:
    """Indices i with hi[i] strictly above the ``side`` bars on each side."""
    out = []
    for i in range(side, len(hi) - side):
        if hi[i] > hi[i - side:i].max() and hi[i] > hi[i + 1:i + side + 1].max():
            out.append(i)
    return out


def nearest_untouched_level(h1_hi: np.ndarray, after_max: np.ndarray, above: float,
                            side: int = PIVOT_SIDE) -> float | None:
    """Nearest 1h pivot high above ``above`` that no later bar (up to the end of the
    range) has traded through. ``after_max[i]`` = max high after hourly bar i."""
    best = None
    for i in hourly_pivot_highs(h1_hi, side):
        lvl = h1_hi[i]
        if lvl > above and after_max[i] < lvl and (best is None or lvl < best):
            best = lvl
    return best


# ---------------------------------------------------------------------------
# per-day driver
# ---------------------------------------------------------------------------

def _at(day: pd.Timestamp, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day.date()} {hhmm}", tz=TZ).tz_convert("UTC")


def collect(m1: pd.DataFrame) -> tuple[list[Trade], dict]:
    mid = pd.DataFrame({k: (m1[k] + m1[f"ask_{k}"]) / 2 for k in ("open", "high", "low", "close")},
                       index=m1.index)
    spread = (m1["ask_close"] - m1["close"]).to_numpy()
    idx = mid.index
    op, hi, lo, cl = (mid[c].to_numpy() for c in ("open", "high", "low", "close"))
    days = pd.DatetimeIndex(sorted(set(idx.tz_convert(TZ).normalize().tz_localize(None))))
    days = days[days.dayofweek < 5]
    pos = lambda t: int(idx.searchsorted(t))  # noqa: E731

    trades: list[Trade] = []
    stats = {"days": 0, "or_ok": 0, "with_level": 0, "no_event": 0}
    prev = None
    for day in days:
        stats["days"] += 1
        a, b = pos(_at(day, OR_START)), pos(_at(day, OR_END))
        if b - a < MIN_OR_BARS or prev is None:
            prev = day
            continue
        stats["or_ok"] += 1
        H, L = hi[a:b].max(), lo[a:b].min()
        tH, tL = int(np.argmax(hi[a:b])), int(np.argmin(lo[a:b]))
        bull = tL < tH
        sgn = 1.0 if bull else -1.0
        # bull-oriented arrays: a bear day is mirrored (price -> -price)
        h_o = hi if bull else -lo
        l_o = lo if bull else -hi
        o_o = sgn * op
        c_o = sgn * cl
        Ho, Lo = (H, L) if bull else (-L, -H)

        # 1h bars over the previous session day 09:30 -> today 09:30, aligned at :30
        la, lb = pos(_at(prev, OR_START)), a
        prev = day
        seg = pd.DataFrame({"h": h_o[la:lb]}, index=idx[la:lb])
        if len(seg) < 60 * (2 * PIVOT_SIDE + 1):
            continue
        h1 = seg["h"].resample("60min", origin=_at(day, OR_START)).max().dropna()
        h1_hi = h1.to_numpy()
        # max after hourly bar i: later hourly bars, then the range itself
        from_i = np.maximum.accumulate(h1_hi[::-1])[::-1]       # max of h1_hi[i:]
        after_max = np.maximum(np.r_[from_i[1:], -np.inf], Ho)  # later bars + the range
        target = nearest_untouched_level(h1_hi, after_max, Ho)
        if target is None:
            continue
        stats["with_level"] += 1

        c, d, e = b, pos(_at(day, ENTRY_END)), pos(_at(day, FLAT))
        ev = classify_path(h_o[c:d], l_o[c:d], o_o[c:d], Ho, Lo)
        if ev is None:
            stats["no_event"] += 1
            continue
        path, j, entry, stop = ev
        ei = c + j
        if not (stop < entry < target):
            continue
        outcome, exit_px = first_passage(h_o[ei:e], l_o[ei:e], c_o[ei:e], 0, entry, stop, target)
        risk = entry - stop
        rt = round_trip_spread("limit" if path == "deep" else "stop", outcome)
        cost_r = spread[ei] * rt / risk
        r_g = (exit_px - entry) / risk
        trades.append(Trade(str(day.date()), day.year, "bull" if bull else "bear", path,
                            entry, stop, target, Ho - Lo, Ho, ei, float(spread[ei]), outcome,
                            r_g, cost_r, r_g - cost_r))
    return trades, stats


# ---------------------------------------------------------------------------
# shuffle placebo: same day, same path, a target at another day's distance
# ---------------------------------------------------------------------------

def _paths(m1: pd.DataFrame, trades: list[Trade]) -> list[tuple[np.ndarray, int, float]]:
    """Per trade: running max of the oriented high after the entry bar, the index of
    the stop hit (or len), and the time-exit close - enough to re-score any target."""
    mid_h = ((m1["high"] + m1["ask_high"]) / 2).to_numpy()
    mid_l = ((m1["low"] + m1["ask_low"]) / 2).to_numpy()
    mid_c = ((m1["close"] + m1["ask_close"]) / 2).to_numpy()
    idx = m1.index
    out = []
    for t in trades:
        e = int(idx.searchsorted(_at(pd.Timestamp(t.day), FLAT)))
        bull = t.direction == "bull"
        hs = mid_h[t.entry_i:e] if bull else -mid_l[t.entry_i:e]
        ls = mid_l[t.entry_i:e] if bull else -mid_h[t.entry_i:e]
        c = mid_c[t.entry_i:e] if bull else -mid_c[t.entry_i:e]
        stop_hit = np.flatnonzero(ls <= t.stop)
        si = int(stop_hit[0]) if len(stop_hit) else len(ls)
        run = np.maximum.accumulate(np.r_[-np.inf, hs[1:]])  # target only from the next bar
        out.append((run, si, float(c[-1])))
    return out


def rescore(run: np.ndarray, si: int, last: float, entry: float, stop: float, target: float) -> float:
    ti = int(np.searchsorted(run, target))  # first index with run >= target
    risk = entry - stop
    if ti < len(run) and ti < si:
        return (target - entry) / risk
    if si < len(run):
        return -1.0
    return (last - entry) / risk


def summarize(trades: list[Trade], paths, shuffles: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame([asdict(t) for t in trades])
    rows = []
    for (dr, pa), g in df.groupby(["direction", "path"]):
        ii = g.index.to_numpy()
        k = ((g["target"] - g["extreme"]) / g["rng"]).to_numpy()  # target beyond H, range units
        real = g["r_gross"].mean()
        sims = np.empty(shuffles)
        for s in range(shuffles):
            kk = rng.permutation(k)
            r = [rescore(*paths[i], trades[i].entry, trades[i].stop,
                         trades[i].extreme + kk[n] * trades[i].rng) for n, i in enumerate(ii)]
            sims[s] = np.mean(r)
        p0 = ((g["entry"] - g["stop"]) / (g["target"] - g["stop"])).mean()
        rows.append({
            "direction": dr, "path": pa, "n": len(g),
            "tp": (g["outcome"] == "tp").mean(), "stop": (g["outcome"] == "stop").mean(),
            "time": (g["outcome"] == "time").mean(), "p0_driftless": p0,
            "reward_R_median": ((g["target"] - g["entry"]) / (g["entry"] - g["stop"])).median(),
            "risk_in_spreads_median": ((g["entry"] - g["stop"]) / g["spread"]).median(),
            "R_gross": real, "cost_R": g["cost_r"].mean(), "R_net": g["r_net"].mean(),
            "placebo_R_gross": sims.mean(), "p_shuffle": float((sims >= real).mean()),
            "R_net_by_year": g.groupby("year")["r_net"].mean().round(3).to_dict(),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", default="US100")
    ap.add_argument("--start", type=int, default=2016)
    ap.add_argument("--end", type=int, default=2021)
    ap.add_argument("--shuffles", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    if args.end >= 2022:
        raise SystemExit("discovery only: 2022-2024 is confirmation, 2025+ is the holdout")

    raw = load_cached(args.asset, args.start, args.end)
    m1, info = build_study_frame(raw, args.asset)
    trades, stats = collect(m1)
    paths = _paths(m1, trades)
    rows = summarize(trades, paths, args.shuffles, args.seed)

    print(f"{args.asset} {args.start}-{args.end}  frame {info}  days {stats}")
    print(f"{'dir':5s} {'path':8s} {'n':>4s} {'tp%':>5s} {'stop%':>6s} {'time%':>6s} {'p0':>5s} "
          f"{'rew R':>6s} {'risk/spr':>8s} {'R gross':>8s} {'placebo':>8s} {'p_shuf':>6s} "
          f"{'cost R':>7s} {'R net':>7s}")
    for r in rows:
        print(f"{r['direction']:5s} {r['path']:8s} {r['n']:4d} {r['tp']:5.0%} {r['stop']:6.0%} "
              f"{r['time']:6.0%} {r['p0_driftless']:5.2f} {r['reward_R_median']:6.2f} "
              f"{r['risk_in_spreads_median']:8.1f} {r['R_gross']:8.3f} {r['placebo_R_gross']:8.3f} "
              f"{r['p_shuffle']:6.3f} {r['cost_R']:7.3f} {r['R_net']:7.3f}")
        print(f"      net R by year: {r['R_net_by_year']}")

    out = Path("artifacts/h11_target_first_quick.json")
    out.write_text(json.dumps({"asset": args.asset, "years": [args.start, args.end], "frame": info,
                               "stats": stats, "groups": rows,
                               "trades": [asdict(t) for t in trades]}, indent=1, default=str))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
