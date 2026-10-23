"""Research engine (S1/S2b): observe recurring behaviour, test it before coding.

**WIP — do not quote the event-study numbers yet.** Self-checking found three
defects, two fixed and one open:

- OPEN: the first/last-hour conditions still return implausible magnitudes
  (≈+4 %/day on US100 15m, i.e. ~10x any plausible move). That points at
  unvetted data/session boundaries in the cached CFD series, not at a real
  effect — it must be traced with a per-day audit before any number is used.
- FIXED: outlier bars (1 644 bars on US100, worst 104x the median move) are now
  masked before the statistics — they made a "4.5 %/day" momentum look real.
- FIXED: the placebo sampled the event set instead of the full series and
  returned NaN; it now samples the full series.


The loop is: an **observation** (recurring price behaviour) → a **mechanism**
(who trades why) → a **testable claim** with a number. This module does the last
step mechanically: define a condition on the data, then measure the *forward
return in the condition's direction*, compared with the round-trip cost, a
placebo (random timestamps, same horizon) and the dose-response (magnitude
quintiles). Nothing here is a strategy — it decides whether writing one is worth
it.

Conditions are pre-registered and mechanical (no chart patterns): prior-day
level breaks, z-score extremes, first-hour and last-hour momentum, range
compression. A condition that fails here is dropped, not "improved".

Caveat, stated up front: overlapping events inflate the t-statistic. Conditions
that fire once per day (first/last hour, compression) are clean; the level/z
conditions are sampled with a minimum bar gap (``min_gap``) to limit the
overlap. Treat ``t`` as a screen, not as proof.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_EVENTS = 30
PASS_T = 3.0
PASS_EFFECT_COST = 2.0


@dataclass
class EventStudy:
    name: str
    mechanism: str
    n: int
    horizon: str
    mean_fwd_pct: float
    hit_rate: float
    t_stat: float
    cost_pct: float
    placebo_mean_pct: float
    quintiles: tuple[float, ...] = ()
    notes: str = ""

    @property
    def effect_cost(self) -> float:
        return self.mean_fwd_pct / self.cost_pct if self.cost_pct else float("inf")

    def verdict(self) -> str:
        if self.n < MIN_EVENTS:
            return "INSUFFICIENT"
        if self.t_stat < PASS_T:
            return "FAIL(t)"
        if self.effect_cost < PASS_EFFECT_COST:
            return "FAIL(cost)"
        if abs(self.placebo_mean_pct) > 0.5 * abs(self.mean_fwd_pct):
            return "FAIL(placebo)"
        return "PASS"

    def summary(self) -> str:
        q = " ".join(f"{x:+.3f}" for x in self.quintiles)
        return (f"{self.name:22s} n={self.n:5d} fwd {self.mean_fwd_pct:+.4f}% cost {self.cost_pct:.4f}% "
                f"eff/cost {self.effect_cost:5.2f} t={self.t_stat:5.2f} hit {self.hit_rate:.0%} "
                f"placebo {self.placebo_mean_pct:+.4f}% quint {q}  {self.verdict()}")


def prepare(m1: pd.DataFrame, timeframe: str, tz: str) -> tuple[pd.DataFrame, pd.Series]:
    df = m1.resample(timeframe).agg({"open": "first", "high": "max", "low": "min",
                                     "close": "last", "ask_close": "last"}).dropna()
    mid = (df["close"] + df["ask_close"]) / 2.0
    return df, mid


def _spread_pct(df: pd.DataFrame) -> pd.Series:
    return (df["ask_close"] - df["close"]) / ((df["close"] + df["ask_close"]) / 2.0)


def _study(name: str, mechanism: str, ts: pd.DatetimeIndex, direction: np.ndarray,
           fwd_pct: np.ndarray, cost_pct: np.ndarray, horizon: str,
           magnitude: np.ndarray | None = None, seed: int = 7,
           placebo_pool: np.ndarray | None = None) -> EventStudy:
    """Shared statistics for a set of events."""
    ok = np.isfinite(fwd_pct) & np.isfinite(cost_pct) & (direction != 0)
    ts, direction, fwd_pct, cost_pct = ts[ok], direction[ok], fwd_pct[ok], cost_pct[ok]
    magnitude = magnitude[ok] if magnitude is not None else None
    n = len(fwd_pct)
    if n == 0:
        return EventStudy(name, mechanism, 0, horizon, 0, 0, 0, 0, 0)
    signed = direction * fwd_pct                     # return in the traded direction
    mean = float(np.mean(signed))
    std = float(np.std(signed, ddof=1)) if n > 1 else 0.0
    t = mean / (std / np.sqrt(n)) if std else 0.0
    hit = float((signed > 0).mean())
    cost = float(np.median(cost_pct))
    # placebo: random bars on the full series, same horizon and direction mix
    placebo = 0.0
    if placebo_pool is not None:
        pool = np.asarray(placebo_pool, dtype=float)
        pool = pool[np.isfinite(pool)]
        if len(pool):
            rng = np.random.default_rng(seed)
            pick = rng.integers(0, len(pool), size=min(n, len(pool)))
            mix = rng.choice(np.unique(direction), size=len(pick))
            placebo = float(np.mean(mix * pool[pick]))
    quint = ()
    if magnitude is not None and n >= 25:
        q = pd.qcut(pd.Series(magnitude).rank(method="first"), 5, labels=False)
        quint = tuple(round(float(np.mean(signed[q == k])) * 100, 4) for k in range(5))
    return EventStudy(name, mechanism, n, horizon, round(mean * 100, 4), round(hit, 3),
                      round(t, 2), round(cost * 100, 4), round(placebo * 100, 4), quint)


# ── Pre-registered conditions ───────────────────────────────────────

def prior_day_break(df: pd.DataFrame, mid: pd.Series, tz: str, horizon: int,
                    gap: int) -> EventStudy:
    """First close beyond yesterday's high/low → continuation (Osler clusters)."""
    day = df.index.tz_convert(tz).date
    for side, sign in (("high", 1), ("low", -1)):
        level = getattr(df, side).groupby(day).max().shift(1) if side == "high" \
            else df["low"].groupby(day).min().shift(1)
        lv = pd.Series(day).map(level).to_numpy(float)
        close = df["close"].to_numpy(float)
        prev = np.r_[np.nan, close[:-1]]
        brk = ((close > lv) & (prev <= lv)) if sign > 0 else ((close < lv) & (prev >= lv))
        idx = _thin(np.flatnonzero(brk), gap)
        fwd = _forward(mid, idx, horizon)
        yield _study(f"pd_{side}_break", "level-break continuation (order clusters)",
                     df.index[idx], np.full(len(idx), sign), fwd, _spread_pct(df).to_numpy()[idx],
                     f"{horizon} bars", magnitude=np.abs(close[idx] / lv[idx] - 1),
                     placebo_pool=_forward(mid, np.arange(len(mid) - horizon), horizon))


def zscore_extreme(df: pd.DataFrame, mid: pd.Series, lookback: int, z: float,
                   horizon: int, gap: int) -> EventStudy:
    """|z| ≥ threshold vs the rolling mean → reversion back to the mean."""
    m = mid.rolling(lookback).mean()
    s = mid.rolling(lookback).std()
    zz = ((mid - m) / s).to_numpy(float)
    ev = np.flatnonzero(np.abs(zz) >= z)
    ev = _thin(ev, gap)
    direction = -np.sign(zz[ev])
    fwd = np.array([direction[k] * (mid.iloc[i + horizon] - mid.iloc[i]) / mid.iloc[i] * 100
                    for k, i in enumerate(ev) if i + horizon < len(mid)])
    ev = ev[:len(fwd)]
    return _study(f"z|{z:g}_reversion", "mean reversion after a stretched move",
                  df.index[ev], direction[:len(fwd)], fwd, _spread_pct(df).to_numpy()[ev],
                  f"{horizon} bars", magnitude=np.abs(zz[ev]),
                  placebo_pool=_forward(mid, np.arange(len(mid) - horizon), horizon))


def first_hour_momentum(df: pd.DataFrame, mid: pd.Series, tz: str) -> EventStudy:
    """Sign of the first hour's move → rest-of-day move (intraday momentum)."""
    loc = df.index.tz_convert(tz)
    minutes = loc.hour * 60 + loc.minute
    day = loc.date
    ev, direction, fwd = [], [], []
    for d in pd.unique(day):
        m = day == d
        first = np.flatnonzero(m & (minutes >= 570) & (minutes < 630))
        rest = np.flatnonzero(m & (minutes >= 630) & (minutes < 960))
        if len(first) < 1 or len(rest) < 3:
            continue
        o, c1, c2 = mid.iloc[first[0] - 1] if first[0] else mid.iloc[0], mid.iloc[first[-1]], mid.iloc[rest[-1]]
        if not np.isfinite([o, c1, c2]).all():
            continue
        sg = np.sign(c1 - o)
        if sg == 0:
            continue
        ev.append(first[-1]); direction.append(sg); fwd.append((c2 - c1) / c1 * 100)
    return _study("first_hour_mom", "intraday momentum (day return persists)",
                  df.index[ev], np.array(direction), np.array(fwd),
                  _spread_pct(df).to_numpy()[ev], "to session close",
                  magnitude=np.abs(np.array(fwd)), placebo_pool=_all_days_rest(df, mid, tz))


def last_hour_momentum(df: pd.DataFrame, mid: pd.Series, tz: str) -> EventStudy:
    """Sign of the day's move by 15:30 → last 30 minutes (the 'C' effect)."""
    loc = df.index.tz_convert(tz)
    minutes = loc.hour * 60 + loc.minute
    day = loc.date
    ev, direction, fwd = [], [], []
    for d in pd.unique(day):
        m = day == d
        upto = np.flatnonzero(m & (minutes <= 930))      # ≤ 15:30
        tail = np.flatnonzero(m & (minutes >= 930) & (minutes <= 960))
        if len(upto) < 3 or len(tail) < 2:
            continue
        o, c1, c2 = mid.iloc[upto[0] - 1] if upto[0] else mid.iloc[0], mid.iloc[upto[-1]], mid.iloc[tail[-1]]
        if not np.isfinite([o, c1, c2]).all():
            continue
        sg = np.sign(c1 - o)
        if sg == 0:
            continue
        ev.append(upto[-1]); direction.append(sg); fwd.append((c2 - c1) / c1 * 100)
    return _study("last_hour_mom", "late-day momentum (gamma/ETF rebalancing)",
                  df.index[ev], np.array(direction), np.array(fwd),
                  _spread_pct(df).to_numpy()[ev], "last 30 min",
                  magnitude=np.abs(np.array(fwd)), placebo_pool=_all_days_tail(df, mid, tz))


def _all_days_rest(df: pd.DataFrame, mid: pd.Series, tz: str) -> np.ndarray:
    """Every day's rest-of-session return in % (the placebo pool for C)."""
    loc = df.index.tz_convert(tz)
    minutes = loc.hour * 60 + loc.minute
    day = loc.date
    out = []
    for d in pd.unique(day):
        m = day == d
        first = np.flatnonzero(m & (minutes >= 570) & (minutes < 630))
        rest = np.flatnonzero(m & (minutes >= 630) & (minutes < 960))
        if len(first) < 1 or len(rest) < 3:
            continue
        c1, c2 = mid.iloc[first[-1]], mid.iloc[rest[-1]]
        if np.isfinite([c1, c2]).all():
            out.append((c2 - c1) / c1 * 100)
    return np.array(out)


def _all_days_tail(df: pd.DataFrame, mid: pd.Series, tz: str) -> np.ndarray:
    """Every day's last-30-minute return in % (the placebo pool for D)."""
    loc = df.index.tz_convert(tz)
    minutes = loc.hour * 60 + loc.minute
    day = loc.date
    out = []
    for d in pd.unique(day):
        m = day == d
        upto = np.flatnonzero(m & (minutes <= 930))
        tail = np.flatnonzero(m & (minutes >= 930) & (minutes <= 960))
        if len(upto) < 3 or len(tail) < 2:
            continue
        c1, c2 = mid.iloc[upto[-1]], mid.iloc[tail[-1]]
        if np.isfinite([c1, c2]).all():
            out.append((c2 - c1) / c1 * 100)
    return np.array(out)


# ── helpers ─────────────────────────────────────────────────────────

def _thin(idx: np.ndarray, gap: int) -> np.ndarray:
    if gap <= 1 or len(idx) == 0:
        return idx
    keep = [idx[0]]
    for i in idx[1:]:
        if i - keep[-1] >= gap:
            keep.append(i)
    return np.array(keep, dtype=int)


def _forward(mid: pd.Series, idx: np.ndarray, horizon: int) -> np.ndarray:
    """Raw forward return in %, no direction applied (that happens in _study)."""
    out = []
    for i in idx:
        if i + horizon < len(mid):
            out.append((mid.iloc[i + horizon] - mid.iloc[i]) / mid.iloc[i] * 100)
    return np.array(out)


def run_all(m1: pd.DataFrame, *, timeframe: str = "15min", tz: str = "America/New_York",
            clean: bool = True, z_cut: float = 12.0) -> list[EventStudy]:
    """The pre-registered condition set, in one call.

    ``clean`` masks bars whose move exceeds ``z_cut`` × the median absolute move.
    The cached CFD data does contain such bars; without this every mean is
    garbage (we measured a "4.5 %/day" first-hour momentum that was pure spikes).
    """
    df, mid = prepare(m1, timeframe, tz)
    if clean:
        r = mid.pct_change()
        scale = float(r.abs().median()) or 1e-9
        mid = mid.where(r.abs() <= z_cut * scale)
    bar_min = {"5min": 5, "15min": 15, "30min": 30, "1h": 60}.get(timeframe, 15)
    h30 = max(int(round(30 / bar_min)), 1)
    studies: list[EventStudy] = []
    studies += list(prior_day_break(df, mid, tz, h30, gap=h30 * 2))
    studies.append(zscore_extreme(df, mid, lookback=60, z=2.0, horizon=h30, gap=h30 * 2))
    studies.append(first_hour_momentum(df, mid, tz))
    studies.append(last_hour_momentum(df, mid, tz))
    return studies


def data_quality(df: pd.DataFrame, mid: pd.Series, *, z_cut: float = 12.0) -> dict:
    """S1 data gate: bar-level outliers that would poison every mean.

    Returns the worst bar returns, how many exceed ``z_cut`` × the median
    absolute move, and a cleaned mid series with those bars removed.
    """
    r = mid.pct_change()
    scale = float(r.abs().median()) or 1e-9
    worst = float(r.abs().max()) / scale
    outliers = r.abs() > z_cut * scale
    cleaned = mid.where(~outliers.shift(0).fillna(False)).interpolate()
    return {
        "bars": int(len(r)), "median_abs_move_pct": round(scale * 100, 4),
        "worst_move_in_median_units": round(worst, 1),
        "outlier_bars": int(outliers.sum()),
        "worst_pct": round(float(r.abs().max()) * 100, 2),
        "cleaned": cleaned,
    }
