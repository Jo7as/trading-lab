"""Pullback: enter a session trend on a measured retrace, with a resting order.

Mechanism. Within a session a directional move often continues, but rarely in a
straight line: it retraces. Liquidity providers who fill that retracement get paid
for it (the same mechanism as the fade candidates, but *with* the trend instead of
against it). The trade is therefore a **limit order into a trend**, which matters
for cost: a resting order sits at the bid (long) and pays no half-spread, whereas
a market entry pays the full spread on the way in.

Signal, all dimensionless and computed only from data up to the bar:

* ``sigma_m`` - mean |mid move| from the session anchor to the same clock minute,
  over the last ``sigma_days`` sessions. This is the noise-area scale factor of
  Zarattini/Baltussen, used here as the unit of trend and retrace size.
* ``trend``   - (mid - anchor) / sigma_m. Anchor is the session open, gap-corrected
  (snapped to the first traded level, so an opening jump is not counted as trend).
* ``retrace`` - (session extreme - mid) / sigma_m, i.e. how far the current price
  has come back from the best level of the day, in the same units.

A long setup needs ``trend >= trend_min`` **and** ``retrace >= pullback_min``;
short is mirrored. The resting entry is the mid at signal time; stop sits a
``stop_buffer`` beyond the retrace extreme; the target is ``target_r`` times the
resulting risk. Fill, stop and target are all resolved on the 1-minute series, so
the intrabar order of stop and target is the real one (from M15 bars it would not
be decidable) — and each order is checked on the side of the book it really
trades at, so the spread is already inside the fills and no separate cost is
charged. What is *not* modelled: fill rate and adverse selection. A touched limit
is assumed to fill at its own price.

Use ``run_backtest`` for the trade log, or ``signals`` for the event study
(response and dose-response without any exit model).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN = np.timedelta64(1, "m")
SESSIONS: dict[str, tuple[str, str, str]] = {
    # name: (timezone, session open, flat time)
    "london": ("Europe/London", "08:00", "16:30"),
    "ny": ("America/New_York", "09:30", "15:55"),
    "both": ("Europe/London", "08:00", "16:30"),
}


@dataclass(frozen=True)
class PullbackConfig:
    asset: str = "US100"
    session: str = "ny"
    sigma_days: int = 14
    trend_min: float = 0.5          # trend in sigma_m
    pullback_min: float = 0.30      # retrace in sigma_m
    pullback_max: float = 3.0       # beyond this it is a reversal, not a retrace
    stop_buffer: float = 0.20       # sigma_m beyond the retrace extreme
    target_r: float = 1.5
    max_hold_minutes: int = 120
    entry: str = "limit"            # limit (resting at the signal mid) | market (next bar's open)
    limit_wait_minutes: int = 10    # how long the resting order stays
    max_cost_r: float = 0.10        # reject if round-trip cost > this x R
    spread_filter: float = 1.5      # reject if spread > this x local median
    require_confirmation: bool = False   # market entry only: a bar closing back with trend


def _mid(df: pd.DataFrame) -> pd.Series:
    return (df["close"] + df["ask_close"]) / 2.0


def session_frame(df: pd.DataFrame, cfg: PullbackConfig) -> pd.DataFrame:
    """Session-anchored frame on the 1-minute grid, one row per bar.

    Both sides of the book are kept explicitly — ``bid_*`` is the side a long
    exits at and a short enters at, ``ask_*`` the other way round — because every
    fill decision below depends on which side it really trades at. ``high``,
    ``low`` and ``open`` stay as the bid aliases they have always been, so the
    event studies that read them keep working.
    """
    tz, open_s, flat_s = SESSIONS[cfg.session]
    local = df.index.tz_convert(tz)
    day = local.normalize()
    mins = local.hour * 60 + local.minute
    o = int(open_s[:2]) * 60 + int(open_s[3:])
    f = int(flat_s[:2]) * 60 + int(flat_s[3:])
    inner = (mins >= o) & (mins <= f)
    out = pd.DataFrame({"day": day, "minute": mins, "mid": _mid(df),
                        "bid": df["close"], "ask": df["ask_close"],
                        "bid_open": df["open"], "bid_high": df["high"], "bid_low": df["low"],
                        "ask_open": df["ask_open"], "ask_high": df["ask_high"],
                        "ask_low": df["ask_low"],
                        "high": df["high"], "low": df["low"], "open": df["open"]})
    out = out[inner]
    # session anchor: first traded mid of the session, i.e. gap-corrected
    out["anchor"] = out.groupby("day")["mid"].transform("first")
    out["dmin"] = out["minute"] - o
    return out


def sigma_by_minute(frame: pd.DataFrame, cfg: PullbackConfig) -> pd.Series:
    """Mean |mid move| from the anchor to the same session minute, over the last N days."""
    move = (frame["mid"] / frame["anchor"] - 1.0).abs()
    by_day_min = pd.DataFrame({"day": frame["day"], "dmin": frame["dmin"], "move": move})
    pivot = by_day_min.pivot_table(index="day", columns="dmin", values="move", aggfunc="mean")
    sigma = pivot.rolling(cfg.sigma_days, min_periods=max(5, cfg.sigma_days // 2)).mean()
    return sigma.stack().rename("sigma")


def signals(df: pd.DataFrame, cfg: PullbackConfig) -> pd.DataFrame:
    """Every bar's trend/retrace state plus the resulting setup direction."""
    f = session_frame(df, cfg)
    sig = sigma_by_minute(f, cfg)
    f = f.join(sig, on=["day", "dmin"])
    f = f[f["sigma"] > 0]
    f["trend"] = (f["mid"] - f["anchor"]) / (f["sigma"] * f["anchor"])
    hi = f.groupby("day")["mid"].cummax()
    lo = f.groupby("day")["mid"].cummin()
    f["retrace_long"] = (hi - f["mid"]) / (f["sigma"] * f["anchor"])
    f["retrace_short"] = (f["mid"] - lo) / (f["sigma"] * f["anchor"])
    state = (f["trend"] >= cfg.trend_min) & (f["retrace_long"] >= cfg.pullback_min) & \
        (f["retrace_long"] <= cfg.pullback_max)
    state_s = (f["trend"] <= -cfg.trend_min) & (f["retrace_short"] >= cfg.pullback_min) & \
        (f["retrace_short"] <= cfg.pullback_max)
    # a *trigger*, not a state: the bar on which the retrace condition turns true.
    # Without this the setup stays true for every following bar of the retrace,
    # which inflates the sample and makes every bar look like a fresh signal.
    prev = pd.Series(state).groupby(f["day"]).shift(1, fill_value=False)
    prev_s = pd.Series(state_s).groupby(f["day"]).shift(1, fill_value=False)
    f["setup"] = np.where(state & ~prev, 1, np.where(state_s & ~prev_s, -1, 0))
    return f


def _resolve(f: pd.DataFrame, i: int, cfg: PullbackConfig) -> dict | None:
    """Fill, then walk forward on the 1-minute bars to stop / target / time.

    Every fill is resolved on the side of the book it actually trades at:

    ============  =========================  ==========================
    order         long                       short
    ============  =========================  ==========================
    entry limit   fills when ask_low <= L    fills when bid_high >= L
    entry market  buys the ask_open          sells the bid_open
    stop          stop-sell on the bid       stop-buy on the ask
    target        limit-sell on the bid      limit-buy on the ask
    time exit     sells the bid              buys the ask
    ============  =========================  ==========================

    The fills therefore already contain the spread, which is why nothing is
    subtracted from the P&L afterwards. (The first version tested a long's
    resting order against the *bid* — met on every bar, so every signal filled
    instantly at the mid — and a short's stop and target against the bid too.
    Both made the strategy look tradeable when it was not.)
    """
    row = f.iloc[i]
    direction = int(row["setup"])
    if direction == 0:
        return None
    long_ = direction > 0
    entry = float(row["mid"])
    sigma_abs = float(row["sigma"]) * float(row["anchor"])
    day = row["day"]
    before = f.iloc[:i + 1].loc[f["day"] == day]
    if long_:
        extreme = float(before["bid_low"].min())      # low a resting buy could reach
        stop = extreme - cfg.stop_buffer * sigma_abs
    else:
        extreme = float(before["ask_high"].max())     # high a resting sell could reach
        stop = extreme + cfg.stop_buffer * sigma_abs
    if abs(entry - stop) <= 0:
        return None
    cost = float(row["ask"] - row["bid"])             # spread at signal time, for the filter
    if cost / abs(entry - stop) > cfg.max_cost_r:
        return None

    # fill: a resting order has to be touched, a market order pays the open
    j: int | None = None
    if cfg.entry == "limit":
        for k in range(i + 1, min(i + 1 + cfg.limit_wait_minutes, len(f))):
            bar = f.iloc[k]
            if bar["day"] != day:
                break
            touched = float(bar["ask_low"]) <= entry if long_ else float(bar["bid_high"]) >= entry
            if touched:
                j = k
                break
        if j is None:
            return None
    else:
        j = i + 1
        if j >= len(f) or f.iloc[j]["day"] != day:
            return None
        entry = float(f.iloc[j]["ask_open"] if long_ else f.iloc[j]["bid_open"])

    # the bracket is the structure's (stop) plus `target_r` times the risk that
    # the *actual* fill implies
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    target = entry + direction * cfg.target_r * risk

    end = min(j + cfg.max_hold_minutes, len(f) - 1)
    exit_px, reason, k = None, "time", end
    for k in range(j, end + 1):
        bar = f.iloc[k]
        if bar["day"] != day:
            k -= 1
            bar = f.iloc[k]
            exit_px, reason = float(bar["bid"] if long_ else bar["ask"]), "session"
            break
        adverse = float(bar["bid_low"] if long_ else bar["ask_high"])
        favourable = float(bar["bid_high"] if long_ else bar["ask_low"])
        hit_stop = adverse <= stop if long_ else adverse >= stop
        hit_tgt = favourable >= target if long_ else favourable <= target
        if hit_stop:                       # stop wins a tie, as everywhere in this repo
            exit_px, reason = stop, "stop"
            break
        if hit_tgt:
            exit_px, reason = target, "target"
            break
    else:
        last = f.iloc[end]
        exit_px, reason = float(last["bid"] if long_ else last["ask"]), "time"
    if exit_px is None:
        return None
    r = direction * (exit_px - entry) / risk
    return {"entry_time": f.index[j], "exit_time": f.index[k], "direction": direction,
            "entry": entry, "stop": stop, "target": target, "exit": exit_px,
            "risk": risk, "r": r, "spread": cost, "cost_r": cost / risk,
            "exit_reason": reason,
            "trend": float(row["trend"]),
            "retrace": float(row["retrace_long"] if long_ else row["retrace_short"])}


def run_backtest(df: pd.DataFrame, cfg: PullbackConfig, *, one_per_day: bool = True
                 ) -> dict:
    """Trade log on the 1-minute series. One setup per day (first that fills)."""
    f = signals(df, cfg)
    trades: list[dict] = []
    seen: set = set()
    for i in range(len(f)):
        if f.iloc[i]["setup"] == 0:
            continue
        day = f.iloc[i]["day"]
        if one_per_day and day in seen:
            continue
        t = _resolve(f, i, cfg)
        if t is None:
            continue
        seen.add(day)
        t["day"] = day.date().isoformat()
        trades.append(t)
    return {"trades": trades, "signals_frame": f}
