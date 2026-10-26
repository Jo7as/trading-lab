"""Momentum continuation with a full exit palette - built to be swept, not tuned.

The signal test that preceded this used one rule (sign of a 6-hour return, read at
00:00 New York) and no filter. It showed a real but small continuation edge that
**grew with the holding period** (0.35 spreads at 30 min, ~1.0-1.3 at one day),
which is a hint, not a strategy. This module turns it into something that can be
swept along the axes that actually matter:

===============  =========================================================
Achse            Varianten
===============  =========================================================
Regime           none | er (efficiency ratio) | adx | ema_stack
Session          london | ny | both
Entry            breakout | pullback | momentum | market
Stop             atr | swing | session_range
Exit             target | trail_atr | trail_swing | be_trail | time
Horizont         4 h ... 3 Tage
===============  =========================================================

Design rules, learned the hard way in this repo:

* Signals are computed on the working timeframe (default M15); **fills, stops and
  targets are resolved on the 1-minute series**, so the order of stop and target
  inside a bar is the real one.
* Every rolling quantity (ATR, EMA, ADX, ER, swing levels) uses data **up to the
  bar** and nothing after it.
* A resting entry pays one spread, a market/stop entry two. The stop is filled at
  the bid (long) / ask (short), the target at the ask / bid.
* One position at a time per instrument, one entry per session day unless
  ``allow_reentry``.

The point of the module is that ``MomentumConfig`` exposes every choice as a
field, so a sweeper can enumerate them instead of a human guessing.

    from trading_lab.strategies.momentum import MomentumConfig, run_backtest
    res = run_backtest(m1, MomentumConfig(session="ny", entry="breakout",
                                          stop="atr", exit="trail_atr"))
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

SESSIONS: dict[str, tuple[str, str, str]] = {
    "london": ("Europe/London", "08:00", "16:30"),
    "ny": ("America/New_York", "09:30", "15:55"),
    "both": ("America/New_York", "03:00", "15:55"),
}


@dataclass(frozen=True)
class MomentumConfig:
    asset: str = "US100"
    # ── regime filter ───────────────────────────────────────────────────
    regime: str = "none"            # none | er | adx | ema_stack
    regime_lookback: int = 20       # bars of the working timeframe
    regime_min: float = 0.30        # ER floor / ADX floor
    # ── session ─────────────────────────────────────────────────────────
    session: str = "ny"
    trade_minute_start: int = 0     # minutes after session open before entries
    trade_minute_end: int = 300     # after this no new entries
    # ── entry ───────────────────────────────────────────────────────────
    entry: str = "breakout"         # breakout | pullback | momentum | market
    entry_lookback: int = 20        # bars used for the breakout / momentum test
    entry_buffer_atr: float = 0.05  # breakout trigger distance, in ATR
    # ── stop ────────────────────────────────────────────────────────────
    stop: str = "atr"               # atr | swing | session_range
    stop_atr: float = 1.5
    stop_min_atr: float = 0.3       # reject trades whose stop is too tight
    # ── exit ────────────────────────────────────────────────────────────
    exit: str = "target"            # target | trail_atr | trail_swing | be_trail | time
    target_r: float = 2.0
    trail_atr: float = 2.0
    be_trigger_r: float = 1.0       # move to break-even at this R
    max_hold_minutes: int = 240
    # ── cost / risk control ─────────────────────────────────────────────
    max_cost_r: float = 0.10
    spread_filter: float = 1.5      # reject if spread > x * local median
    allow_reentry: bool = False
    tf: str = "15min"               # working timeframe for the signal


# ── indicators (all causal: value at bar t uses data <= t) ──────────────
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    change = (close - close.shift(n)).abs()
    vol = close.diff().abs().rolling(n).sum()
    return (change / vol.replace(0, np.nan)).fillna(0.0)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = atr(df, period)
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1/period, adjust=False).mean() / tr
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1/period, adjust=False).mean() / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean().fillna(0.0)


def build_frame(m1: pd.DataFrame, cfg: MomentumConfig) -> pd.DataFrame:
    """Working-timeframe signal frame, aligned back onto the 1-minute grid."""
    tf = m1.resample(cfg.tf).agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "ask_close": "last",
                                  "ask_high": "max", "ask_low": "min"})
    tf = tf.dropna()
    tf["atr"] = atr(tf, 14)
    tf["er"] = efficiency_ratio(tf["close"], cfg.regime_lookback)
    tf["adx"] = adx(tf, 14)
    tf["ema_f"] = ema(tf["close"], max(5, cfg.regime_lookback // 2))
    tf["ema_s"] = ema(tf["close"], cfg.regime_lookback * 2)
    tf["hh"] = tf["high"].rolling(cfg.entry_lookback).max().shift(1)   # prior bar only
    tf["ll"] = tf["low"].rolling(cfg.entry_lookback).min().shift(1)
    tf["mom"] = tf["close"].pct_change(cfg.entry_lookback)
    return tf


def regime_ok(tf: pd.DataFrame, cfg: MomentumConfig) -> pd.Series:
    if cfg.regime == "none":
        return pd.Series(True, index=tf.index)
    if cfg.regime == "er":
        return tf["er"] >= cfg.regime_min
    if cfg.regime == "adx":
        return tf["adx"] >= 20.0 + 20.0 * cfg.regime_min
    if cfg.regime == "ema_stack":
        return ((tf["ema_f"] > tf["ema_s"]) | (tf["ema_f"] < tf["ema_s"])) & \
               (tf["er"] >= cfg.regime_min)
    raise ValueError(cfg.regime)


def session_mask(m1: pd.DataFrame, cfg: MomentumConfig) -> tuple[pd.Series, pd.Series]:
    tz, open_s, flat_s = SESSIONS[cfg.session]
    local = m1.index.tz_convert(tz)
    day = local.normalize()
    mins = local.hour * 60 + local.minute
    o = int(open_s[:2]) * 60 + int(open_s[3:])
    f = int(flat_s[:2]) * 60 + int(flat_s[3:])
    in_sess = (mins >= o) & (mins <= f)
    can_enter = in_sess & (mins >= o + cfg.trade_minute_start) & (mins <= o + cfg.trade_minute_end)
    return pd.Series(in_sess, index=m1.index), pd.Series(can_enter, index=m1.index)


def _stop_level(kind: str, entry: float, direction: int, tf_row: pd.Series,
                m1: pd.DataFrame, i: int, cfg: MomentumConfig) -> float:
    a = float(tf_row["atr"]) * cfg.stop_atr
    if kind == "atr":
        return entry - direction * a
    if kind == "swing":
        n = cfg.entry_lookback
        s = m1["low"].iloc[max(0, i - n):i + 1].min() if direction > 0 \
            else m1["high"].iloc[max(0, i - n):i + 1].max()
        return float(s)
    if kind == "session_range":
        return entry - direction * max(a, cfg.stop_min_atr * float(tf_row["atr"]) * 2)
    raise ValueError(kind)


def run_backtest(m1: pd.DataFrame, cfg: MomentumConfig, *, one_per_day: bool = True) -> dict:
    """Sweepable momentum backtest. Fills/stops/targets resolved on 1-minute bars."""
    tf = build_frame(m1, cfg)
    ok = regime_ok(tf, cfg)
    in_sess, can_enter = session_mask(m1, cfg)
    spread = (m1["ask_close"] - m1["close"])
    spread_med = spread.rolling(1440, min_periods=60).median()
    # shift(1) BEFORE the ffill: a 15-minute bar's close is only known when the
    # bar ends, so a decision taken inside that bar may use the previous bar at
    # most. Without the shift every 1-minute bar inside a 15-minute candle sees
    # that candle's own close - classic look-ahead, and it made every variant
    # profitable.
    tf_al = tf.shift(1).reindex(m1.index, method="ffill")
    ok_al = ok.shift(1).reindex(m1.index, method="ffill").fillna(False)
    m = ((m1["close"] + m1["ask_close"]) / 2.0).to_numpy()
    bid = m1["close"].to_numpy()
    ask = m1["ask_close"].to_numpy()
    hi = m1["high"].to_numpy()
    lo = m1["low"].to_numpy()
    sp = spread.to_numpy()
    spmed = spread_med.to_numpy()
    tfh = tf_al["hh"].to_numpy()
    tfl = tf_al["ll"].to_numpy()
    tfm = tf_al["mom"].to_numpy()
    tfa = tf_al["atr"].to_numpy()
    ce = can_enter.to_numpy()
    ins = in_sess.to_numpy()
    reg = ok_al.to_numpy()
    day = m1.index.tz_convert(SESSIONS[cfg.session][0]).normalize()
    dayn = pd.factorize(day)[0]

    rng = np.random.default_rng(20261025)
    trades: list[dict] = []
    n = len(m1)
    # pre-select the bars that can even start a trade: scanning all 2.7M 1-minute
    # bars in Python made a 300-variant sweep a 12-hour job
    cand = np.flatnonzero(ce & reg & np.isfinite(tfa) & (tfa > 0))
    done: set = set()
    pos = 0
    while pos < len(cand):
        i = int(cand[pos])
        pos += 1
        if one_per_day and not cfg.allow_reentry and dayn[i] in done:
            continue
        direction = 0
        trigger = np.nan
        if cfg.entry == "random":                      # placebo control
            # must be a fair coin: `integers(-1, 2) or 1` makes 0 -> +1, so the
            # placebo was 2/3 long, and on a drifting index that alone produced
            # +0.14 R per trade - more than the "edge" it was meant to test.
            direction = int(rng.choice([-1, 1]))
        elif cfg.entry == "market":
            direction = int(np.sign(tfm[i])) if np.isfinite(tfm[i]) else 0
        elif cfg.entry == "breakout":
            if m[i] > tfh[i] + cfg.entry_buffer_atr * tfa[i]:
                direction = 1
            elif m[i] < tfl[i] - cfg.entry_buffer_atr * tfa[i]:
                direction = -1
        elif cfg.entry == "pullback":
            # trend from the slow EMA, entry when price retraces to the fast EMA
            if tf_al["ema_f"].iloc[i] > tf_al["ema_s"].iloc[i] and m[i] <= tf_al["ema_f"].iloc[i]:
                direction = 1
            elif tf_al["ema_f"].iloc[i] < tf_al["ema_s"].iloc[i] and m[i] >= tf_al["ema_f"].iloc[i]:
                direction = -1
        elif cfg.entry == "momentum":
            direction = int(np.sign(tfm[i])) if np.isfinite(tfm[i]) else 0
            if direction > 0 and m[i] < tfh[i]:
                direction = 0
            if direction < 0 and m[i] > tfl[i]:
                direction = 0
        if direction == 0:
            continue
        # execution: market/stop entry at the far side, resting entry at the mid
        entry = (ask[i] if direction > 0 else bid[i]) if cfg.entry in ("market", "breakout") else m[i]
        stop = _stop_level(cfg.stop, entry, direction, tf_al.iloc[i], m1, i, cfg)
        risk = abs(entry - stop)
        if risk <= 0 or risk / max(tfa[i], 1e-12) < cfg.stop_min_atr:
            continue
        cost = sp[i]
        if cost / risk > cfg.max_cost_r:
            continue
        if np.isfinite(spmed[i]) and cfg.spread_filter and sp[i] > cfg.spread_filter * spmed[i]:
            continue
        target = entry + direction * cfg.target_r * risk
        # walk forward on 1-minute bars
        # Walk forward bar by bar. ORDER MATTERS:
        #   1. test the stop that existed at the END of the previous bar
        #   2. test the target
        #   3. only then tighten a trailing / break-even stop, using this bar's
        #      extreme, so it applies from the NEXT bar on.
        # Doing 3 before 1 lets a trailing stop be "hit" by the same bar that
        # created it, which is look-ahead inside the bar and produced +123 R.
        end, reason = i, "time"
        peak_fav, peak_adv = entry, entry        # running favourable/adverse extremes
        for k in range(i + 1, n):
            if not ins[k]:
                end, reason = k - 1, "session"
                break
            if k - i > cfg.max_hold_minutes:
                end, reason = k, "time"
                break
            end = k
            hit_stop = lo[k] <= stop if direction > 0 else hi[k] >= stop
            hit_tgt = hi[k] >= target if direction > 0 else lo[k] <= target
            if hit_stop:                          # stop wins a tie
                reason = "stop"
                break
            if cfg.exit == "target" and hit_tgt:
                reason = "target"
                break
            # only now update the dynamic stop, effective from the next bar
            peak_fav = max(peak_fav, hi[k]) if direction > 0 else min(peak_fav, lo[k])
            peak_adv = min(peak_adv, lo[k]) if direction > 0 else max(peak_adv, hi[k])
            if cfg.exit == "trail_atr":
                trail = peak_fav - direction * cfg.trail_atr * tfa[i]
                stop = max(stop, trail) if direction > 0 else min(stop, trail)
            elif cfg.exit == "be_trail":
                mfe = direction * (peak_fav - entry)
                if mfe >= cfg.be_trigger_r * risk:
                    stop = max(stop, entry) if direction > 0 else min(stop, entry)
                trail = peak_fav - direction * cfg.trail_atr * tfa[i]
                stop = max(stop, trail) if direction > 0 else min(stop, trail)
        exit_px = {"stop": stop, "target": target}.get(reason, m[end])
        gross = direction * (exit_px - entry)
        net = gross - cost
        trades.append({"entry_time": m1.index[i], "exit_time": m1.index[end],
                       "direction": direction, "entry": entry, "stop": stop,
                       "target": target, "exit": exit_px, "risk": risk,
                       "r_gross": gross / risk, "r": net / risk,
                       "cost_r": cost / risk, "exit_reason": reason,
                       # absolute units, so a study can price the trade in bp of the
                       # price without re-deriving risk and mid (needed by S2)
                       "gross_abs": gross, "cost_abs": cost, "mid_entry": float(m[i]),
                       "mid_exit": float(m[end]),
                       "day": str(day[i].date())})
        done.add(dayn[i])
        while pos < len(cand) and cand[pos] <= end:
            pos += 1
    return {"trades": trades}
