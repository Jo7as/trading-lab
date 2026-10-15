"""Trend Range Rejection (TRR Forever Model) — CB Reversal Timing.

v1 defaults:
  - Assets: USDJPY / XAUUSD (caller chooses)
  - Trending ranges (HH+HL or LH+LL) with avg pullback ≥ 50%
  - Strong trends (pullbacks < 50%) = kill switch (no trade)
  - With-trend + counter-trend at range extremes (allow_counter_trend=True)
  - No CSER / DXY correlation filter
  - H1: regime + range extremes + ATR overextension + AOI levels
  - M1: CB timing (:15/:30/:45), volume stretch, sweep→MSS→50% limit

Claimed discretionary cadence ~1–2 trades/day — optional via max_trades_per_day (0=off).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from ..backtest import _compute_metrics, _pos_to_dict
from ..indicators import atr as atr_indicator
from ..risk import Position, RiskManager
from .smc.structure import MarketStructureEngine


class RangeRegime(str, Enum):
    OFF = "off"  # strong trend or unknown
    BULL_TREND_RANGE = "bull_trend_range"
    BEAR_TREND_RANGE = "bear_trend_range"


@dataclass(frozen=True)
class TrendRangeRejectionConfig:
    asset: str = "USD/JPY"
    asset_label: str = "USDJPY"

    # Macro lookback on H1 (hours ≈ bars)
    lookback_hours: int = 6
    swing_length_h1: int = 2
    swing_length_m1: int = 3
    min_pullback_pct: float = 0.50  # kill if avg pullback below this

    # AOI / fib half-rule
    aoi_atr_tol: float = 0.55  # touch zone width around level
    avoid_mid_band: float = 0.05  # avoid 0.5 ± this of last impulse (dead zone)
    h1_level_lookback: int = 40  # H1 bars for open/close flip levels

    # Overextension of current/last H1 candle into extreme
    atr_period: int = 14
    overext_atr_mult: float = 0.35

    # CB reversal timing (minute-of-hour)
    timing_minutes: tuple[int, ...] = (15, 30, 45)
    timing_tolerance_min: int = 3

    # M1 momentum / volume into AOI
    vol_ma_period: int = 20
    vol_mult: float = 1.0
    body_atr_mult: float = 0.25  # M1 body stretch fallback if volume sparse

    # Entry
    entry_timeout_bars: int = 45  # M1 bars to fill 50% limit after MSS
    sl_buffer_atr: float = 0.15
    tp_mode: str = "impulse_50"  # impulse_50 | fixed_rr
    tp1_r: float = 2.0
    tp1_close_pct: float = 1.0
    move_sl_to_be_on_tp1: bool = True
    max_hold_bars: int = 240  # 4h on M1

    # Risk / cadence
    account_size: float = 10000.0
    risk_per_trade: float = 0.01
    max_daily_loss: float = 0.03
    cooldown_hours: int = 0
    max_concurrent_positions: int = 1
    max_trades_per_day: int = 0  # 0 = unlimited (no daily cap)
    sl_atr_mult: float = 1.0  # RiskConfig protocol compat

    # Counter-trend: also fade the opposite extreme of a trend-range
    # (bull range → short highs; bear range → long lows)
    allow_counter_trend: bool = True

    # Session filter (UTC hours inclusive start, exclusive end). Empty = all.
    # FX/metal: roughly London+NY overlap bias; leave open by default.
    session_start_utc: int | None = None
    session_end_utc: int | None = None


def _unix_seconds(idx: pd.DatetimeIndex) -> np.ndarray:
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    # Dukascopy frames can be datetime64[ms, UTC]; unit is not always ns.
    dtype_s = str(idx.dtype)  # e.g. datetime64[ms, UTC]
    if "[ms" in dtype_s:
        scale = 1e-3
    elif "[us" in dtype_s:
        scale = 1e-6
    elif "[s" in dtype_s:
        scale = 1.0
    else:
        scale = 1e-9
    return idx.asi8.astype(np.float64) * scale


def last_closed_bar_idx(close_unix: np.ndarray, now_unix: float) -> int:
    i = int(np.searchsorted(close_unix, now_unix, side="right") - 1)
    return i if i >= 0 else -1


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    ohlc = df.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return ohlc.dropna(subset=["open", "high", "low", "close"])


def _avg_pullback_ratio(highs: list[float], lows: list[float], bullish: bool) -> float | None:
    """Average retracement of counter-moves vs prior impulse (last few legs)."""
    if len(highs) < 2 or len(lows) < 2:
        return None
    ratios: list[float] = []
    n = min(len(highs), len(lows), 4)
    h = highs[-n:]
    l = lows[-n:]
    if bullish:
        for i in range(1, len(h)):
            impulse = h[i] - l[i - 1] if i - 1 < len(l) else 0.0
            if impulse <= 0:
                continue
            # pullback from high[i-1] area into low[i] if available
            if i < len(l):
                pb = h[i - 1] - l[i]
                if pb > 0:
                    ratios.append(pb / impulse)
    else:
        for i in range(1, len(l)):
            impulse = h[i - 1] - l[i] if i - 1 < len(h) else 0.0
            if impulse <= 0:
                continue
            if i < len(h):
                rally = h[i] - l[i]
                if rally > 0:
                    ratios.append(rally / impulse)
    if not ratios:
        return None
    return float(np.mean(ratios))


def classify_regime(
    swing_highs: list[float],
    swing_lows: list[float],
    min_pullback: float,
) -> RangeRegime:
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return RangeRegime.OFF
    hh = swing_highs[-1] > swing_highs[-2]
    hl = swing_lows[-1] > swing_lows[-2]
    lh = swing_highs[-1] < swing_highs[-2]
    ll = swing_lows[-1] < swing_lows[-2]

    if hh and hl:
        r = _avg_pullback_ratio(swing_highs, swing_lows, bullish=True)
        if r is None:
            return RangeRegime.OFF
        if r < min_pullback:
            return RangeRegime.OFF  # strong trend kill-switch
        return RangeRegime.BULL_TREND_RANGE
    if lh and ll:
        r = _avg_pullback_ratio(swing_highs, swing_lows, bullish=False)
        if r is None:
            return RangeRegime.OFF
        if r < min_pullback:
            return RangeRegime.OFF
        return RangeRegime.BEAR_TREND_RANGE
    return RangeRegime.OFF


def h1_flip_levels(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    end_i: int,
    lookback: int,
    atr_tol: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return (long_zones, short_zones) as (lo, hi) tuples near H1 polarity flips."""
    start = max(1, end_i - lookback + 1)
    long_lv: list[tuple[float, float]] = []
    short_lv: list[tuple[float, float]] = []
    for i in range(start, end_i + 1):
        prev_bear = close[i - 1] < open_[i - 1]
        cur_bull = close[i] > open_[i]
        prev_bull = close[i - 1] > open_[i - 1]
        cur_bear = close[i] < open_[i]
        px = 0.5 * (close[i - 1] + open_[i])
        z = (px - atr_tol, px + atr_tol)
        if prev_bear and cur_bull:
            long_lv.append(z)
        if prev_bull and cur_bear:
            short_lv.append(z)
    return long_lv, short_lv


def in_timing_window(ts: pd.Timestamp, marks: tuple[int, ...], tol: int) -> bool:
    m = int(ts.minute)
    s = int(ts.second)
    # use minute + fraction; bar open time for M1
    for mark in marks:
        if abs(m - mark) <= tol or abs(m + 60 - mark) <= tol or abs(m - 60 - mark) <= tol:
            if abs(m - mark) <= tol:
                return True
            # wrap cases already covered loosely
            if m >= 58 and mark <= tol:
                return True
            if m <= tol and mark >= 60 - tol:
                return True
    _ = s
    return any(abs(m - mark) <= tol for mark in marks)


def manage_position(
    pos: Position,
    idx: int,
    current_ts: float,
    current_price: float,
    current_high: float,
    current_low: float,
    cfg: TrendRangeRejectionConfig,
    rm: RiskManager,
    trades: list[dict],
) -> None:
    if pos.direction == "long":
        if current_low <= pos.stop_loss:
            rm.close_position(pos, pos.stop_loss, "stop_loss", current_ts)
            trades.append(_pos_to_dict(pos))
            return
    else:
        if current_high >= pos.stop_loss:
            rm.close_position(pos, pos.stop_loss, "stop_loss", current_ts)
            trades.append(_pos_to_dict(pos))
            return

    if not pos.tp1_hit and pos.tp1 is not None:
        hit = (pos.direction == "long" and current_high >= pos.tp1) or (
            pos.direction == "short" and current_low <= pos.tp1
        )
        if hit:
            if cfg.tp1_close_pct >= 0.999:
                rm.close_position(pos, pos.tp1, "tp", current_ts)
            else:
                rm.close_position(
                    pos,
                    pos.tp1,
                    "tp1_partial",
                    current_ts,
                    partial=True,
                    move_sl_to_be=cfg.move_sl_to_be_on_tp1,
                )
            if pos.status == "closed":
                trades.append(_pos_to_dict(pos))
            return

    if idx - pos.entry_idx >= cfg.max_hold_bars:
        rm.close_position(pos, current_price, "time_stop", current_ts)
        trades.append(_pos_to_dict(pos))


@dataclass
class _Setup:
    direction: int  # +1 long / -1 short
    regime: RangeRegime
    impulse_low: float
    impulse_high: float
    range_low: float
    range_high: float
    aoi_lo: float
    aoi_hi: float
    hour_key: str
    counter_trend: bool = False
    # M1 state
    sweep_extreme: float | None = None  # Low_B / High_B
    pivot_between: float | None = None  # intervening swing
    mss_done: bool = False
    limit: float | None = None
    sl: float | None = None
    tp: float | None = None
    armed_i: int | None = None


def _candidate_directions(
    regime: RangeRegime, allow_counter: bool
) -> list[tuple[int, bool]]:
    """Return (direction, is_counter) pairs to evaluate.

    With-trend first; optional counter-trend at the opposite extreme.
    """
    if regime == RangeRegime.BULL_TREND_RANGE:
        out = [(1, False)]
        if allow_counter:
            out.append((-1, True))
        return out
    if regime == RangeRegime.BEAR_TREND_RANGE:
        out = [(-1, False)]
        if allow_counter:
            out.append((1, True))
        return out
    return []


def _side_location_ok(
    direction: int,
    px: float,
    mid: float,
    band: float,
    range_low: float,
    range_high: float,
    tol: float,
    m_low: float,
    m_high: float,
    long_z: list[tuple[float, float]],
    short_z: list[tuple[float, float]],
) -> tuple[bool, list[tuple[float, float]]]:
    """Half-rule + AOI check for one side. Returns (ok, zones)."""
    if direction > 0:
        if px > mid - band:
            return False, []
        zones = list(long_z) + [(range_low - tol, range_low + tol)]
    else:
        if px < mid + band:
            return False, []
        zones = list(short_z) + [(range_high - tol, range_high + tol)]

    in_aoi = any(
        z[0] <= px <= z[1] or (m_low <= z[1] and m_high >= z[0]) for z in zones
    )
    if in_aoi:
        return True, zones
    tagged = (direction > 0 and m_low <= range_low + tol) or (
        direction < 0 and m_high >= range_high - tol
    )
    return tagged, zones


def run_backtest(
    df_1m: pd.DataFrame,
    cfg: TrendRangeRejectionConfig,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    empty: dict[str, Any] = {
        "trades": [],
        "metrics": {},
        "equity_curve": [],
        "signals": [],
    }
    if df_1m.empty or len(df_1m) < 500:
        return empty

    df_1m = df_1m.sort_index()
    if "volume" not in df_1m.columns:
        df_1m = df_1m.copy()
        df_1m["volume"] = (df_1m["high"] - df_1m["low"]).clip(lower=0)

    df_h1 = resample_ohlcv(df_1m, "1h")
    if len(df_h1) < cfg.lookback_hours + cfg.atr_period + 5:
        return empty

    eng_h1 = MarketStructureEngine(cfg.swing_length_h1)
    eng_m1 = MarketStructureEngine(cfg.swing_length_m1)
    eng_h1.precompute(df_h1)
    eng_m1.precompute(df_1m)

    atr_h1 = atr_indicator(
        df_h1["high"], df_h1["low"], df_h1["close"], cfg.atr_period
    ).to_numpy(dtype=float)
    atr_m1 = atr_indicator(
        df_1m["high"], df_1m["low"], df_1m["close"], cfg.atr_period
    ).to_numpy(dtype=float)

    h_o = df_h1["open"].to_numpy(dtype=float)
    h_h = df_h1["high"].to_numpy(dtype=float)
    h_l = df_h1["low"].to_numpy(dtype=float)
    h_c = df_h1["close"].to_numpy(dtype=float)
    h_v = df_h1["volume"].to_numpy(dtype=float)
    h_idx = df_h1.index

    m_o = df_1m["open"].to_numpy(dtype=float)
    m_h = df_1m["high"].to_numpy(dtype=float)
    m_l = df_1m["low"].to_numpy(dtype=float)
    m_c = df_1m["close"].to_numpy(dtype=float)
    m_v = df_1m["volume"].to_numpy(dtype=float)
    m_idx = df_1m.index

    vol_ma = (
        pd.Series(m_v)
        .rolling(cfg.vol_ma_period, min_periods=max(5, cfg.vol_ma_period // 2))
        .mean()
        .to_numpy(dtype=float)
    )

    open_unix_1 = _unix_seconds(pd.DatetimeIndex(m_idx))
    close_unix_1 = open_unix_1 + 60.0
    open_unix_h = _unix_seconds(pd.DatetimeIndex(h_idx))
    close_unix_h = open_unix_h + 3600.0

    rm = RiskManager(cfg, account_size=cfg.account_size, backtest_mode=True)
    trades: list[dict] = []
    signals: list[dict] = []
    equity_curve: list[float] = []
    open_pos: Position | None = None
    setup: _Setup | None = None
    trades_day = 0
    day_key = ""

    n = len(df_1m)
    warmup = max(cfg.vol_ma_period + 50, 300)

    for i in range(warmup, n):
        ts = pd.Timestamp(m_idx[i])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        bar_close_unix = float(close_unix_1[i])
        dk = ts.strftime("%Y-%m-%d")
        if dk != day_key:
            day_key = dk
            trades_day = 0

        if open_pos is not None and open_pos.status == "open":
            manage_position(
                open_pos,
                i,
                bar_close_unix,
                float(m_c[i]),
                float(m_h[i]),
                float(m_l[i]),
                cfg,
                rm,
                trades,
            )
            if open_pos.status == "closed":
                open_pos = None

        equity_curve.append(float(rm.state.equity))

        # Session filter
        if cfg.session_start_utc is not None and cfg.session_end_utc is not None:
            hour = int(ts.hour)
            if not (cfg.session_start_utc <= hour < cfg.session_end_utc):
                continue

        # Pending limit fill / timeout
        if (
            open_pos is None
            and setup is not None
            and setup.mss_done
            and setup.limit is not None
            and setup.sl is not None
            and setup.tp is not None
            and setup.armed_i is not None
        ):
            if i - setup.armed_i > cfg.entry_timeout_bars:
                setup = None
            else:
                filled = False
                fill_px = float(setup.limit)
                if setup.direction > 0 and float(m_l[i]) <= fill_px <= float(m_h[i]):
                    filled = True
                elif setup.direction < 0 and float(m_l[i]) <= fill_px <= float(m_h[i]):
                    filled = True
                day_ok = cfg.max_trades_per_day <= 0 or trades_day < cfg.max_trades_per_day
                if filled and day_ok:
                    atr_e = float(atr_m1[i]) if np.isfinite(atr_m1[i]) else abs(fill_px) * 0.001
                    direction = "long" if setup.direction > 0 else "short"
                    open_pos = rm.open_position(
                        entry_price=fill_px,
                        atr=atr_e,
                        direction=direction,
                        entry_time=bar_close_unix,
                        entry_idx=i,
                        stop_loss=float(setup.sl),
                        tp1=float(setup.tp),
                    )
                    if open_pos is not None:
                        open_pos.meta = {
                            "regime": setup.regime.value,
                            "setup": "trr_cb",
                            "hour_key": setup.hour_key,
                            "counter_trend": setup.counter_trend,
                        }
                        trades_day += 1
                        signals.append(
                            {
                                "time": str(ts),
                                "direction": direction,
                                "limit": fill_px,
                                "sl": setup.sl,
                                "tp": setup.tp,
                                "regime": setup.regime.value,
                            }
                        )
                        if verbose:
                            print(
                                f"  [trr] {direction} @ {fill_px:.5f} "
                                f"SL={setup.sl:.5f} TP={setup.tp:.5f} "
                                f"regime={setup.regime.value}"
                            )
                    setup = None
                    continue

        if open_pos is not None:
            continue
        if cfg.max_trades_per_day > 0 and trades_day >= cfg.max_trades_per_day:
            continue

        # Map to last fully closed H1 bar
        i_h = last_closed_bar_idx(close_unix_h, bar_close_unix)
        if i_h < cfg.lookback_hours + 2:
            continue
        atr_h = float(atr_h1[i_h]) if np.isfinite(atr_h1[i_h]) else 0.0
        if atr_h <= 0:
            continue

        # --- Regime from H1 swings in lookback ---
        snap = eng_h1.snapshot_at(df_h1, i_h)
        # Collect swing prices confirmed in lookback window
        lb0 = max(0, i_h - cfg.lookback_hours)
        highs = [
            float(s.price)
            for s in eng_h1._highs
            if s.confirmed_idx <= i_h and lb0 <= s.idx <= i_h
        ]
        lows = [
            float(s.price)
            for s in eng_h1._lows
            if s.confirmed_idx <= i_h and lb0 <= s.idx <= i_h
        ]
        # Prefer engine last swings if window thin
        if len(highs) < 2 and snap.last_swing_high is not None:
            highs = [float(s.price) for s in eng_h1._highs if s.confirmed_idx <= i_h][-4:]
        if len(lows) < 2 and snap.last_swing_low is not None:
            lows = [float(s.price) for s in eng_h1._lows if s.confirmed_idx <= i_h][-4:]

        regime = classify_regime(highs, lows, cfg.min_pullback_pct)
        if regime == RangeRegime.OFF:
            setup = None
            continue

        range_high = max(highs[-3:]) if highs else float(h_h[lb0 : i_h + 1].max())
        range_low = min(lows[-3:]) if lows else float(h_l[lb0 : i_h + 1].min())
        if range_high <= range_low:
            continue

        impulse_low = lows[-1] if lows else range_low
        impulse_high = highs[-1] if highs else range_high
        mid = 0.5 * (impulse_low + impulse_high)
        band = cfg.avoid_mid_band * abs(impulse_high - impulse_low)
        px = float(m_c[i])
        tol = cfg.aoi_atr_tol * atr_h
        long_z, short_z = h1_flip_levels(
            h_o, h_h, h_l, h_c, i_h, cfg.h1_level_lookback, tol
        )

        # Pick first matching side (with-trend preferred, then counter)
        direction = 0
        is_counter = False
        zones: list[tuple[float, float]] = []
        for cand_dir, cand_counter in _candidate_directions(
            regime, cfg.allow_counter_trend
        ):
            ok, z = _side_location_ok(
                cand_dir,
                px,
                mid,
                band,
                range_low,
                range_high,
                tol,
                float(m_l[i]),
                float(m_h[i]),
                long_z,
                short_z,
            )
            if ok:
                direction = cand_dir
                is_counter = cand_counter
                zones = z
                break
        if direction == 0:
            continue

        # Overextension: H1 candle stretch from open toward extreme
        hour_open = float(h_o[i_h])
        hour_floor = ts.replace(minute=0, second=0, microsecond=0)
        live_open = hour_open
        j_live = last_closed_bar_idx(open_unix_h, open_unix_1[i] + 1)
        if j_live >= 0 and pd.Timestamp(h_idx[j_live]).floor("h") == hour_floor:
            live_open = float(h_o[j_live])
        stretch = abs(px - live_open)
        if stretch < cfg.overext_atr_mult * atr_h:
            continue

        # CB timing
        if not in_timing_window(ts, cfg.timing_minutes, cfg.timing_tolerance_min):
            continue

        # Volume / body stretch on M1
        vma = float(vol_ma[i]) if np.isfinite(vol_ma[i]) else 0.0
        body = abs(float(m_c[i]) - float(m_o[i]))
        atr_m = float(atr_m1[i]) if np.isfinite(atr_m1[i]) else 0.0
        vol_ok = (vma > 0 and float(m_v[i]) >= cfg.vol_mult * vma) or (
            atr_m > 0 and body >= cfg.body_atr_mult * atr_m
        )
        if not vol_ok:
            continue

        hour_key = hour_floor.isoformat()
        if (
            setup is None
            or setup.hour_key != hour_key
            or setup.direction != direction
            or setup.counter_trend != is_counter
        ):
            aoi_lo = min(z[0] for z in zones[-5:]) if zones else range_low - tol
            aoi_hi = max(z[1] for z in zones[-5:]) if zones else range_high + tol
            setup = _Setup(
                direction=direction,
                regime=regime,
                impulse_low=float(impulse_low),
                impulse_high=float(impulse_high),
                range_low=float(range_low),
                range_high=float(range_high),
                aoi_lo=float(aoi_lo),
                aoi_hi=float(aoi_hi),
                hour_key=hour_key,
                counter_trend=is_counter,
            )

        if setup.mss_done:
            continue

        # --- M1 sweep + MSS state machine ---
        snap1 = eng_m1.snapshot_at(df_1m, i)
        if direction > 0:
            # Sweep: break prior swing low inside AOI
            if setup.sweep_extreme is None:
                if snap1.last_swing_low is None:
                    continue
                prior = float(snap1.last_swing_low.price)
                if float(m_l[i]) < prior and float(m_l[i]) <= setup.aoi_hi:
                    setup.sweep_extreme = float(m_l[i])
                    # intervening high = last swing high if any
                    if snap1.last_swing_high is not None:
                        setup.pivot_between = float(snap1.last_swing_high.price)
                    else:
                        setup.pivot_between = float(m_h[i])
            else:
                setup.sweep_extreme = min(setup.sweep_extreme, float(m_l[i]))
                if setup.pivot_between is None and snap1.last_swing_high is not None:
                    setup.pivot_between = float(snap1.last_swing_high.price)
                # MSS: close above pivot between
                if (
                    setup.pivot_between is not None
                    and float(m_c[i]) > setup.pivot_between
                    and vol_ok
                ):
                    low_b = float(setup.sweep_extreme)
                    brk = float(m_h[i])
                    impulse = brk - low_b
                    if impulse <= 0:
                        continue
                    limit = low_b + 0.5 * impulse
                    sl = low_b - cfg.sl_buffer_atr * (atr_m if atr_m > 0 else impulse * 0.1)
                    risk = limit - sl
                    if risk <= 0:
                        continue
                    if cfg.tp_mode == "fixed_rr":
                        tp = limit + cfg.tp1_r * risk
                    else:
                        # 50% of last H1 impulse (conservative trending TP)
                        imp_h1 = abs(setup.impulse_high - setup.impulse_low)
                        tp = limit + 0.5 * imp_h1
                        if tp <= limit:
                            tp = limit + cfg.tp1_r * risk
                    setup.limit = limit
                    setup.sl = sl
                    setup.tp = tp
                    setup.mss_done = True
                    setup.armed_i = i
        else:
            if setup.sweep_extreme is None:
                if snap1.last_swing_high is None:
                    continue
                prior = float(snap1.last_swing_high.price)
                if float(m_h[i]) > prior and float(m_h[i]) >= setup.aoi_lo:
                    setup.sweep_extreme = float(m_h[i])
                    if snap1.last_swing_low is not None:
                        setup.pivot_between = float(snap1.last_swing_low.price)
                    else:
                        setup.pivot_between = float(m_l[i])
            else:
                setup.sweep_extreme = max(setup.sweep_extreme, float(m_h[i]))
                if setup.pivot_between is None and snap1.last_swing_low is not None:
                    setup.pivot_between = float(snap1.last_swing_low.price)
                if (
                    setup.pivot_between is not None
                    and float(m_c[i]) < setup.pivot_between
                    and vol_ok
                ):
                    high_b = float(setup.sweep_extreme)
                    brk = float(m_l[i])
                    impulse = high_b - brk
                    if impulse <= 0:
                        continue
                    limit = high_b - 0.5 * impulse
                    sl = high_b + cfg.sl_buffer_atr * (atr_m if atr_m > 0 else impulse * 0.1)
                    risk = sl - limit
                    if risk <= 0:
                        continue
                    if cfg.tp_mode == "fixed_rr":
                        tp = limit - cfg.tp1_r * risk
                    else:
                        imp_h1 = abs(setup.impulse_high - setup.impulse_low)
                        tp = limit - 0.5 * imp_h1
                        if tp >= limit:
                            tp = limit - cfg.tp1_r * risk
                    setup.limit = limit
                    setup.sl = sl
                    setup.tp = tp
                    setup.mss_done = True
                    setup.armed_i = i

    if open_pos is not None and open_pos.status == "open":
        rm.close_position(
            open_pos, float(m_c[-1]), "eod_flat", float(close_unix_1[-1])
        )
        trades.append(_pos_to_dict(open_pos))

    metrics = _compute_metrics(trades, equity_curve, cfg)  # type: ignore[arg-type]
    metrics["setups"] = len(signals)
    return {
        "trades": trades,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "signals": signals,
        "config": cfg,
    }
