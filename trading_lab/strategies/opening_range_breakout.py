"""Opening Range Breakout (ORB) for Nasdaq-100 / index CFDs.

Classic RTH model (NY local):
  1. Opening range = high/low of [09:30, 09:30+orb_minutes)
  2. After the range locks, wait for a **bar close** beyond OR high/low
  3. One trade per day; SL = opposite side of OR; TP = ``tp_range_mult`` × OR width
  4. Optional: skip wide ranges; long-only; flatten by EOD

Designed for 5m bars (confirm_tf = bar period). 1m also works.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..backtest import _compute_metrics, _pos_to_dict
from ..indicators import atr as atr_indicator
from ..risk import Position, RiskManager

_NY = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


@dataclass(frozen=True)
class OpeningRangeBreakoutConfig:
    """Config for NY Opening Range Breakout."""

    asset: str = "NQ"
    asset_label: str = "NAS100"

    # Opening range window (NY local), starting at RTH open
    rth_open: str = "09:30"
    orb_minutes: int = 15

    # After OR locks: accept breakout closes until this NY time
    entry_cutoff: str = "11:00"
    # Force flat by this NY time (0 = disabled)
    eod_flat: str = "16:00"

    # Skip day if OR width / mid > this fraction (0 = no filter)
    max_orb_pct: float = 0.008  # 0.8% — Trade That Swing style

    # TP distance = tp_range_mult × (orb_high - orb_low) from entry
    # 0.5 ≈ high-WR TTS variant; 1.0 = classic 1R range extension
    tp_range_mult: float = 0.5
    # Optional second target (0 = off); partial at TP1 then runner to TP2
    tp2_range_mult: float = 0.0
    tp1_close_pct: float = 1.0

    sl_buffer: float = 1.0  # points beyond opposite OR side
    # Breakout must clear OR by this many points (0 = touch/close beyond)
    breakout_buffer: float = 0.0

    long_only: bool = False
    short_only: bool = False

    atr_period: int = 14
    sl_atr_mult: float = 1.0  # RiskConfig compat (structure SL used)
    tp1_r: float = 1.0  # unused for sizing; RiskConfig Protocol

    account_size: float = 10000.0
    risk_per_trade: float = 0.01
    max_daily_loss: float = 0.03
    cooldown_hours: int = 0
    max_concurrent_positions: int = 1

    bar_minutes: int = 5  # expected bar size for close timing

    @staticmethod
    def search_space() -> list[tuple[str, float, float, float]]:
        return [
            ("orb_minutes", 5, 30, 5),
            ("max_orb_pct", 0.004, 0.015, 0.002),
            ("tp_range_mult", 0.5, 2.0, 0.25),
            ("sl_buffer", 0.0, 5.0, 1.0),
            ("breakout_buffer", 0.0, 5.0, 1.0),
        ]


def _parse_hhmm(s: str) -> time:
    h, m = s.strip().split(":")
    return time(int(h), int(m))


def _ts(val: Any) -> pd.Timestamp:
    t = pd.Timestamp(val)
    if t.tzinfo is None:
        return t.tz_localize(_UTC)
    return t


def _to_ny(ts: Any) -> pd.Timestamp:
    return _ts(ts).tz_convert(_NY)


def _ny_day_key(ts: Any) -> str:
    return _to_ny(ts).date().isoformat()


@dataclass
class _DayCtx:
    key: str
    orb_high: float | None = None
    orb_low: float | None = None
    orb_locked: bool = False
    skipped: bool = False
    traded: bool = False
    direction: str | None = None


def manage_position(
    pos: Position,
    idx: int,
    current_ts: float,
    current_price: float,
    current_high: float,
    current_low: float,
    cfg: OpeningRangeBreakoutConfig,
    rm: RiskManager,
    trades: list[dict],
    *,
    force_flat: bool = False,
) -> None:
    if force_flat:
        rm.close_position(pos, current_price, "eod_flat", current_ts)
        trades.append(_pos_to_dict(pos))
        return

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
            if cfg.tp1_close_pct >= 0.999 or pos.tp2 is None:
                rm.close_position(pos, pos.tp1, "tp", current_ts)
                trades.append(_pos_to_dict(pos))
            else:
                rm.close_position(
                    pos, pos.tp1, "tp1_partial", current_ts, partial=True
                )
            return

    if pos.tp1_hit and pos.tp2 is not None and not pos.tp2_hit:
        hit2 = (pos.direction == "long" and current_high >= pos.tp2) or (
            pos.direction == "short" and current_low <= pos.tp2
        )
        if hit2:
            rm.close_position(pos, pos.tp2, "tp2", current_ts)
            pos.tp2_hit = True
            trades.append(_pos_to_dict(pos))


def run_backtest(
    df: pd.DataFrame,
    cfg: OpeningRangeBreakoutConfig,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Causal ORB backtest on a single OHLCV frame (typically 5m)."""
    empty: dict[str, Any] = {
        "trades": [],
        "metrics": {},
        "equity_curve": [],
        "signals": [],
    }
    if df.empty:
        return empty

    out = df.copy()
    if "volume" not in out.columns:
        out["volume"] = 0.0
    atr_s = atr_indicator(out["high"], out["low"], out["close"], cfg.atr_period)

    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    close = out["close"].to_numpy(dtype=float)
    atr_a = atr_s.to_numpy(dtype=float)
    index = out.index

    rth_open = _parse_hhmm(cfg.rth_open)
    entry_cutoff = _parse_hhmm(cfg.entry_cutoff)
    eod_flat = _parse_hhmm(cfg.eod_flat) if cfg.eod_flat else None
    orb_end_minutes = int(rth_open.hour * 60 + rth_open.minute + cfg.orb_minutes)

    bar_td = timedelta(minutes=int(cfg.bar_minutes))
    rm = RiskManager(cfg, account_size=cfg.account_size, backtest_mode=True)
    trades: list[dict] = []
    signals: list[dict] = []
    equity_curve: list[float] = []
    open_pos: Position | None = None
    day: _DayCtx | None = None

    # Accumulators while OR building
    or_hi = -np.inf
    or_lo = np.inf

    n = len(out)
    for i in range(n):
        bar_open = _ts(index[i])
        bar_close_ts = bar_open + bar_td
        ny = bar_close_ts.tz_convert(_NY)
        day_key = ny.date().isoformat()
        ny_t = ny.time()
        minutes = ny.hour * 60 + ny.minute
        current_ts = float(bar_close_ts.timestamp())

        # New session day
        if day is None or day.key != day_key:
            day = _DayCtx(key=day_key)
            or_hi, or_lo = -np.inf, np.inf

        # Manage open trade
        force_flat = False
        if open_pos is not None and open_pos.status == "open":
            if eod_flat is not None and ny_t >= eod_flat:
                force_flat = True
            manage_position(
                open_pos,
                i,
                current_ts,
                float(close[i]),
                float(high[i]),
                float(low[i]),
                cfg,
                rm,
                trades,
                force_flat=force_flat,
            )
            if open_pos.status == "closed":
                open_pos = None

        # Build / lock opening range using bars whose open is in [rth_open, orb_end)
        bar_ny_open = bar_open.tz_convert(_NY)
        bar_open_min = bar_ny_open.hour * 60 + bar_ny_open.minute
        rth_open_min = rth_open.hour * 60 + rth_open.minute

        if (
            not day.orb_locked
            and bar_open_min >= rth_open_min
            and bar_open_min < orb_end_minutes
        ):
            or_hi = max(or_hi, float(high[i]))
            or_lo = min(or_lo, float(low[i]))

        # Lock OR once first bar fully after orb window starts closing
        if (
            not day.orb_locked
            and or_hi > -np.inf
            and bar_open_min >= orb_end_minutes
        ):
            day.orb_high = float(or_hi)
            day.orb_low = float(or_lo)
            day.orb_locked = True
            mid = 0.5 * (day.orb_high + day.orb_low)
            width = day.orb_high - day.orb_low
            if mid > 0 and cfg.max_orb_pct > 0 and (width / mid) > cfg.max_orb_pct:
                day.skipped = True

        # Entries only after lock, before cutoff, one per day
        if (
            open_pos is None
            and day.orb_locked
            and not day.skipped
            and not day.traded
            and ny_t < entry_cutoff
            and bar_open_min >= orb_end_minutes
        ):
            assert day.orb_high is not None and day.orb_low is not None
            buf = float(cfg.breakout_buffer)
            long_sig = close[i] > day.orb_high + buf
            short_sig = close[i] < day.orb_low - buf
            direction: str | None = None
            if long_sig and not cfg.short_only:
                direction = "long"
            elif short_sig and not cfg.long_only:
                direction = "short"

            if direction is not None:
                width = day.orb_high - day.orb_low
                if direction == "long":
                    entry = float(close[i])  # market on close of breakout bar
                    sl = day.orb_low - float(cfg.sl_buffer)
                    tp1 = entry + float(cfg.tp_range_mult) * width
                    tp2 = (
                        entry + float(cfg.tp2_range_mult) * width
                        if cfg.tp2_range_mult > 0
                        else None
                    )
                else:
                    entry = float(close[i])
                    sl = day.orb_high + float(cfg.sl_buffer)
                    tp1 = entry - float(cfg.tp_range_mult) * width
                    tp2 = (
                        entry - float(cfg.tp2_range_mult) * width
                        if cfg.tp2_range_mult > 0
                        else None
                    )

                allowed, _ = rm.can_trade(current_ts)
                if allowed and width > 0:
                    atr_val = (
                        float(atr_a[i])
                        if i < len(atr_a) and atr_a[i] == atr_a[i]
                        else width
                    )
                    pos = rm.open_position(
                        entry_price=entry,
                        atr=atr_val,
                        direction=direction,
                        entry_time=current_ts,
                        entry_idx=i,
                        stop_loss=sl,
                        tp1=tp1,
                        tp2=tp2,
                    )
                    if pos is not None:
                        pos.meta = {
                            "orb_high": day.orb_high,
                            "orb_low": day.orb_low,
                            "orb_width": width,
                            "day": day.key,
                        }
                        open_pos = pos
                        day.traded = True
                        day.direction = direction
                        signals.append(
                            {
                                "ts": str(bar_close_ts),
                                "direction": direction,
                                "entry": entry,
                                "sl": sl,
                                "tp1": tp1,
                                "orb_high": day.orb_high,
                                "orb_low": day.orb_low,
                            }
                        )

        if verbose and i % 5000 == 0 and i > 0:
            print(
                f"  [{100 * i / n:5.1f}%] trades={len(trades)} "
                f"equity={rm.current_equity:.2f}"
            )

        if open_pos is not None or i % 12 == 0:
            equity_curve.append(rm.current_equity)

    if open_pos is not None and open_pos.status != "closed":
        last_ts = float((_ts(index[-1]) + bar_td).timestamp())
        rm.close_position(open_pos, float(close[-1]), "backtest_end", last_ts)
        trades.append(_pos_to_dict(open_pos))

    metrics = _compute_metrics(trades, equity_curve, cfg)  # type: ignore[arg-type]
    metrics["setups"] = len(signals)
    return {
        "trades": trades,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "signals": signals,
    }
