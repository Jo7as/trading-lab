"""Session-Range-Fade for Gold: fade failed London breakouts of the Asia range.

Rules (all times UTC, bars addressed by their open time):
  1. Asia range = ask high / bid low of bars in [asia_start, asia_end)
  2. Day is tradeable only if range size is within
     [range_min_atr_mult, range_max_atr_mult] × ATR(14, H1) at range close,
     the prior 24h move stayed below trend_day_atr_mult × ATR(daily), the
     window has no data gap and it is not the last trading day of the month
  3. London window [london_start, london_end): a closed bar breaks the range,
     the next closed bar closes back inside → fade at the following bar's open
  4. SL = breakout extreme ± sl_buffer_atr_mult × ATR(14, H1), TP1 = range
     mid (partial, SL to BE), TP2 = opposite range side, flat at london_end

The rulebook's "ATR(14)" is read as H1: the Asia range is ~2.2× H1 ATR in the
median (0.4× daily ATR), so only on H1 does the 1.5–4× band select anything.

open/high/low/close are bid prices, ask_* the ask side: longs buy at ask and
exit on bid, shorts sell at bid and exit on ask.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..backtest import _compute_metrics
from ..indicators import atr as atr_indicator
from ..mr_common import (
    bar_minutes,
    close_at_market,
    epoch_seconds,
    gap_minutes_before,
    manage_stops_and_targets,
    open_sized_position,
    prepare_frame,
    resample_ohlc,
)
from ..risk import Position, RiskManager


@dataclass(frozen=True)
class SessionRangeFadeConfig:
    """Config for the Asia-range / London-fade strategy."""

    asset: str = "XAU/USD"
    asset_label: str = "XAU/USD"
    timeframe: str = "15min"          # M5 or M15

    asia_start_utc_hour: int = 0
    asia_end_utc_hour: int = 7
    london_start_utc_hour: int = 7
    london_end_utc_hour: int = 11

    # Reference ATR for the range-size band and the SL buffer
    atr_timeframe: str = "1h"
    atr_period: int = 14
    range_min_atr_mult: float = 1.5
    range_max_atr_mult: float = 4.0
    sl_buffer_atr_mult: float = 0.3
    tp1_close_pct: float = 0.5        # at range mid; TP2 = opposite range side

    atr_period_daily: int = 14
    trend_day_atr_mult: float = 2.0   # skip after a 24h move > this × daily ATR
    skip_month_end: bool = True
    max_gap_minutes: int = 60

    account_size: float = 10000.0
    risk_per_trade: float = 0.0075
    max_trades_per_day: int = 1
    max_leverage: float = 30.0
    long_only: bool = False

    weekly_circuit_breaker_pct: float = -0.04

    fallback_spread_pct: float = 0.00015  # only when the frame has no ask side
    slippage_pct: float = 0.00002

    # RiskConfig Protocol compat (structure SL/TP used)
    sl_atr_mult: float = 1.0
    tp1_r: float = 1.0
    max_daily_loss: float = 0.03
    cooldown_hours: int = 0

    @staticmethod
    def search_space() -> list[tuple[str, float, float, float]]:
        return [
            ("range_min_atr_mult", 0.5, 3.0, 0.25),
            ("range_max_atr_mult", 2.0, 8.0, 0.5),
            ("sl_buffer_atr_mult", 0.0, 1.5, 0.1),
            ("trend_day_atr_mult", 0.25, 3.0, 0.25),
        ]


def _daily_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """ATR of completed trading days, indexed by the day it may be used on."""
    day = df.index.normalize()
    # Sunday evening bars belong to Monday's trading day
    day = day.where(df.index.dayofweek != 6, day + pd.Timedelta(days=1))
    daily = df.groupby(day).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    atr_d = atr_indicator(daily["high"], daily["low"], daily["close"], period)
    atr_d.iloc[:period] = np.nan
    return atr_d.shift(1)


def _reference_atr(df: pd.DataFrame, cfg: SessionRangeFadeConfig, at: pd.DatetimeIndex) -> np.ndarray:
    """ATR(cfg.atr_timeframe) from bars fully closed at each timestamp in ``at``."""
    bars = resample_ohlc(df[["open", "high", "low", "close"]], cfg.atr_timeframe)
    atr_ref = atr_indicator(bars["high"], bars["low"], bars["close"], cfg.atr_period)
    atr_ref.iloc[: cfg.atr_period] = np.nan
    last_closed = at - pd.Timedelta(minutes=bar_minutes(cfg.atr_timeframe))
    return atr_ref.reindex(last_closed, method="ffill").to_numpy(dtype=float)


def compute_asia_range(df: pd.DataFrame, cfg: SessionRangeFadeConfig) -> pd.DataFrame:
    """Per-day Asia range with filter verdicts.

    ``df`` must already be a prepared frame (UTC, closed bars, ask side).
    Returns one row per UTC day: range_high (ask), range_low (bid), range_mid,
    range_size, atr_ref, atr_daily, valid, skip_reason.
    """
    mins = bar_minutes(cfg.timeframe)
    hours = df.index.hour
    in_asia = (hours >= cfg.asia_start_utc_hour) & (hours < cfg.asia_end_utc_hour)
    asia = df[in_asia]
    if asia.empty:
        return pd.DataFrame()

    day = asia.index.normalize()
    grouped = asia.groupby(day)
    out = grouped.agg(
        range_high=("ask_high", "max"),
        range_low=("low", "min"),
        first_open=("open", "first"),
    )
    bar_ts = pd.Series(asia.index, index=asia.index).groupby(day)
    out["first_bar"] = bar_ts.first()
    out["last_bar"] = bar_ts.last()

    # Missing minutes anywhere in the window, including both edges
    inner_gap = pd.Series(gap_minutes_before(asia.index, mins), index=asia.index)
    same_day = np.r_[False, day[1:] == day[:-1]]
    out["max_gap"] = inner_gap.where(same_day, 0.0).groupby(day).max()
    win_start = out.index + pd.Timedelta(hours=cfg.asia_start_utc_hour)
    win_end = out.index + pd.Timedelta(hours=cfg.asia_end_utc_hour)
    lead = (out["first_bar"] - win_start).dt.total_seconds() / 60.0
    tail = (win_end - out["last_bar"]).dt.total_seconds() / 60.0 - mins
    out["max_gap"] = np.maximum(out["max_gap"], np.maximum(lead, tail))

    out["range_size"] = out["range_high"] - out["range_low"]
    out["range_mid"] = (out["range_high"] + out["range_low"]) / 2.0
    out["atr_ref"] = _reference_atr(df, cfg, win_end)
    out["atr_daily"] = _daily_atr(df, cfg.atr_period_daily).reindex(out.index)

    # Directional move over the 24h before the range starts
    ref_ts = win_start - pd.Timedelta(hours=24)
    pos = df.index.searchsorted(ref_ts, side="left") - 1
    close = df["close"].to_numpy(dtype=float)
    ref_close = np.where(pos >= 0, close[np.clip(pos, 0, None)], np.nan)
    out["move_24h"] = np.abs(out["first_open"].to_numpy(dtype=float) - ref_close)

    # Last business day of the month: the next business day is in another month
    month_end = (out.index + pd.offsets.BDay(1)).month != out.index.month

    reason = pd.Series("", index=out.index, dtype=object)

    def mark(mask: Any, label: str) -> None:
        reason.mask((reason == "") & np.asarray(mask), label, inplace=True)

    mark(out["max_gap"] > cfg.max_gap_minutes, "data_gap")
    warm = out[["atr_ref", "atr_daily"]].isna().any(axis=1) | (out["atr_ref"] <= 0) | (out["atr_daily"] <= 0)
    mark(warm, "atr_warmup")
    mark(out["range_size"] <= 0, "zero_range")
    mark(out["range_size"] < cfg.range_min_atr_mult * out["atr_ref"], "range_too_small")
    mark(out["range_size"] > cfg.range_max_atr_mult * out["atr_ref"], "range_too_large")
    mark(out["move_24h"].isna(), "no_24h_reference")
    mark(out["move_24h"] > cfg.trend_day_atr_mult * out["atr_daily"], "trend_day")
    if cfg.skip_month_end:
        mark(month_end, "month_end")

    out["skip_reason"] = reason
    out["valid"] = reason == ""
    return out.drop(columns=["first_bar", "last_bar", "first_open"])


def generate_signals(
    df: pd.DataFrame,
    cfg: SessionRangeFadeConfig,
    ranges: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """All breakout→reversal fades in the London window, in time order.

    Each signal is built from two fully closed bars (breakout, confirmation)
    and is executable at the open of the bar after them. ``run_backtest``
    decides which ones are taken (one per day, no overlap).
    """
    if ranges is None:
        ranges = compute_asia_range(df, cfg)
    if ranges.empty:
        return []

    mins = bar_minutes(cfg.timeframe)
    step = pd.Timedelta(minutes=mins)
    index = df.index
    hours = index.hour.to_numpy()
    bid_o = df["open"].to_numpy(dtype=float)
    bid_l = df["low"].to_numpy(dtype=float)
    bid_c = df["close"].to_numpy(dtype=float)
    ask_o = df["ask_open"].to_numpy(dtype=float)
    ask_h = df["ask_high"].to_numpy(dtype=float)
    ask_c = df["ask_close"].to_numpy(dtype=float)

    signals: list[dict[str, Any]] = []
    for day, rng in ranges[ranges["valid"]].iterrows():
        lo_i = index.searchsorted(day + pd.Timedelta(hours=cfg.london_start_utc_hour), side="left")
        hi_i = index.searchsorted(day + pd.Timedelta(hours=cfg.london_end_utc_hour), side="left")
        # breakout b, confirmation b+1, entry b+2 — all must open inside the window
        if hi_i - lo_i < 3:
            continue
        r_high = float(rng["range_high"])
        r_low = float(rng["range_low"])
        mid = float(rng["range_mid"])
        atr_ref = float(rng["atr_ref"])
        buffer = cfg.sl_buffer_atr_mult * atr_ref
        sess_high = -np.inf
        sess_low = np.inf

        for b in range(lo_i, hi_i - 2):
            c, e = b + 1, b + 2
            sess_high = max(sess_high, ask_h[b])
            sess_low = min(sess_low, bid_l[b])
            if index[c] - index[b] != step or index[e] - index[c] != step:
                continue

            direction: str | None = None
            if ask_h[b] > r_high and ask_c[c] < r_high and not cfg.long_only:
                direction = "short"
                entry = bid_o[e]
                sl = max(sess_high, ask_h[c]) + buffer
                tp2 = r_low
                ok = sl > entry > mid
            elif bid_l[b] < r_low and bid_c[c] > r_low:
                direction = "long"
                entry = ask_o[e]
                sl = min(sess_low, bid_l[c]) - buffer
                tp2 = r_high
                ok = sl < entry < mid
            if direction is None or not ok:
                continue

            signals.append(
                {
                    "day": day,
                    "direction": direction,
                    "signal_time": index[c] + step,
                    "entry_idx": e,
                    "entry_time": index[e],
                    "entry_price": float(entry),
                    "stop_loss": float(sl),
                    "tp1": mid,
                    "tp2": float(tp2),
                    "range_high": r_high,
                    "range_low": r_low,
                    "atr": atr_ref,
                    "flat_idx": int(hi_i),
                }
            )
    return signals


def run_backtest(
    df: pd.DataFrame,
    cfg: SessionRangeFadeConfig,
    verbose: bool = False,
    now: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Backtest on a bid(+ask) frame of ``cfg.timeframe`` bars."""
    data = prepare_frame(df, cfg.timeframe, cfg.fallback_spread_pct, now)
    rm = RiskManager(cfg, account_size=cfg.account_size, backtest_mode=True)
    equity_curve: list[float] = [rm.current_equity]
    trades: list[dict] = []
    empty_stats: dict[str, Any] = {"days_total": 0, "days_valid": 0, "skip_reasons": {}}

    if len(data) == 0:
        return {
            "trades": [],
            "metrics": {**_compute_metrics([], equity_curve, cfg), **empty_stats, "setups": 0},  # type: ignore[arg-type]
            "equity_curve": equity_curve,
            "signals": [],
            "ranges": pd.DataFrame(),
        }

    ranges = compute_asia_range(data, cfg)
    signals = generate_signals(data, cfg, ranges)

    index = data.index
    n = len(data)
    mins = bar_minutes(cfg.timeframe)
    bar_secs = mins * 60.0
    ts_open = epoch_seconds(index)
    bid ={k: data[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close")}
    ask = {k: data[f"ask_{k}"].to_numpy(dtype=float) for k in ("open", "high", "low", "close")}

    last_exit_idx = -1
    day_trades: dict[pd.Timestamp, int] = {}
    week_key: tuple[int, int] | None = None
    week_start_pnl = 0.0
    breaker_hits = 0
    taken: list[dict[str, Any]] = []

    for sig in signals:
        e = int(sig["entry_idx"])
        day = sig["day"]
        if e <= last_exit_idx or day_trades.get(day, 0) >= cfg.max_trades_per_day:
            continue

        iso = day.isocalendar()
        if (iso.year, iso.week) != week_key:
            week_key = (iso.year, iso.week)
            week_start_pnl = rm.state.cumulative_equity_pnl_pct
        week_pnl = rm.state.cumulative_equity_pnl_pct - week_start_pnl
        if week_pnl <= cfg.weekly_circuit_breaker_pct * 100.0:
            breaker_hits += 1
            continue

        allowed, _ = rm.can_trade(ts_open[e])
        if not allowed:
            continue

        is_long = sig["direction"] == "long"
        slip = sig["entry_price"] * cfg.slippage_pct
        entry = sig["entry_price"] + slip if is_long else sig["entry_price"] - slip
        pos: Position | None = open_sized_position(
            rm,
            entry_price=entry,
            stop_loss=sig["stop_loss"],
            tp1=sig["tp1"],
            tp2=sig["tp2"],
            direction=sig["direction"],
            entry_time=ts_open[e],
            entry_idx=e,
            atr=sig["atr"],
            max_leverage=cfg.max_leverage,
            meta={
                "strategy": "session_range_fade",
                "day": day.date().isoformat(),
                "range_high": sig["range_high"],
                "range_low": sig["range_low"],
            },
        )
        if pos is None:
            continue
        day_trades[day] = day_trades.get(day, 0) + 1
        taken.append(sig)
        px = bid if is_long else ask  # exit side

        j = e
        while pos.status == "open" and j < n:
            if j >= sig["flat_idx"]:
                # first bar at/after london_end: flat at its open
                close_at_market(
                    pos, rm, trades, ts=ts_open[j], price=px["open"][j],
                    reason="time_exit", slippage_pct=cfg.slippage_pct,
                )
                break
            manage_stops_and_targets(
                pos, rm, trades,
                ts=ts_open[j] + bar_secs,
                bar_open=px["open"][j], bar_high=px["high"][j], bar_low=px["low"][j],
                slippage_pct=cfg.slippage_pct,
            )
            j += 1
        if pos.status == "open":
            close_at_market(
                pos, rm, trades, ts=ts_open[-1] + bar_secs, price=px["close"][-1],
                reason="backtest_end", slippage_pct=cfg.slippage_pct,
            )
            j = n
        last_exit_idx = min(j, n - 1)
        equity_curve.append(rm.current_equity)
        if verbose:
            t = trades[-1]
            print(
                f"  [trade] {t['meta']['day']} {t['direction']:5s} "
                f"{t['exit_reason']:14s} pnl={t['pnl_pct']:+.3f}%"
            )

    metrics = _compute_metrics(trades, equity_curve, cfg)  # type: ignore[arg-type]
    metrics["setups"] = len(signals)
    metrics["days_total"] = int(len(ranges))
    metrics["days_valid"] = int(ranges["valid"].sum()) if len(ranges) else 0
    metrics["skip_reasons"] = (
        ranges.loc[~ranges["valid"], "skip_reason"].value_counts().to_dict()
        if len(ranges) else {}
    )
    metrics["circuit_breaker_skips"] = breaker_hits
    metrics["leverage_capped_trades"] = sum(
        1 for t in trades if (t.get("meta") or {}).get("leverage_capped")
    )
    return {
        "trades": trades,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "signals": taken,
        "ranges": ranges,
    }
