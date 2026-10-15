"""Z-score mean reversion (15m): fade stretched closes back to the SMA.

  z = (close - SMA(z_period)) / STD(z_period)
  Tier A: |z| >= z_entry_tier_a            → full risk
  Tier B: z_entry_tier_b <= |z| < tier_a   → tier_b_size_mult × risk
  No entry while ADX >= adx_block_threshold, STD/close <= min_std_pct, the
  z window spans a data gap, or (optionally) outside the session window.

Signals are read from closed bars only and filled at the next bar's open.
Levels are frozen at entry: TP1 = SMA ∓ tp1_std_mult × STD (partial, SL to
break-even), TP2 = SMA, SL = entry ∓ sl_atr_mult × ATR, time stop after
max_hold_bars.

open/high/low/close are bid prices, ask_* the ask side: longs buy at ask and
exit on bid, shorts sell at bid and exit on ask.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..backtest import _compute_metrics
from ..indicators import adx as compute_adx, atr as atr_indicator
from ..mr_common import (
    bar_minutes,
    close_at_market,
    epoch_seconds,
    gap_minutes_before,
    manage_stops_and_targets,
    open_sized_position,
    prepare_frame,
)
from ..risk import Position, RiskManager


@dataclass(frozen=True)
class MeanReversionConfig:
    """Config for the two-tier z-score mean reversion."""

    asset: str = "GC=F"
    asset_label: str = "XAU/USD"
    timeframe: str = "15min"

    z_period: int = 20
    z_entry_tier_a: float = 2.0
    z_entry_tier_b: float = 1.5
    tier_b_size_mult: float = 0.5

    adx_period: int = 14
    adx_block_threshold: float = 32.0
    min_std_pct: float = 0.0005

    require_reversal_candle: bool = False

    atr_period: int = 14
    sl_atr_mult: float = 1.5
    tp1_std_mult: float = 0.5
    tp1_close_pct: float = 0.5
    max_hold_bars: int = 16

    session_filter_enabled: bool = True
    session_start_utc_hour: int = 7
    session_end_utc_hour: int = 20
    max_gap_minutes: int = 60

    account_size: float = 10000.0
    risk_per_trade: float = 0.01
    max_concurrent_positions: int = 1
    max_leverage: float = 30.0
    long_only: bool = False
    cooldown_bars: int = 0

    fallback_spread_pct: float = 0.00015  # only when the frame has no ask side
    slippage_pct: float = 0.00002

    # RiskConfig Protocol compat
    tp1_r: float = 1.0
    max_daily_loss: float = 0.03
    cooldown_hours: int = 0

    @staticmethod
    def search_space() -> list[tuple[str, float, float, float]]:
        return [
            ("z_entry_tier_a", 1.6, 3.0, 0.1),
            ("z_entry_tier_b", 1.0, 2.4, 0.1),
            ("adx_block_threshold", 20.0, 50.0, 2.0),
            ("sl_atr_mult", 0.8, 3.0, 0.1),
            ("max_hold_bars", 4, 48, 2),
        ]


def prepare_dataframe(df: pd.DataFrame, cfg: MeanReversionConfig) -> pd.DataFrame:
    """Indicator columns on a prepared frame: sma, std, z, adx, atr, filters.

    Every value at row i uses bars up to and including i only.
    """
    out = df.copy()
    mins = bar_minutes(cfg.timeframe)

    out["sma"] = out["close"].rolling(cfg.z_period).mean()
    out["std"] = out["close"].rolling(cfg.z_period).std(ddof=0)
    out["std_ok"] = (out["std"] / out["close"]) > cfg.min_std_pct
    # Dead markets blow z up — only divide where the std filter passed
    out["z"] = ((out["close"] - out["sma"]) / out["std"]).where(out["std_ok"])

    out["adx"] = compute_adx(out["high"], out["low"], out["close"], cfg.adx_period)["adx"]
    out["atr"] = atr_indicator(out["high"], out["low"], out["close"], cfg.atr_period)
    warmup = max(cfg.z_period, 2 * cfg.adx_period, cfg.atr_period)
    out.iloc[:warmup, out.columns.get_indexer(["adx", "atr"])] = np.nan

    gap = pd.Series(gap_minutes_before(out.index, mins) > cfg.max_gap_minutes, index=out.index)
    out["gap_ok"] = gap.rolling(cfg.z_period, min_periods=1).sum() == 0

    # Session is checked on the bar close = the open of the entry bar
    close_hour = (out.index + pd.Timedelta(minutes=mins)).hour
    out["in_session"] = (
        (close_hour >= cfg.session_start_utc_hour)
        & (close_hour < cfg.session_end_utc_hour)
    )
    return out


def generate_signals(df: pd.DataFrame, cfg: MeanReversionConfig) -> pd.DataFrame:
    """long_signal / short_signal (1.0/0.0) and tier ('A' / 'B' / '')."""
    out = prepare_dataframe(df, cfg)
    z = out["z"]
    tier_a = max(cfg.z_entry_tier_a, cfg.z_entry_tier_b)
    tier_b = min(cfg.z_entry_tier_a, cfg.z_entry_tier_b)

    tradeable = (
        out["std_ok"]
        & out["gap_ok"]
        & out["adx"].notna()
        & out["atr"].notna()
        & (out["adx"] < cfg.adx_block_threshold)
    )
    if cfg.session_filter_enabled:
        tradeable &= out["in_session"]

    long_cond = tradeable & (z <= -tier_b)
    short_cond = tradeable & (z >= tier_b)
    if cfg.require_reversal_candle:
        prev_close = out["close"].shift(1)
        long_cond &= out["close"] > prev_close
        short_cond &= out["close"] < prev_close
    if cfg.long_only:
        short_cond &= False

    # Tier A is checked first so a bar can never count for both tiers
    tier = pd.Series("", index=out.index, dtype=object)
    tier[long_cond | short_cond] = "B"
    tier[(long_cond | short_cond) & (z.abs() >= tier_a)] = "A"

    out["long_signal"] = long_cond.astype(float)
    out["short_signal"] = short_cond.astype(float)
    out["tier"] = tier
    return out


def manage_position(
    pos: Position,
    idx: int,
    current_ts: float,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    cfg: MeanReversionConfig,
    rm: RiskManager,
    trades: list[dict],
) -> None:
    """One bar of exits on the exit-side prices: SL, TP1, TP2, time stop."""
    manage_stops_and_targets(
        pos, rm, trades,
        ts=current_ts, bar_open=bar_open, bar_high=bar_high, bar_low=bar_low,
        slippage_pct=cfg.slippage_pct,
    )
    if pos.status == "open" and idx - pos.entry_idx + 1 >= cfg.max_hold_bars:
        close_at_market(
            pos, rm, trades, ts=current_ts, price=bar_close,
            reason="time_stop", slippage_pct=cfg.slippage_pct,
        )


def run_backtest(
    df: pd.DataFrame,
    cfg: MeanReversionConfig,
    verbose: bool = False,
    now: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Backtest on a bid(+ask) frame of ``cfg.timeframe`` bars."""
    data = prepare_frame(df, cfg.timeframe, cfg.fallback_spread_pct, now)
    data = generate_signals(data, cfg)

    rm = RiskManager(cfg, account_size=cfg.account_size, backtest_mode=True)
    trades: list[dict] = []
    equity_curve: list[float] = [rm.current_equity]
    tier_trades = {"A": 0, "B": 0}

    n = len(data)
    if n == 0:
        metrics = _compute_metrics([], equity_curve, cfg)  # type: ignore[arg-type]
        metrics["tier_trades"] = tier_trades
        return {
            "trades": [], "metrics": metrics, "equity_curve": equity_curve,
            "tier_stats": tier_trades, "signals_df": data,
        }

    mins = bar_minutes(cfg.timeframe)
    bar_secs = mins * 60.0
    ts_open = epoch_seconds(data.index)
    contiguous_next = np.r_[np.diff(ts_open) == bar_secs, False]
    bid = {k: data[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close")}
    ask = {k: data[f"ask_{k}"].to_numpy(dtype=float) for k in ("open", "high", "low", "close")}
    sma = data["sma"].to_numpy(dtype=float)
    std = data["std"].to_numpy(dtype=float)
    z = data["z"].to_numpy(dtype=float)
    atr_a = data["atr"].to_numpy(dtype=float)
    long_sig = data["long_signal"].to_numpy() > 0
    short_sig = data["short_signal"].to_numpy() > 0
    tier = data["tier"].to_numpy()

    open_pos: Position | None = None
    pending: int | None = None  # signal bar whose order fills at this bar's open
    last_exit_idx = -10**9
    signals_seen = 0

    for i in range(n):
        if pending is not None:
            s = pending
            pending = None
            is_long = bool(long_sig[s])
            raw = ask["open"][i] if is_long else bid["open"][i]
            slip = raw * cfg.slippage_pct
            entry = raw + slip if is_long else raw - slip
            sl_dist = cfg.sl_atr_mult * atr_a[s]
            if is_long:
                sl, tp1, tp2 = entry - sl_dist, sma[s] - cfg.tp1_std_mult * std[s], sma[s]
                room = tp1 > entry
            else:
                sl, tp1, tp2 = entry + sl_dist, sma[s] + cfg.tp1_std_mult * std[s], sma[s]
                room = tp1 < entry
            # Skip when the open already gapped through the first target
            if room and sl_dist > 0:
                is_tier_a = tier[s] == "A"
                open_pos = open_sized_position(
                    rm,
                    entry_price=entry,
                    stop_loss=sl,
                    tp1=tp1,
                    tp2=tp2,
                    direction="long" if is_long else "short",
                    entry_time=ts_open[i],
                    entry_idx=i,
                    atr=atr_a[s],
                    max_leverage=cfg.max_leverage,
                    size_mult=1.0 if is_tier_a else cfg.tier_b_size_mult,
                    meta={
                        "strategy": "mean_reversion",
                        "tier": tier[s],
                        "z": round(float(z[s]), 4),
                    },
                )
                if open_pos is not None:
                    tier_trades[tier[s]] += 1

        if open_pos is not None:
            px = bid if open_pos.direction == "long" else ask  # exit side
            manage_position(
                open_pos, i, ts_open[i] + bar_secs,
                px["open"][i], px["high"][i], px["low"][i], px["close"][i],
                cfg, rm, trades,
            )
            if open_pos.status == "closed":
                open_pos = None
                last_exit_idx = i
                equity_curve.append(rm.current_equity)
                if verbose:
                    t = trades[-1]
                    print(
                        f"  [trade] {t['direction']:5s} tier={t['meta']['tier']} "
                        f"{t['exit_reason']:14s} pnl={t['pnl_pct']:+.3f}%"
                    )

        if open_pos is None and (long_sig[i] or short_sig[i]):
            signals_seen += 1
            if (
                contiguous_next[i]
                and i - last_exit_idx >= cfg.cooldown_bars
                and rm.can_trade(ts_open[i] + bar_secs)[0]
            ):
                pending = i

    if open_pos is not None and open_pos.status != "closed":
        px = bid if open_pos.direction == "long" else ask
        close_at_market(
            open_pos, rm, trades, ts=ts_open[-1] + bar_secs, price=px["close"][-1],
            reason="backtest_end", slippage_pct=cfg.slippage_pct,
        )
        equity_curve.append(rm.current_equity)

    metrics = _compute_metrics(trades, equity_curve, cfg)  # type: ignore[arg-type]
    metrics["tier_trades"] = tier_trades
    metrics["signal_bars_flat"] = signals_seen
    metrics["leverage_capped_trades"] = sum(
        1 for t in trades if (t.get("meta") or {}).get("leverage_capped")
    )
    return {
        "trades": trades,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "tier_stats": tier_trades,
        "signals_df": data,
    }
