"""Signal Generator — MCTFB entry and exit signals.

Pure functions: take indicator DataFrame, return signal DataFrames.
No state, no side effects, no look-ahead bias.

Entry signals are evaluated at candle CLOSE. The trade is executed at
the NEXT candle OPEN to avoid look-ahead bias in backtest mode.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MCTFBConfig
from .indicators import compute_all_indicators


def generate_entry_signals(df: pd.DataFrame, cfg: MCTFBConfig) -> pd.DataFrame:
    """Generate long/short entry signals.

    Returns df with columns:
      long_signal:  1.0 when all 5 confluence conditions are met (long)
      short_signal: 1.0 when all 5 confluence conditions are met (short)

    All conditions evaluated at candle close. shift(1) applied to squeeze
    to avoid using the current candle's bandwidth in the percentile calc
    (slight extra safety against look-ahead).
    """
    out = df.copy()

    # 1. Trend filter
    long_trend = out["trend_up"] > 0
    short_trend = out["trend_down"] > 0

    # 2. Squeeze (shift by 1 to confirm squeeze existed BEFORE breakout candle)
    squeeze_active = out["squeeze"].shift(1) > 0

    # 3. Breakout: close above BB upper (long) / below BB lower (short)
    long_breakout = out["close"] > out["bb_upper"]
    short_breakout = out["close"] < out["bb_lower"]

    # 4. Volume confirmation
    vol_ok = out["vol_above_avg"] > 0

    # 5. Candle body filter
    body_ok = out["body_ratio"] > cfg.min_body_ratio

    # Combine
    out["long_signal"] = (long_trend & squeeze_active & long_breakout &
                          vol_ok & body_ok).astype(float)
    out["short_signal"] = (short_trend & squeeze_active & short_breakout &
                           vol_ok & body_ok).astype(float)

    return out


def generate_exit_signals(
    df: pd.DataFrame, cfg: MCTFBConfig,
    entry_price: float, entry_idx: int, direction: str,
    atr_at_entry: float
) -> dict[str, pd.Series]:
    """Generate exit signal series for a specific trade.

    Returns dict with:
      stop_loss: price level for SL (1.5 × ATR from entry)
      tp1: price level for TP1 (1.5R)
      tp2_exit: boolean — True where supertrend flips against the trade
      time_stop: boolean — True after max_hold_hours candles
    """
    n = len(df)
    sl_dist = cfg.sl_atr_mult * atr_at_entry
    risk = sl_dist  # R = SL distance

    if direction == "long":
        sl_price = entry_price - sl_dist
        tp1_price = entry_price + cfg.tp1_r * risk
        tp2_flip = df["st_dir"] < 0  # supertrend turns bearish
    else:
        sl_price = entry_price + sl_dist
        tp1_price = entry_price - cfg.tp1_r * risk
        tp2_flip = df["st_dir"] > 0  # supertrend turns bullish

    # Time stop: max_hold_hours candles after entry (1h candles)
    time_stop_idx = entry_idx + cfg.max_hold_hours
    time_stop = pd.Series(False, index=df.index)
    if time_stop_idx < n:
        time_stop.iloc[time_stop_idx:] = True

    return {
        "stop_loss": pd.Series(sl_price, index=df.index),
        "tp1": pd.Series(tp1_price, index=df.index),
        "tp2_exit": tp2_flip,
        "time_stop": time_stop,
    }


def prepare_signals(df: pd.DataFrame, cfg: MCTFBConfig) -> pd.DataFrame:
    """Compute indicators + generate entry signals in one call."""
    df_with_indicators = compute_all_indicators(df, cfg)
    return generate_entry_signals(df_with_indicators, cfg)
