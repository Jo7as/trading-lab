"""Indicator Engine — vectorized technical indicators for MCTFB.

All functions take a pandas DataFrame with columns:
  open, high, low, close, volume
and return a pandas Series or DataFrame of the same length.

No look-ahead bias: every indicator at index i uses only data up to and
including index i. Rolling windows use .rolling() which is backward-looking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average — uses ewm which is backward-looking."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI — fully vectorized."""
    deltas = closes.diff()
    gains = deltas.clip(lower=0.0)
    losses = (-deltas.clip(upper=0.0))
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range — Wilder's smoothing."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.DataFrame:
    """Wilder ADX with +DI and -DI."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    alpha = 1.0 / period
    atr_w = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_w.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_w.replace(0.0, np.nan)
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan))
    adx_val = dx.ewm(alpha=alpha, adjust=False).mean()

    return pd.DataFrame(
        {
            "adx": adx_val.fillna(0.0),
            "plus_di": plus_di.fillna(0.0),
            "minus_di": minus_di.fillna(0.0),
        },
        index=high.index,
    )

def bollinger_bands(
    closes: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands: middle, upper, lower, bandwidth, %B."""
    middle = closes.rolling(period).mean()
    std = closes.rolling(period).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = (upper - lower) / middle.replace(0.0, np.nan)
    bandwidth = bandwidth.fillna(0.0)
    pct_b = (closes - lower) / (upper - lower).replace(0.0, np.nan)
    pct_b = pct_b.clip(-1.0, 2.0).fillna(0.5)
    return pd.DataFrame({
        "bb_middle": middle,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_bandwidth": bandwidth,
        "bb_pct_b": pct_b,
    }, index=closes.index)


def bb_squeeze(
    closes: pd.Series, bb_period: int = 20, bb_std: float = 2.0,
    lookback: int = 100, percentile: float = 0.20
) -> pd.Series:
    """Detect Bollinger Band squeeze — bandwidth in bottom Nth percentile."""
    bb = bollinger_bands(closes, bb_period, bb_std)
    bw = bb["bb_bandwidth"]
    # rolling percentile rank (0..1)
    pct_rank = bw.rolling(lookback).rank(pct=True)
    return (pct_rank < percentile).astype(float)


def supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series,
    period: int = 10, multiplier: float = 3.0
) -> pd.DataFrame:
    """Supertrend indicator — returns trend direction and support/resistance line.

    Returns DataFrame with columns:
      supertrend: the supertrend line value
      st_dir: +1 for uptrend (green), -1 for downtrend (red)
    """
    a = atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    upper_basic = hl2 + multiplier * a
    lower_basic = hl2 - multiplier * a

    n = len(close)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    st_line = np.full(n, np.nan)
    st_dir = np.ones(n)  # default uptrend

    if n == 0:
        return pd.DataFrame({
            "supertrend": pd.Series(dtype=float),
            "st_dir": pd.Series(dtype=float),
        })

    final_upper[0] = upper_basic.iloc[0]
    final_lower[0] = lower_basic.iloc[0]
    st_line[0] = upper_basic.iloc[0]
    st_dir[0] = 1.0 if close.iloc[0] >= st_line[0] else -1.0

    for i in range(1, n):
        # carry forward if band is tighter than previous
        if upper_basic.iloc[i] < final_upper[i - 1] or close.iloc[i - 1] > final_upper[i - 1]:
            final_upper[i] = upper_basic.iloc[i]
        else:
            final_upper[i] = final_upper[i - 1]

        if lower_basic.iloc[i] > final_lower[i - 1] or close.iloc[i - 1] < final_lower[i - 1]:
            final_lower[i] = lower_basic.iloc[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # determine direction
        if st_dir[i - 1] == 1.0:  # was uptrend
            if close.iloc[i] < final_lower[i]:
                st_dir[i] = -1.0
                st_line[i] = final_upper[i]
            else:
                st_dir[i] = 1.0
                st_line[i] = final_lower[i]
        else:  # was downtrend
            if close.iloc[i] > final_upper[i]:
                st_dir[i] = 1.0
                st_line[i] = final_lower[i]
            else:
                st_dir[i] = -1.0
                st_line[i] = final_upper[i]

    return pd.DataFrame({
        "supertrend": pd.Series(st_line, index=close.index),
        "st_dir": pd.Series(st_dir, index=close.index),
    })


def volume_above_average(
    volume: pd.Series, period: int = 20, multiplier: float = 1.3
) -> pd.Series:
    """Boolean series: True where volume > multiplier × rolling mean.
    If volume is all zeros (forex via yfinance), return all True (skip filter).
    """
    vol_ma = volume.rolling(period).mean()
    nonzero = (volume > 0).sum()
    if nonzero == 0:
        # No volume data (forex) — skip filter
        return pd.Series(1.0, index=volume.index)
    return (volume > vol_ma * multiplier).astype(float)


def candle_body_ratio(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Body as fraction of total candle range. 1.0 = full body, 0.0 = doji."""
    body = (close - open_).abs()
    candle_range = (high - low).replace(0.0, np.nan)
    return (body / candle_range).fillna(0.0)


def compute_all_indicators(
    df: pd.DataFrame, cfg: "MCTFBConfig"
) -> pd.DataFrame:
    """Compute all indicators needed by the MCTFB strategy.

    Expects df with columns: open, high, low, close, volume.
    Returns a copy of df with indicator columns appended.
    """
    out = df.copy()

    # Trend — EMA200 filter (proven: +19.94% PnL, Sharpe 2.2 on Gold 2y backtest)
    out["ema_fast"] = ema(out["close"], cfg.ema_fast)
    out["ema_slow"] = ema(out["close"], cfg.ema_slow)
    out["trend_up"] = ((out["close"] > out["ema_slow"]) &
                       (out["ema_fast"] > out["ema_slow"])).astype(float)
    out["trend_down"] = ((out["close"] < out["ema_slow"]) &
                         (out["ema_fast"] < out["ema_slow"])).astype(float)

    # Bollinger Bands + Squeeze
    bb = bollinger_bands(out["close"], cfg.bb_period, cfg.bb_std)
    out = pd.concat([out, bb], axis=1)
    out["squeeze"] = bb_squeeze(
        out["close"], cfg.bb_period, cfg.bb_std,
        cfg.squeeze_lookback, cfg.squeeze_threshold
    )

    # ATR
    out["atr"] = atr(out["high"], out["low"], out["close"], cfg.atr_period)

    # Volume
    out["vol_above_avg"] = volume_above_average(
        out["volume"], cfg.volume_ma_period, cfg.volume_mult
    )

    # Candle body
    out["body_ratio"] = candle_body_ratio(
        out["open"], out["high"], out["low"], out["close"]
    )

    # Supertrend
    st = supertrend(
        out["high"], out["low"], out["close"],
        cfg.supertrend_period, cfg.supertrend_mult
    )
    out["supertrend"] = st["supertrend"]
    out["st_dir"] = st["st_dir"]

    # RSI (for reference / future use)
    out["rsi"] = rsi(out["close"], 14)

    # Drop NaN rows from warmup (but keep enough for lookback)
    out = out.dropna(subset=["ema_slow", "atr", "bb_bandwidth"]).copy()

    return out
