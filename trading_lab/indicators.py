"""Indicator Engine — vectorized technical indicators for MCTFB.

All functions take a pandas DataFrame with columns:
  open, high, low, close, volume
and return a pandas Series or DataFrame of the same length.

No look-ahead bias: every indicator at index i uses only data up to and
including index i. Rolling windows use .rolling() which is backward-looking.

Smoothing follows the textbook / TradingView definitions (review 2026-10-27): Wilder
(RSI, ATR, ADX) and EMA are seeded with the SMA of the first ``period`` values and are
NaN before that - see ``seeded_smooth``. Reference values are pinned in
``tests/test_indicators.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def seeded_smooth(x: pd.Series, period: int, alpha: float) -> pd.Series:
    """Recursive smoothing seeded with the SMA of the first ``period`` valid values.

    This is how Wilder (RSI/ATR/ADX, ``alpha = 1/period``) and the standard EMA
    (``alpha = 2/(period+1)``) are defined, and what TradingView's ``ta.rma`` /
    ``ta.ema`` do. ``ewm(adjust=False)`` alone seeds with the FIRST value instead, so
    the first few dozen bars of an EMA(200) or ATR(14) were meaningless numbers that
    looked valid. Values before the seed are NaN (review 2026-10-27).
    """
    x = pd.Series(x, dtype=float)
    pos = np.flatnonzero(x.notna().to_numpy())
    if len(pos) < period:
        return pd.Series(np.nan, index=x.index)
    p = pos[period - 1]
    s = x.copy()
    s.iloc[: p + 1] = np.nan
    s.iloc[p] = float(x.iloc[pos[:period]].mean())
    out = s.ewm(alpha=alpha, adjust=False).mean()
    out.iloc[:p] = np.nan
    return out


def wilder(x: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (RMA): SMA seed, then ``y += (x - y) / period``."""
    return seeded_smooth(x, period, 1.0 / period)


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average, SMA-seeded; NaN during the warm-up."""
    return seeded_smooth(series, period, 2.0 / (period + 1.0))


def rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (SMA-seeded averages). NaN during the warm-up.

    Only gains -> 100, only losses -> 0, flat -> 50. (It used to return 50 whenever
    the average loss was zero, i.e. exactly when RSI should be 100.)
    """
    deltas = closes.diff()
    avg_gain = wilder(deltas.clip(lower=0.0), period)
    avg_loss = wilder((-deltas).clip(lower=0.0), period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.where(avg_loss != 0.0, np.where(avg_gain > 0.0, 100.0, 50.0))
    return out.where(avg_gain.notna())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()],
                     axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range, Wilder's smoothing (SMA seed). NaN during the warm-up."""
    return wilder(true_range(high, low, close), period)


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    """Kaufman efficiency ratio |net move| / sum |moves| over ``n`` bars (0..1)."""
    change = (close - close.shift(n)).abs()
    vol = close.diff().abs().rolling(n).sum()
    return (change / vol.replace(0, np.nan)).fillna(0.0)


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

    tr = true_range(high, low, close)
    # first bar has no previous bar -> no directional movement there
    plus_dm.iloc[0] = minus_dm.iloc[0] = np.nan
    tr = tr.where(close.shift(1).notna())
    atr_w = wilder(tr, period)
    plus_di = 100.0 * wilder(plus_dm, period) / atr_w.replace(0.0, np.nan)
    minus_di = 100.0 * wilder(minus_dm, period) / atr_w.replace(0.0, np.nan)
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan))
    adx_val = wilder(dx, period)

    # Legacy semantics kept on purpose: 0 during the warm-up (callers compare
    # "adx >= threshold"; a NaN would silently flip "block if trending" filters).
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
    """Supertrend (standard definition): ATR bands around hl2 that only tighten,
    direction flips when the close crosses the active band. NaN during the ATR
    warm-up; like TradingView's ``ta.supertrend`` the first valid bar starts down.

    Columns: ``supertrend`` (the active line), ``st_dir`` (+1 up / -1 down).
    """
    n = len(close)
    if n == 0:
        return pd.DataFrame({"supertrend": pd.Series(dtype=float), "st_dir": pd.Series(dtype=float)})
    a = atr(high, low, close, period).to_numpy()
    h, l, c = high.to_numpy(float), low.to_numpy(float), close.to_numpy(float)
    hl2 = (h + l) / 2.0
    ub, lb = hl2 + multiplier * a, hl2 - multiplier * a
    fu, fl = np.full(n, np.nan), np.full(n, np.nan)
    line, d = np.full(n, np.nan), np.full(n, np.nan)
    valid = np.flatnonzero(np.isfinite(a))
    if len(valid):
        k = valid[0]
        fu[k], fl[k] = ub[k], lb[k]
        d[k] = -1.0                      # TradingView ta.supertrend starts in the down state
        line[k] = fu[k]
        for i in range(k + 1, n):
            fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
            fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
            if d[i - 1] > 0:
                d[i] = -1.0 if c[i] < fl[i] else 1.0
            else:
                d[i] = 1.0 if c[i] > fu[i] else -1.0
            line[i] = fl[i] if d[i] > 0 else fu[i]
    return pd.DataFrame({"supertrend": pd.Series(line, index=close.index),
                         "st_dir": pd.Series(d, index=close.index)})


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
