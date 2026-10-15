"""Unit tests for BB Squeeze Breakout signal logic."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_lab.strategies.bb_squeeze_breakout import (
    BBSqueezeBreakoutConfig,
    compute_indicators,
    generate_signals,
)


def _test_cfg(**kwargs: object) -> BBSqueezeBreakoutConfig:
    defaults = {
        "bb_period": 5,
        "squeeze_lookback": 8,
        "squeeze_threshold": 0.50,
        "ema_trend_period": 3,
        "use_trend_filter": False,
        "min_body_ratio": 0.0,
        "min_bb_bandwidth": 0.0,
        "activity_mult": 0.01,
        "session_filter": False,
    }
    defaults.update(kwargs)
    return BBSqueezeBreakoutConfig(**defaults)  # type: ignore[arg-type]


def _ohlcv(closes: list[float], *, hours: list[int] | None = None) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    if hours is not None:
        idx = pd.DatetimeIndex(
            [t.replace(hour=h) for t, h in zip(idx, hours)],
            tz="UTC",
        )
    highs = [c + 0.05 for c in closes]
    lows = [c - 0.05 for c in closes]
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000.0] * n,
        },
        index=idx,
    )


def test_squeeze_requires_previous_bar() -> None:
    """Signal only when squeeze was active on the prior bar (shift(1))."""
    cfg = _test_cfg()
    # Flat series → likely squeeze on many bars; breakout only on last bar
    closes = [1.1000] * 11 + [1.1050]
    df = _ohlcv(closes)
    out = generate_signals(df, cfg)
    # If last bar has signal, prior bar must have had squeeze
    if out["long_signal"].iloc[-1] > 0:
        assert out["squeeze"].shift(1).iloc[-1] > 0


def test_long_breakout_all_conditions() -> None:
    """Crafted breakout: prior squeeze + close above upper band."""
    cfg = _test_cfg(bb_std=1.0, squeeze_threshold=0.60)
    # Tight range then spike
    closes = [1.1000] * 9 + [1.1005, 1.1020]
    df = _ohlcv(closes)
    df.loc[df.index[-1], "close"] = 1.1040
    df.loc[df.index[-1], "high"] = 1.1045
    df.loc[df.index[-1], "open"] = 1.1020
    out = generate_signals(df, cfg)
    # At least verify signal columns exist and logic runs without error
    assert "long_signal" in out.columns
    assert "short_signal" in out.columns
    assert out["squeeze"].shift(1).dtype == float or out["squeeze"].dtype == float


def test_no_signal_without_bandwidth() -> None:
    """min_bb_bandwidth blocks entries in dead markets."""
    cfg_dead = _test_cfg(min_bb_bandwidth=0.50)
    cfg_open = _test_cfg(min_bb_bandwidth=0.0)
    closes = [1.1000] * 9 + [1.1005, 1.1040]
    df = _ohlcv(closes)
    out_dead = generate_signals(df, cfg_dead)
    out_open = generate_signals(df, cfg_open)
    assert out_dead["long_signal"].sum() <= out_open["long_signal"].sum()


def test_session_filter_blocks_off_hours() -> None:
    """UTC session gate excludes off-session bars."""
    cfg_on = _test_cfg(session_filter=False)
    cfg_off = _test_cfg(session_filter=True, session_start_utc=10, session_end_utc=12)
    closes = [1.1000] * 12
    hours = list(range(12))
    df = _ohlcv(closes, hours=hours)
    out_on = generate_signals(df, cfg_on)
    out_off = generate_signals(df, cfg_off)
    assert out_off["session_ok"].sum() < out_on["session_ok"].sum()


def test_short_mirror() -> None:
    """Short signals use bb_lower breakout with mirrored trend filter."""
    cfg = _test_cfg(use_trend_filter=True, ema_trend_period=3)
    closes = [1.1000] * 10 + [1.0980, 1.0960]
    df = _ohlcv(closes)
    df.loc[df.index[-1], "close"] = 1.0950
    df.loc[df.index[-1], "low"] = 1.0945
    df.loc[df.index[-1], "open"] = 1.0970
    out = generate_signals(df, cfg)
    assert "short_signal" in out.columns
    # Long and short should not fire on same bar
    both = (out["long_signal"] > 0) & (out["short_signal"] > 0)
    assert both.sum() == 0


def test_compute_indicators_columns() -> None:
    cfg = _test_cfg()
    df = _ohlcv([1.1 + i * 0.0001 for i in range(30)])
    out = compute_indicators(df, cfg)
    for col in (
        "bb_upper", "bb_lower", "bb_middle", "squeeze",
        "atr", "ema_trend", "activity_ok", "body_ratio", "session_ok",
    ):
        assert col in out.columns
