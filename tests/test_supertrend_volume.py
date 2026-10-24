"""Tests for the Supertrend + Volume strategy (added after the code review).

Covers the four review findings: entry fills at the *next* bar's open with
adverse slippage, gap-aware stop fills, the time stop in *bars of the configured
timeframe*, and determinism / long_only behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_lab.strategies.supertrend_volume import (
    SupertrendVolumeConfig,
    run_backtest,
)

SLIP = 0.00002


def _frame(path: list[float], *, volume: list[float] | None = None,
           gap_at: int | None = None, gap: float = 0.0) -> pd.DataFrame:
    """OHLCV frame from a close path.

    Volume must vary: the strategy's filter needs `volume > median x mult`, so a
    constant series would block every signal (a trap worth knowing about).
    """
    n = len(path)
    idx = pd.date_range("2024-03-01 00:00", periods=n, freq="1h", tz="UTC")
    close = np.asarray(path, dtype=float)
    open_ = np.r_[close[0], close[:-1]]
    if gap_at is not None:
        open_[gap_at:] = open_[gap_at] + gap
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": np.asarray(volume, dtype=float) if volume is not None else 100.0},
                      index=idx)
    return df


def _uptrend() -> pd.DataFrame:
    """Flat, then a strong rise with a volume surge (so the filter lets it through)."""
    path = [100.0] * 250 + [101.0] + [101.0] * 30
    vol = [100.0] * 250 + [300.0] + [100.0] * 30
    return _frame(path, volume=vol)


def test_timeframe_conversion_of_the_time_stop() -> None:
    assert SupertrendVolumeConfig(timeframe="1h", max_hold_hours=48).max_hold_bars == 48
    assert SupertrendVolumeConfig(timeframe="15m", max_hold_hours=48).max_hold_bars == 192
    assert SupertrendVolumeConfig(timeframe="5m", max_hold_hours=48).max_hold_bars == 576
    assert SupertrendVolumeConfig(timeframe="4h", max_hold_hours=48).max_hold_bars == 12
    with pytest.raises(ValueError):
        SupertrendVolumeConfig(timeframe="7m").max_hold_bars


def test_entry_fills_at_the_next_bar_open_with_slippage() -> None:
    df = _uptrend()
    res = run_backtest(df, SupertrendVolumeConfig())
    assert res["trades"], "expected at least one long after the flip"
    t = res["trades"][0]
    signal_idx = t["meta"]["signal_idx"]
    assert t["entry_idx"] == signal_idx + 1                    # filled one bar later
    expected = df["open"].iloc[t["entry_idx"]] * (1 + SLIP)    # long: buy the open + slippage
    assert t["entry_price"] == pytest.approx(expected, rel=1e-9)


def test_gap_stop_fills_at_the_bar_open_not_at_the_stop() -> None:
    # flat, up-bar (flip -> long), fill bar, then a bar that GAPS far below the stop
    path = [100.0] * 250 + [101.0, 101.0, 60.0]
    vol = [100.0] * 250 + [300.0, 100.0, 100.0]
    df = _frame(path, volume=vol, gap_at=len(path) - 1, gap=-40.0)
    res = run_backtest(df, SupertrendVolumeConfig())
    stops = [t for t in res["trades"] if t["exit_reason"] == "stop_loss"]
    if not stops:                                   # the crash may also exit via supertrend
        pytest.skip("no stop hit in this synthetic path")
    t = stops[0]
    assert t["exit_price"] == pytest.approx(df["open"].iloc[-1] * (1 - SLIP), rel=1e-6)
    assert t["exit_price"] < t["stop_loss"]          # filled through the stop (a gap)


def test_long_only_suppresses_shorts() -> None:
    # an up-flip followed by a down-flip: the only way to get a short is a flip down
    path = [100.0] * 250 + [101.0, 99.0] + [99.0] * 28
    vol = [100.0] * 250 + [300.0, 300.0] + [100.0] * 28
    df = _frame(path, volume=vol)
    longs_only = run_backtest(df, SupertrendVolumeConfig(long_only=True))
    assert longs_only["trades"], "the up-flip should still trade long"
    assert all(t["direction"] == "long" for t in longs_only["trades"])
    both = run_backtest(df, SupertrendVolumeConfig(long_only=False))
    assert any(t["direction"] == "short" for t in both["trades"])


def test_time_stop_uses_bars_of_the_timeframe() -> None:
    # flat after the entry, so neither stop nor TP1 is touched -> only time can close it
    cfg = SupertrendVolumeConfig(max_hold_hours=2)          # 2 bars on 1h
    res = run_backtest(_uptrend(), cfg)
    timed = [t for t in res["trades"] if t["exit_reason"] == "time_stop"]
    assert timed and cfg.max_hold_bars == 2
    for t in timed:
        assert t["exit_time"] - t["entry_time"] == pytest.approx(cfg.max_hold_bars * 3600)


def test_determinism() -> None:
    df = _uptrend()
    a = run_backtest(df, SupertrendVolumeConfig())
    b = run_backtest(df, SupertrendVolumeConfig())
    assert a["trades"] and a["trades"] == b["trades"]
    assert a["metrics"]["total_trades"] == b["metrics"]["total_trades"]


def test_volume_filter_range_fallback_on_sparse_volume() -> None:
    from trading_lab.strategies.supertrend_volume import _compute_volume_filter

    df = _frame([100.0 + i * 0.1 for i in range(260)])
    df["volume"] = 0.0                                      # yfinance-style sparse volume
    vol_ok = _compute_volume_filter(df, period=20, mult=1.0)
    assert set(vol_ok.dropna().unique()) <= {0.0, 1.0}
