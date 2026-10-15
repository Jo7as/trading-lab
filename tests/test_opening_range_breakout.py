"""Tests for Opening Range Breakout."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_lab.strategies.opening_range_breakout import (
    OpeningRangeBreakoutConfig,
    run_backtest,
)


def _ny_session_5m_day(date: str = "2024-03-05") -> pd.DataFrame:
    """Build one RTH day of 5m bars in UTC from a known NY date (EST=UTC-5)."""
    # 09:30 NY EST = 14:30 UTC
    start = pd.Timestamp(f"{date} 14:30:00", tz="UTC")
    # until ~16:00 NY = 21:00 UTC
    idx = pd.date_range(start, periods=78, freq="5min", tz="UTC")
    # Flat then breakout up after OR
    close = np.full(len(idx), 18000.0)
    # First 3 bars = OR 09:30-09:45: range 17990-18010
    close[:3] = [18000, 18005, 17995]
    high = close + 5
    low = close - 5
    high[0], low[0] = 18010, 17995
    high[1], low[1] = 18008, 17992
    high[2], low[2] = 18005, 17990
    # Bar at 09:45 (index 3): close above OR high
    close[3] = 18025
    high[3], low[3] = 18030, 18010
    # Then drift up for TP
    for i in range(4, len(idx)):
        close[i] = 18025 + (i - 3) * 2
        high[i] = close[i] + 3
        low[i] = close[i] - 3
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1.0,
        },
        index=idx,
    )


def test_orb_long_breakout_smoke() -> None:
    df = _ny_session_5m_day()
    cfg = OpeningRangeBreakoutConfig(
        max_orb_pct=0.05,  # allow wide synthetic range
        tp_range_mult=0.5,
        sl_buffer=0.0,
        breakout_buffer=0.0,
        long_only=False,
    )
    r = run_backtest(df, cfg)
    assert r["metrics"].get("total_trades", 0) >= 1
    t = [x for x in r["trades"] if x.get("status") == "closed"][0]
    assert t["direction"] == "long"


def test_orb_skip_wide_range() -> None:
    df = _ny_session_5m_day()
    cfg = OpeningRangeBreakoutConfig(max_orb_pct=0.0001)  # tiny → skip
    r = run_backtest(df, cfg)
    assert r["metrics"].get("total_trades", 0) == 0
