"""H3/H5 entry simulation on hand-made bars: fills only on trade-through, no overlap."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _study():
    spec = importlib.util.spec_from_file_location("entry_study", ROOT / "studies" / "crypto_opening_entry_study.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bars(close, low=None, high=None):
    idx = pd.date_range("2024-01-01", periods=len(close), freq="10s", tz="UTC")
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"close": close, "low": close if low is None else np.asarray(low, float),
                         "high": close if high is None else np.asarray(high, float)}, index=idx)


def test_market_entry_and_exit_after_hold():
    st = _study()
    close = np.full(100, 100.0)
    close[6:] = 101.0                                     # +~100 bp after the signal
    bars = _bars(close)
    sig = pd.DataFrame({"side": [1]}, index=bars.index[[0]])
    res = st.simulate(bars, sig, hold="60s", limit_bp=None, wait="30s")
    assert res["filled"].all()
    assert res["gross_bp"].iloc[0] == pytest.approx(np.log(1.01) * 1e4)


def test_limit_needs_trade_through_and_misses_are_kept():
    st = _study()
    close = np.full(100, 100.0)
    low = close.copy()
    low[2] = 99.90                                        # touches the 10 bp limit exactly: no fill
    bars = _bars(close, low=low)
    sig = pd.DataFrame({"side": [1]}, index=bars.index[[0]])
    res = st.simulate(bars, sig, hold="60s", limit_bp=10.0, wait="30s")
    assert not res["filled"].iloc[0]
    low[2] = 99.85                                        # trades through -> filled at 99.90
    res = st.simulate(_bars(close, low=low), sig, hold="60s", limit_bp=10.0, wait="30s")
    assert res["filled"].iloc[0]
    assert res["gross_bp"].iloc[0] == pytest.approx(np.log(100.0 / 99.90) * 1e4)


def test_signals_inside_a_running_trade_are_skipped():
    st = _study()
    bars = _bars(np.full(100, 100.0))
    sig = pd.DataFrame({"side": [1, -1, 1]}, index=bars.index[[0, 3, 10]])
    res = st.simulate(bars, sig, hold="60s", limit_bp=None)   # 6 bars hold -> bar 3 skipped
    assert list(res["ts"]) == [bars.index[0], bars.index[10]]
