"""VPIN risk-filter panel: features strictly before the hour, targets after it."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _study():
    spec = importlib.util.spec_from_file_location("vpin_study", ROOT / "studies" / "crypto_vpin_risk_filter.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_panel_features_do_not_see_the_target_window():
    st = _study()
    n = 3 * 8640
    idx = pd.date_range("2024-01-01", periods=n, freq="10s", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    bars = pd.DataFrame({"close": close}, index=idx)
    t = pd.Timestamp("2024-01-02 12:00", tz="UTC")
    marker = pd.Series(0.0, index=idx)
    marker[idx >= t] = 1.0                       # a "VPIN" that switches on exactly at t
    panel = st.hourly_panel(bars, marker, marker)
    assert panel.loc[t, "vpin"] == 0.0           # value at t comes from the bar before t
    assert panel.loc[t + pd.Timedelta("1h"), "vpin"] == 1.0
    # absolute 1h return from the last price before t to the last price before t+1h
    k0, k1 = idx.get_loc(t) - 1, idx.get_loc(t + pd.Timedelta("1h")) - 1
    assert panel.loc[t, "absret_1h"] == pytest.approx(abs(np.log(close[k1] / close[k0])) * 1e4)
