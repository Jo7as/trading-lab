"""Unit tests for ICT Silver Bullet helpers."""
from __future__ import annotations

import pandas as pd

from trading_lab.strategies.silver_bullet import (
    SilverBulletConfig,
    _fvg_in_pd_zone,
    run_backtest,
    silver_bullet_window,
    silver_bullet_window_key,
)


def test_silver_bullet_windows_ny() -> None:
    # 2024-01-03 is Wednesday; construct NY-local times as UTC offsets carefully
    # 10:30 NY in January = EST = UTC-5 → 15:30 UTC
    ts_ny_am = pd.Timestamp("2024-01-03 15:30:00", tz="UTC")
    assert silver_bullet_window(ts_ny_am) == "ny_am"
    assert silver_bullet_window(ts_ny_am, use_ny_am=False) is None

    ts_london = pd.Timestamp("2024-01-03 08:30:00", tz="UTC")  # 03:30 EST
    assert silver_bullet_window(ts_london) == "london"

    ts_pm = pd.Timestamp("2024-01-03 19:30:00", tz="UTC")  # 14:30 EST
    assert silver_bullet_window(ts_pm) == "ny_pm"

    ts_out = pd.Timestamp("2024-01-03 12:00:00", tz="UTC")  # 07:00 EST
    assert silver_bullet_window(ts_out) is None


def test_window_key_stable() -> None:
    ts = pd.Timestamp("2024-01-03 15:30:00", tz="UTC")
    assert silver_bullet_window_key(ts, "ny_am") == "2024-01-03_ny_am"


def test_fvg_pd_zone() -> None:
    assert _fvg_in_pd_zone(1, 1.10, 1.00, 1.20)  # discount
    assert not _fvg_in_pd_zone(1, 1.15, 1.00, 1.20)
    assert _fvg_in_pd_zone(-1, 1.15, 1.00, 1.20)


def test_run_backtest_smoke() -> None:
    """Synthetic series — must run without error."""
    n15 = 200
    n1 = n15 * 15
    idx15 = pd.date_range("2024-01-02", periods=n15, freq="15min", tz="UTC")
    idx1 = pd.date_range("2024-01-02", periods=n1, freq="1min", tz="UTC")
    # mild uptrend noise
    import numpy as np

    rng = np.random.default_rng(0)
    base15 = 2000 + np.cumsum(rng.normal(0, 0.5, n15))
    df15 = pd.DataFrame(
        {
            "open": base15,
            "high": base15 + 1,
            "low": base15 - 1,
            "close": base15 + 0.2,
            "volume": 1.0,
        },
        index=idx15,
    )
    base1 = np.interp(
        np.arange(n1),
        np.linspace(0, n1 - 1, n15),
        base15,
    ) + rng.normal(0, 0.2, n1)
    df1 = pd.DataFrame(
        {
            "open": base1,
            "high": base1 + 0.5,
            "low": base1 - 0.5,
            "close": base1,
            "volume": 1.0,
        },
        index=idx1,
    )
    cfg = SilverBulletConfig(sl_buffer=0.3, require_pd_zone=False, require_displacement=False)
    r = run_backtest(df15, df1, cfg)
    assert "metrics" in r
    assert "trades" in r
