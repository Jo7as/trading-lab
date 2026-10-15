"""Tests for 15m POI respect engine."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_lab.strategies.smc.poi_respect import (
    RespectParams,
    detect_poi_pairs,
    evaluate_respect_15m,
)


def _synth_bullish_poi_series() -> pd.DataFrame:
    """Build a clear bullish FVG+OB then a respectful retest."""
    rows = []
    # baseline
    for i in range(8):
        c = 100.0 + i * 0.01
        rows.append((c, c + 0.05, c - 0.05, c))
    # bearish OB candle
    rows.append((100.2, 100.25, 99.9, 99.95))
    # impulse mid
    rows.append((99.95, 100.8, 99.9, 100.7))
    # displacement creating bullish FVG (high[i-2]=100.25 < low[i]=100.9)
    rows.append((100.9, 101.2, 100.9, 101.1))
    # move away
    for i in range(5):
        c = 101.2 + i * 0.05
        rows.append((c, c + 0.05, c - 0.02, c))
    # retest into zone then bounce
    rows.append((101.0, 101.05, 100.5, 100.95))  # touch
    rows.append((100.95, 101.4, 100.9, 101.35))  # rejection up

    idx = pd.date_range("2024-01-02", periods=len(rows), freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": 1.0,
        },
        index=idx,
    )


def test_detect_poi_pairs_finds_bullish() -> None:
    df = _synth_bullish_poi_series()
    pois = detect_poi_pairs(df, RespectParams(ob_lookback=5, min_fvg_atr=0.0))
    assert len(pois) >= 1
    assert any(p.direction == 1 for p in pois)


def test_respect_engine_runs() -> None:
    df = _synth_bullish_poi_series()
    # pad with more bars for ATR
    extra = pd.DataFrame(
        {
            "open": np.linspace(101.4, 102.0, 30),
            "high": np.linspace(101.5, 102.1, 30),
            "low": np.linspace(101.3, 101.9, 30),
            "close": np.linspace(101.4, 102.0, 30),
            "volume": 1.0,
        },
        index=pd.date_range("2024-01-03", periods=30, freq="15min", tz="UTC"),
    )
    df = pd.concat([df, extra])
    report = evaluate_respect_15m(df, RespectParams(ob_lookback=5, min_fvg_atr=0.0, horizon_bars=10))
    assert report.n_pois >= 1
    assert "respect_rate" in report.to_dict()


def test_1m_entry_respect_runs() -> None:
    from trading_lab.strategies.smc.poi_respect import evaluate_respect_1m_after_choch

    n = 120
    idx = pd.date_range("2024-01-02", periods=n, freq="1min", tz="UTC")
    # random-walk-ish series — just ensure function completes
    rng = np.random.default_rng(1)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0001, n))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.0002,
            "low": close - 0.0002,
            "close": close,
            "volume": 1.0,
        },
        index=idx,
    )
    report = evaluate_respect_1m_after_choch(
        df,
        RespectParams(ob_lookback=5, horizon_bars=20, min_reaction_atr=0.5, min_fvg_atr=0.0),
        swing_length_1m=2,
        max_chochs=200,
    )
    assert report.to_dict()["respect_rate"] >= 0.0
