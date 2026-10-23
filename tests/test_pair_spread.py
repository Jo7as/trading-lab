"""Smoke test for the pair-spread engine (rejected candidate — kept as a guard).

Checks the two things that matter mechanically: a deviating ratio produces a
trade, and paying the spread (real bid/ask) costs more than the mid-price run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_lab.strategies.pair_spread import PairSpreadConfig, run_backtest

SPREAD = 0.2


def _leg(level: float, ratio_path: np.ndarray, seed: int) -> pd.DataFrame:
    idx = pd.date_range("2024-03-05 09:00", periods=len(ratio_path), freq="15min", tz="UTC")
    rng = np.random.default_rng(seed)
    close = level * ratio_path * (1 + rng.normal(0, 1e-4, len(ratio_path))).cumprod()
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=idx)
    df["ask_close"] = df["close"] + SPREAD
    return df


def _pair() -> dict[str, pd.DataFrame]:
    n = 400
    base = np.ones(n)
    # A/B deviates then reverts, twice
    dev = np.zeros(n)
    dev[120:150] = np.linspace(0, 0.004, 30)
    dev[150:180] = np.linspace(0.004, 0, 30)
    dev[240:270] = np.linspace(0, -0.004, 30)
    dev[270:300] = np.linspace(-0.004, 0, 30)
    return {"A": _leg(100.0, base + dev, 1), "B": _leg(50.0, base, 2)}


def test_spread_engine_trades_and_costs_matter() -> None:
    frames = _pair()
    real = run_backtest(frames, PairSpreadConfig(warmup=60), pair=("A", "B"))
    mid = run_backtest(frames, PairSpreadConfig(warmup=60, mid_prices=True), pair=("A", "B"))

    assert real["metrics"]["total_trades"] >= 1          # the deviation is traded
    assert mid["metrics"]["total_pnl_pct"] > real["metrics"]["total_pnl_pct"]  # spreads hurt
    first = real["trades"][0]
    assert first["direction"] in ("long_spread", "short_spread")
    assert first["exit_reason"] in ("exit", "stop", "time", "flat")
