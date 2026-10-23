"""Tests for the data-quality audit (the corrupt bars that fabricate patterns)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_lab.data_quality import audit, clean

NY = "America/New_York"


def _frame(n: int = 500, *, spike_at: int | None = None, spike: float = 1.0,
           ask_below_bid: bool = False, roll_hours: bool = False) -> pd.DataFrame:
    idx = pd.date_range("2024-03-05 00:00", periods=n, freq="1min", tz=NY).tz_convert("UTC")
    rng = np.random.default_rng(1)
    close = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
    if spike_at is not None:
        close[spike_at:] += spike
    df = pd.DataFrame({"open": close, "high": close + 0.01, "low": close - 0.01, "close": close},
                      index=idx)
    df["ask_close"] = df["close"] + 0.02
    if ask_below_bid:
        df.loc[df.index[10], "ask_close"] = df["close"].iloc[10] - 0.01
    if roll_hours:                      # put a small move into the 22:00-ET roll hour
        m = df.index.tz_convert(NY).hour == 22
        df.loc[m, "close"] += 0.05
        df.loc[m, "open"] = df["close"]
    return df


def test_clean_series_is_ok() -> None:
    rep = audit(_frame(), "TEST", tz=NY)
    assert rep.verdict() == "OK" and rep.corrupt_bars == 0 and rep.ask_lt_bid == 0


def test_extreme_spike_is_flagged_as_corrupt() -> None:
    rep = audit(_frame(spike_at=200, spike=5.0), "TEST", tz=NY)     # +5 on a 100 price
    assert rep.corrupt_bars >= 1
    assert rep.worst_move_pct > 4
    assert "corrupt" in rep.verdict()


def test_ask_below_bid_is_flagged() -> None:
    assert audit(_frame(ask_below_bid=True), "TEST", tz=NY).ask_lt_bid == 1


def test_clean_masks_corrupt_bars_and_the_roll_window() -> None:
    df = _frame(spike_at=200, spike=5.0, roll_hours=True)
    cleaned = clean(df, tz=NY)
    assert cleaned["close"].isna().sum() >= 1                       # the spike is masked
    roll = cleaned.index.tz_convert(NY).hour == 22
    assert cleaned.loc[roll, "close"].isna().all()                  # the roll hour is masked
    # a study reading the cleaned frame sees no fabricated jump
    assert cleaned["close"].pct_change().abs().max() < 0.05
