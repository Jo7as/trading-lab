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


def _day_frame(*, spike_utc_hour: int | None = None, spike: float = 5.0) -> pd.DataFrame:
    """One full UTC day (starting 00:00 New York) with an optional hour-long spike."""
    n = 1440
    idx = pd.date_range("2024-03-05 00:00", periods=n, freq="1min", tz=NY).tz_convert("UTC")
    rng = np.random.default_rng(1)
    close = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
    df = pd.DataFrame({"open": close, "high": close + 0.01, "low": close - 0.01, "close": close},
                      index=idx)
    df["ask_close"] = df["close"] + 0.02
    if spike_utc_hour is not None:
        m = df.index.hour == spike_utc_hour
        df.loc[m, "close"] += spike
        df.loc[m, "open"] = df["close"]
    return df


def test_clean_masks_corrupt_bars() -> None:
    df = _frame(spike_at=200, spike=5.0, roll_hours=True)
    cleaned = clean(df)
    assert cleaned["close"].isna().sum() >= 1                       # the spike is masked
    # a study reading the cleaned frame sees no fabricated jump
    assert cleaned["close"].pct_change().abs().max() < 0.05


def test_clean_masks_the_roll_window_in_utc_not_in_the_asset_timezone() -> None:
    # The roll is 17:00 New York = 21:00/22:00 UTC. Built from the *asset*
    # timezone, the mask blanked 21:00-23:59 NY (01:00-04:59 UTC) and left the
    # real roll in the data — which is how a fabricated jump survived the gate.
    df = _day_frame()
    roll_utc = df.index.hour == 21
    quiet_late_evening_ny = df.index.tz_convert(NY).hour == 22      # 02:00 UTC
    cleaned = clean(df)
    assert cleaned.loc[roll_utc, "close"].isna().all()
    assert cleaned.loc[quiet_late_evening_ny, "close"].notna().all()


def test_audit_counts_roll_outliers_in_utc() -> None:
    in_roll = audit(_day_frame(spike_utc_hour=21), "TEST", tz=NY)
    outside = audit(_day_frame(spike_utc_hour=3), "TEST", tz=NY)
    assert in_roll.roll_window_outliers >= 1
    assert outside.roll_window_outliers == 0
    assert outside.corrupt_bars >= 1        # still an outlier, just not at the roll
