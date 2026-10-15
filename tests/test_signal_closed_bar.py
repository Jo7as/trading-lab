"""Unit tests for signal-loop closed-bar selection."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from trading_lab.signal_loop import bar_timedelta, latest_closed_bar_index


def test_bar_timedelta_1h() -> None:
    assert bar_timedelta("1h") == timedelta(hours=1)
    assert bar_timedelta("60m") == timedelta(hours=1)


def test_latest_closed_when_only_completed_bars() -> None:
    """Dukascopy hist often has no forming bar → last row is already closed."""
    idx = pd.DatetimeIndex(
        [
            "2026-07-24 05:00:00+00:00",
            "2026-07-24 06:00:00+00:00",
            "2026-07-24 07:00:00+00:00",  # 07-08 UTC closed at 08:00
        ]
    )
    now = datetime(2026, 7, 24, 8, 50, tzinfo=timezone.utc)
    i = latest_closed_bar_index(idx, now, timedelta(hours=1))
    assert i == 2  # NOT 1 (old iloc[-2] bug)


def test_latest_closed_skips_forming_bar() -> None:
    idx = pd.DatetimeIndex(
        [
            "2026-07-24 06:00:00+00:00",
            "2026-07-24 07:00:00+00:00",
            "2026-07-24 08:00:00+00:00",  # still forming at 08:50
        ]
    )
    now = datetime(2026, 7, 24, 8, 50, tzinfo=timezone.utc)
    i = latest_closed_bar_index(idx, now, timedelta(hours=1))
    assert i == 1


def test_latest_closed_right_after_hour() -> None:
    idx = pd.DatetimeIndex(
        [
            "2026-07-24 06:00:00+00:00",
            "2026-07-24 07:00:00+00:00",
        ]
    )
    now = datetime(2026, 7, 24, 8, 0, 5, tzinfo=timezone.utc)
    i = latest_closed_bar_index(idx, now, timedelta(hours=1))
    assert i == 1
