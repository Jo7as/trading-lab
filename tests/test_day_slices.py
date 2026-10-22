"""Tests for the shared per-day slicing helper.

Guards the index-space trap that made a whole strategy look profitable: slices
built from a boolean mask must index the FULL frame, not the masked subset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_lab.mr_common import day_slices

NY = "America/New_York"


def _frame(days: tuple[str, ...]) -> pd.DatetimeIndex:
    """A full 24h 1-minute index (00:00–23:58) for each day, in UTC."""
    idx = pd.DatetimeIndex([])
    for d in days:
        idx = idx.append(pd.date_range(f"{d} 00:00", f"{d} 23:58", freq="1min", tz=NY))
    return idx.tz_convert("UTC")


def test_slices_index_the_full_frame_not_the_masked_subset() -> None:
    idx = _frame(("2024-03-05", "2024-03-06"))
    local = idx.tz_convert(NY)
    minutes = local.hour * 60 + local.minute
    mask = (minutes >= 570) & (minutes < 960)              # 09:30–15:59

    days, slices = day_slices(local, mask)

    assert [d.isoformat() for d in days] == ["2024-03-05", "2024-03-06"]
    for sl in slices:
        assert local[sl.start].strftime("%H:%M") == "09:30"     # not 00:00 / 18:00
        assert local[sl.stop - 1].strftime("%H:%M") == "15:59"
        assert sl.stop - sl.start == 390
        assert sl.stop <= len(local)                            # indices into the FULL frame


def test_slices_are_per_day_not_a_rolling_window() -> None:
    """Two days must not share a window: the second slice starts after the first."""
    idx = _frame(("2024-03-05", "2024-03-06", "2024-03-07"))
    local = idx.tz_convert(NY)
    minutes = local.hour * 60 + local.minute
    mask = (minutes >= 570) & (minutes < 960)
    _, slices = day_slices(local, mask)
    assert slices[0].stop <= slices[1].start < slices[2].start
    # the second day's first bar is still 09:30 the next morning (1440 bars later)
    assert local[slices[1].start].strftime("%H:%M") == "09:30"
    assert local[slices[1].start].date().isoformat() == "2024-03-06"


def test_min_bars_drops_short_days() -> None:
    idx = _frame(("2024-03-05", "2024-03-06"))
    local = idx.tz_convert(NY)
    minutes = local.hour * 60 + local.minute
    # on day 2 only 10 session bars exist
    mask = (minutes >= 570) & (minutes < 960)
    mask &= ~((local.normalize().date == pd.Timestamp("2024-03-06").date())
              & (minutes >= 580))
    days, slices = day_slices(local, mask, min_bars=15)
    assert [d.isoformat() for d in days] == ["2024-03-05"]      # day 2 too short


def test_empty_mask() -> None:
    idx = _frame(("2024-03-05",))
    local = idx.tz_convert(NY)
    assert day_slices(local, np.zeros(len(local), dtype=bool)) == ([], [])


def test_orb_2r_session_open_on_a_full_day_frame() -> None:
    """Cross-strategy guard: another session-based strategy must also land on 09:30."""
    from trading_lab.strategies.orb_2r import MARKETS, build_sessions

    idx = _frame(("2024-03-05",))
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                       "volume": 1.0}, index=idx)
    for col in ("open", "high", "low", "close"):
        df[f"ask_{col}"] = df[col] + 1.0
    s = build_sessions(df, MARKETS["US100"])
    expected = pd.Timestamp("2024-03-05 09:30", tz=NY).tz_convert("UTC").value
    assert s.open_ns[0] == expected
    assert s.days[0].isoformat() == "2024-03-05"
