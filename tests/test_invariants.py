"""Tests for the calendar/session invariants (the bug class from the ORB fade)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_lab.mr_common import day_slices
from trading_lab.invariants import (
    assert_invariants,
    held_past,
    outside_session,
    slice_problems,
    spanning_days,
)

NY = "America/New_York"


def _full_day_index(day: str = "2024-03-05") -> pd.DatetimeIndex:
    return pd.date_range(f"{day} 00:00", f"{day} 23:58", freq="1min", tz=NY).tz_convert("UTC")


def _trade(entry: str, exit_: str, tz: str = NY) -> dict:
    return {"entry_iso": pd.Timestamp(entry, tz=tz).isoformat(),
            "exit_iso": pd.Timestamp(exit_, tz=tz).isoformat(),
            "direction": "long"}


def test_slice_problems_accepts_correct_slices() -> None:
    local = _full_day_index().tz_convert(NY)
    minutes = local.hour * 60 + local.minute
    mask = (minutes >= 570) & (minutes < 960)
    _, slices = day_slices(local, mask)
    assert slice_problems(local, slices, session_open="09:30", session_close="16:00") == []


def test_slice_problems_catches_the_offset_bug() -> None:
    """A masked-subset index applied to the full frame is exactly the old bug."""
    local = _full_day_index().tz_convert(NY)
    # the buggy slices: 390-bar windows starting at the frame's beginning
    bad = [slice(0, 390)]
    problems = slice_problems(local, bad, session_open="09:30", session_close="16:00")
    assert problems and "starts" in problems[0]


def test_outside_session_detects_entries_and_exits() -> None:
    good = _trade("2024-03-05 09:35", "2024-03-05 11:00")
    early = _trade("2024-03-05 03:00", "2024-03-05 04:00")
    late = _trade("2024-03-05 15:55", "2024-03-05 17:30")     # exit after the close
    assert outside_session([good], NY, session_open="09:30", session_close="16:00") == []
    bad = outside_session([good, early, late], NY, session_open="09:30", session_close="16:00")
    # `early` violates on both legs (entry 03:00, exit 04:00), `late` on its exit
    assert len(bad) == 3
    assert {b["leg"] for b in bad} == {"entry", "exit"}


def test_spanning_days_flags_overnight_holds() -> None:
    intraday = _trade("2024-03-05 10:00", "2024-03-05 14:00")
    overnight = _trade("2024-03-05 15:00", "2024-03-06 09:35")
    assert spanning_days([intraday], NY) == []
    assert len(spanning_days([intraday, overnight], NY)) == 1


def test_held_past_flat_time() -> None:
    ok = _trade("2024-03-05 10:00", "2024-03-05 15:50")
    late = _trade("2024-03-05 10:00", "2024-03-05 15:58")
    assert held_past([ok], NY, flat_hhmm="15:55") == []
    assert len(held_past([ok, late], NY, flat_hhmm="15:55")) == 1


def test_assert_invariants_raises_with_details() -> None:
    good = _trade("2024-03-05 10:00", "2024-03-05 14:00")
    assert_invariants([good], NY, session_open="09:30", session_close="16:00", flat_hhmm="15:55")
    with pytest.raises(AssertionError):
        assert_invariants([good, _trade("2024-03-05 15:00", "2024-03-06 09:35")], NY,
                          session_open="09:30", session_close="16:00", flat_hhmm="15:55")


def test_orb_rejection_respects_the_session_on_a_full_day_frame() -> None:
    """Cross-strategy guard on real engine output, using a synthetic 24 h frame."""
    from tests.test_orb_rejection_ifvg import _day_full_session
    from trading_lab.strategies.orb_rejection_ifvg import OrbRejConfig, run_backtest

    ov = {0: (99.2, 100.0, 99.1, 99.3), 1: (90.9, 91.0, 90.0, 90.8),
          20: (99.5, 100.2, 99.0, 99.05), 24: (99.5, 100.2, 99.0, 99.05),
          28: (99.5, 100.2, 99.0, 99.05), 30: (98.0, 98.9, 98.0, 98.0),
          31: (98.5, 100.0, 98.5, 99.1), 32: (99.5, 99.9, 99.2, 99.4),
          33: (99.4, 99.5, 99.2, 99.3), 34: (99.6, 99.7, 98.7, 98.8),
          35: (99.4, 99.5, 97.2, 97.4)}
    for m in range(36, 390):
        ov[m] = (89.0, 89.0, 89.0, 89.0)
    trades = run_backtest(_day_full_session(ov), OrbRejConfig(mode="naive"))["trades"]
    assert trades, "expected the synthetic setup to trade"
    assert_invariants(trades, NY, session_open="09:30", session_close="16:00", flat_hhmm="15:55")
