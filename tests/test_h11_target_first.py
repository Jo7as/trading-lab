"""Rules of the H11 quick measurement on tiny hand-made paths (no network)."""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from studies.h11_target_first_quick import (  # noqa: E402
    classify_path, first_passage, nearest_untouched_level, rescore)


def test_deep_path_fills_at_half_and_stops_at_low():
    # range L=100, H=110 -> half 105; bar 1 dips to 104
    hi = np.array([109.0, 107.0, 112.0])
    lo = np.array([106.0, 104.0, 105.0])
    op = np.array([108.0, 106.0, 106.0])
    assert classify_path(hi, lo, op, 110.0, 100.0) == ("deep", 1, 105.0, 100.0)


def test_shallow_path_breaks_high_first_with_gap_fill_at_open():
    hi = np.array([109.0, 113.0])
    lo = np.array([106.0, 111.5])
    op = np.array([108.0, 111.5])
    assert classify_path(hi, lo, op, 110.0, 100.0) == ("shallow", 1, 111.5, 105.0)


def test_ambiguous_bar_and_no_event_are_skipped():
    assert classify_path(np.array([111.0]), np.array([104.0]), np.array([108.0]), 110.0, 100.0) is None
    assert classify_path(np.array([109.0]), np.array([106.0]), np.array([108.0]), 110.0, 100.0) is None


def test_first_passage_stop_wins_ties_and_target_not_on_entry_bar():
    # entry bar high already above target: must not count
    hi = np.array([121.0, 112.0, 121.0])
    lo = np.array([104.0, 106.0, 99.0])
    cl = np.array([106.0, 110.0, 100.0])
    assert first_passage(hi, lo, cl, 0, 105.0, 100.0, 120.0) == ("stop", 100.0)
    hi2 = np.array([106.0, 121.0])
    lo2 = np.array([104.0, 106.0])
    assert first_passage(hi2, lo2, np.array([105.0, 120.0]), 0, 105.0, 100.0, 120.0) == ("tp", 120.0)
    assert first_passage(hi2, lo2, np.array([105.0, 111.0]), 0, 105.0, 100.0, 130.0) == ("time", 111.0)


def test_level_must_be_pivot_above_range_and_untouched():
    #           0    1    2    3    4    5    6    7
    h1 = np.array([100, 101, 130, 101, 100, 102, 125, 101, 100.0])
    after = np.r_[np.maximum.accumulate(h1[::-1])[::-1][1:], -np.inf]
    # pivots at 2 (130) and 6 (125); nearest above 110 is 125
    assert nearest_untouched_level(h1, np.maximum(after, 110.0), 110.0) == 125.0
    # a later bar trades through 125 -> the next untouched level above is 130
    h1b = h1.copy()
    h1b[8] = 126.0
    afterb = np.r_[np.maximum.accumulate(h1b[::-1])[::-1][1:], -np.inf]
    assert nearest_untouched_level(h1b, np.maximum(afterb, 110.0), 110.0) == 130.0


def test_rescore_matches_first_passage():
    hi = np.array([106.0, 112.0, 118.0, 121.0])
    lo = np.array([104.0, 106.0, 110.0, 117.0])
    cl = np.array([105.0, 111.0, 117.0, 120.0])
    run = np.maximum.accumulate(np.r_[-np.inf, hi[1:]])
    si = len(lo)
    for tgt in (115.0, 120.0, 125.0):
        out, px = first_passage(hi, lo, cl, 0, 105.0, 100.0, tgt)
        assert rescore(run, si, float(cl[-1]), 105.0, 100.0, tgt) == (px - 105.0) / 5.0


def test_auc_basics():
    from studies.h11_move_forecast_quick import auc
    y = np.array([0, 0, 1, 1])
    assert auc(np.array([1.0, 2.0, 3.0, 4.0]), y) == 1.0
    assert auc(np.array([4.0, 3.0, 2.0, 1.0]), y) == 0.0
    assert auc(np.array([1.0, 1.0, 1.0, 1.0]), y) == 0.5
    assert auc(np.array([np.nan, 2.0, 3.0, 1.0]), y) == 0.5
