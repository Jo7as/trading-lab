"""H12 mining blocks against the scalar reference / hand-made cases (no network)."""
from __future__ import annotations

import numpy as np

from trading_lab.chart_tokens import sax_word
from trading_lab.pattern_mining import (clustered_z, first_passage_moves, mixed_neighbors,
                                        nonoverlap_count, sax_codes, sax_windows)


def test_sax_codes_match_scalar_sax_word():
    rng = np.random.default_rng(0)
    for n, w, a in ((6, 3, 3), (8, 4, 4), (5, 3, 3), (16, 3, 4)):
        x = rng.normal(size=(50, n)).cumsum(axis=1) + 100
        x[3] = 7.0                      # flat row
        got = sax_codes(x, w, a)
        assert got.tolist() == [sax_word(r, w, a) for r in x]
    y = np.array([[1.0, np.nan, 2.0]])
    assert sax_codes(y, 3, 3).tolist() == [-1]


def test_sax_windows_match_scalar_on_variable_windows():
    rng = np.random.default_rng(1)
    v = rng.normal(size=500).cumsum()
    i = np.array([0, 10, 37, 100, 250, 400])
    j = np.array([7, 90, 38, 200, 251, 499])
    got = sax_windows(v, i, j, 3, 3, min_len=3)
    want = [sax_word(v[a:b], 3, 3) if b - a >= 3 else -1 for a, b in zip(i, j)]
    assert got.tolist() == want


def test_sax_windows_flat_and_price_level():
    v = np.r_[np.full(10, 5.0), np.arange(10.0)]
    got = sax_windows(v - 3.0, np.array([0, 10]), np.array([10, 20]), 3, 3)
    assert got[0] == sax_word(np.full(10, 5.0), 3, 3)
    assert got[1] == sax_word(np.arange(10.0), 3, 3)


def test_first_passage_moves():
    hi = np.array([[100.5, 101.2, 99.0], [100.2, 100.3, 100.4], [101.5, 100.0, 100.0], [100.2, 100.0, np.nan]])
    lo = np.array([[99.8, 100.5, 98.0], [99.7, 98.9, 99.0], [98.5, 100.0, 100.0], [99.9, 98.0, np.nan]])
    p = np.full(4, 100.0)
    m = np.full(4, 1.0)
    # row0 up first (bar 1); row1 down (bar 1); row2 both in bar 0 -> 0; row3 down (bar 1)
    assert first_passage_moves(hi, lo, p, m).tolist() == [1, -1, 0, -1]


def test_mixed_neighbors():
    # radices [3,3] (one SAX letter each) + size 3; code of digits (1,0,2) = 1*9 + 0*3 + 2 = 11
    nb = mixed_neighbors(np.array([11]), (3, 3, 3), (True, True, False))
    got = sorted(int(c) for c in nb[0] if c >= 0)
    # digit0 -> 0 or 2 : 2, 20 ; digit1 -> 1 (0 - 1 invalid) : 14
    assert got == [2, 14, 20]
    assert (mixed_neighbors(np.array([-1]), (3,), (True,)) == -1).all()


def test_clustered_z_and_nonoverlap():
    resid = np.array([1.0, 1.0, -1.0, 1.0])
    cl = np.array([0, 0, 1, 2])
    # cluster sums 2, -1, 1 -> se sqrt(6)
    assert abs(clustered_z(resid, cl) - 2.0 / np.sqrt(6)) < 1e-12
    t = np.array([0, 10, 59, 60, 61, 130]) * 60_000_000_000
    assert nonoverlap_count(t, 60 * 60_000_000_000) == 3
