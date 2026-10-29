"""Statistics of the H11 token study against brute-force versions (no network)."""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from studies.h11_move_forecast_quick import auc  # noqa: E402
from studies.h11_tokens_study import (SHRINK, LoyoScorer, fit_score, grid, shuffle_within,  # noqa: E402
                                      strat_auc)


def test_grid_has_200_unique_readings():
    keys = [r.key for r in grid()]
    assert len(keys) == 200 and len(set(keys)) == 200


def test_strat_auc_matches_brute_force_with_ties():
    rng = np.random.default_rng(0)
    n = 400
    score = rng.integers(0, 6, n).astype(float)
    big = (rng.random(n) < 0.3).astype(float)
    strat = rng.integers(0, 4, n)
    score[:5] = np.nan
    got, per = strat_auc(score, big, strat, 4)
    want, w = 0.0, 0
    for k in range(4):
        m = (strat == k) & ~np.isnan(score)
        a = auc(score[m], big[m])
        assert abs(per[k] - a) < 1e-12
        want += a * m.sum()
        w += m.sum()
    assert abs(got - want / w) < 1e-12


def test_loyo_scorer_matches_naive():
    rng = np.random.default_rng(1)
    n = 300
    code = rng.integers(-1, 7, n)
    year = rng.integers(2016, 2020, n)
    big = (rng.random(n) < 0.25).astype(float)
    got = LoyoScorer(code, year).score(big)
    for i in range(n):
        if code[i] < 0:
            assert np.isnan(got[i])
            continue
        tr = (year != year[i]) & (code >= 0)
        base = big[tr].mean()
        m = tr & (code == code[i])
        want = (big[m].sum() + SHRINK * base) / (m.sum() + SHRINK)
        assert abs(got[i] - want) < 1e-12


def test_fit_score_unseen_token_gets_base_and_missing_stays_nan():
    tr = np.array([1, 1, 2, 2, -1])
    b = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
    out = fit_score(tr, b, np.array([1, 3, -1]))
    base = 0.5
    assert abs(out[0] - (2 + SHRINK * base) / (2 + SHRINK)) < 1e-12
    assert out[1] == base and np.isnan(out[2])


def test_shuffle_within_keeps_group_counts():
    rng = np.random.default_rng(2)
    g = rng.integers(0, 5, 200)
    b = (rng.random(200) < 0.3).astype(float)
    s = shuffle_within(b, g, rng)
    for k in range(5):
        assert s[g == k].sum() == b[g == k].sum()
    assert not np.array_equal(s, b)
