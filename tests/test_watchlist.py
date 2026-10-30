"""Paket 3: power hint and the Benjamini-Hochberg watchlist."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_lab import screening
from trading_lab import watchlist as W


def test_power_grows_with_sharpe_and_years_and_falls_with_trials() -> None:
    assert W.power(1.0, 6) < W.power(1.5, 6) < 1
    assert W.power(0.7, 6) < W.power(0.7, 21)
    assert W.power(0.7, 6, 100) < W.power(0.7, 6, 1)
    assert W.power(0.0, 6, 1) == pytest.approx(0.05, abs=0.001)          # size of the test
    assert W.power(0.5, 0) == 0.0


def test_power_note_warns_only_when_small_edges_are_invisible() -> None:
    assert "kaum nachweisbar" in W.power_note(3, 10)
    assert "kaum nachweisbar" not in W.power_note(30, 1)


def test_family_p_corrects_the_search() -> None:
    assert W.family_p(2.0, 1) == pytest.approx(0.02275, abs=1e-4)
    assert W.family_p(2.0, 10) == pytest.approx(0.2275, abs=1e-3)
    assert W.family_p(0.0, 10) == 1.0


def test_benjamini_hochberg_known_example() -> None:
    p = [0.01, 0.04, 0.03, 0.005, 0.5]
    # sorted .005 .01 .03 .04 .5 against rank/5 x q: .04 sits exactly on the line at q = .05
    assert W.benjamini_hochberg(p, 0.05).tolist() == [True, True, True, True, False]
    assert W.benjamini_hochberg(p, 0.03).tolist() == [True, False, False, True, False]
    assert W.benjamini_hochberg(p, 0.20).tolist() == [True, True, True, True, False]
    assert not W.benjamini_hochberg([0.9, 0.8], 0.2).any()
    assert W.benjamini_hochberg([], 0.2).tolist() == []


def test_noise_gives_a_nearly_empty_list() -> None:
    rng = np.random.default_rng(0)
    hits = [W.benjamini_hochberg(rng.uniform(size=50), 0.2).sum() for _ in range(300)]
    assert np.mean([h > 0 for h in hits]) <= 0.25                         # FDR controlled under the null


def test_build_counts_each_family_once() -> None:
    rows = pd.DataFrame({"family": ["a", "a", "b", "c"], "t": [4.0, 3.9, 0.5, 2.8], "trials": [5, 5, 1, 1]})
    out = W.build(rows)
    assert len(out) == 3 and out.iloc[0]["family"] == "a" and bool(out.iloc[0]["beobachten"])
    assert not bool(out[out["family"] == "b"].iloc[0]["beobachten"])


def test_the_gates_are_untouched() -> None:
    assert (screening.S3_MIN_POS_YEARS, screening.S4_MIN_DSR, screening.S4_MAX_PBO) == (0.70, 0.95, 0.25)
