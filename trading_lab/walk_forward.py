"""Rolling walk-forward splits for the evolution workers.

A single fixed 30 % holdout is not a holdout: ``best_oos`` is the maximum over
unbounded rounds on that one window, so the window gets optimised — just slowly.
This module replaces it with a set of rolling folds; each round trains on
everything up to ``train_end`` and validates on the next slice of history, then
the window advances.

Layout, from the end of the history backwards: every fold has a ``test_frac``
long test window, the folds advance by ``step_frac``, and the last fold's test
window ends at the last bar. ``step_frac`` is capped so that even the **first**
fold trains on at least ``min_train_frac`` of the history — with 5 folds, a 20 %
test window and a 50 % minimum train the step becomes 7.5 % per fold. (The first
specification said 15 %; that leaves fold 0 training on 20 % of the data, which is
why it is derived rather than fixed.)

What this does *not* fix: with ``n_folds`` folds the windows are revisited every
``n_folds`` rounds, so the union of test windows is still seen repeatedly. The
honest answer to that is a frozen confirm window that no round may touch
(``backtests/backtest_orb_2r.py`` keeps one) — this module only removes the
*continuous* maximisation of a single window and makes the fold a recorded fact
of every promotion.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

N_FOLDS = 5
TEST_FRAC = 0.20
STEP_FRAC = 0.15
MIN_TRAIN_FRAC = 0.50


@dataclass(frozen=True)
class Fold:
    """One train/test split of the history."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp          # == the test window's start
    test_end: pd.Timestamp

    @property
    def test_start(self) -> pd.Timestamp:
        return self.train_end

    def describe(self) -> str:
        return (f"fold {self.index}: train {self.train_start:%Y-%m-%d}..{self.train_end:%Y-%m-%d} "
                f"| test {self.test_start:%Y-%m-%d}..{self.test_end:%Y-%m-%d}")


def folds(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    n_folds: int = N_FOLDS,
    test_frac: float = TEST_FRAC,
    step_frac: float = STEP_FRAC,
    min_train_frac: float = MIN_TRAIN_FRAC,
) -> list[Fold]:
    """Rolling folds over ``[start, end]``, laid out backwards from ``end``."""
    span = end - start
    if span <= pd.Timedelta(0) or n_folds < 1:
        return []
    head_room = max(0.0, 1.0 - min_train_frac - test_frac)
    step = min(step_frac, head_room / max(1, n_folds - 1))
    out: list[Fold] = []
    for k in range(n_folds):
        test_end = end - span * step * (n_folds - 1 - k)
        test_start = test_end - span * test_frac
        if test_start - start < span * min_train_frac:
            continue
        out.append(Fold(index=k, train_start=start, train_end=test_start, test_end=test_end))
    return out


def fold_for_round(all_folds: list[Fold], round_index: int) -> Fold:
    """The fold round ``round_index`` (1-based) is validated on, cycling forward."""
    if not all_folds:
        raise ValueError("no folds: the history is too short for a walk-forward split")
    return all_folds[(max(1, round_index) - 1) % len(all_folds)]
