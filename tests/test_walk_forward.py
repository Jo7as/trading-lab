"""The rolling walk-forward split and the promotion bar."""
from __future__ import annotations

import pandas as pd
import pytest

from trading_lab import promotion, walk_forward


def _span() -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp("2016-01-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")


def test_folds_are_laid_out_backwards_from_the_end() -> None:
    start, end = _span()
    folds = walk_forward.folds(start, end)
    assert len(folds) == walk_forward.N_FOLDS
    assert folds[-1].test_end == end                       # the last fold reaches the end
    span = end - start
    for f in folds:
        assert abs((f.test_end - f.test_start) - span * 0.20) < pd.Timedelta(minutes=1)
        assert f.train_start == start
        assert f.train_end == f.test_start
    # advancing, and every fold trains on at least half the history
    for earlier, later in zip(folds, folds[1:]):
        assert later.test_start > earlier.test_start
        assert later.train_end - later.train_start >= span * 0.50


def test_the_step_is_capped_so_the_first_fold_still_has_a_train_window() -> None:
    # The 15 % step from the first specification would leave fold 0 training on
    # 20 % of the history; the cap brings it to 7.5 %.
    start, end = _span()
    first = walk_forward.folds(start, end)[0]
    assert first.train_end - first.train_start >= (end - start) * 0.50


def test_fold_for_round_cycles_forward() -> None:
    start, end = _span()
    folds = walk_forward.folds(start, end)
    picked = [walk_forward.fold_for_round(folds, r).index for r in range(1, 8)]
    assert picked == [0, 1, 2, 3, 4, 0, 1]


def test_a_short_history_does_not_explode() -> None:
    start = pd.Timestamp("2025-01-01", tz="UTC")
    assert walk_forward.folds(start, start) == []
    tiny = walk_forward.folds(start, start + pd.Timedelta(days=30))
    assert tiny and tiny[-1].test_end == start + pd.Timedelta(days=30)
    with pytest.raises(ValueError):
        walk_forward.fold_for_round([], 1)


def test_promotion_bar_rejects_thin_or_damaging_windows() -> None:
    bar = promotion.PromotionBar()
    good = {"sharpe": 0.9, "max_drawdown_pct": -12.0}
    trades = [{"status": "closed"}] * 30
    assert promotion.check(good, trades, bar) == (True, "")

    ok, why = promotion.check(good, trades[:5], bar)
    assert not ok and "oos trades" in why
    ok, why = promotion.check({"sharpe": 0.1, "max_drawdown_pct": -5.0}, trades, bar)
    assert not ok and "sharpe" in why
    ok, why = promotion.check({"sharpe": 1.0, "max_drawdown_pct": -35.0}, trades, bar)
    assert not ok and "drawdown" in why


def test_the_worker_rolls_its_split_and_the_legacy_flag_restores_it(tmp_path, monkeypatch) -> None:
    from trading_lab.mr_evolution_base import MrEvolutionBase

    idx = pd.date_range("2020-01-01", "2026-01-01", freq="15min", tz="UTC")
    frames = {"15min": pd.DataFrame({"close": 1.0, "ask_close": 1.0001}, index=idx)}

    worker = MrEvolutionBase(state_dir=tmp_path, verbose=False)
    train, test, label, fold = worker.round_split(frames, 1)
    assert fold == 0 and "fold 0" in label
    assert len(train["15min"]) and len(test["15min"])
    assert train["15min"].index[-1] <= test["15min"].index[0]

    _, _, _, later = worker.round_split(frames, 2)
    assert later == 1                                     # the window advances
    _, _, _, wrapped = worker.round_split(frames, 1 + walk_forward.N_FOLDS)
    assert wrapped == 0                                   # ... and cycles

    monkeypatch.setenv("LAB_LEGACY_SPLIT", "1")
    legacy = MrEvolutionBase(state_dir=tmp_path, verbose=False)
    _, _, label, fold = legacy.round_split(frames, 3)
    assert fold is None and "legacy" in label


def test_promotion_bar_reads_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("LAB_BEST_OOS_MIN_TRADES", "40")
    monkeypatch.setenv("LAB_BEST_OOS_MIN_SHARPE", "1.1")
    monkeypatch.setenv("LAB_BEST_OOS_MAX_DD_PCT", "8")
    bar = promotion.PromotionBar.from_env()
    assert (bar.min_trades, bar.min_sharpe, bar.max_dd_pct) == (40, 1.1, 8.0)
    assert "n>=40" in bar.describe()
