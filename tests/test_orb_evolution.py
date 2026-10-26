"""The ORB evolution worker's rolling walk-forward split."""
from __future__ import annotations

import pandas as pd
import pytest

from trading_lab import walk_forward
from trading_lab.orb_evolution_worker import OrbEvolutionWorker


def _df() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", "2026-01-01", freq="15min", tz="UTC")
    return pd.DataFrame({"close": 1.0}, index=idx)


def test_orb_worker_rolls_its_split_and_the_legacy_flag_restores_it(
    tmp_path, monkeypatch
) -> None:
    # Avoid touching the network / a real adapter; only the split is exercised.
    monkeypatch.setenv("LAB_DATA_SOURCE", "yfinance")
    df = _df()

    worker = OrbEvolutionWorker(state_dir=tmp_path, verbose=False)
    train, test, label, fold = worker.round_split(df, 1)
    assert fold == 0 and "fold 0" in label
    assert len(train) and len(test)
    assert train.index[-1] <= test.index[0]

    _, _, _, later = worker.round_split(df, 2)
    assert later == 1                                     # the window advances
    _, _, _, wrapped = worker.round_split(df, 1 + walk_forward.N_FOLDS)
    assert wrapped == 0                                   # ... and cycles

    monkeypatch.setenv("LAB_LEGACY_SPLIT", "1")
    legacy = OrbEvolutionWorker(state_dir=tmp_path, verbose=False)
    _, _, label, fold = legacy.round_split(df, 3)
    assert fold is None and "legacy" in label


def test_orb_legacy_split_is_the_old_holdout(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LAB_DATA_SOURCE", "yfinance")
    monkeypatch.setenv("LAB_LEGACY_SPLIT", "1")
    df = _df()
    worker = OrbEvolutionWorker(state_dir=tmp_path, verbose=False)
    train, test, split_ts = worker.split(df)
    assert split_ts == df.index[int(len(df) * (1 - worker.oos_split))]
    assert len(train) + len(test) == len(df)
    assert pytest.approx(len(test) / len(df), abs=1e-3) == worker.oos_split
