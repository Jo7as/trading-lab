"""The ST+Vol evolution worker's rolling walk-forward split and train short gate."""
from __future__ import annotations

import pandas as pd

from trading_lab import walk_forward
from trading_lab.st_evolution_worker import STEvolutionWorker
from trading_lab.strategies.supertrend_volume import SupertrendVolumeConfig


def _df() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", "2026-01-01", freq="1h", tz="UTC")
    return pd.DataFrame({"close": 1.0}, index=idx)


def test_st_worker_rolls_its_split_and_the_legacy_flag_restores_it(
    tmp_path, monkeypatch
) -> None:
    df = _df()

    worker = STEvolutionWorker(state_dir=tmp_path, verbose=False)
    train, test, label, fold = worker.round_split(df, 1)
    assert fold == 0 and "fold 0" in label
    assert len(train) and len(test)
    assert train.index[-1] <= test.index[0]

    _, _, _, later = worker.round_split(df, 2)
    assert later == 1
    _, _, _, wrapped = worker.round_split(df, 1 + walk_forward.N_FOLDS)
    assert wrapped == 0

    monkeypatch.setenv("LAB_LEGACY_SPLIT", "1")
    legacy = STEvolutionWorker(state_dir=tmp_path, verbose=False)
    _, _, label, fold = legacy.round_split(df, 3)
    assert fold is None and "legacy" in label


def test_short_gate_decides_on_train_and_keeps_shorts_when_thin(tmp_path) -> None:
    worker = STEvolutionWorker(state_dir=tmp_path, verbose=False)
    cfg = SupertrendVolumeConfig(long_only=False)

    # Too few train shorts is not evidence against them — keep shorts on.
    thin = [{"status": "closed", "direction": "short", "pnl_pct": -5.0}]
    assert worker._enforce_short_gate(cfg, thin).long_only is False

    # Enough train shorts, but the train short leg loses — then go long-only.
    losing = [{"status": "closed", "direction": "short", "pnl_pct": -1.0}] * 40
    assert worker._enforce_short_gate(cfg, losing).long_only is True
