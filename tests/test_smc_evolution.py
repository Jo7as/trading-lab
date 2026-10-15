"""Tests for SMC evolution wiring: liquidity flags, search space, worker helpers."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from trading_lab.exploration import SMC_RESEARCH_SEEDS, random_smc_config
from trading_lab.smc_evolution_worker import SmcEvolutionWorker, _slice_mtf
from trading_lab.strategies.smc.liquidity import LiquidityEngine
from trading_lab.strategies.smc_ict import SmcIctConfig


def _session_df() -> pd.DataFrame:
    """Two NY days of 15m bars covering Asia + London windows."""
    # 2024-01-02 00:00 UTC = 2024-01-01 19:00 NY (Asia start)
    idx = pd.date_range(
        "2024-01-02 00:00:00", periods=96, freq="15min", tz="UTC"
    )
    # Day1 higher, day2 lower — PDH/PDL distinct
    highs = [1.10 + (i % 20) * 0.0001 for i in range(96)]
    lows = [h - 0.0005 for h in highs]
    closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    opens = closes[:]
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100.0] * len(idx),
        },
        index=idx,
    )


def test_liquidity_flags_filter_kinds() -> None:
    df = _session_df()
    all_eng = LiquidityEngine()
    all_eng.build(df)
    kinds_all = {lv.kind for lv in all_eng.levels}
    assert any(k.startswith("asia_") for k in kinds_all)
    assert any(k.startswith("london_") for k in kinds_all)
    assert "pdh" in kinds_all or "pdl" in kinds_all

    asia_only = LiquidityEngine(
        use_asia=True,
        use_london=False,
        use_pd=False,
        use_pw=False,
        use_swing_liq=False,
    )
    asia_only.build(df)
    kinds = {lv.kind for lv in asia_only.levels}
    assert kinds
    assert all(k.startswith("asia_") for k in kinds)

    pd_only = LiquidityEngine(
        use_asia=False,
        use_london=False,
        use_pd=True,
        use_pw=False,
        use_swing_liq=False,
    )
    pd_only.build(df)
    kinds_pd = {lv.kind for lv in pd_only.levels}
    assert kinds_pd <= {"pdh", "pdl"}


def test_search_and_categorical_space() -> None:
    nums = SmcIctConfig.search_space()
    cats = SmcIctConfig.categorical_space()
    names_n = {n for n, *_ in nums}
    names_c = {n for n, _ in cats}
    # Safe knobs present
    assert "entry_edge_offset" in names_n
    assert "sl_buffer" in names_n
    assert "tp1_r" in names_n
    assert "ob_lookback" in names_n
    assert "sl_mode" in names_c
    assert "tp_mode" in names_c
    assert "move_sl_to_be_on_tp1" in names_c
    assert "min_irl_r" in names_n
    assert "use_asia" in names_c
    assert "ob_range" in names_c
    # Playbook locks must not be evolvable
    assert "swing_length_4h" not in names_n
    assert "risk_per_trade" not in names_n
    assert "entry_limit_mode" not in names_c
    assert "use_swing_liq" not in names_c
    assert "nested_choch_vs_pullback" not in names_c
    assert "require_1m_fvg" not in names_c
    assert "long_only" not in names_c


def test_flow_evolution_space() -> None:
    flow = SmcIctConfig.flow_evolution_space()
    names = {n for n, _ in flow}
    assert "flow_a_sweep_close_back" in names
    assert "flow_c_require_15m_mss" in names
    assert "flow_killzone" in names
    assert "entry_limit_mode" not in names


def test_pin_strategy_locks() -> None:
    cfg = SmcIctConfig(
        entry_limit_mode="ob_edge",
        use_swing_liq=True,
        nested_choch_vs_pullback=False,
        require_1m_fvg=False,
        long_only=True,
        swing_length_4h=2,
    )
    pinned = SmcIctConfig.pin_strategy_locks(cfg)
    assert pinned.entry_limit_mode == "fvg_edge"
    assert pinned.use_swing_liq is False
    assert pinned.nested_choch_vs_pullback is True
    assert pinned.require_1m_fvg is True
    assert pinned.long_only is False
    assert pinned.swing_length_4h == 10


def test_mutate_one_changes_single_field() -> None:
    import dataclasses

    worker = SmcEvolutionWorker(state_dir=Path("/tmp/smc-test-state-unused"))
    cfg = SmcIctConfig(asset="EUR/USD", asset_label="EUR/USD", sl_buffer=0.0008)
    found = False
    for _ in range(40):
        mut = worker._mutate_one(cfg)
        if mut is None:
            continue
        new_cfg, var, old, new = mut
        found = True
        assert old != new
        assert getattr(new_cfg, var) == new
        diffs = [
            f.name
            for f in dataclasses.fields(SmcIctConfig)
            if getattr(cfg, f.name) != getattr(new_cfg, f.name)
        ]
        assert diffs == [var]
        break
    assert found


def test_random_smc_config_and_seeds() -> None:
    cfg = random_smc_config()
    assert cfg.asset == "EUR/USD"
    assert any([cfg.use_asia, cfg.use_london, cfg.use_pd, cfg.use_pw])
    assert len(SMC_RESEARCH_SEEDS) >= 4
    for label, seed in SMC_RESEARCH_SEEDS:
        assert seed.asset == "EUR/USD"
        assert label


def test_slice_mtf_aligns() -> None:
    idx_1m = pd.date_range("2020-01-01", periods=1000, freq="1min", tz="UTC")
    idx_15 = pd.date_range("2020-01-01", periods=100, freq="15min", tz="UTC")
    idx_4h = pd.date_range("2020-01-01", periods=20, freq="4h", tz="UTC")

    def _df(idx):
        return pd.DataFrame(
            {
                "open": 1.1,
                "high": 1.11,
                "low": 1.09,
                "close": 1.1,
                "volume": 1.0,
            },
            index=idx,
        )

    a, b, c = _slice_mtf(
        _df(idx_4h),
        _df(idx_15),
        _df(idx_1m),
        idx_1m[100],
        idx_1m[500],
    )
    assert len(c) == 400
    assert a.index[0] >= idx_1m[100]
    assert c.index[-1] < idx_1m[500]


def test_worker_load_save_strategy(tmp_path: Path) -> None:
    # Copy minimal goal/strategy
    (tmp_path / "goal.yaml").write_text(
        "asset: EUR/USD\nasset_label: EUR/USD\nmin_trades_per_week: 1\n"
        "target_trades_per_week: 3\ntarget_return_30d: 0.04\nmax_drawdown: 0.08\n"
        "min_sharpe: 0.5\n"
    )
    baseline = SmcIctConfig(
        asset="EUR/USD", asset_label="EUR/USD", sl_buffer=0.0008, ob_lookback=15
    )
    worker = SmcEvolutionWorker(state_dir=tmp_path, verbose=False)
    worker._save_strategy(baseline, increment=False)
    loaded = worker._load_strategy()
    assert loaded.ob_lookback == 15
    assert loaded.sl_buffer == pytest.approx(0.0008)
    mutated = replace(loaded, ob_lookback=10)
    worker._save_strategy(mutated, increment=True)
    again = worker._load_strategy()
    assert again.ob_lookback == 10
    raw = (tmp_path / "strategy.yaml").read_text()
    assert "version:" in raw
