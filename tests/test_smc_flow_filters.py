"""Tests for SMC flow filter helpers (Stages A–C)."""
from __future__ import annotations

from dataclasses import dataclass

from trading_lab.strategies.smc.flow_filters import (
    dealing_range_from_swings,
    m15_align_ok_legacy,
    m15_mss_ok,
    premium_discount_ok,
    sweep_close_back_ok,
)
from trading_lab.strategies.smc.types import StructureSnapshot, SwingPoint, TrendBias


@dataclass
class _FlowCfg:
    flow_a_sweep_close_back: bool = False
    flow_a_sweep_close_mode: str = "body"
    flow_b_premium_discount: bool = False
    flow_b_range_mode: str = "15m_swing"
    flow_b_equilibrium: float = 0.5
    flow_c_require_15m_mss: bool = False
    flow_c_displacement: bool = False
    flow_c_displacement_atr_mult: float = 1.0
    flow_c_align_fallback: bool = True
    flow_killzone: str = "off"
    flow_ny_session_only: bool = False
    flow_sell_side_sweeps_only: bool = False


def test_sweep_close_back_bullish_pdl() -> None:
    cfg = _FlowCfg(flow_a_sweep_close_back=True)
    # Sell-side raid at 1.10, close back above
    assert sweep_close_back_ok(
        cfg,
        open_=1.101,
        high=1.102,
        low=1.099,
        close=1.1015,
        level=1.10,
        kind="pdl",
    )
    # Breakout close — reject
    assert not sweep_close_back_ok(
        cfg,
        open_=1.099,
        high=1.101,
        low=1.098,
        close=1.0985,
        level=1.10,
        kind="pdl",
    )


def test_sweep_close_back_bearish_pdh() -> None:
    cfg = _FlowCfg(flow_a_sweep_close_back=True)
    assert sweep_close_back_ok(
        cfg,
        open_=1.199,
        high=1.201,
        low=1.198,
        close=1.1995,
        level=1.20,
        kind="pdh",
    )
    assert not sweep_close_back_ok(
        cfg,
        open_=1.201,
        high=1.202,
        low=1.200,
        close=1.2015,
        level=1.20,
        kind="pdh",
    )


def test_sweep_close_back_disabled_passes() -> None:
    cfg = _FlowCfg(flow_a_sweep_close_back=False)
    assert sweep_close_back_ok(
        cfg,
        open_=1.0,
        high=1.0,
        low=0.99,
        close=0.99,
        level=1.0,
        kind="pdh",
    )


def test_premium_discount() -> None:
    cfg = _FlowCfg(flow_b_premium_discount=True)
    lo, hi = 1.0, 1.2
    eq = 1.1
    assert premium_discount_ok(cfg, TrendBias.BULLISH, eq - 0.01, lo, hi)
    assert not premium_discount_ok(cfg, TrendBias.BULLISH, eq + 0.01, lo, hi)
    assert premium_discount_ok(cfg, TrendBias.BEARISH, eq + 0.01, lo, hi)
    assert not premium_discount_ok(cfg, TrendBias.BEARISH, eq - 0.01, lo, hi)


def test_m15_mss_requires_choch_after_sweep() -> None:
    cfg = _FlowCfg(flow_c_require_15m_mss=True, flow_c_displacement=False)
    snap = StructureSnapshot(bias=TrendBias.BEARISH, last_choch_dir=-1)
    assert m15_mss_ok(
        cfg,
        bias=TrendBias.BEARISH,
        snap=snap,
        choch_bar_i=10,
        sweep_bar_i=8,
        open_=1.0,
        close=0.99,
        atr=0.001,
    )
    assert not m15_mss_ok(
        cfg,
        bias=TrendBias.BEARISH,
        snap=snap,
        choch_bar_i=5,
        sweep_bar_i=8,
        open_=1.0,
        close=0.99,
        atr=0.001,
    )


def test_m15_align_legacy() -> None:
    cfg = _FlowCfg()
    assert m15_align_ok_legacy(
        cfg, bias=TrendBias.BULLISH, bias_15m=TrendBias.BULLISH, choch_dir=0
    )
    assert m15_align_ok_legacy(
        cfg, bias=TrendBias.BULLISH, bias_15m=TrendBias.NEUTRAL, choch_dir=1
    )


def test_dealing_range_from_swings() -> None:
    snap = StructureSnapshot(
        last_swing_low=SwingPoint(1, None, 1.05, -1, 5),
        last_swing_high=SwingPoint(3, None, 1.15, 1, 7),
    )
    assert dealing_range_from_swings(snap) == (1.05, 1.15)


def test_sweep_kind_and_ny_session_filters() -> None:
    from trading_lab.strategies.smc.flow_filters import (
        ny_session_ok,
        sweep_kind_allowed,
    )

    off = _FlowCfg()
    assert sweep_kind_allowed(off, "pdh")
    assert ny_session_ok(off, 3)

    on = _FlowCfg(flow_ny_session_only=True, flow_sell_side_sweeps_only=True)
    assert sweep_kind_allowed(on, "asia_low")
    assert sweep_kind_allowed(on, "london_low")
    assert sweep_kind_allowed(on, "pdl")
    assert not sweep_kind_allowed(on, "pdh")
    assert not sweep_kind_allowed(on, "asia_high")
    assert ny_session_ok(on, 8)  # ny_am
    assert ny_session_ok(on, 13)  # ny_pm
    assert not ny_session_ok(on, 3)  # london
    assert not ny_session_ok(on, 20)  # asia


def test_killzone_modes() -> None:
    from trading_lab.strategies.smc.flow_filters import ny_session_ok

    ny_am = _FlowCfg(flow_killzone="ny_am")
    assert ny_session_ok(ny_am, 8)
    assert not ny_session_ok(ny_am, 13)
    assert not ny_session_ok(ny_am, 3)

    lon_ny = _FlowCfg(flow_killzone="london_ny")
    assert ny_session_ok(lon_ny, 3)
    assert ny_session_ok(lon_ny, 8)
    assert ny_session_ok(lon_ny, 13)
    assert not ny_session_ok(lon_ny, 20)

    # Explicit killzone wins over legacy flag
    both = _FlowCfg(flow_killzone="london", flow_ny_session_only=True)
    assert ny_session_ok(both, 3)
    assert not ny_session_ok(both, 8)


def test_premium_discount_disabled_passes() -> None:
    cfg = _FlowCfg(flow_b_premium_discount=False)
    assert premium_discount_ok(cfg, TrendBias.BULLISH, 1.5, 1.0, 1.2)
