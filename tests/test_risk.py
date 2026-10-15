"""Tests for fixed-account risk management and P&L accounting."""
from __future__ import annotations

from dataclasses import dataclass

from trading_lab.risk import RiskManager, Position


@dataclass(frozen=True)
class _Cfg:
    risk_per_trade: float = 0.01
    sl_atr_mult: float = 1.0
    tp1_r: float = 2.0
    tp1_close_pct: float = 0.75
    max_daily_loss: float = 0.03
    cooldown_hours: int = 4


def test_fixed_sizing_does_not_compound() -> None:
    cfg = _Cfg()
    rm = RiskManager(cfg, account_size=10_000)

    pos1 = rm.open_position(100.0, atr=2.0, direction="long", entry_time=1.0, entry_idx=0)
    assert pos1 is not None
    size1 = pos1.size

    # Simulate a large win
    rm.close_position(pos1, 110.0, "test", exit_time=2.0)
    assert rm.current_equity > 10_000

    pos2 = rm.open_position(100.0, atr=2.0, direction="long", entry_time=3.0, entry_idx=1)
    assert pos2 is not None
    assert abs(pos2.size - size1) < 1e-9


def test_partial_then_final_pnl_is_weighted() -> None:
    cfg = _Cfg(tp1_close_pct=0.5)
    rm = RiskManager(cfg, account_size=10_000)

    pos = rm.open_position(100.0, atr=2.0, direction="long", entry_time=1.0, entry_idx=0)
    assert pos is not None

    # TP1 at +4% price move
    rm.close_position(pos, 104.0, "tp1_partial", exit_time=2.0, partial=True)
    assert pos.tp1_hit
    assert pos.status == "open"
    mid_equity = rm.current_equity

    # Remainder stopped at breakeven
    rm.close_position(pos, 100.0, "stop_loss", exit_time=3.0)
    assert pos.status == "closed"
    assert pos.pnl_pct is not None
    # Total account P&L should be positive (from TP1 leg only)
    assert pos.pnl_pct > 0
    assert rm.current_equity == mid_equity


def test_backtest_mode_resets_daily_pnl_on_simulated_day() -> None:
    cfg = _Cfg(max_daily_loss=0.03)
    rm = RiskManager(cfg, account_size=10_000, backtest_mode=True)

    day1 = 1_700_000_000.0
    pos = rm.open_position(100.0, atr=2.0, direction="long", entry_time=day1, entry_idx=0)
    assert pos is not None
    rm.close_position(pos, 98.0, "stop_loss", exit_time=day1 + 3600)
    assert rm.state.daily_pnl < 0

    # Next simulated calendar day — daily loss budget should reset
    day2 = day1 + 86_400
    allowed, reason = rm.can_trade(day2)
    assert allowed, reason
    assert rm.state.daily_pnl == 0.0


def test_loss_triggers_cooldown_on_net_trade() -> None:
    cfg = _Cfg()
    rm = RiskManager(cfg, account_size=10_000)

    pos = rm.open_position(100.0, atr=2.0, direction="long", entry_time=1.0, entry_idx=0)
    assert pos is not None
    rm.close_position(pos, 98.0, "stop_loss", exit_time=2.0)

    allowed, reason = rm.can_trade(2.0)
    assert not allowed
    assert "cooldown" in reason
