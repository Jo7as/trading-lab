"""Tests for the reusable prop-firm simulator (FTMO + Lucid presets, phases)."""
from __future__ import annotations

import numpy as np
import pytest

from trading_lab.prop_sim import (
    FTMO,
    INSTANT_25K,
    LUCID,
    ONE_STEP as ONE_STEP_PRESET,
    FIRMS,
    PropFirm,
    _one_funded,
    daily_pnl,
    evaluate,
    get_firm,
    risk_usd_for,
    run_phase,
    simulate_funded,
)

# a 25k one-step firm: 5 % target, 2.4 % daily, 4 % EOD-trailing total
ONE_STEP = PropFirm(name="test-25k", account_size=25_000.0, phase_targets=(0.05,),
                    max_daily_loss_frac=0.024, max_total_loss_frac=0.04,
                    drawdown_type="eod_trailing", time_limit_days=365)


def test_presets_and_lookup() -> None:
    assert FIRMS["ftmo"] is FTMO and get_firm("FTMO") is FTMO
    assert FTMO.drawdown_type == "static" and LUCID.drawdown_type == "eod_trailing"
    assert FTMO.phase_targets == (0.10, 0.05)          # 2-step
    assert FTMO.daily_limit_usd == 5_000.0 and FTMO.total_limit_usd == 10_000.0
    assert risk_usd_for(25_000, 0.5) == 125.0


def test_pass_target() -> None:
    outcome, days, traded = run_phase(np.array([200.0] * 10), ONE_STEP, 0.05)
    assert outcome == "pass" and days == 7 and traded == 7   # 200 × 7 = 1400 ≥ 1250


def test_daily_breach() -> None:
    assert run_phase(np.array([-600.0]), ONE_STEP, 0.05)[0] == "daily"
    assert run_phase(np.array([-599.0]), ONE_STEP, 0.05)[0] == "timeout"


def test_eod_trailing_breach() -> None:
    seq = np.array([300.0, 300.0, -400.0, -400.0, -400.0])   # peak 25 600 → floor 24 600
    assert run_phase(seq, ONE_STEP, 0.05) == ("total", 5, 5)


def test_static_breach_ignores_the_peak() -> None:
    static = PropFirm(name="static", account_size=25_000.0, phase_targets=(0.05,),
                      max_daily_loss_frac=0.05, max_total_loss_frac=0.10,
                      drawdown_type="static", time_limit_days=365)
    # rises to 26 000 then bleeds to 22 000 (< 25 000 − 10 %); no single −1 250 day
    seq = np.array([1000.0, -1000.0, -1000.0, -1000.0, -1000.0])
    assert run_phase(seq, static, 0.05) == ("total", 5, 5)


def test_minimum_trading_days_delays_the_pass() -> None:
    firm = PropFirm(name="mindays", account_size=25_000.0, phase_targets=(0.05,),
                    max_daily_loss_frac=0.024, max_total_loss_frac=0.04,
                    drawdown_type="eod_trailing", min_trading_days=4, time_limit_days=365)
    assert run_phase(np.array([500.0] * 5), firm, 0.05) == ("pass", 4, 4)     # target on day 3, pass on day 4
    assert run_phase(np.array([500.0] * 3), firm, 0.05)[0] == "timeout"        # data ends before 4 days


def test_two_phase_attempt_needs_both_phases() -> None:
    good = np.full(300, 300.0)          # strongly positive
    res = evaluate(good, FTMO, runs=50, block=20, seed=1)
    assert res.attempts == 51 and res.pass_rate > 0.5
    assert len(res.phase_pass_rate) == 2 and all(0.0 <= r <= 1.0 for r in res.phase_pass_rate)


def test_daily_pnl_uses_the_calendar() -> None:
    trades = [{"day": "2024-03-05", "r": 2.0}, {"day": "2024-03-05", "r": -1.0},
              {"day": "2024-03-07", "r": 1.5}]

    class D:
        def __init__(self, s): self.s = s
        def date(self): return self.s

    arr = daily_pnl(trades, risk_usd=100.0,
                    days=[D("2024-03-05"), D("2024-03-06"), D("2024-03-07")])
    assert list(arr) == [100.0, 0.0, 150.0]


def test_presets_include_one_step_and_instant() -> None:
    assert FIRMS["one_step"] is ONE_STEP_PRESET and ONE_STEP_PRESET.phase_targets == (0.08,)
    assert INSTANT_25K.is_instant and not FTMO.is_instant
    assert set(FIRMS) >= {"ftmo", "lucid", "one_step", "instant_25k", "instant_100k"}


def test_instant_funded_survival_and_payout() -> None:
    firm = PropFirm(name="instant-1k", account_size=1_000.0, phase_targets=(),
                    max_daily_loss_frac=0.10, max_total_loss_frac=0.20,
                    drawdown_type="static", payout_split=1.0, payout_cycle_days=2)
    # +100/day; balance hits +200 at day 2 and +200 again at day 4 → 400 total
    ok, days, payout = _one_funded(np.array([100.0] * 4), firm, 252)
    assert ok and days == 4 and payout == pytest.approx(400.0)


def test_instant_daily_breach_and_static_floor() -> None:
    firm = PropFirm(name="instant-1k", account_size=1_000.0, phase_targets=(),
                    max_daily_loss_frac=0.10, max_total_loss_frac=0.20,
                    drawdown_type="static", payout_split=1.0, payout_cycle_days=0)
    assert _one_funded(np.array([-100.0]), firm, 252) == (False, 1, 0.0)      # daily limit
    assert _one_funded(np.array([-90.0, -90.0, -90.0]), firm, 252)[0] is False  # 700 < 800 floor


def test_simulate_funded_shape() -> None:
    rng = np.random.default_rng(0)
    stream = rng.normal(150, 300, 400)
    res = simulate_funded(stream, INSTANT_25K, runs=200, seed=1)
    assert res.attempts == 201 and 0.0 <= res.survival_rate <= 1.0
    assert res.ev_per_account_usd == pytest.approx(res.mean_payout_usd)


def test_evaluate_reports_failure_reasons() -> None:
    bleed = np.full(300, -200.0)        # steady losses → trailing total breach
    res = evaluate(bleed, LUCID, runs=50, seed=2)
    assert res.pass_rate == 0.0 and res.fails.get("total", 0) > 0
    flat = np.zeros(300)                # nothing happens → timeout
    assert evaluate(flat, LUCID, runs=20, seed=3).fails.get("timeout", 0) > 0
