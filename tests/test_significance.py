"""Tests for the shared significance / acceptance path."""
from __future__ import annotations

import numpy as np

from trading_lab import acceptance, significance
from trading_lab.significance import (
    MAX_PVALUE,
    MIN_PERCENTILE,
    assess,
    bootstrap_ci,
    percentile_of,
    promotion_gate,
    returns_from_trades,
    sign_flip_pvalue,
)


def test_bootstrap_ci_of_constant_series_is_a_point():
    lo, hi = bootstrap_ci([2.0] * 30)
    assert lo == hi == 2.0


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1.0, 400)
    lo, hi = bootstrap_ci(x, seed=1)
    assert lo < x.mean() < hi


def test_sign_flip_detects_a_strong_edge():
    assert sign_flip_pvalue([1.0] * 25) < MAX_PVALUE


def test_sign_flip_is_quiet_on_zero_mean_noise():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, 200)  # mean ~ 0
    assert sign_flip_pvalue(x) > MAX_PVALUE


def test_percentile_of_counts_how_much_null_is_beaten():
    null = np.arange(100.0)  # 0..99
    assert percentile_of(50.0, null) == 50.0
    assert percentile_of(1000.0, null) == 100.0
    assert percentile_of(-1.0, null) == 0.0


def test_returns_from_trades_prefers_r_then_pnl_and_skips_open():
    trades = [
        {"status": "closed", "r": 1.5},
        {"status": "closed", "pnl_pct": 2.0},
        {"status": "open", "r": 99.0},
        {"r": -0.5},  # no status → treated as closed
    ]
    assert returns_from_trades(trades) == [1.5, 2.0, -0.5]


def test_assess_generic_passes_on_a_clear_edge_and_fails_on_noise():
    assert assess([1.0] * 25).passed
    rng = np.random.default_rng(7)
    assert not assess(rng.normal(0.0, 1.0, 60)).passed


def test_assess_rejects_too_few_trades():
    sig = assess([1.0] * 3, min_trades=10)
    assert not sig.passed and sig.reason == "n<10"


def test_assess_random_entry_uses_the_percentile():
    returns = [1.0] * 12
    null = np.random.default_rng(0).normal(0.0, 5.0, 1000)
    good = assess(returns, observed=12.0, null_samples=null)
    assert good.method == "random-entry" and good.passed and good.percentile >= MIN_PERCENTILE
    bad = assess(returns, observed=-50.0, null_samples=null)
    assert not bad.passed


def test_promotion_gate_is_on_by_default_and_can_be_disabled(monkeypatch):
    weak = [{"status": "closed", "pnl_pct": 0.1}] * 5  # too few → fails when enabled
    ok, sig = promotion_gate(weak)
    assert ok is False and not sig.passed

    monkeypatch.setenv("LAB_SIGNIFICANCE", "0")
    ok, _ = promotion_gate(weak)
    assert ok is True


def test_run_acceptance_requires_significance_and_extra_criteria():
    strong = [{"status": "closed", "r": 1.0}] * 25
    passed = acceptance.run_acceptance(strong, extra={"oos_positive": True, "cost_ok": True})
    assert passed.passed

    blocked = acceptance.run_acceptance(strong, extra={"oos_positive": True, "cost_ok": False})
    assert not blocked.passed
    assert blocked.significance.passed and not blocked.criteria["cost_ok"]
