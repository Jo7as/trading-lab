"""Tests for the shared significance / acceptance path."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from trading_lab import acceptance, significance
from trading_lab.significance import (
    MAX_PVALUE,
    MIN_PERCENTILE,
    T_DAILY_MIN,
    assess,
    assess_trades,
    bootstrap_ci,
    percentile_of,
    promotion_gate,
    returns_from_trades,
    sign_flip_pvalue,
)


def _daily_log(values) -> list[dict]:
    """One closed trade per calendar day — the shape S4 judges."""
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return [{"status": "closed", "exit_time": (base + timedelta(days=i)).isoformat(),
             "r": float(v)} for i, v in enumerate(values)]


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


def test_promotion_gate_enforces_when_there_is_evidence_but_it_fails():
    weak = [{"status": "closed", "pnl_pct": 0.1}] * 5  # n < MIN_TRADES → fails
    allowed, sig, evaluated = promotion_gate(weak)
    assert evaluated is True and sig.passed is False and allowed is False


def test_promotion_gate_fails_open_without_trades():
    allowed, sig, evaluated = promotion_gate([])
    assert allowed is True and evaluated is False


def test_promotion_gate_fails_open_without_any_closed_trade():
    allowed, sig, evaluated = promotion_gate([{"status": "open", "r": 9.0}] * 5)
    assert allowed is True and evaluated is False


def test_promotion_gate_blocks_a_log_without_a_return_series():
    # Closed trades, but none of r/pnl_pct/equity_pnl_pct: a logging/schema bug,
    # not "nothing to evaluate" — it must not promote unchallenged.
    allowed, sig, evaluated = promotion_gate([{"status": "closed", "pnl_usd": -3.0}] * 20)
    assert allowed is False and evaluated is True and not sig.passed
    assert "no return series" in sig.reason


def test_promotion_gate_flag_mode_records_a_missing_return_series(monkeypatch):
    monkeypatch.setenv("LAB_SIGNIFICANCE_MODE", "flag")
    allowed, sig, evaluated = promotion_gate([{"status": "closed", "pnl_usd": 1.0}])
    assert allowed is True and evaluated is True and not sig.passed


def test_promotion_gate_can_be_disabled(monkeypatch):
    monkeypatch.setenv("LAB_SIGNIFICANCE", "0")
    allowed, sig, evaluated = promotion_gate([{"status": "closed", "pnl_pct": 0.1}] * 5)
    assert allowed is True and evaluated is False


def test_promotion_gate_flag_mode_records_but_allows(monkeypatch):
    monkeypatch.setenv("LAB_SIGNIFICANCE_MODE", "flag")
    allowed, sig, evaluated = promotion_gate([{"status": "closed", "pnl_pct": 0.1}] * 5)
    assert allowed is True and evaluated is True and not sig.passed


def test_promotion_gate_passes_on_a_clear_edge():
    allowed, sig, evaluated = promotion_gate([{"status": "closed", "r": 1.0}] * 25)
    assert allowed is True and evaluated is True and sig.passed


def test_significance_as_dict_is_json_serializable():
    d = assess([1.0] * 25).as_dict()
    assert d["method"] == "sign-flip" and d["passed"] is True
    assert d["n"] == 25
    import json

    json.dumps(d)  # must not raise


def test_run_acceptance_requires_significance_and_extra_criteria():
    strong = [{"status": "closed", "r": 1.0}] * 25
    passed = acceptance.run_acceptance(strong, extra={"oos_positive": True, "cost_ok": True})
    assert passed.passed

    blocked = acceptance.run_acceptance(strong, extra={"oos_positive": True, "cost_ok": False})
    assert not blocked.passed
    assert blocked.significance.passed and not blocked.criteria["cost_ok"]


# ── S4: a trade log with a clock is judged on the DAILY series ──────────────

def test_assess_trades_uses_the_daily_t_when_a_clock_exists():
    strong = [0.5 + (0.01 if i % 2 else -0.01) for i in range(60)]
    sig = assess_trades(_daily_log(strong))
    assert sig.method == "daily-t" and sig.passed and sig.t >= T_DAILY_MIN


def test_assess_trades_fails_a_flat_daily_series_with_an_evidence_reason():
    flat = [0.1 if i % 2 else -0.1 for i in range(60)]      # mean 0 → no edge
    sig = assess_trades(_daily_log(flat))
    assert sig.method == "daily-t" and not sig.passed and sig.reason.startswith("t_daily")


def test_assess_trades_falls_back_to_per_trade_without_a_clock():
    sig = assess_trades([{"status": "closed", "r": 1.0}] * 25)
    assert sig.method == "sign-flip" and sig.passed


def test_promotion_gate_uses_the_daily_series_when_the_log_has_a_clock():
    strong = [0.5 + (0.01 if i % 2 else -0.01) for i in range(60)]
    allowed, sig, evaluated = promotion_gate(_daily_log(strong))
    assert allowed and evaluated and sig.method == "daily-t" and sig.passed


def test_promotion_gate_blocks_a_clocked_log_with_no_daily_edge():
    flat = [0.1 if i % 2 else -0.1 for i in range(60)]
    allowed, sig, evaluated = promotion_gate(_daily_log(flat))
    assert not allowed and evaluated and sig.method == "daily-t" and not sig.passed


def test_run_acceptance_judges_a_clocked_log_on_the_daily_series():
    strong = [0.5 + (0.01 if i % 2 else -0.01) for i in range(60)]
    rep = acceptance.run_acceptance(_daily_log(strong))
    assert rep.passed and rep.significance.method == "daily-t" and rep.n_days == 60


def test_acceptance_report_summary_renders():
    """Regression: the report used ``significance.t`` before it existed."""
    txt = acceptance.run_acceptance([{"status": "closed", "r": 0.5}] * 25).summary()
    assert "ACCEPTANCE" in txt and "t je Tag" in txt
