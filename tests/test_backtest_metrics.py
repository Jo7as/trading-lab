"""Tests for the shared metric builder (`_compute_metrics`).

Covers the two things the metrics are honest about now:
  - max drawdown comes from the equity curve when one is given, else from the
    cumulative trade-PnL stream;
  - Sharpe is annualized with the realized trade frequency, not a magic constant.
"""
from __future__ import annotations

import math

from trading_lab.backtest import _compute_metrics
from trading_lab.score import trades_per_week


def _closed(pnl: float, entry_time: float, direction: str = "long") -> dict:
    return {
        "status": "closed",
        "pnl_pct": pnl,
        "entry_time": entry_time,
        "direction": direction,
    }


def test_drawdown_uses_equity_curve_when_given() -> None:
    # Positive trade PnL stream would show zero drawdown; the equity curve dips.
    trades = [_closed(1.0, 1_700_000_000.0), _closed(1.0, 1_700_086_400.0)]
    equity = [10_000.0, 11_000.0, 9_000.0, 10_500.0]

    m = _compute_metrics(trades, equity)

    # peak 11_000 -> trough 9_000 is -18.18%
    assert m["max_drawdown_pct"] == -18.18


def test_drawdown_falls_back_to_trade_pnl_stream() -> None:
    trades = [_closed(2.0, 1_700_000_000.0), _closed(-3.0, 1_700_086_400.0)]

    # No curve, and a single-point curve, both fall back to the PnL stream.
    assert _compute_metrics(trades)["max_drawdown_pct"] == -3.0
    assert _compute_metrics(trades, [10_000.0])["max_drawdown_pct"] == -3.0


def test_sharpe_annualizes_with_realized_frequency() -> None:
    pnls = [1.0, 3.0]
    mean = sum(pnls) / len(pnls)
    std = math.sqrt(((1.0 - 2.0) ** 2 + (3.0 - 2.0) ** 2) / (len(pnls) - 1))

    weekly = [_closed(1.0, 1_700_000_000.0), _closed(3.0, 1_700_000_000.0 + 7 * 86_400)]
    biweekly = [_closed(1.0, 1_700_000_000.0), _closed(3.0, 1_700_000_000.0 + 14 * 86_400)]

    weekly_factor = max(trades_per_week(weekly), 1.0) * 52.0
    biweekly_factor = max(trades_per_week(biweekly), 1.0) * 52.0
    assert weekly_factor != biweekly_factor

    expected_weekly = round((mean / std) * math.sqrt(weekly_factor), 3)
    expected_biweekly = round((mean / std) * math.sqrt(biweekly_factor), 3)

    assert _compute_metrics(weekly)["sharpe"] == expected_weekly
    assert _compute_metrics(biweekly)["sharpe"] == expected_biweekly
    assert _compute_metrics(weekly)["sharpe"] != _compute_metrics(biweekly)["sharpe"]
