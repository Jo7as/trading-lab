"""Score trades against goal.yaml.

score(trades, goal) -> float in [-1, +1]

Composite of:
  - realised return vs target
  - drawdown vs max
  - Sharpe vs min
  - trade frequency vs target (trades/week)
"""
from __future__ import annotations

import math
from typing import Any


def trades_per_week(
    trades: list[dict[str, Any]], period_days: float | None = None
) -> float:
    """Closed trades per calendar week over the evaluation window."""
    closed = [t for t in trades if t.get("status") == "closed"]
    if not closed:
        return 0.0

    if period_days is None:
        times = [t.get("entry_time") for t in closed if t.get("entry_time")]
        if len(times) >= 2:
            period_days = (max(times) - min(times)) / 86400
        else:
            period_days = 7.0

    weeks = max(period_days / 7.0, 1.0)
    return len(closed) / weeks


def _frequency_component(tpw: float, goal: dict[str, Any]) -> float:
    """Reward configs in the min–target trades/week band."""
    min_tpw = goal.get("min_trades_per_week", 3)
    target = goal.get("target_trades_per_week", 5)
    if tpw < min_tpw:
        return max(-1.0, -1.0 + (tpw / min_tpw if min_tpw > 0 else 0.0))
    if tpw <= target:
        span = max(target - min_tpw, 0.1)
        return (tpw - min_tpw) / span
    # Mild decay above target — extra trades OK up to ~2× target
    return max(0.4, 1.0 - (tpw - target) / max(target, 1.0))


def score(
    trades: list[dict[str, Any]],
    goal: dict[str, Any],
    period_days: float | None = None,
) -> float:
    """Return a composite score in [-1, +1]."""
    if not trades:
        return -0.5  # penalise zero-trade configs in evolution

    pnls = [t.get("pnl_pct", 0) for t in trades]
    n = len(pnls)
    tpw = trades_per_week(trades, period_days)

    # realised return (sum of per-trade % returns)
    total_return = sum(pnls)

    # max drawdown from running cumulative
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        dd = cum - peak
        max_dd = min(max_dd, dd)

    # Sharpe (per-trade, annualised × sqrt(252*24) ~ rough)
    mean = total_return / n
    variance = sum((p - mean) ** 2 for p in pnls) / n if n > 1 else 0
    std = math.sqrt(variance) if variance > 0 else 0.0001
    sharpe = (mean / std) * math.sqrt(252 * 24) if std > 0 else 0.0

    # normalise components to [-1, +1]
    target = goal.get("target_return_30d", 0.08) * 100  # as %
    max_dd_limit = goal.get("max_drawdown", 0.05) * 100
    min_sharpe = goal.get("min_sharpe", 1.2)

    # return component: +1 at target, 0 at 0, -1 at -target
    return_component = max(-1, min(1, total_return / target)) if target > 0 else 0.0

    # drawdown component: +1 at 0 dd, -1 at max_dd_limit
    dd_component = max(-1, min(1, 1 + (max_dd / max_dd_limit))) if max_dd_limit > 0 else 0.0

    # sharpe component: +1 at 2*min, 0 at min, -1 below 0
    sharpe_component = max(-1, min(1, (sharpe - min_sharpe) / min_sharpe)) if min_sharpe > 0 else 0.0

    freq_component = _frequency_component(tpw, goal)

    # weighted composite — frequency weighted heavily for evolution
    composite = (
        0.30 * return_component
        + 0.25 * dd_component
        + 0.20 * sharpe_component
        + 0.25 * freq_component
    )
    composite = max(-1, min(1, composite))

    # apply failure floor — 0% means any negative composite is amplified
    failure_below = goal.get("failure_below", -0.02)
    if composite < failure_below:
        composite = max(-1, composite * 1.5)  # steeply negative

    return round(composite, 4)
