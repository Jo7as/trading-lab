"""The return target, written down in one place.

**Target: 1-2 % per month** (compounded 12.7-26.8 % p.a.), decided 2026-10-26.
The retired 2-4 % band needed a net Sharpe of 2.7-6.0, which nothing reaches;
1-2 % is what ``docs/GOAL.md`` already called the honest ceiling (Sharpe 0.8-1.8
at 15-20 % volatility, at a ~20-30 % drawdown).

Every consumer reads the numbers from here — the evolution workers' fallback
goals, ``score.py`` and the study gates — because the target had drifted into
five different values across the repo (0.01 / 0.02 / 0.04 / 0.06 / 0.08 as a
30-day target, plus a 0.27 annual gate).

Two things this module deliberately is *not*:

* it is not evidence. The band says what counts as enough; it never replaces the
  significance and cost gates (:mod:`trading_lab.significance`).
* it is not the score. ``score.py`` normalises returns with the target, so
  changing the number changes the search pressure of the hill-climb. Consumers
  that only *report* against the goal should use the constants here; the
  selection weights stay where they are.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

MONTHS_PER_YEAR = 12

# ── The band ────────────────────────────────────────────────────────

TARGET_RETURN_MONTH_LOW = 0.01
TARGET_RETURN_MONTH_HIGH = 0.02
#: Midpoint of the band — the value a fresh ``goal.yaml`` gets.
TARGET_RETURN_MONTH = (TARGET_RETURN_MONTH_LOW + TARGET_RETURN_MONTH_HIGH) / 2.0

# goal.yaml and score.py express the target over a 30-day window; a window month
# is the same as a calendar month here.
TARGET_RETURN_30D = TARGET_RETURN_MONTH
TARGET_RETURN_30D_LOW = TARGET_RETURN_MONTH_LOW
TARGET_RETURN_30D_HIGH = TARGET_RETURN_MONTH_HIGH


def annualised(monthly: float, months: int = MONTHS_PER_YEAR) -> float:
    """Compounded annual return of a constant monthly return."""
    return (1.0 + monthly) ** months - 1.0


#: Annual equivalents of the band: +12.7 % … +26.8 %.
TARGET_RETURN_YEAR_LOW = annualised(TARGET_RETURN_MONTH_LOW)
TARGET_RETURN_YEAR_HIGH = annualised(TARGET_RETURN_MONTH_HIGH)
TARGET_RETURN_YEAR = annualised(TARGET_RETURN_MONTH)


def required_sharpe(monthly: float, vol_annual: float) -> float:
    """Net Sharpe a constant ``monthly`` return needs at ``vol_annual``."""
    if vol_annual <= 0:
        raise ValueError("vol_annual must be > 0")
    return annualised(monthly) / vol_annual


# ── Quality bars that go with the band ──────────────────────────────

#: ``docs/GOAL.md`` drawdown gate. With 15-20 % volatility the expected drawdown
#: is ~1.5-2.5x vol, i.e. 22-38 % — so the drawdown, not the return, is the
#: binding constraint. The goal.yaml templates used 0.08, which no 1-2 %/month
#: strategy can satisfy at a sensible volatility (0.08 would imply ~4 % vol).
MAX_DRAWDOWN = 0.20

#: A floor, not the band's requirement: 1 %/month at 20 % vol needs Sharpe 0.63,
#: 2 %/month at 15 % needs 1.79. A strategy well below this cannot reach the band.
MIN_SHARPE = 0.5

#: Frequency band the workers score against. Deliberately the values the code
#: already defaulted to (score.py: 3/5) — the return target changed, not the
#: frequency expectation. Strategy templates override them anyway.
MIN_TRADES_PER_WEEK = 3.0
TARGET_TRADES_PER_WEEK = 5.0


def band_summary() -> str:
    """One line for CLIs and reports."""
    return (
        f"goal {TARGET_RETURN_MONTH_LOW * 100:.0f}-{TARGET_RETURN_MONTH_HIGH * 100:.0f} %/month "
        f"({TARGET_RETURN_YEAR_LOW * 100:.1f}-{TARGET_RETURN_YEAR_HIGH * 100:.1f} %/yr), "
        f"net Sharpe ~{required_sharpe(TARGET_RETURN_MONTH_LOW, 0.20):.2f}-"
        f"{required_sharpe(TARGET_RETURN_MONTH_HIGH, 0.15):.2f}, max DD {MAX_DRAWDOWN * 100:.0f} %"
    )


# ── Loading ─────────────────────────────────────────────────────────


def default_goal() -> dict[str, Any]:
    """The goal a worker uses when its state dir has no ``goal.yaml``."""
    return {
        "target_return_30d": TARGET_RETURN_30D,
        "max_drawdown": MAX_DRAWDOWN,
        "min_sharpe": MIN_SHARPE,
        "min_trades_per_week": MIN_TRADES_PER_WEEK,
        "target_trades_per_week": TARGET_TRADES_PER_WEEK,
    }


def load_goal(state_dir: str | Path, **extra_defaults: Any) -> dict[str, Any]:
    """Read ``<state_dir>/goal.yaml``, filling anything missing from the defaults.

    Replaces the per-worker ``load_yaml(state_dir / "goal.yaml")`` calls, which
    raised ``FileNotFoundError`` (or, for an empty file, returned ``None`` and
    crashed later on ``.get``) whenever a state dir had no goal yet — the case
    for every local run, since goal.yaml only ships as a deploy template.

    Values in the file always win; ``extra_defaults`` are strategy-specific
    floors (e.g. a looser frequency band for SMC).
    """
    goal: dict[str, Any] = {**default_goal(), **extra_defaults}
    path = Path(state_dir) / "goal.yaml"
    if path.exists():
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except (OSError, yaml.YAMLError):
            raw = {}
        if isinstance(raw, dict):
            goal.update({k: v for k, v in raw.items() if v is not None})
    return goal
