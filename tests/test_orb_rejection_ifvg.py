"""Tests for the ORB rejection fade with inverse FVG (ablation-first)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trading_lab.strategies.orb_rejection_ifvg import (
    OrbRejConfig,
    bearish_engulfing,
    bullish_engulfing,
    rejection_candle,
    run_backtest,
)

NY = "America/New_York"
SPREAD = 1.0
CFG = OrbRejConfig(tick_size=0.25, commission_points=0.0, mode="full",
                   entry_start="09:45", entry_end="11:30")


def _day(overrides: dict[int, tuple[float, float, float, float]],
         base: float = 96.0, day: str = "2024-03-05") -> pd.DataFrame:
    idx = pd.date_range(f"{day} 09:30", f"{day} 15:59", freq="1min", tz=NY).tz_convert("UTC")
    n = len(idx)
    o = np.full(n, base)
    h, l, c = o.copy(), o.copy(), o.copy()
    for m, (po, ph, pl, pc) in overrides.items():
        o[m], h[m], l[m], c[m] = po, ph, pl, pc
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1.0}, index=idx)
    for col in ("open", "high", "low", "close"):
        df[f"ask_{col}"] = df[col] + SPREAD
    return df


def _short_day(last: float = 89.0,
               trigger: tuple[float, float, float, float] | None = None) -> pd.DataFrame:
    """ORB 90–100; three rejections at ORH; a bullish FVG broken, then an
    engulfing trigger — a textbook Setup A short toward ORL."""
    ov = {
        0: (99.2, 100.0, 99.1, 99.3),          # sets orh=100 (small range → calm ATR)
        1: (90.9, 91.0, 90.0, 90.8),           # sets orl=90
        20: (99.5, 100.2, 99.0, 99.05),        # touch + rejection candle
        24: (99.5, 100.2, 99.0, 99.05),
        28: (99.5, 100.2, 99.0, 99.05),
        30: (98.0, 98.9, 98.0, 98.0),          # high 98.9 = bullish FVG bottom
        31: (98.5, 100.0, 98.5, 99.1),         # impulsive leg + rejection candle
        32: (99.5, 99.9, 99.2, 99.4),          # low 99.2 > high[30] → bullish FVG
        33: (99.4, 99.5, 99.2, 99.3),
        34: (99.6, 99.7, 98.7, 98.8),          # closes below the FVG → inversion
        35: trigger or (99.4, 99.5, 97.2, 97.4),  # bearish engulfing trigger (big body)
    }
    for m in range(36, 390):
        ov[m] = (last, last, last, last)
    return _day(ov)


# ── primitives ──────────────────────────────────────────────────────


def test_rejection_candle_upper_wick() -> None:
    assert rejection_candle(99.5, 100.2, 99.0, 99.1, 0.5)          # upper wick ≥ half
    assert not rejection_candle(99.5, 99.6, 99.0, 99.55, 0.5)      # tiny wick


def test_engulfing_candles() -> None:
    assert bearish_engulfing(98.6, 97.2, 98.8, 97.9)               # red engulfs prior
    assert not bearish_engulfing(97.2, 97.9, 98.6, 98.8)           # wrong direction
    assert bullish_engulfing(97.2, 98.6, 97.0, 97.4)


# ── end to end ──────────────────────────────────────────────────────


def test_full_setup_goes_short_to_opposite_edge() -> None:
    res = run_backtest(_short_day(), CFG)
    assert len(res["trades"]) == 1
    t = res["trades"][0]
    assert t["direction"] == "short"
    assert t["exit_reason"] == "target"
    assert t["target"] == pytest.approx(90.0 + 2 * 0.25)           # front-run ORL
    assert t["r"] > 0


def test_ablation_filters_do_not_add_trades() -> None:
    naive = run_backtest(_short_day(), replace(CFG, mode="naive"))
    full = run_backtest(_short_day(), replace(CFG, mode="full"))
    assert len(full["trades"]) >= 1 and len(naive["trades"]) >= len(full["trades"])


def test_momentum_lock_disables_the_fade() -> None:
    # price accepts above ORH (three closes beyond) before any setup → no short
    ov = {0: (95.0, 100.0, 90.0, 95.0)}
    for m in (20, 21, 22):
        ov[m] = (101.0, 101.5, 100.5, 101.2)
    df = _day(ov)
    assert run_backtest(df, CFG)["trades"] == []


def test_min_reward_risk_blocks_a_poor_trade() -> None:
    # huge ORB → the opposite edge is far but the stop is far too; force a high gate
    assert run_backtest(_short_day(), replace(CFG, min_reward_risk=99.0))["trades"] == []


def test_no_setup_without_engulf_under_full_mode() -> None:
    # a small trigger bar → engulfing fails, so the full mode stays quiet
    assert run_backtest(_short_day(trigger=(98.6, 98.7, 98.4, 98.5)), CFG)["trades"] == []
