"""Tests for the exit-attribution report (is the R target actually doing work?)."""
from __future__ import annotations

import pandas as pd
import pytest

from trading_lab.exit_attribution import (
    attribution,
    holding_minutes,
    progress,
    report,
    verdict,
)


def _trade(reason: str, r: float, *, direction: str = "long", entry: float = 100.0,
           stop: float = 99.0, target: float = 102.0, exit: float = 100.0,
           minutes: int = 60) -> dict:
    """A trade opened 09:30 and closed ``minutes`` later."""
    start = pd.Timestamp("2026-01-02T09:30:00+00:00")
    return {
        "exit_reason": reason, "r": r, "direction": direction,
        "entry": entry, "stop": stop, "target": target, "exit": exit,
        "entry_time": start.isoformat(),
        "exit_time": (start + pd.Timedelta(minutes=minutes)).isoformat(),
    }


def test_attribution_groups_and_shares_add_up() -> None:
    trades = [_trade("target", 2.0)] * 2 + [_trade("stop", -1.0)] * 5 + [_trade("flat", 0.1)] * 3
    a = attribution(trades)
    assert a.loc["stop", "n"] == 5
    assert a.loc["target", "mean_r"] == pytest.approx(2.0)
    assert a["share"].sum() == pytest.approx(1.0)
    assert a["share_of_r"].sum() == pytest.approx(1.0)
    assert a.loc["target", "win_rate"] == pytest.approx(1.0)
    assert a.loc["stop", "win_rate"] == pytest.approx(0.0)


def test_attribution_marks_target_and_time_groups() -> None:
    a = attribution([_trade("target", 2.0), _trade("flat", 0.0), _trade("stop", -1.0)])
    assert bool(a.loc["target", "is_target"]) is True
    assert bool(a.loc["flat", "is_time"]) is True
    assert bool(a.loc["stop", "is_target"]) is False


def test_verdict_calls_a_rarely_hit_target_decorative() -> None:
    trades = [_trade("target", 2.0)] + [_trade("flat", 0.05)] * 6 + [_trade("stop", -1.0)] * 3
    v = verdict(attribution(trades))
    assert "dekorativ" in v                      # 10 % target, 60 % time exits
    heavy = [_trade("target", 2.0)] * 6 + [_trade("stop", -1.0)] * 4
    assert "echter Parameter" in verdict(attribution(heavy))
    no_target = [_trade("stop", -1.0), _trade("time_stop", 0.2)]
    assert "reiner" in verdict(attribution(no_target))


def test_progress_is_the_fraction_of_the_entry_to_target_distance() -> None:
    long_half = _trade("flat", 0.0, entry=100.0, target=102.0, exit=101.0)
    assert progress([long_half]).iloc[0]["progress"] == pytest.approx(0.5)
    short_half = _trade("flat", 0.0, direction="short", entry=100.0, target=98.0, exit=99.0)
    assert progress([short_half]).iloc[0]["progress"] == pytest.approx(0.5)
    at_target = _trade("target", 2.0, entry=100.0, target=102.0, exit=102.0)
    assert progress([at_target]).iloc[0]["progress"] == pytest.approx(1.0)


def test_holding_minutes_per_group() -> None:
    trades = [_trade("flat", 0.0, minutes=300), _trade("flat", 0.0, minutes=120),
              _trade("stop", -1.0, minutes=30)]
    hm = holding_minutes(trades)
    assert hm.loc["flat", "median_minutes"] == pytest.approx(210.0)
    assert hm.loc["stop", "median_minutes"] == pytest.approx(30.0)


def test_report_contains_the_table_and_the_verdict() -> None:
    trades = [_trade("target", 2.0)] + [_trade("flat", 0.1)] * 5 + [_trade("stop", -1.0)] * 4
    txt = report(trades, "UNIT")
    assert "Exit-Attribution: UNIT" in txt
    assert "Ziel-Fortschritt" in txt
    assert "Haltedauer" in txt
    assert "-> " in txt


def test_pnl_key_is_configurable_and_missing_keys_raise() -> None:
    trades = [{"exit_reason": "target", "pnl_pct": 1.5}]
    a = attribution(trades, r_key="pnl_pct")
    assert a.loc["target", "sum_r"] == pytest.approx(1.5)
    with pytest.raises(ValueError, match="pnl_pct"):
        attribution(trades)
