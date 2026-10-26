"""The return target has one home: ``trading_lab.goal``.

Before this, the target lived in ten places with five different values (0.01 /
0.02 / 0.04 / 0.06 / 0.08 as a 30-day target, plus a 0.27 annual gate). These
tests keep the band in one place and keep the deploy templates on it.
"""
from __future__ import annotations

import pathlib

import pytest
import yaml

from trading_lab import goal

TEMPLATES = pathlib.Path(__file__).resolve().parents[1] / "deploy" / "state-templates"


def test_band_is_one_to_two_percent_per_month():
    assert goal.TARGET_RETURN_MONTH_LOW == 0.01
    assert goal.TARGET_RETURN_MONTH_HIGH == 0.02
    assert goal.TARGET_RETURN_MONTH_LOW <= goal.TARGET_RETURN_MONTH <= goal.TARGET_RETURN_MONTH_HIGH


def test_annualisation_matches_compounding():
    assert goal.annualised(0.01) == pytest.approx(0.1268, abs=1e-4)
    assert goal.annualised(0.02) == pytest.approx(0.2682, abs=1e-4)
    assert goal.TARGET_RETURN_YEAR_LOW == pytest.approx(0.1268, abs=1e-4)
    assert goal.TARGET_RETURN_YEAR_HIGH == pytest.approx(0.2682, abs=1e-4)


def test_required_sharpe_of_the_band():
    # 1 %/month at 20 % vol needs ~0.63; 2 %/month at 15 % vol needs ~1.79.
    assert goal.required_sharpe(0.01, 0.20) == pytest.approx(0.634, abs=1e-3)
    assert goal.required_sharpe(0.02, 0.15) == pytest.approx(1.788, abs=1e-3)
    with pytest.raises(ValueError):
        goal.required_sharpe(0.01, 0.0)


def test_band_summary_names_the_band_and_the_annual_equivalent():
    text = goal.band_summary()
    assert "1-2 %/month" in text
    assert "12.7" in text and "26.8" in text


def test_default_goal_carries_the_band():
    g = goal.default_goal()
    assert g["target_return_30d"] == goal.TARGET_RETURN_30D
    assert g["max_drawdown"] == goal.MAX_DRAWDOWN
    assert g["min_sharpe"] == goal.MIN_SHARPE


def test_load_goal_falls_back_to_the_band_without_a_file(tmp_path):
    g = goal.load_goal(tmp_path)
    assert g["target_return_30d"] == goal.TARGET_RETURN_30D
    assert g["max_drawdown"] == goal.MAX_DRAWDOWN


def test_load_goal_prefers_the_file_and_keeps_strategy_defaults(tmp_path):
    (tmp_path / "goal.yaml").write_text(
        "target_return_30d: 0.02\nasset: EUR/USD\nunset:\n"
    )
    g = goal.load_goal(tmp_path, asset="XXX", target_trades_per_week=3.0)
    assert g["target_return_30d"] == 0.02       # file wins
    assert g["asset"] == "EUR/USD"              # file wins
    assert g["target_trades_per_week"] == 3.0   # strategy default survives
    assert g["min_sharpe"] == goal.MIN_SHARPE   # filled from the band
    assert "unset" not in g                     # explicit nulls are dropped


def test_load_goal_survives_a_broken_file(tmp_path):
    (tmp_path / "goal.yaml").write_text("target_return_30d: [unclosed\n")
    assert goal.load_goal(tmp_path)["target_return_30d"] == goal.TARGET_RETURN_30D


def test_no_worker_reads_its_own_target_default():
    """The workers must not carry their own numbers again."""
    root = pathlib.Path(__file__).resolve().parents[1]
    hits: list[str] = []
    for path in sorted((root / "trading_lab").glob("*evolution*.py")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if "target_return_30d" in line and any(
                f", {v})" in line for v in ("0.01", "0.02", "0.04", "0.06", "0.08")
            ):
                hits.append(f"{path}:{number}: {line.strip()}")
    assert not hits, hits


def test_deploy_templates_sit_inside_the_band():
    files = sorted(TEMPLATES.glob("*/goal.yaml"))
    assert files, "no deploy goal templates found"
    for path in files:
        data = yaml.safe_load(path.read_text()) or {}
        target = data["target_return_30d"]
        assert goal.TARGET_RETURN_MONTH_LOW <= target <= goal.TARGET_RETURN_MONTH_HIGH, path
        assert data["max_drawdown"] == goal.MAX_DRAWDOWN, path
