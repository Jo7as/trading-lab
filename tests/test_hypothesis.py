"""Tests for the S2 hypothesis-card gate."""
from __future__ import annotations

import textwrap

import pytest

from trading_lab.hypothesis import load_hypothesis

GOOD = textwrap.dedent("""
    id: H-2026-001
    titel: t
    beobachtung: quiet hours revert
    mechanismus: liquidity provision after a stretched move
    behauptung: "mean forward return over 60m is > +0.05%"
    dosis_wirkung: grows with |z|
    placebo: random timestamps show nothing
    horizont: 60m
    kosten_ratio: 5.8
    kill_kriterien: [t < 3, effect < 2x cost]
    status: offen
""")


def _write(tmp_path, body: str):
    p = tmp_path / "h.yaml"
    p.write_text(textwrap.dedent(body))
    return load_hypothesis(p)


def test_a_complete_card_passes(tmp_path) -> None:
    assert _write(tmp_path, GOOD).validate() == []


def test_missing_mechanism_is_rejected(tmp_path) -> None:
    card = _write(tmp_path, GOOD.replace("mechanismus: liquidity provision after a stretched move",
                                         "mechanismus: ''"))
    assert any("mechanism" in p for p in card.validate())


def test_claim_without_a_number_is_rejected(tmp_path) -> None:
    card = _write(tmp_path, GOOD.replace('behauptung: "mean forward return over 60m is > +0.05%"',
                                         'behauptung: "it tends to revert"'))
    assert any("number" in p for p in card.validate())


def test_missing_kill_criteria_is_rejected(tmp_path) -> None:
    card = _write(tmp_path, GOOD.replace("kill_kriterien: [t < 3, effect < 2x cost]",
                                         "kill_kriterien: []"))
    assert any("kill criteria" in p for p in card.validate())


def test_cost_gate_rejects_a_hopeless_horizon(tmp_path) -> None:
    card = _write(tmp_path, GOOD.replace("kosten_ratio: 5.8", "kosten_ratio: 1.2"))
    assert any("cost gate" in p for p in card.validate())
