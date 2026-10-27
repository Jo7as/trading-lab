"""Strategy specs: the template is not a valid spec, a complete one is, and every H-012 gap is caught."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from trading_lab import strategy_spec as S

TEMPLATE = Path(__file__).resolve().parent.parent / "docs" / "specs" / "TEMPLATE.yaml"

GOOD = {
    "id": "S-001", "strategie": "demo", "titel": "Rücksetzer im Aufwärtstrend", "status": "entwurf",
    "idee": {"mechanismus": "kurzfristige Übertreibung im intakten Trend", "quellen": ["Connors 2008"]},
    "handel": {"maerkte": ["US100", "SP500"], "zeitrahmen": "1D", "kontotyp": "prop_standard",
               "max_haltedauer": "5 Handelstage", "wochenende_halten": False, "news_fenster": "egal"},
    "entry": {"bedingungen": ["Schluss > SMA200", "RSI(2) < 10"], "ausfuehrung": "Market zum nächsten Open"},
    "stop_loss": {"regel": "Einstieg - 2 x ATR(14, 1D)", "begruendung": "Volatilität"},
    "take_profit": {"regel": "Schluss > SMA5 -> Exit am nächsten Open", "begruendung": "Mean Reversion"},
    "exit_sonst": ["Freitag 21:00 UTC schließen"],
    "risiko": {"risiko_je_trade_pct": 0.5, "max_offene_positionen": 2,
               "korrelation": "US100+SP500 zusammen max. 1 R"},
    "kosten": {"profil": "ftmo", "swaps": "Dukascopy-Swaps je Nacht", "slippage": "0,5 bp je Fill"},
    "daten": {"quelle": "Dukascopy 1m", "bekannte_probleme": [], "warmup": "200 Tage"},
    "varianten": [{"name": "basis", "aenderung": ""}],
    "referenz": "Connors/Alvarez: SPY 1995-2007",
    "tests": ["tests/test_strategy_spec.py"],
}


def test_template_is_not_a_valid_spec() -> None:
    c = S.validate(yaml.safe_load(TEMPLATE.read_text(encoding="utf-8")), "TEMPLATE")
    assert not c.ok
    assert any("stop_loss" in e for e in c.errors) and any("referenz" in e for e in c.errors)


def test_a_complete_spec_passes() -> None:
    c = S.validate(GOOD)
    assert c.ok, c.errors
    assert not c.agreed                      # entwurf - only the user agrees


def test_agreed_needs_who_and_when() -> None:
    s = {**GOOD, "status": "abgestimmt"}
    assert not S.validate(s).ok
    s.update(abgestimmt_von="Jonas", abgestimmt_am="2026-10-27")
    assert S.validate(s).agreed


def test_every_h012_gap_is_an_error() -> None:
    s = copy.deepcopy(GOOD)
    s["stop_loss"]["regel"] = ""
    s["take_profit"] = {"regel": "keiner", "begruendung": ""}
    s["risiko"]["korrelation"] = ""
    s["kosten"]["swaps"] = ""
    s["referenz"] = ""
    s["tests"] = []
    errs = " | ".join(S.validate(s).errors)
    for word in ("stop_loss", "take_profit", "korrelation", "swaps", "referenz", "tests"):
        assert word in errs, word


def test_prop_standard_cannot_hold_over_the_weekend() -> None:
    s = copy.deepcopy(GOOD)
    s["handel"]["wochenende_halten"] = True
    assert any("Wochenende" in e for e in S.validate(s).errors)


def test_absolute_slippage_and_unrolled_futures_cfds_are_caught() -> None:
    s = copy.deepcopy(GOOD)
    s["kosten"]["slippage"] = "1 Punkt je Fill"
    s["handel"]["maerkte"] = ["WTI"]
    errs = " | ".join(S.validate(s).errors)
    assert "absoluten Punkten" in errs and "Rolls" in errs


def test_a_strategy_without_spec_file_is_reported() -> None:
    c = S.load("gibt_es_nicht")
    assert not c.agreed and "keine Spezifikation" in c.errors[0]
