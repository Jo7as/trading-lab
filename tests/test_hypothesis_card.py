import copy
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

from trading_lab.hypothesis_card import main, validiere  # noqa: E402

AUDIT_OK = {i: True for i in [
    "US100", "US30", "SP500", "GER40", "UK100", "JP225",
    "XAUUSD", "XAGUSD", "COPPER", "BRENT", "UST10Y",
]}


def lade(name):
    return yaml.safe_load((ROOT / "docs" / "hypotheses" / name).read_text(encoding="utf-8"))


def registrierte_karte():
    karte = lade("H-001.yaml")
    karte["status"] = "registriert"
    # Stufe deklarieren und deren Kill-Kriterium erfuellen - sonst warnt der
    # Validator (fehlendes Kriterium -> Stufen-Default gilt), und der Test unten
    # prueft "keine Warnungen".
    karte["stufe"] = "S1"
    karte["kill_kriterien"] = {"min_trades": 100}
    karte["s0"].update(kosten_rt_bp=2.0, typische_bewegung_bp=40.0, s0_quote=0.05, min_effekt_bp=4.0)
    return karte


def _ohne_s0(status="entwurf"):
    """A card with the S0 block emptied — tests the validator, not our own cards."""
    karte = lade("H-001.yaml")
    karte["status"] = status
    for feld in ("kosten_rt_bp", "typische_bewegung_bp", "s0_quote", "min_effekt_bp"):
        karte["s0"][feld] = None
    return karte


def test_entwurf_mit_fehlendem_s0_warnt():
    erg = validiere(_ohne_s0("entwurf"), audit=AUDIT_OK)
    assert erg.gueltig, erg.fehler
    assert any("s0" in w for w in erg.warnungen)


def test_registrierte_karte_vollstaendig_gueltig():
    erg = validiere(registrierte_karte(), audit=AUDIT_OK)
    assert erg.gueltig, erg.fehler
    assert not erg.warnungen


def test_registriert_ohne_s0_wird_abgelehnt():
    erg = validiere(_ohne_s0("registriert"), audit=AUDIT_OK)
    assert any("s0: Werte fehlen" in m for m in erg.fehler)


def test_registriert_ohne_audit_wird_abgelehnt():
    erg = validiere(registrierte_karte(), audit=None)
    assert any("Audit-Status fehlt" in m for m in erg.fehler)


def test_instrument_ohne_audit_freigabe():
    audit = dict(AUDIT_OK, GER40=False)
    erg = validiere(registrierte_karte(), audit=audit)
    assert any("audit_passed" in m and "GER40" in m for m in erg.fehler)


def test_s0_gate_warnt_statt_zu_blockieren():
    """Die S0-Quote ist eine Prognose - sie darf den Test nicht verhindern.

    Vorher: `s0_quote 0.2 > grenze 0.1` -> Fehler -> "Hypothese nicht testen".
    Jetzt: Warnung. Der Kandidat geht in S1 (brutto); scheitert er, dann an S2 auf
    echten Ergebnissen. Ein Vorhersage-Wert darf keine Strategie toeten.
    """
    karte = registrierte_karte()
    karte["s0"].update(kosten_rt_bp=6.0, typische_bewegung_bp=30.0, s0_quote=0.2, min_effekt_bp=12.0)
    erg = validiere(karte, audit=AUDIT_OK)
    assert erg.gueltig, erg.fehler
    assert any("S0-Quote" in w for w in erg.warnungen)
    assert not any("nicht testen" in m for m in erg.fehler)


def test_s0_quote_inkonsistent():
    karte = registrierte_karte()
    karte["s0"]["s0_quote"] = 0.01
    erg = validiere(karte, audit=AUDIT_OK)
    assert any("passt nicht" in m for m in erg.fehler)


def test_min_effekt_zu_klein_warnt_statt_zu_blockieren():
    """Auch der Mindest-Effekt ist eine Schaetzung, kein Ergebnis - Warnung, kein Fehler."""
    karte = registrierte_karte()
    karte["s0"]["min_effekt_bp"] = 3.0
    erg = validiere(karte, audit=AUDIT_OK)
    assert erg.gueltig, erg.fehler
    assert any("min_effekt_bp" in w for w in erg.warnungen)


def test_kurzer_horizont_ohne_begruendung():
    karte = registrierte_karte()
    karte["horizont_min"] = 5
    erg = validiere(karte, audit=AUDIT_OK)
    assert any("horizont_min" in m for m in erg.fehler)


def test_kurzer_horizont_mit_begruendung_erlaubt():
    karte = registrierte_karte()
    karte["horizont_min"] = 30
    karte["s0"]["kurz_horizont_begruendung"] = "S0-Quote auf 30 Min für US500 vorab 0,05"
    erg = validiere(karte, audit=AUDIT_OK)
    assert erg.gueltig, erg.fehler


def test_kill_kriterium_ohne_begruendung_nicht_lockern():
    """Ein lockereres Kriterium als der Stufen-Default braucht eine Begruendung.

    Die Feldnamen sind jetzt stufenbezogen (S1: min_trades, S3: vorzeichen_jahre_min,
    S4: t_daily_min) - vorher waren es globale Namen mit t_min auf der Trade-Reihe.
    """
    karte = registrierte_karte()
    karte["kill_kriterien"]["min_trades"] = 10        # S1-Default ist 100
    erg = validiere(karte, audit=AUDIT_OK)
    assert any("min_trades" in m and "lockerer" in m for m in erg.fehler)


def test_kill_kriterium_lockerung_mit_begruendung_erlaubt():
    """Vorher war jede Lockerung verboten - ein falsch gesetzter Wert war fuer immer falsch."""
    karte = registrierte_karte()
    karte["kill_kriterien"]["min_trades"] = 10
    karte["abweichung_begruendung"] = "hochfrequenter Kandidat, 10 Trades je Fenster reichen"
    erg = validiere(karte, audit=AUDIT_OK)
    assert erg.gueltig, erg.fehler


def test_kill_kriterien_verschaerfen_erlaubt():
    karte = registrierte_karte()
    karte["kill_kriterien"]["min_trades"] = 500
    assert validiere(karte, audit=AUDIT_OK).gueltig


def test_kill_kriterien_je_stufe():
    """S4 ist streng, S1 nicht - dieselbe Karte, zwei Stufen."""
    karte = registrierte_karte()
    karte["stufe"] = "S4"
    karte["kill_kriterien"] = {"min_trades": 100}
    erg = validiere(karte, audit=AUDIT_OK)
    assert any("t_daily_min" in w for w in erg.warnungen)   # S4-Kriterium fehlt -> Default


def test_s0_grenze_lockerung_nur_mit_begruendung():
    """Eine lockerere S0-Grenze ist erlaubt, aber begruendungspflichtig."""
    karte = registrierte_karte()
    karte["s0"]["grenze"] = 0.2
    assert any("s0.grenze" in w for w in validiere(karte, audit=AUDIT_OK).warnungen)
    karte["abweichung_begruendung"] = "Grenze auf dem korrigierten Datenrahmen neu kalibriert"
    assert not any("s0.grenze" in w for w in validiere(karte, audit=AUDIT_OK).warnungen)


def test_zu_viele_zustaende():
    karte = registrierte_karte()
    karte["zustaende"].append(copy.deepcopy(karte["zustaende"][0]))
    erg = validiere(karte, audit=AUDIT_OK)
    assert any("höchstens 2 Familien" in m for m in erg.fehler)


def test_absolute_einheit_abgelehnt():
    karte = registrierte_karte()
    karte["zustaende"][0]["einheit"] = "pips"
    erg = validiere(karte, audit=AUDIT_OK)
    assert any("dimensionslos" in m for m in erg.fehler)


def test_unbekannte_familie():
    karte = registrierte_karte()
    karte["zustaende"][0]["familie"] = "Z11"
    erg = validiere(karte, audit=AUDIT_OK)
    assert any("Z1..Z10" in m for m in erg.fehler)


@pytest.mark.parametrize("begriff", ["Bull-Flagge", "Order Block", "Fair Value Gap", "Kopf-Schulter", "Doji"])
def test_formen_begriffe_abgelehnt(begriff):
    karte = registrierte_karte()
    karte["ereignis_definition"] = f"Kurs bildet eine {begriff} an der Marke"
    erg = validiere(karte, audit=AUDIT_OK)
    assert any("nicht erlaubt" in m for m in erg.fehler)


def test_kein_falscher_treffer_bei_teilwort():
    # "Keil" darf nicht in "Steilheit" anschlagen, "Hammer" nicht in "Hammerschmidt"
    karte = registrierte_karte()
    karte["beobachtung"] = karte["beobachtung"] + " Steilheit der Bewegung laut Hammerschmidt."
    assert validiere(karte, audit=AUDIT_OK).gueltig


def test_fehlender_mechanismus():
    karte = registrierte_karte()
    karte["mechanismus"]["wer"] = ""
    erg = validiere(karte, audit=AUDIT_OK)
    assert any("mechanismus.wer" in m for m in erg.fehler)


def test_varianten_budget_je_stufe():
    """S1/S2 sind unbegrenzt - dort gehoert die Exploration hin, nicht in einen Deckel.

    Vorher: `MAX_VARIANTEN = 10`, also hartes Limit fuer *jede* Karte. Das war das
    Gegenteil von "hunderte Varianten testen".
    """
    karte = registrierte_karte()
    karte["varianten"] = [{"name": f"v{i}", "aenderung": "x"} for i in range(600)]
    # S1: unbegrenzt
    assert not any("Budget" in w for w in validiere(karte, audit=AUDIT_OK).warnungen)
    # S3: Budget 500 -> 600 uebersteigt es (Warnung, kein Fehler)
    karte["stufe"] = "S3"
    karte["kill_kriterien"] = {"vorzeichen_jahre_min": 0.7, "nachbar_anteil_min": 0.5}
    assert any("Budget" in w for w in validiere(karte, audit=AUDIT_OK).warnungen)


def test_validate_abschnitt_nicht_erlaubt():
    karte = registrierte_karte()
    karte["datenabschnitt"] = "holdout"
    erg = validiere(karte, audit=AUDIT_OK)
    assert any("datenabschnitt" in m for m in erg.fehler)


def test_falsche_id():
    karte = registrierte_karte()
    karte["id"] = "H1"
    assert any("id" in m for m in validiere(karte, audit=AUDIT_OK).fehler)


def test_cli_exit_codes(tmp_path):
    gut = tmp_path / "gut.yaml"
    gut.write_text(yaml.safe_dump(registrierte_karte(), allow_unicode=True), encoding="utf-8")
    schlecht_karte = registrierte_karte()
    schlecht_karte["mechanismus"]["warum"] = ""
    schlecht = tmp_path / "schlecht.yaml"
    schlecht.write_text(yaml.safe_dump(schlecht_karte, allow_unicode=True), encoding="utf-8")
    audit = tmp_path / "audit.yaml"
    audit.write_text(yaml.safe_dump(AUDIT_OK), encoding="utf-8")

    assert main([str(gut), "--audit", str(audit)]) == 0
    assert main([str(gut), str(schlecht), "--audit", str(audit)]) == 1
