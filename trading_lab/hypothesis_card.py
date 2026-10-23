"""Validator für Hypothesen-Karten (PATTERN_RESEARCH.md, Abschnitt 3).

Aufruf:
    python tools/validate_hypothesis.py hypotheses/H-001.yaml [weitere.yaml ...] [--audit data/audit_status.yaml]

Exit-Code 0 = alle Karten gültig, 1 = mindestens eine Karte ungültig.
Karten im Status "entwurf" dürfen noch fehlende S0-Zahlen haben (nur Warnung);
ab "registriert" ist alles Pflicht.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

STATUS = {"entwurf", "registriert", "kill", "bestanden_train", "bestanden_validate"}
FAMILIEN = {f"Z{i}" for i in range(1, 11)}
EINHEITEN = {"sigma", "spread", "perzentil", "session_min", "z", "quote", "verhaeltnis", "korrelation"}
DATENABSCHNITTE = {"train", "exploration"}
DOSIS_ERWARTUNG = {"monoton_steigend", "monoton_fallend"}

# Begriffe für Formen-Scanning und diskretionäre Strukturen (PATTERN_RESEARCH Abschnitt 4)
VERBOTENE_BEGRIFFE = [
    "flagge", "wimpel", "keil", "dreieck", "kopf-schulter", "kopf schulter", "schulter-kopf",
    "head and shoulders", "doppelboden", "doppeltop", "double top", "double bottom",
    "kerzenmuster", "candlestick", "engulfing", "hammer", "doji",
    "order block", "orderblock", "fair value gap", "fvg", "liquidity sweep", "smc",
    "break of structure", "change of character",
]

KILL_DEFAULTS = {
    # Feld: (Default, Richtung "min" = Wert darf nur größer werden)
    "t_min": 3.0,
    "placebo_mindestabstand": 0.5,
    "dosis_rangkorrelation_min": 0.8,
    "vorzeichen_jahre_min": 0.7,
    "netto_effekt_min_bp": 0.0,
}

MAX_ZUSTAENDE = 2
MAX_VARIANTEN = 10
MIN_HORIZONT = 60
S0_GRENZE_DEFAULT = 0.10


@dataclass
class Ergebnis:
    pfad: str
    fehler: list[str] = field(default_factory=list)
    warnungen: list[str] = field(default_factory=list)

    @property
    def gueltig(self) -> bool:
        return not self.fehler


def _leer(wert) -> bool:
    return wert is None or (isinstance(wert, (str, list, dict)) and len(wert) == 0)


def _ist_zahl(wert) -> bool:
    return isinstance(wert, (int, float)) and not isinstance(wert, bool)


def _texte(karte: dict) -> str:
    """Alle Freitexte der Karte, kleingeschrieben, für die Suche nach verbotenen Begriffen."""
    teile = [
        str(karte.get(k, "")) for k in
        ("titel", "beobachtung", "ort", "zeit", "ereignis_definition", "behauptung", "placebo")
    ]
    for z in karte.get("zustaende") or []:
        if isinstance(z, dict):
            teile.append(str(z.get("definition", "")))
    return " ".join(teile).lower()


def validiere(karte: dict, pfad: str = "<karte>", audit: dict | None = None) -> Ergebnis:
    erg = Ergebnis(pfad)
    f, w = erg.fehler, erg.warnungen

    if not isinstance(karte, dict):
        f.append("Karte ist kein YAML-Objekt")
        return erg

    status = karte.get("status")
    if status not in STATUS:
        f.append(f"status muss einer von {sorted(STATUS)} sein")
    entwurf = status == "entwurf"

    if not re.fullmatch(r"H-\d{3}", str(karte.get("id", ""))):
        f.append("id muss das Format H-<3 Ziffern> haben")

    for feld in ("titel", "beobachtung", "ort", "zeit", "ereignis_definition", "richtung",
                 "behauptung", "placebo"):
        if _leer(karte.get(feld)):
            f.append(f"Pflichtfeld '{feld}' fehlt oder ist leer")

    # Mechanismus
    mech = karte.get("mechanismus") or {}
    for feld in ("wer", "warum"):
        if _leer(mech.get(feld)):
            f.append(f"mechanismus.{feld} fehlt (wer handelt warum?)")
    if _leer(mech.get("quellen")):
        f.append("mechanismus.quellen: mindestens eine Quelle angeben")

    # Zustände
    zustaende = karte.get("zustaende") or []
    if not zustaende:
        f.append("zustaende: mindestens eine Zustandsfamilie angeben")
    if len(zustaende) > MAX_ZUSTAENDE:
        f.append(f"zustaende: höchstens {MAX_ZUSTAENDE} Familien kombinieren (sind {len(zustaende)})")
    for i, z in enumerate(zustaende):
        if not isinstance(z, dict):
            f.append(f"zustaende[{i}] ist kein Objekt")
            continue
        if z.get("familie") not in FAMILIEN:
            f.append(f"zustaende[{i}].familie muss eine von Z1..Z10 sein")
        if _leer(z.get("definition")):
            f.append(f"zustaende[{i}].definition fehlt")
        if z.get("einheit") not in EINHEITEN:
            f.append(f"zustaende[{i}].einheit muss dimensionslos sein: {sorted(EINHEITEN)}")

    # Verbotene Begriffe
    text = _texte(karte)
    treffer = [b for b in VERBOTENE_BEGRIFFE if re.search(rf"(?<![a-zäöü]){re.escape(b)}(?![a-zäöü])", text)]
    if treffer:
        f.append(f"Formen-/Strukturbegriffe sind nicht erlaubt (PATTERN_RESEARCH Abschnitt 4): {treffer}")

    # Dosis-Wirkung
    dosis = karte.get("dosis_wirkung") or {}
    if _leer(dosis.get("staerke_variable")):
        f.append("dosis_wirkung.staerke_variable fehlt")
    if dosis.get("erwartung") not in DOSIS_ERWARTUNG:
        f.append(f"dosis_wirkung.erwartung muss einer von {sorted(DOSIS_ERWARTUNG)} sein")

    # Horizont und S0
    horizont = karte.get("horizont_min")
    s0 = karte.get("s0") or {}
    if not _ist_zahl(horizont) or horizont <= 0:
        f.append("horizont_min muss eine positive Zahl sein")
    elif horizont < MIN_HORIZONT and _leer(s0.get("kurz_horizont_begruendung")):
        f.append(f"horizont_min < {MIN_HORIZONT} nur mit s0.kurz_horizont_begruendung erlaubt")

    grenze = s0.get("grenze", S0_GRENZE_DEFAULT)
    if not _ist_zahl(grenze) or grenze > S0_GRENZE_DEFAULT:
        f.append(f"s0.grenze darf höchstens {S0_GRENZE_DEFAULT} sein (nur verschärfen)")

    s0_zahlen = {k: s0.get(k) for k in ("kosten_rt_bp", "typische_bewegung_bp", "s0_quote", "min_effekt_bp")}
    fehlend = [k for k, v in s0_zahlen.items() if not _ist_zahl(v)]
    if fehlend:
        meldung = f"s0: Werte fehlen {fehlend} (vor Registrierung aus Train-Daten berechnen)"
        (w if entwurf else f).append(meldung)
    else:
        kosten, bewegung, quote, min_eff = (s0_zahlen[k] for k in
                                            ("kosten_rt_bp", "typische_bewegung_bp", "s0_quote", "min_effekt_bp"))
        if kosten <= 0 or bewegung <= 0:
            f.append("s0: kosten_rt_bp und typische_bewegung_bp müssen > 0 sein")
        else:
            if abs(quote - kosten / bewegung) > 0.005:
                f.append(f"s0.s0_quote ({quote}) passt nicht zu kosten/bewegung ({kosten / bewegung:.4f})")
            if _ist_zahl(grenze) and quote > grenze:
                f.append(f"S0-Gate nicht bestanden: s0_quote {quote} > grenze {grenze} → Hypothese nicht testen")
            if min_eff < 2 * kosten - 1e-9:
                f.append(f"s0.min_effekt_bp muss >= 2 x kosten_rt_bp sein ({2 * kosten})")

    # Kill-Kriterien: nur verschärfen
    kill = karte.get("kill_kriterien") or {}
    for feld, default in KILL_DEFAULTS.items():
        wert = kill.get(feld)
        if not _ist_zahl(wert):
            f.append(f"kill_kriterien.{feld} fehlt")
        elif wert < default:
            f.append(f"kill_kriterien.{feld} = {wert} ist lockerer als der Default {default}")

    # Varianten
    varianten = karte.get("varianten") or []
    if not varianten:
        f.append("varianten: mindestens die Basisvariante angeben")
    if len(varianten) > MAX_VARIANTEN:
        f.append(f"varianten: höchstens {MAX_VARIANTEN} (Trial-Budget), sind {len(varianten)}")

    # Datenabschnitt
    if karte.get("datenabschnitt") not in DATENABSCHNITTE:
        f.append(f"datenabschnitt muss einer von {sorted(DATENABSCHNITTE)} sein (validate/holdout erst nach Freigabe)")

    # Instrumente und Audit
    instrumente = karte.get("instrumente") or []
    if not instrumente:
        f.append("instrumente: mindestens ein Instrument angeben")
    if audit is not None:
        nicht_frei = [i for i in instrumente if audit.get(i) is not True]
        if nicht_frei:
            meldung = f"Instrumente ohne audit_passed: {nicht_frei}"
            (w if entwurf else f).append(meldung)
    elif not entwurf:
        f.append("Audit-Status fehlt: ab Status 'registriert' muss --audit angegeben werden")

    return erg


def lade_yaml(pfad: Path):
    with pfad.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hypothesen-Karten prüfen")
    parser.add_argument("karten", nargs="+", type=Path)
    parser.add_argument("--audit", type=Path, help="YAML: {Instrument: true/false} (audit_passed)")
    args = parser.parse_args(argv)

    audit = lade_yaml(args.audit) if args.audit else None
    alle_gueltig = True
    for pfad in args.karten:
        erg = validiere(lade_yaml(pfad), str(pfad), audit)
        alle_gueltig &= erg.gueltig
        print(f"{'OK  ' if erg.gueltig else 'FAIL'} {pfad}")
        for m in erg.fehler:
            print(f"   FEHLER:  {m}")
        for m in erg.warnungen:
            print(f"   WARNUNG: {m}")
    return 0 if alle_gueltig else 1


if __name__ == "__main__":
    sys.exit(main())
