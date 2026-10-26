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

# "kill" = getestet und keine Kante. "nicht_nachweisbar" = der Test hatte zu wenig
# Power, um zu entscheiden (das war der Fehler, der die alten Karten zu Unrecht
# getoetet hat). "widerlegt" = der Effekt war ein Artefakt (z.B. Look-ahead).
STATUS = {"entwurf", "registriert", "kill", "nicht_nachweisbar", "widerlegt",
          "bestanden_train", "bestanden_validate"}
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

# Kill-Kriterien je **Stufe** des Trichters (trading_lab/screening.py), nicht global.
#
# Was vorher falsch war: alle Kriterien galten gleichzeitig, und `t_min = 3.0` wurde
# auf der *Trade*-Reihe geprüft. Mit sd(R) ~ 4 braucht ein 0.1-R-Edge dafür ~20.000
# Trades - kein Kandidat in diesem Repo konnte das erreichen, also starben die
# Kandidaten an fehlender Teststärke statt an fehlender Kante. Ab jetzt gilt:
# S1 ist locker (nur "es bewegt sich was"), erst S4 ist streng.
KILL_JE_STUFE: dict[str, dict] = {
    "S1": {"min_trades": 100},
    "S2": {},                                    # gross_bp > cost_bp
    "S3": {"vorzeichen_jahre_min": 0.7, "nachbar_anteil_min": 0.5},
    # t_daily_min 3.0 gilt nur, wenn die Zahl der getesteten Varianten unbekannt ist;
    # der Trichter nimmt sonst DSR bzw. screening.required_t(n_trials) (1 Variante: 1.645).
    "S4": {"t_daily_min": 3.0, "placebo_vorsprung_min": 0.0, "pbo_max": 0.25},
    "S5": {},                                    # Holdout bestanden
}
STUFEN = tuple(KILL_JE_STUFE)

# Abweichungen von einem Stufen-Kriterium sind erlaubt, aber nur mit Begruendung
# (Feld `abweichung_begruendung`). Ohne Begruendung bleibt die Pruefung scharf -
# vorher war jede Lockerung verboten, also war ein falsch gesetzter Wert fuer immer
# falsch.
MAX_ZUSTAENDE = 2
MIN_HORIZONT = 60
S0_GRENZE_DEFAULT = 0.10

# Varianten sind kein Deckel mehr, sondern ein Budget **je Stufe**. Exploration darf
# breit sein; die Disziplin kommt ueber das Versuchsregister und den Trichter, nicht
# ueber ein Limit von 10 (das war das Gegenteil von "hunderte Varianten testen").
VARIANTEN_BUDGET: dict[str, int | None] = {"S1": None, "S2": None, "S3": 500,
                                           "S4": 50, "S5": 5}


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

    begruendet = not _leer(karte.get("abweichung_begruendung"))
    grenze = s0.get("grenze", S0_GRENZE_DEFAULT)
    if not _ist_zahl(grenze) or grenze > S0_GRENZE_DEFAULT:
        # A4: eine lockerere Grenze ist erlaubt, wenn sie begruendet ist. Vorher war
        # jede Lockerung verboten, also war ein falsch gesetzter Wert fuer immer falsch.
        if not begruendet:
            w.append(f"s0.grenze {grenze} ist lockerer als der Default "
                     f"{S0_GRENZE_DEFAULT} - nur mit abweichung_begruendung erlaubt")

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
            # A1: die S0-Quote ist eine **Prognose**, kein Ergebnis. Sie darf den Test
            # nicht verhindern - sie wandert in die Warnungen. Kosten sind Stufe S2 des
            # Trichters und werden dort auf echten Ergebnissen angewendet.
            if _ist_zahl(grenze) and quote > grenze:
                w.append(f"S0-Quote {quote:.3f} > {grenze}: die Kostenschaetzung ist "
                         f"unguenstig, aber das ist eine Prognose. Der Kandidat geht in "
                         f"S1 (brutto) und scheitert, falls ueberhaupt, erst an S2.")
            if min_eff < 2 * kosten - 1e-9:
                w.append(f"s0.min_effekt_bp ({min_eff}) < 2 x kosten_rt_bp "
                         f"({2 * kosten}): ebenfalls nur eine Schaetzung - S2 entscheidet.")

    # A3: Kill-Kriterien je **Stufe** statt global. Fehlt ein Kriterium, gilt der
    # Stufen-Default (Warnung, kein Fehler); ist es lockerer als der Default, braucht
    # es eine Begruendung.
    stufe = str(karte.get("stufe") or "S1").upper()
    if stufe not in STUFEN:
        f.append(f"stufe muss eine von {list(STUFEN)} sein, ist {stufe!r}")
        stufe = "S1"
    kill = karte.get("kill_kriterien") or {}
    for feld, default in KILL_JE_STUFE[stufe].items():
        wert = kill.get(feld)
        if wert is None:
            w.append(f"kill_kriterien.{feld} fehlt ({stufe}) - Default {default} gilt")
        elif _ist_zahl(wert) and wert < default and not begruendet:
            f.append(f"kill_kriterien.{feld} = {wert} ist lockerer als {default} "
                     f"({stufe}) - nur mit abweichung_begruendung erlaubt")

    # A2: Varianten sind ein Budget **je Stufe**, kein globaler Deckel. S1/S2 sind
    # unbegrenzt, weil dort die Exploration hingehoert.
    varianten = karte.get("varianten") or []
    if not varianten:
        f.append("varianten: mindestens die Basisvariante angeben")
    budget = VARIANTEN_BUDGET.get(stufe)
    n_var = karte.get("n_varianten", len(varianten))
    if budget is not None and _ist_zahl(n_var) and n_var > budget:
        w.append(f"{n_var} Varianten uebersteigen das Budget {budget} fuer {stufe}; "
                 f"S1/S2 sind unbegrenzt - breite Exploration gehoert dorthin.")

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
