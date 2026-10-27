"""Strategy specifications: written down and agreed BEFORE a strategy is coded or tested.

Why: on 2026-10-27 a "trend strategy" (H-012) was coded straight from an idea - no
stop-loss, no take-profit, risk parity that ignored correlation, guessed financing,
no tests, no reference to compare with. The pipeline rejected it correctly, and the
verdict still said nothing about the idea. The pipeline only judges what it is given;
this module makes sure what it is given is a well-defined strategy.

A spec lives in ``docs/specs/<strategy>.yaml`` (template: ``docs/specs/TEMPLATE.yaml``).
``validate`` lists every missing or weak field. ``status: abgestimmt`` is set by the
USER, never by an agent; the guided check (``studies/pruefen.py``) and the batch runner
look the spec up and mark every run without an agreed spec as
"UNGEPRÜFTE UMSETZUNG".

    uv run python -m trading_lab.strategy_spec docs/specs/my_idea.yaml
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SPEC_DIR = Path(__file__).resolve().parent.parent / "docs" / "specs"
REPO = SPEC_DIR.parent.parent
STATUS = ("entwurf", "abgestimmt")
KONTOTYP = ("prop_standard", "prop_swing", "eigenkapital")
UNCHECKED = "UNGEPRÜFTE UMSETZUNG"


@dataclass
class SpecCheck:
    path: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    status: str = "fehlt"

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def agreed(self) -> bool:
        return self.ok and self.status == "abgestimmt"

    def summary(self) -> str:
        head = f"{self.path}: {'OK' if self.ok else 'FEHLER'} (status {self.status})"
        return "\n".join([head] + [f"  FEHLER:  {e}" for e in self.errors]
                         + [f"  WARNUNG: {w}" for w in self.warnings])


def _empty(v) -> bool:
    return v is None or (isinstance(v, (str, list, dict)) and len(v) == 0) or (isinstance(v, str) and not v.strip())


def validate(spec: dict, path: str = "<spec>") -> SpecCheck:
    c = SpecCheck(path)
    e, w = c.errors, c.warnings
    if not isinstance(spec, dict):
        e.append("Spec ist kein YAML-Objekt")
        return c
    c.status = str(spec.get("status", ""))
    if c.status not in STATUS:
        e.append(f"status muss {STATUS} sein")
    if not re.fullmatch(r"S-\d{3}", str(spec.get("id", ""))):
        e.append("id muss S-<3 Ziffern> sein")
    for k in ("strategie", "titel"):
        if _empty(spec.get(k)):
            e.append(f"{k} fehlt")
    if c.status == "abgestimmt" and (_empty(spec.get("abgestimmt_von")) or _empty(spec.get("abgestimmt_am"))):
        e.append("status 'abgestimmt' braucht abgestimmt_von und abgestimmt_am (Freigabe durch den Nutzer)")

    idee = spec.get("idee") or {}
    if _empty(idee.get("mechanismus")):
        e.append("idee.mechanismus fehlt - warum sollte die Strategie funktionieren?")
    if _empty(idee.get("quellen")):
        w.append("idee.quellen leer")

    h = spec.get("handel") or {}
    for k in ("maerkte", "zeitrahmen", "max_haltedauer"):
        if _empty(h.get(k)):
            e.append(f"handel.{k} fehlt")
    if h.get("kontotyp") not in KONTOTYP:
        e.append(f"handel.kontotyp muss {KONTOTYP} sein")
    if h.get("kontotyp") == "prop_standard" and h.get("wochenende_halten"):
        e.append("prop_standard (funded) darf nicht übers Wochenende halten - kontotyp prop_swing wählen")

    en = spec.get("entry") or {}
    if _empty(en.get("bedingungen")):
        e.append("entry.bedingungen fehlen")
    if _empty(en.get("ausfuehrung")):
        e.append("entry.ausfuehrung fehlt (Market/Limit, auf welcher Kerze?)")

    sl = spec.get("stop_loss") or {}
    if _empty(sl.get("regel")):
        e.append("stop_loss.regel fehlt - ohne Stop kein festes Risiko je Trade (H-012)")
    tp = spec.get("take_profit") or {}
    if _empty(tp.get("regel")):
        e.append("take_profit.regel fehlt (festes Ziel, Trailing, Zeit-Exit oder Gegensignal)")
    elif str(tp.get("regel")).strip().lower() in ("keiner", "kein", "none") and _empty(tp.get("begruendung")):
        e.append("take_profit 'keiner' nur mit Begründung")

    r = spec.get("risiko") or {}
    rp = r.get("risiko_je_trade_pct")
    if not isinstance(rp, (int, float)) or not 0 < rp <= 2:
        e.append("risiko.risiko_je_trade_pct muss zwischen 0 und 2 % liegen")
    if len(h.get("maerkte") or []) > 1 and _empty(r.get("korrelation")):
        e.append("risiko.korrelation fehlt - mehrere Märkte, wie wird korreliertes Risiko begrenzt?")

    k = spec.get("kosten") or {}
    try:
        from . import costs
        costs.get(k.get("profil"))
    except Exception:
        e.append("kosten.profil unbekannt (trading_lab/costs.py)")
    long_hold = not re.search(r"\b(intraday|minute|stunde)", str(h.get("max_haltedauer", "")).lower())
    if long_hold and _empty(k.get("swaps")):
        e.append("kosten.swaps fehlt - Haltedauer über Nacht braucht ein Finanzierungsmodell")
    if _empty(k.get("slippage")):
        w.append("kosten.slippage leer")
    elif re.search(r"punkt|point", str(k.get("slippage")).lower()):
        e.append("kosten.slippage in absoluten Punkten - relativ (bp) angeben")

    d = spec.get("daten") or {}
    if _empty(d.get("quelle")):
        e.append("daten.quelle fehlt")
    maerkte = {str(m).upper() for m in h.get("maerkte") or []}
    if maerkte & {"WTI", "BRENT", "COPPER", "UST10Y"} and _empty(d.get("bekannte_probleme")):
        e.append("Futures-basierte CFDs (Öl/Kupfer/Anleihen): daten.bekannte_probleme muss die Rolls behandeln")

    var = spec.get("varianten") or []
    if not var:
        e.append("varianten: mindestens die Basisvariante")
    elif len(var) > 10:
        w.append(f"{len(var)} vorab festgelegte Varianten - jede kostet Beweiskraft in S4")

    if _empty(spec.get("referenz")):
        e.append("referenz fehlt - womit wird das Ergebnis auf Plausibilität geprüft?")

    tests = spec.get("tests") or []
    if not tests:
        e.append("tests fehlen - jede Regel braucht einen Test auf einem kleinen Kursverlauf")
    for t in tests:
        if not (REPO / str(t)).exists():
            (e if c.status == "abgestimmt" else w).append(f"Testdatei {t} existiert nicht")
    return c


def load(strategy: str, *, directory: Path = SPEC_DIR) -> SpecCheck:
    """The spec check for a registry strategy (status 'fehlt' if there is none)."""
    path = directory / f"{strategy}.yaml"
    if not path.exists():
        c = SpecCheck(str(path))
        c.errors.append(f"keine Spezifikation {path.relative_to(REPO) if path.is_relative_to(REPO) else path}"
                        " - Vorlage: docs/specs/TEMPLATE.yaml")
        return c
    return validate(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Strategie-Spezifikationen prüfen")
    ap.add_argument("specs", nargs="+", type=Path)
    a = ap.parse_args(argv)
    ok = True
    for p in a.specs:
        c = validate(yaml.safe_load(p.read_text(encoding="utf-8")), str(p))
        ok &= c.ok
        print(c.summary())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
