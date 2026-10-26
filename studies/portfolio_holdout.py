#!/usr/bin/env python3
"""Ein eingefrorenes Portfolio EINMAL gegen den Holdout testen - Schritt für Schritt.

  uv run python studies/portfolio_holdout.py --liste
  uv run python studies/portfolio_holdout.py TK8 --train      # nur Train ansehen (kein Holdout)
  uv run python studies/portfolio_holdout.py TK8              # der eine Test (fragt nach 'HOLDOUT')

Ablauf (immer gleich):
  1. Portfolio zeigen: Komponenten, festes Risiko, vorab festgelegte Kriterien, Train-Ergebnis
  2. Sperre prüfen: Datei seit dem Einfrieren unverändert? (sonst Abbruch)
  3. Ledger prüfen: wurde dieser Holdout (oder der einer Komponente) schon angesehen?
  4. Bestätigung: 'HOLDOUT' eintippen
  5. Holdout rechnen: je Komponente, dann kombiniert
  6. Urteil nach den vorab festgelegten Kriterien + Prop-Firm-Simulation
  7. Bericht + Ledger-Eintrag (artifacts/pruefung/)
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

from trading_lab import portfolio as P  # noqa: E402
from trading_lab.guided import LEDGER, OUT_ROOT  # noqa: E402


def _h(k: int, title: str) -> None:
    print("\n" + "=" * 78 + f"\n  SCHRITT {k}/7: {title}\n" + "=" * 78)


def _pause() -> None:
    try:
        a = input("\n  [Enter] = weiter,  a = abbrechen > ").strip().lower()
    except EOFError:
        a = ""
    if a == "a":
        raise SystemExit("  Abgebrochen.")


def _print_stats(s: dict) -> None:
    labels = [("pct_pro_monat_mittel", "% pro Monat (Mittel)"), ("pct_pro_monat_median", "% pro Monat (Median)"),
              ("schlechtester_monat_pct", "schlechtester Monat %"), ("positive_monate", "Anteil positiver Monate"),
              ("max_drawdown_pct", "max. Drawdown %"), ("schlechtester_tag_pct", "schlechtester Tag %"),
              ("t_taeglich", "t (Tagesreihe)"), ("summe_pct", "Summe % im Fenster")]
    print(f"  Zeitraum {s['von']} .. {s['bis']} ({s['monate']} Monate)")
    for k, lab in labels:
        print(f"  {lab:28s} {s[k]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id", nargs="?")
    ap.add_argument("--liste", action="store_true")
    ap.add_argument("--train", action="store_true", help="nur das Train-Ergebnis zeigen")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    if args.liste or not args.id:
        for f in sorted(P.PF_DIR.glob("*.yaml")):
            pf = P.load(f.stem)
            lock = "eingefroren" if P.is_unchanged(pf.id) else "NICHT eingefroren / geändert"
            print(f"  {pf.id:5s} {pf.name:16s} {pf.raw.get('rolle', ''):9s} {len(pf.components)} Komponenten, "
                  f"Risiko {pf.risk_per_trade_pct} %/Trade  [{lock}]")
        return 0

    pf = P.load(args.id)
    _h(1, f"Portfolio {pf.id} - {pf.name}")
    print(f"  {pf.raw.get('beschreibung', '')}")
    print(f"  Rolle: {pf.raw.get('rolle')}   festes Risiko je Trade: {pf.risk_per_trade_pct} %")
    for comp in pf.raw["komponenten"]:
        print(f"\n   - {comp['strategy']}:{comp['asset']}")
        for k, v in (comp.get("regeln") or {}).items():
            print(f"       {k:26s} {v}")
    print("\n  Vorab festgelegte Holdout-Kriterien:")
    print("  " + pf.criteria.get("erklaerung", json.dumps(pf.criteria)))
    print("\n  Train-Ergebnis (in-sample, optimistisch):")
    _print_stats({**pf.raw["train_ergebnis"], "summe_pct": "-"})
    if args.train:
        return 0
    _pause()

    _h(2, "Sperre prüfen")
    ok, why = P.check(pf.id)
    if not ok:
        for w in why:
            print(f"  NEIN: {w}")
        print("  Ein Holdout-Test gegen veränderbare Regeln oder Kriterien ist kein Test - Abbruch.")
        return 1
    lock = json.loads((P.PF_DIR / f"{pf.id}.lock").read_text())
    print(f"  OK - Datei unverändert seit {lock['eingefroren']} (SHA-256 {lock['sha256'][:16]}...)")
    for f in lock.get("code_sha256", {}):
        print(f"  OK - Code unverändert: {f}")

    _h(3, "Ledger prüfen")
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    seen = [k for k in [f"portfolio:{pf.id}"] + [c.label for c in pf.components] if k in ledger]
    if seen:
        print("  ACHTUNG - diese Holdout-Daten wurden schon angesehen: " + ", ".join(seen))
        print("  Ein weiterer Blick ist kein unabhängiger Test mehr; der Bericht vermerkt das.")
    else:
        print("  Noch nie angesehen - dies wäre der eine unabhängige Test.")

    _h(4, "Bestätigung")
    print("  Nach diesem Schritt ist der Holdout (ab 2025-01-01) für alle Komponenten verbraucht.")
    try:
        a = input("  Zum Öffnen 'HOLDOUT' eintippen, [Enter] = nicht öffnen > ").strip()
    except EOFError:
        a = ""
    if a != "HOLDOUT":
        print("  Nicht geöffnet.")
        return 0

    _h(5, "Holdout rechnen")
    res = P.evaluate(pf, window="holdout", progress=lambda m: print("   " + m))
    s = res.stats()
    print("\n  Je Komponente (Summe R im Holdout):")
    for k, v in s["komponenten"].items():
        print(f"   {k:22s} {v['r_summe']:+8.2f} R   ({v['trades_tage']} Handelstage)")
    print("\n  Kombiniert, beim festen Risiko:")
    _print_stats(s)
    _pause()

    _h(6, "Urteil nach den vorab festgelegten Kriterien")
    v = res.verdict()
    for k, ok in v["kriterien"].items():
        print(f"   {'OK  ' if ok else 'NEIN'} {k}")
    print(f"\n  ERGEBNIS: {'BESTANDEN' if v['bestanden'] else 'NICHT BESTANDEN'}")
    tr = pf.raw["train_ergebnis"]["pct_pro_monat_mittel"]
    if s["pct_pro_monat_mittel"] is not None and tr:
        print(f"  Holdout {s['pct_pro_monat_mittel']:.2f} %/Monat gegen Train {tr:.2f} %/Monat "
              f"-> {100 * s['pct_pro_monat_mittel'] / tr:.0f} % des Trainingswerts")
    prop = {f: res.prop(f) for f in ("ftmo", "one_step", "instant_100k")}
    print("\n  Prop-Firm-Simulation auf den Holdout-Tagen (Block-Bootstrap):")
    for f, txt in prop.items():
        print("   " + txt)

    _h(7, "Bericht und Ledger")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / f"{stamp}_portfolio_{pf.id}"
    out.mkdir(parents=True, exist_ok=True)
    rep = {"portfolio": pf.id, "name": pf.name, "zeit": stamp, "lock": lock, "schon_gesehen": seen,
           "holdout": s, "urteil": v, "prop": prop, "train": pf.raw["train_ergebnis"]}
    (out / "ergebnis.json").write_text(json.dumps(rep, indent=1, default=str, ensure_ascii=False))
    md = [f"# Holdout-Test {pf.id} - {pf.name}", "", f"**Ergebnis: {'BESTANDEN' if v['bestanden'] else 'NICHT BESTANDEN'}**"
          + (" (Daten waren schon gesehen - kein unabhängiger Test)" if seen else ""), "",
          "| Kriterium | erfüllt |", "|---|---|"] + [f"| {k} | {'ja' if ok else 'nein'} |" for k, ok in v["kriterien"].items()]
    md += ["", "| Kennzahl | Train (in-sample) | Holdout |", "|---|---|---|"]
    for k in ("pct_pro_monat_mittel", "pct_pro_monat_median", "schlechtester_monat_pct", "positive_monate",
              "max_drawdown_pct", "schlechtester_tag_pct", "t_taeglich"):
        md.append(f"| {k} | {pf.raw['train_ergebnis'].get(k)} | {s.get(k)} |")
    md += ["", "## Komponenten im Holdout", ""] + [f"- {k}: {x['r_summe']:+.2f} R" for k, x in s["komponenten"].items()]
    md += ["", "## Prop-Firm-Simulation (Holdout)", ""] + [f"- {t}" for t in prop.values()]
    (out / "bericht.md").write_text("\n".join(md))
    entry = {"datum": datetime.now().isoformat(timespec="seconds"), "via": f"portfolio:{pf.id}"}
    ledger[f"portfolio:{pf.id}"] = {**entry, "wiederholt": bool(seen)}
    for c in pf.components:
        ledger.setdefault(c.label, entry)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(ledger, indent=1))
    print(f"  Bericht: {out / 'bericht.md'}\n  Ledger aktualisiert: {LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
