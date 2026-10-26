#!/usr/bin/env python3
"""Geführte Strategie-Prüfung - der feste Ablauf für jede Strategie.

Jede Strategie läuft durch dieselben 12 Schritte (trading_lab/guided.py): Umgebung,
Auswahl, Daten, Basis-Backtest, Suchraum, Optimierung, S1 Brutto/Placebo, S2 Kosten,
S3 Robustheit, S4 Signifikanz, S5 Holdout (nur mit Bestätigung), Prop-Firm-Einschätzung
und Bericht. Nach jedem Schritt: Erklärung, Zwischenergebnis, Rückfrage.

  uv run python studies/pruefen.py
  uv run python studies/pruefen.py --strategie momentum --instrument US100
  uv run python studies/pruefen.py --liste
"""
from __future__ import annotations

import argparse
import sys
import warnings

sys.path.insert(0, ".")

from trading_lab.guided import BUDGETS, STEPS, GuidedCheck  # noqa: E402
from trading_lab.registry import DESCRIPTIONS, NOT_ADAPTED, REGISTRY  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategie", choices=sorted(REGISTRY))
    ap.add_argument("--instrument")
    ap.add_argument("--budget", choices=sorted(BUDGETS), help="s/n/g = schnell/normal/gruendlich")
    ap.add_argument("--liste", action="store_true", help="Strategien, Instrumente und die 12 Schritte zeigen")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    if args.liste:
        print("Die 12 Schritte (für jede Strategie gleich):")
        for i, s in enumerate(STEPS, 1):
            print(f"  {i:2d}. {s}")
        print("\nStrategien:")
        for n, (_, assets) in REGISTRY.items():
            print(f"  {n:20s} {', '.join(assets)}\n  {'':20s} {DESCRIPTIONS.get(n, '')}")
        for n, why in NOT_ADAPTED.items():
            print(f"  (nicht im Ablauf) {n}: {why}")
        return 0
    budget = None
    if args.budget:
        from trading_lab.registry import BUDGET_SCALE
        scale = BUDGET_SCALE.get(args.strategie or "", 1.0)
        budget = max(20, int(BUDGETS[args.budget][1] * scale))
    GuidedCheck(strategy=args.strategie, asset=args.instrument, budget=budget).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
