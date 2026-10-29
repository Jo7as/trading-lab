#!/usr/bin/env python3
"""Prop-firm check for one strategy variant: pass / survival / payouts per firm and risk.

Uses the strategy's full cost model (``optimizer.run_variant``) on TRAIN only (the
holdout belongs to the pipeline, S5). Firms: the 2026 presets in
``trading_lab/prop_sim.py`` (rules: ``docs/PROP_FIRMS.md``).

  .venv/bin/python studies/prop_firms.py noise_area_momentum US100 '{"pruefung": "30min", "trailing": "band"}'
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import replace

sys.path.insert(0, ".")

import pandas as pd  # noqa: E402

from trading_lab import prop_sim  # noqa: E402
from trading_lab.optimizer import run_variant  # noqa: E402
from trading_lab.registry import HOLDOUT_START, REGISTRY, load_cached  # noqa: E402

FIRMS = ("fundednext_instant", "fundingpips_zero", "the5ers_hyper_growth", "ftmo_1step", "ftmo")


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy")
    ap.add_argument("asset")
    ap.add_argument("params", nargs="?", default="{}")
    ap.add_argument("--risks", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--runs", type=int, default=2000)
    a = ap.parse_args()
    spec = REGISTRY[a.strategy][0](a.asset)
    m1 = load_cached(a.asset)
    m1 = m1[m1.index < HOLDOUT_START]
    tr = run_variant(spec, replace(spec.base, **json.loads(a.params)), m1)
    risks = tuple(float(x) for x in a.risks.split(","))
    rows = prop_sim.risk_sweep(tr, FIRMS, risks, runs=a.runs)
    print(f"{a.strategy} {a.asset} {a.params}: {len(tr)} Trades, Training bis {HOLDOUT_START.date()}\n")
    inst = pd.DataFrame([r for r in rows if r["type"] == "instant"])
    if len(inst):
        print("Instant-Konten (1 Jahr, Block-Bootstrap): überlebt / mit Auszahlung / Ø Auszahlung in % des Kontos")
        for _, r in inst.iterrows():
            print(f"  {r.firm:22s} {r.risk_pct:4.2f}%/Trade  überlebt {r.survive_1y:5.0%}  Auszahlung {r.payout_rate:5.0%}"
                  f"  Ø {r.payout_pct_of_account:5.1f}% des Kontos{'' if r.verified else '  (Regeln ungeprüft)'}")
    ch = pd.DataFrame([r for r in rows if r["type"] == "challenge"])
    if len(ch):
        print("\nChallenges: bestanden / Median Tage / Gründe fürs Scheitern")
        for _, r in ch.iterrows():
            print(f"  {r.firm:22s} {r.risk_pct:4.2f}%/Trade  bestanden {r.pass_rate:5.0%}  Median {r.median_days:5.0f} Tage"
                  f"  {r.fails}")
    print("\nGebühren sind nicht eingetragen: Auszahlung minus Kontopreis = Erwartungswert.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
