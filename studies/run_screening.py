#!/usr/bin/env python3
"""Run a variants table through the screening funnel (S1 -> ... -> S5).

The funnel is the one place the order of checks lives (see
``trading_lab/screening.py``). This runner is deliberately dumb: it reads a table,
runs the stages, and prints what each stage removed. The default stops at **S4** -
the holdout only runs with an explicit release, because it can be spent once.

Input format (one row per variant, missing values allowed):

    name, n, gross_mean, placebo_gross, gross_bp, cost_bp,
    pos_years, n_years, t_daily, placebo_edge, pbo, oos_pass

Producers: every card's study should write this via its own ``--json`` flag
(see docs/PIPELINE_AUDIT.md, step C2). Missing columns do not drop a row - the
stage that needs them reports that it skipped, which is the rule: nothing falls
out silently.

  .venv/bin/python studies/run_screening.py artifacts/momentum_sweep.json
  .venv/bin/python studies/run_screening.py table.json --up-to S3
  .venv/bin/python studies/run_screening.py table.json --holdout 2026-10-26
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd  # noqa: E402

from trading_lab.screening import DEFAULT_UP_TO, run_funnel  # noqa: E402


def load_variants(path: Path) -> pd.DataFrame:
    """Read a variants table from json (records) or csv."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "variants" in payload:
        payload = payload["variants"]
    return pd.DataFrame(payload)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", type=Path, help="variants.json (or .csv)")
    ap.add_argument("--up-to", default=DEFAULT_UP_TO,
                    choices=("S1", "S2", "S3", "S4", "all"))
    ap.add_argument("--holdout", default=None, metavar="FREIGABE_DATUM",
                    help="gibt den Holdout frei (z.B. 2026-10-26). Ohne das laeuft S5 nie.")
    ap.add_argument("--json", default="", help="Ergebnis hierhin schreiben")
    args = ap.parse_args()

    if not args.table.exists():
        print(f"Tabelle nicht gefunden: {args.table}", file=sys.stderr)
        return 2
    df = load_variants(args.table)
    if df.empty:
        print("Tabelle ist leer", file=sys.stderr)
        return 1

    up_to = None if args.up_to == "all" else args.up_to
    if up_to is None and not args.holdout:
        print("--up-to all ohne --holdout: der Holdout bleibt gesperrt (S5 wird "
              "uebersprungen, nicht still ausgefuehrt).", file=sys.stderr)

    funnel = run_funnel(df, up_to=up_to, holdout_freigabe=args.holdout)
    print(f"### {args.table.name}: {len(df)} Varianten")
    print(funnel.summary())

    for s in funnel.stages:
        if s.n_in and len(s.dropped) == 0 and not s.note:
            print(f"  HINWEIS {s.name}: cuttet nichts - entweder wirkungslos oder "
                  f"der Input ist schon gefiltert")
        elif s.note:
            print(f"  HINWEIS {s.name}: {s.note}")

    if not funnel.survivors.empty:
        cols = [c for c in ("name", "asset", "entry", "exit", "gross_mean",
                            "gross_bp", "cost_bp", "t_daily") if c in funnel.survivors]
        print("\n### Ueberlebende")
        print(funnel.survivors[cols].head(25).round(4).to_string(index=False))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({
            "input": str(args.table),
            "up_to": args.up_to,
            "holdout_freigabe": args.holdout,
            "stages": [{"name": s.name, "kept": len(s.kept), "dropped": len(s.dropped),
                        "note": s.note} for s in funnel.stages],
            "survivors": funnel.survivors.to_dict(orient="records"),
        }, indent=1, default=str))
        print(f"\nErgebnis -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
