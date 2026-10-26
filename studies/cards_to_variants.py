#!/usr/bin/env python3
"""Bring the registered hypothesis cards into the funnel's variants format.

The nine old cards were judged by hand under the *old* logic (a cost prediction
blocking registration, significance on the per-trade series, everything AND-ed).
Several of them may have died for lack of power rather than lack of edge, and the
only way to know is to put them through the same funnel as any new candidate.

What this can and cannot do today:

* **Can**: read everything the cards hold in structured form - the S0 block, the
  kill criteria, the variant list, the script reference added in step C1.
* **Cannot**: invent the numbers that only a re-run produces (``n``, ``pos_years``,
  the daily t). Those need each card's study to emit ``--json`` (step C2 in
  docs/PIPELINE_AUDIT.md). They are written as ``null`` on purpose: the funnel then
  **skips** the stage that needs them with a note, instead of dropping the row.

So the output is an honest inventory: which cards are re-runnable, which have their
evidence only as prose, and which are not reproducible at all.

  .venv/bin/python studies/cards_to_variants.py
  .venv/bin/python studies/cards_to_variants.py --out artifacts/cards_variants.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

CARDS = Path("docs/hypotheses")


def read_cards(pattern: str = "H-*.yaml") -> pd.DataFrame:
    """One row per card, with the structured fields the funnel understands."""
    rows: list[dict] = []
    for f in sorted(CARDS.glob(pattern)):
        d = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        s0 = d.get("s0") or {}
        text = str(d.get("ergebnis") or "")
        kill = d.get("kill_kriterien") or {}
        skript = str(d.get("skript") or "")
        # Trades / Jahre stehen nicht strukturiert in den Karten, aber wenn der
        # Ergebnistext sie nennt, ist das besser als ein blankes null.
        m_n = re.search(r"\b(\d{2,5})\s*Trades?\b", text)
        m_j = re.search(r"(\d)\s*(?:von|/)\s*(\d)\s*Jahre", text)
        rows.append({
            "name": d.get("id"),
            "card": f.name,
            "status": d.get("status"),
            "stufe": d.get("stufe"),
            "skript": skript,
            "reproduzierbar": bool(skript) and "UNGEKLAERT" not in skript,
            "n": int(m_n.group(1)) if m_n else None,
            "pos_years": int(m_j.group(1)) if m_j else None,
            "n_years": int(m_j.group(2)) if m_j else None,
            # Nur die Kosten aus S0. Die "typische Bewegung" ist eine PROGNOSE und
            # darf nicht als Brutto-Ergebnis in S2 landen - sonst entscheidet dort
            # wieder eine Erwartung statt eines Backtests (Review 2026-10-26).
            "cost_bp": s0.get("kosten_rt_bp"),
            "gross_bp": None,
            "s0_bewegung_prognose_bp": s0.get("typische_bewegung_bp"),
            "placebo_gross": None,
            "gross_mean": None,
            "t_daily": None,
            "placebo_edge": None,
            "pbo": None,
            "oos_pass": None,
            "kill_kriterien": ",".join(sorted(kill)) or None,
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", default="H-*.yaml")
    ap.add_argument("--out", default="artifacts/cards_variants.json")
    args = ap.parse_args()

    df = read_cards(args.pattern)
    if df.empty:
        print("keine Karten gefunden", file=sys.stderr)
        return 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(df.to_dict(orient="records"), indent=1))

    print(f"### {len(df)} Karten inventarisiert -> {args.out}\n")
    cols = ["name", "status", "stufe", "reproduzierbar", "n", "cost_bp", "gross_bp",
            "skript"]
    print(df[cols].to_string(index=False))
    n_rep = int(df["reproduzierbar"].sum())
    print(f"\n    reproduzierbar (Study bekannt): {n_rep} von {len(df)}")
    if n_rep < len(df):
        fehlt = df.loc[~df["reproduzierbar"], "name"].tolist()
        print(f"    offen: {', '.join(fehlt)}")
    print("\n    Was noch fehlt, damit der Trichter sie bewerten kann:")
    for c in ("n", "pos_years", "t_daily"):
        n_missing = int(df[c].isna().sum())
        if n_missing:
            print(f"      {c}: {n_missing} Karten ohne Wert -> Study braucht --json (C2)")
    print("\n    --json gibt es inzwischen fuer studies/run_h001.py (H-001) und")
    print("    studies/hypotheses_60m.py (H-002/H-003). Offen: multimarket_check,")
    print("    carry_final, carry_real_swaps (deren Logs sind schon netto).")
    print("    Naechster Schritt: diese Tabellen durch studies/run_screening.py schicken.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
