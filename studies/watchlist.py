#!/usr/bin/env python3
"""Paket 3: the watchlist over everything measured so far (TRAIN numbers only, no holdout is opened).

Families: every strategy x market of the optimiser run (``artifacts/optimize/variants.json``) and the
portfolio studies S-005, S-006, S-008, S-009. Best variant per family, corrected for the search inside the
family, Benjamini-Hochberg at 20 %. Confirmed basket blocks are left out; ``holdout`` marks what already
saw the 2025+ data. The list releases nothing - see ``docs/PRUEFABLAUF.md``. Writes ``docs/BEOBACHTUNG.md``.

  .venv/bin/python studies/watchlist.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd  # noqa: E402

from trading_lab import watchlist as W  # noqa: E402

SEEN = {"momentum:US100": "TK6, nicht bestanden", "momentum:SP500": "TK6, nicht bestanden",
        "supertrend:USDJPY": "TK6, nicht bestanden", "supertrend:GER40": "TK6, nicht bestanden",
        "supertrend:SP500": "TK6, nicht bestanden", "orb_classic:US100": "TK6, nicht bestanden",
        "noise_area_momentum:SP500": "nicht bestanden", "noise_area_momentum:US100": "formal, t 0,41"}


def load_rows() -> pd.DataFrame:
    rows = []
    for r in json.loads(Path("artifacts/optimize/variants.json").read_text()):
        if r.get("t_daily") is None or r["t_daily"] != r["t_daily"]:
            continue
        trials = r.get("n_eff") or r.get("n_trials") or 1
        rows.append({"family": f"{r['strategy']}:{r['asset']}", "t": float(r["t_daily"]), "trials": float(trials),
                     "net_r": r.get("net_mean"), "dsr": r.get("dsr")})
    for name, path in (("pre_fomc:INDIZES (S-009)", "artifacts/pre_fomc/funnel.json"),
                       ("turn_of_month:INDIZES (S-008)", "artifacts/tom/funnel.json"),
                       ("weekly_level_break:POOL8 (S-005)", "artifacts/weekly_level/funnel.json")):
        data = json.loads(Path(path).read_text())
        for r in data if isinstance(data, list) else [data]:
            rows.append({"family": name, "t": float(r["t_daily"]), "trials": float(r.get("n_trials") or 1),
                         "net_r": r.get("net_mean"), "dsr": r.get("dsr")})
    return pd.DataFrame(rows)


def main() -> int:
    df = W.build(load_rows())
    df["holdout"] = [SEEN.get(f, "unberührt") for f in df["family"]]
    watch = df[df["beobachten"]]
    pd.set_option("display.width", 200)
    print(f"{len(df)} Familien, {len(watch)} auf der Beobachtungsliste (FDR 20 %)")
    print(df.head(15)[["family", "t", "trials", "p", "beobachten", "net_r", "dsr", "holdout"]].round(3).to_string(index=False))
    s9 = df[df["family"].str.contains("S-009")]
    print("\nS-009:", s9[["t", "trials", "p", "beobachten"]].round(3).to_dict("records"))
    lines = ["# Beobachtungsliste", "",
             "Automatisch aus `studies/watchlist.py` (nur Trainingszahlen). Eine Familie = Strategie x Markt, beste",
             "Variante korrigiert für die Suche in der Familie, Benjamini-Hochberg bei 20 % FDR.", "",
             "**Die Liste gibt nichts frei** - weder Handel noch Korb. Sie nennt nur, was eine Prüfung auf frischen",
             "Daten oder im Korb verdient.", "",
             f"{len(df)} Familien geprüft, {len(watch)} auf der Liste.", "",
             "| Familie | t | eff. Versuche | p (korrigiert) | DSR | Holdout |", "|---|---|---|---|---|---|"]
    for _, r in watch.iterrows():
        dsr = "-" if r["dsr"] is None or r["dsr"] != r["dsr"] else f"{r['dsr']:.2f}"
        lines.append(f"| {r['family']} | {r['t']:.2f} | {r['trials']:.0f} | {r['p']:.3f} | {dsr} | {r['holdout']} |")
    lines += ["", "## Am nächsten dran (nicht auf der Liste, nur zur Information)", "",
              "| Familie | t | eff. Versuche | p (korrigiert) | Holdout |", "|---|---|---|---|---|"]
    for _, r in df[~df["beobachten"]].head(6).iterrows():
        lines.append(f"| {r['family']} | {r['t']:.2f} | {r['trials']:.0f} | {r['p']:.3f} | {r['holdout']} |")
    Path("docs/BEOBACHTUNG.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
