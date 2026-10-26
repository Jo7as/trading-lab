#!/usr/bin/env python3
"""Build and freeze the named portfolios from an optimiser run - reproducibly.

Each component is written with
  * ``params``       - the optimised parameters,
  * ``voll_config``  - EVERY field of the strategy config (defaults included), so a
                       later change of a default cannot change the strategy unnoticed,
  * ``regeln``       - entry / stop / target / exit in words, with the real numbers
                       (``trading_lab/rules.py``),
  * ``herkunft``     - which optimiser row it came from and how it scored.
The lock (``<ID>.lock``) holds the SHA-256 of the YAML and of every source file the
result depends on (``trading_lab.portfolio.code_hashes``).

Re-freezing is only allowed with ``--neu`` and only while the portfolio's holdout has
provably not been opened (no ``portfolio:<ID>`` entry in the holdout ledger).

  uv run python studies/portfolio_build.py            # build + freeze (refuses if locked)
  uv run python studies/portfolio_build.py --neu      # rebuild before the holdout was opened
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict, replace

sys.path.insert(0, ".")
sys.path.insert(0, "studies")

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import optimize_report as rep  # noqa: E402
from trading_lab import portfolio as P  # noqa: E402
from trading_lab.guided import LEDGER  # noqa: E402
from trading_lab.registry import REGISTRY  # noqa: E402
from trading_lab.rules import describe  # noqa: E402

#: the selection rule for TK8 - written down, not hand-picked
ROBUST = dict(stage="S3", min_placebo_edge=0.05, max_pbo=0.25)
TARGET_DD = 8.0          # risk per trade is set so the train max drawdown is this


def _clean(v):
    return v.item() if hasattr(v, "item") else v


def select(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["reached"] = rep.stage_reached(rows)
    ok = rows[(rows["reached"] == ROBUST["stage"]) & (rows["placebo_edge"] > ROBUST["min_placebo_edge"])
              & (rows["pbo"] <= ROBUST["max_pbo"])]
    return ok.sort_values("dsr", ascending=False).groupby(["strategy", "asset"]).head(1)


def component(r: pd.Series) -> dict:
    spec = REGISTRY[r["strategy"]][0](r["asset"])
    cfg = replace(spec.base, **r["params"])
    return {
        "strategy": r["strategy"], "asset": r["asset"],
        "params": {k: _clean(v) for k, v in r["params"].items()},
        "regeln": describe(r["strategy"], cfg),
        "voll_config": {k: _clean(v) for k, v in asdict(cfg).items()},
        "kosten_modell": {"spread_von_engine": spec.charges_spread,
                          "einstieg_versatz_min": spec.entry_offset_min},
        "herkunft": {"variante": r["name"], "train_n": int(r["n"]),
                     "train_netto_R": round(float(r["net_mean"]), 3), "dsr": round(float(r["dsr"]), 3),
                     "pbo": round(float(r["pbo"]), 3), "placebo_vorsprung": round(float(r["placebo_edge"]), 3)},
    }


def build(pid: str, name: str, rolle: str, desc: str, comps: list[dict], *, neu: bool) -> dict:
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    if f"portfolio:{pid}" in ledger:
        raise SystemExit(f"{pid}: Holdout wurde am {ledger[f'portfolio:{pid}']['datum']} geöffnet - "
                         "nicht mehr veränderbar.")
    lock = P.PF_DIR / f"{pid}.lock"
    if lock.exists():
        if not neu:
            raise SystemExit(f"{pid} ist eingefroren; --neu nur, solange der Holdout ungeöffnet ist.")
        lock.unlink()
    d = {"id": pid, "name": name, "rolle": rolle, "beschreibung": desc,
         "erstellt": pd.Timestamp.now().strftime("%Y-%m-%d"),
         "quelle": "studies/optimize_all.py (Budget 150, Train 2019-01-01..2024-12-31), "
                   "Auswahl: studies/portfolio_build.py",
         "auswahlregel": ROBUST, "risiko_je_trade_pct": None,
         "holdout_kriterien": {"t_min": 1.645, "max_dd_pct": 10.0, "max_tag_pct": 5.0,
                               "erklaerung": "Vorab festgelegt. Bestanden, wenn im Holdout (ab 2025-01-01): "
                                             "Tages-t >= 1,645 (einseitig 5 %, eine Hypothese, keine Deflation), "
                                             "Monatsmittel > 0, max. Drawdown <= 10 %, schlechtester Tag <= 5 % "
                                             "- alles beim festen Risiko."},
         "komponenten": comps}
    path = P.PF_DIR / f"{pid}.yaml"
    d["risiko_je_trade_pct"] = 1.0
    path.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False))
    dd_at_1 = P.evaluate(P.load(pid), window="train").stats()["max_drawdown_pct"]
    d["risiko_je_trade_pct"] = round(TARGET_DD / dd_at_1, 2)
    d["risiko_begruendung"] = f"so gewählt, dass der maximale Drawdown auf Train ~{TARGET_DD:.0f} % beträgt"
    path.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False))
    st = P.evaluate(P.load(pid), window="train").stats()
    d["train_ergebnis"] = {k: st[k] for k in ("von", "bis", "monate", "pct_pro_monat_mittel",
                                              "pct_pro_monat_median", "schlechtester_monat_pct",
                                              "positive_monate", "max_drawdown_pct",
                                              "schlechtester_tag_pct", "t_taeglich")}
    d["train_ergebnis"]["hinweis"] = "in-sample (auf diesen Daten optimiert) - optimistisch, kein Beweis"
    path.write_text(f"# Portfolio {pid} - {name}. EINGEFROREN: nicht ändern (siehe {pid}.lock).\n"
                    + yaml.safe_dump(d, allow_unicode=True, sort_keys=False))
    P.freeze(pid)
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neu", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    rows = pd.DataFrame(json.loads((rep.OUT / "variants.json").read_text()))
    best = select(rows)
    comps = [component(r) for _, r in best.iterrows()]
    us = [c for c in comps if c["asset"] == "US100"]
    m1 = [c for c in comps if c["strategy"] == "momentum" and c["asset"] == "US100"]
    sets = [(f"TK{len(comps)}", f"Trendkorb {len(comps)}", "primaer",
             f"Alle {len(comps)} Varianten, die S1-S3 bestanden haben, beim Placebo klar vorne liegen "
             f"(> +{ROBUST['min_placebo_edge']} R) und PBO <= {ROBUST['max_pbo']} haben. Der Haupttest.", comps),
            (f"US{len(us)}", {1: "US100-Solo", 2: "US100-Duo", 3: "US100-Trio"}.get(len(us), f"US100-{len(us)}"),
             "sekundaer", f"Alle Komponenten auf US100. Nur zur Information - bestätigend ist allein TK{len(comps)}.", us),
            ("M1", "Momentum solo", "sekundaer",
             f"Nur Momentum US100. Nur zur Information - bestätigend ist allein TK{len(comps)}.", m1)]
    for pid, name, rolle, desc, cs in sets:
        if not cs:
            print(f"{pid}: keine Komponenten - übersprungen")
            continue
        d = build(pid, name, rolle, desc, cs, neu=args.neu)
        t = d["train_ergebnis"]
        print(f"{pid} {name}: {len(cs)} Komponenten, Risiko {d['risiko_je_trade_pct']} %/Trade, "
              f"Train {t['pct_pro_monat_mittel']:.2f} %/Monat (Median {t['pct_pro_monat_median']:.2f}), "
              f"max DD {t['max_drawdown_pct']:.1f} %  -> eingefroren")
        for c in cs:
            h = c["herkunft"]
            print(f"    {c['strategy']}:{c['asset']:7s} n={h['train_n']:4d} netto {h['train_netto_R']:+.3f} R "
                  f"DSR {h['dsr']:.3f} PBO {h['pbo']:.2f} Placebo {h['placebo_vorsprung']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
