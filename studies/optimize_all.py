#!/usr/bin/env python3
"""Run every strategy through optimiser + screening funnel, offline, from the m1 cache.

Process per (strategy, instrument):

  1. search on TRAIN (2019 .. holdout start) with ``trading_lab.optimizer``:
     objective = lower confidence bound of the net expectancy per trade,
     every distinct setting counted as a trial;
  2. shortlist -> grid neighbours, replay placebo, years, daily t, Deflated Sharpe,
     PBO -> one funnel row per variant;
  3. ``screening.run_funnel`` S1 -> S4 over ALL rows of all strategies;
  4. the holdout (S5) only with ``--holdout <release>``, and only for S4 survivors.

Data: only what is already in ``artifacts/cache/m1`` - this runner never downloads
(``load_cached``). Results: ``artifacts/optimize/``.

  .venv/bin/python studies/optimize_all.py --list
  .venv/bin/python studies/optimize_all.py --only momentum:US100 --budget 40
  .venv/bin/python studies/optimize_all.py --jobs 4 --holdout 2026-10-26
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_lab import screening  # noqa: E402
from trading_lab.optimizer import grid_size, optimise  # noqa: E402
from trading_lab.registry import (BUDGET_SCALE, HOLDOUT_START, NOT_ADAPTED,  # noqa: E402,F401
                                  REGISTRY, SHORTLIST, load_cached)

OUT = Path("artifacts/optimize")


def _job(name: str, asset: str, budget: int, holdout: str | None) -> dict:
    warnings.filterwarnings("ignore")
    t0 = time.time()
    m1 = load_cached(asset)
    if m1.empty:
        return {"name": name, "asset": asset, "error": "keine Daten im Cache"}
    spec = REGISTRY[name][0](asset)
    from trading_lab import strategy_spec
    sc = strategy_spec.load(name)
    spec_status = "abgestimmt" if sc.agreed else strategy_spec.UNCHECKED
    print(f"  start {spec.label}  (Grid {grid_size(spec.grid)}, Budget {budget})"
          + ("" if sc.agreed else f"  [{spec_status}: keine abgestimmte docs/specs/{name}.yaml]"), flush=True)
    try:
        top_net, top_gross = SHORTLIST.get(name, (6, 3))
        res = optimise(spec, m1, holdout_start=HOLDOUT_START,
                       budget=max(20, int(budget * BUDGET_SCALE.get(name, 1.0))),
                       top_net=top_net, top_gross=top_gross,
                       holdout_release=holdout, verbose=True)
    except Exception as exc:
        return {"name": name, "asset": asset, "error": repr(exc)}
    for r in res.rows:
        r["spezifikation"] = spec_status
    payload = {"name": name, "asset": asset, "n_trials": res.n_trials, "pbo": res.pbo,
               "spezifikation": spec_status,
               "grid_size": grid_size(spec.grid), "seconds": round(time.time() - t0, 1),
               "note": res.note, "rows": res.rows, "trials": res.trial_log()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}_{asset}.json").write_text(json.dumps(payload, indent=1, default=str))
    print(f"  fertig {spec.label}: {res.n_trials} Trials, PBO {res.pbo:.2f}, "
          f"{time.time() - t0:.0f}s", flush=True)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="z.B. momentum:US100,orb_2r (Komma-getrennt)")
    ap.add_argument("--budget", type=int, default=150, help="Trials der Suche je Strategie x Instrument")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--holdout", default=None, metavar="FREIGABE",
                    help="gibt S5 frei (nur fuer S4-Ueberlebende). Ohne: Ende bei S4.")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="nur vorhandene JSONs auswerten")
    args = ap.parse_args()

    jobs = [(n, a) for n, (_, assets) in REGISTRY.items() for a in assets]
    if args.only:
        want = [w.strip() for w in args.only.split(",") if w.strip()]
        jobs = [(n, a) for n, a in jobs if any(w == n or w == f"{n}:{a}" for w in want)]
    if args.list:
        for n, a in jobs:
            print(f"{n}:{a}")
        for n, why in NOT_ADAPTED.items():
            print(f"(nicht adaptiert) {n}: {why}")
        return 0

    if not args.report_only:
        print(f"### {len(jobs)} Laeufe, Budget {args.budget}, {args.jobs} parallel, "
              f"Train < {HOLDOUT_START.date()}, Holdout {'FREI: ' + args.holdout if args.holdout else 'gesperrt'}")
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(_job, n, a, args.budget, args.holdout) for n, a in jobs]
            for f in as_completed(futs):
                r = f.result()
                if r.get("error"):
                    print(f"  FEHLER {r['name']}:{r['asset']}: {r['error']}", flush=True)
                    (OUT / f"{r['name']}_{r['asset']}.error.txt").write_text(r["error"])
    return report(holdout=args.holdout)


def report(holdout: str | None) -> int:
    rows = []
    runs = []
    for p in sorted(OUT.glob("*.json")):
        if p.name in ("variants.json", "funnel.json"):
            continue
        d = json.loads(p.read_text())
        runs.append({k: d.get(k) for k in ("name", "asset", "n_trials", "pbo", "grid_size", "seconds", "note")})
        rows += d.get("rows", [])
    if not rows:
        print("keine Ergebnisse")
        return 1
    df = pd.DataFrame(rows)
    (OUT / "variants.json").write_text(json.dumps(rows, indent=1, default=str))
    funnel = screening.run_funnel(df, up_to=None if holdout else "S4", holdout_freigabe=holdout)
    print("\n" + funnel.summary())
    stages = {s.name.split()[0]: s for s in funnel.stages}
    best = df.sort_values("objective_lcb", ascending=False)
    cols = ["name", "n", "gross_mean", "net_mean", "placebo_edge", "pos_years", "n_years",
            "neighbour_positive", "t_daily", "dsr", "pbo"]
    print("\n### Beste Variante je Strategie x Instrument (Train, netto)")
    top = best.groupby(["strategy", "asset"], sort=False).head(1)
    print(top[cols].round(3).to_string(index=False))
    out = {"runs": runs, "stages": [{"name": s.name, "kept": len(s.kept), "dropped": len(s.dropped),
                                      "skipped": len(s.skipped), "note": s.note,
                                      "kept_names": list(s.kept.get("name", []))}
                                     for s in funnel.stages],
           "not_adapted": NOT_ADAPTED}
    (OUT / "funnel.json").write_text(json.dumps(out, indent=1, default=str))
    for tag in ("S2", "S3", "S4", "S5"):
        if tag in stages and len(stages[tag].kept):
            print(f"\n### nach {tag}: {len(stages[tag].kept)}")
            print(stages[tag].kept[cols].round(3).head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
