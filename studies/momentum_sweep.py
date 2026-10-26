#!/usr/bin/env python3
"""Stage 1 (exploration): sweep the momentum space BROAD and judge it on gross.

Process, in this order - the order matters and the reverse is what killed the first
nine candidates:

  1. build the strategy with every choice as a parameter,
  2. sweep **all** variants and look at **gross** results (no cost filter, no gate),
  3. only the variants that survive this get cost, walk-forward and years applied.

A cost gate applied *before* the sweep is a filter on an assumption: it says "the
move is probably too small" without ever seeing whether the strategy actually
captures more than a trade's worth of it. That is backwards. Costs belong in
stage 2, as a hurdle on results that exist, not as a prediction that stops them
from existing.

The only things kept in stage 1 are correctness constraints:

* no look-ahead (signals shifted, intrabar stop/target order respected),
* one trade per session day,
* a fair-coin placebo in the same row: with ~2,000 variants, a positive gross
  number means nothing unless it beats a random direction through the very same
  exit machinery. That is the one "gate" worth having here - it separates
  "the exit model has a positive skew" from "the signal chose a side".

Judging criteria for stage 2 shortlist: gross R per trade, share of years with
positive gross, and the margin over the placebo.

  .venv/bin/python studies/momentum_sweep.py
  .venv/bin/python studies/momentum_sweep.py --assets US100,GER40 --max-variants 400
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_lab.minute_data import DEFAULT_CACHE, load_minute_bid_ask  # noqa: E402
from trading_lab.strategies.momentum import MomentumConfig, run_backtest  # noqa: E402

GRID = {
    "entry": ("breakout", "pullback", "momentum", "market"),
    "stop": ("atr", "swing"),
    "exit": ("target", "trail_atr", "be_trail", "time"),
    "regime": ("none", "er", "adx"),
    "session": ("ny", "london"),
    "target_r": (1.0, 2.0, 3.0),
    "stop_atr": (1.0, 1.5, 2.5),
    "max_hold_minutes": (120, 240, 480),
}


def years_of(trades: list[dict]) -> dict[int, float]:
    out: dict[int, float] = {}
    for t in trades:
        y = pd.Timestamp(t["entry_time"]).year
        out[y] = out.get(y, 0.0) + t["r_gross"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", default="US100,GER40,XAUUSD,EURUSD")
    ap.add_argument("--max-variants", type=int, default=0,
                    help="0 = all combinations")
    ap.add_argument("--start-year", type=int, default=2019)
    ap.add_argument("--json", default="artifacts/momentum_sweep.json")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    cached = {f.name.split("_1m_")[0] for f in DEFAULT_CACHE.glob("*_1m_bid_*.pkl")}

    keys = list(GRID)
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    if args.max_variants and args.max_variants < len(combos):
        # NICHT combos[:n]: itertools.product zaehlt die letzte Achse am schnellsten,
        # also bestuenden die "ersten 20" nur aus breakout/atr/target - eine
        # Stichprobe, die genau eine Entry-Art testet und den Rest nie sieht.
        # Stattdessen eine deterministisch gezogene, ueber den ganzen Raum verteilte
        # Auswahl.
        rng = np.random.default_rng(7)
        pick = rng.choice(len(combos), args.max_variants, replace=False)
        combos = [combos[i] for i in sorted(pick)]
    print(f"### Stufe 1: breiter Sweep, BRUTTO beurteilt ({len(combos)} Varianten "
          f"x {len(assets)} Instrumente = {len(combos)*len(assets)})")
    print("    Kein Kostenfilter. Placebo (Zufallsrichtung) laeuft in jeder Zeile mit.\n")

    rows: list[dict] = []
    t0 = time.time()
    total = len(assets) * (len(combos) + 48)
    done = 0
    for asset in assets:
        if asset not in cached:
            continue
        m1 = load_minute_bid_ask(asset, args.start_year, verbose=False)
        # one placebo per (exit, stop, hold) - the signal is irrelevant for it
        placebo_cfg: dict[tuple, float] = {}
        for ex, st, hold in itertools.product(GRID["exit"], GRID["stop"],
                                              GRID["max_hold_minutes"]):
            for sess in GRID["session"]:
                key = (ex, st, hold, sess)
                if key in placebo_cfg:
                    continue
                cfg = MomentumConfig(asset=asset, entry="random", stop=st, exit=ex,
                                     session=sess, max_hold_minutes=hold, max_cost_r=1.0)
                tr = run_backtest(m1, cfg)["trades"]
                placebo_cfg[key] = float(np.mean([t["r_gross"] for t in tr])) if tr else np.nan
                done += 1
        print(f"  {asset}: Placebos fertig ({done}/{total} Laeufe, {time.time()-t0:.0f}s)",
              flush=True)
        for i, combo in enumerate(combos):
            kw = dict(zip(keys, combo))
            cfg = MomentumConfig(asset=asset, max_cost_r=1.0,
                                 spread_filter=0.0, **kw)
            tr = run_backtest(m1, cfg)["trades"]
            done += 1
            if i % 20 == 0:
                eta = (time.time() - t0) / max(done, 1) * (total - done)
                print(f"    {asset} {i:4d}/{len(combos)}  ({done}/{total}, "
                      f"{time.time()-t0:.0f}s, ETA {eta/60:.1f} min)", flush=True)
            if len(tr) < 100:
                continue
            g = np.array([t["r_gross"] for t in tr])
            yrs = years_of(tr)
            pos_years = sum(1 for v in yrs.values() if v > 0)
            plc = placebo_cfg.get((kw["exit"], kw["stop"], kw["max_hold_minutes"],
                                   kw["session"]), np.nan)
            # bp of the price: the unit S2 needs. Taken from the trade's own mid, not
            # from an average, so a variant whose entries cluster in a high-price
            # regime is not silently flattered.
            gross_bp = float(np.mean([t["gross_abs"] / t["mid_entry"] * 1e4 for t in tr]))
            cost_bp = float(np.mean([t["cost_abs"] / t["mid_entry"] * 1e4 for t in tr]))
            rows.append({"asset": asset, **kw, "n": len(tr),
                         "gross_mean": float(g.mean()), "gross_sum": float(g.sum()),
                         "gross_bp": gross_bp, "cost_bp": cost_bp,
                         "sd": float(g.std(ddof=1)),
                         "t": float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))),
                         "pos_years": pos_years, "n_years": len(yrs),
                         "placebo_gross": plc, "edge": float(g.mean()) - (plc if np.isfinite(plc) else 0.0)})
        print(f"  {asset} fertig  ({time.time()-t0:.0f}s, {len(rows)} Zeilen)", flush=True)

    df = pd.DataFrame(rows)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(args.json, orient="records")
    print(f"\n### {len(df)} Varianten mit >= 100 Trades")
    print("\n### Top 20 nach Vorsprung gegen den Placebo (brutto)")
    cols = ["asset", "entry", "stop", "exit", "regime", "session", "target_r",
            "stop_atr", "max_hold_minutes", "n", "gross_mean", "placebo_gross",
            "edge", "t", "pos_years", "n_years"]
    top = df.sort_values("edge", ascending=False).head(20)
    print(top[cols].round(4).to_string(index=False))
    print("\n### Top 10 nach brutto Ertrag pro Trade (ohne Placebo-Vergleich)")
    print(df.sort_values("gross_mean", ascending=False).head(10)[cols].round(4).to_string(index=False))
    print("\n### Wie viele Varianten schlagen ihren Placebo?")
    n_pos = int((df["edge"] > 0).sum())
    print(f"    {n_pos} von {len(df)} ({100*n_pos/max(len(df),1):.0f}%)  "
          f"(bei reinem Rauschen waeren es ~50%)")
    print("    Erwartung: wenn die Exit-Maschinerie selbst positiv ist, schlagen")
    print("    fast alle ihren Placebo - dann ist der Placebo falsch kalibriert.")
    # Verteilung der Placebo-Werte zur Kontrolle
    pb = df["placebo_gross"].dropna()
    print(f"\n    Placebo brutto: Median {pb.median():+.4f}, p5 {pb.quantile(0.05):+.4f}, "
          f"p95 {pb.quantile(0.95):+.4f}")
    print(f"    (muss um 0 streuen; ein Median deutlich != 0 heisst: die Exit-Maschine"
          f" selbst hat eine Verzerrung)")
    print(f"\n    Detail -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
