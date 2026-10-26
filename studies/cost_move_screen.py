#!/usr/bin/env python3
"""S0 sweep: is there *any* market x horizon where a move can pay its spread?

`trading_lab.cost_gate.horizon_gate` answers the question for one pair. This runs
it across every instrument with cached 1-minute bid+ask data and four horizons
(15 min, 1 h, 4 h, 1 day), and prints the ratio `E|move| / round-trip cost`.

It is the *necessary* condition, not a strategy: no directional signal can
capture more than the average absolute move, so a ratio below ~2 means every
idea built on that pair/horizon is dead on arrival. The gate exists in
``AGENTS.md`` but had only been applied pointwise - this is the map.

  .venv/bin/python studies/cost_move_screen.py
  .venv/bin/python studies/cost_move_screen.py --start-year 2019 --horizons 15,60,240
"""
from __future__ import annotations

import argparse
import sys
import warnings

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_lab.cost_gate import DEFAULT_K, horizon_gate  # noqa: E402
from trading_lab.hypothesis_card import S0_GRENZE_DEFAULT  # noqa: E402
from trading_lab.minute_data import (  # noqa: E402
    DEFAULT_CACHE,
    INSTRUMENTS,
    load_minute_bid_ask,
)

HORIZONS = (15, 60, 240, 1440)

# Two thresholds, and they are not the same thing:
#   cost_gate.DEFAULT_K = 3  -> the *necessary* condition (a move can pay the spread)
#   hypothesis_card gate     -> S0-Quote = cost/move <= 0.10, i.e. ratio >= 10.
# The card gate is the one a hypothesis must pass before it is tested at all.
CARD_K = 1.0 / S0_GRENZE_DEFAULT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-year", type=int, default=2019)
    ap.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS))
    ap.add_argument("--assets", default=",".join(sorted(INSTRUMENTS)))
    ap.add_argument("--csv", default="", help="write the matrix to this CSV")
    ap.add_argument("--allow-download", action="store_true",
                    help="also screen instruments missing from the cache (hits the network)")
    ap.add_argument("--k", type=float, default=CARD_K,
                    help=f"PASS threshold for move/cost (default {CARD_K:g} = the card's S0 gate)")
    args = ap.parse_args()
    horizons = [int(h) for h in args.horizons.split(",")]
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]

    warnings.filterwarnings("ignore")
    table: dict[str, dict[int, float]] = {}
    detail: list[dict] = []
    cached = {f.name.split("_1m_")[0] for f in DEFAULT_CACHE.glob("*_1m_bid_*.pkl")}
    for asset in assets:
        if not args.allow_download and asset not in cached:
            print(f"  {asset:8s} nicht im Cache - uebersprungen (--allow-download)", flush=True)
            continue
        try:
            m1 = load_minute_bid_ask(asset, args.start_year, verbose=False)
        except Exception as exc:                       # noqa: BLE001 - keep going
            print(f"  {asset:8s} keine Daten ({type(exc).__name__})", flush=True)
            continue
        if m1 is None or len(m1) < 5000:
            continue
        row: dict[int, float] = {}
        for h in horizons:
            g = horizon_gate(m1, h)
            row[h] = g.ratio_taker
            detail.append({"asset": asset, "minutes": h, "ratio_taker": g.ratio_taker,
                           "ratio_limit": g.ratio_limit, "move_pct": g.move_pct,
                           "cost_taker_pct": g.cost_taker_pct, "verdict": g.verdict()})
        table[asset] = row
        print(f"  {asset:8s} {len(m1):>9,d} bars  {m1.index[0].date()}..{m1.index[-1].date()}",
              flush=True)

    df = pd.DataFrame(table).T
    df.index.name = "asset"
    df.columns = [f"{h}m" if h < 1440 else "1d" for h in horizons]

    print("\n### S0: E|move| / round-trip spread  (taker, i.e. full spread both legs)")
    print(f"    PASS >= {args.k:g}  (= S0-Quote <= {1/args.k:.2f})   "
          f"MARG >= {DEFAULT_K:g} (notwendig)   darunter FAIL\n")
    print(f"    {'asset':8s}" + "".join(f"{c:>18s}" for c in df.columns))
    print(f"    {'':8s}" + "".join(f"{'ratio   q_min':>18s}" for _ in df.columns))
    for asset, r in df.iterrows():
        cells = ""
        for v in r:
            if not np.isfinite(v):
                cells += f"{'-':>18s}"; continue
            cells += f"{v:8.2f} {100/v:6.1f}%" + ("  DEAD" if v < 1 else "      ")
        print(f"    {asset:8s}{cells}")
    dead = df.index[(df < 1).any(axis=1)]
    print(f"\n    Wirklich tot (ratio < 1 irgendwo, selbst ein perfektes Signal verliert): "
          f"{', '.join(dead) if len(dead) else 'keiner'}")
    print("\n    Noetige Capture Ratio q_min = 1/ratio, je Horizont (Median / Bereich):")
    for c in df.columns:
        q = (1 / df[c]).dropna()
        print(f"      {c:>5s}  Median {q.median()*100:5.1f}%   {q.min()*100:5.1f}% .. {q.max()*100:5.1f}%")
    print("    -> q_min ist eine HUERDE, keine Todesgrenze: ein Signal muss diesen Anteil der")
    print("       mittleren Bewegung einfangen. Nur q_min > 100 % ist unmoeglich.")
    print("\n    Fuer ein reines Richtungssignal ist q_min = 2p - 1, also die noetige")
    print("    Trefferquote p_min = (1 + 1/ratio) / 2:")
    for c in df.columns:
        pmin = ((1 + 1 / df[c]) / 2).dropna()
        pmin = pmin[pmin <= 1]
        if len(pmin):
            print(f"      {c:>5s}  Median {pmin.median()*100:5.1f}%   "
                  f"{pmin.min()*100:5.1f}% .. {pmin.max()*100:5.1f}%   (n={len(pmin)})")
        else:
            print(f"      {c:>5s}  keine - selbst 100 % Trefferquote reicht nicht")
    print("    Zum Vergleich: die hier gemessenen Strategien lagen bei 46-48 % Trefferquote.")

    print(f"\n    Bestes Verhaeltnis je Markt (ueber die Horizonte):")
    best = df.max(axis=1).sort_values(ascending=False)
    for asset, v in best.items():
        print(f"      {asset:8s} {v:6.2f}  (bei {df.loc[asset].idxmax()})")
    strict = df.max(axis=1) >= args.k
    print(f"\n    Karten-Gate (q_min <= {1/args.k*100:.0f}% irgendwo): {int(strict.sum())} von "
          f"{len(best)} -> {', '.join(sorted(df.index[strict])) or '(keiner)'}")
    loose = df.max(axis=1) >= DEFAULT_K
    print(f"    nur notwendige Bedingung (ratio >= {DEFAULT_K:g}): {int(loose.sum())} von {len(best)}")

    if args.csv:
        pd.DataFrame(detail).to_csv(args.csv, index=False)
        print(f"    Detail -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
