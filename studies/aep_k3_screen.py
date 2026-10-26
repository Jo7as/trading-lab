#!/usr/bin/env python3
"""AEP Stage 0: can the start library pass its own K3 cost gate?

The Adaptive-Edge-Portfolio plan (``Neue Idee pläne/``) promises a start library
of ~22 candidate units and lists K3 as a **mandatory** admission criterion:

    K3: Kosten / typische Bewegung über den Zielhorizont <= 0.10, für das Cluster

That needs no backtest - only the bid/ask data and the holding time each
candidate's rules imply. This checks it for every unit in the plan's tables 10,
Anhang A and Teil XI, and cross-references which candidates already exist as
registered, *killed* cards in this repo (re-testing those would be a re-test of a
refuted hypothesis under a new name).

Horizons are read off the rules in Anhang A (time stop, or session end); where a
candidate holds until a session close the range of plausible entry times is
covered by two values.

  .venv/bin/python studies/aep_k3_screen.py
  .venv/bin/python studies/aep_k3_screen.py --csv artifacts/aep_k3.csv
"""
from __future__ import annotations

import argparse
import sys
import warnings

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_lab.cost_gate import horizon_gate  # noqa: E402
from trading_lab.hypothesis_card import S0_GRENZE_DEFAULT  # noqa: E402
from trading_lab.minute_data import DEFAULT_CACHE, load_minute_bid_ask  # noqa: E402

# The plan's clusters (Abschnitt 10).
CLUSTERS: dict[str, tuple[str, ...]] = {
    "USD-Majors": ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"),
    "JPY-Paare": ("USDJPY", "EURJPY", "GBPJPY", "AUDJPY"),
    "EUR-Crosses": ("EURGBP", "EURCHF"),
    "AUD/NZD/CAD-Crosses": ("AUDNZD", "AUDCAD"),
    "EUR/GBP-Paare": ("EURUSD", "GBPUSD", "EURGBP"),
    "GBP/EUR-Paare": ("GBPUSD", "EURUSD"),
    "Alle FX": ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD"),
    "Gold": ("XAUUSD",),
    "US-Indizes": ("SP500", "US100", "US30"),
    "Europa-Indizes": ("GER40", "UK100"),
    "Krypto": ("BTCUSD",),
}

# (candidate, cluster, horizons in minutes, already-refuted repo card or "")
UNITS: list[tuple[str, str, tuple[int, ...], str]] = [
    ("K-01 F-Asien", "EUR-Crosses", (60, 120), ""),
    ("K-01 F-Asien", "AUD/NZD/CAD-Crosses", (60, 120), ""),
    ("K-02 F-Tag", "EUR/GBP-Paare", (60, 120), "verwandt mit verworfener M3 Mean-Reversion"),
    ("K-02 F-Tag", "USD-Majors", (60, 120), "verwandt mit verworfener M3 Mean-Reversion"),
    ("K-03 Marken-Fade", "USD-Majors", (90, 180), ""),
    ("K-03 Marken-Fade", "JPY-Paare", (90, 180), ""),
    ("K-03 Marken-Fade", "Gold", (90, 180), ""),
    ("K-04 Paar-Spread", "AUD/NZD/CAD-Crosses", (240, 240), "Pair-Spreads getestet: gross PF ~ 1.0"),
    ("K-05 Marken-Durchbruch", "USD-Majors", (120, 240), ""),
    ("K-05 Marken-Durchbruch", "JPY-Paare", (120, 240), ""),
    ("K-05 Marken-Durchbruch", "Gold", (120, 240), ""),
    ("K-06 London-Breakout", "GBP/EUR-Paare", (240, 420), "H-004 kill (Asian-Range-Breakout)"),
    ("K-06 London-Breakout", "JPY-Paare", (240, 420), "H-004 kill (Asian-Range-Breakout)"),
    ("K-07 Noise-Area", "USD-Majors", (390, 600), "H-001 kill"),
    ("K-08 Gold-COMEX", "Gold", (240, 390), ""),
    ("K-09 Kompression", "USD-Majors", (180, 300), ""),
    ("K-09 Kompression", "JPY-Paare", (180, 300), ""),
    ("K-10 Waehrungsstaerke", "Alle FX", (60, 120), "verwandt mit H-006 kill"),
    ("K-11a Fixing London", "USD-Majors", (60, 120), "H-005 kill (London-16:00-Fix)"),
    ("K-11b Fixing Tokio", "JPY-Paare", (60, 120), "verwandt mit H-005 kill"),
    ("K-12 Monatsende", "USD-Majors", (360, 360), "verwandt mit H-005 kill"),
    ("K-15 Index-Regime", "US-Indizes", (60, 120), ""),
    ("K-16 Schlussstunde", "US-Indizes", (60, 60), "H-003 kill"),
    ("K-16 Schlussstunde", "Europa-Indizes", (60, 60), "H-003 kill"),
    ("K-17 Bitcoin", "Krypto", (90, 120), ""),
    ("K-18 Prop-Flow", "USD-Majors", (115, 115), "eigene Hypothese ohne Studie"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-year", type=int, default=2019)
    ap.add_argument("--csv", default="")
    ap.add_argument("--cost-model", choices=("market", "limit"), default="market",
                    help="market = both legs cross the spread; limit = one leg is a "
                         "resting order and pays no spread (halves the cost)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    cached = {f.name.split("_1m_")[0] for f in DEFAULT_CACHE.glob("*_1m_bid_*.pkl")}
    need = sorted({i for c in CLUSTERS.values() for i in c})
    frames: dict[str, pd.DataFrame] = {}
    for inst in need:
        if inst not in cached:
            continue
        try:
            m1 = load_minute_bid_ask(inst, args.start_year, verbose=False)
            if m1 is not None and len(m1) > 5000:
                frames[inst] = m1
        except Exception:                                   # noqa: BLE001
            pass
    missing = sorted(set(need) - set(frames))
    print(f"### AEP Stufe 0 - K3-Vorabpruefung  (Grenze: Kosten/Bewegung <= "
          f"{S0_GRENZE_DEFAULT}, jedes Bein)")
    print(f"    Kostenmodell: {args.cost_model}"
          + ("  (Limit-Einstieg, ein Bein zahlt keinen Spread)" if args.cost_model == "limit"
             else "  (Market-Einstieg, beide Beine zahlen den Spread)") + "\n")
    print(f"    {len(frames)} von {len(need)} Instrumenten im Cache"
          + (f"; fehlen: {', '.join(missing)}" if missing else "") + "\n")

    rows: list[dict] = []
    for cand, cluster, horizons, refuted in UNITS:
        insts = [i for i in CLUSTERS[cluster] if i in frames]
        if not insts:
            continue
        # worst leg over all instruments and all plausible horizons
        worst_q, worst_inst, worst_min = -1.0, "", 0
        per_inst: dict[str, float] = {}
        for inst in insts:
            attr = "ratio_taker" if args.cost_model == "market" else "ratio_limit"
            q = max(1.0 / getattr(horizon_gate(frames[inst], m), attr) for m in horizons)
            per_inst[inst] = q
            if q > worst_q:
                worst_q, worst_inst, worst_min = q, inst, max(horizons)
        all_pass = all(q <= S0_GRENZE_DEFAULT for q in per_inst.values())
        any_pass = any(q <= S0_GRENZE_DEFAULT for q in per_inst.values())
        n_pass = sum(q <= S0_GRENZE_DEFAULT for q in per_inst.values())
        rows.append({
            "unit": f"{cand} x {cluster}", "candidate": cand, "cluster": cluster,
            "horizons": "/".join(str(h) for h in horizons), "n_legs": len(insts),
            "n_pass": n_pass, "worst_inst": worst_inst, "worst_q": worst_q,
            "all_pass": all_pass, "any_pass": any_pass, "refuted": refuted,
        })
        marks = " ".join(f"{i}:{q:.3f}{'' if q <= S0_GRENZE_DEFAULT else '!'}"
                         for i, q in per_inst.items())
        flag = "OK " if all_pass else ("teil " if any_pass else "FAIL")
        print(f"    {cand:22s} x {cluster:22s} {str(horizons):9s} {flag}  {marks}")

    df = pd.DataFrame(rows)
    if df.empty:
        return 1
    fresh = df[~df["refuted"].str.contains("kill|verworfen", na=False, regex=True)]
    print(f"\n    Einheiten: {len(df)}   bestehen K3 vollstaendig: {int(df['all_pass'].sum())}"
          f"   teilweise: {int((df['any_pass'] & ~df['all_pass']).sum())}"
          f"   gar nicht: {int((~df['any_pass']).sum())}")
    print(f"    davon ohne bereits widerlegte Karte: {len(fresh)} Einheiten, "
          f"{int(fresh['all_pass'].sum())} bestehen K3")
    print("\n    Einheiten, die K3 bestehen UND keine tote Karte haben:")
    good = fresh[fresh["all_pass"]]
    if good.empty:
        print("      keine - die Startbibliothek traegt nicht.")
    else:
        for _, r in good.iterrows():
            print(f"      {r.unit:48s} Horizonte {r.horizons:9s} "
                  f"engstes Bein {r.worst_inst} ({r.worst_q:.3f})")
    print("\n    Ohne Kostengate bestuenden diese Strategien nie (Einheiten mit 0 Beinen OK):")
    for _, r in df[df["n_pass"] == 0].iterrows():
        print(f"      {r.unit:48s} bestes Bein schon {r.worst_q:.3f}")
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\n    Detail -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
