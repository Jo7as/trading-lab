#!/usr/bin/env python3
"""Final report for ``studies/optimize_all.py``: ranking, near-misses, prop-firm test.

Reads ``artifacts/optimize/variants.json`` (written by the optimiser run) and

1. reports how far every strategy x instrument got in the funnel (S1..S5),
2. lists the S4 survivors and the **near-misses** (S1-S3 passed, DSR >= 0.80) -
   the near-misses are reported, never promoted,
3. re-runs the top candidates on TRAIN and puts their daily R stream through the
   prop-firm simulator (``trading_lab.prop_sim``: historical replay + block
   bootstrap) at the risk per trade that targets 1.5 %/month on train.
   The holdout is only used for candidates that already have ``oos_n`` (= released
   and passed S4 inside the optimiser run).

Writes ``artifacts/optimize/report.md``.

  .venv/bin/python studies/optimize_report.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "studies")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_lab.registry import HOLDOUT_START, REGISTRY, load_cached  # noqa: E402
from trading_lab import prop_sim, screening  # noqa: E402
from trading_lab.guided import GuidedCheck  # noqa: E402
from trading_lab.optimizer import normalise  # noqa: E402

OUT = Path("artifacts/optimize")
NEAR_DSR = 0.80
FIRMS = ("ftmo", "one_step", "instant_100k")


def stage_reached(df: pd.DataFrame) -> pd.Series:
    """Last funnel stage each row passed (S0 = failed S1)."""
    reached = pd.Series("S0", index=df.index)
    cur = df
    for tag, fn in (("S1", screening.s1_explore), ("S2", screening.s2_cost),
                    ("S3", screening.s3_robust), ("S4", screening.s4_significant)):
        s = fn(cur)
        keep = pd.concat([s.kept, s.skipped]) if len(s.skipped) else s.kept
        reached.loc[keep.index] = tag
        cur = keep
        if cur.empty:
            break
    if "oos_pass" in df:
        reached.loc[df.index[(reached == "S4") & (df["oos_pass"] == True)]] = "S5"  # noqa: E712
    return reached


STAGE_TEXT = {
    "S1": "Brutto > 0, besser als die Gegenrichtung (Placebo), mindestens 100 Trades. Keine Kosten.",
    "S2": "Jetzt erst die gemessenen Kosten (echte Bid/Ask-Fills): netto > 0 je Trade?",
    "S3": ">= 70 % der Jahre netto positiv und die Nachbar-Einstellungen mehrheitlich positiv (Plateau statt Spitze).",
    "S4": "Deflated Sharpe >= 0.95 (Suchaufwand herausgerechnet), Placebo-Vorsprung > 0, PBO <= 0.25.",
    "S5": "Holdout ab 2025, einmalig, nur für S4-Bestandene: >= 20 Trades und netto > 0.",
}


def step_by_step(df: pd.DataFrame) -> list[str]:
    """The batch result told the same way as the guided check: stage by stage,
    with who left and why (one line per strategy x instrument, its best row)."""
    out: list[str] = []
    cur = df.copy()
    fns = (("S1", screening.s1_explore), ("S2", screening.s2_cost),
           ("S3", screening.s3_robust), ("S4", screening.s4_significant))
    for tag, fn in fns:
        st = fn(cur)
        out += [f"### {tag}: {len(st.kept)} von {st.n_in} bestanden", "", STAGE_TEXT[tag], ""]
        gone = pd.concat([st.dropped, st.skipped if tag not in screening.CARRY_SKIPPED else st.skipped.iloc[0:0]])
        if len(gone):
            best = gone.sort_values("objective_lcb", ascending=False).groupby(["strategy", "asset"], sort=False).head(1)
            out += ["| raus | netto R | Grund |", "|---|---|---|"]
            for _, r in best.iterrows():
                why = GuidedCheck._reason(tag, r) if r["name"] in set(st.dropped["name"]) else "Wert fehlt (kein Nachweis)"
                out.append(f"| {r['strategy']}:{r['asset']} (beste Variante) | {r['net_mean']:+.3f} | {why} |")
            out.append("")
        kept_ids = sorted({f"{a}:{b}" for a, b in zip(st.kept["strategy"], st.kept["asset"])})
        out += [f"Weiter: {', '.join(kept_ids) if kept_ids else 'niemand'}", ""]
        cur = pd.concat([st.kept, st.skipped]) if len(st.skipped) and tag in screening.CARRY_SKIPPED else st.kept
        if cur.empty:
            out += ["Keine Variante mehr übrig - die folgenden Stufen prüfen nichts mehr.", ""]
            break
    if not cur.empty and "oos_pass" in cur:
        ok = cur[cur["oos_pass"] == True]  # noqa: E712
        out += [f"### S5: {len(ok)} von {len(cur)} bestanden", "", STAGE_TEXT["S5"], ""]
        for _, r in cur.iterrows():
            out.append(f"- {r['name']}: Holdout n={r.get('oos_n')}, netto {r.get('oos_net_mean')}, bestanden {r.get('oos_pass')}")
        out.append("")
    return out


def prop_check(row: pd.Series) -> dict:
    name, asset = row["strategy"], row["asset"]
    spec = REGISTRY[name][0](asset)
    m1 = load_cached(asset)
    tr_all = normalise(spec.run(replace(spec.base, **row["params"]), spec.prepare(m1)), m1, spec.risk_pct,
                       spec.entry_offset_min, spec.charges_spread)
    out = {}
    for label, part in (("train", tr_all[tr_all["entry_time"] < HOLDOUT_START]),
                        ("holdout", tr_all[tr_all["entry_time"] >= HOLDOUT_START])):
        if label == "holdout" and pd.isna(row.get("oos_n")):
            out[label] = "gesperrt (S4 nicht bestanden)"
            continue
        if part.empty:
            continue
        months = max((part["exit_time"].max() - part["entry_time"].min()).days / 30.44, 1.0)
        r_month = part["r"].sum() / months
        risk_pct = float(np.clip(1.5 / r_month, 0.1, 2.0)) if r_month > 0 else 0.5
        days = pd.bdate_range(part["entry_time"].min().tz_localize(None).normalize(),
                              part["exit_time"].max().tz_localize(None).normalize())
        log = [{"day": str(t.tz_localize(None).date()), "r": r}
               for t, r in zip(part["exit_time"], part["r"])]
        res = {}
        for f in FIRMS:
            firm = prop_sim.get_firm(f)
            daily = prop_sim.daily_pnl(log, risk_usd=prop_sim.risk_usd_for(firm.account_size, risk_pct),
                                       days=[d.date() for d in days])
            if firm.is_instant:
                fr = prop_sim.simulate_funded(daily, firm, runs=2000)
                res[f] = fr.summary() if hasattr(fr, "summary") else str(fr)
            else:
                res[f] = prop_sim.evaluate(daily, firm, runs=2000).summary()
        out[label] = {"n": int(len(part)), "r_per_month": round(float(r_month), 3),
                      "risk_pct_for_1.5pct_month": round(risk_pct, 3),
                      "pct_per_month_at_0.5pct_risk": round(float(r_month) * 0.5, 2), "firms": res}
    return out


def main() -> int:
    rows = json.loads((OUT / "variants.json").read_text())
    df = pd.DataFrame(rows)
    df["reached"] = stage_reached(df)
    order = {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5}
    df["rank"] = df["reached"].map(order)
    best = (df.sort_values(["rank", "dsr", "objective_lcb"], ascending=False)
              .groupby(["strategy", "asset"], sort=False).head(1))
    lines = ["# Optimiser + Trichter - Bericht", "",
             f"Train: 2019-01-01 .. {HOLDOUT_START.date()} (exklusiv), Holdout ab {HOLDOUT_START.date()}.",
             f"{len(df)} Kandidaten-Zeilen aus {df.groupby(['strategy', 'asset']).ngroups} Laeufen.", "",
             "## Wie weit kam jede Strategie (beste Variante je Instrument)", "",
             "| Strategie | Instrument | Stufe | n | brutto R | netto R | Placebo-Vorsprung | Jahre + | Nachbarn + | t/Tag | n_eff | DSR | PBO |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in best.sort_values(["rank", "dsr"], ascending=False).iterrows():
        nb = "-" if pd.isna(r["neighbour_positive"]) else f"{r['neighbour_positive']:.0%}"
        lines.append(f"| {r['strategy']} | {r['asset']} | **{r['reached']}** | {r['n']} | "
                     f"{r['gross_mean']:+.3f} | {r['net_mean']:+.3f} | {r['placebo_edge']:+.3f} | "
                     f"{r['pos_years']}/{r['n_years']} | {nb} | {r['t_daily']:.2f} | "
                     f"{r['n_eff']:.0f} | {r['dsr']:.3f} | {r['pbo']:.2f} |")
    lines += ["", "## Schritt für Schritt durch den Trichter", ""] + step_by_step(df)
    cands = df[(df["rank"] >= 4) | ((df["rank"] >= 3) & (df["dsr"] >= NEAR_DSR))]
    cands = (cands.sort_values(["rank", "dsr"], ascending=False)
                  .groupby(["strategy", "asset"], sort=False).head(1).head(8))
    lines += ["", f"## Kandidaten (S4 bestanden oder knapp: S3 + DSR >= {NEAR_DSR})", ""]
    if cands.empty:
        lines.append("Keine.")
    for _, r in cands.iterrows():
        print(f"  Prop-Test {r['name']} ...", flush=True)
        pc = prop_check(r)
        lines += [f"### {r['name']} - Stufe {r['reached']}", "",
                  f"Parameter: `{json.dumps(r['params'], sort_keys=True)}`", "",
                  f"Train: n={r['n']}, netto {r['net_mean']:+.3f} R/Trade, t/Tag {r['t_daily']:.2f}, "
                  f"DSR {r['dsr']:.3f} (n_eff {r['n_eff']:.0f}), PBO {r['pbo']:.2f}, "
                  f"Jahre {r['pos_years']}/{r['n_years']}", ""]
        if not pd.isna(r.get("oos_n")):
            lines.append(f"Holdout: n={int(r['oos_n'])}, netto {r['oos_net_mean']:+.3f} R/Trade, "
                         f"t/Tag {r['oos_t_daily']:.2f}, bestanden: {bool(r['oos_pass'])}")
            lines.append("")
        lines.append("```")
        lines.append(json.dumps(pc, indent=1, ensure_ascii=False))
        lines.append("```")
        lines.append("")
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines[:40]))
    print(f"\n-> {OUT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
