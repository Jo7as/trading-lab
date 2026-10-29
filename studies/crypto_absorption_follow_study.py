"""H9: follow the flow after absorption at SMC levels (pre-registration: docs/FINDINGS.md,
2026-10-29 "Crypto H9"). Measurement only; decisive sample 2023, 2024 = H8's sample reported
separately (it generated the hypothesis).

Event = H8's absorption (strong taker flow into a first-touched level, 5-min window closes back on
the approach side), traded in the FLOW direction +d from the end of the window, 4 h (1 h
secondary). Controls: the same rule at randomly shifted levels (3 seeds) and random direction on
the same events. Net = gross - Bybit taker round trip. t clustered by UTC day.

    uv run python studies/crypto_absorption_follow_study.py [--years 2023 2024] [--venue binance_um]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from trading_lab import costs  # noqa: E402
from trading_lab import level_flow as lf  # noqa: E402
from trading_lab import orderflow as of  # noqa: E402
from trading_lab.adapters import crypto_trades as ct  # noqa: E402
from trading_lab.indicators import atr as atr_fn  # noqa: E402

MARKETS = ("BTC", "SOL")
BIN = {"BTC": 25.0, "SOL": 0.10}
HORIZONS = {"1h": "1h", "4h": "4h"}
SEEDS = (1, 2, 3)
OB = ("ob_bear", "ob_bull")


def touches(market: str, year: int, venue: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All first-touch events at real and control levels (unclassified), one market-year."""
    if year >= 2025:
        raise ValueError("2025+ is the holdout")
    bars = ct.load_bars(venue, market, date(year, 1, 1), date(year, 12, 31), allow_download=False)
    m15 = of.resample(bars, "15min").dropna(subset=["close"])
    a15 = atr_fn(m15["high"], m15["low"], m15["close"], 14)
    a15.index = a15.index + pd.Timedelta("15min")            # known at the end of the bar
    atr10 = a15.reindex(bars.index, method="ffill")
    levels = (lf.period_levels(bars, "day", "pd") + lf.period_levels(bars, "week", "pw")
              + lf.structure_levels(m15) + lf.profile_levels_daily(bars, BIN[market]))
    real = lf.touch_events(bars, levels, atr10, horizons=HORIZONS)
    ctl = pd.concat([lf.touch_events(bars, lf.control_levels(levels, seed=s), atr10, horizons=HORIZONS)
                     for s in SEEDS], ignore_index=True)
    cost = costs.crypto_round_trip_bp(market, "bybit_perp")
    for df in (real, ctl):
        df["market"], df["year"], df["cost"] = market, year, cost
    return real, ctl


def signed(ev: pd.DataFrame) -> pd.DataFrame:
    out = ev.copy()
    for h in HORIZONS:
        out[f"sig_{h}"] = out["trade"] * out[f"ret_{h}"]
    return out


def stats(df: pd.DataFrame, h: str) -> dict:
    if len(df) < 10:
        return {"n": len(df), "gross": np.nan, "t": np.nan, "net": np.nan}
    x = df[f"sig_{h}"]
    return {"n": len(df), "gross": float(x.mean()), "t": lf.day_clustered_t(x, df["ts"]),
            "net": float((x - df["cost"]).mean())}


def fmt(s: dict) -> str:
    if not np.isfinite(s["gross"]):
        return f"{s['n']} | - | - | -"
    return f"{s['n']} | {s['gross']:+.1f} | {s['t']:+.2f} | {s['net']:+.1f}"


def control_cols(r: pd.DataFrame, c: pd.DataFrame, h: str) -> str:
    if len(r) < 10 or len(c) < 10:
        return "- | - | - | -"
    diff, t_diff = lf.cluster_diff_t(r[f"sig_{h}"], r["ts"], c[f"sig_{h}"], c["ts"])
    p = lf.random_direction_p(r[f"ret_{h}"], r["trade"])
    return f"{c[f'sig_{h}'].mean():+.1f} | {diff:+.1f} | {t_diff:+.2f} | {p:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    ap.add_argument("--venue", default="binance_um")
    a = ap.parse_args()
    reals, ctls = [], []
    for year in a.years:
        for m in MARKETS:
            r, c = touches(m, year, a.venue)
            reals.append(r)
            ctls.append(c)
    real_all, ctl_all = pd.concat(reals, ignore_index=True), pd.concat(ctls, ignore_index=True)
    real, ctl = signed(lf.follow_absorption(real_all)), signed(lf.follow_absorption(ctl_all))

    head = "| n | gross bp | t (day) | net bp | control gross | real - control | t diff | p random dir |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [f"# H9 follow the flow after absorption (flow + prices: {a.venue})", "",
             "Signed bp per event from the end of the 5-min window, trade = +d (flow direction); "
             "t clustered by UTC day; net = gross - Bybit taker round trip "
             f"(BTC {costs.crypto_round_trip_bp('BTC', 'bybit_perp'):.1f}, "
             f"SOL {costs.crypto_round_trip_bp('SOL', 'bybit_perp'):.1f} bp). Control = same levels "
             f"moved 0.15-0.5 %, seeds {SEEDS}. p random dir = one-sided, 2000 draws.", ""]

    first = a.years[0]
    r0, c0 = real[real["year"] == first], ctl[ctl["year"] == first]
    if first == 2023:
        s = stats(r0, "4h")
        diff, t_diff = lf.cluster_diff_t(r0["sig_4h"], r0["ts"], c0["sig_4h"], c0["ts"])
        p = lf.random_direction_p(r0["ret_4h"], r0["trade"])
        by_m = {m: float(r0.loc[r0["market"] == m, "sig_4h"].mean()) for m in MARKETS}
        verdict, cond = lf.h9_verdict(gross=s["gross"], t_gross=s["t"], net=s["net"], diff=diff,
                                      t_diff=t_diff, p_random=p, gross_by_market=by_m)
        lines += ["## Pre-registered decision (2023, BTC + SOL pooled, 4 h)", "",
                  f"n {s['n']}, gross {s['gross']:+.1f} bp (t {s['t']:+.2f}), net {s['net']:+.1f} bp, "
                  f"real - control {diff:+.1f} bp (t {t_diff:+.2f}), random-direction p {p:.3f}, "
                  + ", ".join(f"{m} gross {v:+.1f}" for m, v in by_m.items()), "",
                  "Conditions: " + ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in cond.items()), "",
                  f"**Verdict: {verdict}**", ""]

    for h in HORIZONS:
        lines += [f"## Per year and market, {h}", "", "| year | market " + head, "|---|---" + sep]
        for year in a.years:
            for m in (*MARKETS, "both"):
                sel_r = (real["year"] == year) & ((real["market"] == m) | (m == "both"))
                sel_c = (ctl["year"] == year) & ((ctl["market"] == m) | (m == "both"))
                lines.append(f"| {year} | {m} | {fmt(stats(real[sel_r], h))} | "
                             f"{control_cols(real[sel_r], ctl[sel_c], h)} |")
        lines.append("")

    h = "4h"
    lines += [f"## {first} per quarter, both markets, {h}", "", "| quarter " + head, "|---" + sep]
    for q in range(1, 5):
        rq, cq = r0[r0["ts"].dt.quarter == q], c0[c0["ts"].dt.quarter == q]
        lines.append(f"| Q{q} | {fmt(stats(rq, h))} | {control_cols(rq, cq, h)} |")
    lines += ["", f"## Order blocks only, both markets, {h}", "", "| year " + head, "|---" + sep]
    for year in a.years:
        ro = real[(real["year"] == year) & real["kind"].isin(OB)]
        co = ctl[(ctl["year"] == year) & ctl["kind"].isin(OB)]
        lines.append(f"| {year} | {fmt(stats(ro, h))} | {control_cols(ro, co, h)} |")
    lines += ["", f"## Non-overlapping (per market, >= {h} apart), both markets", "",
              "| year | n | gross bp | t (day) | net bp |", "|---|---|---|---|---|"]
    for year in a.years:
        ro = lf.non_overlapping(real[real["year"] == year], h)
        lines.append(f"| {year} | {fmt(stats(ro, h))} |")

    lines += ["", f"## Comparison rows (real levels, both markets, {h})", "",
              "| year | rule | n | gross bp | t (day) | net bp |", "|---|---|---|---|---|---|"]
    every = real_all.assign(trade=real_all["d"])
    lab = lf.classify(real_all)
    for year in a.years:
        e = signed(every[every["year"] == year])
        cas = signed(lab[(lab["year"] == year) & (lab["label"] == "cascade")])
        fade = signed(lab[(lab["year"] == year) & (lab["label"] == "absorption")])
        lines.append(f"| {year} | follow +d at every touch | {fmt(stats(e, h))} |")
        lines.append(f"| {year} | cascade (flow in, close through), +d | {fmt(stats(cas, h))} |")
        lines.append(f"| {year} | absorption faded (H8), -d | {fmt(stats(fade, h))} |")
    lines += ["", f"Events: absorption real {len(real)}, control {len(ctl)}; all touches real "
              f"{len(real_all)}, control {len(ctl_all)}."]

    text = "\n".join(lines)
    out = ct.CACHE.parent.parent / "crypto" / f"absorption_follow_{a.venue}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    real.to_pickle(out.with_suffix(".events.pkl"))
    print(text)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
