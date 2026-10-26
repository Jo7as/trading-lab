#!/usr/bin/env python3
"""H-005 (London-16:00-Fix) and H-006 (relative value between linked majors).

Both cards were killed by the S0 *prediction* (cost quote 0.12 > 0.10) without a
single trade being simulated. This study tests them for the first time, on the
TRAIN years only (2019-2024), through the same funnel as every other candidate:
gross first, then real bid/ask, then years, then the daily t.

H-005: drift 10:00-10:59 NY >= k spreads -> enter at 11:00 NY *against* the drift
(market, real bid/ask), exit 12:00 NY (variant: 11:30). Placebo: the same set-up one
hour earlier (pre 08:00-08:59, entry 09:00) - no benchmark flow at that time.
Variant ``nur_fix_tage``: month-end days only.

H-006: relative return since 03:00 NY of pair A minus pair B, on a 5-minute grid,
z-scored against the same clock time over the previous 14 days. First |z| >= 2
after 04:00 NY -> short the out-performer, long the under-performer (two legs at
real bid/ask), exit at |z| <= 0.5 or 16:00 NY. Pairs: EURUSD/GBPUSD and
AUDUSD/NZDUSD; placebo: EURGBP/AUDNZD (no common leg).

Units: bp of the price, per event (two legs summed for H-006).

  .venv/bin/python studies/run_h005_h006.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "studies")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_lab.registry import HOLDOUT_START, load_cached  # noqa: E402
from trading_lab import screening  # noqa: E402

NY = "America/New_York"
OUT = Path("artifacts/h005_h006.json")


def _frame(asset: str) -> pd.DataFrame:
    m1 = load_cached(asset)
    m1 = m1[m1.index < HOLDOUT_START]
    f = pd.DataFrame({"bid": m1["close"], "ask": m1["ask_close"]})
    f["mid"] = (f["bid"] + f["ask"]) / 2
    f["spread"] = f["ask"] - f["bid"]
    f.index = f.index.tz_convert(NY)
    return f


def _at(f: pd.DataFrame, hhmm: str) -> pd.DataFrame:
    """Row at (or the last one before) hh:mm each NY day."""
    day = f.index.normalize()
    t = pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))
    target = day + t
    sub = f[(f.index <= target) & (f.index > target - pd.Timedelta(minutes=5))]
    return sub.groupby(sub.index.normalize()).last()


def h005(asset: str, *, pre: tuple[str, str], entry: str, exit_: str, k: float,
         month_end_only: bool = False) -> pd.DataFrame:
    f = _frame(asset)
    a, b, e, x = _at(f, pre[0]), _at(f, pre[1]), _at(f, entry), _at(f, exit_)
    df = pd.concat({"a": a["mid"], "b": b["mid"], "sp": b["spread"],
                    "e_bid": e["bid"], "e_ask": e["ask"], "e_mid": e["mid"],
                    "x_bid": x["bid"], "x_ask": x["ask"], "x_mid": x["mid"]}, axis=1).dropna()
    df = df[df.index.dayofweek < 5]
    drift = (df["b"] - df["a"]) / df["sp"].replace(0, np.nan)
    df["dose"] = drift.abs()
    ev = df[df["dose"] >= k].copy()
    if month_end_only:
        last_bd = ev.index.to_series().groupby(ev.index.to_period("M")).transform("max")
        ev = ev[ev.index == last_bd]
    d = -np.sign(drift.loc[ev.index])                    # against the pre-fix flow
    ev["dir"] = d
    ev["gross_bp"] = d * (ev["x_mid"] - ev["e_mid"]) / ev["e_mid"] * 1e4
    fill_in = np.where(d > 0, ev["e_ask"], ev["e_bid"])
    fill_out = np.where(d > 0, ev["x_bid"], ev["x_ask"])
    ev["net_bp"] = d * (fill_out - fill_in) / ev["e_mid"] * 1e4
    ev["exit_time"] = ev.index
    return ev


def _grid5(f: pd.DataFrame) -> pd.DataFrame:
    return f.resample("5min").last().dropna()


def h006(pa: str, pb: str, *, z_in: float = 2.0, z_out: float = 0.5,
         start: str = "04:00", end: str = "16:00") -> pd.DataFrame:
    A, B = _grid5(_frame(pa)), _grid5(_frame(pb))
    j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
    j = j[j.index.dayofweek < 5]
    day = j.index.normalize()
    clock = j.index - day
    anchor = j[clock == pd.Timedelta(hours=3)]
    anchor = anchor.set_axis(anchor.index.normalize())      # keyed by day, not by 03:00
    j = j[clock >= pd.Timedelta(hours=3)]
    day, clock = j.index.normalize(), j.index - j.index.normalize()
    base = anchor[["mid_a", "mid_b"]].reindex(day).to_numpy()
    rel = np.log(j["mid_a"].to_numpy() / base[:, 0]) - np.log(j["mid_b"].to_numpy() / base[:, 1])
    j = j.assign(rel=rel, day=day, clock=clock).dropna(subset=["rel"])
    # z against the same clock time over the previous 14 days (no look-ahead)
    piv = j.pivot_table(index="day", columns="clock", values="rel")
    sd = piv.rolling(14, min_periods=10).std().shift(1)
    z = (piv / sd).stack().rename("z")
    j = j.join(z, on=["day", "clock"])
    t0, t1 = pd.Timedelta(start + ":00"), pd.Timedelta(end + ":00")
    rows = []
    for d, g in j.groupby("day"):
        g = g[(g["clock"] >= t0) & (g["clock"] <= t1)]
        hit = g.index[g["z"].abs() >= z_in]
        if len(hit) == 0:
            continue
        i = hit[0]
        zi = g.at[i, "z"]
        after = g.loc[i:]
        out = after.index[after["z"].abs() <= z_out]
        o = out[0] if len(out) else after.index[-1]
        s = -np.sign(zi)            # z > 0: A out-performed -> short A, long B
        ga = s * (g.at[o, "mid_a"] - g.at[i, "mid_a"]) / g.at[i, "mid_a"]
        gb = -s * (g.at[o, "mid_b"] - g.at[i, "mid_b"]) / g.at[i, "mid_b"]
        na = s * ((g.at[o, "bid_a"] if s > 0 else g.at[o, "ask_a"])
                  - (g.at[i, "ask_a"] if s > 0 else g.at[i, "bid_a"])) / g.at[i, "mid_a"]
        nb = -s * ((g.at[o, "bid_b"] if -s > 0 else g.at[o, "ask_b"])
                   - (g.at[i, "ask_b"] if -s > 0 else g.at[i, "bid_b"])) / g.at[i, "mid_b"]
        rows.append({"exit_time": o, "dose": abs(zi), "gross_bp": (ga + gb) * 1e4,
                     "net_bp": (na + nb) * 1e4})
    return pd.DataFrame(rows).set_index("exit_time", drop=False) if rows else pd.DataFrame()


def summarise(name: str, ev: pd.DataFrame, placebo: pd.DataFrame | None) -> dict:
    if ev.empty:
        return {"name": name, "n": 0}
    yrs = ev.groupby(ev["exit_time"].dt.year)["net_bp"].sum()
    daily = ev.groupby(ev["exit_time"].dt.normalize())["net_bp"].sum()
    q = pd.qcut(ev["dose"], 5, labels=False, duplicates="drop")
    dose = ev.groupby(q)["gross_bp"].mean().round(2).tolist()
    pg = float(placebo["gross_bp"].mean()) if placebo is not None and len(placebo) else None
    g = float(ev["gross_bp"].mean())
    return {"name": name, "n": int(len(ev)), "gross_mean": g, "net_mean": float(ev["net_bp"].mean()),
            "placebo_gross": pg, "placebo_edge": (g - pg) if pg is not None else None,
            "t_gross": screening.t_stat(ev["gross_bp"]), "t_daily": screening.t_stat(daily),
            "pos_years": int((yrs > 0).sum()), "n_years": int(len(yrs)),
            "dose_quintiles_gross_bp": dose, "gross_bp": None, "cost_bp": None,
            "neighbour_positive": None, "n_trials": None, "pbo": None, "oos_pass": None}


def main() -> int:
    rows = []
    for asset in ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD"):
        for k in (1.0, 2.0):
            base = h005(asset, pre=("10:00", "10:59"), entry="11:00", exit_="12:00", k=k)
            plac = h005(asset, pre=("08:00", "08:59"), entry="09:00", exit_="10:00", k=k)
            rows.append(summarise(f"H-005 {asset} k={k} 11->12", base, plac))
            rows.append(summarise(f"H-005 {asset} k={k} 11->11:30",
                                  h005(asset, pre=("10:00", "10:59"), entry="11:00", exit_="11:30", k=k),
                                  h005(asset, pre=("08:00", "08:59"), entry="09:00", exit_="09:30", k=k)))
            rows.append(summarise(f"H-005 {asset} k={k} nur Monatsende",
                                  h005(asset, pre=("10:00", "10:59"), entry="11:00", exit_="12:00", k=k,
                                       month_end_only=True), None))
        print(f"  H-005 {asset} fertig", flush=True)
    placebo6 = h006("EURGBP", "AUDNZD")
    for pa, pb in (("EURUSD", "GBPUSD"), ("AUDUSD", "NZDUSD")):
        for z_in in (2.0, 2.5):
            rows.append(summarise(f"H-006 {pa}/{pb} z>={z_in}", h006(pa, pb, z_in=z_in), placebo6))
            rows.append(summarise(f"H-006 {pa}/{pb} z>={z_in} overlap",
                                  h006(pa, pb, z_in=z_in, start="08:00", end="11:00"), placebo6))
        print(f"  H-006 {pa}/{pb} fertig", flush=True)
    rows.append(summarise("H-006 PLACEBO EURGBP/AUDNZD", placebo6, None))
    n_var = len(rows)
    for r in rows:
        r["n_trials"] = n_var
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df[["name", "n", "gross_mean", "net_mean", "placebo_edge", "t_gross", "t_daily",
              "pos_years", "n_years", "dose_quintiles_gross_bp"]].round(3).to_string(index=False))
    f = screening.run_funnel(df.drop(columns=["dose_quintiles_gross_bp"]), up_to="S4")
    print("\n" + f.summary())
    OUT.write_text(json.dumps({"rows": rows, "stages": [{"name": s.name, "kept": list(s.kept.get("name", []))}
                                                        for s in f.stages]}, indent=1, default=str))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
