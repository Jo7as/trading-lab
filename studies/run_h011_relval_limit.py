#!/usr/bin/env python3
"""H-011: relative value AUDUSD/NZDUSD with LIMIT entries and venue costs (train only).

Follow-up of H-006, which found a real gross reversion (+1.7..+3.1 bp, placebo ~0)
that died at S2 because two market legs paid the (too wide) Dukascopy spread and the
model charged no commission. This study changes ONLY the execution, not the signal:

Signal (unchanged from H-006): log return since 03:00 NY of A minus B on a 5-minute
grid, z-scored against the same clock time over the previous 14 days. First |z| >= z_in
inside the window -> short the out-performer, long the under-performer; exit at
|z| <= 0.5 or 16:00 NY.

Execution, resolved on 1-minute bid/ask:
  * ``market``: both legs cross the spread at the signal bar (taker in, taker out);
  * ``limit``:  both legs rest at the touch (buy at the bid, sell at the ask) for
    ``wait`` minutes. A buy limit fills when the ask trades down to it, a sell limit
    when the bid trades up to it. If only one leg fills, the other is completed at
    market at the end of the wait (pays the spread); if none fills, no trade.
  Exit is always at market. Commission per fill from ``trading_lab.costs``.

Reported per variant: events, fill rate, gross (mid-to-mid) and net bp per event,
placebo (EURGBP/AUDNZD through the same machinery), share of positive years, the daily
t, dose quintiles, and the adverse-selection check (gross of filled vs unfilled).

TRAIN ONLY (2019-2024). The holdout for these pairs has never been opened.

  .venv/bin/python studies/run_h011_relval_limit.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_lab import costs, screening  # noqa: E402
from trading_lab.registry import HOLDOUT_START, load_cached  # noqa: E402

NY = "America/New_York"
OUT = Path("artifacts/h011_relval_limit.json")
PRIOR_TRIALS = 8          # H-006 already looked at 8 variants of this signal


def _m1(asset: str, profile: str) -> pd.DataFrame:
    m = load_cached(asset)
    m = m[m.index < HOLDOUT_START]
    return costs.adjust_frame(m, asset, profile)


def _grid5(m1: pd.DataFrame) -> pd.DataFrame:
    g = pd.DataFrame({"bid": m1["close"], "ask": m1["ask_close"]}).resample("5min").last().dropna()
    g["mid"] = (g["bid"] + g["ask"]) / 2
    g.index = g.index.tz_convert(NY)
    return g


def signals(A5: pd.DataFrame, B5: pd.DataFrame, *, z_in: float, z_out: float = 0.5,
            start: str = "04:00", end: str = "16:00") -> pd.DataFrame:
    """Entry/exit times and the sign of leg A (H-006 signal, no look-ahead)."""
    j = A5[["mid"]].join(B5[["mid"]], lsuffix="_a", rsuffix="_b", how="inner")
    j = j[j.index.dayofweek < 5]
    day, clock = j.index.normalize(), j.index - j.index.normalize()
    anchor = j[clock == pd.Timedelta(hours=3)]
    anchor = anchor.set_axis(anchor.index.normalize())
    j = j[clock >= pd.Timedelta(hours=3)]
    day, clock = j.index.normalize(), j.index - j.index.normalize()
    base = anchor[["mid_a", "mid_b"]].reindex(day).to_numpy()
    rel = np.log(j["mid_a"].to_numpy() / base[:, 0]) - np.log(j["mid_b"].to_numpy() / base[:, 1])
    j = j.assign(rel=rel, day=day, clock=clock).dropna(subset=["rel"])
    piv = j.pivot_table(index="day", columns="clock", values="rel")
    sd = piv.rolling(14, min_periods=10).std().shift(1)
    j = j.join((piv / sd).stack().rename("z"), on=["day", "clock"])
    t0, t1 = pd.Timedelta(start + ":00"), pd.Timedelta(end + ":00")
    rows = []
    for d, g in j.groupby("day"):
        g = g[(g["clock"] >= t0) & (g["clock"] <= pd.Timedelta("16:00:00"))]
        win = g[g["clock"] <= t1]
        hit = win.index[win["z"].abs() >= z_in]
        if len(hit) == 0:
            continue
        i = hit[0]
        after = g.loc[i:]
        out = after.index[after["z"].abs() <= z_out]
        o = out[0] if len(out) else after.index[-1]
        rows.append({"entry": i, "exit": o, "sign_a": -np.sign(g.at[i, "z"]), "dose": abs(g.at[i, "z"])})
    return pd.DataFrame(rows)


def _leg(m1: pd.DataFrame, t_sig: pd.Timestamp, t_exit: pd.Timestamp, side: int, mode: str,
         wait: int, profile: str, asset: str) -> dict | None:
    """One leg. ``side`` +1 = buy, -1 = sell. Prices relative to the leg's mid."""
    idx = m1.index
    i = idx.searchsorted(t_sig.tz_convert("UTC") + pd.Timedelta(minutes=5))   # bar after the 5-min signal bar
    x = idx.searchsorted(t_exit.tz_convert("UTC") + pd.Timedelta(minutes=5))
    if i >= len(idx) or x >= len(idx) or x <= i:
        return None
    bid, ask = m1["close"].to_numpy(), m1["ask_close"].to_numpy()
    mid0 = (bid[i] + ask[i]) / 2
    filled, fill_px, fill_i = False, np.nan, i
    if mode == "market":
        filled, fill_px = True, (ask[i] if side > 0 else bid[i])
    else:
        lim = bid[i] if side > 0 else ask[i]
        lo_a, hi_b = m1["ask_low"].to_numpy(), m1["high"].to_numpy()
        for k in range(i + 1, min(i + 1 + wait, x)):
            if (side > 0 and lo_a[k] <= lim) or (side < 0 and hi_b[k] >= lim):
                filled, fill_px, fill_i = True, lim, k
                break
    out_px = bid[x] if side > 0 else ask[x]
    mid_x = (bid[x] + ask[x]) / 2
    com = costs.commission_bp_side(asset, mid0, profile)
    return {"filled": filled, "fill_px": fill_px, "fill_i": fill_i, "mid0": mid0, "mid_x": mid_x,
            "out_px": out_px, "i_end_wait": min(i + wait, x - 1), "com_bp": com,
            "gross_bp": side * (mid_x - mid0) / mid0 * 1e4}


def simulate(pa: str, pb: str, *, z_in: float, start: str, end: str, mode: str, wait: int,
             profile: str) -> pd.DataFrame:
    ma, mb = _m1(pa, profile), _m1(pb, profile)
    sig = signals(_grid5(ma), _grid5(mb), z_in=z_in, start=start, end=end)
    rows = []
    bid_a, ask_a = ma["close"].to_numpy(), ma["ask_close"].to_numpy()
    bid_b, ask_b = mb["close"].to_numpy(), mb["ask_close"].to_numpy()
    for _, s in sig.iterrows():
        sa = int(s["sign_a"])
        la = _leg(ma, s["entry"], s["exit"], sa, mode, wait, profile, pa)
        lb = _leg(mb, s["entry"], s["exit"], -sa, mode, wait, profile, pb)
        if la is None or lb is None:
            continue
        gross = la["gross_bp"] + lb["gross_bp"]
        if not (la["filled"] or lb["filled"]):
            rows.append({"exit_time": s["exit"], "dose": s["dose"], "gross_bp": gross,
                         "filled": False, "net_bp": np.nan})
            continue
        net = 0.0
        for leg, side, bid, ask in ((la, sa, bid_a, ask_a), (lb, -sa, bid_b, ask_b)):
            px = leg["fill_px"]
            if not leg["filled"]:                      # complete at market after the wait
                k = leg["i_end_wait"]
                px = ask[k] if side > 0 else bid[k]
            net += side * (leg["out_px"] - px) / leg["mid0"] * 1e4 - 2 * leg["com_bp"]
        rows.append({"exit_time": s["exit"], "dose": s["dose"], "gross_bp": gross,
                     "filled": True, "both_filled": la["filled"] and lb["filled"], "net_bp": net})
    return pd.DataFrame(rows)


def summarise(name: str, ev: pd.DataFrame, placebo: pd.DataFrame | None, n_trials: int) -> dict:
    if ev.empty:
        return {"name": name, "n": 0}
    tr = ev[ev["filled"]]
    yrs = tr.groupby(pd.to_datetime(tr["exit_time"]).dt.year)["net_bp"].sum()
    daily = tr.groupby(pd.to_datetime(tr["exit_time"]).dt.tz_convert("UTC").dt.normalize())["net_bp"].sum()
    q = pd.qcut(tr["dose"], 5, labels=False, duplicates="drop") if len(tr) >= 25 else None
    pg = float(placebo.loc[placebo["filled"], "gross_bp"].mean()) if placebo is not None and len(placebo) else None
    g = float(tr["gross_bp"].mean()) if len(tr) else float("nan")
    unf = ev[~ev["filled"]]
    return {
        "name": name, "signale": int(len(ev)), "n": int(len(tr)),
        "fill_rate": round(float(ev["filled"].mean()), 3),
        "beide_gefuellt": round(float(tr.get("both_filled", pd.Series(dtype=bool)).mean()), 3) if len(tr) else None,
        "gross_mean": g, "net_mean": float(tr["net_bp"].mean()) if len(tr) else float("nan"),
        "gross_unfilled": float(unf["gross_bp"].mean()) if len(unf) else None,
        "placebo_gross": pg, "placebo_edge": (g - pg) if pg is not None else None,
        "t_daily": screening.t_stat(daily), "pos_years": int((yrs > 0).sum()), "n_years": int(len(yrs)),
        "dose_quintiles_gross_bp": (tr.groupby(q)["gross_bp"].mean().round(2).tolist() if q is not None else None),
        "gross_bp": None, "cost_bp": None, "neighbour_positive": None,
        "n_trials": n_trials, "pbo": None, "oos_pass": None,
    }


def main() -> int:
    warnings.filterwarnings("ignore")
    grid = [(z, st, en) for z in (2.0, 2.5) for st, en in (("04:00", "16:00"), ("08:00", "11:00"))]
    execs = [("market", 0), ("limit", 10)]
    profiles = ["ftmo", "ecn_raw"]
    n_trials = PRIOR_TRIALS + len(grid) * len(execs) * len(profiles)
    rows, results = [], {}
    for prof in profiles:
        for mode, wait in execs:
            plac = simulate("EURGBP", "AUDNZD", z_in=2.0, start="04:00", end="16:00", mode=mode,
                            wait=wait, profile=prof)
            for z, st, en in grid:
                ev = simulate("AUDUSD", "NZDUSD", z_in=z, start=st, end=en, mode=mode, wait=wait, profile=prof)
                name = f"AUD/NZD z>={z} {st}-{en} {mode} [{prof}]"
                r = summarise(name, ev, plac, n_trials)
                r.update({"profil": prof, "ausfuehrung": mode, "z_in": z, "fenster": f"{st}-{en}"})
                rows.append(r)
                print(f"  {name:42s} n={r['n']:4d} fill={r['fill_rate']:.2f} brutto {r['gross_mean']:+.2f} "
                      f"netto {r['net_mean']:+.2f} bp  t/Tag {r['t_daily']:+.2f}  Jahre {r['pos_years']}/{r['n_years']}",
                      flush=True)
    df = pd.DataFrame(rows)
    f = screening.run_funnel(df.drop(columns=["dose_quintiles_gross_bp"]), up_to="S4")
    print("\n" + f.summary())
    OUT.write_text(json.dumps({"n_trials": n_trials, "rows": rows,
                               "stages": [{"name": s.name, "kept": list(s.kept.get("name", []))} for s in f.stages]},
                              indent=1, default=str))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
