#!/usr/bin/env python3
"""S-005 weekly level break: funnel over the 8 pre-registered variants, all 8 markets pooled.

Spec: ``docs/specs/weekly_level_break.yaml`` - NORMAL track (found in our own data): DSR with
8 trials, 70 % positive years, neighbours, placebo, PBO. Costs: ftmo profile + 0.5 bp
slippage, measured FundedNext swaps (``artifacts/broker_costs/fundednext_demo.json``).

  .venv/bin/python studies/weekly_level_check.py [--holdout 2026-10-29]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_lab import costs, screening  # noqa: E402
from trading_lab.registry import HOLDOUT_START, load_cached  # noqa: E402
from trading_lab.strategies import weekly_level_break as W  # noqa: E402

MARKETS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "US100", "SP500", "GER40", "XAUUSD"]
GRID = {"zeitrahmen": ("15min", "1h"), "art": ("durch", "alle"), "ziel_r": (3.0, 5.0)}
SWAPS = json.loads(Path("artifacts/broker_costs/fundednext_demo.json").read_text())["swaps_pa"]
OUT = Path("artifacts/weekly_level")
PLACEBO_RUNS = 100


def preps(holdout: bool) -> dict:
    out = {}
    for a in MARKETS:
        m1 = costs.adjust_frame(load_cached(a), a, "ftmo")
        if not holdout:
            m1 = m1[m1.index < HOLDOUT_START]
        out[a] = W.prepare(m1, a)
    return out


def run(ps: dict, params: dict, *, mid: bool = False, since=None, until=None) -> list[dict]:
    tr = []
    for a, p in ps.items():
        cfg = W.WeeklyBreakConfig(**params, fin_pa=tuple(SWAPS[a]) if a in SWAPS else None,
                                  slippage_bp=0.0 if mid else 0.5)
        tr += W.run_backtest(W.on_mid(p) if mid else p, cfg)["trades"]
    if since is not None:
        tr = [t for t in tr if t["entry_time"] >= since]
    if until is not None:
        tr = [t for t in tr if t["entry_time"] < until]
    return tr


def placebo(ps: dict, params: dict, real_by_asset: dict, seed: int) -> float:
    rng = np.random.default_rng(seed)
    rs = []
    for a, p in ps.items():
        cfg = W.WeeklyBreakConfig(**params, slippage_bp=0.0)
        ov = W.placebo_override(p, cfg, real_by_asset.get(a, []), rng)
        rs += [t["r"] for t in W.run_backtest(W.on_mid(p), cfg, override=ov)["trades"]]
    return float(np.mean(rs)) if rs else np.nan


def daily(trades: list[dict], start, end) -> pd.Series:
    s = pd.Series(0.0, index=pd.bdate_range(start, end))
    for t in trades:
        d = t["exit_time"].tz_convert(None).normalize()
        if d in s.index:
            s[d] += t["r"]
    return s


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", metavar="FREIGABE")
    a = ap.parse_args()
    hs = HOLDOUT_START
    ps = preps(False)
    keys = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    res = {}
    for k in keys:
        net = run(ps, k)
        gross = run(ps, k, mid=True)
        res[tuple(k.values())] = {"params": k, "net": net, "gross": gross,
                                  "daily": daily(net, "2019-01-01", hs.tz_localize(None) - pd.Timedelta(days=1))}
    mon = pd.DataFrame({key: pd.Series({t["exit_time"].strftime("%Y-%m"): 0.0 for t in v["net"]}).add(
        pd.Series([t["r"] for t in v["net"]], index=[t["exit_time"].strftime("%Y-%m") for t in v["net"]]).groupby(level=0).sum(),
        fill_value=0) for key, v in res.items()}).fillna(0.0).T
    pbo = screening.pbo(mon) if mon.shape[0] >= 5 else np.nan
    rows = []
    for key, v in res.items():
        by_asset = {}
        for t in v["net"]:
            by_asset.setdefault(t["asset"], []).append(t)
        pl = float(np.nanmean([placebo(ps, v["params"], by_asset, s) for s in range(PLACEBO_RUNS)]))
        g = float(np.mean([t["r"] for t in v["gross"]]))
        n_ = float(np.mean([t["r"] for t in v["net"]]))
        yrs = pd.Series([t["r"] for t in v["net"]], index=[t["exit_time"].year for t in v["net"]]).groupby(level=0).sum()
        nbs = [res[k2] for k2 in res if sum(x != y for x, y in zip(k2, key)) == 1]
        rows.append({"name": f"weekly_level_break:POOL#{len(rows)}", "strategy": "weekly_level_break", "asset": "POOL8",
                     "params": v["params"], "n": len(v["net"]), "gross_mean": g, "net_mean": n_,
                     "placebo_gross": pl, "placebo_edge": g - pl,
                     "pos_years": int((yrs > 0).sum()), "n_years": int(len(yrs)),
                     "neighbour_positive": float(np.mean([np.mean([t["r"] for t in x["net"]]) > 0 for x in nbs])),
                     "t_daily": float(screening.t_stat(v["daily"])), "n_trials": len(keys),
                     "dsr": screening.deflated_sharpe(v["daily"].to_numpy(), n_trials=len(keys)), "pbo": pbo,
                     "per_year_r": {int(k): round(float(x), 1) for k, x in yrs.items()},
                     "hit": float(np.mean([t["exit_reason"] == "ziel" for t in v["net"]]))})
        r = rows[-1]
        print(f"  {v['params']}: n {r['n']}, brutto {g:+.3f} R, netto {n_:+.3f} R, Placebo {pl:+.3f} R, "
              f"Ziel erreicht {r['hit']:.0%}, Jahre {r['pos_years']}/{r['n_years']} {r['per_year_r']}, "
              f"t {r['t_daily']:.2f}, DSR {r['dsr']:.2f}", flush=True)
    print(f"  PBO über die 8 Varianten: {pbo:.2f}")
    if a.holdout:
        full = preps(True)
        for row in rows:
            tr = run(full, row["params"], since=hs)
            rr = np.array([t["r"] for t in tr])
            row.update({"oos_n": len(tr), "oos_mean_r": float(rr.mean()) if len(rr) else None,
                        "oos_sum_r": float(rr.sum()), "oos_pass": bool(len(rr) >= 30 and rr.mean() > 0)})
    f = screening.run_funnel(pd.DataFrame(rows), up_to="S5" if a.holdout else "S4", holdout_freigabe=a.holdout)
    print("\n" + f.summary())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("funnel_holdout.json" if a.holdout else "funnel.json")).write_text(json.dumps(rows, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
