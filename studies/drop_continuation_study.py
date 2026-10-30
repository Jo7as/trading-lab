#!/usr/bin/env python3
"""S-010 stage 2: the five variants of ``docs/specs/drop_continuation.yaml`` as real trades.

2018-05..2024 only, the holdout 2025+ stays closed. Verdict rule: end of the spec.

  uv run python studies/drop_continuation_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import studies.h12_mining_study as h12  # noqa: E402
from trading_lab.chart_tokens import to_ns  # noqa: E402
from trading_lab.data_audit import build_study_frame  # noqa: E402
from trading_lab.market_structure import BOS_DN, BOS_UP, analyze, mid_ohlc  # noqa: E402
from trading_lab.pattern_mining import clustered_z  # noqa: E402
from trading_lab.strategies.drop_continuation import (HOLD_MIN, MIN_NS, VARIANTS, is_signal, last_bos,  # noqa: E402
                                                      pick_trades, variant_ok, walk)

MARKETS = ("US100", "US30", "GER40")
THR = {"US100": 1.87, "US30": 1.78, "GER40": 1.66}
OUT = Path("artifacts/drop_continuation.json")
PLACEBO = 500
Z_MIN, YEARS_MIN, MARKETS_MIN = 2.33, 5, 2


def market_signals(market: str) -> list[dict]:
    rows, _ = h12.market_rows(market, (2016, 2024))
    thr = float(rows["thr"].iloc[0])
    assert abs(thr - THR[market]) < 0.01, (market, thr)
    m1, _ = build_study_frame(h12.load_cached(market, 2016, 2024), market)
    ts = to_ns(m1.index)
    b = [m1[k].to_numpy() for k in ("open", "high", "low", "close",
                                    "ask_open", "ask_high", "ask_low", "ask_close")]
    bars15 = mid_ohlc(m1).resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    ev = analyze(bars15).bars["ext_event"].to_numpy()
    end15 = to_ns(bars15.index) + 15 * MIN_NS
    out = []
    for r in rows.itertuples():
        if r.period is None or not is_signal(r.b5, r.trend):
            continue
        i = int(np.searchsorted(ts, r.t))
        e = int(np.searchsorted(ts, r.t + HOLD_MIN * MIN_NS))
        if i >= len(ts) or ts[i] >= r.t + 5 * MIN_NS or e <= i:
            continue
        m = thr * r.atr
        short = walk(*b, i, e, m, -1)
        long_ = walk(*b, i, e, m, 1)
        out.append({"market": market, "t": int(r.t), "exit_t": int(r.t + HOLD_MIN * MIN_NS), "day": str(r.day.date()),
                    "year": r.day.year, "clock": r.clock, "newhl": r.newhl, "volq": r.volq,
                    "bos15": last_bos(ev, end15, int(r.t), BOS_UP, BOS_DN),
                    "r": short.r, "outcome": short.outcome, "r_long": long_.r})
    print(f"  {market}: {len(out)} signals", flush=True)
    return out


def evaluate(trades: list[dict], rng: np.random.Generator) -> dict:
    d = pd.DataFrame(trades)
    by_y = d.groupby("year")["r"].mean()
    by_m = d.groupby("market")["r"].mean()
    z = clustered_z(d["r"].to_numpy(), d["day"].to_numpy())
    flip = rng.random((PLACEBO, len(d))) < 0.5
    plc = np.where(flip, d["r"].to_numpy(), d["r_long"].to_numpy()).mean(axis=1)
    res = {"trades": len(d), "per_week": len(d) / (d["day"].nunique() and pd.to_datetime(d["day"]).agg(["min", "max"]).diff().iloc[-1].days / 7),
           "mean_r": float(d["r"].mean()), "z_day": z, "tp": float((d["outcome"] == "tp").mean()),
           "stop": float((d["outcome"] == "stop").mean()), "time": float((d["outcome"] == "time").mean()),
           "by_year": by_y.round(3).to_dict(), "by_market": by_m.round(3).to_dict(),
           "years_pos": int((by_y > 0).sum()), "markets_pos": int((by_m > 0).sum()),
           "placebo_mean_r": float(plc.mean()), "placebo_p95": float(np.quantile(plc, 0.95))}
    res["passed"] = bool(res["mean_r"] > 0 and z >= Z_MIN and res["years_pos"] >= YEARS_MIN
                         and res["markets_pos"] >= MARKETS_MIN)
    return res


def main() -> int:
    sig = [s for m in MARKETS for s in market_signals(m)]
    rng = np.random.default_rng(10)
    report = {}
    for v in VARIANTS:
        keep = [s for s in sig if variant_ok(v, clock=s["clock"], bos15=s["bos15"], newhl=s["newhl"], volq=s["volq"])]
        report[v] = evaluate(pick_trades(keep), rng)
        r = report[v]
        print(f"{v:11s} n {r['trades']:5d} {r['per_week']:4.1f}/wk  net R {r['mean_r']:+.4f}  z {r['z_day']:5.2f}  "
              f"tp/stop/time {r['tp']:.0%}/{r['stop']:.0%}/{r['time']:.0%}  years+ {r['years_pos']}/7  markets+ "
              f"{r['markets_pos']}/3  placebo {r['placebo_mean_r']:+.4f} (95% {r['placebo_p95']:+.4f})  "
              f"{'PASS' if r['passed'] else 'fail'}\n            years {r['by_year']}\n            markets {r['by_market']}")
    OUT.write_text(json.dumps(report, indent=1, default=str))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
