#!/usr/bin/env python3
"""Asymmetric RR: does the momentum edge survive a proper (target, stop) sweep?

The grid found a real, robust signal: `trend_continue` beat its random placebo in
**960 of 960 cells** (+0.31 spreads), while `trend_fade` lost in all 960. But only
11 of 3840 cells had E[R] > 0, and they were almost all EURUSD with T=8, S=2 -
i.e. a target four times the stop. With a symmetric ratio the edge (+0.31 spreads)
is simply smaller than the cost (1 spread) and has to lose.

So this sweeps the asymmetric region properly, and controls the two things that
could be artefacts rather than edge:

1. **Discretisation.** Stop and target are resolved on 1-minute bars. Within one
   bar the *order* is unknown, so the bracket convention decides the result. The
   repo uses "stop wins"; a target of 20 spreads with a stop of 1 spreads makes
   that convention very load-bearing. Everything is therefore reported under
   three conventions: stop wins, target wins, and a coin-flip tie-break.
2. **Execution side.** The path uses the mid, but a long's stop fills at the bid
   and its target at the ask. That shifts the effective stop *and* target by half
   a spread each way, which matters precisely when S is 1-2 spreads. In the code
   the ``execution`` axis enters only through the scalar round-trip charge, taken
   from the single convention in ``trading_lab.cost_gate``: ``"full"`` = both
   legs cross the book = ``round_trip_spread("market", "market")`` = 1.0 x spread,
   ``"half"`` = resting entry / taker exit = ``round_trip_spread("limit",
   "market")`` = 0.5 x spread. The path itself stays on the mid.

One trigger per day (the first), so the sample is not dominated by overlapping
windows of the same move.

  .venv/bin/python studies/asymmetric_rr.py
  .venv/bin/python studies/asymmetric_rr.py --assets EURUSD,USDJPY,US100
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_lab.minute_data import DEFAULT_CACHE, load_minute_bid_ask  # noqa: E402
from trading_lab.cost_gate import round_trip_spread  # noqa: E402

ASSETS = ("EURUSD", "USDJPY", "GBPUSD", "US100", "GER40", "XAUUSD")
TARGETS = (5, 8, 12, 20, 30)
STOPS = (1.0, 1.5, 2.0, 3.0)
HORIZONS = (30, 60, 120)
INF = 10 ** 9

# Round-trip spread charge per ``execution`` convention, units of the FULL spread.
COST_PER_ROUND_TRIP = {"full": round_trip_spread("market", "market"),   # 1.0
                       "half": round_trip_spread("limit", "market")}    # 0.5


def directions(m1: pd.DataFrame, kind: str, tf: str = "60min", lookback: int = 6) -> np.ndarray:
    mid = (m1["close"] + m1["ask_close"]) / 2.0
    if kind == "random":
        rng = np.random.default_rng(12345)
        return rng.choice([-1.0, 0.0, 1.0], size=len(mid))
    r = mid.resample(tf).last().pct_change(lookback)
    sig = np.sign(r.reindex(mid.index, method="ffill")).fillna(0.0)
    return (-sig if kind == "trend_fade" else sig).to_numpy()


def daily_first_triggers(direction: np.ndarray, idx: pd.DatetimeIndex) -> np.ndarray:
    """The first bar with a non-zero direction on each local day."""
    day = idx.tz_convert("America/New_York").normalize()
    nz = pd.Series(direction != 0, index=day)
    pos = pd.Series(np.arange(len(direction)), index=day)
    return pos[nz].groupby(level=0).first().to_numpy()


def sweep(m1: pd.DataFrame, direction: np.ndarray, horizon: int, *, tie: str,
          execution: str) -> dict:
    """{(T,S): (p, er)} for one signal, one window, one bracket convention."""
    # use the mid HIGH/LOW, not just closes: a stop that is touched within a bar but
    # closes back is still a stop. Ignoring the intrabar extremes biases every test
    # in favour of the target - the placebo at T=20/S=1 gave 10% where a random walk
    # must give 1/21 = 4.8%.
    mid = (m1["close"] + m1["ask_close"]) / 2.0
    hi = (m1["high"] + m1["ask_high"]) / 2.0
    lo = (m1["low"] + m1["ask_low"]) / 2.0
    spread = float((m1["ask_close"] - m1["close"]).median())
    mh, ml = hi.to_numpy(), lo.to_numpy()
    starts = daily_first_triggers(direction, mid.index)
    starts = starts[starts + horizon < len(mh)]
    # noqa: E501
    if len(starts) < 100:
        return {}
    dirs = direction[starts]
    entry_mid = mid.to_numpy()[starts][:, None]
    base = entry_mid                                       # entry at the mid
    win = np.arange(1, horizon + 1)
    up_path = mh[starts[:, None] + win[None, :]]
    dn_path = ml[starts[:, None] + win[None, :]]
    hi_run = np.maximum.accumulate(up_path, axis=1)        # running mid high since entry
    lo_run = np.minimum.accumulate(dn_path, axis=1)        # running mid low since entry
    # Two separate series, one per side. Using the *high* run for both directions
    # (as the first version did) means the stop is never reached, because a running
    # maximum only moves up.
    long_ = dirs[:, None] > 0
    favourable = np.where(long_, hi_run - base, base - lo_run) / spread
    adverse = np.where(long_, base - lo_run, hi_run - base) / spread

    def first(mask: np.ndarray) -> np.ndarray:
        return np.where(mask.any(axis=1), np.argmax(mask, axis=1), INF)

    t_tgt = {T: first(favourable >= T) for T in TARGETS}
    t_stp = {S: first(adverse >= S) for S in STOPS}
    out = {}
    for T in TARGETS:
        for S in STOPS:
            tt, ts = t_tgt[T], t_stp[S]
            if tie == "stop":
                win = tt < ts
            elif tie == "target":
                win = tt <= ts
            else:
                rng = np.random.default_rng(hash((T, int(S * 10), horizon)) % (2 ** 32))
                coin = rng.random(len(tt)) < 0.5
                win = np.where(tt == ts, coin, tt < ts)
            p = float(np.mean(win))
            cost = COST_PER_ROUND_TRIP[execution]
            out[(T, S)] = (p, p * T - (1 - p) * S - cost)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", default=",".join(ASSETS))
    ap.add_argument("--signals", default="random,trend_continue")
    ap.add_argument("--start-year", type=int, default=2019)
    ap.add_argument("--json", default="artifacts/asymmetric_rr.json")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    kinds = [k.strip() for k in args.signals.split(",") if k.strip()]
    cached = {f.name.split("_1m_")[0] for f in DEFAULT_CACHE.glob("*_1m_bid_*.pkl")}

    rows: list[dict] = []
    for asset in assets:
        if asset not in cached:
            continue
        m1 = load_minute_bid_ask(asset, args.start_year, verbose=False)
        for kind in kinds:
            d = directions(m1, kind)
            for h in HORIZONS:
                for tie in ("stop", "target", "coin"):
                    for ex in ("full", "half"):
                        g = sweep(m1, d, h, tie=tie, execution=ex)
                        for (T, S), (p, er) in g.items():
                            rows.append({"asset": asset, "signal": kind, "minutes": h,
                                         "tie": tie, "execution": ex,
                                         "T": T, "S": S, "p": p, "er": er})
        print(f"  {asset} ok", flush=True)

    df = pd.DataFrame(rows)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(args.json, orient="records")
    print(f"\n### Asymmetrisches RR — {len(df)} Zellen, ein Trigger pro Tag\n")
    base = df[(df.tie == "stop") & (df.execution == "full")]
    print("### P(Ziel vor Stop) und E[R], Konvention 'Stop gewinnt', volle Kosten")
    for h in HORIZONS:
        print(f"\n  --- Horizont {h} Min ---")
        for kind in kinds:
            sub = base[(base.minutes == h) & (base.signal == kind)]
            print(f"    {kind}")
            print(f"      {'Ziel':>5s}" + "".join(f"{('S=' + str(S)):>14s}" for S in STOPS))
            for T in TARGETS:
                cells = ""
                for S in STOPS:
                    v = sub[(sub["T"] == T) & (sub["S"] == S)]
                    cells += f"{v.p.iloc[0]*100:6.1f}%{v.er.iloc[0]:+7.2f}" if len(v) else f"{'-':>13s}"
                print(f"      {T:5d}" + cells)

    print("\n### Robustheit der positiven Zellen gegen die Kontrollachsen")
    pos = df[df.er > 0]
    print(f"    Zellen mit E[R] > 0 insgesamt: {len(pos)} von {len(df)}")
    if not pos.empty:
        key = ["asset", "signal", "minutes", "T", "S"]
        robust = pos.groupby(key).size().reset_index(name="n_conventions")
        full = robust[robust.n_conventions == 6].sort_values(key)
        print(f"    robust in ALLEN 6 Konventionen (3 Tie-Breaks x 2 Ausfuehrungen): "
              f"{len(full)} von {robust.shape[0]} positiven Kombinationen")
        if not full.empty:
            print(full.to_string(index=False))
    print("\n### Schlaegt trend_continue den Placebo? (Vorsprung E[R], je Konvention)")
    for tie in ("stop", "target", "coin"):
        for ex in ("full", "half"):
            piv = df[(df.tie == tie) & (df.execution == ex)].pivot_table(
                index=["asset", "minutes", "T", "S"], columns="signal", values="er")
            if "trend_continue" not in piv or "random" not in piv:
                continue
            w = (piv["trend_continue"] - piv["random"]).dropna()
            print(f"    tie={tie:6s} exec={ex:4s}: besser in {int((w>0).sum()):4d}/{len(w):4d} "
                  f"({100*(w>0).mean():4.0f}%)  Median {w.median():+.3f}")
    print(f"\n    Detail -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
