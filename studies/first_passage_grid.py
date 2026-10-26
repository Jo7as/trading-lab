#!/usr/bin/env python3
"""First-passage grid: the ex-ante economics of a signal, over everything.

This is the test the MFE table cannot give. MFE says how far price ever travels;
it does not say whether the target arrives *before* the stop, and a signal with a
big MFE but an equally big MAE looks impressive there and is worthless in
practice (exactly what the pullback turned out to be).

So: for every signal, market, session and holding window, walk the real 1-minute
path from each trigger and record the **first** time the target and the first time
the stop is reached. From that, for every (target, stop) pair in spreads:

    p    = P(target first)          (stop wins ties - the repo's convention)
    E[R] = p*T - (1-p)*S - cost     cost is per sample, from the realized exit:
                                    a target hit is a resting (limit) exit = 0.0,
                                    a stop or horizon exit is a market exit = 0.5,
                                    with a limit entry (``round_trip_spread``).

The random-direction placebo is computed on identical bars, so the comparison is
like-for-like: a signal only counts if it beats its own placebo in the same row.

Vectorisation: one (starts x horizon) path matrix per (asset, session, horizon),
then an argmax per level. A loop per (T,S) pair would be ~20x slower.

  .venv/bin/python studies/first_passage_grid.py
  .venv/bin/python studies/first_passage_grid.py --assets US100,GER40 --stride 5
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
from trading_lab.data_audit import build_study_frame  # noqa: E402
from trading_lab.cost_gate import round_trip_spread  # noqa: E402

ASSETS = ("US100", "US30", "SP500", "GER40", "XAUUSD", "EURUSD", "GBPUSD", "USDJPY")
HORIZONS = (15, 30, 60)
TARGETS = (1, 2, 3, 5, 8)
STOPS = (2, 3, 5, 8)
SIGNALS = ("random", "trend_continue", "trend_fade", "pullback")
INF = 10 ** 9


def direction_series(m1: pd.DataFrame, kind: str, *, tf: str = "60min",
                     lookback: int = 6, session: str = "ny") -> np.ndarray:
    mid = (m1["close"] + m1["ask_close"]) / 2.0
    if kind == "random":
        rng = np.random.default_rng(12345)
        return rng.choice([-1.0, 0.0, 1.0], size=len(mid))
    if kind == "pullback":
        from trading_lab.strategies.pullback import PullbackConfig, signals
        f = signals(m1, PullbackConfig(session=session))
        s = pd.Series(f["setup"].to_numpy(), index=f.index)
        return s.reindex(mid.index).fillna(0.0).to_numpy()
    tf_ret = mid.resample(tf).last().pct_change(lookback)
    sig = np.sign(tf_ret.reindex(mid.index, method="ffill")).fillna(0.0)
    return (-sig if kind == "trend_fade" else sig).to_numpy()


def passage_grid(m1: pd.DataFrame, direction: np.ndarray, horizon: int, *,
                 stride: int) -> dict[tuple[int, int], tuple[float, float]]:
    """{(T,S): (p, E[R])} for one signal and one holding window."""
    mid = (m1["close"] + m1["ask_close"]) / 2.0
    spread = float((m1["ask_close"] - m1["close"]).median())
    m = mid.to_numpy()
    starts = np.flatnonzero(direction != 0)
    starts = starts[starts + horizon < len(m)]
    if len(starts) == 0:
        return {}
    if stride > 1:
        starts = starts[::stride]
    # path matrix: favourable move (positive = our direction) in spreads
    base = m[starts][:, None]
    path = m[starts[:, None] + np.arange(1, horizon + 1)[None, :]]
    signed = (path - base) / spread * direction[starts][:, None]
    # first passage per level. Target = first time signed >= T (in our favour);
    # stop = first time signed <= -S (against us). Getting these two comparisons
    # the same way round is the whole bug this function exists to avoid.
    def first_time(mask: np.ndarray) -> np.ndarray:
        return np.where(mask.any(axis=1), np.argmax(mask, axis=1), INF)
    t_target = {T: first_time(signed >= T) for T in TARGETS}
    t_stop = {S: first_time(signed <= -S) for S in STOPS}
    out: dict[tuple[int, int], tuple[float, float]] = {}
    for T in TARGETS:
        for S in STOPS:
            win = t_target[T] < t_stop[S]                  # stop wins a tie
            p = float(np.mean(win))
            # limit entry: a winner exits at its resting target (limit, 0.0), a
            # loser through its stop / the holding horizon (market/time, 0.5).
            cost = np.where(win, round_trip_spread("limit", "target"),
                            round_trip_spread("limit", "time"))
            er = float(np.mean(np.where(win, float(T), -float(S)) - cost))
            out[(T, S)] = (p, er)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", default=",".join(ASSETS))
    ap.add_argument("--sessions", default="ny,london")
    ap.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS))
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--start-year", type=int, default=2019)
    ap.add_argument("--json", default="artifacts/first_passage_grid.json")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    horizons = [int(h) for h in args.horizons.split(",")]
    cached = {f.name.split("_1m_")[0] for f in DEFAULT_CACHE.glob("*_1m_bid_*.pkl")}

    rows: list[dict] = []
    for asset in assets:
        if asset not in cached:
            continue
        m1, info = build_study_frame(load_minute_bid_ask(asset, args.start_year, verbose=False), asset)
        dirs = {(s, k): direction_series(m1, k, session=s)
                for s in sessions for k in SIGNALS}
        grids: dict = {}
        for session in sessions:
            for h in horizons:
                for kind in SIGNALS:
                    g = passage_grid(m1, dirs[(session, kind)], h, stride=args.stride)
                    grids[(session, h, kind)] = g
                    for (T, S), (p, er) in g.items():
                        rows.append({"asset": asset, "session": session, "minutes": h,
                                     "signal": kind, "T": T, "S": S,
                                     "p": p, "er": er})
        print(f"  {asset} ok ({len(dirs)} signal series; audit kept {info['kept']}, "
              f"dropped {info['dropped']}, thin days {info['thin_days']})", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        print("keine Ergebnisse")
        return 1
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(args.json, orient="records")

    print(f"\n### First-Passage-Gitter: P(Ziel vor Stop) und E[R] nach Limit-Einstieg "
          f"(Ziel {round_trip_spread('limit', 'target'):.1f}, Stop/Zeit "
          f"{round_trip_spread('limit', 'time'):.1f} x Spread)")
    print("    T = Ziel in Spreads, S = Stop in Spreads. 'random' ist der Placebo")
    print("    auf denselben Bars - ein Signal zaehlt nur, wenn es ihn schlaegt.\n")
    for metric, label in (("p", "Trefferquote"), ("er", "E[R]")):
        print(f"  === {label} ===")
        for session in sessions:
            for h in horizons:
                sub = df[(df.session == session) & (df.minutes == h)]
                # best cell per signal, and whether it beats its placebo
                print(f"\n    --- {session} / {h} Min ---")
                print(f"    {'signal':16s}" + "".join(f"{'T' + str(T) + '/S' + str(S):>9s}"
                                                     for T in TARGETS for S in STOPS))
                for kind in SIGNALS:
                    line = ""
                    for T in TARGETS:
                        for S in STOPS:
                            v = sub[(sub.signal == kind) & (sub.T == T) & (sub.S == S)][metric]
                            line += f"{v.iloc[0]:9.2f}" if len(v) else f"{'-':>9s}"
                    print(f"    {kind:16s}{line}")
        print()
    # the question that matters: does any signal beat its placebo, per cell?
    piv = df.pivot_table(index=["asset", "session", "minutes", "T", "S"],
                         columns="signal", values="er")
    if "random" in piv.columns:
        for kind in ("trend_continue", "trend_fade", "pullback"):
            if kind not in piv.columns:
                continue
            win = piv[kind] - piv["random"]
            n_pos = int((win > 0).sum()); n_tot = int(win.notna().sum())
            print(f"    {kind:16s} schlaegt den Placebo in {n_pos} von {n_tot} Zellen "
                  f"({100*n_pos/max(n_tot,1):.0f}%), Median-Vorsprung {win.median():+.2f} Spreads")
    print("\n    Zellen mit E[R] > 0 UND Vorsprung gegen den Placebo:")
    pos = df[df.er > 0]
    print(f"      {len(pos)} von {len(df)} Zellen haben ueberhaupt E[R] > 0 "
          f"(Zufallserwartung bei {len(df)} Zellen: kein struktureller Grund fuer 0)")
    if not pos.empty:
        print(pos.sort_values("er", ascending=False).head(15).round(3).to_string(index=False))
    print(f"\n    Detail -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
