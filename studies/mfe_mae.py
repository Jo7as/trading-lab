#!/usr/bin/env python3
"""MFE/MAE: the ex-ante economics of a signal, independent of any exit model.

The S0 gate compares the spread with ``E|move|`` over a holding time. That is the
wrong quantity, and it is why the gate could only ever say "maybe": a trade never
needs to capture the average *absolute* move, only the move **in its own
direction**, and only as far as its target. The right measure is the one an
execution desk uses:

    MFE(direction, H) = max over (t, t+H] of  dir * (mid - mid_t)   # best excursion
    MAE(direction, H) = max over (t, t+H] of -dir * (mid - mid_t)   # worst excursion

expressed **in units of the spread**. That is exit-model free: once the
distribution of MFE is known you can price *any* (target, stop) pair, and reading
off "how often does price run k spreads our way" is exactly the question of how
much a signal can pay.

What it answers:

* ``P(MFE >= k)`` per k - the reachable-target curve, in spread units.
* ``P(target first | stop)`` for a grid of (T, S) - the actual hit rate of a
  1:k setup, from which E[R] = p*k - (1-p) follows directly.
* the *break-even* target: the k at which p*k - (1-p) - cost crosses zero.

Costs follow the single convention in ``trading_lab.cost_gate``, in units of the
FULL spread:

    entry -> exit   limit / TP        market / stop / time
    limit               0.0                    0.5
    market              0.5                    1.0

A resting (limit) entry with a limit/target exit is free in this model; a resting
entry whose exit crosses the book (stop/time) costs 0.5; a both-legs market trade
costs 1.0. The MFE/MAE table is quoted in spread units, so it is read under
either entry assumption. The first-passage grid below prices the (limit entry,
market/time exit) level = ``round_trip_spread("limit", "market")`` = 0.5 x spread,
the conservative side because a trade that does not reach its target exits through
its stop.

  .venv/bin/python studies/mfe_mae.py
  .venv/bin/python studies/mfe_mae.py --assets US100,GER40 --session ny
"""
from __future__ import annotations

import argparse
import sys
import warnings

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_lab.minute_data import DEFAULT_CACHE, load_minute_bid_ask  # noqa: E402
from trading_lab.data_audit import build_study_frame  # noqa: E402
from trading_lab.cost_gate import round_trip_spread  # noqa: E402

# The signal generators we want to price. Each returns, per bar, a direction
# (+1/-1/0) and nothing else - the exits are what this script measures.
SIGNALS = ("random", "trend_continue", "trend_fade", "pullback")


def direction_series(m1: pd.DataFrame, kind: str, *, tf: str = "60min",
                     lookback: int = 6, session: str = "ny") -> pd.Series:
    """A direction per bar on the 1-minute grid, computed from data up to the bar."""
    mid = (m1["close"] + m1["ask_close"]) / 2.0
    r = mid.pct_change()
    if kind == "random":
        rng = np.random.default_rng(11)
        return pd.Series(rng.choice([-1.0, 0.0, 1.0], size=len(mid)), index=mid.index)
    if kind == "trend_continue":
        # sign of the return over the last `lookback` bars of `tf`
        tf_ret = mid.resample(tf).last().pct_change(lookback)
        sig = np.sign(tf_ret.reindex(mid.index, method="ffill"))
        return sig.reindex(mid.index).fillna(0.0)
    if kind == "trend_fade":
        tf_ret = mid.resample(tf).last().pct_change(lookback)
        sig = -np.sign(tf_ret.reindex(mid.index, method="ffill"))
        return sig.reindex(mid.index).fillna(0.0)
    if kind == "pullback":
        from trading_lab.strategies.pullback import PullbackConfig, signals
        f = signals(m1, PullbackConfig(session=session))
        s = pd.Series(f["setup"].to_numpy(), index=f.index)
        return s.reindex(mid.index).fillna(0.0)     # back onto the full 1m grid
    raise ValueError(kind)


def mfe_mae(m1: pd.DataFrame, direction: pd.Series, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """MFE/MAE in spread units, for every bar where direction != 0."""
    mid = (m1["close"] + m1["ask_close"]) / 2.0
    # the spread must be in the SAME units as the price difference below,
    # i.e. absolute, not relative - otherwise indices are off by the price level
    spread = float((m1["ask_close"] - m1["close"]).median())
    d = direction.to_numpy()
    m = mid.to_numpy()
    # forward running max/min over the next `horizon` bars: reverse, roll, reverse
    s = pd.Series(m)
    fwd_max = s[::-1].rolling(horizon + 1, min_periods=1).max()[::-1].to_numpy()
    fwd_min = s[::-1].rolling(horizon + 1, min_periods=1).min()[::-1].to_numpy()
    up = (fwd_max - m) / spread           # best move up, in spreads
    dn = (m - fwd_min) / spread           # best move down, in spreads
    ok = (d != 0) & np.isfinite(up) & np.isfinite(dn)
    mfe = np.where(d[ok] > 0, up[ok], dn[ok])
    mae = np.where(d[ok] > 0, dn[ok], up[ok])
    return mfe, mae


def first_passage(m1: pd.DataFrame, direction: pd.Series, horizon: int,
                  T: float, S: float, *, stride: int = 5) -> float:
    """P(target T hit before stop S) by walking the actual 1-minute path.

    MFE/MAE alone cannot answer this: ``mfe >= T and mae < S`` counts only the
    paths where the stop was *never* touched, and ``mfe >= T and mae >= S`` still
    happens when the target came first. The order of the two events needs the
    path, so this walks it (every ``stride``-th bar as a start point, which keeps
    the estimate unbiased and the cost sane). Ties inside one bar go to the stop,
    the conservative reading used everywhere else in this repo.
    """
    mid = (m1["close"] + m1["ask_close"]) / 2.0
    spread = float((m1["ask_close"] - m1["close"]).median())
    m = mid.to_numpy()
    d = direction.to_numpy()
    starts = np.flatnonzero(d != 0)
    starts = starts[starts + horizon < len(m)]
    if stride > 1:
        starts = starts[::stride]
    up_lvl, dn_lvl = T * spread, S * spread
    hit = 0
    for i in starts:
        base = m[i]
        seg = m[i + 1: i + horizon + 1]
        if d[i] > 0:
            tgt, stp = seg >= base + up_lvl, seg <= base - dn_lvl
        else:
            tgt, stp = seg <= base - up_lvl, seg >= base + dn_lvl
        it = np.argmax(tgt) if tgt.any() else 10 ** 9
        is_ = np.argmax(stp) if stp.any() else 10 ** 9
        hit += it < is_                                  # stop wins a tie
    return hit / len(starts) if len(starts) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", default="US100,GER40,XAUUSD,EURUSD")
    ap.add_argument("--session", default="ny", choices=("london", "ny"))
    ap.add_argument("--horizon", type=int, default=60,
                    help="forward window in 1m bars; beyond ~120 min BOTH extremes "
                         "are almost always touched and the table stops discriminating")
    ap.add_argument("--start-year", type=int, default=2019)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    cached = {f.name.split("_1m_")[0] for f in DEFAULT_CACHE.glob("*_1m_bid_*.pkl")}

    K = (0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20)
    print(f"### MFE in Spread-Einheiten, Horizont {args.horizon} Min, Session {args.session}")
    print("    'random' ist der Placebo-Balken: so weit laeuft der Kurs auch ohne Signal.")
    print("    P(MFE>=k) = Anteil der Bars, in denen der Kurs k Spreads in Signalrichtung laeuft.")
    print(f"    Kosten: Limit-Einstieg + Market/Zeit-Exit = "
          f"{round_trip_spread('limit', 'market'):.1f} x Spread; "
          f"Market/Market = {round_trip_spread('market', 'market'):.1f} x Spread.\n")
    for asset in assets:
        if asset not in cached:
            continue
        m1, info = build_study_frame(load_minute_bid_ask(asset, args.start_year, verbose=False), asset)
        print(f"  --- {asset} ---  (audit: kept {info['kept']}, dropped {info['dropped']}, "
              f"thin days {info['thin_days']})")
        print(f"    {'signal':16s}" + "".join(f"{('>=' + str(k)):>8s}" for k in K)
              + f"{'n':>10s}{'MFE med':>10s}")
        out = {}
        for kind in SIGNALS:
            d = direction_series(m1, kind, session=args.session)
            mfe, mae = mfe_mae(m1, d, args.horizon)
            if len(mfe) < 200:
                continue
            out[kind] = (mfe, mae)
            probs = [float((mfe >= k).mean()) for k in K]
            print(f"    {kind:16s}" + "".join(f"{p*100:7.1f}%" for p in probs)
                  + f"{len(mfe):10d}{np.median(mfe):10.2f}")
        # first-passage grid: the actual order of target and stop on the 1m path
        for kind in ("random", "trend_continue", "pullback"):
            if kind not in out:
                continue
            d = direction_series(m1, kind, session=args.session)
            print(f"\n    {kind}: P(Ziel vor Stop) aus dem 1m-Pfad "
                  f"(Stop in Spread-Einheiten), "
                  f"E[R] nach {round_trip_spread('limit', 'market'):.1f} Spread "
                  f"(Limit-Einstieg + Market/Zeit-Exit)")
            print(f"      {'Ziel':>6s}" + "".join(f"{('S=' + str(s)):>12s}" for s in (2, 3, 5, 8)))
            for T in (1, 2, 3, 5, 8):
                cells = ""
                for S in (2, 3, 5, 8):
                    p = first_passage(m1, d, args.horizon, T, S)
                    er = p * T - (1 - p) * S - round_trip_spread("limit", "market")
                    cells += f"{p*100:6.1f}%{er:+6.2f}"
                print(f"      {T:6.1f}" + cells)
            print("      (Zelle: Trefferquote und E[R] in Spread-Einheiten; positiv = handelbar)")
        print()
    print("    Lesart: E[R] > 0 heisst, dass dieses (Ziel, Stop)-Paar nach Kosten traegt,")
    print("    ohne dass eine Haltedauer oder ein Exit-Modell angenommen werden musste.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
