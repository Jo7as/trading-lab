#!/usr/bin/env python3
"""H-010 fill rate and adverse selection for the pullback limit entry.

``pullback.run_backtest`` assumes a resting limit order fills whenever the market
touches its price. Two things it deliberately does *not* model:

* the **fill rate** - how often the touch actually happens inside the wait window
  (``limit_wait_minutes``) before the session ends, and
* the **adverse selection** - the fills you *do* get are the ones where price ran
  through your level, which may be exactly the losers.

This script measures both, on exactly the setups H-010 trades. The setup is built
with the module helpers (``session_frame`` / ``signals``) and the bracket (stop at
the retrace extreme minus ``stop_buffer`` sigma, target ``target_r`` times the
risk) is read from ``PullbackConfig`` - nothing is re-derived by hand, so the
numbers stay comparable with the backtest. For filled signals the realised R is
taken straight from ``pullback._resolve``.

Convention - the side of the book each order really trades at, identical to
``_resolve``:

* long  resting limit fills when ``ask_low <= level``;
* short resting limit fills when ``bid_high >= level``.

Boundary: this measures fill **rate** (touch / no touch) only. Queue position and
partial fills are *not* modelled - a touched limit is assumed to fill in full at
its own price, exactly as the backtest does. The fill rate here is therefore an
upper bound on execution quality, and the adverse-selection gap a lower bound on
how much of it is real.

  .venv/bin/python studies/h010_fill_rate.py
  .venv/bin/python studies/h010_fill_rate.py --assets US100 --years 2024
"""
from __future__ import annotations

import argparse
import sys
import warnings

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_lab.minute_data import DEFAULT_CACHE, load_minute_bid_ask  # noqa: E402
from trading_lab.strategies.pullback import (  # noqa: E402
    PullbackConfig,
    _resolve,
    session_frame,
    signals,
)

# Defaults mirror studies/h010_variants.py (its assets, sessions, start year) and
# the PullbackConfig gate defaults (trend_min / pullback_min).
ASSETS = "US100,SP500,GER40,XAUUSD,EURUSD,USDJPY"
SESSIONS = "ny,london"
START_YEAR = 2019
TREND_MIN = 0.5
PULLBACK_MIN = 0.30

# R difference (filled minus counterfactual unfilled) below which the filled set
# is called materially worse. 0.10 R is ~ a tenth of a trade's risk.
ADVERSE_R_DIFF = 0.10


def _day_starts(days: np.ndarray) -> dict:
    """First positional index of every day in the (day-sorted) signal frame."""
    starts: dict = {}
    prev = None
    for idx, d in enumerate(days):
        if d != prev:
            starts[d] = idx
            prev = d
    return starts


def _bracket(f: pd.DataFrame, i: int, cfg: PullbackConfig, starts: dict
             ) -> tuple[int, bool, float, float, float, object]:
    """(direction, is_long, level, stop, risk, day) - stop/risk exactly as _resolve."""
    row = f.iloc[i]
    direction = int(row["setup"])
    long_ = direction > 0
    entry = float(row["mid"])
    sigma_abs = float(row["sigma"]) * float(row["anchor"])
    day = row["day"]
    before = f.iloc[starts[day]:i + 1]
    if long_:
        extreme = float(before["bid_low"].min())      # low a resting buy could reach
        stop = extreme - cfg.stop_buffer * sigma_abs
    else:
        extreme = float(before["ask_high"].max())     # high a resting sell could reach
        stop = extreme + cfg.stop_buffer * sigma_abs
    return direction, long_, entry, stop, abs(entry - stop), day


def _find_fill(f: pd.DataFrame, i: int, entry: float, long_: bool, day,
               cfg: PullbackConfig) -> int | None:
    """First bar inside the wait window that touches the resting limit, else None."""
    for k in range(i + 1, min(i + 1 + cfg.limit_wait_minutes, len(f))):
        bar = f.iloc[k]
        if bar["day"] != day:
            break
        touched = float(bar["ask_low"]) <= entry if long_ else float(bar["bid_high"]) >= entry
        if touched:
            return k
    return None


def _excursion(f: pd.DataFrame, mid: np.ndarray, j: int, direction: int,
               entry: float, spread: float, cfg: PullbackConfig) -> tuple[float, float]:
    """(MFE, MAE) in spreads from the fill price, over max_hold_minutes within the day."""
    day = f.iloc[j]["day"]
    end = j
    last = min(j + cfg.max_hold_minutes, len(f) - 1)
    for k in range(j, last + 1):
        if f.iloc[k]["day"] != day:
            break
        end = k
    seg = mid[j:end + 1]
    if direction > 0:
        return (seg.max() - entry) / spread, (entry - seg.min()) / spread
    return (entry - seg.min()) / spread, (seg.max() - entry) / spread


def _counterfactual(f: pd.DataFrame, i: int, direction: int, long_: bool,
                    stop: float, target: float, risk: float,
                    cfg: PullbackConfig) -> dict | None:
    """R of a market entry at the end of the fill window, same bracket.

    Stop/target levels are the ones the filled bracket would have used; only the
    entry moves, to the mid of the last bar of the (failed) fill window. Only a
    comparison number - the unfilled signals are never traded.
    """
    day = f.iloc[i]["day"]
    j = i
    for k in range(i + 1, min(i + 1 + cfg.limit_wait_minutes, len(f))):
        if f.iloc[k]["day"] != day:
            break
        j = k
    if j <= i:
        return None
    entry = float(f.iloc[j]["mid"])
    end = min(j + cfg.max_hold_minutes, len(f) - 1)
    exit_px, reason = None, "time"
    for k in range(j, end + 1):
        bar = f.iloc[k]
        if bar["day"] != day:
            k -= 1
            bar = f.iloc[k]
            exit_px, reason = float(bar["bid"] if long_ else bar["ask"]), "session"
            break
        adverse = float(bar["bid_low"] if long_ else bar["ask_high"])
        favourable = float(bar["bid_high"] if long_ else bar["ask_low"])
        if adverse <= stop if long_ else adverse >= stop:
            exit_px, reason = stop, "stop"
            break
        if favourable >= target if long_ else favourable <= target:
            exit_px, reason = target, "target"
            break
    else:
        last = f.iloc[end]
        exit_px, reason = float(last["bid"] if long_ else last["ask"]), "time"
    if exit_px is None:
        return None
    return {"r": direction * (exit_px - entry) / risk, "exit_reason": reason,
            "entry": entry}


def measure(asset: str, session: str, m1: pd.DataFrame, cfg: PullbackConfig) -> dict | None:
    """Fill rate + adverse selection for one (asset, session)."""
    f = session_frame(m1, cfg)     # full session frame (all bars) - spread reference
    s = signals(m1, cfg)           # the signal frame run_backtest uses
    spread = float((f["ask"] - f["bid"]).median())
    if not np.isfinite(spread) or spread <= 0:
        return None

    mid = s["mid"].to_numpy()
    setup = s["setup"].to_numpy()
    days = s["day"].to_numpy()
    starts = _day_starts(days)

    # one setup per day: the first trigger, as run_backtest(one_per_day=True) reads them
    seen: set = set()
    first: list[int] = []
    for i in range(len(s)):
        if setup[i] == 0 or days[i] in seen:
            continue
        seen.add(days[i])
        first.append(i)

    bars_fill: list[int] = []
    adv: list[float] = []
    mfe: list[float] = []
    mae: list[float] = []
    r_fill: list[float] = []
    r_cf: list[float] = []
    n = 0
    for i in first:
        # _resolve needs the signal frame (s), not the bare session frame: it reads
        # the setup/trend/retrace columns that only signals() adds.
        direction, long_, entry, stop, risk, day = _bracket(s, i, cfg, starts)
        if risk <= 0:
            continue
        cost = float(s.iloc[i]["ask"] - s.iloc[i]["bid"])
        if cost / risk > cfg.max_cost_r:          # same guard as _resolve
            continue
        n += 1
        target = entry + direction * cfg.target_r * risk
        j = _find_fill(s, i, entry, long_, day, cfg)
        if j is not None:
            bars_fill.append(j - i)
            seg = mid[i:j + 1]
            adv.append((entry - seg.min()) / spread if long_
                       else (seg.max() - entry) / spread)
            m, x = _excursion(s, mid, j, direction, entry, spread, cfg)
            mfe.append(m)
            mae.append(x)
            t = _resolve(s, i, cfg)
            r_fill.append(t["r"] if t is not None else np.nan)
        else:
            cf = _counterfactual(s, i, direction, long_, stop, target, risk, cfg)
            if cf is not None:
                r_cf.append(cf["r"])

    if n == 0:
        return None
    n_filled = len(bars_fill)
    r_fill_arr = np.array([x for x in r_fill if np.isfinite(x)])
    r_cf_arr = np.array(r_cf)
    mean_fill = float(r_fill_arr.mean()) if len(r_fill_arr) else float("nan")
    mean_cf = float(r_cf_arr.mean()) if len(r_cf_arr) else float("nan")
    diff = mean_fill - mean_cf if np.isfinite(mean_fill) and np.isfinite(mean_cf) else float("nan")
    return {
        "n": n, "n_filled": n_filled,
        "fill_rate": 100.0 * n_filled / n,
        "bars": float(np.median(bars_fill)) if bars_fill else float("nan"),
        "adv": float(np.median(adv)) if adv else float("nan"),
        "mfe": float(np.median(mfe)) if mfe else float("nan"),
        "mae": float(np.median(mae)) if mae else float("nan"),
        "r_fill": mean_fill, "r_cf": mean_cf, "r_diff": diff,
        "adverse": bool(np.isfinite(diff) and diff < -ADVERSE_R_DIFF),
    }


def _fmt(x: float, nd: int, sign: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", default=ASSETS,
                    help="Komma-Liste (Default wie h010_variants)")
    ap.add_argument("--session", default=SESSIONS,
                    help="Komma-Liste (Default wie h010_variants)")
    ap.add_argument("--years", type=int, default=START_YEAR,
                    help="Startjahr, Daten ab diesem Jahr (Default wie h010_variants)")
    ap.add_argument("--trend-min", type=float, default=TREND_MIN)
    ap.add_argument("--pullback-min", type=float, default=PULLBACK_MIN)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    sessions = [s.strip() for s in args.session.split(",") if s.strip()]
    cached = {f.name.split("_1m_")[0] for f in DEFAULT_CACHE.glob("*_1m_bid_*.pkl")}

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("### H-010 Fill-Rate & Adverse Selection (Limit-Einstieg, erstes Setup je Tag)")
    emit("    Buchseite: Long-Limit fuellt bei ask_low<=Level, Short bei bid_high>=Level.")
    emit("    advMove = Gegenlauf des Mids vor dem Fill (Spreads); MFE/MAE ab Fill-Preis.")
    emit("    R_fill = realisiertes R (pullback._resolve); R_nofill = kontrafaktischer")
    emit("    Market-Einstieg am Fills-Fensterende, gleiches Bracket, nur Vergleichszahl.")
    emit("    Grenze: nur Touch-Fill-Rate, keine Queue-Position/Teilfills -> Obergrenze.")
    emit("")
    emit(f"    {'session':8s}{'asset':9s}{'n':>6s}{'fill%':>8s}{'bars':>7s}"
         f"{'advMove':>9s}{'MFE':>8s}{'MAE':>8s}{'R_fill':>9s}{'R_nofill':>10s}"
         f"{'R_diff':>8s}  flag")

    frames = {}
    for asset in assets:
        if asset not in cached:
            continue
        m1 = load_minute_bid_ask(asset, args.years, verbose=False)
        if not m1.empty:
            frames[asset] = m1

    cells = 0
    flagged = 0
    for session in sessions:
        for asset, m1 in frames.items():
            cfg = PullbackConfig(asset=asset, session=session,
                                 trend_min=args.trend_min, pullback_min=args.pullback_min)
            r = measure(asset, session, m1, cfg)
            if r is None:
                continue
            cells += 1
            if r["adverse"]:
                flagged += 1
            flag = "ADVERSE SELECTION" if r["adverse"] else ""
            emit(f"    {session:8s}{asset:9s}{r['n']:6d}{r['fill_rate']:7.1f}%"
                 f"{_fmt(r['bars'], 1):>7s}{_fmt(r['adv'], 2):>9s}"
                 f"{_fmt(r['mfe'], 2):>8s}{_fmt(r['mae'], 2):>8s}"
                 f"{_fmt(r['r_fill'], 2, True):>9s}{_fmt(r['r_cf'], 2, True):>10s}"
                 f"{_fmt(r['r_diff'], 2, True):>8s}  {flag}")
        emit("")

    emit(f"    Zellen: {cells}, davon Adverse Selection: {flagged}")
    emit("    Lesart: R_fill < R_nofill - 0.10 -> die gefuellten Signale sind materiell")
    emit("    schlechter als die nicht gefuellten (Adverse Selection). Die Fill-Annahme")
    emit("    des Backtests ueberschaetzt dann die Qualitaet der tatsaechlich gefuellten Trades.")

    out = "artifacts/h010_fill_rate.txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n    -> Report: {out}")
    return 0 if cells else 1


if __name__ == "__main__":
    raise SystemExit(main())
