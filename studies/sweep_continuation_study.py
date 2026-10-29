#!/usr/bin/env python3
"""Step A for S-003: does price CONTINUE after a sweep? A measurement, not a strategy.

Claim to check (Mahadzva, SSRN 10.2139/ssrn.7430998, abstract): a swept level tends to
predict continuation rather than the textbook reversal. Our own definitions (the paper's
exact ones were not accessible):

Events on the bar's CLOSE, three level types from ``trading_lab.market_structure``:
  * ext  - the external swing target (the last unbroken external swing high/low)
  * int  - the internal swing target
  * pool - an EQH/EQL pool (either level)
and two kinds:
  * reclaim - wick beyond the level, close back inside (the classic "sweep")
  * through - close beyond the level (BOS/CHoCH for swings, "taken" for pools)

Measured: forward move IN THE SWEEP DIRECTION after h bars, in ATR units, minus the
unconditional drift of all bars in that direction (excess). Split by whether the sweep
went WITH the structure trend of that timeframe (prior bar) or AGAINST it. Overlapping
windows inflate t, so events are thinned to one per h bars per group. Each row also shows
the round-trip cost in ATR (ftmo profile) - the hurdle a strategy would have to clear.

TRAIN ONLY (2019-2024).

  .venv/bin/python studies/sweep_continuation_study.py
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
from trading_lab.indicators import atr as _atr  # noqa: E402
from trading_lab.market_structure import StructureConfig, analyze, mid_ohlc  # noqa: E402
from trading_lab.mr_common import resample_ohlc  # noqa: E402
from trading_lab.registry import HOLDOUT_START, load_cached  # noqa: E402

MARKETS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "US100", "SP500", "GER40", "XAUUSD"]
TFS = ["15min", "1h"]
HORIZONS = [1, 4, 12, 24]
OUT = Path("artifacts/sweep_continuation.json")


def events(b: pd.DataFrame) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """(level, kind) -> (bar indices, direction +1 for a swept high / -1 for a swept low)."""
    def pick(up: np.ndarray, dn: np.ndarray):
        iu, idn = np.flatnonzero(up), np.flatnonzero(dn)
        return np.r_[iu, idn], np.r_[np.ones(len(iu), int), -np.ones(len(idn), int)]
    ev = {}
    for lv in ("ext", "int"):
        e = b[f"{lv}_event"].to_numpy()
        ev[(lv, "reclaim")] = pick(b[f"{lv}_swing_sweep_high"].to_numpy() == 1, b[f"{lv}_swing_sweep_low"].to_numpy() == 1)
        ev[(lv, "through")] = pick(e > 0, e < 0)
    ph = (b["ext_eqh_sweep"].to_numpy() == 1) | (b["int_eqh_sweep"].to_numpy() == 1)
    pl = (b["ext_eql_sweep"].to_numpy() == 1) | (b["int_eql_sweep"].to_numpy() == 1)
    th = (b["ext_eqh_taken"].to_numpy() == 1) | (b["int_eqh_taken"].to_numpy() == 1)
    tl = (b["ext_eql_taken"].to_numpy() == 1) | (b["int_eql_taken"].to_numpy() == 1)
    ev[("pool", "reclaim")] = pick(ph, pl)
    ev[("pool", "through")] = pick(th, tl)
    return ev


def thin(idx: np.ndarray, gap: int) -> np.ndarray:
    """Keep events at least ``gap`` bars apart (sorted), so forward windows do not overlap."""
    keep, last = [], -10**9
    for i in np.sort(idx):
        if i - last >= gap:
            keep.append(i)
            last = i
    return np.array(keep, dtype=int)


def study_one(asset: str, tf: str) -> list[dict]:
    m1 = load_cached(asset)
    m1 = m1[m1.index < HOLDOUT_START]
    if m1.empty:
        return []
    df = mid_ohlc(resample_ohlc(m1, tf))
    res = analyze(df, StructureConfig())
    b = res.bars
    c = df["close"].to_numpy()
    a = _atr(df["high"], df["low"], df["close"], 14).to_numpy()
    trend_prev = np.r_[0, b["ext_trend"].to_numpy()[:-1]]          # structure trend BEFORE the event bar
    # round-trip cost in ATR: median liquid spread (ftmo profile) + 2 commissions, over median ATR
    raw = resample_ohlc(m1, tf)
    sp = float(np.nanmedian((raw["ask_close"] - raw["close"]).to_numpy()))
    px = float(np.nanmedian(c))
    cost = sp + 2 * costs.commission_bp_side(asset, px, "ftmo") * 1e-4 * px
    cost_atr = cost / float(np.nanmedian(a))
    rows = []
    n = len(c)
    for h in HORIZONS:
        fwd = np.full(n, np.nan)
        fwd[:-h] = (c[h:] - c[:-h]) / a[:-h]
        base = {d: float(np.nanmean(d * fwd)) for d in (1, -1)}
        for (lv, kind), (idx, dirn) in events(b).items():
            for align in ("mit", "gegen"):
                sel = (trend_prev[idx] == dirn) if align == "mit" else (trend_prev[idx] == -dirn)
                ii, dd = idx[sel], dirn[sel]
                ok = ii < n - h
                ii, dd = ii[ok], dd[ok]
                if len(ii) == 0:
                    continue
                keep = np.isin(ii, thin(ii, h))
                ii, dd = ii[keep], dd[keep]
                x = dd * fwd[ii] - np.array([base[int(d)] for d in dd])
                x = x[np.isfinite(x)]
                if len(x) < 30:
                    continue
                rows.append({"asset": asset, "tf": tf, "h": h, "level": lv, "kind": kind, "trend": align,
                             "n": int(len(x)), "excess_atr": float(x.mean()), "t": float(screening.t_stat(x)),
                             "hit": float((x + np.array([base[int(d)] for d in dd])[: len(x)] > 0).mean()),
                             "cost_atr": cost_atr})
    return rows


def main() -> int:
    warnings.filterwarnings("ignore")
    rows = []
    for asset in MARKETS:
        for tf in TFS:
            r = study_one(asset, tf)
            rows += r
            print(f"  {asset:7s} {tf:6s} {len(r)} Zellen", flush=True)
    df = pd.DataFrame(rows)
    OUT.write_text(json.dumps(rows, indent=1))
    pd.set_option("display.width", 220)
    # the headline: pooled over markets, per level / kind / trend / horizon
    g = df.groupby(["kind", "level", "trend", "h"]).apply(
        lambda x: pd.Series({"Zellen": len(x), "Ø Überschuss (ATR)": x.excess_atr.mean(),
                             "Anteil Zellen > 0": (x.excess_atr > 0).mean(),
                             "Zellen mit t>=2": (x.t >= 2).sum(), "Zellen mit t<=-2": (x.t <= -2).sum(),
                             "Ø Kosten (ATR)": x.cost_atr.mean()}))
    print("\n### Über alle Märkte und beide Zeitrahmen")
    print(g.round(3).to_string())
    print(f"\n{len(df)} Zellen -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
