#!/usr/bin/env python3
"""Step C: do ROUND numbers behave differently? (Osler 2003) - a measurement, no strategy.

Osler (NY Fed SR 125, JF 2003), with a dealing bank's order book: take-profit orders cluster
just BEFORE round numbers (prices reverse there), stop-loss orders just BEHIND them (prices
accelerate after a break). Test on our 1m data, TRAIN only (2019-2024):

* levels: the round grid (FX x.xx00 and x.xx50 = step 0.0050, JPY 0.50, gold 5.0) and a
  CONTROL grid of the same density shifted by 37 % of the step - same rule, not round;
* event: the first bar of a UTC day that trades through a level coming from the other side
  (previous close below, high >= level - or mirrored); per level and day only the first;
  the bar's close decides: back on the old side = ``abprall`` (bounce), beyond = ``durch``;
* trade from the bar's close: ``abprall`` against the touch (range fade), ``durch`` with it
  (cascade); stop 1 ATR(14) of the timeframe, target k ATR, at most H bars, stop first;
* effect = mean R at round levels minus mean R at control levels (two-sample t), next to the
  round-trip cost in R (measured one-way bp x 2 / ATR).

Timeframes 1m, 5m, 15m, 30m, 1h, 2h, 4h.

  .venv/bin/python studies/round_number_study.py
"""
from __future__ import annotations

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402

from trading_lab.indicators import atr as _atr  # noqa: E402
from trading_lab.market_structure import mid_ohlc  # noqa: E402
from trading_lab.mr_common import resample_ohlc  # noqa: E402
from trading_lab.registry import HOLDOUT_START, load_cached  # noqa: E402

MARKETS = {"EURUSD": 0.005, "GBPUSD": 0.005, "AUDUSD": 0.005, "USDCAD": 0.005, "EURGBP": 0.005,
           "USDJPY": 0.5, "XAUUSD": 5.0}
TFS = {"1min": 60, "5min": 48, "15min": 32, "30min": 24, "1h": 24, "2h": 18, "4h": 12}
KS = {"abprall": (1.0, 2.0), "durch": (2.0, 3.0)}
CONTROL_SHIFT = 0.37
OUT = Path("artifacts/round_numbers.json")


def touches(o, h, l, c, step: float, offset: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bar, side +1 = touched from below, level) - first touch of each level per day is
    filtered by the caller."""
    prev = np.r_[np.nan, c[:-1]]
    up_lvl = (np.floor((prev - offset) / step) + 1) * step + offset       # first level above prev close
    dn_lvl = (np.ceil((prev - offset) / step) - 1) * step + offset        # first level below
    up = h >= up_lvl
    dn = l <= dn_lvl
    both = up & dn                                                        # outside bar - ambiguous, skip
    up, dn = up & ~both, dn & ~both
    iu, idn = np.flatnonzero(up), np.flatnonzero(dn)
    bars = np.r_[iu, idn]
    side = np.r_[np.ones(len(iu), int), -np.ones(len(idn), int)]
    lvl = np.r_[up_lvl[iu], dn_lvl[idn]]
    return bars, side, lvl


def brackets(h, l, c, a, idx, d, k: float, H: int) -> np.ndarray:
    """R of a bracket from close[idx]: stop 1 ATR, target k ATR, H bars, stop first."""
    n = len(c)
    ok = idx + H < n
    idx, d = idx[ok], d[ok]
    hw = sliding_window_view(h, H)[idx + 1]
    lw = sliding_window_view(l, H)[idx + 1]
    e, r = c[idx], a[idx]
    stop = e - d * r
    tgt = e + d * k * r
    hit_s = np.where(d[:, None] > 0, lw <= stop[:, None], hw >= stop[:, None])
    hit_t = np.where(d[:, None] > 0, hw >= tgt[:, None], lw <= tgt[:, None])
    fs = np.where(hit_s.any(1), hit_s.argmax(1), H)
    ft = np.where(hit_t.any(1), hit_t.argmax(1), H)
    end = d * (c[idx + H] - e) / r
    out = np.where(fs <= ft, np.where(fs < H, -1.0, end), np.where(ft < H, k, end))
    full = np.full(len(ok), np.nan)
    full[ok] = out
    return full


def one(asset: str) -> list[dict]:
    warnings.filterwarnings("ignore")
    m1 = load_cached(asset)
    m1 = m1[m1.index < HOLDOUT_START]
    cost_bp = json.loads(Path("artifacts/tsmom/costs.json").read_text())[asset]
    step = MARKETS[asset]
    rows = []
    for tf, H in TFS.items():
        df = mid_ohlc(m1 if tf == "1min" else resample_ohlc(m1, tf))
        o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
        a = _atr(df["high"], df["low"], df["close"], 14).to_numpy()
        day = df.index.normalize().asi8
        cost_r = float(np.nanmedian(2 * cost_bp * 1e-4 * c / a))
        res = {}
        for grid, off in (("rund", 0.0), ("kontrolle", CONTROL_SHIFT * step)):
            bars, side, lvl = touches(o, h, l, c, step, off)
            order = np.argsort(bars, kind="stable")
            bars, side, lvl = bars[order], side[order], lvl[order]
            first = ~pd.DataFrame({"d": day[bars], "l": np.round(lvl / step * 1000)}).duplicated(keep="first").to_numpy()
            bars, side, lvl = bars[first], side[first], lvl[first]
            ok = np.isfinite(a[bars]) & (a[bars] > 0)
            bars, side, lvl = bars[ok], side[ok], lvl[ok]
            beyond = side * (c[bars] - lvl) > 0
            for kind, mask in (("abprall", ~beyond), ("durch", beyond)):
                d = (-side if kind == "abprall" else side)[mask]
                for k in KS[kind]:
                    res[(grid, kind, k)] = brackets(h, l, c, a, bars[mask], d, k, H)
        for kind in KS:
            for k in KS[kind]:
                x, y = res[("rund", kind, k)], res[("kontrolle", kind, k)]
                x, y = x[np.isfinite(x)], y[np.isfinite(y)]
                if len(x) < 30 or len(y) < 30:
                    continue
                t = (x.mean() - y.mean()) / np.sqrt(x.var() / len(x) + y.var() / len(y))
                rows.append({"asset": asset, "tf": tf, "art": kind, "k": k, "n_rund": len(x), "n_kontrolle": len(y),
                             "r_rund": float(x.mean()), "r_kontrolle": float(y.mean()),
                             "effekt": float(x.mean() - y.mean()), "t": float(t), "kosten_r": cost_r})
    print(f"  {asset}: {len(rows)} Zellen", flush=True)
    return rows


def main() -> int:
    warnings.filterwarnings("ignore")
    with ProcessPoolExecutor(4) as ex:
        rows = [r for part in ex.map(one, list(MARKETS)) for r in part]
    OUT.write_text(json.dumps(rows, indent=1))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    g = df.groupby(["art", "k", "tf"], sort=False).apply(lambda x: pd.Series({
        "Ø R rund": x.r_rund.mean(), "Ø R Kontrolle": x.r_kontrolle.mean(), "Effekt": x.effekt.mean(),
        "Märkte Effekt>0": f"{int((x.effekt > 0).sum())}/{len(x)}", "t>=2": int((x.t >= 2).sum()),
        "Kosten R": x.kosten_r.mean(), "Ø Ereignisse/Markt": x.n_rund.mean()}))
    print("\n### Runde Zahlen vs. Kontroll-Levels, Mittel über 7 Märkte (Training 2019-2024)")
    print(g.round(3).to_string())
    print(f"\n{len(df)} Zellen -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
