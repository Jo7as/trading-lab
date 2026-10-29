#!/usr/bin/env python3
"""Step D: our own range determination - time-at-price levels (measurement, no strategy).

Idea (Tengelin & Sopasakis 2020, docs/PAPERS.md): support/resistance from WHERE the market spent its
time, not from drawn lines. For each 17:00-NY trading day, from the PREVIOUS day's 1m mid closes:

* POC = price bin with the most minutes (50 equal bins across the day's range);
* value area = bins around the POC holding 70 % of the minutes -> VAH (top), VAL (bottom).

Control (same positions, no time information): the previous day's range at 85 % / 50 % / 15 %
(VAH <-> r85, POC <-> r50, VAL <-> r15). If time-at-price matters, the profile levels must behave
differently from the control levels.

Event: the first bar of the day that trades through a level coming from the other side (previous
close on one side, high/low across it). The bar's close decides: back = ``abprall`` (trade against the
touch, range fade), beyond = ``durch`` (trade with it). Bracket from the bar close: stop 1 ATR(14) of
the timeframe, target k ATR, at most H bars, stop first (``round_number_study.brackets``). Effect =
profile minus control (two-sample t), next to the round-trip cost in R. TRAIN only (2019-2024).

  .venv/bin/python studies/value_area_study.py
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

from studies.round_number_study import brackets  # noqa: E402
from trading_lab.indicators import atr as _atr  # noqa: E402
from trading_lab.market_structure import mid_ohlc  # noqa: E402
from trading_lab.mr_common import resample_ohlc  # noqa: E402
from trading_lab.registry import HOLDOUT_START, load_cached  # noqa: E402

MARKETS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD", "US100", "SP500", "GER40"]
TFS = {"15min": 32, "1h": 24}
KS = {"abprall": (1.0, 2.0), "durch": (2.0, 3.0)}
PAIRS = (("vah", "r85"), ("poc", "r50"), ("val", "r15"))
OUT = Path("artifacts/value_area.json")
NY = "America/New_York"


def trading_day(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.tz_convert(NY) - pd.Timedelta(hours=17)).normalize().tz_localize(None).asi8


def profile_levels(closes: np.ndarray, bins: int = 50, share: float = 0.70) -> dict[str, float]:
    """POC / VAH / VAL from time at price, plus the range control levels."""
    lo, hi = float(np.min(closes)), float(np.max(closes))
    if not hi > lo:
        return {}
    cnt, edges = np.histogram(closes, bins=bins, range=(lo, hi))
    mid = (edges[:-1] + edges[1:]) / 2
    p = int(np.argmax(cnt))
    a = b = p
    total, inside = cnt.sum(), cnt[p]
    while inside < share * total:                      # grow towards the busier neighbour
        left = cnt[a - 1] if a > 0 else -1
        right = cnt[b + 1] if b < bins - 1 else -1
        if right >= left:
            b += 1
            inside += cnt[b]
        else:
            a -= 1
            inside += cnt[a]
    rng = hi - lo
    return {"poc": mid[p], "vah": edges[b + 1], "val": edges[a],
            "r85": lo + 0.85 * rng, "r50": lo + 0.50 * rng, "r15": lo + 0.15 * rng}


def first_touches(c: np.ndarray, h: np.ndarray, l: np.ndarray, rows: np.ndarray, level: float
                  ) -> tuple[int, int] | None:
    """(bar, side) of the first bar in ``rows`` crossing ``level`` from the other side."""
    for i in rows:
        if i == 0:
            continue
        if c[i - 1] < level <= h[i]:
            return int(i), 1
        if c[i - 1] > level >= l[i]:
            return int(i), -1
    return None


def one(asset: str) -> list[dict]:
    warnings.filterwarnings("ignore")
    m1 = load_cached(asset)
    m1 = m1[m1.index < HOLDOUT_START]
    mid1 = mid_ohlc(m1)
    cost_bp = json.loads(Path("artifacts/tsmom/costs.json").read_text())[asset]
    d1 = trading_day(mid1.index)
    days = np.unique(d1)
    closes_by_day = {d: mid1["close"].to_numpy()[d1 == d] for d in days}
    rows = []
    for tf, H in TFS.items():
        df = mid_ohlc(resample_ohlc(m1, tf))
        h, l, c = (df[x].to_numpy(float) for x in ("high", "low", "close"))
        a = _atr(df["high"], df["low"], df["close"], 14).to_numpy()
        bd = trading_day(df.index)
        cost_r = float(np.nanmedian(2 * cost_bp * 1e-4 * c / a))
        ev = {k: [] for k in ("poc", "vah", "val", "r85", "r50", "r15")}
        for prev, day in zip(days[:-1], days[1:]):
            closes = closes_by_day[prev]
            if len(closes) < 600:                             # thin / holiday session
                continue
            lv = profile_levels(closes)
            if not lv:
                continue
            idx = np.flatnonzero(bd == day)
            for name, level in lv.items():
                t = first_touches(c, h, l, idx, level)
                if t is not None and np.isfinite(a[t[0]]) and a[t[0]] > 0:
                    ev[name].append((t[0], t[1], level))
        for prof, ctrl in PAIRS:
            res = {}
            for name in (prof, ctrl):
                e = np.array(ev[name]) if ev[name] else np.zeros((0, 3))
                bars, side, lvl = e[:, 0].astype(int), e[:, 1].astype(int), e[:, 2]
                beyond = side * (c[bars] - lvl) > 0
                for kind, mask in (("abprall", ~beyond), ("durch", beyond)):
                    d = (-side if kind == "abprall" else side)[mask]
                    for k in KS[kind]:
                        res[(name, kind, k)] = brackets(h, l, c, a, bars[mask], d, k, H)
            for kind in KS:
                for k in KS[kind]:
                    x, y = res[(prof, kind, k)], res[(ctrl, kind, k)]
                    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
                    if len(x) < 30 or len(y) < 30:
                        continue
                    t = (x.mean() - y.mean()) / np.sqrt(x.var() / len(x) + y.var() / len(y))
                    rows.append({"asset": asset, "tf": tf, "level": prof, "kontrolle": ctrl, "art": kind, "k": k,
                                 "n": len(x), "n_ctrl": len(y), "r_profil": float(x.mean()),
                                 "r_kontrolle": float(y.mean()), "effekt": float(x.mean() - y.mean()),
                                 "t": float(t), "kosten_r": cost_r})
    print(f"  {asset}: {len(rows)} Zellen", flush=True)
    return rows


def main() -> int:
    warnings.filterwarnings("ignore")
    with ProcessPoolExecutor(4) as ex:
        rows = [r for part in ex.map(one, MARKETS) for r in part]
    OUT.write_text(json.dumps(rows, indent=1))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    g = df.groupby(["level", "art", "k", "tf"], sort=False).apply(lambda x: pd.Series({
        "Ø R Profil": x.r_profil.mean(), "Ø R Kontrolle": x.r_kontrolle.mean(), "Effekt": x.effekt.mean(),
        "Märkte>0": f"{int((x.effekt > 0).sum())}/{len(x)}", "t>=2": int((x.t >= 2).sum()),
        "netto R Profil": (x.r_profil - x.kosten_r).mean(), "Ø Ereignisse": x.n.mean()}))
    print("\n### Zeit-am-Preis-Levels (POC/VAH/VAL des Vortags) vs. Spannen-Levels, 8 Märkte, 2019-2024")
    print(g.round(3).to_string())
    print(f"\n{len(df)} Zellen -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
