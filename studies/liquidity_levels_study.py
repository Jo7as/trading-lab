#!/usr/bin/env python3
"""Step B: big moves after time-based liquidity levels? A measurement, not a strategy.

Levels from ``trading_lab.liquidity``: previous day (pd), previous week (pw), Asia range
(asia), opening range (or). Event = the FIRST minute of a period beyond the level; the
signal bar containing it is a ``reclaim`` (close back inside, a sweep) or a ``through``
(close beyond, a break).

Question: does a trade from the event bar's close reach a FAR target more often than a
trade from an arbitrary bar? Bracket per event, walked bar by bar on mid OHLC of the signal
timeframe (stop checked first inside a bar):

  * ``mit``   - in the direction of the exceed (continuation / stop cascade),
  * ``gegen`` - against it (the classic sweep reversal);
  * stop ``atr``: 1 ATR from the close; stop ``extrem`` (only ``gegen``): beyond the event
    bar's extreme + 0.1 ATR - the tight SMC stop;
  * target k x stop distance, k in {1, 2, 3, 5}; at most H bars (8 h on 15m, 24 h on 1h);
    flat at the end of the window (mark-to-market); windows across a gap > 6 h (weekend)
    are dropped.

Result in R of that bracket; ``excess`` = minus the same bracket from EVERY bar of that
timeframe in the same direction (baseline). ``net`` subtracts the round trip (median
spread + ftmo commission) over each trade's stop distance. TRAIN ONLY (2019-2024).

  .venv/bin/python studies/liquidity_levels_study.py
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

from trading_lab import costs, liquidity, screening  # noqa: E402
from trading_lab.indicators import atr as _atr  # noqa: E402
from trading_lab.market_structure import mid_ohlc  # noqa: E402
from trading_lab.mr_common import resample_ohlc  # noqa: E402
from trading_lab.registry import HOLDOUT_START, load_cached  # noqa: E402

MARKETS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "US100", "SP500", "GER40", "XAUUSD"]
TFS = {"15min": 32, "1h": 24}
KS = (1, 2, 3, 5)
OUT = Path("artifacts/liquidity_levels.json")


def bracket(h: np.ndarray, l: np.ndarray, c: np.ndarray, i: int, d: int, risk: float, k: float, n: int) -> float:
    """R of a bracket entered at close[i]: stop ``risk`` away, target ``k * risk``, n bars."""
    e = c[i]
    stop, tgt = e - d * risk, e + d * k * risk
    for j in range(i + 1, min(i + 1 + n, len(c))):
        if (l[j] <= stop) if d > 0 else (h[j] >= stop):
            return -1.0
        if (h[j] >= tgt) if d > 0 else (l[j] <= tgt):
            return float(k)
    j = min(i + n, len(c) - 1)
    return float(d * (c[j] - e) / risk)


def study_one(asset: str) -> list[dict]:
    warnings.filterwarnings("ignore")
    m1 = load_cached(asset)
    m1 = m1[m1.index < HOLDOUT_START]
    if m1.empty:
        return []
    tabs = liquidity.tables(m1, asset)
    rows = []
    for tf, H in TFS.items():
        raw = resample_ohlc(m1, tf)
        df = mid_ohlc(raw)
        o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
        a = _atr(df["high"], df["low"], df["close"], 14).to_numpy()
        n = len(c)
        st = df.index.as_unit("ns").asi8
        gap = np.r_[False, np.diff(st) > pd.Timedelta(hours=6).value].cumsum()   # weekend id
        win_ok = np.zeros(n, bool)
        win_ok[: n - H] = gap[H:] == gap[: n - H]
        sp = float(np.nanmedian((raw["ask_close"] - raw["close"]).to_numpy())) if "ask_close" in raw else 0.0
        px = float(np.nanmedian(c))
        cost = sp + 2 * costs.commission_bp_side(asset, px, "ftmo") * 1e-4 * px
        # baseline: every 4th bar (enough, keeps it fast), both directions, both stop types
        base_idx = np.flatnonzero(win_ok & np.isfinite(a))[::4]
        base = {}
        for d in (1, -1):
            for k in KS:
                base[("atr", d, k)] = np.mean([bracket(h, l, c, i, d, a[i], k, H) for i in base_idx])
                ext = [bracket(h, l, c, i, d, (c[i] - l[i] if d > 0 else h[i] - c[i]) + 0.1 * a[i], k, H)
                       for i in base_idx]
                base[("extrem", d, k)] = np.mean(ext)
        for name, tab in tabs.items():
            evs = liquidity.events(m1, raw, tf, tab)
            for kind in ("reclaim", "through"):
                for align in ("mit", "gegen"):
                    sel = [e for e in evs if e.kind == kind and win_ok[e.bar] and np.isfinite(a[e.bar])]
                    if len(sel) < 30:
                        continue
                    for stop_kind in (("atr", "extrem") if align == "gegen" else ("atr",)):
                        for k in KS:
                            r, net, ex = [], [], []
                            for e in sel:
                                d = e.side if align == "mit" else -e.side
                                i = e.bar
                                risk = a[i] if stop_kind == "atr" else \
                                    ((c[i] - l[i]) if d > 0 else (h[i] - c[i])) + 0.1 * a[i]
                                x = bracket(h, l, c, i, d, risk, k, H)
                                r.append(x)
                                ex.append(x - base[(stop_kind, d, k)])
                                net.append(x - cost / risk)
                            r, net, ex = np.array(r), np.array(net), np.array(ex)
                            rows.append({"asset": asset, "tf": tf, "level": name, "kind": kind, "align": align,
                                         "stop": stop_kind, "k": k, "n": len(r),
                                         "r": float(r.mean()), "excess": float(ex.mean()),
                                         "t_excess": float(screening.t_stat(ex)),
                                         "net": float(net.mean()), "t_net": float(screening.t_stat(net)),
                                         "hit": float((r >= k - 1e-9).mean())})
    print(f"  {asset}: {len(rows)} Zellen", flush=True)
    return rows


def main() -> int:
    warnings.filterwarnings("ignore")
    with ProcessPoolExecutor(4) as ex:
        rows = [r for part in ex.map(study_one, MARKETS) for r in part]
    OUT.write_text(json.dumps(rows, indent=1))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    g = df.groupby(["kind", "align", "stop", "level", "k"]).apply(
        lambda x: pd.Series({"Zellen": len(x), "Ø R": x.r.mean(), "Ø Überschuss": x.excess.mean(),
                             "t>=2": int((x.t_excess >= 2).sum()), "t<=-2": int((x.t_excess <= -2).sum()),
                             "Ø netto": x.net.mean(), "netto>0 & t>=2": int(((x.net > 0) & (x.t_net >= 2)).sum())}))
    print("\n### Über alle Märkte und beide Zeitrahmen")
    print(g.round(3).to_string())
    best = df[(df.net > 0) & (df.t_net >= 2)].sort_values("t_net", ascending=False)
    print(f"\n### Zellen netto > 0 mit t >= 2: {len(best)} von {len(df)}")
    print(best.head(30).round(3).to_string(index=False))
    print(f"\n{len(df)} Zellen -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
