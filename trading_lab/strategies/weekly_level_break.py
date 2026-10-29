"""S-005: weekly level break - continuation after the first break of last week's high/low.

Spec (agreed with the user 2026-10-29): ``docs/specs/weekly_level_break.yaml``; tests:
``tests/test_weekly_level_break.py``. Origin: step B (``studies/liquidity_levels_study.py``).

 1. Levels: previous week's high/low (``trading_lab.liquidity.previous_week``).
 2. Event: the FIRST minute of the week beyond the level (``liquidity.events``); variant
    ``art="durch"`` keeps it only when the signal bar (15m/1h) also closes beyond.
 3. Trade in the direction of the break, market at the first minute after the signal bar's
    close; stop = entry -/+ 1 ATR(14) of the signal timeframe, target = k x stop.
 4. Time exit after 32 (15m) / 24 (1h) signal bars, Friday 21:00 UTC cut, one position per
    market. Fills, stop-first and the Friday rule: ``breakout_fade_fx.execute`` (S-004).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .. import liquidity
from ..indicators import atr as _atr
from ..market_structure import mid_ohlc
from ..mr_common import resample_ohlc
from .breakout_fade_fx import Prepared as _Fills, entry_allowed, execute

HOLD = {"15min": 32, "1h": 24}


@dataclass(frozen=True)
class WeeklyBreakConfig:
    zeitrahmen: str = "1h"          # 15min | 1h
    art: str = "durch"              # durch | alle
    ziel_r: float = 3.0             # 3 | 5
    stop_atr: float = 1.0
    slippage_bp: float = 0.5
    fin_pa: tuple | None = None     # measured (long, short) financing p.a. of this market


@dataclass
class Prepared:
    asset: str
    fills: _Fills
    tfs: dict                       # tf -> {"start_ns", "atr", "events"}


def prepare(m1: pd.DataFrame, asset: str, tfs: tuple[str, ...] = ("15min", "1h")) -> Prepared:
    fills = _Fills(asset=asset, m1_ns=m1.index.as_unit("ns").asi8,
                   bid={k: m1[k].to_numpy(float) for k in ("open", "high", "low", "close")},
                   ask={k: m1[f"ask_{k}"].to_numpy(float) for k in ("open", "high", "low", "close")}, tfs={})
    table = liquidity.previous_week(m1)
    out = {}
    for tf in tfs:
        bars = mid_ohlc(resample_ohlc(m1, tf))
        out[tf] = {"start_ns": bars.index.as_unit("ns").asi8,
                   "atr": _atr(bars["high"], bars["low"], bars["close"], 14).to_numpy(),
                   "events": liquidity.events(m1, bars, tf, table)}
    return Prepared(asset, fills, out)


def signals(events: list, art: str) -> list:
    """Rule 2: events kept by the variant (all are already first-per-week-and-side)."""
    return [e for e in events if art == "alle" or e.kind == "through"]


def first_fill(fills: _Fills, start_ns: int, d: int) -> tuple[int, float]:
    k = int(np.searchsorted(fills.m1_ns, start_ns, "left"))
    if k >= len(fills.m1_ns):
        return -1, np.nan
    return k, float(fills.ask["open"][k] if d > 0 else fills.bid["open"][k])


def run_backtest(p: Prepared, cfg: WeeklyBreakConfig, *, override: list[tuple[int, int]] | None = None) -> dict:
    """``override`` = [(bar, direction)] replaces the events (placebo)."""
    s = p.tfs[cfg.zeitrahmen]
    width = pd.Timedelta(cfg.zeitrahmen).value
    hold = HOLD[cfg.zeitrahmen]
    cand = override if override is not None else [(e.bar, e.side) for e in signals(s["events"], cfg.art)]
    trades, busy = [], -1
    slip = cfg.slippage_bp * 1e-4
    for i, d in sorted(cand):
        a = s["atr"][i]
        if not np.isfinite(a) or a <= 0 or i + 1 >= len(s["start_ns"]):
            continue
        start = int(s["start_ns"][i]) + width                 # signal bar close
        if start < busy or not entry_allowed(start):
            continue
        k0, px = first_fill(p.fills, start, d)
        if k0 < 0:
            continue
        entry = px * (1 + d * slip)
        stop = entry - d * cfg.stop_atr * a
        target = entry + d * cfg.ziel_r * cfg.stop_atr * a
        deadline = int(s["start_ns"][i]) + (1 + hold) * width
        tr = execute(p.fills, cfg, d, int(p.fills.m1_ns[k0]), stop, target, deadline, fin_pa=cfg.fin_pa)
        if tr is None:
            continue
        tr.update({"asset": p.asset, "signal_bar": i})
        trades.append(tr)
        busy = int(tr["exit_time"].value)
    return {"trades": trades}


def placebo_override(p: Prepared, cfg: WeeklyBreakConfig, real: list[dict], rng: np.random.Generator
                     ) -> list[tuple[int, int]]:
    """Same number of entries, random bars at the same UTC hour, random direction."""
    s = p.tfs[cfg.zeitrahmen]
    hours = pd.to_datetime(s["start_ns"], utc=True).hour
    out = []
    for t in real:
        pool = np.flatnonzero(hours == hours[t["signal_bar"]])
        out.append((int(rng.choice(pool)), int(rng.choice([-1, 1]))))
    return out


def on_mid(p: Prepared) -> Prepared:
    """Same market on mid prices (no spread) - for the gross numbers."""
    mid = {k: (p.fills.bid[k] + p.fills.ask[k]) / 2 for k in p.fills.bid}
    return replace(p, fills=replace(p.fills, bid=mid, ask=mid))


__all__ = ["WeeklyBreakConfig", "Prepared", "prepare", "signals", "run_backtest", "placebo_override", "on_mid", "HOLD"]
