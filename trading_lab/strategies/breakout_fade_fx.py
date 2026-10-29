"""S-004: Breakout-Fade FX - fade a close through an internal swing in the trend direction.

Spec (agreed with the user 2026-10-29): ``docs/specs/breakout_fade_fx.yaml``; origin: step A
(``studies/sweep_continuation_study.py``) found that on FX a close above an internal swing
high in an uptrend tends to pull back over the next bars. Each rule is a small function
with a test in ``tests/test_breakout_fade_fx.py``.

Short (long mirrored):
 1. external structure trend of the signal timeframe, as of the PREVIOUS bar, is up
    (``trendfilter="egal"`` drops this);
 2. the signal bar closes above the internal swing high (internal BOS/CHoCH up);
 3. optional session filter: signal bar closes 07:00-17:00 UTC;
 4. short at the next bar's open (bid, + slippage); stop = signal-bar high + x ATR;
 5. exit after 2 / 4 / 8 bars at market, or ``level_retest``: target = the broken swing
    high, at the latest after 8 bars; Friday 21:00 UTC hard cut; one position per pair.
Inside a minute the stop is checked first; in the entry minute only the stop counts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..costs import asset_class
from ..indicators import atr as _atr
from ..market_structure import BOS_DN, BOS_UP, CHOCH_DN, CHOCH_UP, StructureConfig, analyze, mid_ohlc
from ..mr_common import resample_ohlc
from .fib_retracement import _friday_cut_ns, friday_entry_allowed, rollover_weights

TFS = ("1h", "15min")
FIN_RATE_FX = 0.010
SESSION_UTC = (7, 17)               # signal-bar close inside [07:00, 17:00] UTC


@dataclass(frozen=True)
class BreakoutFadeConfig:
    asset: str = "EURUSD"
    zeitrahmen: str = "1h"          # 1h | 15min
    stop_atr: float = 1.5           # 1.0 | 1.5 | 2.0
    exit: str = "zeit_4"            # zeit_2 | zeit_4 | zeit_8 | level_retest
    session: str = "alle"           # alle | london_ny
    trendfilter: str = "an"         # an | egal
    min_stop_atr: float = 0.5
    retest_max_bars: int = 8
    slippage_bp: float = 0.5


@dataclass
class Prepared:
    asset: str
    m1_ns: np.ndarray
    bid: dict
    ask: dict
    tfs: dict


def prepare(m1: pd.DataFrame, asset: str, tfs: tuple[str, ...] = TFS) -> Prepared:
    p = Prepared(asset=asset, m1_ns=m1.index.as_unit("ns").asi8,
                 bid={k: m1[k].to_numpy(float) for k in ("open", "high", "low", "close")},
                 ask={k: m1[f"ask_{k}"].to_numpy(float) for k in ("open", "high", "low", "close")}, tfs={})
    for tf in tfs:
        df = mid_ohlc(resample_ohlc(m1, tf))
        b = analyze(df, StructureConfig()).bars
        p.tfs[tf] = {"close_ns": df.index.as_unit("ns").asi8 + pd.Timedelta(tf).value,
                     "high": df["high"].to_numpy(), "low": df["low"].to_numpy(),
                     "atr": _atr(df["high"], df["low"], df["close"], 14).to_numpy(),
                     "int_event": b["int_event"].to_numpy(), "ext_trend": b["ext_trend"].to_numpy(),
                     "int_swing_high": b["int_swing_high"].to_numpy(), "int_swing_low": b["int_swing_low"].to_numpy()}
    return p


# ── rules 1-3: signals ─────────────────────────────────────────────────────

def signals(int_event: np.ndarray, ext_trend: np.ndarray, close_ns: np.ndarray, cfg: BreakoutFadeConfig
            ) -> list[tuple[int, int]]:
    """(bar, trade direction) - a break UP is faded with a SHORT (-1), a break down with a long."""
    prev = np.r_[0, ext_trend[:-1]]
    hours = pd.to_datetime(close_ns, utc=True)
    mins = hours.hour * 60 + hours.minute
    out = []
    for i, e in enumerate(int_event):
        if e in (BOS_UP, CHOCH_UP):
            brk = 1
        elif e in (BOS_DN, CHOCH_DN):
            brk = -1
        else:
            continue
        if cfg.trendfilter == "an" and prev[i] != brk:
            continue
        if cfg.session == "london_ny" and not (SESSION_UTC[0] * 60 <= mins[i] <= SESSION_UTC[1] * 60):
            continue
        out.append((i, -brk))
    return out


# ── rules 4-5: execution ───────────────────────────────────────────────────

def entry_allowed(t_ns: int) -> bool:
    """No entry from Friday 21:00 UTC to the weekend's end (prop standard account); shared
    rule with the Fibonacci strategy's standard-account variant."""
    return friday_entry_allowed(t_ns)


def execute(prep: Prepared, cfg, d: int, start_ns: int, stop: float, target: float,
            deadline_ns: int, *, fin_pa: tuple[float, float] | None = None) -> dict | None:
    """Market entry at the first minute >= ``start_ns``; stop first each minute, target
    (if finite) only from the minute after entry; flat at ``deadline_ns`` (open of the
    first minute at/after it, i.e. at the bar's close) or the Friday cut.

    ``fin_pa`` = measured (long, short) financing p.a. (negative = pays, S-005); without it
    the S-004 rule: FX pays 1 % p.a., tripled on Wednesday. ``cfg`` needs ``slippage_bp``."""
    t, bid, ask = prep.m1_ns, prep.bid, prep.ask
    n = len(t)
    k0 = int(np.searchsorted(t, start_ns, "left"))
    if k0 >= n or not entry_allowed(int(t[k0])):
        return None
    slip = cfg.slippage_bp * 1e-4
    entry = (ask["open"][k0] if d > 0 else bid["open"][k0]) * (1 + d * slip)
    if d * (entry - stop) <= 0:
        return None
    if np.isfinite(target) and d * (target - entry) <= 0:
        return None                                  # target already passed at the open
    ex_h, ex_l, ex_o = ((bid["high"], bid["low"], bid["open"]) if d > 0 else (ask["high"], ask["low"], ask["open"]))
    hard = min(deadline_ns, _friday_cut_ns(int(t[k0])))
    k, px, reason = k0, np.nan, "data_end"
    while k < n:
        if t[k] >= hard:
            px, reason = ex_o[k] * (1 - d * slip), "zeit"
            break
        adverse = ex_l[k] if d > 0 else ex_h[k]
        if d * (adverse - stop) <= 0:
            fill = (min(stop, ex_o[k]) if d > 0 else max(stop, ex_o[k])) if k > k0 else stop
            px, reason = fill * (1 - d * slip), "stop"
            break
        if k > k0 and np.isfinite(target):
            favour = ex_h[k] if d > 0 else ex_l[k]
            if d * (favour - target) >= 0:
                px, reason = target, "ziel"
                break
        k += 1
    if not np.isfinite(px):
        k = n - 1
        px = (bid if d > 0 else ask)["close"][k]
    risk = d * (entry - stop)
    if fin_pa is not None:
        rate = fin_pa[0] if d > 0 else fin_pa[1]
        fin = -rate / 365.0 * rollover_weights(int(t[k0]), int(t[k]),
                                               triple_dow=2 if asset_class(prep.asset) == "fx" else 4) * entry
    else:
        fin = FIN_RATE_FX / 365.0 * rollover_weights(int(t[k0]), int(t[k]), triple_dow=2) * entry \
            if asset_class(prep.asset) == "fx" else 0.0
    return {"entry_time": pd.Timestamp(t[k0], tz="UTC"), "exit_time": pd.Timestamp(t[k], tz="UTC"),
            "direction": d, "entry": entry, "stop": stop, "target": target if np.isfinite(target) else None,
            "r": (d * (px - entry) - fin) / risk, "swap_r": fin / risk, "exit_reason": reason}


def run_backtest(prep: Prepared, cfg: BreakoutFadeConfig) -> dict:
    s = prep.tfs[cfg.zeitrahmen]
    cn, a = s["close_ns"], s["atr"]
    nbar = len(cn)
    hold = {"zeit_2": 2, "zeit_4": 4, "zeit_8": 8, "level_retest": cfg.retest_max_bars}[cfg.exit]
    trades, busy = [], -1
    for i, d in signals(s["int_event"], s["ext_trend"], cn, cfg):
        if not np.isfinite(a[i]) or cn[i] < busy or i + hold >= nbar:
            continue
        # fading a break UP (d = -1): stop above the signal bar's high
        stop = (s["high"][i] + cfg.stop_atr * a[i]) if d < 0 else (s["low"][i] - cfg.stop_atr * a[i])
        target = np.nan
        if cfg.exit == "level_retest":
            target = s["int_swing_high"][i] if d < 0 else s["int_swing_low"][i]
        tr = execute(prep, cfg, d, int(cn[i]), stop, target, int(cn[i + hold]))
        if tr is None or d * (tr["entry"] - stop) < cfg.min_stop_atr * a[i]:
            continue
        trades.append(tr)
        busy = int(tr["exit_time"].value)
    return {"trades": trades}
