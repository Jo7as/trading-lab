"""ORB Rejection Fade with inverse FVG ("ORB Ping-Pong") — spec-driven v0.

Spec: ``strategy_orb_rejection_ifvg.yaml``. Fade the opening-range edges when
price repeatedly *fails* there (rejection candles + bearish/bullish engulfing +
an inverted fair-value gap), targeting the opposite edge. A momentum breakout
locks the failed side for the day.

Ablation-first: ``mode`` selects how many filters are active, so the runner can
show whether each one adds anything:

    naive  → fade the first touch of the edge (no touch count, no candle, no iFVG)
    touches→ require ``min_touches`` separate touches
    engulf → + a rejection candle and an engulfing trigger bar
    full   → + the inverted fair-value gap

Side convention throughout: **+1 = long at ORL, −1 = short at ORH.**
Execution on 1-minute RTH bars (America/New_York). Costs: historical bid/ask plus
entry/stop slippage and a point commission. Results in R and account-% through
the shared pipeline. One position at a time; ping-pong (opposite side after an
exit) is allowed up to ``max_trades_per_day``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ..backtest import _compute_metrics
from ..indicators import atr as atr_indicator
from ..mr_common import ASK_COLS, ensure_utc, with_ask
from .orb_2r import skipped_days

MODES = ("naive", "touches", "engulf", "full", "random", "breakout")


@dataclass(frozen=True)
class OrbRejConfig:
    asset: str = "US100"
    asset_label: str = "US100"
    tz: str = "America/New_York"
    calendar: str = "nyse"
    session_open: str = "09:30"
    session_close: str = "16:00"
    or_minutes: int = 15
    entry_start: str = "09:45"
    entry_end: str = "11:30"
    flat_time: str = "15:55"

    mode: str = "full"                 # naive | touches | engulf | full | random | breakout
    entry_mode: str = "close"          # close (touch bar) | next_open (bar after the signal)

    touch_tolerance_ticks: float = 2.0
    reject_close_buffer_frac: float = 0.05   # × or_size
    min_bars_between_touches: int = 3
    min_touches: int = 3

    wick_ratio: float = 0.5
    rej_candle_min: int = 2
    rej_candle_lookback: int = 10

    engulf_min: float = 1.0
    engulf_min_range_atr: float = 1.5
    atr_len: int = 14

    ifvg_lookback: int = 15
    min_fvg_ticks: float = 1.0
    trigger_gap_bars: int = 2

    breakout_close_buffer_frac: float = 0.10  # × or_size
    acceptance_bars: int = 3

    stop_buffer_ticks: float = 2.0
    min_stop_points: float = 2.0
    max_stop_points: float = 10.0
    target: str = "opposite_edge"      # opposite_edge | mid_then_edge
    tp1_close_pct: float = 0.5
    front_run_ticks: float = 2.0
    min_reward_risk: float = 1.5

    risk_per_trade_pct: float = 0.5
    max_leverage: float = 10.0
    tick_size: float = 0.25
    slippage_ticks_entry: float = 1.0
    slippage_ticks_stop: float = 2.0
    commission_points: float = 0.0

    max_trades_per_day: int = 2
    consec_loss_stop: int = 2
    breakout_target_r: float = 2.0     # classic-ORB-breakout baseline target
    random_seed: int = 7

    @staticmethod
    def search_space() -> list[tuple[str, float, float, float]]:
        return [
            ("min_touches", 2, 4, 1),
            ("ifvg_lookback", 5, 30, 5),
            ("trigger_gap_bars", 1, 4, 1),
            ("min_reward_risk", 1.0, 2.0, 0.25),
            ("or_minutes", 5, 30, 5),
        ]


# ── Primitives ──────────────────────────────────────────────────────


def opening_range(h: np.ndarray, l: np.ndarray, or_bars: int) -> tuple[float, float]:
    return float(h[:or_bars].max()), float(l[:or_bars].min())


def rejection_candle(o: float, h: float, l: float, c: float, wick_ratio: float) -> bool:
    """Bearish rejection: long upper wick, red candle (short-side level)."""
    rng = h - l
    if rng <= 0:
        return False
    return (h - max(o, c)) >= wick_ratio * rng and c < o


def bullish_rejection_candle(o: float, h: float, l: float, c: float, wick_ratio: float) -> bool:
    """Bullish rejection: long lower wick, green candle (long-side level)."""
    rng = h - l
    if rng <= 0:
        return False
    return (min(o, c) - l) >= wick_ratio * rng and c > o


def bearish_engulfing(o0: float, c0: float, o1: float, c1: float, engulf_min: float = 1.0) -> bool:
    body0, body1 = abs(c0 - o0), abs(c1 - o1)
    return (c0 < o0 and o0 >= c1 and c0 <= o1 and body0 >= engulf_min * body1)


def bullish_engulfing(o0: float, c0: float, o1: float, c1: float, engulf_min: float = 1.0) -> bool:
    body0, body1 = abs(c0 - o0), abs(c1 - o1)
    return (c0 > o0 and o0 <= c1 and c0 >= o1 and body0 >= engulf_min * body1)


# ── Prepared data ───────────────────────────────────────────────────


@dataclass
class Prepared:
    ts: np.ndarray
    local: pd.DatetimeIndex
    days: list[date]
    day_slices: list[slice]
    bid: dict[str, np.ndarray]
    ask: dict[str, np.ndarray]
    atr1m: np.ndarray


def _hhmm_min(hhmm: str) -> int:
    h, m = (int(x) for x in hhmm.split(":"))
    return h * 60 + m


def prepare(data: pd.DataFrame, cfg: OrbRejConfig) -> Prepared:
    if not all(c in data.columns for c in ASK_COLS):
        data = with_ask(data, 0.0)
    df = ensure_utc(data)
    local = df.index.tz_convert(cfg.tz)
    ts = df.index.as_unit("ns").asi8
    bid = {k: df[k].to_numpy(float) for k in ("open", "high", "low", "close")}
    ask = {k: df[f"ask_{k}"].to_numpy(float) for k in ("open", "high", "low", "close")}
    atr1m = atr_indicator(df["high"], df["low"], df["close"], cfg.atr_len).to_numpy(float)

    minutes = local.hour * 60 + local.minute
    rth = (minutes >= _hhmm_min(cfg.session_open)) & (minutes < _hhmm_min(cfg.session_close))
    idx_rth = np.flatnonzero(rth)
    day_arr = local[idx_rth].normalize().to_numpy()
    # rows are time-sorted, so each day is contiguous: one vectorised split
    uniq, starts, counts = np.unique(day_arr, return_index=True, return_counts=True)
    days, slices = [], []
    for u, s, c in zip(uniq, starts, counts):
        if c < cfg.or_minutes + 5:
            continue
        days.append(pd.Timestamp(u).date())
        slices.append(slice(int(s), int(s) + int(c)))

    skip: set[date] = set()
    if days:
        for year in range(days[0].year, days[-1].year + 1):
            skip |= skipped_days(cfg.calendar, year)
    keep = [(d, s) for d, s in zip(days, slices) if d.weekday() < 5 and d not in skip]
    return Prepared(ts=ts, local=local, days=[d for d, _ in keep],
                    day_slices=[s for _, s in keep], bid=bid, ask=ask, atr1m=atr1m)


# ── Inverse FVG ─────────────────────────────────────────────────────


def inversion_flags(h: np.ndarray, l: np.ndarray, c: np.ndarray,
                    side: int, cfg: OrbRejConfig) -> np.ndarray:
    """Per-bar flag: bar ``k`` closes back through an FVG formed in the lookback.

    ``side = -1`` (short at ORH): a bullish FVG (``low[b] > high[b-2]``) whose
    bottom the bar closes below → the gap inverts to resistance.
    ``side = +1`` (long at ORL): a bearish FVG (``high[b] < low[b-2]``) whose
    top the bar closes above → the gap inverts to support.
    """
    n = len(c)
    flags = np.zeros(n, dtype=bool)
    need = cfg.min_fvg_ticks * cfg.tick_size
    lb = cfg.ifvg_lookback
    for k in range(n):
        for b in range(max(2, k - lb), k):
            if side < 0:
                level = h[b - 2]
                if l[b] > level and (l[b] - level) >= need and c[k] < level:
                    flags[k] = True
                    break
            else:
                level = l[b - 2]
                if h[b] < level and (level - h[b]) >= need and c[k] > level:
                    flags[k] = True
                    break
    return flags


# ── Engine ──────────────────────────────────────────────────────────


def _simulate(cfg: OrbRejConfig, i0: int, flat_i: int, side: int,
              hi: np.ndarray, lo: np.ndarray, op: np.ndarray, cl: np.ndarray,
              stop: float, target: float) -> tuple[int, float, str]:
    """Bracket exit on the day's 1m arrays; stop first when both fall in a bar."""
    slip = cfg.slippage_ticks_stop * cfg.tick_size
    for j in range(i0 + 1, flat_i + 1):
        hit_stop = lo[j] <= stop if side > 0 else hi[j] >= stop
        hit_tp = hi[j] >= target if side > 0 else lo[j] <= target
        if hit_stop:
            px = min(stop, op[j]) if side > 0 else max(stop, op[j])
            return j, (px - slip if side > 0 else px + slip), "stop"
        if hit_tp:
            return j, float(target), "target"
    return flat_i, float(cl[flat_i]), "flat"


def run_backtest(data: pd.DataFrame, cfg: OrbRejConfig, *, verbose: bool = False,
                 prepared: Prepared | None = None) -> dict[str, Any]:
    if cfg.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    p = prepared if prepared is not None else prepare(data, cfg)
    slip_e = cfg.slippage_ticks_entry * cfg.tick_size
    tob = cfg.touch_tolerance_ticks * cfg.tick_size
    sob = cfg.stop_buffer_ticks * cfg.tick_size
    tick = cfg.tick_size

    trades: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    pnl_stream: list[float] = []
    rng = np.random.default_rng(cfg.random_seed)

    for d, sl in zip(p.days, p.day_slices):
        o, h, l, c = (p.bid[k][sl] for k in ("open", "high", "low", "close"))
        b_hi, b_lo, b_op, b_cl = (p.bid[k][sl] for k in ("high", "low", "open", "close"))
        a_hi, a_lo, a_op, a_cl = (p.ask[k][sl] for k in ("high", "low", "open", "close"))
        a_c = a_cl
        atr = p.atr1m[sl]
        mins = (p.local[sl].hour * 60 + p.local[sl].minute)
        n = len(c)
        orh, orl = opening_range(h, l, cfg.or_minutes)
        mid = (orh + orl) / 2.0
        osize = max(orh - orl, tick)
        buf = cfg.breakout_close_buffer_frac * osize
        close_buf = cfg.reject_close_buffer_frac * osize

        gap = cfg.trigger_gap_bars
        inv: dict[int, np.ndarray | None] = {1: None, -1: None}
        if cfg.mode == "full":
            inv = {1: inversion_flags(h, l, c, 1, cfg), -1: inversion_flags(h, l, c, -1, cfg)}

        i_e0 = int(np.searchsorted(mins, _hhmm_min(cfg.entry_start), side="left"))
        i_e1 = int(np.searchsorted(mins, _hhmm_min(cfg.entry_end), side="left"))
        i_flat = int(np.searchsorted(mins, _hhmm_min(cfg.flat_time), side="left")) - 1
        i_flat = n - 1 if i_flat <= i_e0 else min(i_flat, n - 1)

        # side +1 = long at ORL, -1 = short at ORH
        touches: dict[int, list[int]] = {1: [], -1: []}
        last_touch = {1: -10**9, -1: -10**9}
        locked = {1: False, -1: False}
        run_len = {1: 0, -1: 0}
        busy_until = -1
        trades_today = 0
        consec_losses = 0
        fired = {1: False, -1: False}      # classic-breakout baseline: first break per side

        def entry_for(side: int, i: int) -> tuple[int, float]:
            """Entry bar and price: the signal bar's close, or the next bar's open."""
            if cfg.entry_mode == "next_open":
                j = i + 1
                if j >= n:                     # signal on the last bar → no next bar
                    return n, 0.0
                return j, ((a_op[j] + slip_e) if side > 0 else (b_op[j] - slip_e))
            return i, ((a_c[i] + slip_e) if side > 0 else (c[i] - slip_e))

        def enter(i0: int, side: int, entry: float, risk: float, target: float) -> None:
            nonlocal busy_until, trades_today, consec_losses
            size = min(cfg.risk_per_trade_pct / 100.0 / (risk / entry), cfg.max_leverage)
            stop_abs = entry - risk if side > 0 else entry + risk
            if side > 0:
                d_hi, d_lo, d_op, d_cl = b_hi, b_lo, b_op, b_cl
            else:
                d_hi, d_lo, d_op, d_cl = a_hi, a_lo, a_op, a_cl
            i_exit, exit_price, reason = _simulate(cfg, i0, i_flat, side, d_hi, d_lo, d_op, d_cl, stop_abs, target)
            move = (exit_price - entry) if side > 0 else (entry - exit_price)
            move_net = move - 2 * cfg.commission_points
            r = move_net / risk
            trades.append({
                "day": d.isoformat(), "direction": "long" if side > 0 else "short",
                "status": "closed", "pnl_pct": round((move_net / entry) * size * 100.0, 4),
                "entry_time": float(p.ts[sl.start + i0]) / 1e9,
                "entry_iso": pd.Timestamp(p.ts[sl.start + i0], tz="UTC").isoformat(),
                "exit_time": float(p.ts[sl.start + i_exit]) / 1e9,
                "exit_iso": pd.Timestamp(p.ts[sl.start + i_exit], tz="UTC").isoformat(),
                "entry": round(entry, 4), "stop": round(stop_abs, 4), "target": round(target, 4),
                "exit": round(exit_price, 4), "exit_reason": reason,
                "risk_points": round(risk, 4), "r": round(r, 4), "mode": cfg.mode,
            })
            signals.append({"day": d.isoformat(), "direction": side})
            pnl_stream.append((move_net / entry) * size * 100.0)
            trades_today += 1
            consec_losses = consec_losses + 1 if r < 0 else 0
            busy_until = i_exit

        for i in range(i_e0, min(i_e1, i_flat) + 1):
            # momentum regime: a side is locked once price accepts beyond it
            for side in (1, -1):
                edge = orl if side > 0 else orh
                beyond = (c[i] < edge - buf) if side > 0 else (c[i] > edge + buf)
                run_len[side] = run_len[side] + 1 if beyond else 0
                if run_len[side] >= cfg.acceptance_bars:
                    locked[side] = True
            if h[i] >= orh - tob and c[i] <= orh + close_buf and i - last_touch[-1] >= cfg.min_bars_between_touches:
                touches[-1].append(i); last_touch[-1] = i
            if l[i] <= orl + tob and c[i] >= orl - close_buf and i - last_touch[1] >= cfg.min_bars_between_touches:
                touches[1].append(i); last_touch[1] = i

            if i <= busy_until or trades_today >= cfg.max_trades_per_day or consec_losses >= cfg.consec_loss_stop:
                continue

            # ── classic ORB breakout baseline (first close beyond an edge, stop mid)
            if cfg.mode == "breakout":
                if not fired[1] and c[i] > orh:
                    i_e, entry = entry_for(1, i)
                    if i_e <= i_flat:
                        fired[1] = True
                        risk = float(np.clip(abs(entry - mid), cfg.min_stop_points, cfg.max_stop_points))
                        enter(i_e, 1, entry, risk, entry + cfg.breakout_target_r * risk)
                elif not fired[-1] and c[i] < orl:
                    i_e, entry = entry_for(-1, i)
                    if i_e <= i_flat:
                        fired[-1] = True
                        risk = float(np.clip(abs(entry - mid), cfg.min_stop_points, cfg.max_stop_points))
                        enter(i_e, -1, entry, risk, entry - cfg.breakout_target_r * risk)
                continue

            # ── random-direction baseline: same touch bars, same stop/target distances
            if cfg.mode == "random":
                cands = [sd for sd in (1, -1) if touches[sd] and touches[sd][-1] == i]
                if not cands or not (orl <= c[i] <= orh):
                    continue
                ts_ = cands[0]
                side = int(rng.choice((1, -1)))
                i_e, entry = entry_for(side, i)
                if i_e > i_flat:
                    continue
                ext = (max(h[j] for j in touches[ts_][-5:] + [i]) if ts_ < 0
                       else min(l[j] for j in touches[ts_][-5:] + [i]))
                raw = (ext + sob - entry) if ts_ < 0 else (entry - (ext - sob))
                risk = float(np.clip(abs(raw), cfg.min_stop_points, cfg.max_stop_points))
                level = (orl + cfg.front_run_ticks * tick) if ts_ < 0 else (orh - cfg.front_run_ticks * tick)
                reward = abs(entry - level)
                if reward / risk < cfg.min_reward_risk:
                    continue
                enter(i_e, side, entry, risk, entry - reward if side < 0 else entry + reward)
                continue

            # ── fade modes: naive / touches / engulf / full
            for side in (1, -1):
                if locked[side] or not (orl <= c[i] <= orh):
                    continue
                tok = touches[side]
                if cfg.mode == "naive":
                    if not tok:
                        continue
                elif len(tok) < cfg.min_touches:
                    continue

                if cfg.mode in ("engulf", "full"):
                    recent = range(max(1, i - cfg.rej_candle_lookback), i + 1)
                    if side > 0:
                        rej = sum(bullish_rejection_candle(o[j], h[j], l[j], c[j], cfg.wick_ratio) for j in recent)
                        eng = bullish_engulfing(o[i], c[i], o[i - 1], c[i - 1], cfg.engulf_min)
                    else:
                        rej = sum(rejection_candle(o[j], h[j], l[j], c[j], cfg.wick_ratio) for j in recent)
                        eng = bearish_engulfing(o[i], c[i], o[i - 1], c[i - 1], cfg.engulf_min)
                    if rej < cfg.rej_candle_min or not eng or (h[i] - l[i]) < cfg.engulf_min_range_atr * atr[i]:
                        continue
                if cfg.mode == "full" and inv[side] is not None and not inv[side][max(0, i - gap):i + 1].any():
                    continue

                i_e, entry = entry_for(side, i)
                if i_e > i_flat:
                    continue
                ref = [h[j] for j in tok[-5:]] + [h[i]] if side < 0 else [l[j] for j in tok[-5:]] + [l[i]]
                extreme = max(ref) if side < 0 else min(ref)
                raw = (extreme + sob - entry) if side < 0 else (entry - (extreme - sob))
                risk = float(np.clip(abs(raw), cfg.min_stop_points, cfg.max_stop_points))
                target = (orh - cfg.front_run_ticks * tick) if side > 0 else (orl + cfg.front_run_ticks * tick)
                reward = abs(entry - target)
                if risk <= 0 or reward / risk < cfg.min_reward_risk:
                    continue
                enter(i_e, side, entry, risk, target)
                break

    equity = [10000.0]
    cum = 0.0
    for pnl in pnl_stream:
        cum += pnl
        equity.append(10000.0 * (1 + cum / 100.0))
    return {"trades": trades, "signals": signals, "equity_curve": equity,
            "metrics": _compute_metrics(trades, equity, cfg)}
