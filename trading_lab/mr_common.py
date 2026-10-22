"""Shared helpers for the mean-reversion candidates.

Used by ``strategies/session_range_fade.py`` and ``strategies/mean_reversion.py``:
  - UTC / closed-bar / gap guards
  - bid+ask frames (historical spread; fixed fallback when no ask side exists)
  - fixed-fractional sizing (RiskManager.calculate_position_size caps at 1x
    notional, which turns a 1% risk into ~0.15% on tight intraday stops)
  - exit handling shared by both strategies (SL, TP1 partial + BE, TP2)
"""
from __future__ import annotations
from .env import env

import os
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import _pos_to_dict
from .risk import Position, RiskManager

ASK_COLS = ("ask_open", "ask_high", "ask_low", "ask_close")

_TF_RE = re.compile(r"^(\d+)\s*(m|min|h)$")


def bar_minutes(timeframe: str) -> int:
    """'15min' / '15m' / '1h' → minutes."""
    m = _TF_RE.match(timeframe.strip().lower())
    if not m:
        raise ValueError(f"Unsupported timeframe {timeframe!r}")
    n = int(m.group(1))
    return n * 60 if m.group(2) == "h" else n


def adapter_interval(timeframe: str) -> str:
    """'15min' → '15m' (market data adapter spelling)."""
    mins = bar_minutes(timeframe)
    return f"{mins // 60}h" if mins % 60 == 0 else f"{mins}m"


def ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Sorted, de-duplicated frame with an explicit tz-aware UTC index."""
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out.index = idx
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def drop_unclosed_last_bar(
    df: pd.DataFrame, minutes: int, now: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Drop the last row while its candle is still forming (lookahead guard)."""
    if df.empty:
        return df
    now_ts = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    if df.index[-1] + pd.Timedelta(minutes=minutes) > now_ts:
        return df.iloc[:-1]
    return df


def with_ask(df: pd.DataFrame, fallback_spread_pct: float) -> pd.DataFrame:
    """Guarantee ask_* columns. open/high/low/close are always the bid side."""
    if all(c in df.columns for c in ASK_COLS):
        return df
    out = df.copy()
    spread = out["close"] * float(fallback_spread_pct)
    for side in ("open", "high", "low", "close"):
        out[f"ask_{side}"] = out[side] + spread
    return out


def prepare_frame(
    df: pd.DataFrame,
    timeframe: str,
    fallback_spread_pct: float,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """UTC index, closed bars only, ask side present."""
    out = ensure_utc(df)
    out = drop_unclosed_last_bar(out, bar_minutes(timeframe), now)
    return with_ask(out, fallback_spread_pct)


def day_slices(local: pd.DatetimeIndex, mask: np.ndarray, *, min_bars: int = 1
               ) -> tuple[list[date], list[slice]]:
    """Per-day slices into the **full** frame, from a boolean bar mask.

    ``local`` is the full frame's index in the strategy's timezone; ``mask``
    marks the bars that belong to a session (e.g. RTH). Returns ``(days,
    slices)`` where every slice indexes the *original* frame — so
    ``bid["open"][sl]`` is that day's session bars.

    The trap this exists to prevent: building positions from the **masked**
    subset (``np.flatnonzero(mask)`` then ``unique(..., return_index=True)``)
    and then applying them to the full frame. That mixes two index spaces and
    silently walks windows out of phase with the clock — the bug that made a
    whole strategy look profitable. Use this helper instead of open-coding it,
    and keep a 24 h-frame test for any new session logic.
    """
    pos = np.flatnonzero(mask)
    if pos.size == 0:
        return [], []
    days_arr = local[pos].normalize().to_numpy()
    uniq, starts, counts = np.unique(days_arr, return_index=True, return_counts=True)
    days: list[date] = []
    slices: list[slice] = []
    for u, s, c in zip(uniq, starts, counts):
        if c < min_bars:
            continue
        days.append(pd.Timestamp(u).date())
        slices.append(slice(int(pos[s]), int(pos[s + c - 1]) + 1))
    return days, slices


def epoch_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    """Bar open times as unix seconds, independent of the index resolution."""
    return index.as_unit("ns").asi8.astype("float64") / 1e9


def gap_minutes_before(index: pd.DatetimeIndex, minutes: int) -> np.ndarray:
    """Missing minutes between bar i-1 and bar i (0 for contiguous bars)."""
    gaps = np.zeros(len(index))
    if len(index) > 1:
        gaps[1:] = np.diff(epoch_seconds(index)) / 60.0 - minutes
    return np.clip(gaps, 0.0, None)


def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate a finer bid(+ask) frame to ``timeframe``; empty buckets dropped."""
    agg: dict[str, str] = {
        "open": "first", "high": "max", "low": "min", "close": "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"
    if all(c in df.columns for c in ASK_COLS):
        agg.update(
            ask_open="first", ask_high="max", ask_low="min", ask_close="last"
        )
    out = df.resample(f"{bar_minutes(timeframe)}min", label="left", closed="left").agg(agg)
    return out.dropna(subset=["close"])


def period_days(df: pd.DataFrame) -> float:
    """Calendar days covered by the frame."""
    if len(df) < 2:
        return 1.0
    return max((df.index[-1] - df.index[0]).total_seconds() / 86400.0, 1.0)


# ── Sizing / fills ──────────────────────────────────────────────────


def open_sized_position(
    rm: RiskManager,
    *,
    entry_price: float,
    stop_loss: float,
    tp1: float,
    tp2: float | None,
    direction: str,
    entry_time: float,
    entry_idx: int,
    atr: float,
    max_leverage: float,
    size_mult: float = 1.0,
    meta: dict[str, Any] | None = None,
) -> Position | None:
    """Fixed-fractional position: size = risk% / stop distance%, capped by leverage."""
    dist_pct = abs(entry_price - stop_loss) / entry_price if entry_price > 0 else 0.0
    if dist_pct <= 0:
        return None
    risk = rm.get_current_risk_per_trade() * float(size_mult)
    uncapped = risk / dist_pct
    size = min(uncapped, float(max_leverage))
    if size <= 0:
        return None
    pos = Position(
        entry_time=entry_time,
        entry_idx=entry_idx,
        entry_price=entry_price,
        direction=direction,
        size=size,
        stop_loss=stop_loss,
        tp1=tp1,
        atr_at_entry=atr,
        tp2=tp2,
    )
    pos.meta = dict(meta or {})
    pos.meta["risk_pct"] = round(size * dist_pct * 100.0, 6)
    pos.meta["leverage_capped"] = bool(uncapped > max_leverage)
    rm.state.trades_today += 1
    return pos


def close_at_market(
    pos: Position,
    rm: RiskManager,
    trades: list[dict],
    *,
    ts: float,
    price: float,
    reason: str,
    slippage_pct: float,
) -> None:
    """Market exit on the exit-side price, slippage against the position."""
    slip = price * float(slippage_pct)
    fill = price - slip if pos.direction == "long" else price + slip
    rm.close_position(pos, fill, reason, ts)
    trades.append(_pos_to_dict(pos))


def manage_stops_and_targets(
    pos: Position,
    rm: RiskManager,
    trades: list[dict],
    *,
    ts: float,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    slippage_pct: float,
) -> None:
    """SL → TP1 (partial, SL to break-even) → TP2 for one bar.

    Prices are the exit side: bid for longs, ask for shorts. When SL and a
    target fall into the same bar the stop wins (conservative). Stops that are
    gapped through fill at the bar open, not at the stop level.
    """
    is_long = pos.direction == "long"

    stopped = bar_low <= pos.stop_loss if is_long else bar_high >= pos.stop_loss
    if stopped:
        level = min(pos.stop_loss, bar_open) if is_long else max(pos.stop_loss, bar_open)
        reason = "breakeven_stop" if pos.sl_moved_to_be else "stop_loss"
        close_at_market(
            pos, rm, trades, ts=ts, price=level, reason=reason,
            slippage_pct=slippage_pct,
        )
        return

    if not pos.tp1_hit:
        hit = bar_high >= pos.tp1 if is_long else bar_low <= pos.tp1
        if hit:
            if rm.cfg.tp1_close_pct >= 0.999 or pos.tp2 is None:
                rm.close_position(pos, pos.tp1, "tp1", ts)
                trades.append(_pos_to_dict(pos))
            else:
                rm.close_position(pos, pos.tp1, "tp1_partial", ts, partial=True)
        return

    if pos.tp2 is not None:
        hit2 = bar_high >= pos.tp2 if is_long else bar_low <= pos.tp2
        if hit2:
            pos.tp2_hit = True
            rm.close_position(pos, pos.tp2, "tp2", ts)
            trades.append(_pos_to_dict(pos))


# ── Data loading (bid + ask, optional local cache) ──────────────────


def _cache_path(asset: str, interval: str, side: str, start: datetime) -> Path | None:
    cache_dir = env("LAB_DATA_CACHE", "").strip()
    if not cache_dir:
        return None
    safe = re.sub(r"[^A-Za-z0-9]+", "", asset)
    today = datetime.now(timezone.utc).date().isoformat()
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{safe}_{interval}_{side}_{start.date().isoformat()}_{today}.pkl"


async def _fetch_side(
    asset: str, interval: str, start: datetime, side: str, source: str
) -> pd.DataFrame:
    from .adapters.market_data import get_market_adapter

    cached = _cache_path(asset, interval, side, start)
    if cached is not None and cached.exists():
        return pd.read_pickle(cached)

    if side == "ask":
        if source not in ("dukascopy", "duka"):
            return pd.DataFrame()
        import dukascopy_python as duka

        from .adapters.dukascopy_adapter import DukascopyAdapter

        adapter: Any = DukascopyAdapter(offer_side=duka.OFFER_SIDE_ASK)
    else:
        adapter = get_market_adapter(source)

    df = await adapter.fetch_history(asset, interval=interval, start=start)
    if cached is not None and not df.empty:
        df.to_pickle(cached)
    return df


async def load_bid_ask(
    asset: str,
    timeframe: str,
    start: datetime,
    *,
    source: str | None = None,
) -> pd.DataFrame:
    """Bid OHLCV plus ask_* columns from the market data adapter.

    Without an ask side (non-Dukascopy source or empty ask fetch) only the bid
    frame is returned; strategies then apply their fixed fallback spread.
    Set LAB_DATA_CACHE to a directory to cache downloads per day.
    """
    src = (source or env("LAB_DATA_SOURCE", "dukascopy")).strip().lower()
    interval = adapter_interval(timeframe)
    bid = await _fetch_side(asset, interval, start, "bid", src)
    if bid.empty:
        return bid
    bid = ensure_utc(bid)
    ask = await _fetch_side(asset, interval, start, "ask", src)
    if ask.empty:
        return bid
    ask = ensure_utc(ask)
    out = bid.join(
        ask[["open", "high", "low", "close"]].add_prefix("ask_"), how="inner"
    )
    # A crossed quote would hand out negative spreads — never let ask sit below bid
    for side in ("open", "high", "low", "close"):
        out[f"ask_{side}"] = np.maximum(out[f"ask_{side}"], out[side])
    return out
