"""Causal market structure: swings, BoS, ChoCh (body-close only)."""
from __future__ import annotations

from bisect import bisect_right
from typing import Any

import numpy as np
import pandas as pd

from .types import StructureSnapshot, SwingPoint, TrendBias


def detect_swings(
    high: np.ndarray,
    low: np.ndarray,
    index: Any,
    swing_length: int,
) -> list[SwingPoint]:
    """Confirm pivots only after `swing_length` bars on both sides (causal).

    A swing high at i is emitted when we reach bar i+L and high[i] is max
    of high[i-L : i+L+1]. Same for lows. No forward data beyond confirm bar.
    """
    n = len(high)
    L = max(1, int(swing_length))
    out: list[SwingPoint] = []
    if n < 2 * L + 1:
        return out

    for confirm in range(L, n):
        i = confirm - L
        if i < L:
            continue
        left = i - L
        right = confirm + 1  # exclusive; confirm == i+L
        window_h = high[left:right]
        window_l = low[left:right]
        if high[i] == np.max(window_h):
            out.append(
                SwingPoint(
                    idx=i,
                    ts=index[i],
                    price=float(high[i]),
                    kind=1,
                    confirmed_idx=confirm,
                )
            )
        if low[i] == np.min(window_l):
            out.append(
                SwingPoint(
                    idx=i,
                    ts=index[i],
                    price=float(low[i]),
                    kind=-1,
                    confirmed_idx=confirm,
                )
            )
    return out


def swings_up_to(
    swings: list[SwingPoint], bar_idx: int
) -> list[SwingPoint]:
    """Swings whose confirmation index is <= bar_idx (no lookahead)."""
    return [s for s in swings if s.confirmed_idx <= bar_idx]


def _update_bias_from_swings(swings: list[SwingPoint]) -> TrendBias:
    """External structure bias from last two highs and lows."""
    highs = [s for s in swings if s.kind == 1]
    lows = [s for s in swings if s.kind == -1]
    if len(highs) < 2 or len(lows) < 2:
        return TrendBias.NEUTRAL
    hh = highs[-1].price > highs[-2].price
    hl = lows[-1].price > lows[-2].price
    lh = highs[-1].price < highs[-2].price
    ll = lows[-1].price < lows[-2].price
    if hh and hl:
        return TrendBias.BULLISH
    if lh and ll:
        return TrendBias.BEARISH
    return TrendBias.NEUTRAL


def _bias_from_last(
    last_highs: list[SwingPoint], last_lows: list[SwingPoint]
) -> TrendBias:
    if len(last_highs) < 2 or len(last_lows) < 2:
        return TrendBias.NEUTRAL
    hh = last_highs[-1].price > last_highs[-2].price
    hl = last_lows[-1].price > last_lows[-2].price
    lh = last_highs[-1].price < last_highs[-2].price
    ll = last_lows[-1].price < last_lows[-2].price
    if hh and hl:
        return TrendBias.BULLISH
    if lh and ll:
        return TrendBias.BEARISH
    return TrendBias.NEUTRAL


def body_closes_beyond(close: float, level: float, direction: int) -> bool:
    """Body close beyond level (wick alone does not count — use close only)."""
    if direction > 0:
        return close > level
    return close < level


def structure_break_at_bar(
    close: float,
    swings: list[SwingPoint],
    bias: TrendBias,
) -> tuple[int, int, float | None]:
    """Detect BoS / ChoCh on this closed bar via body close.

    Returns (bos_dir, choch_dir, broken_level).
    bos_dir / choch_dir: 1 bullish, -1 bearish, 0 none.
    """
    highs = [s for s in swings if s.kind == 1]
    lows = [s for s in swings if s.kind == -1]
    if not highs or not lows:
        return 0, 0, None

    last_high = highs[-1].price
    last_low = lows[-1].price
    bos = 0
    choch = 0
    level: float | None = None

    bullish_break = close > last_high
    bearish_break = close < last_low

    if bias == TrendBias.BULLISH:
        if bullish_break:
            bos, level = 1, last_high
        elif bearish_break:
            choch, level = -1, last_low
    elif bias == TrendBias.BEARISH:
        if bearish_break:
            bos, level = -1, last_low
        elif bullish_break:
            choch, level = 1, last_high
    else:
        if bullish_break:
            choch, level = 1, last_high
        elif bearish_break:
            choch, level = -1, last_low

    return bos, choch, level


def _break_from_levels(
    close: float,
    last_high: float | None,
    last_low: float | None,
    bias: TrendBias,
) -> tuple[int, int, float | None]:
    if last_high is None or last_low is None:
        return 0, 0, None
    bos = 0
    choch = 0
    level: float | None = None
    bullish_break = close > last_high
    bearish_break = close < last_low
    if bias == TrendBias.BULLISH:
        if bullish_break:
            bos, level = 1, last_high
        elif bearish_break:
            choch, level = -1, last_low
    elif bias == TrendBias.BEARISH:
        if bearish_break:
            bos, level = -1, last_low
        elif bullish_break:
            choch, level = 1, last_high
    else:
        if bullish_break:
            choch, level = 1, last_high
        elif bearish_break:
            choch, level = -1, last_low
    return bos, choch, level


class MarketStructureEngine:
    """Incremental structure tracker for one timeframe."""

    def __init__(self, swing_length: int = 5):
        self.swing_length = max(1, int(swing_length))
        self._all_swings: list[SwingPoint] = []
        self._confirm_keys: list[int] = []
        self._highs: list[SwingPoint] = []
        self._lows: list[SwingPoint] = []
        self._high_confirm: list[int] = []
        self._low_confirm: list[int] = []
        self._closes: np.ndarray | None = None
        self._precomputed = False

    def precompute(self, df: pd.DataFrame) -> None:
        """Precompute all causal swings for the series."""
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        self._all_swings = detect_swings(high, low, df.index, self.swing_length)
        self._confirm_keys = [s.confirmed_idx for s in self._all_swings]
        self._highs = [s for s in self._all_swings if s.kind == 1]
        self._lows = [s for s in self._all_swings if s.kind == -1]
        self._high_confirm = [s.confirmed_idx for s in self._highs]
        self._low_confirm = [s.confirmed_idx for s in self._lows]
        self._closes = df["close"].to_numpy(dtype=float)
        self._precomputed = True

    def _prefix(self, keys: list[int], items: list[SwingPoint], bar_idx: int) -> list[SwingPoint]:
        n = bisect_right(keys, bar_idx)
        return items[:n]

    def newly_confirmed_swings(self, bar_idx: int) -> list[SwingPoint]:
        """Swings whose confirmation bar is exactly `bar_idx`."""
        if not self._precomputed:
            return []
        out: list[SwingPoint] = []
        # _all_swings is ordered by confirmed_idx
        for s in self._all_swings:
            if s.confirmed_idx == bar_idx:
                out.append(s)
            elif s.confirmed_idx > bar_idx:
                break
        return out

    def nested_choch_dir(
        self, df: pd.DataFrame, bar_idx: int, htf_bias: TrendBias
    ) -> int:
        """1m ChoCh for nested MTF entry relative to HTF bias.

        After HTF is set, LTF pullbacks build *counter-trend* structure. The
        confirmation break back in HTF direction is a ChoCh even when the
        rolling 1m bias already flipped (that flip would otherwise label it BOS).

        Classifies the bar's body break against pullback bias = opposite(HTF).
        Returns ±1 when that break is a ChoCh in HTF direction, else 0.
        """
        if htf_bias == TrendBias.NEUTRAL:
            return 0
        if not self._precomputed:
            self.precompute(df)
        if self._closes is None or not (0 <= bar_idx < len(self._closes)):
            return 0

        nh = bisect_right(self._high_confirm, bar_idx)
        nl = bisect_right(self._low_confirm, bar_idx)
        ph_end = nh
        while ph_end > 0 and self._highs[ph_end - 1].idx >= bar_idx:
            ph_end -= 1
        pl_end = nl
        while pl_end > 0 and self._lows[pl_end - 1].idx >= bar_idx:
            pl_end -= 1

        close = float(self._closes[bar_idx])
        last_h = self._highs[ph_end - 1].price if ph_end else None
        last_l = self._lows[pl_end - 1].price if pl_end else None
        pullback_bias = TrendBias(-int(htf_bias))
        _bos, choch, _ = _break_from_levels(close, last_h, last_l, pullback_bias)
        # ChoCh vs pullback is in HTF direction by construction
        return choch if choch == int(htf_bias) else 0

    def snapshot_at(self, df: pd.DataFrame, bar_idx: int) -> StructureSnapshot:
        """Structure known at the close of `bar_idx` (inclusive). O(log S)."""
        if not self._precomputed:
            self.precompute(df)

        nh = bisect_right(self._high_confirm, bar_idx)
        nl = bisect_right(self._low_confirm, bar_idx)
        # Trim confirmed swings whose pivot is still at bar_idx (not prior)
        ph_end = nh
        while ph_end > 0 and self._highs[ph_end - 1].idx >= bar_idx:
            ph_end -= 1
        pl_end = nl
        while pl_end > 0 and self._lows[pl_end - 1].idx >= bar_idx:
            pl_end -= 1

        last_high = self._highs[nh - 1] if nh else None
        last_low = self._lows[nl - 1] if nl else None
        bias = _bias_from_last(
            self._highs[max(0, nh - 4) : nh],
            self._lows[max(0, nl - 4) : nl],
        )
        bias_break = _bias_from_last(
            self._highs[max(0, ph_end - 4) : ph_end],
            self._lows[max(0, pl_end - 4) : pl_end],
        )

        bos, choch = 0, 0
        if self._closes is not None and 0 <= bar_idx < len(self._closes):
            close = float(self._closes[bar_idx])
            last_h = self._highs[ph_end - 1].price if ph_end else None
            last_l = self._lows[pl_end - 1].price if pl_end else None
            bos, choch, _ = _break_from_levels(close, last_h, last_l, bias_break)

        end = bisect_right(self._confirm_keys, bar_idx)
        swings_tail = self._all_swings[max(0, end - 16) : end]

        return StructureSnapshot(
            bias=bias,
            last_swing_high=last_high,
            last_swing_low=last_low,
            last_bos_dir=bos,
            last_choch_dir=choch,
            swings=swings_tail,
        )

    def bias_at(self, df: pd.DataFrame, bar_idx: int) -> TrendBias:
        return self.snapshot_at(df, bar_idx).bias
