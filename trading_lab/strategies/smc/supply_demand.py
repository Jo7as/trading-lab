"""Causal Fair Value Gap and Order Block detection with zone queues."""
from __future__ import annotations

from collections import deque
from typing import Any, Literal

import numpy as np
import pandas as pd

from .types import Zone

MitigationMode = Literal["wick", "close"]


def detect_fvg_at(
    high: np.ndarray,
    low: np.ndarray,
    index: Any,
    i: int,
) -> Zone | None:
    """FVG confirmed at bar i close using candles i-2, i-1, i.

    Bullish: high[i-2] < low[i]
    Bearish: low[i-2] > high[i]
    """
    if i < 2:
        return None
    if high[i - 2] < low[i]:
        return Zone(
            direction=1,
            top=float(low[i]),
            bottom=float(high[i - 2]),
            created_idx=i,
            created_ts=index[i],
            kind="fvg",
        )
    if low[i - 2] > high[i]:
        return Zone(
            direction=-1,
            top=float(low[i - 2]),
            bottom=float(high[i]),
            created_idx=i,
            created_ts=index[i],
            kind="fvg",
        )
    return None


def detect_all_fvgs(df: pd.DataFrame) -> list[Zone]:
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    out: list[Zone] = []
    for i in range(2, len(df)):
        z = detect_fvg_at(high, low, df.index, i)
        if z is not None:
            out.append(z)
    return out


def _is_bearish_candle(o: float, c: float) -> bool:
    return c < o


def _is_bullish_candle(o: float, c: float) -> bool:
    return c > o


def order_block_for_fvg(
    df: pd.DataFrame,
    fvg: Zone,
    *,
    lookback: int = 10,
) -> Zone | None:
    """Last opposing candle before the FVG displacement (impulsive leg).

    Bullish FVG → last bearish candle in (fvg.created_idx - lookback, fvg-1].
    Bearish FVG → last bullish candle in that window.
    """
    i = fvg.created_idx
    start = max(0, i - lookback)
    # Displacement candle is typically i; search bars before i (and mid i-1)
    search_end = i  # exclusive of confirming bar? include i-1 and earlier
    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)

    ob_idx: int | None = None
    if fvg.direction > 0:
        for j in range(search_end - 1, start - 1, -1):
            if j < 0:
                break
            if _is_bearish_candle(opens[j], closes[j]):
                ob_idx = j
                break
    else:
        for j in range(search_end - 1, start - 1, -1):
            if j < 0:
                break
            if _is_bullish_candle(opens[j], closes[j]):
                ob_idx = j
                break

    if ob_idx is None:
        return None

    top = float(max(opens[ob_idx], closes[ob_idx], highs[ob_idx]))
    bottom = float(min(opens[ob_idx], closes[ob_idx], lows[ob_idx]))
    # Prefer body range; clip to high/low already done. Use full candle OHLC.
    top = float(highs[ob_idx])
    bottom = float(lows[ob_idx])

    return Zone(
        direction=fvg.direction,
        top=top,
        bottom=bottom,
        created_idx=ob_idx,
        created_ts=df.index[ob_idx],
        kind="ob",
    )


def combine_ob_fvg(ob: Zone, fvg: Zone) -> Zone:
    """POI = OB that left an FVG — store merged bounds."""
    return Zone(
        direction=fvg.direction,
        top=max(ob.top, fvg.top),
        bottom=min(ob.bottom, fvg.bottom),
        created_idx=fvg.created_idx,
        created_ts=fvg.created_ts,
        kind="ob_fvg",
    )


def is_mitigated(
    zone: Zone,
    high: float,
    low: float,
    close: float,
    mode: MitigationMode = "wick",
) -> bool:
    """Zone fully filled: price traded through the entire gap/OB."""
    if mode == "close":
        if zone.direction > 0:
            # bullish FVG mitigated when close through bottom
            return close <= zone.bottom
        return close >= zone.top
    # wick: any trade that covers the opposite edge fully through zone
    if zone.direction > 0:
        return low <= zone.bottom
    return high >= zone.top


class SupplyDemandEngine:
    """Queue of untested OB+FVG POIs."""

    def __init__(
        self,
        *,
        ob_require_fvg: bool = True,
        fvg_mitigation: MitigationMode = "wick",
        max_zones: int = 32,
        ob_lookback: int = 10,
    ):
        self.ob_require_fvg = ob_require_fvg
        self.fvg_mitigation = fvg_mitigation
        self.max_zones = max_zones
        self.ob_lookback = ob_lookback
        self.zones: deque[Zone] = deque()
        self._last_scanned = 1  # last bar index examined for new FVGs

    def reset(self) -> None:
        self.zones.clear()
        self._last_scanned = 1

    def update_through(self, df: pd.DataFrame, bar_idx: int) -> None:
        """Add new OB+FVG zones and mitigate with bars up to bar_idx."""
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)

        start = max(2, self._last_scanned + 1)
        for i in range(start, bar_idx + 1):
            fvg = detect_fvg_at(high, low, df.index, i)
            if fvg is not None:
                if self.ob_require_fvg:
                    ob = order_block_for_fvg(df, fvg, lookback=self.ob_lookback)
                    if ob is not None:
                        zone = combine_ob_fvg(ob, fvg)
                        self.zones.append(zone)
                        while len(self.zones) > self.max_zones:
                            self.zones.popleft()
                else:
                    self.zones.append(fvg)
                    while len(self.zones) > self.max_zones:
                        self.zones.popleft()

            self._mitigate_bar(float(high[i]), float(low[i]), float(close[i]))

        self._last_scanned = max(self._last_scanned, bar_idx)

    def _mitigate_bar(self, high: float, low: float, close: float) -> None:
        keep: deque[Zone] = deque()
        for z in self.zones:
            if z.mitigated:
                continue
            if is_mitigated(z, high, low, close, self.fvg_mitigation):
                z.mitigated = True
                continue
            keep.append(z)
        self.zones = keep

    def active_zones(self, direction: int | None = None) -> list[Zone]:
        out = [z for z in self.zones if not z.mitigated]
        if direction is not None:
            out = [z for z in out if z.direction == direction]
        return out

    def zone_containing(
        self, price: float, direction: int | None = None
    ) -> Zone | None:
        for z in reversed(self.active_zones(direction)):
            if z.contains(price):
                return z
        return None

    def zone_touched(
        self, high: float, low: float, direction: int | None = None
    ) -> Zone | None:
        for z in reversed(self.active_zones(direction)):
            if z.touched(high, low):
                return z
        return None
