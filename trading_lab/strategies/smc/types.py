"""Shared SMC/ICT types."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


class TrendBias(IntEnum):
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1


class TradeState(str, Enum):
    IDLE = "IDLE"
    HTF_ALIGNED = "HTF_ALIGNED"  # 4h bias set
    SWEEP_TOUCHED = "SWEEP_TOUCHED"  # Stage A: wick raid, await close-back (optional)
    SWEPT = "SWEPT"  # liquidity sweep confirmed
    IN_POI = "IN_POI"  # 15m OB+FVG armed, waiting retest (Stage D)
    M15_MSS = "M15_MSS"  # Stage C: 15m MSS/displacement confirmed
    WAIT_FOR_CHOCH = "WAIT_FOR_CHOCH"  # watch 1m ChoCh
    ENTRY_PENDING = "ENTRY_PENDING"
    IN_TRADE = "IN_TRADE"


@dataclass(frozen=True)
class SwingPoint:
    """Confirmed external swing pivot."""

    idx: int
    ts: Any
    price: float
    kind: int  # 1 = swing high, -1 = swing low
    confirmed_idx: int


@dataclass
class Zone:
    """Untested supply/demand or FVG zone."""

    direction: int  # 1 = bullish (demand), -1 = bearish (supply)
    top: float
    bottom: float
    created_idx: int
    created_ts: Any
    kind: str  # "fvg" | "ob" | "ob_fvg"
    mitigated: bool = False

    @property
    def mid(self) -> float:
        return 0.5 * (self.top + self.bottom)

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top

    def touched(self, high: float, low: float) -> bool:
        return low <= self.top and high >= self.bottom


@dataclass
class LiquidityLevel:
    """Session / prior-day / prior-week liquidity pool."""

    price: float
    kind: str  # asia_high, asia_low, london_high, london_low, pdh, pdl, pwh, pwl
    created_ts: Any
    swept: bool = False
    created_unix: float = 0.0  # for fast sweep checks (avoid pd.Timestamp in hot loop)


@dataclass
class PendingEntry:
    """Limit order waiting to be filled after 1m ChoCh."""

    direction: str  # "long" | "short"
    limit_price: float
    stop_loss: float
    swing_sl_price: float
    zone_top: float
    zone_bottom: float
    placed_idx: int
    placed_ts: Any
    timeout_bars: int
    invalidation_level: float


@dataclass
class StructureSnapshot:
    """Cached structure for one timeframe at a closed bar."""

    bias: TrendBias = TrendBias.NEUTRAL
    last_swing_high: SwingPoint | None = None
    last_swing_low: SwingPoint | None = None
    last_bos_dir: int = 0
    last_choch_dir: int = 0
    swings: list[SwingPoint] = field(default_factory=list)
