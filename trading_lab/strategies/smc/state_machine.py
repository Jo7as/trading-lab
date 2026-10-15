"""SMC trade lifecycle state machine — nested MTF entry.

Flow:
  IDLE → HTF_ALIGNED (4h bias)
       → SWEPT (liquidity sweep)
       → WAIT_FOR_CHOCH (15m aligns/ChoCh toward 4h; immediately watch 1m)
       → ENTRY_PENDING (1m ChoCh + 1m OB pullback limit)
       → IN_TRADE

15m OB+FVG is optional context (stored when available). POI retest is NOT
required before 1m ChoCh — that can be re-enabled later via evolution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .types import PendingEntry, TradeState, TrendBias, Zone


@dataclass
class SmcContext:
    """Mutable context shared across engines during backtest/live ticks."""

    state: TradeState = TradeState.IDLE
    bias: TrendBias = TrendBias.NEUTRAL  # 4h bias
    last_sweep_kind: str | None = None
    poi_zone: Zone | None = None  # optional 15m context
    pending: PendingEntry | None = None
    recent_sweep: bool = False
    m15_aligned: bool = False
    # Flow stages A–C (see SMC_FLOW_SPEC.md)
    sweep_level: float | None = None
    sweep_extreme: float | None = None
    dealing_range_low: float | None = None
    dealing_range_high: float | None = None
    m15_mss_bar_i: int = 0


@dataclass
class StateMachine:
    """Nested: 4h → Sweep → 15m align → watch 1m ChoCh → 1m OB entry."""

    ctx: SmcContext = field(default_factory=SmcContext)

    def reset_setup(self) -> None:
        self.ctx.state = TradeState.IDLE
        self.ctx.last_sweep_kind = None
        self.ctx.poi_zone = None
        self.ctx.pending = None
        self.ctx.recent_sweep = False
        self.ctx.m15_aligned = False
        self.ctx.sweep_level = None
        self.ctx.sweep_extreme = None
        self.ctx.dealing_range_low = None
        self.ctx.dealing_range_high = None
        self.ctx.m15_mss_bar_i = 0

    def on_4h_bias(self, bias: TrendBias) -> None:
        """Primary trend filter: 4h only."""
        prev = self.ctx.bias
        self.ctx.bias = bias
        if bias == TrendBias.NEUTRAL:
            if self.ctx.state not in (TradeState.IN_TRADE, TradeState.ENTRY_PENDING):
                self.reset_setup()
                self.ctx.bias = TrendBias.NEUTRAL
            return
        if prev != bias and self.ctx.state not in (
            TradeState.IN_TRADE,
            TradeState.ENTRY_PENDING,
        ):
            self.reset_setup()
            self.ctx.bias = bias
            self.ctx.state = TradeState.HTF_ALIGNED
            return
        if self.ctx.state == TradeState.IDLE:
            self.ctx.state = TradeState.HTF_ALIGNED

    def on_htf_bias(self, bias: TrendBias) -> None:
        self.on_4h_bias(bias)

    def on_sweep(self, kind: str, sweep_dir: int = 0) -> bool:
        """Record liquidity sweep in favor of 4h bias."""
        if self.ctx.bias == TrendBias.NEUTRAL:
            return False
        if self.ctx.state not in (
            TradeState.HTF_ALIGNED,
            TradeState.SWEPT,
            TradeState.IN_POI,
            TradeState.WAIT_FOR_CHOCH,
        ):
            return False
        want_low = self.ctx.bias == TrendBias.BULLISH
        is_low = kind.endswith("_low") or kind in ("pdl", "pwl")
        is_high = kind.endswith("_high") or kind in ("pdh", "pwh")
        hit = (want_low and is_low) or ((not want_low) and is_high)
        if not hit:
            return False
        self.ctx.recent_sweep = True
        self.ctx.last_sweep_kind = kind
        if self.ctx.state == TradeState.HTF_ALIGNED:
            self.ctx.state = TradeState.SWEPT
        return True

    def on_15m_align(
        self,
        bias_15m: TrendBias,
        choch_dir: int,
        poi: Zone | None,
    ) -> bool:
        """After sweep: 15m aligns or ChoCh toward 4h → start watching 1m.

        POI is optional context. Returns True when 1m watch is armed.
        """
        if self.ctx.bias == TrendBias.NEUTRAL or not self.ctx.recent_sweep:
            return False
        if self.ctx.state not in (
            TradeState.SWEPT,
            TradeState.IN_POI,
            TradeState.WAIT_FOR_CHOCH,
        ):
            return False

        aligned = bias_15m == self.ctx.bias
        choch_ok = choch_dir == int(self.ctx.bias) and choch_dir != 0
        if not (aligned or choch_ok):
            # Do not clear m15_aligned — 15m can chop after we've started
            # watching; wiping it makes cancel_pending drop the HTF sweep.
            return False

        self.ctx.m15_aligned = True
        if poi is not None and poi.direction == int(self.ctx.bias):
            self.ctx.poi_zone = poi
        # Immediately watch 1m for ChoCh (no 15m POI retest gate)
        self.ctx.state = TradeState.WAIT_FOR_CHOCH
        return True

    def on_poi_retest(self, high: float, low: float) -> bool:
        """Optional filter (disabled in main path; kept for evolution/experiments)."""
        if self.ctx.state not in (TradeState.IN_POI, TradeState.WAIT_FOR_CHOCH):
            return False
        z = self.ctx.poi_zone
        if z is None or z.mitigated:
            return False
        return z.touched(high, low)

    def on_poi_touch(self, zone: Zone) -> None:
        """Store optional 15m POI context after sweep."""
        if not self.ctx.recent_sweep or self.ctx.bias == TrendBias.NEUTRAL:
            return
        if zone.direction != int(self.ctx.bias):
            return
        if self.ctx.state in (TradeState.SWEPT, TradeState.WAIT_FOR_CHOCH, TradeState.IN_POI):
            self.ctx.poi_zone = zone

    def on_choch(self, choch_dir: int) -> bool:
        """1m ChoCh in 4h direction while watching — arm pullback entry."""
        if self.ctx.state != TradeState.WAIT_FOR_CHOCH:
            return False
        if choch_dir == 0 or choch_dir != int(self.ctx.bias):
            return False
        self.ctx.state = TradeState.ENTRY_PENDING
        return True

    def set_pending(self, pending: PendingEntry) -> None:
        self.ctx.pending = pending
        self.ctx.state = TradeState.ENTRY_PENDING

    def on_fill(self) -> None:
        """Fill consumes the sweep — next trade needs a fresh session/PD/PW sweep."""
        self.ctx.state = TradeState.IN_TRADE
        self.ctx.pending = None
        self.ctx.recent_sweep = False
        self.ctx.m15_aligned = False
        self.ctx.last_sweep_kind = None
        self.ctx.poi_zone = None
        self.ctx.sweep_level = None
        self.ctx.sweep_extreme = None
        self.ctx.dealing_range_low = None
        self.ctx.dealing_range_high = None
        self.ctx.m15_mss_bar_i = 0

    def on_trade_closed(self) -> None:
        """Back to HTF bias; do not reuse the prior sweep."""
        bias = self.ctx.bias
        self.reset_setup()
        if bias != TrendBias.NEUTRAL:
            self.ctx.bias = bias
            self.ctx.state = TradeState.HTF_ALIGNED

    def on_new_trading_day(self) -> None:
        """NY day rollover — stale sweeps expire; wait for today's liquidity."""
        if self.ctx.state == TradeState.IN_TRADE:
            self.ctx.recent_sweep = False
            self.ctx.m15_aligned = False
            self.ctx.last_sweep_kind = None
            self.ctx.poi_zone = None
            self.ctx.pending = None
            return
        bias = self.ctx.bias
        self.reset_setup()
        if bias != TrendBias.NEUTRAL:
            self.ctx.bias = bias
            self.ctx.state = TradeState.HTF_ALIGNED

    def cancel_pending(self) -> None:
        self.ctx.pending = None
        if self.ctx.recent_sweep and self.ctx.m15_aligned:
            self.ctx.state = TradeState.WAIT_FOR_CHOCH
        elif self.ctx.recent_sweep:
            # Keep HTF sweep; wait for 15m to re-affirm
            self.ctx.state = TradeState.SWEPT
        elif self.ctx.bias != TrendBias.NEUTRAL:
            self.ctx.state = TradeState.HTF_ALIGNED
            self.ctx.poi_zone = None
            self.ctx.m15_aligned = False
        else:
            self.reset_setup()
