"""Risk Manager — position sizing, loss limits, cooldown, SL/TP calculation.

Stateful: tracks daily P&L, weekly P&L, cooldown timers, and open positions.
Designed for both backtest (path-dependent loop) and live execution.

Backtest sizing uses a fixed account_size (default 10_000): each trade risks
risk_per_trade × account_size regardless of cumulative P&L (no compounding).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol


class RiskConfig(Protocol):
    risk_per_trade: float
    sl_atr_mult: float
    tp1_r: float
    tp1_close_pct: float
    max_daily_loss: float
    cooldown_hours: int


@dataclass
class Position:
    """Represents an open or closed trade."""
    entry_time: float          # unix timestamp
    entry_idx: int             # candle index in the DataFrame
    entry_price: float
    direction: str             # "long" | "short"
    size: float                # fraction of account (0.0–1.0)
    stop_loss: float
    tp1: float
    atr_at_entry: float
    tp1_hit: bool = False
    tp2: float | None = None   # optional second target (e.g. SMC fixed RR)
    tp2_hit: bool = False
    sl_moved_to_be: bool = False
    exit_time: float | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    pnl_pct: float | None = None       # total account-level % PnL when closed
    equity_pnl_pct: float = 0.0        # running account-level % across legs
    status: str = "open"               # "open" | "closed"
    partial_closed: bool = False
    remaining_size: float = 1.0        # fraction of position still open
    meta: dict[str, Any] | None = None  # optional setup tags (SMC attribution)


@dataclass
class RiskState:
    """Mutable risk state tracked across trades."""
    equity: float = 10000.0
    cumulative_equity_pnl_pct: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    day_start_ts: float = 0.0
    week_start_ts: float = 0.0
    last_loss_ts: float = 0.0
    in_cooldown: bool = False
    trades_today: int = 0
    total_trades: int = 0
    consecutive_losses: int = 0


class RiskManager:
    """Centralized risk management with fixed-account backtest sizing."""

    def __init__(
        self, cfg: RiskConfig, account_size: float = 10000.0, *, backtest_mode: bool = False
    ):
        self.cfg = cfg
        self.account_size = account_size
        self.backtest_mode = backtest_mode
        self._sim_clock_initialized = False
        self.state = RiskState()
        self.state.equity = account_size
        if backtest_mode:
            # Anchored on first simulated timestamp, not wall clock
            self.state.day_start_ts = 0.0
            self.state.week_start_ts = 0.0
        else:
            self.state.day_start_ts = time.time()
            self.state.week_start_ts = time.time()

    @property
    def current_equity(self) -> float:
        """Non-compounded equity: account_size + linear P&L accumulation."""
        return self.account_size * (
            1 + self.state.cumulative_equity_pnl_pct / 100
        )

    # ── Daily / weekly reset ───────────────────────────────────────

    def _maybe_reset_periods(self, now: float) -> None:
        """Reset daily/weekly P&L if a new day/week has started."""
        day_seconds = 86400
        week_seconds = 7 * day_seconds

        if self.backtest_mode and not self._sim_clock_initialized:
            self.state.day_start_ts = now
            self.state.week_start_ts = now
            self._sim_clock_initialized = True

        if now - self.state.day_start_ts >= day_seconds:
            self.state.daily_pnl = 0.0
            self.state.trades_today = 0
            self.state.day_start_ts = now

        if now - self.state.week_start_ts >= week_seconds:
            self.state.weekly_pnl = 0.0
            self.state.week_start_ts = now

    # ── Cooldown ───────────────────────────────────────────────────

    def _check_cooldown(self, now: float) -> bool:
        """Return True if still in cooldown."""
        if not self.state.in_cooldown:
            return False
        cooldown_seconds = self.cfg.cooldown_hours * 3600
        if now - self.state.last_loss_ts >= cooldown_seconds:
            self.state.in_cooldown = False
            return False
        return True

    # ── Gate check ─────────────────────────────────────────────────

    def can_trade(self, now: float) -> tuple[bool, str]:
        """Check if new trades are allowed. Returns (allowed, reason)."""
        self._maybe_reset_periods(now)

        if self._check_cooldown(now):
            remaining = self.cfg.cooldown_hours * 3600 - (now - self.state.last_loss_ts)
            return False, f"cooldown {remaining:.0f}s remaining"

        if self.state.daily_pnl <= -self.cfg.max_daily_loss:
            return False, f"daily loss limit hit ({self.state.daily_pnl:.2%})"

        return True, "ok"

    def get_current_risk_per_trade(self) -> float:
        """Return adjusted risk per trade based on daily P&L."""
        if self.state.daily_pnl <= -self.cfg.max_daily_loss * 0.5:
            return self.cfg.risk_per_trade * 0.5
        return self.cfg.risk_per_trade

    # ── Position sizing ────────────────────────────────────────────

    def calculate_position_size(
        self, entry_price: float, stop_loss: float
    ) -> float:
        """Position size as fraction of fixed account (no compounding)."""
        risk_per_trade = self.get_current_risk_per_trade()
        risk_amount = self.account_size * risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0:
            return 0.0
        units = risk_amount / risk_per_unit
        notional = units * entry_price
        size = notional / self.account_size if self.account_size > 0 else 0.0
        return min(max(size, 0.0), 1.0)

    # ── SL / TP calculation ────────────────────────────────────────

    def calculate_sl_tp(
        self, entry_price: float, atr: float, direction: str
    ) -> tuple[float, float]:
        """Calculate stop-loss and take-profit-1 levels."""
        sl_dist = self.cfg.sl_atr_mult * atr
        risk = sl_dist

        if direction == "long":
            sl = entry_price - sl_dist
            tp1 = entry_price + self.cfg.tp1_r * risk
        else:
            sl = entry_price + sl_dist
            tp1 = entry_price - self.cfg.tp1_r * risk

        return sl, tp1

    def _price_move_pct(self, pos: Position, exit_price: float) -> float:
        if pos.direction == "long":
            return (exit_price - pos.entry_price) / pos.entry_price * 100
        return (pos.entry_price - exit_price) / pos.entry_price * 100

    def _apply_leg_pnl(self, pos: Position, leg_equity_pnl: float) -> None:
        """Book account-level P&L from one close leg (percent of account)."""
        pos.equity_pnl_pct += leg_equity_pnl
        self.state.cumulative_equity_pnl_pct += leg_equity_pnl
        self.state.equity = self.current_equity
        frac = leg_equity_pnl / 100
        self.state.daily_pnl += frac
        self.state.weekly_pnl += frac

    # ── Trade lifecycle ────────────────────────────────────────────

    def open_position(
        self,
        entry_price: float,
        atr: float,
        direction: str,
        entry_time: float,
        entry_idx: int,
        *,
        stop_loss: float | None = None,
        tp1: float | None = None,
        tp2: float | None = None,
    ) -> Position | None:
        """Open a new position with risk-managed sizing.

        Optional stop_loss/tp1/tp2 override ATR-derived levels (structure/fixed RR).
        """
        self._maybe_reset_periods(entry_time)
        sl_atr, tp1_atr = self.calculate_sl_tp(entry_price, atr, direction)
        sl = float(stop_loss) if stop_loss is not None else sl_atr
        tp1_level = float(tp1) if tp1 is not None else tp1_atr
        size = self.calculate_position_size(entry_price, sl)
        if size <= 0:
            return None

        pos = Position(
            entry_time=entry_time,
            entry_idx=entry_idx,
            entry_price=entry_price,
            direction=direction,
            size=size,
            stop_loss=sl,
            tp1=tp1_level,
            atr_at_entry=atr,
            tp2=float(tp2) if tp2 is not None else None,
        )
        self.state.trades_today += 1
        return pos

    def close_position(
        self, pos: Position, exit_price: float, reason: str, exit_time: float,
        partial: bool = False,
        *,
        move_sl_to_be: bool = True,
    ) -> float:
        """Close (or partially close) a position.

        Returns realized account-level P&L in percent (of fixed account_size).
        On partial close, ``move_sl_to_be`` (default True) moves stop to entry
        so runners are protected; set False for ICT-style ERL runners.
        """
        self._maybe_reset_periods(exit_time)
        price_pnl_pct = self._price_move_pct(pos, exit_price)
        close_fraction = (
            self.cfg.tp1_close_pct if partial else pos.remaining_size
        )
        leg_equity_pnl = price_pnl_pct * pos.size * close_fraction

        if partial:
            pos.partial_closed = True
            pos.remaining_size -= close_fraction
            pos.tp1_hit = True
            if move_sl_to_be:
                pos.sl_moved_to_be = True
                pos.stop_loss = pos.entry_price
            self._apply_leg_pnl(pos, leg_equity_pnl)
            return leg_equity_pnl

        self._apply_leg_pnl(pos, leg_equity_pnl)

        pos.exit_time = exit_time
        pos.exit_price = exit_price
        pos.exit_reason = reason
        pos.pnl_pct = round(pos.equity_pnl_pct, 4)
        pos.status = "closed"
        pos.remaining_size = 0.0

        self.state.total_trades += 1
        if pos.equity_pnl_pct < 0:
            self.state.consecutive_losses += 1
            self.state.last_loss_ts = exit_time
            self.state.in_cooldown = True
        else:
            self.state.consecutive_losses = 0

        return leg_equity_pnl

    def get_stats(self) -> dict[str, Any]:
        """Return current risk state as dict."""
        return {
            "account_size": round(self.account_size, 2),
            "equity": round(self.current_equity, 2),
            "cumulative_pnl_pct": round(self.state.cumulative_equity_pnl_pct, 4),
            "daily_pnl": round(self.state.daily_pnl, 4),
            "weekly_pnl": round(self.state.weekly_pnl, 4),
            "in_cooldown": self.state.in_cooldown,
            "trades_today": self.state.trades_today,
            "total_trades": self.state.total_trades,
            "consecutive_losses": self.state.consecutive_losses,
        }
