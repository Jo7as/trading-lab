"""Supertrend + Volume Intraday Strategy for Gold (GC=F).

Designed for 1h candles via yfinance. Produces 5-8 trades/week.
Suitable for FTMO-style prop firm evaluation (5% daily DD, 10% overall DD).

This module reuses the existing indicator functions (atr, supertrend, ema)
and RiskManager from trading_lab. It only adds:
  - Supertrend flip detection
  - Volume/ATR confirmation filter
  - Modified position management with Supertrend trailing
  - New config dataclass with searchable parameter space

Usage:
  from trading_lab.strategies.supertrend_volume import (
      SupertrendVolumeConfig, run_backtest
  )

  cfg = SupertrendVolumeConfig()
  result = run_backtest(df, cfg)
  print(result["metrics"])
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..config import load_config_from_yaml  # reuse YAML loader pattern
from ..indicators import atr, ema, supertrend
from ..risk import RiskManager, Position
from ..backtest import _compute_metrics, _pos_to_dict, print_metrics


@dataclass(frozen=True)
class SupertrendVolumeConfig:
    """Configuration for the Supertrend + Volume intraday strategy.

    All parameters have explicit bounds for the hill-climb optimizer.
    """

    # ── Asset ──────────────────────────────────────────────────────
    asset: str = "XAU/USD"
    asset_label: str = "XAU/USD"

    # ── Supertrend (core signal generator) ─────────────────────────
    supertrend_period: int = 7
    supertrend_mult: float = 1.5     # sensitive — more flips than 2.0+

    # ── Volume filter ──────────────────────────────────────────────
    volume_ma_period: int = 20
    volume_mult: float = 1.0

    # ── Trend filter (EMA alignment) ────────────────────────────────
    ema_fast: int = 3                # 3/3 = no EMA filter → more signals
    ema_slow: int = 3

    # ── ATR / volatility filter ────────────────────────────────────
    atr_period: int = 14
    min_atr_pct: float = 0.0

    # ── Entry modes ───────────────────────────────────────────────
    pullback_entry: bool = False     # optional — adds freq but hurts PnL if always on
    pullback_tolerance: float = 0.002

    # ── Account / risk management ──────────────────────────────────
    account_size: float = 10000.0    # fixed notional base (no compounding)
    risk_per_trade: float = 0.01     # 1% of account_size per trade
    sl_atr_mult: float = 1.5         # Stop loss = 1.5 × ATR
    tp1_r: float = 2.0               # TP1 at 2.0R
    tp1_close_pct: float = 0.60      # Close 60% at TP1
    max_hold_hours: int = 48         # Time stop (candle count for 1h)
    max_daily_loss: float = 0.03      # 3% daily loss limit
    cooldown_hours: int = 0          # no pause after loss → higher frequency
    max_concurrent_positions: int = 1
    long_only: bool = True           # shorts enabled only via OOS gate

    # ── Execution model (added 2026-10, reviewed) ──────────────────
    timeframe: str = "1h"            # bar size of `df`; drives the time stop
    slippage_pct: float = 0.00002    # adverse slippage per fill (2 bp, as elsewhere)
    random_entry: bool = False       # baseline: same signal bars, random direction
    random_seed: int = 7

    # ── Searchable parameter space for hill-climb optimizer ────────
    @staticmethod
    def search_space() -> list[tuple[str, float, float, float]]:
        """Return (param_name, min, max, step) for each searchable parameter."""
        return [
            # Core Supertrend — controls signal frequency (most sensitive)
            ("supertrend_period",     7,    14,   1),
            ("supertrend_mult",       1.4,  3.0,  0.2),

            # Volume / volatility filter — controls signal quality
            ("volume_ma_period",      10,   50,   5),
            ("volume_mult",           0.8,  1.5,  0.1),
            ("min_atr_pct",           0.0,  0.002, 0.0005),

            # Trend filter — controls signal alignment
            ("ema_fast",              3,    50,   3),
            ("ema_slow",              30,   150,  10),

            # Risk management — controls per-trade risk and exit
            ("sl_atr_mult",           1.0,  2.5,  0.2),
            ("tp1_r",                 1.5,  3.5,  0.2),
            ("tp1_close_pct",         0.40, 0.90, 0.05),
            ("max_hold_hours",        4,    48,   2),
            ("cooldown_hours",        0,    8,    1),
        ]

    # ── Derived: the time stop must not depend on the bar size ─────
    _BARS_PER_HOUR = {"1m": 60.0, "5m": 12.0, "15m": 4.0, "30m": 2.0,
                      "1h": 1.0, "4h": 0.25, "1d": 1.0 / 24.0}

    @property
    def bars_per_hour(self) -> float:
        if self.timeframe not in self._BARS_PER_HOUR:
            raise ValueError(f"unknown timeframe {self.timeframe!r}; "
                             f"known: {sorted(self._BARS_PER_HOUR)}")
        return self._BARS_PER_HOUR[self.timeframe]

    @property
    def max_hold_bars(self) -> int:
        """``max_hold_hours`` converted to bars of ``timeframe``."""
        return max(1, int(round(self.max_hold_hours * self.bars_per_hour)))


def _compute_volume_filter(
    df: pd.DataFrame, period: int, mult: float
) -> pd.Series:
    """Volume filter using rolling median, with bar-range fallback for sparse data.

    When yfinance returns mostly-zero volume (common for GC=F), falls back to
    comparing candle range (high-low) against its rolling median.

    Returns float Series: 1.0 = activity OK, 0.0 = too quiet.
    """
    volume = df["volume"]
    vol_median = volume.rolling(period).median()
    vol_ok = (volume > vol_median * mult).astype(float)

    nonzero_ratio = (volume > 0).rolling(100).mean()
    sparse_mask = (nonzero_ratio < 0.5).fillna(False)

    if sparse_mask.any():
        bar_range = df["high"] - df["low"]
        range_median = bar_range.rolling(period).median()
        range_ok = (bar_range > range_median * mult).astype(float)
        vol_ok = vol_ok.where(~sparse_mask, range_ok)

    return vol_ok


def compute_indicators(
    df: pd.DataFrame, cfg: SupertrendVolumeConfig
) -> pd.DataFrame:
    """Compute all indicators needed for signal generation.

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume
            Must be sorted ascending by timestamp.
        cfg: Strategy configuration.

    Returns:
        Copy of df with indicator columns appended:
            atr, atr_pct, supertrend, st_dir, st_flip_up, st_flip_down,
            vol_ok, ema_fast, ema_slow, atr_ok
    """
    out = df.copy()

    # ATR and volatility ratio
    out["atr"] = atr(
        out["high"], out["low"], out["close"], cfg.atr_period
    )
    out["atr_pct"] = out["atr"] / out["close"]

    # Supertrend
    st = supertrend(
        out["high"], out["low"], out["close"],
        cfg.supertrend_period, cfg.supertrend_mult
    )
    out["supertrend"] = st["supertrend"]
    out["st_dir"] = st["st_dir"]

    # Detect Supertrend direction flips (the core entry signal)
    out["st_flip_up"] = (
        (out["st_dir"] == 1.0) & (out["st_dir"].shift(1) == -1.0)
    ).astype(float)
    out["st_flip_down"] = (
        (out["st_dir"] == -1.0) & (out["st_dir"].shift(1) == 1.0)
    ).astype(float)

    # Volume filter (median-based, range fallback for sparse yfinance data)
    out["vol_ok"] = _compute_volume_filter(
        out, cfg.volume_ma_period, cfg.volume_mult
    )

    # EMA trend filter
    out["ema_fast_val"] = ema(out["close"], cfg.ema_fast)
    out["ema_slow_val"] = ema(out["close"], cfg.ema_slow)

    # ATR minimum volatility filter
    out["atr_ok"] = (out["atr_pct"] > cfg.min_atr_pct).astype(float)

    return out


def generate_signals(
    df: pd.DataFrame, cfg: SupertrendVolumeConfig
) -> pd.DataFrame:
    """Generate long/short entry signals.

    Entry conditions (all must be true at candle close):
      LONG (flip):
        1. Supertrend flips from -1 to +1 (st_flip_up)
        2. Close > Supertrend line (confirms flip)
        3. Volume > median × volume_mult (or ATR proxy if volume sparse)
        4. EMA(fast) > EMA(slow) (trend alignment)
        5. ATR/price > min_atr_pct (minimum volatility)

      LONG (pullback, if pullback_entry):
        1. Supertrend already bullish (st_dir == 1 for 2+ bars)
        2. Wick touches Supertrend line (low within tolerance)
        3. Close > Supertrend line
        4. Same volume / EMA / ATR filters as flip

      SHORT: mirror of long

    Returns df with long_signal and short_signal columns (1.0 = signal, 0.0 = no signal).
    """
    out = compute_indicators(df, cfg)

    # Trend filter — optional. If ema_fast == ema_slow, skip filter
    if cfg.ema_fast != cfg.ema_slow:
        trend_up = (out["ema_fast_val"] > out["ema_slow_val"]).astype(float)
        trend_down = (out["ema_fast_val"] < out["ema_slow_val"]).astype(float)
    else:
        # Degenerate case: no EMA trend filter (allow both directions)
        trend_up = pd.Series(1.0, index=out.index)
        trend_down = pd.Series(1.0, index=out.index)

    flip_long = (
        (out["st_flip_up"] > 0) &
        (out["close"] > out["supertrend"]) &
        (out["vol_ok"] > 0) &
        (trend_up > 0) &
        (out["atr_ok"] > 0)
    )
    flip_short = (
        (out["st_flip_down"] > 0) &
        (out["close"] < out["supertrend"]) &
        (out["vol_ok"] > 0) &
        (trend_down > 0) &
        (out["atr_ok"] > 0)
    )

    if cfg.pullback_entry:
        st_up = (out["st_dir"] == 1) & (out["st_dir"].shift(1) == 1)
        st_down = (out["st_dir"] == -1) & (out["st_dir"].shift(1) == -1)
        tol = 1.0 + cfg.pullback_tolerance
        touch_long = out["low"] <= out["supertrend"] * tol
        touch_short = out["high"] >= out["supertrend"] / tol
        pullback_long = (
            st_up & touch_long &
            (out["close"] > out["supertrend"]) &
            (out["st_flip_up"] == 0) &
            (out["vol_ok"] > 0) & (trend_up > 0) & (out["atr_ok"] > 0)
        )
        pullback_short = (
            st_down & touch_short &
            (out["close"] < out["supertrend"]) &
            (out["st_flip_down"] == 0) &
            (out["vol_ok"] > 0) & (trend_down > 0) & (out["atr_ok"] > 0)
        )
        long_cond = flip_long | pullback_long
        short_cond = flip_short | pullback_short
    else:
        long_cond = flip_long
        short_cond = flip_short

    out["long_signal"] = long_cond.astype(float)
    out["short_signal"] = short_cond.astype(float)

    return out


def manage_position(
    pos: Position,
    row: pd.Series,
    idx: int,
    current_ts: float,
    current_price: float,
    current_high: float,
    current_low: float,
    cfg: SupertrendVolumeConfig,
    rm: RiskManager,
    trades: list[dict],
) -> None:
    """Check all exit conditions for an open position.

    Exit hierarchy (checked in order):
      1. Stop loss (intrabar via high/low)
      2. TP1 (partial close, move SL to breakeven)
      3. After TP1: Supertrend trailing exit
         - Ratchet SL to Supertrend line (only in favorable direction)
         - Exit on Supertrend flip against position
      4. Time stop (max_hold_hours candles after entry)
    """
    slip = cfg.slippage_pct

    # 1. Stop loss (intrabar; a gap fills at the bar's open, not at the stop)
    if pos.direction == "long":
        if current_low <= pos.stop_loss:
            fill = min(pos.stop_loss, float(row["open"])) * (1 - slip)
            rm.close_position(pos, fill, "stop_loss", current_ts)
            trades.append(_pos_to_dict(pos))
            return
    else:
        if current_high >= pos.stop_loss:
            fill = max(pos.stop_loss, float(row["open"])) * (1 + slip)
            rm.close_position(pos, fill, "stop_loss", current_ts)
            trades.append(_pos_to_dict(pos))
            return

    # 2. TP1 (if not yet hit — partial close)
    if not pos.tp1_hit:
        if pos.direction == "long" and current_high >= pos.tp1:
            rm.close_position(
                pos, pos.tp1, "tp1_partial", current_ts, partial=True
            )
            return
        elif pos.direction == "short" and current_low <= pos.tp1:
            rm.close_position(
                pos, pos.tp1, "tp1_partial", current_ts, partial=True
            )
            return

    # 3. After TP1: trail using Supertrend line + exit on flip
    if pos.tp1_hit:
        st_dir = float(row["st_dir"])
        st_line = float(row["supertrend"])

        if pos.direction == "long":
            # Ratchet SL up to Supertrend line (never lower it)
            pos.stop_loss = max(pos.stop_loss, st_line)
            if st_dir < 0:
                rm.close_position(
                    pos, current_price * (1 - slip), "supertrend_exit", current_ts
                )
                trades.append(_pos_to_dict(pos))
                return
        else:
            # Ratchet SL down to Supertrend line (never raise it)
            pos.stop_loss = min(pos.stop_loss, st_line)
            if st_dir > 0:
                rm.close_position(
                    pos, current_price * (1 + slip), "supertrend_exit", current_ts
                )
                trades.append(_pos_to_dict(pos))
                return

    # 4. Time stop (bars of the configured timeframe)
    candles_held = idx - pos.entry_idx
    if candles_held >= cfg.max_hold_bars:
        fill = current_price * (1 - slip) if pos.direction == "long" else current_price * (1 + slip)
        rm.close_position(pos, fill, "time_stop", current_ts)
        trades.append(_pos_to_dict(pos))
        return


def run_backtest(
    df: pd.DataFrame,
    cfg: SupertrendVolumeConfig,
    verbose: bool = False,
) -> dict[str, Any]:
    """Full backtest of Supertrend + Volume strategy.

    Args:
        df: OHLCV DataFrame (open, high, low, close, volume),
            sorted ascending by timestamp.
        cfg: Strategy configuration.
        verbose: Print progress every 500 candles.

    Returns:
        dict with keys: trades, metrics, equity_curve, signals_df
    """
    # Compute indicators + signals
    df = generate_signals(df, cfg)

    # Initialize risk manager (fixed account sizing, no compounding)
    rm = RiskManager(cfg, account_size=cfg.account_size, backtest_mode=True)  # type: ignore[arg-type]

    # State
    open_pos: Position | None = None
    trades: list[dict] = []
    equity_curve: list[float] = [rm.current_equity]

    # Drop warmup rows where indicators are NaN
    df = df.dropna(
        subset=["atr", "ema_slow_val", "supertrend", "st_dir"]
    ).copy()

    if len(df) == 0:
        return {
            "trades": [],
            "metrics": _compute_metrics([], equity_curve, cfg),
            "equity_curve": equity_curve,
            "signals_df": df,
        }

    pending: str | None = None       # direction armed at the previous close
    rng = np.random.default_rng(cfg.random_seed)

    for i in range(len(df)):
        row = df.iloc[i]
        current_ts = (
            float(row.name.timestamp())
            if hasattr(row.name, "timestamp")
            else float(i)
        )
        open_price = float(row["open"])
        current_price = float(row["close"])
        current_high = float(row["high"])
        current_low = float(row["low"])

        # ── Fill a pending entry at THIS bar's open (signal was the prev close) ──
        if open_pos is None and pending is not None:
            direction = pending
            pending = None
            allowed, _ = rm.can_trade(current_ts)
            atr_val = float(row["atr"])
            if allowed and atr_val > 0:
                slip = cfg.slippage_pct
                fill = open_price * (1 + slip) if direction == "long" else open_price * (1 - slip)
                pos = rm.open_position(
                    entry_price=fill, atr=atr_val, direction=direction,
                    entry_time=current_ts, entry_idx=i,
                )
                if pos:
                    pos.meta = {"signal_idx": i - 1}
                    open_pos = pos
                    if verbose:
                        print(f"  [trade] OPEN {direction} @ {fill:.2f} "
                              f"SL={pos.stop_loss:.2f} TP1={pos.tp1:.2f}")

        if verbose and i % 500 == 0 and i > 0:
            pct = i / len(df) * 100
            print(
                f"  [{pct:5.1f}%] candle {i}/{len(df)} | "
                f"trades={len(trades)} | equity={rm.current_equity:.2f}"
            )

        # ── Manage open position ─────────────────────────────────
        if open_pos is not None:
            manage_position(
                open_pos, row, i, current_ts, current_price,
                current_high, current_low, cfg, rm, trades,
            )
            if open_pos.status == "closed":
                open_pos = None

        # ── Arm a new signal at this bar's close (filled next bar's open) ──
        if open_pos is None and pending is None:
            has_signal = row["long_signal"] > 0 or row["short_signal"] > 0
            if has_signal and cfg.random_entry:
                pending = "long" if rng.random() < 0.5 else "short"
            elif row["long_signal"] > 0:
                pending = "long"
            elif row["short_signal"] > 0 and not cfg.long_only:
                pending = "short"

        equity_curve.append(rm.current_equity)

    # Close any remaining position at backtest end
    if open_pos is not None and open_pos.status != "closed":
        last_row = df.iloc[-1]
        last_price = float(last_row["close"])
        last_ts = (
            float(last_row.name.timestamp())
            if hasattr(last_row.name, "timestamp")
            else float(len(df))
        )
        rm.close_position(open_pos, last_price, "backtest_end", last_ts)
        trades.append(_pos_to_dict(open_pos))

    # Compute metrics (reuse existing metric calculator)
    metrics = _compute_metrics(trades, equity_curve, cfg)

    return {
        "trades": trades,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "signals_df": df,
    }


# ── CLI entry point for standalone testing ──────────────────────────
if __name__ == "__main__":
    import asyncio
    from ..adapters.yfinance_adapter import YFinanceAdapter

    async def main():
        adapter = YFinanceAdapter()
        df = await adapter.fetch_history("GC=F", interval="1h", period="730d")
        print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")

        cfg = SupertrendVolumeConfig()
        result = run_backtest(df, cfg, verbose=True)
        print_metrics(result["metrics"])

    asyncio.run(main())
