"""Vectorized backtest engine for MCTFB strategy.

Runs the full strategy on historical OHLCV data and produces:
  - Trade log (list of closed trades)
  - Performance metrics (win rate, Sharpe, max DD, expectancy, profit factor)

Path-dependent simulation: iterates candle-by-candle to handle
partial closes, trailing stops, and cooldown logic. This cannot be
fully vectorized due to state dependencies (position state, risk state).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from .score import trades_per_week

from .config import MCTFBConfig
from .indicators import compute_all_indicators
from .signals import generate_entry_signals
from .risk import RiskManager, Position


def backtest(
    df: pd.DataFrame, cfg: MCTFBConfig, verbose: bool = False
) -> dict[str, Any]:
    """Run a full backtest of the MCTFB strategy.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns: open, high, low, close, volume.
        Must be sorted ascending by timestamp.
    cfg : MCTFBConfig
        Strategy configuration.
    verbose : bool
        Print progress every 1000 candles.

    Returns
    -------
    dict with keys: trades, metrics, equity_curve, signals_df
    """
    # 1. Compute indicators + entry signals
    df = compute_all_indicators(df, cfg)
    df = generate_entry_signals(df, cfg)

    # 2. Initialize risk manager
    rm = RiskManager(cfg, account_size=10000.0)

    # 3. State
    open_pos: Position | None = None
    trades: list[dict] = []
    equity_curve: list[float] = [rm.current_equity]
    timestamps: list = [df.index[0]] if len(df) > 0 else []

    # 4. Candle-by-candle simulation
    for i in range(len(df)):
        row = df.iloc[i]
        current_ts = float(row.name.timestamp()) if hasattr(row.name, "timestamp") else float(i)
        current_price = float(row["close"])
        current_high = float(row["high"])
        current_low = float(row["low"])

        if verbose and i % 1000 == 0 and i > 0:
            done, total = i, len(df)
            pct = done / total * 100
            print(f"  [{pct:5.1f}%] candle {done}/{total} | "
                  f"trades={len(trades)} | equity={rm.current_equity:.2f}")

        # ── Manage open position ───────────────────────────────
        if open_pos is not None:
            _manage_position(
                open_pos, rm, row, i, current_ts, current_price,
                current_high, current_low, cfg, trades
            )
            if open_pos.status == "closed":
                open_pos = None

        # ── Check for new entry ────────────────────────────────
        if open_pos is None:
            allowed, reason = rm.can_trade(current_ts)
            if not allowed:
                pass  # in cooldown or daily loss limit
            else:
                # Execute at next candle OPEN to avoid look-ahead
                # But for simplicity in the backtest, we execute at this candle's
                # close (which is the same as next candle's open in practice for 1h data)
                atr_val = float(row["atr"]) if not np.isnan(row["atr"]) else 0.0
                if atr_val > 0:
                    if row["long_signal"] > 0:
                        pos = rm.open_position(
                            entry_price=current_price,
                            atr=atr_val,
                            direction="long",
                            entry_time=current_ts,
                            entry_idx=i,
                        )
                        if pos:
                            open_pos = pos
                            if verbose:
                                print(f"  [trade] OPEN long @ {current_price:.2f} "
                                      f"SL={pos.stop_loss:.2f} TP1={pos.tp1:.2f}")
                    elif row["short_signal"] > 0:
                        pos = rm.open_position(
                            entry_price=current_price,
                            atr=atr_val,
                            direction="short",
                            entry_time=current_ts,
                            entry_idx=i,
                        )
                        if pos:
                            open_pos = pos
                            if verbose:
                                print(f"  [trade] OPEN short @ {current_price:.2f} "
                                      f"SL={pos.stop_loss:.2f} TP1={pos.tp1:.2f}")

        # Track equity
        equity_curve.append(rm.current_equity)
        timestamps.append(row.name)

    # Close any remaining position at last close
    if open_pos is not None:
        last_row = df.iloc[-1]
        last_price = float(last_row["close"])
        last_ts = float(last_row.name.timestamp()) if hasattr(last_row.name, "timestamp") else float(len(df))
        rm.close_position(open_pos, last_price, "backtest_end", last_ts)
        trades.append(_pos_to_dict(open_pos))

    # 5. Compute metrics
    metrics = _compute_metrics(trades, equity_curve, cfg)

    return {
        "trades": trades,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "equity_timestamps": timestamps,
        "signals_df": df,
    }


def _manage_position(
    pos: Position, rm: RiskManager, row: pd.Series, idx: int,
    current_ts: float, current_price: float,
    current_high: float, current_low: float,
    cfg: MCTFBConfig, trades: list[dict]
) -> None:
    """Check SL, TP1, TP2 (supertrend), and time stop for an open position."""

    # Check stop loss first (use high/low for intrabar testing)
    if pos.direction == "long":
        if current_low <= pos.stop_loss:
            rm.close_position(pos, pos.stop_loss, "stop_loss", current_ts)
            trades.append(_pos_to_dict(pos))
            return
    else:
        if current_high >= pos.stop_loss:
            rm.close_position(pos, pos.stop_loss, "stop_loss", current_ts)
            trades.append(_pos_to_dict(pos))
            return

    # Check TP1 (if not yet hit)
    if not pos.tp1_hit:
        if pos.direction == "long" and current_high >= pos.tp1:
            rm.close_position(pos, pos.tp1, "tp1_partial", current_ts, partial=True)
            return
        elif pos.direction == "short" and current_low <= pos.tp1:
            rm.close_position(pos, pos.tp1, "tp1_partial", current_ts, partial=True)
            return

    # Check TP2 (supertrend flip) — only after TP1 hit
    if pos.tp1_hit:
        st_dir = float(row["st_dir"])
        if pos.direction == "long" and st_dir < 0:
            rm.close_position(pos, current_price, "supertrend_exit", current_ts)
            trades.append(_pos_to_dict(pos))
            return
        elif pos.direction == "short" and st_dir > 0:
            rm.close_position(pos, current_price, "supertrend_exit", current_ts)
            trades.append(_pos_to_dict(pos))
            return

    # Check time stop
    candles_held = idx - pos.entry_idx
    if candles_held >= cfg.max_hold_hours:
        rm.close_position(pos, current_price, "time_stop", current_ts)
        trades.append(_pos_to_dict(pos))
        return


def _pos_to_dict(pos: Position) -> dict:
    """Convert a Position to a serializable dict."""
    d = asdict(pos)
    return d


def _compute_side_metrics(trades: list[dict]) -> dict[str, dict[str, float | int]]:
    """Per-direction stats from closed trades."""
    closed = [t for t in trades if t.get("status") == "closed"]
    out: dict[str, dict[str, float | int]] = {}
    for side in ("long", "short"):
        side_trades = [t for t in closed if t.get("direction") == side]
        pnls = [t.get("pnl_pct", 0.0) or 0.0 for t in side_trades]
        wins = [p for p in pnls if p > 0]
        out[side] = {
            "trades": len(side_trades),
            "pnl_pct": round(sum(pnls), 2),
            "win_rate": round(len(wins) / len(pnls) * 100, 2) if pnls else 0.0,
        }
    return out


def _max_drawdown_from_equity(equity_curve: list[float] | None) -> float | None:
    """Peak-to-trough drawdown in % from an equity curve.

    Returns None when the curve has fewer than two usable points, so the caller
    can fall back to the trade-PnL stream.
    """
    if not equity_curve or len(equity_curve) < 2:
        return None
    peak = float(equity_curve[0])
    max_dd = 0.0
    for value in equity_curve:
        v = float(value)
        peak = max(peak, v)
        if peak > 0:
            max_dd = min(max_dd, (v - peak) / peak * 100.0)
    return max_dd


def _max_drawdown_from_pnls(pnls: list[float]) -> float:
    """Fallback drawdown in % from the cumulative trade-PnL stream."""
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)
    return max_dd


def _compute_metrics(
    trades: list[dict],
    equity_curve: list[float] | None = None,
    cfg: MCTFBConfig | None = None,
) -> dict[str, Any]:
    """Compute performance metrics from the trade list and equity curve.

    ``equity_curve`` drives max drawdown when it has at least two points
    (intratrade dips included); otherwise the drawdown falls back to the
    cumulative trade-PnL stream. ``cfg`` is accepted for backward compatibility
    and is not used.
    """
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "by_side": _compute_side_metrics([]),
        }

    # Only fully closed trades
    closed = [t for t in trades if t.get("status") == "closed"]
    pnls = [t.get("pnl_pct", 0.0) or 0.0 for t in closed]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0.0

    # Max drawdown: prefer the equity curve (real intratrade path), else the
    # trade-PnL stream.
    equity_dd = _max_drawdown_from_equity(equity_curve)
    max_dd = equity_dd if equity_dd is not None else _max_drawdown_from_pnls(pnls)

    # Sharpe (per-trade, annualized with sqrt of the realized trade frequency)
    if len(pnls) > 1:
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls, ddof=1)
        if std_pnl > 0:
            trades_per_year = max(trades_per_week(closed), 1.0) * 52.0
            sharpe = (mean_pnl / std_pnl) * np.sqrt(trades_per_year)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Expectancy in R (risk multiples)
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    if avg_loss != 0:
        expectancy_r = (win_rate / 100 * avg_win + (1 - win_rate / 100) * avg_loss) / abs(avg_loss)
    else:
        expectancy_r = 0.0

    # Profit factor
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "total_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 2),
        "total_pnl_pct": round(total_pnl, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "expectancy_r": round(expectancy_r, 3),
        "profit_factor": round(profit_factor, 3),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "trades_per_week": round(trades_per_week(closed), 2),
        "by_side": _compute_side_metrics(trades),
    }


def print_metrics(metrics: dict[str, Any], label: str = "BACKTEST") -> None:
    """Print metrics in a readable format."""
    print("\n" + "=" * 60)
    print(f"{label} RESULTS")
    print("=" * 60)
    print(f"  Total trades:     {metrics['total_trades']}")
    if metrics.get("trades_per_week"):
        print(f"  Trades/week:      {metrics['trades_per_week']}")
    print(f"  Wins/Losses:      {metrics.get('wins', 0)}W / {metrics.get('losses', 0)}L")
    print(f"  Win rate:         {metrics['win_rate']}%")
    print(f"  Total PnL:        {metrics['total_pnl_pct']}%")
    print(f"  Max Drawdown:     {metrics['max_drawdown_pct']}%")
    print(f"  Sharpe:           {metrics['sharpe']}")
    print(f"  Expectancy (R):   {metrics['expectancy_r']}")
    print(f"  Profit Factor:    {metrics['profit_factor']}")
    print(f"  Avg Win:          {metrics.get('avg_win_pct', 0)}%")
    print(f"  Avg Loss:         {metrics.get('avg_loss_pct', 0)}%")
    by_side = metrics.get("by_side")
    if by_side:
        for side in ("long", "short"):
            s = by_side[side]
            if s["trades"] > 0:
                print(
                    f"  {side.capitalize():5s} side:      "
                    f"{s['trades']} trades | PnL {s['pnl_pct']:+.2f}% | WR {s['win_rate']:.1f}%"
                )
    print("=" * 60)
