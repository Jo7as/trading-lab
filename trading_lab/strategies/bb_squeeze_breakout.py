"""BB Squeeze Breakout strategy for Forex H1.

Volatility compression (Bollinger bandwidth in bottom percentile) followed by
a close breakout beyond the upper/lower band. Distinct signal type from
Supertrend flip strategies — targets expansion after squeeze.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..backtest import _compute_metrics, _pos_to_dict
from ..indicators import (
    atr,
    bb_squeeze,
    bollinger_bands,
    candle_body_ratio,
    ema,
)
from ..risk import Position, RiskManager
from .supertrend_volume import _compute_volume_filter


@dataclass(frozen=True)
class BBSqueezeBreakoutConfig:
    """Configuration for BB Squeeze Breakout on 1h candles."""

    asset: str = "EUR/USD"
    asset_label: str = "EUR/USD"

    bb_period: int = 20
    bb_std: float = 2.0
    squeeze_lookback: int = 100
    squeeze_threshold: float = 0.20

    ema_trend_period: int = 50
    use_trend_filter: bool = True

    activity_period: int = 20
    activity_mult: float = 1.0

    min_body_ratio: float = 0.40
    min_bb_bandwidth: float = 0.005

    session_filter: bool = False
    session_start_utc: int = 7
    session_end_utc: int = 17

    atr_period: int = 14
    sl_atr_mult: float = 1.5
    tp1_r: float = 2.0
    tp1_close_pct: float = 0.60
    max_hold_hours: int = 24
    fail_back_inside_bands: bool = True

    account_size: float = 10000.0
    risk_per_trade: float = 0.01
    max_daily_loss: float = 0.03
    cooldown_hours: int = 0
    max_concurrent_positions: int = 1
    long_only: bool = False

    @staticmethod
    def search_space() -> list[tuple[str, float, float, float]]:
        return [
            ("squeeze_threshold", 0.10, 0.35, 0.05),
            ("bb_std", 1.5, 2.5, 0.2),
            ("ema_trend_period", 20, 100, 10),
            ("activity_mult", 0.6, 1.5, 0.1),
            ("min_body_ratio", 0.20, 0.60, 0.10),
            ("sl_atr_mult", 1.0, 2.5, 0.2),
            ("tp1_r", 1.5, 3.0, 0.2),
            ("max_hold_hours", 12, 48, 6),
        ]


def _session_ok(index: pd.Index, cfg: BBSqueezeBreakoutConfig) -> pd.Series:
    if not cfg.session_filter:
        return pd.Series(1.0, index=index)
    hours = pd.Series(index.hour, index=index)
    in_session = (hours >= cfg.session_start_utc) & (hours < cfg.session_end_utc)
    return in_session.astype(float)


def compute_indicators(df: pd.DataFrame, cfg: BBSqueezeBreakoutConfig) -> pd.DataFrame:
    """Append BB, squeeze, ATR, trend EMA, activity, and session columns."""
    out = df.copy()
    if "volume" not in out.columns:
        out["volume"] = 0.0

    bb = bollinger_bands(out["close"], cfg.bb_period, cfg.bb_std)
    out = pd.concat([out, bb], axis=1)

    out["squeeze"] = bb_squeeze(
        out["close"],
        cfg.bb_period,
        cfg.bb_std,
        cfg.squeeze_lookback,
        cfg.squeeze_threshold,
    )
    out["atr"] = atr(out["high"], out["low"], out["close"], cfg.atr_period)
    out["atr_pct"] = out["atr"] / out["close"]
    out["ema_trend"] = ema(out["close"], cfg.ema_trend_period)
    out["activity_ok"] = _compute_volume_filter(
        out, cfg.activity_period, cfg.activity_mult
    )
    out["body_ratio"] = candle_body_ratio(
        out["open"], out["high"], out["low"], out["close"]
    )
    out["session_ok"] = _session_ok(out.index, cfg)
    return out


def generate_signals(df: pd.DataFrame, cfg: BBSqueezeBreakoutConfig) -> pd.DataFrame:
    """Generate long/short squeeze-breakout entry signals.

    LONG (all at candle close):
      1. squeeze.shift(1) > 0 — squeeze on prior bar
      2. close > bb_upper
      3. close > ema_trend (if use_trend_filter)
      4. activity_ok, body_ratio, bandwidth, session gates

    SHORT: mirrored.
    """
    out = compute_indicators(df, cfg)

    squeeze_prev = out["squeeze"].shift(1) > 0
    bw_ok = out["bb_bandwidth"] >= cfg.min_bb_bandwidth
    body_ok = out["body_ratio"] >= cfg.min_body_ratio
    activity = out["activity_ok"] > 0
    session = out["session_ok"] > 0

    if cfg.use_trend_filter:
        trend_long = out["close"] > out["ema_trend"]
        trend_short = out["close"] < out["ema_trend"]
    else:
        trend_long = pd.Series(True, index=out.index)
        trend_short = pd.Series(True, index=out.index)

    long_breakout = out["close"] > out["bb_upper"]
    short_breakout = out["close"] < out["bb_lower"]

    long_cond = (
        squeeze_prev & long_breakout & trend_long & activity & body_ok & bw_ok & session
    )
    short_cond = (
        squeeze_prev & short_breakout & trend_short & activity & body_ok & bw_ok & session
    )

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
    cfg: BBSqueezeBreakoutConfig,
    rm: RiskManager,
    trades: list[dict],
) -> None:
    """Exit hierarchy: SL, TP1 partial, failed breakout, time stop."""
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

    if not pos.tp1_hit:
        if pos.direction == "long" and current_high >= pos.tp1:
            rm.close_position(pos, pos.tp1, "tp1_partial", current_ts, partial=True)
            return
        if pos.direction == "short" and current_low <= pos.tp1:
            rm.close_position(pos, pos.tp1, "tp1_partial", current_ts, partial=True)
            return

    if cfg.fail_back_inside_bands:
        if pos.direction == "long" and current_price < float(row["bb_upper"]):
            rm.close_position(pos, current_price, "failed_breakout", current_ts)
            trades.append(_pos_to_dict(pos))
            return
        if pos.direction == "short" and current_price > float(row["bb_lower"]):
            rm.close_position(pos, current_price, "failed_breakout", current_ts)
            trades.append(_pos_to_dict(pos))
            return

    if idx - pos.entry_idx >= cfg.max_hold_hours:
        rm.close_position(pos, current_price, "time_stop", current_ts)
        trades.append(_pos_to_dict(pos))


def run_backtest(
    df: pd.DataFrame,
    cfg: BBSqueezeBreakoutConfig,
    verbose: bool = False,
) -> dict[str, Any]:
    """Full candle-by-candle backtest."""
    data = generate_signals(df, cfg)
    data = data.dropna(subset=["bb_middle", "bb_upper", "bb_lower", "atr", "ema_trend"]).copy()

    if len(data) == 0:
        empty_metrics = _compute_metrics([], [cfg.account_size], cfg)  # type: ignore[arg-type]
        empty_metrics["squeeze_pct"] = 0.0
        empty_metrics["breakout_signals"] = {"long": 0, "short": 0}
        return {
            "trades": [],
            "metrics": empty_metrics,
            "equity_curve": [cfg.account_size],
            "signals_df": data,
        }

    rm = RiskManager(cfg, account_size=cfg.account_size, backtest_mode=True)  # type: ignore[arg-type]
    open_pos: Position | None = None
    trades: list[dict] = []
    equity_curve: list[float] = [rm.current_equity]

    squeeze_bars = int((data["squeeze"] > 0).sum())
    long_signals = int((data["long_signal"] > 0).sum())
    short_signals = int((data["short_signal"] > 0).sum())

    for i in range(len(data)):
        row = data.iloc[i]
        current_ts = float(row.name.timestamp()) if hasattr(row.name, "timestamp") else float(i)
        current_price = float(row["close"])
        current_high = float(row["high"])
        current_low = float(row["low"])

        if open_pos is not None:
            manage_position(
                open_pos, row, i, current_ts, current_price,
                current_high, current_low, cfg, rm, trades,
            )
            if open_pos.status == "closed":
                open_pos = None

        if open_pos is None:
            allowed, _ = rm.can_trade(current_ts)
            if allowed:
                atr_val = float(row["atr"])
                if atr_val > 0:
                    if row["long_signal"] > 0:
                        pos = rm.open_position(
                            entry_price=current_price, atr=atr_val,
                            direction="long", entry_time=current_ts, entry_idx=i,
                        )
                        if pos:
                            open_pos = pos
                    elif row["short_signal"] > 0 and not cfg.long_only:
                        pos = rm.open_position(
                            entry_price=current_price, atr=atr_val,
                            direction="short", entry_time=current_ts, entry_idx=i,
                        )
                        if pos:
                            open_pos = pos

        equity_curve.append(rm.current_equity)

    if open_pos is not None and open_pos.status != "closed":
        last_row = data.iloc[-1]
        last_price = float(last_row["close"])
        last_ts = float(last_row.name.timestamp()) if hasattr(last_row.name, "timestamp") else float(len(data))
        rm.close_position(open_pos, last_price, "backtest_end", last_ts)
        trades.append(_pos_to_dict(open_pos))

    metrics = _compute_metrics(trades, equity_curve, cfg)  # type: ignore[arg-type]
    total_bars = len(data) or 1
    metrics["squeeze_pct"] = round(squeeze_bars / total_bars * 100, 1)
    metrics["breakout_signals"] = {"long": long_signals, "short": short_signals}

    return {
        "trades": trades,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "signals_df": data,
    }
