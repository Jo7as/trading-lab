"""ADX Regime-Hybrid: Supertrend in trends, BB mean-reversion in ranges."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from ..backtest import _compute_metrics, _pos_to_dict
from ..indicators import adx as compute_adx, bollinger_bands, rsi
from ..risk import Position, RiskManager
from .supertrend_volume import (
    SupertrendVolumeConfig,
    generate_signals as st_generate_signals,
    manage_position as st_manage_position,
)

Regime = Literal["trend", "range", "idle"]


@dataclass(frozen=True)
class RegimeHybridConfig:
    """ADX regime switch between ST trend and BB mean-reversion."""

    asset: str = "EUR/USD"
    asset_label: str = "EUR/USD"

    adx_period: int = 14
    adx_trend_on: float = 25.0
    adx_range_on: float = 15.0

    supertrend_period: int = 7
    supertrend_mult: float = 1.5
    volume_ma_period: int = 20
    volume_mult: float = 1.0
    ema_fast: int = 3
    ema_slow: int = 3
    atr_period: int = 14
    min_atr_pct: float = 0.0
    pullback_entry: bool = False
    pullback_tolerance: float = 0.002

    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 35.0
    rsi_overbought: float = 65.0
    min_bb_bandwidth: float = 0.01
    mr_sl_atr_mult: float = 1.5
    mr_max_hold_hours: int = 8

    account_size: float = 10000.0
    risk_per_trade: float = 0.01
    sl_atr_mult: float = 1.5
    tp1_r: float = 2.0
    tp1_close_pct: float = 0.60
    max_hold_hours: int = 48
    max_daily_loss: float = 0.03
    cooldown_hours: int = 0
    max_concurrent_positions: int = 1
    long_only: bool = False

    def to_st_config(self) -> SupertrendVolumeConfig:
        return SupertrendVolumeConfig(
            asset=self.asset,
            asset_label=self.asset_label,
            supertrend_period=self.supertrend_period,
            supertrend_mult=self.supertrend_mult,
            volume_ma_period=self.volume_ma_period,
            volume_mult=self.volume_mult,
            ema_fast=self.ema_fast,
            ema_slow=self.ema_slow,
            atr_period=self.atr_period,
            min_atr_pct=self.min_atr_pct,
            pullback_entry=self.pullback_entry,
            pullback_tolerance=self.pullback_tolerance,
            account_size=self.account_size,
            risk_per_trade=self.risk_per_trade,
            sl_atr_mult=self.sl_atr_mult,
            tp1_r=self.tp1_r,
            tp1_close_pct=self.tp1_close_pct,
            max_hold_hours=self.max_hold_hours,
            max_daily_loss=self.max_daily_loss,
            cooldown_hours=self.cooldown_hours,
            max_concurrent_positions=self.max_concurrent_positions,
            long_only=self.long_only,
        )


def compute_regime_series(
    adx_vals: pd.Series, cfg: RegimeHybridConfig
) -> pd.Series:
    """Hysteresis regime labels per bar."""
    regimes: list[Regime] = []
    last: Regime = "idle"
    for val in adx_vals:
        if pd.isna(val):
            regimes.append(last)
            continue
        if val >= cfg.adx_trend_on:
            last = "trend"
        elif val <= cfg.adx_range_on:
            last = "range"
        regimes.append(last)
    return pd.Series(regimes, index=adx_vals.index, dtype=object)


def prepare_dataframe(df: pd.DataFrame, cfg: RegimeHybridConfig) -> pd.DataFrame:
    """ST signals + MR signals + regime column."""
    st_cfg = cfg.to_st_config()
    out = st_generate_signals(df, st_cfg)
    adx_df = compute_adx(out["high"], out["low"], out["close"], cfg.adx_period)
    out = pd.concat([out, adx_df], axis=1)
    out["regime"] = compute_regime_series(out["adx"], cfg)

    bb = bollinger_bands(out["close"], cfg.bb_period, cfg.bb_std)
    out = pd.concat([out, bb], axis=1)
    out["rsi"] = rsi(out["close"], cfg.rsi_period)
    prev_close = out["close"].shift(1)

    bw_ok = out["bb_bandwidth"] > cfg.min_bb_bandwidth
    mr_long = (
        (out["close"] < out["bb_lower"])
        & (out["close"] > prev_close)
        & (out["rsi"] < cfg.rsi_oversold)
        & bw_ok
    )
    mr_short = (
        (out["close"] > out["bb_upper"])
        & (out["close"] < prev_close)
        & (out["rsi"] > cfg.rsi_overbought)
        & bw_ok
    )
    out["mr_long"] = mr_long.astype(float)
    out["mr_short"] = mr_short.astype(float)
    return out


def _open_mr_position(
    rm: RiskManager,
    *,
    entry_price: float,
    atr_val: float,
    bb_middle: float,
    direction: str,
    entry_time: float,
    entry_idx: int,
    cfg: RegimeHybridConfig,
) -> Position | None:
    sl_dist = cfg.mr_sl_atr_mult * atr_val
    if direction == "long":
        sl = entry_price - sl_dist
        tp1 = float(bb_middle)
        if tp1 <= entry_price:
            tp1 = entry_price + sl_dist
    else:
        sl = entry_price + sl_dist
        tp1 = float(bb_middle)
        if tp1 >= entry_price:
            tp1 = entry_price - sl_dist

    size = rm.calculate_position_size(entry_price, sl)
    if size <= 0:
        return None

    return Position(
        entry_time=entry_time,
        entry_idx=entry_idx,
        entry_price=entry_price,
        direction=direction,
        size=size,
        stop_loss=sl,
        tp1=tp1,
        atr_at_entry=atr_val,
    )


def manage_mr_position(
    pos: Position,
    row: pd.Series,
    idx: int,
    current_ts: float,
    current_price: float,
    current_high: float,
    current_low: float,
    cfg: RegimeHybridConfig,
    rm: RiskManager,
    trades: list[dict],
) -> None:
    """MR exits: SL, TP at bb_middle, time stop."""
    if pos.direction == "long":
        if current_low <= pos.stop_loss:
            rm.close_position(pos, pos.stop_loss, "mr_stop_loss", current_ts)
            trades.append(_pos_to_dict(pos))
            return
        if current_high >= pos.tp1:
            rm.close_position(pos, pos.tp1, "mr_tp_middle", current_ts)
            trades.append(_pos_to_dict(pos))
            return
    else:
        if current_high >= pos.stop_loss:
            rm.close_position(pos, pos.stop_loss, "mr_stop_loss", current_ts)
            trades.append(_pos_to_dict(pos))
            return
        if current_low <= pos.tp1:
            rm.close_position(pos, pos.tp1, "mr_tp_middle", current_ts)
            trades.append(_pos_to_dict(pos))
            return

    if idx - pos.entry_idx >= cfg.mr_max_hold_hours:
        rm.close_position(pos, current_price, "mr_time_stop", current_ts)
        trades.append(_pos_to_dict(pos))


def run_backtest(
    df: pd.DataFrame,
    cfg: RegimeHybridConfig,
    verbose: bool = False,
) -> dict[str, Any]:
    """Backtest ADX regime-hybrid strategy."""
    st_cfg = cfg.to_st_config()
    data = prepare_dataframe(df, cfg)
    data = data.dropna(
        subset=["atr", "ema_slow_val", "supertrend", "st_dir", "adx", "bb_middle"]
    ).copy()

    if len(data) == 0:
        return {
            "trades": [],
            "metrics": _compute_metrics([], [cfg.account_size], st_cfg),
            "equity_curve": [cfg.account_size],
            "regime_stats": {},
            "module_stats": {},
        }

    rm = RiskManager(cfg, account_size=cfg.account_size, backtest_mode=True)
    open_pos: Position | None = None
    entry_module: Literal["trend", "mr"] | None = None
    trades: list[dict] = []
    equity_curve: list[float] = [rm.current_equity]

    regime_counts = {"trend": 0, "range": 0, "idle": 0}
    module_trades = {"trend": 0, "mr": 0}

    for i in range(len(data)):
        row = data.iloc[i]
        regime = str(row["regime"])
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        current_ts = float(row.name.timestamp()) if hasattr(row.name, "timestamp") else float(i)
        current_price = float(row["close"])
        current_high = float(row["high"])
        current_low = float(row["low"])

        if open_pos is not None:
            if entry_module == "trend":
                prev_len = len(trades)
                st_manage_position(
                    open_pos, row, i, current_ts, current_price,
                    current_high, current_low, st_cfg, rm, trades,
                )
                if open_pos.status == "closed":
                    if len(trades) > prev_len:
                        trades[-1]["entry_module"] = entry_module
                    module_trades["trend"] += 1
                    open_pos = None
                    entry_module = None
            else:
                prev_len = len(trades)
                manage_mr_position(
                    open_pos, row, i, current_ts, current_price,
                    current_high, current_low, cfg, rm, trades,
                )
                if open_pos.status == "closed":
                    if len(trades) > prev_len:
                        trades[-1]["entry_module"] = entry_module
                    module_trades["mr"] += 1
                    open_pos = None
                    entry_module = None

        if open_pos is None:
            allowed, _ = rm.can_trade(current_ts)
            if allowed:
                atr_val = float(row.get("atr", 0))
                if atr_val > 0 and regime == "trend":
                    if row.get("long_signal", 0) > 0:
                        pos = rm.open_position(
                            entry_price=current_price, atr=atr_val,
                            direction="long", entry_time=current_ts, entry_idx=i,
                        )
                        if pos:
                            open_pos = pos
                            entry_module = "trend"
                    elif row.get("short_signal", 0) > 0 and not cfg.long_only:
                        pos = rm.open_position(
                            entry_price=current_price, atr=atr_val,
                            direction="short", entry_time=current_ts, entry_idx=i,
                        )
                        if pos:
                            open_pos = pos
                            entry_module = "trend"
                elif atr_val > 0 and regime == "range":
                    if row.get("mr_long", 0) > 0:
                        pos = _open_mr_position(
                            rm,
                            entry_price=current_price,
                            atr_val=atr_val,
                            bb_middle=float(row["bb_middle"]),
                            direction="long",
                            entry_time=current_ts,
                            entry_idx=i,
                            cfg=cfg,
                        )
                        if pos:
                            open_pos = pos
                            entry_module = "mr"
                    elif row.get("mr_short", 0) > 0 and not cfg.long_only:
                        pos = _open_mr_position(
                            rm,
                            entry_price=current_price,
                            atr_val=atr_val,
                            bb_middle=float(row["bb_middle"]),
                            direction="short",
                            entry_time=current_ts,
                            entry_idx=i,
                            cfg=cfg,
                        )
                        if pos:
                            open_pos = pos
                            entry_module = "mr"

        equity_curve.append(rm.current_equity)

    if open_pos is not None and open_pos.status != "closed":
        last_row = data.iloc[-1]
        last_price = float(last_row["close"])
        last_ts = float(last_row.name.timestamp()) if hasattr(last_row.name, "timestamp") else float(len(data))
        rm.close_position(open_pos, last_price, "backtest_end", last_ts)
        t = _pos_to_dict(open_pos)
        t["entry_module"] = entry_module
        trades.append(t)
        if entry_module:
            module_trades[entry_module] += 1

    total_bars = sum(regime_counts.values()) or 1
    regime_stats = {
        k: round(v / total_bars * 100, 1) for k, v in regime_counts.items()
    }
    metrics = _compute_metrics(trades, equity_curve, st_cfg)
    metrics["regime_stats"] = regime_stats
    metrics["module_trades"] = module_trades

    return {
        "trades": trades,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "regime_stats": regime_stats,
        "module_stats": module_trades,
        "signals_df": data,
    }
