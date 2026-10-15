"""ICT Silver Bullet — time-boxed SMC scalp (not nested 4h→15m→1m).

Checklist (NY local windows 03–04 / 10–11 / 14–15):
  1. Mark nearest BSL/SSL on 15m at window open
  2. Only *form* setups inside the one-hour window
  3. Sweep one side → draw = opposing pool
  4. 1m MSS (ChoCh toward draw) + displacement FVG
  5. Limit into FVG; SL beyond FVG candle extreme; TP = fixed ``tp_r`` (default 3R)

Fill/management may extend past the window clock (ICT: do not exit early).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..backtest import _compute_metrics, _pos_to_dict
from ..indicators import atr as atr_indicator
from ..risk import Position, RiskManager
from .smc.liquidity import LiquidityEngine
from .smc.structure import MarketStructureEngine
from .smc.supply_demand import detect_fvg_at
from .smc.types import TrendBias

_NY = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")

_SB_WINDOWS: tuple[tuple[str, time, time], ...] = (
    ("london", time(3, 0), time(4, 0)),
    ("ny_am", time(10, 0), time(11, 0)),
    ("ny_pm", time(14, 0), time(15, 0)),
)


@dataclass(frozen=True)
class SilverBulletConfig:
    """Config for ICT Silver Bullet."""

    asset: str = "XAU/USD"
    asset_label: str = "XAU/USD"

    use_london: bool = True
    use_ny_am: bool = True
    use_ny_pm: bool = True

    swing_length_15m: int = 5
    swing_length_1m: int = 3
    use_asia: bool = True
    use_london_liq: bool = True
    use_pd: bool = True
    use_pw: bool = True
    use_swing_liq: bool = True
    swing_liq_length: int = 5

    entry_edge_offset: float = 0.0
    entry_timeout_bars: int = 60
    sl_buffer: float = 0.50
    require_fvg: bool = True
    require_pd_zone: bool = True
    # ChoCh bar body must be >= mult * ATR (displacement)
    require_displacement: bool = True
    displacement_atr_mult: float = 1.0

    tp_r: float = 3.0
    tp1_close_pct: float = 1.0
    move_sl_to_be_on_tp1: bool = False
    max_hold_bars: int = 180
    atr_period: int = 14
    sl_atr_mult: float = 1.0

    account_size: float = 10000.0
    risk_per_trade: float = 0.01
    max_daily_loss: float = 0.03
    cooldown_hours: int = 0
    max_concurrent_positions: int = 1
    long_only: bool = False

    @property
    def tp1_r(self) -> float:
        return self.tp_r


@dataclass
class _WindowCtx:
    key: str
    name: str
    bsl: float | None = None
    ssl: float | None = None
    mid: float | None = None
    swept_side: str | None = None
    sweep_extreme: float | None = None
    sweep_i: int = -1
    want: int = 0
    draw: float | None = None
    pending: dict[str, Any] | None = None
    traded: bool = False


def _ts(val: Any) -> pd.Timestamp:
    t = pd.Timestamp(val)
    if t.tzinfo is None:
        return t.tz_localize(_UTC)
    return t


def _to_ny(ts: Any) -> pd.Timestamp:
    return _ts(ts).tz_convert(_NY)


def silver_bullet_window(
    ts: Any,
    *,
    use_london: bool = True,
    use_ny_am: bool = True,
    use_ny_pm: bool = True,
) -> str | None:
    """Return SB window name if ``ts`` is inside an enabled window."""
    t = _to_ny(ts).time()
    enabled = {"london": use_london, "ny_am": use_ny_am, "ny_pm": use_ny_pm}
    for name, start, end in _SB_WINDOWS:
        if not enabled.get(name, True):
            continue
        if start <= t < end:
            return name
    return None


def silver_bullet_window_key(ts: Any, name: str) -> str:
    return f"{_to_ny(ts).date().isoformat()}_{name}"


def _unix_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    return index.asi8.astype(np.float64) / 1e9


def last_closed_bar_idx(close_unix: np.ndarray, now_unix: float) -> int:
    i = int(np.searchsorted(close_unix, now_unix, side="right") - 1)
    return i if i >= 0 else -1


def _nearest_pools(
    liq: LiquidityEngine, price: float, asof: Any
) -> tuple[float | None, float | None]:
    levels = liq.active_levels(asof)
    above = sorted(
        float(lv.price)
        for lv in levels
        if LiquidityEngine.is_buy_side(lv.kind) and float(lv.price) > price
    )
    below = sorted(
        (
            float(lv.price)
            for lv in levels
            if LiquidityEngine.is_sell_side(lv.kind) and float(lv.price) < price
        ),
        reverse=True,
    )
    return (above[0] if above else None, below[0] if below else None)


def _fvg_in_pd_zone(
    direction: int, fvg_mid: float, range_low: float, range_high: float
) -> bool:
    if range_high <= range_low:
        return True
    eq = range_low + 0.5 * (range_high - range_low)
    if direction > 0:
        return fvg_mid <= eq
    return fvg_mid >= eq


def manage_position(
    pos: Position,
    idx: int,
    current_ts: float,
    current_price: float,
    current_high: float,
    current_low: float,
    cfg: SilverBulletConfig,
    rm: RiskManager,
    trades: list[dict],
) -> None:
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

    if not pos.tp1_hit and pos.tp1 is not None:
        hit = (pos.direction == "long" and current_high >= pos.tp1) or (
            pos.direction == "short" and current_low <= pos.tp1
        )
        if hit:
            if cfg.tp1_close_pct >= 0.999:
                rm.close_position(pos, pos.tp1, "tp", current_ts)
            else:
                rm.close_position(
                    pos,
                    pos.tp1,
                    "tp1_partial",
                    current_ts,
                    partial=True,
                    move_sl_to_be=cfg.move_sl_to_be_on_tp1,
                )
            if pos.status == "closed":
                trades.append(_pos_to_dict(pos))
            return

    if idx - pos.entry_idx >= cfg.max_hold_bars:
        rm.close_position(pos, current_price, "time_stop", current_ts)
        trades.append(_pos_to_dict(pos))


def _try_arm(
    *,
    df_1m: pd.DataFrame,
    eng_1m: MarketStructureEngine,
    i: int,
    wctx: _WindowCtx,
    cfg: SilverBulletConfig,
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    close: np.ndarray,
    atr_val: float,
) -> bool:
    want = wctx.want
    if want == 0:
        return False
    htf = TrendBias.BULLISH if want > 0 else TrendBias.BEARISH
    if eng_1m.nested_choch_dir(df_1m, i, htf) != want:
        return False

    if cfg.require_displacement:
        body = abs(float(close[i]) - float(open_[i]))
        need = float(atr_val) * float(cfg.displacement_atr_mult)
        if atr_val != atr_val or body < need:  # NaN atr
            return False

    fvg = detect_fvg_at(high, low, df_1m.index, i)
    if fvg is None or fvg.direction != want:
        return False

    if cfg.require_pd_zone and wctx.ssl is not None and wctx.bsl is not None:
        mid = 0.5 * (float(fvg.top) + float(fvg.bottom))
        if not _fvg_in_pd_zone(want, mid, float(wctx.ssl), float(wctx.bsl)):
            return False

    direction = "long" if want > 0 else "short"
    off = float(cfg.entry_edge_offset)
    if direction == "long":
        limit = float(fvg.bottom) + off
        sl = float(low[i]) - float(cfg.sl_buffer)
        if wctx.sweep_extreme is not None:
            sl = min(sl, float(wctx.sweep_extreme) - float(cfg.sl_buffer))
        inv = float(wctx.sweep_extreme) if wctx.sweep_extreme is not None else sl
    else:
        limit = float(fvg.top) - off
        sl = float(high[i]) + float(cfg.sl_buffer)
        if wctx.sweep_extreme is not None:
            sl = max(sl, float(wctx.sweep_extreme) + float(cfg.sl_buffer))
        inv = float(wctx.sweep_extreme) if wctx.sweep_extreme is not None else sl

    if direction == "long" and sl >= limit:
        sl = limit - max(float(cfg.sl_buffer), 1e-6)
    if direction == "short" and sl <= limit:
        sl = limit + max(float(cfg.sl_buffer), 1e-6)

    wctx.pending = {
        "direction": direction,
        "limit": limit,
        "sl": sl,
        "invalidation": inv,
        "placed_i": i,
    }
    return True


def run_backtest(
    df_15m: pd.DataFrame,
    df_1m: pd.DataFrame,
    cfg: SilverBulletConfig,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Causal Silver Bullet backtest: 15m parent liquidity + 1m execution."""
    empty: dict[str, Any] = {
        "trades": [],
        "metrics": {},
        "equity_curve": [],
        "signals": [],
    }
    if df_1m.empty or df_15m.empty:
        return empty

    eng_15m = MarketStructureEngine(cfg.swing_length_15m)
    eng_1m = MarketStructureEngine(cfg.swing_length_1m)
    eng_15m.precompute(df_15m)
    eng_1m.precompute(df_1m)

    liq = LiquidityEngine(
        use_asia=cfg.use_asia,
        use_london=cfg.use_london_liq,
        use_pd=cfg.use_pd,
        use_pw=cfg.use_pw,
        use_swing_liq=cfg.use_swing_liq,
        swing_liq_length=cfg.swing_liq_length,
    )
    liq.build(df_15m)

    atr = atr_indicator(df_1m["high"], df_1m["low"], df_1m["close"], cfg.atr_period).to_numpy(dtype=float)
    high_1 = df_1m["high"].to_numpy(dtype=float)
    low_1 = df_1m["low"].to_numpy(dtype=float)
    open_1 = df_1m["open"].to_numpy(dtype=float)
    close_1 = df_1m["close"].to_numpy(dtype=float)
    idx_1 = df_1m.index

    open_unix_1 = _unix_seconds(pd.DatetimeIndex(idx_1))
    close_unix_1 = open_unix_1 + 60.0
    open_unix_15 = _unix_seconds(pd.DatetimeIndex(df_15m.index))
    close_unix_15 = open_unix_15 + 15.0 * 60.0

    rm = RiskManager(cfg, account_size=cfg.account_size, backtest_mode=True)
    trades: list[dict] = []
    signals: list[dict] = []
    equity_curve: list[float] = []
    open_pos: Position | None = None
    wctx: _WindowCtx | None = None

    n = len(df_1m)
    for i in range(n):
        bar_close_unix = float(close_unix_1[i])
        bar_close = _ts(idx_1[i]) + timedelta(minutes=1)

        if open_pos is not None and open_pos.status == "open":
            manage_position(
                open_pos,
                i,
                bar_close_unix,
                float(close_1[i]),
                float(high_1[i]),
                float(low_1[i]),
                cfg,
                rm,
                trades,
            )
            if open_pos.status == "closed":
                open_pos = None

        win_name = silver_bullet_window(
            bar_close,
            use_london=cfg.use_london,
            use_ny_am=cfg.use_ny_am,
            use_ny_pm=cfg.use_ny_pm,
        )

        if win_name is not None:
            key = silver_bullet_window_key(bar_close, win_name)
            if wctx is None or wctx.key != key:
                i15 = last_closed_bar_idx(close_unix_15, bar_close_unix)
                px = float(close_1[i])
                asof = df_15m.index[i15] if i15 >= 0 else bar_close
                bsl, ssl = _nearest_pools(liq, px, asof)
                if i15 >= 0:
                    snap15 = eng_15m.snapshot_at(df_15m, i15)
                    if snap15.last_swing_high is not None:
                        sh = float(snap15.last_swing_high.price)
                        if sh > px and (bsl is None or sh < bsl):
                            bsl = sh
                    if snap15.last_swing_low is not None:
                        slv = float(snap15.last_swing_low.price)
                        if slv < px and (ssl is None or slv > ssl):
                            ssl = slv
                wctx = _WindowCtx(
                    key=key, name=win_name, bsl=bsl, ssl=ssl, mid=px
                )
        elif wctx is not None and wctx.pending is None and not wctx.traded:
            if wctx.swept_side is None:
                wctx = None

        if (
            open_pos is None
            and wctx is not None
            and wctx.pending is not None
            and not wctx.traded
        ):
            pend = wctx.pending
            if i - int(pend["placed_i"]) > int(cfg.entry_timeout_bars):
                wctx.pending = None
            else:
                direction = str(pend["direction"])
                limit = float(pend["limit"])
                sl = float(pend["sl"])
                inv = pend.get("invalidation")
                dead = False
                if inv is not None:
                    if direction == "long" and low_1[i] < float(inv):
                        dead = True
                    if direction == "short" and high_1[i] > float(inv):
                        dead = True
                hit = (direction == "long" and low_1[i] <= limit) or (
                    direction == "short" and high_1[i] >= limit
                )
                if dead:
                    wctx.pending = None
                elif hit:
                    allowed, _ = rm.can_trade(bar_close_unix)
                    if allowed and not (direction == "short" and cfg.long_only):
                        risk = abs(limit - sl) or max(abs(limit) * 1e-4, 1e-6)
                        tp = (
                            limit + cfg.tp_r * risk
                            if direction == "long"
                            else limit - cfg.tp_r * risk
                        )
                        atr_val = (
                            float(atr[i])
                            if i < len(atr) and atr[i] == atr[i]
                            else risk
                        )
                        pos = rm.open_position(
                            entry_price=limit,
                            atr=atr_val,
                            direction=direction,
                            entry_time=bar_close_unix,
                            entry_idx=i,
                            stop_loss=sl,
                            tp1=tp,
                            tp2=None,
                        )
                        if pos is not None:
                            pos.meta = {
                                "sb_window": wctx.name,
                                "sweep_side": wctx.swept_side,
                                "bsl": wctx.bsl,
                                "ssl": wctx.ssl,
                                "draw": wctx.draw,
                            }
                            open_pos = pos
                            wctx.traded = True
                            wctx.pending = None
                            signals.append(
                                {
                                    "ts": str(bar_close),
                                    "window": wctx.name,
                                    "direction": direction,
                                    "entry": limit,
                                    "sl": sl,
                                    "tp": tp,
                                }
                            )

        if (
            open_pos is None
            and win_name is not None
            and wctx is not None
            and not wctx.traded
            and wctx.pending is None
        ):
            if wctx.swept_side is None:
                if wctx.bsl is not None and high_1[i] > wctx.bsl:
                    wctx.swept_side = "buy"
                    wctx.sweep_extreme = float(high_1[i])
                    wctx.sweep_i = i
                    wctx.want = -1
                    wctx.draw = wctx.ssl
                elif wctx.ssl is not None and low_1[i] < wctx.ssl:
                    wctx.swept_side = "sell"
                    wctx.sweep_extreme = float(low_1[i])
                    wctx.sweep_i = i
                    wctx.want = 1
                    wctx.draw = wctx.bsl

            if wctx.swept_side is not None and i > wctx.sweep_i:
                atr_i = float(atr[i]) if i < len(atr) else float("nan")
                _try_arm(
                    df_1m=df_1m,
                    eng_1m=eng_1m,
                    i=i,
                    wctx=wctx,
                    cfg=cfg,
                    high=high_1,
                    low=low_1,
                    open_=open_1,
                    close=close_1,
                    atr_val=atr_i,
                )

        if verbose and i % 20000 == 0 and i > 0:
            print(
                f"  [{100 * i / n:5.1f}%] trades={len(trades)} "
                f"equity={rm.current_equity:.2f}"
            )

        if open_pos is not None or i % 60 == 0:
            equity_curve.append(rm.current_equity)

    if open_pos is not None and open_pos.status != "closed":
        last_ts = float(close_unix_1[-1])
        rm.close_position(
            open_pos, float(close_1[-1]), "backtest_end", last_ts
        )
        trades.append(_pos_to_dict(open_pos))

    metrics = _compute_metrics(trades, equity_curve, cfg)  # type: ignore[arg-type]
    metrics["setups"] = len(signals)
    return {
        "trades": trades,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "signals": signals,
    }
