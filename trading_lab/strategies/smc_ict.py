"""SMC/ICT multi-timeframe strategy (nested entry).

Flow: 4h bias → liquidity sweep → 15m align/ChoCh
      → watch 1m for ChoCh → 1m FVG limit.
Exits (P0): SL beyond sweep extreme; TP1 nearest IRL, TP2 ERL/HTF draw
      (RR fallback when pools missing). BE after TP1 optional (off by default).

15m OB+FVG is optional context; POI retest is not required before 1m ChoCh.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd

from ..backtest import _compute_metrics, _pos_to_dict
from ..indicators import atr as atr_indicator
from ..risk import Position, RiskManager
from .smc.flow_filters import (
    dealing_range_from_swings,
    dealing_range_sweep_bar,
    is_high_side_liquidity,
    m15_mss_ok,
    ny_session_ok,
    premium_discount_ok,
    session_from_ny_hour,
    sweep_close_back_ok,
    sweep_kind_allowed,
)
from .smc.liquidity import LiquidityEngine
from .smc.state_machine import StateMachine
from .smc.structure import MarketStructureEngine
from .smc.supply_demand import SupplyDemandEngine, detect_fvg_at, order_block_for_fvg
from .smc.types import LiquidityLevel, PendingEntry, TradeState, TrendBias, Zone


@dataclass(frozen=True)
class SmcIctConfig:
    """Configuration for SMC/ICT MTF strategy."""

    asset: str = "XAU/USD"
    asset_label: str = "XAU/USD"

    tf_bias: str = "4h"
    tf_poi: str = "15m"
    tf_entry: str = "1m"

    swing_length_4h: int = 10  # external 4h structure (L=3 flips on internal rallies)
    swing_length_15m: int = 4
    swing_length_1m: int = 3

    ob_require_fvg: bool = True
    fvg_mitigation: Literal["wick", "close"] = "wick"
    ob_range: Literal["wick", "body"] = "wick"  # grid favorite: wick
    max_active_zones: int = 32
    ob_lookback: int = 15  # grid favorite on EURUSD 2020 1m entry
    # Playbook: enter on 1m FVG after ChoCh (OB-only fallback off by default)
    require_1m_fvg: bool = True
    # Nested 1m ChoCh: classify vs pullback bias (opposite HTF), not rolling 1m BOS
    nested_choch_vs_pullback: bool = True
    # After arming, allow tighter FVG refine for this many 1m bars, then lock
    choch_refine_bars: int = 5

    asia_start: str = "19:00"
    asia_end: str = "00:00"
    london_start: str = "02:00"
    london_end: str = "05:00"
    use_asia: bool = True
    use_london: bool = True
    use_pd: bool = True
    use_pw: bool = True
    # Local swing highs/lows — off by default (false sweeps); session/PD/PW only
    use_swing_liq: bool = False
    swing_liq_length: int = 5
    # After 15m align: look back for 1m ChoCh already printed in the impulse
    choch_lookback_bars: int = 90  # 1m bars (~1.5h)

    # Limit placement: research showed pullbacks mostly hit FVG/overlap
    entry_limit_mode: Literal["fvg_edge", "fvg_mid", "ob_edge", "merged_edge"] = (
        "fvg_edge"
    )
    limit_at_ob_edge: bool = True  # legacy; ignored when entry_limit_mode set
    entry_timeout_bars: int = 90
    # Short: sell slightly below FVG top (proximal); long: buy slightly above FVG bottom
    entry_edge_offset: float = 0.00005  # ~0.5 pip EUR
    # SL: 1m OB (production baseline) vs structure swing / sweep
    sl_mode: Literal["ob", "swing", "max", "sweep"] = "ob"
    sl_buffer: float = 0.50  # absolute buffer beyond OB/swing/sweep (forex ~0.0001–0.0002)
    # If sweep SL is farther than this from entry, fall back to OB/swing (0 = no cap)
    max_sl_dist: float = 0.0012  # ~12 pips EUR

    atr_period: int = 14
    sl_atr_mult: float = 1.0  # unused when structure SL set; keeps RiskConfig happy
    # Exits: fixed R-multiples (baseline) vs liquidity IRL/ERL (P0, evolvable)
    tp_mode: Literal["rr", "liquidity"] = "rr"
    tp1_r: float = 1.5  # RR mode / liquidity fallback
    tp2_r: float = 3.0
    min_irl_r: float = 1.0  # IRL must be ≥ this many R beyond entry
    min_erl_r: float = 2.0  # ERL must be ≥ this many R (else RR stretch)
    max_irl_r: float = 3.0  # skip pools farther than this for TP1
    max_erl_r: float = 6.0  # hard cap on TP2 distance in R
    tp_fallback_rr: bool = True  # if no IRL/ERL pools, use tp1_r/tp2_r
    tp1_close_pct: float = 0.50
    # After TP1: move SL to entry (baseline); off for ICT ERL runners
    move_sl_to_be_on_tp1: bool = True
    max_hold_bars: int = 360  # 1m bars (~6h)

    account_size: float = 10000.0
    risk_per_trade: float = 0.01
    max_daily_loss: float = 0.03
    cooldown_hours: int = 0
    max_concurrent_positions: int = 1
    long_only: bool = False

    # --- Flow gates (ablation defaults: B + sell-side) ---
    # A: sweep rejection (close back inside after wick raid)
    flow_a_sweep_close_back: bool = False
    flow_a_sweep_close_mode: Literal["wick", "body"] = "body"
    # B: premium/discount vs dealing range after sweep
    flow_b_premium_discount: bool = True
    flow_b_range_mode: Literal["15m_swing", "sweep_bar"] = "15m_swing"
    flow_b_equilibrium: float = 0.5
    # C: 15m MSS after sweep (ChoCh + optional displacement), not bias-only align
    flow_c_require_15m_mss: bool = False
    flow_c_displacement: bool = False
    flow_c_displacement_atr_mult: float = 1.0
    flow_c_align_fallback: bool = True  # when C off: keep aligned OR 15m ChoCh
    # Attribution filters (TP analysis 2024–now with Stage B)
    # Prefer flow_killzone; flow_ny_session_only kept as legacy alias for "ny"
    flow_killzone: Literal["off", "ny", "ny_am", "london", "london_ny"] = "off"
    flow_ny_session_only: bool = False  # legacy → treated as killzone "ny"
    flow_sell_side_sweeps_only: bool = True  # asia_low / london_low / pdl only

    # Playbook invariants — never mutate these in evolution
    STRATEGY_LOCKS: ClassVar[dict[str, Any]] = {
        "nested_choch_vs_pullback": True,
        "use_swing_liq": False,
        "entry_limit_mode": "fvg_edge",
        "require_1m_fvg": True,
        "long_only": False,
        "swing_length_4h": 10,  # HTF bias stability
    }

    @staticmethod
    def search_space() -> list[tuple[str, float, float, float]]:
        """Numeric knobs safe to evolve (exit/fill/geometry — not flow)."""
        return [
            # Entry buffer + fill quality
            ("entry_edge_offset", 0.0, 0.00025, 0.00005),
            ("entry_timeout_bars", 20, 120, 5),
            ("sl_buffer", 0.00005, 0.0005, 0.00005),
            ("max_sl_dist", 0.0004, 0.0025, 0.0002),
            # ChoCh arming / refine window
            ("choch_lookback_bars", 30, 180, 15),
            ("choch_refine_bars", 0, 15, 1),
            # Structure (LTF only — 4h locked)
            ("swing_length_15m", 3, 8, 1),
            ("swing_length_1m", 3, 6, 1),
            ("ob_lookback", 5, 20, 5),
            # Exits — RR fallback / partial sizing (liquidity mode uses pools first)
            ("tp1_r", 1.0, 2.5, 0.25),
            ("tp2_r", 2.0, 5.0, 0.5),
            ("min_irl_r", 0.5, 2.0, 0.25),
            ("min_erl_r", 1.5, 5.0, 0.5),
            ("max_irl_r", 1.5, 4.0, 0.5),
            ("max_erl_r", 3.0, 8.0, 1.0),
            ("tp1_close_pct", 0.25, 0.75, 0.25),
            ("max_hold_bars", 60, 720, 30),
        ]

    @staticmethod
    def categorical_space() -> list[tuple[str, list]]:
        """Discrete knobs safe to evolve (filters / SL anatomy — not flow)."""
        return [
            ("sl_mode", ["ob", "swing", "max", "sweep"]),
            ("tp_mode", ["rr", "liquidity"]),
            ("move_sl_to_be_on_tp1", [True, False]),
            ("ob_range", ["wick", "body"]),
            ("fvg_mitigation", ["wick", "close"]),
            ("ob_require_fvg", [True, False]),
            # Liquidity quality filters (at least one source kept by mutate guard)
            ("use_asia", [True, False]),
            ("use_london", [True, False]),
            ("use_pd", [True, False]),
            ("use_pw", [True, False]),
        ]

    @staticmethod
    def flow_evolution_space() -> list[tuple[str, list]]:
        """Flow gates — mutate one at a time; ablation order A → B → C."""
        return [
            ("flow_a_sweep_close_back", [True, False]),
            ("flow_a_sweep_close_mode", ["body", "wick"]),
            ("flow_b_premium_discount", [True, False]),
            ("flow_b_range_mode", ["15m_swing", "sweep_bar"]),
            ("flow_c_require_15m_mss", [True, False]),
            ("flow_c_displacement", [True, False]),
            ("flow_c_align_fallback", [True, False]),
            ("flow_killzone", ["off", "ny", "ny_am", "london", "london_ny"]),
            ("flow_ny_session_only", [True, False]),
            ("flow_sell_side_sweeps_only", [True, False]),
        ]

    @classmethod
    def pin_strategy_locks(cls, cfg: "SmcIctConfig") -> "SmcIctConfig":
        """Force playbook invariants onto a config."""
        return replace(cfg, **cls.STRATEGY_LOCKS)


@dataclass
class EntryZone1m:
    """1m entry anatomy after ChoCh — OB/FVG kept separate for limit pricing."""

    direction: int
    ob: Zone | None
    fvg: Zone | None
    merged: Zone

    @property
    def anchor(self) -> Zone:
        """Preferred zone for limit (FVG if present)."""
        if self.fvg is not None:
            return self.fvg
        if self.ob is not None:
            return self.ob
        return self.merged


_TF_DELTA = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
}


def _ts(val: Any) -> pd.Timestamp:
    t = pd.Timestamp(val)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t


def last_closed_bar_idx(
    df: pd.DataFrame, now: Any, interval: timedelta
) -> int:
    """Highest index whose candle is fully closed at `now` (open + interval <= now)."""
    if df.empty:
        return -1
    now_ts = _ts(now)
    opens = df.index
    lo, hi = 0, len(df) - 1
    best = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        close_at = _ts(opens[mid]) + interval
        if close_at <= now_ts:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def advance_closed_idx(
    df: pd.DataFrame, now: Any, interval: timedelta, last_i: int
) -> int:
    """Monotonic advance of last closed bar index (O(1) amortized)."""
    if last_i < 0:
        return last_closed_bar_idx(df, now, interval)
    n = len(df)
    i = last_i
    now_ts = _ts(now)
    while i + 1 < n and (_ts(df.index[i + 1]) + interval) <= now_ts:
        i += 1
    return i


def _advance_closed_ts(
    close_ts: np.ndarray, now_ts: float, last_i: int
) -> int:
    """Advance using precomputed bar-close unix timestamps (seconds)."""
    n = len(close_ts)
    if n == 0:
        return -1
    if last_i < 0:
        lo, hi, best = 0, n - 1, -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if close_ts[mid] <= now_ts:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best
    i = last_i
    while i + 1 < n and close_ts[i + 1] <= now_ts:
        i += 1
    return i


def _unix_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    """Bar open times as unix seconds (float64).

    Dukascopy frames may store ms-resolution ints in ``asi8``; casting ns→float
    loses precision, so scale in int64 first and detect the unit from magnitude.
    """
    if len(index) == 0:
        return np.array([], dtype=np.float64)
    raw = index.asi8
    sample = abs(int(raw[len(raw) // 2]))
    if sample > 10**17:  # nanoseconds
        return (raw // 10**9).astype(np.float64)
    if sample > 10**14:  # microseconds
        return (raw // 10**6).astype(np.float64)
    if sample > 10**11:  # milliseconds
        return (raw // 10**3).astype(np.float64)
    return raw.astype(np.float64)  # already seconds


def _bar_close_ts(index: pd.DatetimeIndex, interval: timedelta) -> np.ndarray:
    """Unix seconds when each bar closes (open + interval)."""
    return _unix_seconds(index) + interval.total_seconds()


def _nested_1m_choch_dir(
    eng_1m: MarketStructureEngine,
    df_1m: pd.DataFrame,
    bar_i: int,
    want: int,
    cfg: SmcIctConfig,
) -> int:
    """1m ChoCh in setup direction for nested entry.

    Default: classify vs pullback bias (opposite HTF) so confirmation breaks
    are ChoCh even when rolling 1m bias already flipped to HTF (else BOS).
    Fallback: classic snapshot ChoCh only.
    """
    if want == 0:
        return 0
    if cfg.nested_choch_vs_pullback:
        return eng_1m.nested_choch_dir(df_1m, bar_i, TrendBias(want))
    snap = eng_1m.snapshot_at(df_1m, bar_i)
    return want if snap.last_choch_dir == want else 0


def _arm_pending_from_choch(
    df_1m: pd.DataFrame,
    eng_1m: MarketStructureEngine,
    choch_i: int,
    placed_i: int,
    placed_ts: Any,
    cfg: SmcIctConfig,
    sm: StateMachine,
) -> bool:
    """Arm 1m FVG limit from ChoCh bar `choch_i`; timeout from `placed_i`."""
    want = int(sm.ctx.bias)
    choch_dir = _nested_1m_choch_dir(eng_1m, df_1m, choch_i, want, cfg)
    if not sm.on_choch(choch_dir):
        return False
    direction_i = want
    entry_z = _entry_zone_1m(df_1m, choch_i, direction_i, cfg)
    if entry_z is None:
        sm.cancel_pending()
        return False
    direction = "long" if direction_i > 0 else "short"
    snap_1m = eng_1m.snapshot_at(df_1m, choch_i)
    swing_sl = (
        snap_1m.last_swing_low.price
        if direction == "long" and snap_1m.last_swing_low
        else (
            snap_1m.last_swing_high.price
            if direction == "short" and snap_1m.last_swing_high
            else (
                entry_z.merged.bottom if direction == "long" else entry_z.merged.top
            )
        )
    )
    limit, sl = _limit_and_sl(
        entry_z,
        direction,
        swing_sl,
        cfg,
        sweep_extreme=sm.ctx.sweep_extreme,
    )
    if cfg.sl_mode == "sweep" and sm.ctx.sweep_extreme is not None:
        inv = float(sm.ctx.sweep_extreme)
    else:
        inv = (
            snap_1m.last_swing_low.price
            if direction == "long" and snap_1m.last_swing_low
            else (
                snap_1m.last_swing_high.price
                if direction == "short" and snap_1m.last_swing_high
                else limit
            )
        )
    if direction == "long" and sl >= limit:
        sl = limit - max(cfg.sl_buffer, 1e-6)
    if direction == "short" and sl <= limit:
        sl = limit + max(cfg.sl_buffer, 1e-6)
    anchor = entry_z.anchor
    sm.set_pending(
        PendingEntry(
            direction=direction,
            limit_price=limit,
            stop_loss=sl,
            swing_sl_price=float(swing_sl),
            zone_top=anchor.top,
            zone_bottom=anchor.bottom,
            placed_idx=placed_i,
            placed_ts=placed_ts,
            timeout_bars=cfg.entry_timeout_bars,
            invalidation_level=inv,
        )
    )
    return True


def _lookback_arm_choch(
    df_1m: pd.DataFrame,
    eng_1m: MarketStructureEngine,
    *,
    now_i: int,
    sweep_i: int,
    m15_choch_i: int,
    cfg: SmcIctConfig,
    sm: StateMachine,
) -> bool:
    """Arm FVG from the latest nested 1m ChoCh after the 15m impulse."""
    if sm.ctx.state != TradeState.WAIT_FOR_CHOCH or sm.ctx.pending is not None:
        return False
    want = int(sm.ctx.bias)
    if want == 0:
        return False
    lookback = max(1, int(cfg.choch_lookback_bars))
    anchor = m15_choch_i if m15_choch_i > 0 else sweep_i
    start = max(0, now_i - lookback, anchor)
    # Most recent ChoCh in window (confirmation of the impulse, not first micro)
    for j in range(now_i, start - 1, -1):
        if _nested_1m_choch_dir(eng_1m, df_1m, j, want, cfg) != want:
            continue
        if _arm_pending_from_choch(
            df_1m, eng_1m, j, now_i, df_1m.index[now_i], cfg, sm
        ):
            return True
    return False


def manage_position(
    pos: Position,
    idx: int,
    current_ts: float,
    current_price: float,
    current_high: float,
    current_low: float,
    cfg: SmcIctConfig,
    rm: RiskManager,
    trades: list[dict],
) -> None:
    """Exit hierarchy: SL → TP1 partial → TP2 → time stop."""
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
            rm.close_position(
                pos,
                pos.tp1,
                "tp1_partial",
                current_ts,
                partial=True,
                move_sl_to_be=bool(cfg.move_sl_to_be_on_tp1),
            )
            return
        if pos.direction == "short" and current_low <= pos.tp1:
            rm.close_position(
                pos,
                pos.tp1,
                "tp1_partial",
                current_ts,
                partial=True,
                move_sl_to_be=bool(cfg.move_sl_to_be_on_tp1),
            )
            return

    if pos.tp1_hit and pos.tp2 is not None and not pos.tp2_hit:
        if pos.direction == "long" and current_high >= pos.tp2:
            rm.close_position(pos, pos.tp2, "tp2", current_ts)
            pos.tp2_hit = True
            trades.append(_pos_to_dict(pos))
            return
        if pos.direction == "short" and current_low <= pos.tp2:
            rm.close_position(pos, pos.tp2, "tp2", current_ts)
            pos.tp2_hit = True
            trades.append(_pos_to_dict(pos))
            return

    candles_held = idx - pos.entry_idx
    if candles_held >= cfg.max_hold_bars:
        rm.close_position(pos, current_price, "time_stop", current_ts)
        trades.append(_pos_to_dict(pos))
        return


def _entry_zone_1m(
    df_1m: pd.DataFrame,
    choch_idx: int,
    direction: int,
    cfg: SmcIctConfig,
) -> EntryZone1m | None:
    """1m OB (+ FVG) that caused the ChoCh displacement."""
    high = df_1m["high"].to_numpy(dtype=float)
    low = df_1m["low"].to_numpy(dtype=float)
    opens = df_1m["open"].to_numpy(dtype=float)
    closes = df_1m["close"].to_numpy(dtype=float)

    def _ob_range(j: int) -> tuple[float, float]:
        if cfg.ob_range == "body":
            return max(float(opens[j]), float(closes[j])), min(
                float(opens[j]), float(closes[j])
            )
        return float(high[j]), float(low[j])

    for i in range(choch_idx, max(1, choch_idx - 5) - 1, -1):
        fvg = detect_fvg_at(high, low, df_1m.index, i)
        if fvg is None or fvg.direction != direction:
            continue
        ob = order_block_for_fvg(df_1m, fvg, lookback=cfg.ob_lookback)
        if ob is not None and cfg.ob_range == "body":
            top, bottom = _ob_range(ob.created_idx)
            ob = Zone(
                direction=ob.direction,
                top=top,
                bottom=bottom,
                created_idx=ob.created_idx,
                created_ts=ob.created_ts,
                kind="ob",
            )
        if ob is None:
            return EntryZone1m(
                direction=direction,
                ob=None,
                fvg=fvg,
                merged=fvg,
            )
        merged = Zone(
            direction=direction,
            top=max(ob.top, fvg.top),
            bottom=min(ob.bottom, fvg.bottom),
            created_idx=fvg.created_idx,
            created_ts=fvg.created_ts,
            kind="ob_fvg",
        )
        return EntryZone1m(direction=direction, ob=ob, fvg=fvg, merged=merged)

    # Fallback: last opposing candle as OB without FVG (off when require_1m_fvg)
    if cfg.require_1m_fvg:
        return None
    for j in range(choch_idx - 1, max(-1, choch_idx - cfg.ob_lookback) - 1, -1):
        if j < 0:
            break
        bullish_ob = direction > 0 and closes[j] < opens[j]
        bearish_ob = direction < 0 and closes[j] > opens[j]
        if not (bullish_ob or bearish_ob):
            continue
        top, bottom = _ob_range(j)
        ob = Zone(
            direction=direction,
            top=top,
            bottom=bottom,
            created_idx=j,
            created_ts=df_1m.index[j],
            kind="ob",
        )
        return EntryZone1m(direction=direction, ob=ob, fvg=None, merged=ob)
    return None


def _limit_price(entry: EntryZone1m, direction: str, cfg: SmcIctConfig) -> float:
    """Limit on FVG edge/mid by default (research: touches land on FVG/overlap)."""
    mode = cfg.entry_limit_mode
    if mode.startswith("fvg") and entry.fvg is not None:
        z = entry.fvg
    elif mode.startswith("ob") and entry.ob is not None:
        z = entry.ob
    else:
        z = entry.merged

    if mode.endswith("mid"):
        return float(z.mid)
    # Proximal edge for pullback: after displacement, price is beyond the FVG;
    # limit rests on the near side (short → top, long → bottom).
    off = float(getattr(cfg, "entry_edge_offset", 0.0) or 0.0)
    if direction == "long":
        return float(z.bottom) + off
    return float(z.top) - off


def _stop_from_ob_or_swing(
    entry: EntryZone1m,
    direction: str,
    swing_price: float,
    cfg: SmcIctConfig,
    *,
    sweep_extreme: float | None = None,
    entry_price: float | None = None,
) -> float:
    """SL beyond sweep extreme (ICT) or at 1m OB / structure swing.

    When ``sl_mode=sweep`` but the stop would exceed ``max_sl_dist`` from the
    limit/entry, fall back to OB/swing so deep wicks do not starve R-multiples.
    """
    buf = float(cfg.sl_buffer)
    mode = getattr(cfg, "sl_mode", "ob")

    def _structure_sl() -> float:
        ob_px: float | None = None
        if entry.ob is not None:
            ob_px = float(entry.ob.bottom if direction == "long" else entry.ob.top)
        swing_px = float(swing_price)
        # For fallback after a too-wide sweep, prefer OB then swing
        sub = mode if mode in ("ob", "swing", "max") else "ob"
        if sub == "ob" and ob_px is not None:
            anchor = ob_px
        elif sub == "max" and ob_px is not None:
            anchor = (
                min(ob_px, swing_px) if direction == "long" else max(ob_px, swing_px)
            )
        elif ob_px is not None and mode == "sweep":
            anchor = ob_px
        else:
            anchor = swing_px
        if direction == "long":
            return anchor - buf
        return anchor + buf

    if mode == "sweep" and sweep_extreme is not None:
        if direction == "long":
            sl = float(sweep_extreme) - buf
        else:
            sl = float(sweep_extreme) + buf
        ref = float(entry_price) if entry_price is not None else None
        cap = float(getattr(cfg, "max_sl_dist", 0.0) or 0.0)
        if ref is not None and cap > 0 and abs(ref - sl) > cap:
            return _structure_sl()
        return sl
    return _structure_sl()


def _limit_and_sl(
    entry: EntryZone1m,
    direction: str,
    swing_price: float,
    cfg: SmcIctConfig,
    *,
    sweep_extreme: float | None = None,
) -> tuple[float, float]:
    limit = _limit_price(entry, direction, cfg)
    sl = _stop_from_ob_or_swing(
        entry,
        direction,
        swing_price,
        cfg,
        sweep_extreme=sweep_extreme,
        entry_price=limit,
    )
    return float(limit), float(sl)


def _rr_targets(
    entry: float, sl: float, direction: str, cfg: SmcIctConfig
) -> tuple[float, float]:
    risk = abs(entry - sl)
    if risk <= 0:
        risk = max(abs(entry) * 1e-4, 1e-6)
    if direction == "long":
        return entry + cfg.tp1_r * risk, entry + cfg.tp2_r * risk
    return entry - cfg.tp1_r * risk, entry - cfg.tp2_r * risk


def _risk_distance(entry: float, sl: float) -> float:
    risk = abs(entry - sl)
    if risk <= 0:
        return max(abs(entry) * 1e-4, 1e-6)
    return risk


def _exit_targets(
    entry: float,
    sl: float,
    direction: str,
    cfg: SmcIctConfig,
    *,
    liq: LiquidityEngine | None = None,
    asof_ts: Any = None,
    dealing_high: float | None = None,
    dealing_low: float | None = None,
) -> tuple[float, float]:
    """TP1/TP2: liquidity IRL/ERL when enabled, else fixed R-multiples."""
    if getattr(cfg, "tp_mode", "rr") != "liquidity" or liq is None:
        return _rr_targets(entry, sl, direction, cfg)

    risk = _risk_distance(entry, sl)
    min_irl = float(getattr(cfg, "min_irl_r", 1.0)) * risk
    max_erl = float(getattr(cfg, "max_erl_r", 6.0)) * risk
    max_irl = float(getattr(cfg, "max_irl_r", 3.0)) * risk
    irl, erl = liq.trade_targets(
        entry,
        direction,
        asof_ts,
        dealing_high=dealing_high,
        dealing_low=dealing_low,
        min_dist=min_irl,
        max_dist=max_erl,
    )
    # Prefer an IRL inside max_irl_r; if nearest is too far, fall back to RR TP1
    if irl is not None and abs(float(irl) - entry) > max_irl + 1e-12:
        irl = None
        erl = None

    if irl is None:
        return _rr_targets(entry, sl, direction, cfg)

    tp1 = float(irl)
    min_erl = float(getattr(cfg, "min_erl_r", 2.0)) * risk
    tp2: float | None = None
    if erl is not None:
        erl_f = float(erl)
        erl_r = abs(erl_f - entry)
        farther = (direction == "long" and erl_f > tp1) or (
            direction == "short" and erl_f < tp1
        )
        if farther and erl_r >= min_erl:
            # Clamp to max_erl_r so runners stay reachable
            if direction == "long":
                tp2 = min(erl_f, entry + max_erl)
            else:
                tp2 = max(erl_f, entry - max_erl)

    if tp2 is None:
        if getattr(cfg, "tp_fallback_rr", True):
            _, tp2_rr = _rr_targets(entry, sl, direction, cfg)
            if direction == "long":
                tp2 = max(float(tp2_rr), tp1 + 0.25 * risk)
                tp2 = min(tp2, entry + max_erl)
            else:
                tp2 = min(float(tp2_rr), tp1 - 0.25 * risk)
                tp2 = max(tp2, entry - max_erl)
        elif direction == "long":
            tp2 = entry + max(min_erl, abs(tp1 - entry) * 1.5)
            tp2 = min(max(tp2, tp1 + 0.25 * risk), entry + max_erl)
        else:
            tp2 = entry - max(min_erl, abs(tp1 - entry) * 1.5)
            tp2 = max(min(tp2, tp1 - 0.25 * risk), entry - max_erl)

    # Ensure TP2 is strictly beyond TP1
    if direction == "long" and tp2 <= tp1:
        tp2 = min(tp1 + 0.25 * risk, entry + max_erl)
        if tp2 <= tp1:
            return _rr_targets(entry, sl, direction, cfg)
    if direction == "short" and tp2 >= tp1:
        tp2 = max(tp1 - 0.25 * risk, entry - max_erl)
        if tp2 >= tp1:
            return _rr_targets(entry, sl, direction, cfg)

    return tp1, float(tp2)


def _setup_tags_at_fill(
    sm: StateMachine,
    *,
    direction: str,
    entry: float,
    sl: float,
    now: Any,
) -> dict[str, Any]:
    """Causal setup features for TP attribution (no lookahead)."""
    ts = _ts(now)
    ny = ts.tz_convert("America/New_York")
    berlin = ts.tz_convert("Europe/Berlin")
    hour_ny = int(ny.hour)
    session = session_from_ny_hour(hour_ny)

    lo = sm.ctx.dealing_range_low
    hi = sm.ctx.dealing_range_high
    pd_side = "unknown"
    eq = None
    if lo is not None and hi is not None and hi > lo:
        eq = lo + 0.5 * (hi - lo)
        if entry <= eq:
            pd_side = "discount"
        else:
            pd_side = "premium"

    risk = abs(entry - sl)
    sweep_dist = None
    if sm.ctx.sweep_extreme is not None and risk > 0:
        sweep_dist = abs(entry - float(sm.ctx.sweep_extreme)) / risk

    kind = sm.ctx.last_sweep_kind or "unknown"
    if kind.endswith("_low") or kind in ("pdl", "pwl"):
        liq_family = "sell_side"
    elif kind.endswith("_high") or kind in ("pdh", "pwh"):
        liq_family = "buy_side"
    else:
        liq_family = "other"
    if kind.startswith("asia"):
        liq_src = "asia"
    elif kind.startswith("london"):
        liq_src = "london"
    elif kind in ("pdh", "pdl"):
        liq_src = "pd"
    elif kind in ("pwh", "pwl"):
        liq_src = "pw"
    else:
        liq_src = "other"

    return {
        "direction": direction,
        "sweep_kind": kind,
        "liq_src": liq_src,
        "liq_family": liq_family,
        "session": session,
        "hour_ny": hour_ny,
        "hour_berlin": int(berlin.hour),
        "weekday": int(berlin.weekday()),  # 0=Mon
        "pd_side": pd_side,
        "eq": eq,
        "dealing_lo": lo,
        "dealing_hi": hi,
        "sweep_extreme": sm.ctx.sweep_extreme,
        "sweep_to_entry_r": round(sweep_dist, 3) if sweep_dist is not None else None,
        "sl_dist": round(risk, 6),
        "m15_mss_bar_i": sm.ctx.m15_mss_bar_i,
        "bias": int(sm.ctx.bias),
    }


def _try_record_sweep(
    *,
    sm: StateMachine,
    liq: LiquidityEngine,
    cfg: SmcIctConfig,
    sw: LiquidityLevel,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> bool:
    """Stage A gate: accept wick sweep only if close-back (when enabled)."""
    if not sweep_close_back_ok(
        cfg,
        open_=open_,
        high=high,
        low=low,
        close=close,
        level=float(sw.price),
        kind=sw.kind,
    ):
        liq.restore_level(sw)
        return False
    if not sm.on_sweep(sw.kind, 0):
        return False
    sm.ctx.sweep_level = float(sw.price)
    if is_high_side_liquidity(sw.kind):
        sm.ctx.sweep_extreme = float(high)
    else:
        sm.ctx.sweep_extreme = float(low)
    return True


def _set_dealing_range(
    sm: StateMachine,
    cfg: SmcIctConfig,
    *,
    snap_15m,
    high: float,
    low: float,
    prior_high: float | None = None,
    prior_low: float | None = None,
) -> None:
    """Stage B: store dealing range after a confirmed sweep."""
    if cfg.flow_b_range_mode == "15m_swing" and snap_15m is not None:
        dr = dealing_range_from_swings(snap_15m)
        if dr is not None:
            sm.ctx.dealing_range_low, sm.ctx.dealing_range_high = dr
            return
    lo, hi = dealing_range_sweep_bar(
        high=high, low=low, prior_high=prior_high, prior_low=prior_low
    )
    sm.ctx.dealing_range_low = lo
    sm.ctx.dealing_range_high = hi


def _premium_discount_gate(
    sm: StateMachine,
    cfg: SmcIctConfig,
    price: float,
) -> bool:
    """Stage B: True if gate disabled or price sits in correct half."""
    if not cfg.flow_b_premium_discount:
        return True
    lo = sm.ctx.dealing_range_low
    hi = sm.ctx.dealing_range_high
    if lo is None or hi is None:
        return False
    return premium_discount_ok(cfg, sm.ctx.bias, price, lo, hi)


def _m15_mss_gate(
    sm: StateMachine,
    cfg: SmcIctConfig,
    *,
    snap_15m,
    i15: int,
    sweep_15m_i: int,
    open_: float,
    close: float,
    atr: float,
) -> bool:
    """Stage C: require 15m ChoCh after sweep (+ optional displacement).

    When disabled, always True — legacy align/ChoCh stays in ``on_15m_align``.
    """
    if not cfg.flow_c_require_15m_mss:
        return True
    if not m15_mss_ok(
        cfg,
        bias=sm.ctx.bias,
        snap=snap_15m,
        choch_bar_i=i15,
        sweep_bar_i=sweep_15m_i,
        open_=open_,
        close=close,
        atr=atr,
    ):
        return False
    sm.ctx.m15_mss_bar_i = i15
    return True


def run_backtest(
    df_4h: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_1m: pd.DataFrame,
    cfg: SmcIctConfig,
    verbose: bool = False,
) -> dict[str, Any]:
    """Path-dependent MTF backtest on closed bars only."""
    for name, df in (("4h", df_4h), ("15m", df_15m), ("1m", df_1m)):
        if df is None or df.empty:
            raise ValueError(f"df_{name} is empty")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

    df_1m = df_1m.copy()
    if "volume" not in df_1m.columns:
        df_1m["volume"] = 0.0
    df_1m["atr"] = atr_indicator(
        df_1m["high"], df_1m["low"], df_1m["close"], cfg.atr_period
    )
    df_15m = df_15m.copy()
    df_15m["atr"] = atr_indicator(
        df_15m["high"], df_15m["low"], df_15m["close"], cfg.atr_period
    )

    eng_4h = MarketStructureEngine(cfg.swing_length_4h)
    eng_15m = MarketStructureEngine(cfg.swing_length_15m)
    eng_1m = MarketStructureEngine(cfg.swing_length_1m)
    eng_4h.precompute(df_4h)
    eng_15m.precompute(df_15m)
    eng_1m.precompute(df_1m)

    liq = LiquidityEngine(
        asia_start=cfg.asia_start,
        asia_end=cfg.asia_end,
        london_start=cfg.london_start,
        london_end=cfg.london_end,
        use_asia=cfg.use_asia,
        use_london=cfg.use_london,
        use_pd=cfg.use_pd,
        use_pw=cfg.use_pw,
        use_swing_liq=cfg.use_swing_liq,
        swing_liq_length=cfg.swing_liq_length,
    )
    liq.build(df_15m)

    sd_15m = SupplyDemandEngine(
        ob_require_fvg=cfg.ob_require_fvg,
        fvg_mitigation=cfg.fvg_mitigation,
        max_zones=cfg.max_active_zones,
        ob_lookback=cfg.ob_lookback,
    )

    sm = StateMachine()
    rm = RiskManager(cfg, account_size=cfg.account_size, backtest_mode=True)  # type: ignore[arg-type]

    open_pos: Position | None = None
    trades: list[dict] = []
    equity_curve: list[float] = [rm.current_equity]
    signal_log: list[dict] = []

    delta_4h = _TF_DELTA[cfg.tf_bias]
    delta_15m = _TF_DELTA[cfg.tf_poi]
    last_4h_i = -1
    last_15m_i = -1
    last_sweep_i = 0
    last_sweep_15m_i = 0
    last_15m_choch_i = 0
    snap_4h = None
    snap_15m = None
    # Pre-extract 1m numpy for fast path
    opens_1m = df_1m["open"].to_numpy(dtype=float)
    closes_1m = df_1m["close"].to_numpy(dtype=float)
    highs_1m = df_1m["high"].to_numpy(dtype=float)
    lows_1m = df_1m["low"].to_numpy(dtype=float)
    atrs_1m = df_1m["atr"].to_numpy(dtype=float)
    times_1m = df_1m.index
    ts_1m = _unix_seconds(times_1m)
    ts_15m = _unix_seconds(df_15m.index)
    close_ts_4h = _bar_close_ts(df_4h.index, delta_4h)
    close_ts_15m = _bar_close_ts(df_15m.index, delta_15m)
    opens_15m = df_15m["open"].to_numpy(dtype=float)
    closes_15m = df_15m["close"].to_numpy(dtype=float)
    highs_15m = df_15m["high"].to_numpy(dtype=float)
    lows_15m = df_15m["low"].to_numpy(dtype=float)
    atrs_15m = df_15m["atr"].to_numpy(dtype=float)

    n = len(df_1m)
    warmup = max(cfg.atr_period + 5, cfg.swing_length_1m * 2 + 5)
    # Precompute NY calendar-day ids once (per-bar tz_convert is far too slow)
    ny_day_ids = times_1m.tz_convert("America/New_York").floor("D").asi8
    last_ny_day = None

    for i in range(warmup, n):
        now = times_1m[i]
        current_ts = float(ts_1m[i])
        price = float(closes_1m[i])
        high = float(highs_1m[i])
        low = float(lows_1m[i])
        atr_val = float(atrs_1m[i]) if not np.isnan(atrs_1m[i]) else 0.0

        # New NY calendar day → expire stale sweeps (need today's session/PD/PW)
        ny_day = int(ny_day_ids[i])
        if last_ny_day is not None and ny_day != last_ny_day:
            sm.on_new_trading_day()
            last_15m_choch_i = 0
        last_ny_day = ny_day

        # Manage open trade first
        if open_pos is not None:
            manage_position(
                open_pos, i, current_ts, price, high, low, cfg, rm, trades
            )
            if open_pos.status == "closed":
                open_pos = None
                sm.on_trade_closed()

        # HTF cache refresh on new closed bars
        i4 = _advance_closed_ts(close_ts_4h, current_ts, last_4h_i)
        i15 = _advance_closed_ts(close_ts_15m, current_ts, last_15m_i)

        if i4 >= 0 and i4 != last_4h_i:
            last_4h_i = i4
            snap_4h = eng_4h.snapshot_at(df_4h, i4)
        if i15 >= 0 and i15 != last_15m_i:
            last_15m_i = i15
            snap_15m = eng_15m.snapshot_at(df_15m, i15)
            sd_15m.update_through(df_15m, i15)
            # Track 15m ChoCh even while in a trade (impulse anchor for lookback)
            if (
                snap_4h is not None
                and snap_15m.last_choch_dir != 0
                and snap_15m.last_choch_dir == int(snap_4h.bias)
            ):
                # Map 15m bar → 1m index at/before this 15m open
                last_15m_choch_i = int(
                    times_1m.searchsorted(df_15m.index[i15], side="right") - 1
                )
                last_15m_choch_i = max(0, last_15m_choch_i)
            for sw in liq.detect_sweeps(
                float(highs_15m[i15]),
                float(lows_15m[i15]),
                float(ts_15m[i15]),
                allow_kind=lambda k: sweep_kind_allowed(cfg, k),
            ):
                if _try_record_sweep(
                    sm=sm,
                    liq=liq,
                    cfg=cfg,
                    sw=sw,
                    open_=float(opens_15m[i15]),
                    high=float(highs_15m[i15]),
                    low=float(lows_15m[i15]),
                    close=float(closes_15m[i15]),
                ):
                    prior_h = float(highs_15m[i15 - 1]) if i15 > 0 else None
                    prior_l = float(lows_15m[i15 - 1]) if i15 > 0 else None
                    _set_dealing_range(
                        sm,
                        cfg,
                        snap_15m=snap_15m,
                        high=float(highs_15m[i15]),
                        low=float(lows_15m[i15]),
                        prior_high=prior_h,
                        prior_low=prior_l,
                    )
                    last_sweep_i = i
                    last_sweep_15m_i = i15
                    last_15m_choch_i = 0  # new sweep → wait for fresh 15m ChoCh
        if i4 < 0 or snap_4h is None:
            if i % 60 == 0:
                equity_curve.append(rm.current_equity)
            continue

        # 4h bias only — 15m may align after the sweep
        sm.on_4h_bias(snap_4h.bias)

        # Sweeps on 1m wick
        for sw in liq.detect_sweeps(
            high, low, current_ts, allow_kind=lambda k: sweep_kind_allowed(cfg, k)
        ):
            if _try_record_sweep(
                sm=sm,
                liq=liq,
                cfg=cfg,
                sw=sw,
                open_=float(opens_1m[i]),
                high=high,
                low=low,
                close=price,
            ):
                prior_h = float(highs_1m[i - 1]) if i > 0 else None
                prior_l = float(lows_1m[i - 1]) if i > 0 else None
                _set_dealing_range(
                    sm,
                    cfg,
                    snap_15m=snap_15m,
                    high=high,
                    low=low,
                    prior_high=prior_h,
                    prior_low=prior_l,
                )
                last_sweep_i = i
                if i15 >= 0:
                    last_sweep_15m_i = i15

        # After sweep: 15m align/ChoCh → immediately watch 1m for ChoCh
        atr_15 = (
            float(atrs_15m[i15])
            if i15 >= 0 and not np.isnan(atrs_15m[i15])
            else 0.0
        )
        if (
            sm.ctx.recent_sweep
            and sm.ctx.state
            in (TradeState.SWEPT, TradeState.IN_POI, TradeState.WAIT_FOR_CHOCH)
            and snap_15m is not None
            and i15 >= 0
            and _premium_discount_gate(sm, cfg, float(closes_15m[i15]))
            and _m15_mss_gate(
                sm,
                cfg,
                snap_15m=snap_15m,
                i15=i15,
                sweep_15m_i=last_sweep_15m_i,
                open_=float(opens_15m[i15]),
                close=float(closes_15m[i15]),
                atr=atr_15,
            )
        ):
            want = int(sm.ctx.bias)
            poi = sd_15m.zone_touched(
                float(highs_15m[i15]),
                float(lows_15m[i15]),
                direction=want,
            )
            if poi is None:
                zones = sd_15m.active_zones(direction=want)
                poi = zones[-1] if zones else None
            was_waiting = sm.ctx.state == TradeState.WAIT_FOR_CHOCH
            if sm.on_15m_align(snap_15m.bias, snap_15m.last_choch_dir, poi):
                # First align after sweep: arm FVG from ChoCh already printed
                # in the impulse (don't wait for the next future 1m ChoCh).
                if (
                    open_pos is None
                    and not was_waiting
                    and sm.ctx.pending is None
                ):
                    _lookback_arm_choch(
                        df_1m,
                        eng_1m,
                        now_i=i,
                        sweep_i=last_sweep_i,
                        m15_choch_i=last_15m_choch_i,
                        cfg=cfg,
                        sm=sm,
                    )

        # Pending limit management
        if open_pos is None and sm.ctx.state == TradeState.ENTRY_PENDING and sm.ctx.pending:
            pend = sm.ctx.pending
            # Timeout
            if i - pend.placed_idx >= pend.timeout_bars:
                sm.cancel_pending()
            else:
                # Invalidation: opposing structure body break
                invalidated = False
                if pend.direction == "long" and price < pend.invalidation_level:
                    invalidated = True
                elif pend.direction == "short" and price > pend.invalidation_level:
                    invalidated = True
                if invalidated:
                    sm.cancel_pending()
                else:
                    filled = False
                    if pend.direction == "long" and low <= pend.limit_price <= high:
                        filled = True
                    elif pend.direction == "short" and low <= pend.limit_price <= high:
                        filled = True
                    if filled and atr_val > 0:
                        hour_ny = int(
                            _ts(now).tz_convert("America/New_York").hour
                        )
                        if not ny_session_ok(cfg, hour_ny):
                            sm.cancel_pending()
                        else:
                            allowed, _ = rm.can_trade(current_ts)
                            if allowed:
                                if pend.direction == "short" and cfg.long_only:
                                    sm.cancel_pending()
                                else:
                                    tp1, tp2 = _exit_targets(
                                        pend.limit_price,
                                        pend.stop_loss,
                                        pend.direction,
                                        cfg,
                                        liq=liq,
                                        asof_ts=now,
                                        dealing_high=sm.ctx.dealing_range_high,
                                        dealing_low=sm.ctx.dealing_range_low,
                                    )
                                    pos = rm.open_position(
                                        entry_price=pend.limit_price,
                                        atr=atr_val,
                                        direction=pend.direction,
                                        entry_time=current_ts,
                                        entry_idx=i,
                                        stop_loss=pend.stop_loss,
                                        tp1=tp1,
                                        tp2=tp2,
                                    )
                                    if pos is not None:
                                        tags = _setup_tags_at_fill(
                                            sm,
                                            direction=pend.direction,
                                            entry=pend.limit_price,
                                            sl=pend.stop_loss,
                                            now=now,
                                        )
                                        tags["tp_mode"] = getattr(
                                            cfg, "tp_mode", "rr"
                                        )
                                        tags["sl_mode"] = getattr(
                                            cfg, "sl_mode", "ob"
                                        )
                                        tags["tp1"] = tp1
                                        tags["tp2"] = tp2
                                        pos.meta = tags
                                        open_pos = pos
                                        sm.on_fill()
                                        signal_log.append(
                                            {
                                                "ts": str(now),
                                                "direction": pend.direction,
                                                "entry": pend.limit_price,
                                                "sl": pend.stop_loss,
                                                "tp1": tp1,
                                                "tp2": tp2,
                                                **tags,
                                            }
                                        )

        # Arm / refine on nested 1m ChoCh (vs pullback bias).
        # Short refine window: tighter FVG right after displacement, then lock
        # so later micro-breaks cannot steal the pullback limit.
        if open_pos is None and sm.ctx.state in (
            TradeState.WAIT_FOR_CHOCH,
            TradeState.ENTRY_PENDING,
        ):
            want = int(sm.ctx.bias)
            choch_dir = _nested_1m_choch_dir(eng_1m, df_1m, i, want, cfg)
            if choch_dir != 0:
                can_arm = True
                if (
                    sm.ctx.state == TradeState.ENTRY_PENDING
                    and sm.ctx.pending is not None
                ):
                    refine = max(0, int(cfg.choch_refine_bars))
                    age = i - int(sm.ctx.pending.placed_idx)
                    if age > refine:
                        can_arm = False
                    else:
                        old = float(sm.ctx.pending.limit_price)
                        ez = _entry_zone_1m(df_1m, i, want, cfg)
                        if ez is None:
                            can_arm = False
                        else:
                            direction = "long" if want > 0 else "short"
                            new_lim = _limit_price(ez, direction, cfg)
                            tighter = (want < 0 and new_lim < old - 1e-12) or (
                                want > 0 and new_lim > old + 1e-12
                            )
                            if tighter:
                                sm.cancel_pending()
                            else:
                                can_arm = False
                if can_arm and sm.ctx.state == TradeState.WAIT_FOR_CHOCH:
                    _arm_pending_from_choch(
                        df_1m, eng_1m, i, i, now, cfg, sm
                    )

        if verbose and i % 5000 == 0 and i > 0:
            print(
                f"  [{100 * i / n:5.1f}%] i={i} state={sm.ctx.state.value} "
                f"trades={len(trades)} equity={rm.current_equity:.2f}"
            )

        # Equity for metrics is trade-based; sparse curve is enough for plots
        if open_pos is not None or i % 60 == 0:
            equity_curve.append(rm.current_equity)

    if open_pos is not None and open_pos.status != "closed":
        last = df_1m.iloc[-1]
        last_ts = float(_ts(last.name).timestamp())
        rm.close_position(open_pos, float(last["close"]), "backtest_end", last_ts)
        trades.append(_pos_to_dict(open_pos))

    metrics = _compute_metrics(trades, equity_curve, cfg)  # type: ignore[arg-type]
    metrics["setups"] = len(signal_log)
    metrics["final_state"] = sm.ctx.state.value

    return {
        "trades": trades,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "signals": signal_log,
    }
