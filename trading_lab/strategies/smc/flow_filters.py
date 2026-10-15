"""Optional ICT flow gates (Stages A–C) — pure, causal helpers.

See SMC_FLOW_SPEC.md. Defaults in SmcIctConfig keep all gates off (baseline flow).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .types import StructureSnapshot, TrendBias

if TYPE_CHECKING:
    pass


class FlowConfig(Protocol):
    flow_a_sweep_close_back: bool
    flow_a_sweep_close_mode: str
    flow_b_premium_discount: bool
    flow_b_range_mode: str
    flow_b_equilibrium: float
    flow_c_require_15m_mss: bool
    flow_c_displacement: bool
    flow_c_displacement_atr_mult: float
    flow_c_align_fallback: bool
    flow_ny_session_only: bool
    flow_killzone: str
    flow_sell_side_sweeps_only: bool


# Attribution winners (TP2 / PnL) — sell-side session + PD lows
_SELL_SIDE_SWEEP_KINDS = frozenset({"asia_low", "london_low", "pdl"})
_NY_SESSIONS = frozenset({"ny_am", "ny_pm"})

# Killzone session sets (NY local clock via session_from_ny_hour)
_KILLZONE_SESSIONS: dict[str, frozenset[str]] = {
    "off": frozenset(),
    "ny": frozenset({"ny_am", "ny_pm"}),
    "ny_am": frozenset({"ny_am"}),
    "london": frozenset({"london"}),
    "london_ny": frozenset({"london", "ny_am", "ny_pm"}),
}


def session_from_ny_hour(hour_ny: int) -> str:
    if 19 <= hour_ny or hour_ny < 2:
        return "asia"
    if 2 <= hour_ny < 5:
        return "london"
    if 7 <= hour_ny < 11:
        return "ny_am"
    if 11 <= hour_ny < 16:
        return "ny_pm"
    return "other"


def sweep_kind_allowed(cfg: FlowConfig, kind: str) -> bool:
    """Attribution filter: only asia_low / london_low / pdl when enabled."""
    if not getattr(cfg, "flow_sell_side_sweeps_only", False):
        return True
    return kind in _SELL_SIDE_SWEEP_KINDS


def _resolved_killzone(cfg: FlowConfig) -> str:
    """Prefer ``flow_killzone``; legacy ``flow_ny_session_only`` → ``ny``."""
    mode = str(getattr(cfg, "flow_killzone", "off") or "off")
    if mode != "off":
        return mode
    if getattr(cfg, "flow_ny_session_only", False):
        return "ny"
    return "off"


def ny_session_ok(cfg: FlowConfig, hour_ny: int) -> bool:
    """Fill only inside configured killzone (NY / NY AM / London / …)."""
    mode = _resolved_killzone(cfg)
    if mode == "off":
        return True
    allowed = _KILLZONE_SESSIONS.get(mode)
    if not allowed:
        return True
    return session_from_ny_hour(hour_ny) in allowed


# Back-compat alias
killzone_ok = ny_session_ok


def is_high_side_liquidity(kind: str) -> bool:
    return kind.endswith("_high") or kind in ("pdh", "pwh", "swing_high")


def sweep_close_back_ok(
    cfg: FlowConfig,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    level: float,
    kind: str,
) -> bool:
    """Stage A: wick through level + close back inside (rejection)."""
    if not cfg.flow_a_sweep_close_back:
        return True
    is_high = is_high_side_liquidity(kind)
    if is_high:
        # Buy-side raid → bearish setup: wick above, close below level
        wick_through = high > level
        if cfg.flow_a_sweep_close_mode == "body":
            return wick_through and close < level
        return wick_through and low < level and close < level
    # Sell-side raid → bullish setup: wick below, close above level
    wick_through = low < level
    if cfg.flow_a_sweep_close_mode == "body":
        return wick_through and close > level
    return wick_through and high > level and close > level


def dealing_range_from_swings(
    snap: StructureSnapshot,
) -> tuple[float, float] | None:
    """15m swing high/low at snapshot (causal)."""
    if snap.last_swing_high is None or snap.last_swing_low is None:
        return None
    lo = float(snap.last_swing_low.price)
    hi = float(snap.last_swing_high.price)
    if hi <= lo:
        return None
    return lo, hi


def dealing_range_sweep_bar(
    *,
    high: float,
    low: float,
    prior_high: float | None = None,
    prior_low: float | None = None,
) -> tuple[float, float]:
    """Minimal range: sweep bar wicks, optionally expanded by prior bar."""
    lo, hi = low, high
    if prior_low is not None:
        lo = min(lo, prior_low)
    if prior_high is not None:
        hi = max(hi, prior_high)
    return lo, hi


def premium_discount_ok(
    cfg: FlowConfig,
    bias: TrendBias,
    price: float,
    range_low: float,
    range_high: float,
) -> bool:
    """Stage B: long in discount, short in premium relative to dealing range."""
    if not cfg.flow_b_premium_discount:
        return True
    if range_high <= range_low:
        return False
    eq = range_low + (range_high - range_low) * float(cfg.flow_b_equilibrium)
    if bias == TrendBias.BULLISH:
        return price <= eq
    if bias == TrendBias.BEARISH:
        return price >= eq
    return False


def displacement_bar_ok(
    cfg: FlowConfig,
    *,
    open_: float,
    close: float,
    atr: float,
) -> bool:
    if not cfg.flow_c_displacement:
        return True
    if atr <= 0:
        return False
    body = abs(close - open_)
    return body >= atr * float(cfg.flow_c_displacement_atr_mult)


def m15_mss_ok(
    cfg: FlowConfig,
    *,
    bias: TrendBias,
    snap: StructureSnapshot,
    choch_bar_i: int,
    sweep_bar_i: int,
    open_: float,
    close: float,
    atr: float,
) -> bool:
    """Stage C: 15m ChoCh in HTF direction after sweep, optional displacement."""
    if bias == TrendBias.NEUTRAL:
        return False
    if not cfg.flow_c_require_15m_mss:
        return True
    want = int(bias)
    if snap.last_choch_dir != want:
        return False
    if choch_bar_i < sweep_bar_i:
        return False
    return displacement_bar_ok(cfg, open_=open_, close=close, atr=atr)


def m15_align_ok_legacy(
    cfg: FlowConfig,
    *,
    bias: TrendBias,
    bias_15m: TrendBias,
    choch_dir: int,
) -> bool:
    """Current production path when flow_c_require_15m_mss is false."""
    if bias == TrendBias.NEUTRAL:
        return False
    aligned = bias_15m == bias
    choch_ok = choch_dir == int(bias) and choch_dir != 0
    if cfg.flow_c_require_15m_mss:
        return False
    if cfg.flow_c_align_fallback:
        return aligned or choch_ok
    return choch_ok
