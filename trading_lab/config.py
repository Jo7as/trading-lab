"""MCTFB Strategy Configuration — all adjustable parameters in one place.

Every variable the reflection loop can optimize has explicit bounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MCTFBConfig:
    # ── Asset ──────────────────────────────────────────────────────
    asset: str = "GC=F"           # Yahoo Finance ticker (GC=F = Gold futures)
    asset_label: str = "XAU/USD"  # Human-readable

    # ── Trend filter ───────────────────────────────────────────────
    ema_fast: int = 21
    ema_slow: int = 200

    # ── Bollinger Bands ────────────────────────────────────────────
    bb_period: int = 20
    bb_std: float = 2.0
    squeeze_lookback: int = 100
    squeeze_threshold: float = 0.20  # bottom 20th percentile

    # ── Volume ─────────────────────────────────────────────────────
    volume_ma_period: int = 20
    volume_mult: float = 1.3

    # ── Candle body filter ─────────────────────────────────────────
    min_body_ratio: float = 0.50

    # ── ATR / Supertrend ───────────────────────────────────────────
    atr_period: int = 14
    supertrend_period: int = 10
    supertrend_mult: float = 3.0

    # ── Risk management ────────────────────────────────────────────
    risk_per_trade: float = 0.01    # 1% of equity
    sl_atr_mult: float = 1.0       # stop loss = 1.0 × ATR (tight)
    tp1_r: float = 2.0             # first TP at 2.0R (reward > risk)
    tp1_close_pct: float = 0.75    # close 75% at TP1 (less trail risk)
    max_hold_hours: int = 72
    max_daily_loss: float = 0.03   # 3% daily loss limit
    cooldown_hours: int = 4        # cooldown after a loss
    max_concurrent_positions: int = 1

    # ── Reflection ─────────────────────────────────────────────────
    reflection_every: int = 25

    # ── Searchable parameter space for reflection loop ─────────────
    # (key, min, max, step) — used by reflect.py
    @staticmethod
    def search_space() -> list[tuple[str, float, float, float]]:
        return [
            ("bb_std",              1.5, 2.5, 0.2),
            ("squeeze_threshold",   0.10, 0.30, 0.04),
            ("volume_mult",         1.2, 2.5, 0.2),
            ("sl_atr_mult",         0.8, 2.0, 0.2),
            ("tp1_r",               1.5, 3.0, 0.2),
            ("tp1_close_pct",       0.50, 0.90, 0.05),
            ("supertrend_mult",     2.0, 4.0, 0.3),
            ("max_hold_hours",      48, 120, 12),
            ("ema_fast",            9, 50, 3),
            ("ema_slow",            100, 300, 20),
        ]


def load_config_from_yaml(path: Path) -> MCTFBConfig:
    """Load a MCTFBConfig from a strategy.yaml file."""
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f)
    # flatten nested yaml into config fields
    kwargs: dict[str, Any] = {}
    field_names = {f.name for f in __import__("dataclasses").fields(MCTFBConfig)}
    for key, val in data.items():
        if key in field_names:
            kwargs[key] = val
    return MCTFBConfig(**kwargs)


def config_to_dict(cfg: MCTFBConfig) -> dict[str, Any]:
    """Serialize config back to a plain dict for YAML."""
    from dataclasses import asdict
    return asdict(cfg)


def config_to_yaml(cfg: MCTFBConfig) -> str:
    """Serialize config to a YAML string."""
    return yaml.dump(config_to_dict(cfg), default_flow_style=False, sort_keys=False)
