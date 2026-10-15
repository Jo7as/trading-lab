"""Research seeds and random configs for evolution exploration jumps."""
from __future__ import annotations

import random

from .strategies.smc_ict import SmcIctConfig
from .strategies.supertrend_volume import SupertrendVolumeConfig

# Hand-picked from test_st_final.py — outperform evolved v3311 on frequency/edge
RESEARCH_SEEDS: list[tuple[str, SupertrendVolumeConfig]] = [
    (
        "st10_m2_ema21100",
        SupertrendVolumeConfig(
            supertrend_period=10,
            supertrend_mult=2.0,
            volume_mult=1.0,
            ema_fast=21,
            ema_slow=100,
            sl_atr_mult=1.5,
            tp1_r=2.0,
            tp1_close_pct=0.6,
            cooldown_hours=0,
            max_hold_hours=48,
            long_only=True,
        ),
    ),
    (
        "st7_m15_ema21100",
        SupertrendVolumeConfig(
            supertrend_period=7,
            supertrend_mult=1.5,
            volume_mult=1.0,
            ema_fast=21,
            ema_slow=100,
            sl_atr_mult=1.5,
            tp1_r=2.0,
            cooldown_hours=0,
            max_hold_hours=48,
            long_only=True,
        ),
    ),
    (
        "st7_m15_noema",
        SupertrendVolumeConfig(
            supertrend_period=7,
            supertrend_mult=1.5,
            volume_mult=1.0,
            ema_fast=3,
            ema_slow=3,
            min_atr_pct=0.0,
            sl_atr_mult=1.5,
            tp1_r=2.0,
            cooldown_hours=0,
            max_hold_hours=48,
            long_only=True,
        ),
    ),
    (
        "st10_m2_longshort",
        SupertrendVolumeConfig(
            supertrend_mult=2.0,
            volume_mult=1.0,
            ema_fast=21,
            ema_slow=100,
            sl_atr_mult=1.5,
            tp1_r=2.0,
            cooldown_hours=0,
            max_hold_hours=24,
            long_only=False,
        ),
    ),
    # Higher frequency — target ~2–3+ trades/week on 1h GC=F
    (
        "hf_st7_m18_ema2150_long",
        SupertrendVolumeConfig(
            supertrend_period=7,
            supertrend_mult=1.8,
            volume_mult=1.0,
            ema_fast=21,
            ema_slow=50,
            min_atr_pct=0.0,
            sl_atr_mult=1.5,
            tp1_r=2.0,
            tp1_close_pct=0.6,
            cooldown_hours=0,
            max_hold_hours=48,
            pullback_entry=True,
            long_only=True,
        ),
    ),
    (
        "hf_st7_m18_ema2150_longshort",
        SupertrendVolumeConfig(
            supertrend_period=7,
            supertrend_mult=1.8,
            volume_mult=1.0,
            ema_fast=21,
            ema_slow=50,
            min_atr_pct=0.0,
            sl_atr_mult=1.5,
            tp1_r=2.0,
            tp1_close_pct=0.6,
            cooldown_hours=0,
            max_hold_hours=48,
            pullback_entry=True,
            long_only=False,
        ),
    ),
    (
        "hf_st7_m15_noema_long",
        SupertrendVolumeConfig(
            supertrend_period=7,
            supertrend_mult=1.5,
            volume_mult=1.0,
            ema_fast=3,
            ema_slow=3,
            min_atr_pct=0.0,
            sl_atr_mult=1.5,
            tp1_r=2.0,
            cooldown_hours=0,
            max_hold_hours=48,
            pullback_entry=True,
            long_only=True,
        ),
    ),
    (
        "hf_st7_m15_noema_longshort",
        SupertrendVolumeConfig(
            supertrend_period=7,
            supertrend_mult=1.5,
            volume_mult=1.0,
            ema_fast=3,
            ema_slow=3,
            min_atr_pct=0.0,
            sl_atr_mult=1.5,
            tp1_r=2.0,
            cooldown_hours=0,
            max_hold_hours=48,
            pullback_entry=True,
            long_only=False,
        ),
    ),
]

# JPY research seeds — USD/JPY 1h (Dukascopy 2020+)
JPY_RESEARCH_SEEDS: list[tuple[str, SupertrendVolumeConfig]] = [
    (
        "jpy_ls01_long",
        SupertrendVolumeConfig(
            supertrend_period=7,
            supertrend_mult=1.5,
            volume_mult=1.0,
            ema_fast=3,
            ema_slow=3,
            min_atr_pct=0.0,
            sl_atr_mult=1.5,
            tp1_r=2.0,
            tp1_close_pct=0.6,
            cooldown_hours=0,
            max_hold_hours=48,
            pullback_entry=False,
            long_only=True,
        ),
    ),
    (
        "jpy_ls01_pullback",
        SupertrendVolumeConfig(
            supertrend_period=7,
            supertrend_mult=1.5,
            volume_mult=1.0,
            ema_fast=3,
            ema_slow=3,
            min_atr_pct=0.0,
            sl_atr_mult=1.5,
            tp1_r=2.0,
            tp1_close_pct=0.6,
            cooldown_hours=0,
            max_hold_hours=48,
            pullback_entry=True,
            long_only=False,
        ),
    ),
    (
        "jpy_st10_m18_ema40_long",
        SupertrendVolumeConfig(
            supertrend_period=10,
            supertrend_mult=1.8,
            volume_mult=1.4,
            ema_fast=3,
            ema_slow=40,
            min_atr_pct=0.0005,
            sl_atr_mult=2.0,
            tp1_r=3.0,
            tp1_close_pct=0.5,
            cooldown_hours=3,
            max_hold_hours=24,
            pullback_entry=True,
            long_only=True,
        ),
    ),
    (
        "jpy_hf_pullback_long",
        SupertrendVolumeConfig(
            supertrend_period=7,
            supertrend_mult=1.5,
            volume_mult=1.0,
            ema_fast=3,
            ema_slow=3,
            min_atr_pct=0.0,
            sl_atr_mult=1.5,
            tp1_r=2.0,
            tp1_close_pct=0.6,
            cooldown_hours=0,
            max_hold_hours=48,
            pullback_entry=True,
            long_only=True,
        ),
    ),
]


def random_config() -> SupertrendVolumeConfig:
    """Sample a random config from the searchable parameter space."""
    kwargs: dict = {}
    for var_name, lo, hi, step in SupertrendVolumeConfig.search_space():
        steps = int((hi - lo) / step)
        # Bias supertrend_mult toward lower values (more signals)
        if var_name == "supertrend_mult":
            idx = random.randint(0, max(0, int(steps * 0.6)))
        else:
            idx = random.randint(0, max(0, steps))
        val = lo + idx * step
        if var_name in (
            "supertrend_period", "volume_ma_period", "ema_fast", "ema_slow",
            "max_hold_hours", "cooldown_hours",
        ):
            val = int(val)
        else:
            val = round(float(val), 4)
        kwargs[var_name] = val
    kwargs["pullback_entry"] = random.random() < 0.75
    kwargs["long_only"] = random.random() < 0.6
    return SupertrendVolumeConfig(**kwargs)


# ---------------------------------------------------------------------------
# SMC / ICT research seeds (EUR/USD nested MTF)
# ---------------------------------------------------------------------------

_SMC_EUR_BASE = dict(
    asset="EUR/USD",
    asset_label="EUR/USD",
    sl_buffer=0.00005,
    sl_mode="ob",
    max_sl_dist=0.0012,
    tp_mode="rr",
    move_sl_to_be_on_tp1=True,
    entry_edge_offset=0.00005,
    require_1m_fvg=True,
    nested_choch_vs_pullback=True,
    use_swing_liq=False,
    swing_length_4h=10,
    ob_lookback=15,
    entry_limit_mode="fvg_edge",
    ob_range="wick",
    entry_timeout_bars=60,
    choch_lookback_bars=90,
    choch_refine_bars=5,
    max_hold_bars=360,
)

SMC_RESEARCH_SEEDS: list[tuple[str, SmcIctConfig]] = [
    (
        "smc_baseline_fvg_edge",
        SmcIctConfig(**_SMC_EUR_BASE),
    ),
    (
        "smc_entry_buffer_1pip",
        SmcIctConfig(**{**_SMC_EUR_BASE, "entry_edge_offset": 0.00010}),
    ),
    (
        "smc_sl_swing",
        SmcIctConfig(**{**_SMC_EUR_BASE, "sl_mode": "swing", "sl_buffer": 0.00010}),
    ),
    (
        "smc_tight_ltf_swings",
        SmcIctConfig(
            **{
                **_SMC_EUR_BASE,
                "swing_length_15m": 3,
                "swing_length_1m": 3,
            }
        ),
    ),
    (
        "smc_asia_london_only",
        SmcIctConfig(
            **{
                **_SMC_EUR_BASE,
                "use_asia": True,
                "use_london": True,
                "use_pd": False,
                "use_pw": False,
            }
        ),
    ),
    (
        "smc_pd_pw_only",
        SmcIctConfig(
            **{
                **_SMC_EUR_BASE,
                "use_asia": False,
                "use_london": False,
                "use_pd": True,
                "use_pw": True,
            }
        ),
    ),
    (
        "smc_wider_rr",
        SmcIctConfig(
            **{
                **_SMC_EUR_BASE,
                "tp1_r": 2.0,
                "tp2_r": 4.0,
                "max_hold_bars": 360,
            }
        ),
    ),
    (
        "smc_quick_tp1",
        SmcIctConfig(
            **{
                **_SMC_EUR_BASE,
                "tp1_r": 1.0,
                "tp1_close_pct": 0.75,
                "tp2_r": 2.5,
            }
        ),
    ),
    (
        "smc_lookback10_body",
        SmcIctConfig(
            **{
                **_SMC_EUR_BASE,
                "ob_lookback": 10,
                "ob_range": "body",
            }
        ),
    ),
]


def random_smc_config(**overrides) -> SmcIctConfig:
    """Sample a random SMC config from numeric + categorical search spaces."""
    kwargs: dict = dict(_SMC_EUR_BASE)
    for var_name, lo, hi, step in SmcIctConfig.search_space():
        steps = int(round((hi - lo) / step))
        idx = random.randint(0, max(0, steps))
        val = lo + idx * step
        if var_name in (
            "swing_length_15m",
            "swing_length_1m",
            "ob_lookback",
            "entry_timeout_bars",
            "choch_lookback_bars",
            "choch_refine_bars",
            "max_hold_bars",
        ):
            val = int(round(val))
        else:
            val = round(float(val), 6)
        kwargs[var_name] = val
    for var_name, choices in SmcIctConfig.categorical_space():
        kwargs[var_name] = random.choice(choices)
    # Keep at least one liquidity source on
    if not any(kwargs.get(k) for k in ("use_asia", "use_london", "use_pd", "use_pw")):
        kwargs["use_pd"] = True
    kwargs.update(overrides)
    return SmcIctConfig.pin_strategy_locks(SmcIctConfig(**kwargs))
