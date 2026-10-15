"""Evolution worker for the two-tier z-score mean reversion (15m).

Searches z_entry_tier_a/b, adx_block_threshold, sl_atr_mult, max_hold_bars and
the require_reversal_candle / session_filter_enabled flags. One asset per
worker (Gold or USD/JPY) — set it in goal.yaml or via MR_EVOLUTION_ASSET.
Loop and persistence live in ``mr_evolution_base``.
"""
from __future__ import annotations
from .env import env

import asyncio
import os
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from .mr_evolution_base import MrEvolutionBase, env_int
from .strategies.mean_reversion import MeanReversionConfig, run_backtest

_BOOL_FLIP = ("require_reversal_candle", "session_filter_enabled")


class MeanReversionEvolutionWorker(MrEvolutionBase):
    """Z-score mean-reversion evolution on 15m bid+ask bars."""

    MODE = "mr_evolution"
    VERSION_PREFIX = "mr"
    BASELINE_FILE = "mean_reversion_baseline.yaml"
    DEFAULT_ASSET = "GC=F"
    BASE_TIMEFRAME = "15min"
    TIMEFRAMES = ("15min",)
    config_cls = MeanReversionConfig

    def _backtest(self, df: pd.DataFrame, cfg: MeanReversionConfig) -> dict[str, Any]:
        return run_backtest(df, cfg, verbose=False)

    def _valid(self, cfg: MeanReversionConfig) -> bool:
        return cfg.z_entry_tier_b <= cfg.z_entry_tier_a

    def _extra_mutation(
        self, cfg: MeanReversionConfig
    ) -> tuple[MeanReversionConfig, str, Any, Any] | None:
        if random.random() >= 0.2:
            return None
        var = random.choice(_BOOL_FLIP)
        old = getattr(cfg, var)
        return replace(cfg, **{var: not old}), var, old, not old

    def _random_seed(self, base: MeanReversionConfig) -> MeanReversionConfig:
        tier_a = random.choice([1.8, 2.0, 2.2, 2.5, 3.0])
        return replace(
            base,
            z_entry_tier_a=tier_a,
            z_entry_tier_b=random.choice([b for b in (1.2, 1.5, 1.8, 2.0, 2.4) if b <= tier_a]),
            adx_block_threshold=random.choice([22.0, 26.0, 32.0, 40.0, 50.0]),
            sl_atr_mult=random.choice([1.0, 1.5, 2.0, 2.5, 3.0]),
            max_hold_bars=random.choice([8, 16, 24, 32, 48]),
            require_reversal_candle=random.choice([True, False]),
            session_filter_enabled=random.choice([True, False]),
        )

    def _describe(self, cfg: MeanReversionConfig) -> str:
        return (
            f"z=[A {cfg.z_entry_tier_a}, B {cfg.z_entry_tier_b}] adx<{cfg.adx_block_threshold} "
            f"sl={cfg.sl_atr_mult}×ATR hold={cfg.max_hold_bars} "
            f"reversal={cfg.require_reversal_candle} session={cfg.session_filter_enabled}"
        )


def main() -> None:
    state = env("LAB_STATE")
    worker = MeanReversionEvolutionWorker(
        state_dir=Path(state) if state else None,
        year_start=env_int("MR_EVOLUTION_YEAR_START", 2020),
        num_candidates=env_int("MR_EVOLUTION_CANDIDATES", 12),
        forward_eval_days=env_int("MR_EVOLUTION_FORWARD_DAYS", 365),
        asset=env("MR_EVOLUTION_ASSET") or None,
        verbose=True,
    )
    print("Mean-Reversion Evolution Worker")
    asyncio.run(worker.run_forever())


if __name__ == "__main__":
    main()
