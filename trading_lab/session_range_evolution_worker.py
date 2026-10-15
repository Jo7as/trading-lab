"""Evolution worker for the Session-Range-Fade (Gold, Asia range / London fade).

Searches range_min/max_atr_mult, sl_buffer_atr_mult, trend_day_atr_mult and the
signal timeframe (M5 vs M15). Loop and persistence live in ``mr_evolution_base``.
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
from .strategies.session_range_fade import SessionRangeFadeConfig, run_backtest

_TIMEFRAMES = ("5min", "15min")


class SessionRangeEvolutionWorker(MrEvolutionBase):
    """Session-Range-Fade evolution on XAU/USD 5m/15m bid+ask bars."""

    MODE = "session_range_evolution"
    VERSION_PREFIX = "srf"
    BASELINE_FILE = "session_range_baseline.yaml"
    DEFAULT_ASSET = "XAU/USD"
    BASE_TIMEFRAME = "5min"
    TIMEFRAMES = _TIMEFRAMES
    config_cls = SessionRangeFadeConfig

    def _backtest(self, df: pd.DataFrame, cfg: SessionRangeFadeConfig) -> dict[str, Any]:
        return run_backtest(df, cfg, verbose=False)

    def _valid(self, cfg: SessionRangeFadeConfig) -> bool:
        return cfg.range_min_atr_mult < cfg.range_max_atr_mult

    def _extra_mutation(
        self, cfg: SessionRangeFadeConfig
    ) -> tuple[SessionRangeFadeConfig, str, Any, Any] | None:
        if random.random() >= 0.15:
            return None
        new = random.choice([tf for tf in _TIMEFRAMES if tf != cfg.timeframe])
        return replace(cfg, timeframe=new), "timeframe", cfg.timeframe, new

    def _random_seed(self, base: SessionRangeFadeConfig) -> SessionRangeFadeConfig:
        return replace(
            base,
            timeframe=random.choice(_TIMEFRAMES),
            range_min_atr_mult=random.choice([0.75, 1.0, 1.5, 2.0, 2.5]),
            range_max_atr_mult=random.choice([3.0, 4.0, 5.0, 6.0]),
            sl_buffer_atr_mult=random.choice([0.0, 0.2, 0.3, 0.5, 1.0]),
            trend_day_atr_mult=random.choice([0.5, 1.0, 1.5, 2.0, 3.0]),
        )

    def _describe(self, cfg: SessionRangeFadeConfig) -> str:
        return (
            f"tf={cfg.timeframe} range=[{cfg.range_min_atr_mult}, {cfg.range_max_atr_mult}]"
            f"×ATR({cfg.atr_timeframe}) sl_buffer={cfg.sl_buffer_atr_mult} "
            f"trend_day={cfg.trend_day_atr_mult}"
        )


def main() -> None:
    state = env("LAB_STATE")
    worker = SessionRangeEvolutionWorker(
        state_dir=Path(state) if state else None,
        year_start=env_int("SRF_EVOLUTION_YEAR_START", 2020),
        num_candidates=env_int("SRF_EVOLUTION_CANDIDATES", 12),
        forward_eval_days=env_int("SRF_EVOLUTION_FORWARD_DAYS", 365),
        verbose=True,
    )
    print("Session-Range-Fade Evolution Worker")
    asyncio.run(worker.run_forever())


if __name__ == "__main__":
    main()
