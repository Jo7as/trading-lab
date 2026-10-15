"""Evolution worker for the opening range breakout with a fixed R target.

Searches range_minutes, signal_bar_minutes, signal_window_minutes, target_r,
stop_buffer_spreads, max_cost_r and the max-range filter on 1m bid+ask bars.
Session clock and cost assumptions are not evolved: they always come from the
market preset. One market per worker (US100, GER40 or XAUUSD) — set it in
goal.yaml or via ORB2R_EVOLUTION_ASSET. Loop and persistence live in
``mr_evolution_base``.

The strategy reports R; ``risk_per_trade`` from goal.yaml turns that into the
account % the shared score() expects.
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

from .backtest import _compute_metrics
from .minute_data import load_minute_bid_ask
from .mr_evolution_base import Frames, MrEvolutionBase, env_int
from .strategies.orb_2r import MARKETS, OrbConfig, Sessions, build_sessions, run_backtest

_SIGNAL_BARS = (1, 5, 15)
_RANGE_MULTS = (0.0, 2.0, 2.5, 3.0)
_PRESET_FIELDS = ("tz", "calendar", "session_open", "flat_time", "slippage", "slippage_pct", "commission")


class Orb2rEvolutionWorker(MrEvolutionBase):
    """ORB evolution on 1m bid+ask bars, results scored in account %."""

    MODE = "orb2r_evolution"
    VERSION_PREFIX = "orb2r"
    BASELINE_FILE = "orb2r_baseline.yaml"
    DEFAULT_ASSET = "US100"
    BASE_TIMEFRAME = "1min"
    TIMEFRAMES = ("1min",)
    config_cls = OrbConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # train / forward / OOS frames are scored hundreds of times — prepare each once
        self._sessions: dict[int, tuple[pd.DataFrame, Sessions]] = {}

    def _pin(self, cfg: OrbConfig) -> OrbConfig:
        cfg = super()._pin(cfg)
        preset = MARKETS.get(cfg.asset)
        if preset is None:
            return cfg
        return replace(cfg, **{f: getattr(preset, f) for f in _PRESET_FIELDS})

    def _backtest(self, df: pd.DataFrame, cfg: OrbConfig) -> dict[str, Any]:
        cached = self._sessions.get(id(df))
        if cached is None or cached[0] is not df:
            cached = (df, build_sessions(df, cfg))
            self._sessions[id(df)] = cached
        risk_pct = float(self.goal.get("risk_per_trade", 0.005)) * 100.0
        trades = [
            {
                **t,
                "status": "closed",
                "entry_time": pd.Timestamp(t["entry_time"]).timestamp(),
                "exit_time": pd.Timestamp(t["exit_time"]).timestamp(),
                "pnl_pct": round(t["r"] * risk_pct, 4),
                "meta": {"strategy": "orb_2r", "risk_pct": risk_pct},
            }
            for t in run_backtest(cached[1], cfg)["trades"]
        ]
        return {"trades": trades, "metrics": _compute_metrics(trades, [], cfg)}  # type: ignore[arg-type]

    def _valid(self, cfg: OrbConfig) -> bool:
        return cfg.signal_window_minutes >= cfg.signal_bar_minutes

    def _extra_mutation(self, cfg: OrbConfig) -> tuple[OrbConfig, str, Any, Any] | None:
        if random.random() >= 0.25:
            return None
        var, choices = random.choice(
            [("signal_bar_minutes", _SIGNAL_BARS), ("max_range_mult", _RANGE_MULTS)]
        )
        old = getattr(cfg, var)
        new = random.choice([c for c in choices if c != old])
        return replace(cfg, **{var: new}), var, old, new

    def _random_seed(self, base: OrbConfig) -> OrbConfig:
        return replace(
            base,
            range_minutes=random.choice([5, 10, 15, 30, 45, 60]),
            signal_bar_minutes=random.choice(_SIGNAL_BARS),
            signal_window_minutes=random.choice([60, 120, 180, 240]),
            target_r=random.choice([1.0, 1.5, 2.0, 3.0, 4.0]),
            stop_buffer_spreads=random.choice([0.0, 1.0, 2.0]),
            max_cost_r=random.choice([0.02, 0.03, 0.04, 0.06, 0.08]),
            max_range_mult=random.choice(_RANGE_MULTS),
        )

    def _describe(self, cfg: OrbConfig) -> str:
        return (
            f"open={cfg.session_open} {cfg.tz} range={cfg.range_minutes}m "
            f"signal={cfg.signal_bar_minutes}m window={cfg.signal_window_minutes}m "
            f"target={cfg.target_r}R buffer={cfg.stop_buffer_spreads}×spread "
            f"cost≤{cfg.max_cost_r}R range≤{cfg.max_range_mult}×avg"
        )

    async def load_frames(self) -> Frames:
        asset, _ = self._goal_asset()
        print(f"\n  Loading {asset} 1min (bid+ask) from {self.year_start} ...")
        df = await asyncio.to_thread(load_minute_bid_ask, asset, self.year_start)
        print(f"  Loaded {len(df)} bars")
        return {} if df.empty else {self.BASE_TIMEFRAME: df}


def main() -> None:
    state = env("LAB_STATE")
    worker = Orb2rEvolutionWorker(
        state_dir=Path(state) if state else None,
        year_start=env_int("ORB2R_EVOLUTION_YEAR_START", 2016),
        num_candidates=env_int("ORB2R_EVOLUTION_CANDIDATES", 12),
        forward_eval_days=env_int("ORB2R_EVOLUTION_FORWARD_DAYS", 730),
        asset=env("ORB2R_EVOLUTION_ASSET") or None,
        verbose=True,
    )
    print("ORB 2R Evolution Worker")
    asyncio.run(worker.run_forever())


if __name__ == "__main__":
    main()
