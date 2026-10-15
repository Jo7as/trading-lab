"""Evolution Worker for Supertrend+Volume intraday strategy.

Runs iterative backtest with Hill-Climb + Simulated Annealing on Railway.
Loads 2 years of Gold (GC=F) 1h candles, plays chronologically,
reflects after every 15 trades, evolves strategy parameters.

The strategy is in trading_lab.strategies.supertrend_volume.
"""
from __future__ import annotations
from .env import env

import asyncio
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .adapters.market_data import get_market_adapter
from .config import load_config_from_yaml
from .score import score, trades_per_week
from .strategies.supertrend_volume import (
    SupertrendVolumeConfig,
    generate_signals as st_generate_signals,
    manage_position as st_manage_position,
    run_backtest as st_run_backtest,
    compute_indicators as st_compute_indicators,
)
from .risk import RiskManager, Position
from .backtest import _compute_metrics, _pos_to_dict, print_metrics
from .reflect import (
    load_yaml, save_prior, append_hypothesis, write_strategy, next_version,
    prune_state_volume,
    STATE_DIR, HYPOTHESES_PATH, TRADES_PATH, HISTORY_DIR,
)

from .exploration import RESEARCH_SEEDS, random_config

# Annealing: reject clearly worse configs; cap acceptance probability
_MAX_ANNEAL_DROP = 0.05
_MAX_ANNEAL_P = 0.25
_MIN_SHORT_TRADES_OOS = 30


def _loc_in_df(df: pd.DataFrame, ts: Any) -> int:
    """Integer row position for a timestamp/index label in df."""
    loc = df.index.get_loc(ts)
    if isinstance(loc, slice):
        return loc.start if loc.start is not None else 0
    if isinstance(loc, np.ndarray):
        if loc.dtype == bool:
            return int(np.flatnonzero(loc)[-1])
        return int(loc[0])
    return int(loc)


def _forward_eval_slice(df: pd.DataFrame, current_ts: Any, bars: int, min_bars: int = 50) -> pd.DataFrame | None:
    """Slice forward window from raw OHLCV using candle timestamp, not dropna index."""
    pos = _loc_in_df(df, current_ts)
    start = pos + 1
    end = min(len(df), start + bars)
    if end - start < min_bars:
        return None
    return df.iloc[start:end].copy()


class STEvolutionWorker:
    """Evolution worker for Supertrend+Volume strategy."""

    def __init__(
        self,
        state_dir: Path | None = None,
        period: str = "730d",
        interval: str = "1h",
        reflect_interval: int = 15,
        eval_window: int = 25,
        num_candidates: int = 20,
        oos_split: float = 0.30,
        forward_eval_bars: int = 200,
        explore_every: int = 20,
        explore_holdout_split: float = 0.30,
        verbose: bool = True,
    ):
        self.state_dir = state_dir or STATE_DIR
        self.period = period
        self.interval = interval
        self.reflect_interval = reflect_interval
        self.eval_window = eval_window
        self.num_candidates = num_candidates
        self.oos_split = oos_split
        self.forward_eval_bars = forward_eval_bars
        self.explore_every = explore_every
        self.explore_holdout_split = explore_holdout_split
        self.verbose = verbose
        self.adapter = get_market_adapter()
        self.goal = load_yaml(self.state_dir / "goal.yaml")
        self.evolution_round = 0
        self.stagnation_rounds = 0
        self._last_version: str | None = None
        self._reflect_pause_remaining = self._load_reflect_pause()

    @staticmethod
    def _config_snapshot(cfg: SupertrendVolumeConfig) -> tuple[tuple[str, Any], ...]:
        """Hashable config fingerprint (ignores YAML version field)."""
        return tuple(sorted(asdict(cfg).items()))

    def _goal_asset(self) -> tuple[str, str]:
        asset = str(self.goal.get("asset", "GC=F"))
        label = str(self.goal.get("asset_label", asset))
        return asset, label

    def _pin_goal_asset(self, cfg: SupertrendVolumeConfig) -> SupertrendVolumeConfig:
        asset, label = self._goal_asset()
        if cfg.asset == asset and cfg.asset_label == label:
            return cfg
        return replace(cfg, asset=asset, asset_label=label)

    def _enforce_goal_asset(self, strategy_path: Path) -> tuple[SupertrendVolumeConfig, bool]:
        """Restore baseline when strategy.yaml asset drifts from goal.yaml."""
        goal_asset, _ = self._goal_asset()
        if not strategy_path.exists():
            baseline_path = self._baseline_path()
            if baseline_path.exists():
                shutil.copy2(baseline_path, strategy_path)
            return self._pin_goal_asset(self._load_strategy(strategy_path)), True

        raw = yaml.safe_load(strategy_path.read_text()) or {}
        file_asset = str(raw.get("asset", ""))
        if file_asset == goal_asset:
            return self._pin_goal_asset(self._load_strategy(strategy_path)), False

        print(
            f"  [asset] DRIFT detected: {file_asset!r} != {goal_asset!r} "
            f"— restoring {self._baseline_path().name}"
        )
        if strategy_path.exists():
            save_prior(raw)
        baseline_path = self._baseline_path()
        shutil.copy2(baseline_path, strategy_path)
        baseline = self._pin_goal_asset(self._load_strategy(baseline_path))
        self._save_strategy(baseline, increment=False)
        self.stagnation_rounds = 0
        self._clear_invalid_best_oos()
        return baseline, True

    def _clear_invalid_best_oos(self) -> None:
        """Drop best_oos snapshot when it belongs to the wrong asset."""
        goal_asset, _ = self._goal_asset()
        meta_path = self.state_dir / "best_oos_meta.json"
        strat_path = self.state_dir / "best_oos.yaml"
        if not strat_path.exists():
            return
        try:
            raw = yaml.safe_load(strat_path.read_text()) or {}
        except Exception:
            raw = {}
        if str(raw.get("asset", "")) == goal_asset:
            return
        strat_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        print(f"  [asset] cleared invalid best_oos (was {raw.get('asset', '?')!r})")

    async def run_forever(self) -> None:
        """Main loop: each round is a FULL chronological 2-year backtest."""
        pruned = prune_state_volume()
        if any(pruned.values()):
            print(
                f"  [volume] pruned history={pruned.get('history_removed', 0)} "
                f"hypotheses_lines={pruned.get('hypotheses_lines_removed', 0)} "
                f"trades_lines={pruned.get('trades_lines_removed', 0)}"
            )

        while True:
            self.evolution_round += 1
            self._reflect_pause_remaining = self._load_reflect_pause()
            force_hf = env("LAB_FORCE_HF", "").strip() in ("1", "true", "yes")
            self._force_hf = force_hf
            reset_applied = self._apply_pending_reset()
            baseline_restored = self._enforce_hf_baseline(required=force_hf)
            print(f"\n{'='*60}")
            print(f"ST+VOL EVOLUTION ROUND #{self.evolution_round}")
            print(f"{'='*60}")

            # Load current strategy (restore baseline if asset drifted from goal.yaml)
            strategy_path = self.state_dir / "strategy.yaml"
            cfg, asset_restored = self._enforce_goal_asset(strategy_path)
            round_start_snapshot = self._config_snapshot(cfg)
            ver_before = yaml.safe_load(strategy_path.read_text()).get("version", "?") if strategy_path.exists() else "?"

            print(f"  account:     ${cfg.account_size:,.0f} fixed (no compounding)")
            print(f"  asset:       {cfg.asset}")
            print(f"  config:      st_period={cfg.supertrend_period}, "
                  f"st_mult={cfg.supertrend_mult}, "
                  f"vol_mult={cfg.volume_mult}, "
                  f"sl_atr={cfg.sl_atr_mult}, tp1_r={cfg.tp1_r}, "
                  f"tp1_close={cfg.tp1_close_pct}, "
                  f"max_hold={cfg.max_hold_hours}")
            print(f"  period:      {self.period}")
            print(f"  reflect:     every {self.reflect_interval} trades")
            print(f"  candidates:  {self.num_candidates} per reflection")
            print(f"  OOS holdout: {self.oos_split:.0%} of data (never used for evolution)")
            print(f"  exploration: every {self.explore_every} stagnant rounds")
            reflect_frozen = force_hf or reset_applied or baseline_restored or asset_restored or self._reflect_pause_remaining > 0
            if force_hf:
                print(f"  [hf] FORCE mode — HF baseline locked, no evolution")
            elif self._reflect_pause_remaining > 0:
                print(f"  [reflect] PAUSED — {self._reflect_pause_remaining} round(s) remaining")
            if reflect_frozen:
                print(f"  [reflect] evolution FROZEN this round")

            # Load data
            data_source = env("LAB_DATA_SOURCE", "yfinance")
            evolution_start = env("EVOLUTION_START", "").strip()
            print(f"\n  Loading historical data ({data_source})...")
            if evolution_start:
                df = await self.adapter.fetch_history(
                    cfg.asset,
                    interval=self.interval,
                    period=self.period,
                    start=datetime.fromisoformat(evolution_start),
                )
            else:
                df = await self.adapter.fetch_history(
                    cfg.asset, interval=self.interval, period=self.period
                )
            if df.empty:
                print("  ERROR: No data. Waiting 60s.")
                await asyncio.sleep(60)
                continue

            print(f"  Loaded {len(df)} candles ({df.index[0]} to {df.index[-1]})")

            # Train / OOS split — evolution only sees in-sample data
            split_idx = int(len(df) * (1 - self.oos_split))
            df_train = df.iloc[:split_idx].copy()
            df_oos = df.iloc[split_idx:].copy()
            print(f"  Train:       {len(df_train)} candles (in-sample)")
            print(f"  OOS:         {len(df_oos)} candles ({df_oos.index[0]} → {df_oos.index[-1]})")

            # Exploration jump when stuck at local optimum (between-round stagnation)
            if not reflect_frozen and self.stagnation_rounds >= self.explore_every:
                cfg = self._exploration_jump(df_train, cfg, ver_before)

            # Run chronological backtest with reflection on train only
            result = self._run_iteration(df_train, cfg, freeze_reflection=reflect_frozen)

            # Print in-sample results
            print_metrics(result["metrics"], label="IN-SAMPLE (train)")
            self._print_trades_per_week(
                result["trades"], label="In-sample", period_days=len(df_train) / 24
            )
            print(f"  Reflections:     {result['reflection_count']}")
            print(f"  Evolution round: {self.evolution_round}")

            self._check_goal(result["metrics"], label="In-sample", period_days=len(df_train) / 24)

            # Out-of-sample validation with frozen evolved config
            evolved_cfg = self._load_strategy(strategy_path)
            oos_result = st_run_backtest(df_oos, evolved_cfg, verbose=False)
            print_metrics(oos_result["metrics"], label="OUT-OF-SAMPLE (holdout)")
            self._print_trades_per_week(
                oos_result["trades"], label="OOS", period_days=len(df_oos) / 24
            )
            self._check_goal(oos_result["metrics"], label="OOS", period_days=len(df_oos) / 24)
            if not force_hf:
                evolved_cfg = self._enforce_short_gate(evolved_cfg, oos_result["trades"])
            elif self.verbose and not evolved_cfg.long_only:
                print(f"  [short-gate] SKIPPED — HF force lock (long+short)")
            result["oos_metrics"] = oos_result["metrics"]

            # Save in-sample trades only
            self._save_trades(result["trades"])

            print(f"\n  Round #{self.evolution_round} complete. "
                  f"Config: st_mult={evolved_cfg.supertrend_mult}, "
                  f"vol_mult={evolved_cfg.volume_mult}, "
                  f"sl_atr={evolved_cfg.sl_atr_mult}, tp1_r={evolved_cfg.tp1_r}, "
                  f"long_only={evolved_cfg.long_only}")
            ver_after = yaml.safe_load(strategy_path.read_text()).get("version", "?") if strategy_path.exists() else "?"

            if strategy_path.exists():
                from .candidate_leaderboard import update_best_oos_tracker

                strat_raw = yaml.safe_load(strategy_path.read_text()) or {}
                goal_asset, _ = self._goal_asset()
                if str(strat_raw.get("asset", "")) != goal_asset:
                    print(
                        f"  [best-oos] skipped — wrong asset "
                        f"{strat_raw.get('asset', '?')!r} (goal {goal_asset!r})"
                    )
                elif update_best_oos_tracker(
                    self.state_dir,
                    strat_raw,
                    float(oos_result["metrics"]["total_pnl_pct"]),
                    evolution_round=self.evolution_round,
                ):
                    print(
                        f"  [best-oos] new record: "
                        f"{oos_result['metrics']['total_pnl_pct']:.2f}% OOS (v{ver_after})"
                    )

            round_end_snapshot = self._config_snapshot(evolved_cfg)
            config_changed_this_round = round_end_snapshot != round_start_snapshot

            if config_changed_this_round:
                self.stagnation_rounds = 0
                if self.verbose:
                    print(f"  Config updated v{ver_before} -> v{ver_after} (params changed)")
            else:
                self.stagnation_rounds += 1
                if self.verbose:
                    print(f"  Stagnation: {self.stagnation_rounds} rounds without "
                          f"net config change (v{ver_after})")
            if reflect_frozen and self._reflect_pause_remaining > 0:
                if result["reflection_count"] == 0:
                    self._reflect_pause_remaining -= 1
                    self._save_reflect_pause(self._reflect_pause_remaining)
                    print(f"  [reflect] pause: {self._reflect_pause_remaining} round(s) remaining")
                else:
                    print(
                        f"  [reflect] WARN pause active but {result['reflection_count']} "
                        f"reflections ran — not decrementing"
                    )

            await self._maybe_refresh_leaderboard()

            print(f"  Starting next round...")

    def _load_reflect_pause(self) -> int:
        """Rounds left without reflection (volume file or LAB_NO_REFLECT_ROUNDS env)."""
        path = self.state_dir / "reflect_pause.yaml"
        if path.exists():
            try:
                return max(0, int(load_yaml(path).get("remaining", 0)))
            except (TypeError, ValueError):
                pass
        return max(0, int(env("LAB_NO_REFLECT_ROUNDS", "0")))

    def _save_reflect_pause(self, remaining: int) -> None:
        path = self.state_dir / "reflect_pause.yaml"
        with open(path, "w") as f:
            yaml.dump({"remaining": remaining}, f, default_flow_style=False)

    def _baseline_path(self) -> Path:
        env = env("LAB_BASELINE_PATH", "").strip()
        if env:
            p = Path(env)
            if p.exists():
                return p
        for candidate in (
            self.state_dir / "jpy_baseline.yaml",
            self.state_dir / "hf_baseline.yaml",
            Path("/app/jpy_baseline.yaml"),
            Path("/app/hf_baseline.yaml"),
        ):
            if candidate.exists():
                return candidate
        return self.state_dir / "hf_baseline.yaml"

    def _enforce_hf_baseline(self, *, required: bool = False) -> bool:
        """Re-apply baseline yaml when paused or LAB_FORCE_HF is active."""
        if not required and self._reflect_pause_remaining <= 0:
            return False
        baseline_path = self._baseline_path()
        if not baseline_path.exists():
            if required:
                print(f"  [hf] WARN baseline missing: {baseline_path}")
            return False
        strategy_path = self.state_dir / "strategy.yaml"
        current = self._load_strategy(strategy_path)
        baseline = self._load_strategy(baseline_path)
        if self._config_snapshot(current) == self._config_snapshot(baseline):
            return False
        if strategy_path.exists():
            save_prior(yaml.safe_load(strategy_path.read_text()))
        shutil.copy2(baseline_path, strategy_path)
        side = "long+short" if not baseline.long_only else "long-only"
        print(
            f"  [hf] Restored {baseline_path.name} ({side}, "
            f"st_mult={baseline.supertrend_mult}, tp1_r={baseline.tp1_r})"
        )
        return True

    def _enforce_hf_baseline_during_pause(self) -> bool:
        """Backward-compatible alias."""
        return self._enforce_hf_baseline(required=False)

    def _apply_pending_reset(self) -> bool:
        """Apply pending_hf_reset.yaml at round boundary (survives running worker)."""
        reset_path = self.state_dir / "pending_hf_reset.yaml"
        if not reset_path.exists():
            return False
        strategy_path = self.state_dir / "strategy.yaml"
        try:
            if strategy_path.exists():
                save_prior(yaml.safe_load(strategy_path.read_text()))
            shutil.copy2(reset_path, strategy_path)
            reset_path.unlink(missing_ok=True)
        except Exception as exc:
            print(f"  [reset] ERROR applying pending_hf_reset.yaml: {exc}")
            return False
        new_ver = yaml.safe_load(strategy_path.read_text()).get("version", "?")
        self.stagnation_rounds = 0
        self._reflect_pause_remaining = max(self._reflect_pause_remaining, 3)
        self._save_reflect_pause(self._reflect_pause_remaining)
        print(f"  [reset] Applied pending_hf_reset.yaml -> strategy.yaml (v{new_ver})")
        print(f"  [reset] Reflection frozen this round ({self._reflect_pause_remaining} pause rounds)")
        return True

    def _load_strategy(self, path: Path) -> SupertrendVolumeConfig:
        """Load strategy from YAML, falling back to defaults."""
        if not path.exists():
            return SupertrendVolumeConfig()
        raw = yaml.safe_load(path.read_text())
        # Filter to only valid SupertrendVolumeConfig fields
        import dataclasses
        valid = {f.name for f in dataclasses.fields(SupertrendVolumeConfig)}
        kwargs = {k: v for k, v in raw.items() if k in valid}
        try:
            return self._pin_goal_asset(SupertrendVolumeConfig(**kwargs))
        except Exception:
            return self._pin_goal_asset(SupertrendVolumeConfig())

    def _save_strategy(self, cfg: SupertrendVolumeConfig, increment: bool = False) -> None:
        """Save config to strategy.yaml."""
        path = self.state_dir / "strategy.yaml"
        raw = yaml.safe_load(path.read_text()) if path.exists() else {}
        old_ver = raw.get("version", "01")
        if increment and raw:
            save_prior(raw)
        new_ver = next_version(old_ver) if increment else old_ver
        d = asdict(self._pin_goal_asset(cfg))
        d["version"] = new_ver
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)
        if increment and self.verbose:
            print(f"    [strategy] saved v{new_ver}")

    def _run_iteration(
        self,
        df: pd.DataFrame,
        cfg: SupertrendVolumeConfig,
        *,
        freeze_reflection: bool = False,
    ) -> dict[str, Any]:
        """Run full chronological backtest with Hill-Climb reflection."""
        cfg = self._pin_goal_asset(cfg)

        # Compute indicators + signals
        df_ind = st_generate_signals(df, cfg)

        # Drop warmup
        df_ind = df_ind.dropna(subset=["atr", "ema_slow_val", "supertrend", "st_dir"]).copy()
        warmup_end = 0

        # Init — fixed account sizing, no compounding
        rm = RiskManager(cfg, account_size=cfg.account_size, backtest_mode=True)  # type: ignore
        open_pos: Position | None = None
        trades: list[dict] = []
        equity_curve: list[float] = [rm.current_equity]
        trades_since_reflection = 0
        reflection_count = 0

        current_cfg = cfg
        current_df = df_ind

        if self.verbose:
            print(f"\n  Starting backtest: {len(current_df)} candles")
            print()

        for i in range(len(current_df)):
            row = current_df.iloc[i]
            current_ts = float(row.name.timestamp()) if hasattr(row.name, "timestamp") else float(i)
            current_price = float(row["close"])
            current_high = float(row["high"])
            current_low = float(row["low"])

            if self.verbose and i % 2000 == 0 and i > 0:
                pct = i / len(current_df) * 100
                print(f"    [{pct:5.1f}%] candle {i}/{len(current_df)} | "
                      f"trades={len(trades)} | equity={rm.current_equity:.2f}")

            # Manage open position
            if open_pos is not None:
                st_manage_position(
                    open_pos, row, i, current_ts, current_price,
                    current_high, current_low, current_cfg, rm, trades,
                )
                if open_pos.status == "closed":
                    open_pos = None
                    trades_since_reflection += 1

            # Reflection
            if (
                not freeze_reflection
                and trades_since_reflection >= self.reflect_interval
                and open_pos is None
            ):
                reflection_count += 1
                new_cfg, new_df = self._hill_climb_reflect(
                    trades, current_cfg, df, row.name, reflection_count
                )
                if new_cfg is not current_cfg:
                    current_cfg = new_cfg
                if new_df is not None:
                    current_df = new_df
                    try:
                        i = _loc_in_df(current_df, row.name)
                    except KeyError:
                        pass
                trades_since_reflection = 0

            # New entry
            if open_pos is None:
                allowed, _ = rm.can_trade(current_ts)
                if allowed:
                    atr_val = float(row.get("atr", 0))
                    if atr_val > 0:
                        if row.get("long_signal", 0) > 0:
                            pos = rm.open_position(
                                entry_price=current_price, atr=atr_val,
                                direction="long", entry_time=current_ts, entry_idx=i,
                            )
                            if pos:
                                open_pos = pos
                                if self.verbose:
                                    print(f"    [trade] OPEN long @ {current_price:.2f} "
                                          f"SL={pos.stop_loss:.2f} TP1={pos.tp1:.2f}")
                        elif row.get("short_signal", 0) > 0 and not current_cfg.long_only:
                            pos = rm.open_position(
                                entry_price=current_price, atr=atr_val,
                                direction="short", entry_time=current_ts, entry_idx=i,
                            )
                            if pos:
                                open_pos = pos
                                if self.verbose:
                                    print(f"    [trade] OPEN short @ {current_price:.2f} "
                                          f"SL={pos.stop_loss:.2f} TP1={pos.tp1:.2f}")

            equity_curve.append(rm.current_equity)

        # Close remaining
        if open_pos is not None and open_pos.status != "closed":
            last_row = current_df.iloc[-1]
            last_price = float(last_row["close"])
            last_ts = float(last_row.name.timestamp()) if hasattr(last_row.name, "timestamp") else float(len(current_df))
            rm.close_position(open_pos, last_price, "backtest_end", last_ts)
            trades.append(_pos_to_dict(open_pos))

        metrics = _compute_metrics(trades, equity_curve, current_cfg)
        metrics["reflections"] = reflection_count

        return {
            "trades": trades,
            "metrics": metrics,
            "equity_curve": equity_curve,
            "reflection_count": reflection_count,
            "final_config": asdict(current_cfg),
        }

    def _score_on_df(self, df: pd.DataFrame, cfg: SupertrendVolumeConfig) -> tuple[float, int]:
        """Run backtest on df and return (score, n_closed_trades)."""
        result = st_run_backtest(df, cfg, verbose=False)
        closed = [t for t in result["trades"] if t.get("status") == "closed"]
        if len(closed) < 2:
            return -1.0, len(closed)
        period_days = len(df) / 24  # 1h candles
        return score(closed, self.goal, period_days=period_days), len(closed)

    def _exploration_jump(
        self,
        df_train: pd.DataFrame,
        cfg: SupertrendVolumeConfig,
        ver_before: str,
    ) -> SupertrendVolumeConfig:
        """Try research seeds + random configs when evolution stagnates."""
        if self.verbose:
            print(f"\n  [explore] stagnation={self.stagnation_rounds} — "
                  f"testing research seeds + random configs")

        explore_idx = int(len(df_train) * (1 - self.explore_holdout_split))
        df_explore = df_train.iloc[explore_idx:].copy()
        if self.verbose:
            print(f"  [explore] scoring on holdout: {len(df_explore)} candles "
                  f"({df_explore.index[0]} → {df_explore.index[-1]})")

        baseline_score, baseline_n = self._score_on_df(df_explore, cfg)
        best_score = baseline_score
        best_cfg = cfg
        best_label = "current"

        candidates: list[tuple[str, SupertrendVolumeConfig]] = list(RESEARCH_SEEDS)
        for i in range(5):
            candidates.append((f"random_{i}", random_config()))

        for label, cand in candidates:
            cand = self._pin_goal_asset(cand)
            try:
                cand_score, cand_n = self._score_on_df(df_explore, cand)
            except Exception:
                continue
            if cand_n < 5:
                continue
            if cand_score > best_score:
                best_score = cand_score
                best_cfg = cand
                best_label = label

        if best_cfg is not cfg:
            best_cfg = self._maybe_strip_shorts(df_explore, best_cfg, best_label)
            if self.verbose:
                print(f"  [explore] ADOPT {best_label} | score {baseline_score:.4f} -> "
                      f"{best_score:.4f} (+{best_score - baseline_score:.4f}) | "
                      f"st_mult={best_cfg.supertrend_mult} ema={best_cfg.ema_fast}/"
                      f"{best_cfg.ema_slow} long_only={best_cfg.long_only}")
            self._log_hypothesis(
                0, baseline_score, best_score,
                f"explore:{best_label}", ver_before, asdict(best_cfg), baseline_n,
            )
            self._save_strategy(best_cfg, increment=True)
            return best_cfg

        if self.verbose:
            print(f"  [explore] no seed beat current (baseline={baseline_score:.4f})")
        return cfg

    def _print_trades_per_week(
        self, trades: list[dict], label: str = "", period_days: float | None = None
    ) -> None:
        closed = [t for t in trades if t.get("status") == "closed"]
        if not closed:
            return
        tpw = trades_per_week(closed, period_days)
        target = self.goal.get("target_trades_per_week", 5)
        min_tpw = self.goal.get("min_trades_per_week", 3)
        status = "OK" if tpw >= min_tpw else "LOW"
        prefix = f"  Trades/week ({label}):" if label else "  Trades/week:"
        print(f"{prefix} {tpw:.1f} (target {min_tpw}–{target}) [{status}]")

    def _short_side_stats(self, trades: list[dict]) -> tuple[int, float]:
        shorts = [
            t for t in trades
            if t.get("status") == "closed" and t.get("direction") == "short"
        ]
        pnl = sum(t.get("pnl_pct", 0.0) or 0.0 for t in shorts)
        return len(shorts), pnl

    def _shorts_allowed_oos(self, trades: list[dict]) -> tuple[bool, str]:
        n_short, short_pnl = self._short_side_stats(trades)
        if n_short < _MIN_SHORT_TRADES_OOS:
            return False, f"only {n_short} short trades (need {_MIN_SHORT_TRADES_OOS})"
        if short_pnl <= 0:
            return False, f"short PnL {short_pnl:+.2f}% <= 0"
        return True, f"{n_short} shorts, PnL {short_pnl:+.2f}%"

    def _enforce_short_gate(
        self, cfg: SupertrendVolumeConfig, oos_trades: list[dict]
    ) -> SupertrendVolumeConfig:
        """Keep shorts only when OOS short leg is profitable with enough samples."""
        if cfg.long_only:
            return cfg
        allowed, reason = self._shorts_allowed_oos(oos_trades)
        if allowed:
            if self.verbose:
                print(f"  [short-gate] PASS — {reason}")
            return cfg
        gated = replace(cfg, long_only=True)
        if self.verbose:
            print(f"  [short-gate] FAIL ({reason}) — forcing long_only=True")
        self._save_strategy(gated, increment=True)
        return gated

    def _maybe_strip_shorts(
        self,
        df: pd.DataFrame,
        cfg: SupertrendVolumeConfig,
        label: str,
    ) -> SupertrendVolumeConfig:
        """On exploration adopt: drop shorts if holdout short leg loses."""
        if cfg.long_only:
            return cfg
        result = st_run_backtest(df, cfg, verbose=False)
        n_short, short_pnl = self._short_side_stats(result["trades"])
        if n_short == 0:
            return cfg
        if short_pnl > 0:
            return cfg
        if self.verbose:
            print(f"  [explore] {label}: short holdout PnL {short_pnl:+.2f}% "
                  f"({n_short} trades) — adopting long-only variant")
        return replace(cfg, long_only=True)

    def _hill_climb_reflect(
        self,
        trades: list[dict],
        cfg: SupertrendVolumeConfig,
        original_df: pd.DataFrame,
        current_ts: Any,
        reflection_count: int,
    ) -> tuple[SupertrendVolumeConfig, pd.DataFrame | None]:
        """Hill-climb with simulated annealing."""

        closed = [t for t in trades if t.get("status") == "closed"]
        if len(closed) < 5:
            return cfg, None

        # Temperature — boost when stagnating to escape local optima
        cycle = self.evolution_round % 100
        temperature = max(0.08, 0.6 * (0.98 ** cycle))
        if self.stagnation_rounds >= 5:
            temperature = max(temperature, 0.25)

        # Forward window aligned by timestamp (not dropna row index)
        eval_df = _forward_eval_slice(original_df, current_ts, self.forward_eval_bars)
        if eval_df is None:
            if self.verbose:
                print(f"    [reflect #{reflection_count}] skip — insufficient forward bars "
                      f"after {current_ts}")
            return cfg, None

        # Baseline on SAME forward window (apples-to-apples — was broken before)
        baseline_score, baseline_n = self._score_on_df(eval_df, cfg)
        if baseline_n < 2:
            return cfg, None

        if self.verbose:
            print(f"\n    [reflect #{reflection_count}] forward baseline={baseline_score:.4f} "
                  f"temp={temperature:.4f} bars={len(eval_df)} stagnation={self.stagnation_rounds}")

        # Generate K candidates
        search_space = SupertrendVolumeConfig.search_space()
        candidates = []
        seen_keys = set()

        for _k in range(self.num_candidates):
            var_name, lo, hi, step = random.choice(search_space)
            old_val = getattr(cfg, var_name)
            direction = random.choice([-1, 1])
            new_val = round(float(old_val) + direction * step, 4)
            new_val = max(lo, min(hi, new_val))
            if isinstance(old_val, int) or var_name in (
                "supertrend_period", "volume_ma_period", "ema_fast", "ema_slow",
                "max_hold_hours", "cooldown_hours",
            ):
                new_val = int(new_val)

            key = f"{var_name}={new_val}"
            if key in seen_keys or new_val == old_val:
                continue
            seen_keys.add(key)

            cand_cfg = replace(cfg, **{var_name: new_val})
            candidates.append((cand_cfg, var_name, old_val, new_val))

        if not candidates:
            return cfg, None

        # Backtest each candidate
        best_score = baseline_score
        best_cfg = cfg
        best_var = None
        best_old = None
        best_new = None
        wildcards: list[tuple] = []

        for cand_cfg, var_name, old_val, new_val in candidates:
            try:
                cand_score, cand_n = self._score_on_df(eval_df, cand_cfg)
                if cand_n < 2:
                    continue
            except Exception:
                continue

            if cand_score > best_score:
                best_score = cand_score
                best_cfg = cand_cfg
                best_var = var_name
                best_old = old_val
                best_new = new_val

            if cand_score <= baseline_score:
                wildcards.append((cand_cfg, var_name, old_val, new_val, cand_score))

        # Decision
        if best_cfg is not cfg:
            new_df = st_generate_signals(original_df, best_cfg)
            new_df = new_df.dropna(subset=["atr", "ema_slow_val", "supertrend", "st_dir"]).copy()
            if self.verbose:
                improvement = best_score - baseline_score
                print(f"    [reflect #{reflection_count}] KEEP {best_var}: {best_old} -> {best_new} "
                      f"| score {baseline_score:.4f} -> {best_score:.4f} (+{improvement:.4f})")
            self._log_hypothesis(reflection_count, baseline_score, best_score,
                                 best_var, best_old, best_new, baseline_n)
            self._save_strategy(best_cfg, increment=True)
            return best_cfg, new_df

        elif wildcards and temperature > 0.03:
            wildcard_cfg, wildcard_var, wildcard_old, wildcard_new, wildcard_score = random.choice(wildcards)
            delta = wildcard_score - baseline_score
            if delta >= -_MAX_ANNEAL_DROP:
                p_accept = min(_MAX_ANNEAL_P, math.exp(delta / temperature)) if temperature > 0 else 0.0
                if random.random() < p_accept:
                    new_df = st_generate_signals(original_df, wildcard_cfg)
                    new_df = new_df.dropna(subset=["atr", "ema_slow_val", "supertrend", "st_dir"]).copy()
                    if self.verbose:
                        print(f"    [reflect #{reflection_count}] ANNEAL {wildcard_var}: {wildcard_old} -> {wildcard_new} "
                              f"| score {baseline_score:.4f} -> {wildcard_score:.4f} ({delta:+.4f}) "
                              f"[ANNEAL p={p_accept:.3f}]")
                    self._log_hypothesis(reflection_count, baseline_score, wildcard_score,
                                         wildcard_var, wildcard_old, wildcard_new, baseline_n)
                    self._save_strategy(wildcard_cfg, increment=True)
                    return wildcard_cfg, new_df

        if self.verbose:
            print(f"    [reflect #{reflection_count}] no improvement — keeping config")
        return cfg, None

    def _log_hypothesis(self, count: int, old_score: float, new_score: float,
                        var: str, old_val, new_val, trades_n: int) -> None:
        path = self.state_dir / "strategy.yaml"
        strategy = load_yaml(path) if path.exists() else {}
        old_ver = strategy.get("version", "01")
        new_ver = next_version(old_ver)
        hyp = {
            "ts": time.time(),
            "mode": "st_vol_evolution",
            "round": self.evolution_round,
            "reflection": count,
            "version": f"{old_ver} -> {new_ver}",
            "baseline_score": round(old_score, 4),
            "new_score": round(new_score, 4),
            "improvement": round(new_score - old_score, 4),
            "trades_examined": trades_n,
            "variable_changed": var,
            "old_value": old_val,
            "new_value": new_val,
        }
        append_hypothesis(hyp)

    def _check_goal(
        self, metrics: dict, label: str = "", period_days: float = 730
    ) -> bool:
        target = self.goal.get("target_return_30d", 0.06) * 100
        max_dd = self.goal.get("max_drawdown", 0.08) * 100
        min_sharpe = self.goal.get("min_sharpe", 0.8)
        total_pnl = metrics.get("total_pnl_pct", 0)
        est_30d = total_pnl / (period_days / 30) if period_days > 0 else 0
        actual_dd = abs(metrics.get("max_drawdown_pct", 0))
        actual_sharpe = metrics.get("sharpe", 0)

        prefix = f"  Goal check ({label}):" if label else "  Goal check:"
        print(f"\n{prefix}")
        print(f"    Return/30d:  {est_30d:.2f}% (target: {target:.0f}%) {'PASS' if est_30d >= target else 'FAIL'}")
        print(f"    Max DD:      {actual_dd:.2f}% (limit: {max_dd:.0f}%) {'PASS' if actual_dd <= max_dd else 'FAIL'}")
        print(f"    Sharpe:      {actual_sharpe:.3f} (min: {min_sharpe}) {'PASS' if actual_sharpe >= min_sharpe else 'FAIL'}")

        return est_30d >= target and actual_dd <= max_dd and actual_sharpe >= min_sharpe

    async def _maybe_refresh_leaderboard(self) -> None:
        """Refresh candidate leaderboard JSON every 5 evolution rounds."""
        if self.evolution_round % 5 != 0:
            return
        if env("LAB_LEADERBOARD_REFRESH", "1").strip().lower() in (
            "0",
            "false",
            "no",
        ):
            return
        strategy_path = self.state_dir / "strategy.yaml"
        asset = "GC=F"
        if strategy_path.exists():
            raw = yaml.safe_load(strategy_path.read_text()) or {}
            asset = str(raw.get("asset", "GC=F"))
        try:
            from trading_lab.candidate_leaderboard import (
                dashboard_profile_for_asset,
                refresh_dashboard,
            )

            profile = dashboard_profile_for_asset(asset, self.state_dir)
            if profile is None:
                return
            data_source = env("LAB_DATA_SOURCE", "dukascopy")
            start = env("EVOLUTION_START", "2020-01-01")
            await refresh_dashboard(profile, data_source=data_source, start=start)
            print(
                f"  [dashboard] {profile.title} leaderboard refreshed "
                f"(round {self.evolution_round})"
            )
        except Exception as exc:
            print(f"  [dashboard] refresh failed: {exc}")

    def _save_trades(self, trades: list[dict]) -> None:
        path = self.state_dir / "trades.jsonl"
        with open(path, "w") as f:
            for t in trades:
                f.write(json.dumps(t, default=str) + "\n")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="ST+Vol Evolution Worker")
    parser.add_argument("--period", default="730d")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--reflect-interval", type=int, default=15)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    worker = STEvolutionWorker(
        period=args.period,
        interval=args.interval,
        reflect_interval=args.reflect_interval,
        num_candidates=args.candidates,
        verbose=args.verbose,
    )

    print("Supertrend+Volume Evolution Worker")
    print(f"  period:     {args.period}")
    print(f"  interval:   {args.interval}")
    print(f"  reflect:    every {args.reflect_interval} trades")
    print(f"  candidates: {args.candidates}")
    print()

    asyncio.run(worker.run_forever())


if __name__ == "__main__":
    main()
