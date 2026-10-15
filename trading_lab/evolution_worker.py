"""Evolution Worker — runs iterative backtest with Hill-Climb on Railway.

Loads 2 years of historical Gold data, plays it chronologically candle-by-candle,
and after every 15 closed trades runs a Hill-Climb reflection:
  - Tests 8 candidate configs on the last 25 trades
  - Keeps only changes that improve the score
  - Writes improved strategy.yaml to state
  - Continues with the new config

Runs in a loop: when the full 2-year backtest completes, it starts over
with the evolved strategy, allowing continuous improvement.

The goal is to evolve the strategy until it meets:
  - 6% return per 30 days
  - 8% max drawdown
  - 0.8 Sharpe ratio

Usage (in Railway):
  uv run python -m trading_lab.evolution_worker

Or locally:
  uv run python -m trading_lab.evolution_worker --period 730d --verbose
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import MCTFBConfig, load_config_from_yaml, config_to_dict, config_to_yaml
from .adapters.yfinance_adapter import YFinanceAdapter
from .indicators import compute_all_indicators
from .signals import generate_entry_signals
from .risk import RiskManager, Position
from .score import score
from .backtest import _manage_position, _pos_to_dict, _compute_metrics, print_metrics
from .reflect import load_yaml, load_trades, save_prior, append_hypothesis, write_strategy, next_version, STATE_DIR, HYPOTHESES_PATH, TRADES_PATH, HISTORY_DIR


class EvolutionWorker:
    """Runs iterative backtest with Hill-Climb reflection, continuously."""

    def __init__(
        self,
        state_dir: Path | None = None,
        period: str = "730d",
        interval: str = "1h",
        reflect_interval: int = 15,
        eval_window: int = 25,
        num_candidates: int = 20,
        verbose: bool = True,
        ):
        self.state_dir = state_dir or STATE_DIR
        self.period = period
        self.interval = interval
        self.reflect_interval = reflect_interval
        self.eval_window = eval_window
        self.num_candidates = num_candidates
        self.verbose = verbose
        self.adapter = YFinanceAdapter()

        # Load goal
        self.goal = load_yaml(self.state_dir / "goal.yaml")

        # Evolution stats
        self.evolution_round = 0
        self.best_score_ever = -2.0
        self.best_config_ever: MCTFBConfig | None = None

    async def run_forever(self) -> None:
        """Main loop: each round is a FULL chronological 2-year backtest.

        Round 1: start with v01, play all 13.740 candles, reflect after every 15 trades.
        Round 2: start with the evolved strategy from round 1, play all candles again.
        Each round is a complete chronological pass — NOT restarting after 15 trades.
        """
        while True:
            self.evolution_round += 1
            print(f"\n{'='*60}")
            print(f"EVOLUTION ROUND #{self.evolution_round}")
            print(f"{'='*60}")

            # Load current strategy (inherits improvements from previous round)
            strategy_path = self.state_dir / "strategy.yaml"
            cfg = load_config_from_yaml(strategy_path) if strategy_path.exists() else MCTFBConfig()

            print(f"  asset:       {cfg.asset}")
            print(f"  strategy:    v{self._get_version()}")
            print(f"  config:      sl_atr_mult={cfg.sl_atr_mult}, tp1_r={cfg.tp1_r}, "
                  f"squeeze_threshold={cfg.squeeze_threshold}, "
                  f"supertrend_mult={cfg.supertrend_mult}, "
                  f"volume_mult={cfg.volume_mult}")
            print(f"  period:      {self.period}")
            print(f"  reflect:     every {self.reflect_interval} trades")
            print(f"  candidates:  {self.num_candidates} per reflection")

            # Load historical data
            print(f"\n  Loading historical data...")
            df = await self.adapter.fetch_history(cfg.asset, interval=self.interval, period=self.period)
            if df.empty:
                print("  ERROR: No data loaded. Waiting 60s and retrying.")
                await asyncio.sleep(60)
                continue

            print(f"  Loaded {len(df)} candles ({df.index[0]} to {df.index[-1]})")

            # Run one full chronological backtest (all candles, reflect every 15 trades)
            result = self._run_iteration(df, cfg)

            # Print results
            print_metrics(result["metrics"])
            print(f"  Reflections:     {result['reflection_count']}")
            print(f"  Evolution round: {self.evolution_round}")

            # Check if goal is reached
            goal_reached = self._check_goal(result["metrics"])
            if goal_reached:
                print(f"\n  *** GOAL REACHED in round {self.evolution_round}! ***")
                print(f"  Strategy v{self._get_version()} meets all targets.")
                print(f"  Continuing to next round for further optimization.")

            # The _run_iteration returns final_config as a dict, but we need
            # the actual MCTFBConfig object. The problem is that _run_iteration
            # evolves current_cfg internally but only returns a dict.
            # Fix: return current_cfg directly from _run_iteration.
            # For now, we trust the strategy.yaml that was saved during
            # reflections — it contains the latest evolved config.
            # Reload it to get the evolved config for next round.
            evolved_cfg = load_config_from_yaml(strategy_path) if strategy_path.exists() else MCTFBConfig()

            # Save trades
            self._save_trades(result["trades"])

            print(f"\n  Round #{self.evolution_round} complete. "
                  f"Strategy evolved to v{self._get_version()}. "
                  f"Config: sl_atr_mult={evolved_cfg.sl_atr_mult}, "
                  f"tp1_r={evolved_cfg.tp1_r}, "
                  f"squeeze_threshold={evolved_cfg.squeeze_threshold}, "
                  f"supertrend_mult={evolved_cfg.supertrend_mult}, "
                  f"volume_mult={evolved_cfg.volume_mult}")
            print(f"  Starting next round with evolved config...")

    def _run_iteration(self, df: pd.DataFrame, cfg: MCTFBConfig) -> dict[str, Any]:
        """Run one full iterative backtest with Hill-Climb reflection."""

        # Compute indicators + signals
        df_ind = compute_all_indicators(df, cfg)
        df_ind = generate_entry_signals(df_ind, cfg)

        # Find warmup boundary
        warmup_end = 0
        for i in range(len(df_ind)):
            if not np.isnan(df_ind.iloc[i]["atr"]) and not np.isnan(df_ind.iloc[i]["ema_slow"]):
                warmup_end = i
                break

        # Initialize
        rm = RiskManager(cfg, account_size=10000.0)
        open_pos: Position | None = None
        trades: list[dict] = []
        equity_curve: list[float] = [rm.current_equity]
        trades_since_reflection = 0
        reflection_count = 0

        current_cfg = cfg
        current_df = df_ind

        if self.verbose:
            print(f"\n  Starting backtest: {len(current_df)} candles, warmup={warmup_end}")
            print(f"  Initial config: ema_fast={current_cfg.ema_fast}, "
                  f"sl_atr_mult={current_cfg.sl_atr_mult}, tp1_r={current_cfg.tp1_r}, "
                  f"squeeze_threshold={current_cfg.squeeze_threshold}")
            print()

        for i in range(warmup_end, len(current_df)):
            row = current_df.iloc[i]
            current_ts = float(row.name.timestamp()) if hasattr(row.name, "timestamp") else float(i)
            current_price = float(row["close"])
            current_high = float(row["high"])
            current_low = float(row["low"])

            if self.verbose and i % 2000 == 0 and i > warmup_end:
                pct = i / len(current_df) * 100
                print(f"    [{pct:5.1f}%] candle {i}/{len(current_df)} | "
                      f"trades={len(trades)} | equity={rm.current_equity:.2f}")

            # Manage open position
            if open_pos is not None:
                _manage_position(
                    open_pos, rm, row, i, current_ts, current_price,
                    current_high, current_low, current_cfg, trades
                )
                if open_pos.status == "closed":
                    open_pos = None
                    trades_since_reflection += 1

            # Check for reflection
            if trades_since_reflection >= self.reflect_interval and open_pos is None:
                reflection_count += 1
                new_cfg, new_df = self._hill_climb_reflect(
                    trades, current_cfg, df, i, reflection_count
                )
                if new_cfg is not current_cfg:
                    current_cfg = new_cfg
                if new_df is not None:
                    current_df = new_df
                trades_since_reflection = 0

            # Check for new entry
            if open_pos is None:
                allowed, _ = rm.can_trade(current_ts)
                if allowed:
                    atr_val = float(row["atr"]) if not np.isnan(row.get("atr", 0)) else 0.0
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
                        elif row.get("short_signal", 0) > 0:
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

        # Close remaining position
        if open_pos is not None:
            last_row = current_df.iloc[-1]
            last_price = float(last_row["close"])
            last_ts = float(last_row.name.timestamp()) if hasattr(last_row.name, "timestamp") else float(len(current_df))
            rm.close_position(open_pos, last_price, "backtest_end", last_ts)
            trades.append(_pos_to_dict(open_pos))

        # Final metrics
        metrics = _compute_metrics(trades, equity_curve, current_cfg)
        metrics["reflections"] = reflection_count
        metrics["final_config"] = config_to_dict(current_cfg)

        return {
            "trades": trades,
            "metrics": metrics,
            "equity_curve": equity_curve,
            "reflection_count": reflection_count,
            "final_config": config_to_dict(current_cfg),
        }

    def _hill_climb_reflect(
        self,
        trades: list[dict],
        cfg: MCTFBConfig,
        original_df: pd.DataFrame,
        current_candle: int,
        reflection_count: int,
    ) -> tuple[MCTFBConfig, pd.DataFrame | None]:
        """Simulated Annealing: test K candidates, accept improvements always,
        accept worse solutions with probability exp(-delta/temperature).

        Temperature decreases over rounds, so early rounds explore freely
        and later rounds converge (exploit).
        """

        import random
        import math
        from .backtest import backtest as _bt

        closed = [t for t in trades if t.get("status") == "closed"]
        if len(closed) < 5:
            if self.verbose:
                print(f"\n    [reflect #{reflection_count}] only {len(closed)} trades — skipping\n")
            return cfg, None

        recent = closed[-self.eval_window:]
        baseline_score = score(recent, self.goal)

        if self.verbose:
            print(f"\n    [reflect #{reflection_count}] baseline_score={baseline_score:.4f} "
                  f"on last {len(recent)} trades")

        # Extract eval window from original_df
        eval_start = max(0, current_candle - 500)
        eval_df = original_df.iloc[eval_start:current_candle + 1].copy()

        # Temperature schedule with reheating every 300 rounds.
        cycle = self.evolution_round % 300
        if cycle < 250:
            # Cooling phase: 1.0 -> ~0.05
            temperature = max(0.05, 1.0 * (0.997 ** cycle))
        else:
            # Reheat phase: ramp from 0.05 up to 0.5
            reheat_progress = (cycle - 250) / 50.0
            temperature = 0.05 + reheat_progress * 0.45  # 0.05 -> 0.50

        if self.verbose:
            print(f"    [reflect #{reflection_count}] temperature={temperature:.4f} "
                  f"(round {self.evolution_round})")

        # Generate K candidates
        search_space = MCTFBConfig.search_space()
        candidates = []
        seen_keys = set()

        for _k in range(self.num_candidates):
            var_name, lo, hi, step = random.choice(search_space)
            old_val = getattr(cfg, var_name)
            direction = random.choice([-1, 1])
            new_val = round(float(old_val) + direction * step, 4)
            new_val = max(lo, min(hi, new_val))
            if isinstance(old_val, int) or var_name in ("ema_fast", "max_hold_hours"):
                new_val = int(new_val)

            key = f"{var_name}={new_val}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if new_val == old_val:
                continue

            cand_cfg = replace(cfg, **{var_name: new_val})
            candidates.append((cand_cfg, var_name, old_val, new_val))

        if not candidates:
            if self.verbose:
                print(f"    [reflect #{reflection_count}] no valid candidates\n")
            return cfg, None

        # Backtest each candidate and pick the best
        best_score = baseline_score
        best_cfg = cfg
        best_var = None
        best_old = None
        best_new = None
        best_is_improvement = False

        # Also track a "wildcard" candidate for annealing (random worse solution)
        wildcard_cfg = None
        wildcard_var = None
        wildcard_old = None
        wildcard_new = None
        wildcard_score = None

        for cand_cfg, var_name, old_val, new_val in candidates:
            try:
                result = _bt(eval_df, cand_cfg, verbose=False)
                cand_trades = result["trades"]
                cand_closed = [t for t in cand_trades if t.get("status") == "closed"]
                if len(cand_closed) < 3:
                    continue
                cand_recent = cand_closed[-self.eval_window:]
                cand_score = score(cand_recent, self.goal)
            except Exception:
                continue

            # Track best improvement
            if cand_score > best_score:
                best_score = cand_score
                best_cfg = cand_cfg
                best_var = var_name
                best_old = old_val
                best_new = new_val
                best_is_improvement = True

            # Track a random non-improving candidate for annealing
            if cand_score <= baseline_score and wildcard_cfg is None:
                wildcard_cfg = cand_cfg
                wildcard_var = var_name
                wildcard_old = old_val
                wildcard_new = new_val
                wildcard_score = cand_score

        # Decision: accept improvement, or annealing-accept a worse solution
        if best_is_improvement:
            # Always accept improvements
            new_df = compute_all_indicators(original_df, best_cfg)
            new_df = generate_entry_signals(new_df, best_cfg)
            if self.verbose:
                improvement = best_score - baseline_score
                print(f"    [reflect #{reflection_count}] KEEP {best_var}: {best_old} -> {best_new} "
                      f"| score {baseline_score:.4f} -> {best_score:.4f} (+{improvement:.4f}) [IMPROVE]")

            self._log_hypothesis(reflection_count, baseline_score, best_score,
                                 best_var, best_old, best_new, len(recent))
            self._save_strategy(best_cfg, increment_version=True)
            return best_cfg, new_df

        elif wildcard_cfg is not None and temperature > 0.01:
            # Simulated Annealing: accept worse solution with probability p = exp(-delta/T)
            delta = wildcard_score - baseline_score  # negative (worse)
            p_accept = math.exp(delta / temperature) if temperature > 0 else 0

            if random.random() < p_accept:
                # Accept the worse solution to escape local maximum
                new_df = compute_all_indicators(original_df, wildcard_cfg)
                new_df = generate_entry_signals(new_df, wildcard_cfg)
                if self.verbose:
                    print(f"    [reflect #{reflection_count}] ANNEAL {wildcard_var}: {wildcard_old} -> {wildcard_new} "
                          f"| score {baseline_score:.4f} -> {wildcard_score:.4f} ({delta:+.4f}) "
                          f"[ANNEAL p={p_accept:.3f} T={temperature:.4f}]")

                self._log_hypothesis(reflection_count, baseline_score, wildcard_score,
                                     wildcard_var, wildcard_old, wildcard_new, len(recent))
                self._save_strategy(wildcard_cfg, increment_version=True)
                return wildcard_cfg, new_df
            else:
                if self.verbose:
                    print(f"    [reflect #{reflection_count}] no improvement, annealing rejected "
                          f"(p={p_accept:.3f}) — keeping config\n")
                return cfg, None

        else:
            if self.verbose:
                print(f"    [reflect #{reflection_count}] no improvement — keeping config\n")
            return cfg, None

    def _log_hypothesis(self, count: int, old_score: float, new_score: float,
                        var: str, old_val, new_val, trades_n: int) -> None:
        """Log reflection result to hypotheses.jsonl and save prior."""
        strategy_path = self.state_dir / "strategy.yaml"
        strategy = load_yaml(strategy_path) if strategy_path.exists() else {}
        prior_path = save_prior(strategy)
        old_version = strategy.get("version", "01")
        new_version = next_version(old_version)

        hyp = {
            "ts": time.time(),
            "mode": "hill_climb_evolution",
            "round": self.evolution_round,
            "reflection": count,
            "version": f"{old_version} -> {new_version}",
            "baseline_score": round(old_score, 4),
            "new_score": round(new_score, 4),
            "improvement": round(new_score - old_score, 4),
            "trades_examined": trades_n,
            "variable_changed": var,
            "old_value": old_val,
            "new_value": new_val,
        }
        append_hypothesis(hyp)

    def _check_goal(self, metrics: dict) -> bool:
        """Check if the strategy meets all goals."""
        target = self.goal.get("target_return_30d", 0.06) * 100
        max_dd = self.goal.get("max_drawdown", 0.08) * 100
        min_sharpe = self.goal.get("min_sharpe", 0.8)

        # Estimate 30d return from total (rough: scale by 30/total_days)
        # Assuming 730 days of data
        total_days = 730
        total_pnl = metrics.get("total_pnl_pct", 0)
        est_30d_return = total_pnl / (total_days / 30) if total_days > 0 else 0
        actual_dd = abs(metrics.get("max_drawdown_pct", 0))
        actual_sharpe = metrics.get("sharpe", 0)

        print(f"\n  Goal check:")
        print(f"    Return/30d:  {est_30d_return:.2f}% (target: {target:.0f}%) {'PASS' if est_30d_return >= target else 'FAIL'}")
        print(f"    Max DD:      {actual_dd:.2f}% (limit: {max_dd:.0f}%) {'PASS' if actual_dd <= max_dd else 'FAIL'}")
        print(f"    Sharpe:      {actual_sharpe:.3f} (min: {min_sharpe}) {'PASS' if actual_sharpe >= min_sharpe else 'FAIL'}")

        return (est_30d_return >= target and
                actual_dd <= max_dd and
                actual_sharpe >= min_sharpe)

    def _get_version(self) -> str:
        strategy_path = self.state_dir / "strategy.yaml"
        if strategy_path.exists():
            raw = yaml.safe_load(strategy_path.read_text())
            return raw.get("version", "01")
        return "01"

    def _save_strategy(self, cfg: MCTFBConfig, increment_version: bool = False) -> None:
        """Save config to strategy.yaml. If increment_version, bump version number."""
        strategy_path = self.state_dir / "strategy.yaml"
        raw = yaml.safe_load(strategy_path.read_text()) if strategy_path.exists() else {}
        old_version = raw.get("version", "01")
        if increment_version:
            new_version = next_version(old_version)
        else:
            new_version = old_version
        new_dict = config_to_dict(cfg)
        new_dict["version"] = new_version
        with open(strategy_path, "w") as f:
            yaml.dump(new_dict, f, default_flow_style=False, sort_keys=False)
        if increment_version and self.verbose:
            print(f"    [strategy] saved v{new_version} to strategy.yaml")

    def _save_trades(self, trades: list[dict]) -> None:
        trades_path = self.state_dir / "trades.jsonl"
        with open(trades_path, "w") as f:
            for t in trades:
                f.write(json.dumps(t, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolution Worker — iterative backtest with Hill-Climb")
    parser.add_argument("--period", default="730d", help="History period (default 730d)")
    parser.add_argument("--interval", default="1h", help="Candle interval (default 1h)")
    parser.add_argument("--reflect-interval", type=int, default=15, help="Trades between reflections")
    parser.add_argument("--candidates", type=int, default=8, help="Candidates per reflection")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    worker = EvolutionWorker(
        period=args.period,
        interval=args.interval,
        reflect_interval=args.reflect_interval,
        num_candidates=args.candidates,
        verbose=args.verbose,
    )

    print("MCTFB Evolution Worker")
    print(f"  period:     {args.period}")
    print(f"  interval:   {args.interval}")
    print(f"  reflect:    every {args.reflect_interval} trades")
    print(f"  candidates: {args.candidates} per reflection")
    print()

    asyncio.run(worker.run_forever())


if __name__ == "__main__":
    main()
