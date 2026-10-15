"""Iterative backtest with self-improving reflection loop.

Runs the MCTFB strategy on historical data, and after every N closed trades:
  1. Reflects on the results (score-guided random search)
  2. Updates the strategy parameters
  3. Re-computes indicators with the new parameters
  4. Continues the backtest with the updated strategy

This simulates the real self-improving agent behavior on historical data.

Usage:
  uv run python -m trading_lab.iterative_backtest --asset GC=F --period 730d --reflect-interval 15
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .adapters.yfinance_adapter import YFinanceAdapter, _clean_ohlcv
from .config import MCTFBConfig, load_config_from_yaml, config_to_yaml
from .indicators import compute_all_indicators
from .signals import generate_entry_signals
from .risk import RiskManager, Position
from .score import score
from .reflect import (
    load_yaml, load_trades, save_prior, append_hypothesis,
    write_strategy, next_version, STATE_DIR,
    HYPOTHESES_PATH, TRADES_PATH, HISTORY_DIR,
)
from .backtest import _manage_position, _pos_to_dict, _compute_metrics, print_metrics


def iterative_backtest(
    df: pd.DataFrame,
    cfg: MCTFBConfig,
    reflect_interval: int = 15,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run backtest with reflection after every N closed trades.

    After each reflection, indicators are re-computed with the new
    parameters for the remaining candles. This means the strategy
    genuinely evolves during the backtest.
    """
    state_dir = STATE_DIR

    # Load goal for scoring
    goal = load_yaml(state_dir / "goal.yaml")

    # Initialize
    rm = RiskManager(cfg, account_size=10000.0)
    open_pos: Position | None = None
    trades: list[dict] = []
    equity_curve: list[float] = [rm.current_equity]
    trades_since_reflection = 0
    reflection_count = 0

    # We process the DataFrame in segments: after each reflection,
    # we re-compute indicators from the current point forward with
    # the new config. But we need lookback data for indicators, so
    # we keep a rolling window.

    # Pre-compute indicators for the full dataset with initial config
    # (we'll re-compute segments after each reflection)
    df_ind = compute_all_indicators(df, cfg)
    df_ind = generate_entry_signals(df_ind, cfg)

    # Find the warmup boundary (first non-NaN row)
    warmup_end = 0
    for i in range(len(df_ind)):
        if not np.isnan(df_ind.iloc[i]["atr"]) and not np.isnan(df_ind.iloc[i]["ema_slow"]):
            warmup_end = i
            break

    if verbose:
        print(f"\n[iterative] starting backtest: {len(df_ind)} candles, "
              f"warmup={warmup_end}, reflect every {reflect_interval} trades")
        print(f"[iterative] initial config: ema_fast={cfg.ema_fast}, "
              f"sl_atr_mult={cfg.sl_atr_mult}, tp1_r={cfg.tp1_r}, "
              f"squeeze_threshold={cfg.squeeze_threshold}")
        print()

    current_cfg = cfg
    current_df = df_ind

    for i in range(warmup_end, len(current_df)):
        row = current_df.iloc[i]
        current_ts = float(row.name.timestamp()) if hasattr(row.name, "timestamp") else float(i)
        current_price = float(row["close"])
        current_high = float(row["high"])
        current_low = float(row["low"])

        if verbose and i % 2000 == 0 and i > warmup_end:
            pct = i / len(current_df) * 100
            print(f"  [{pct:5.1f}%] candle {i}/{len(current_df)} | "
                  f"trades={len(trades)} | equity={rm.current_equity:.2f} | "
                  f"v{current_cfg_to_version(current_cfg)}")

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
        if trades_since_reflection >= reflect_interval and open_pos is None:
            reflection_count += 1
            new_cfg, new_df = _do_reflection(
                trades, current_cfg, df, state_dir, goal,
                i, reflection_count, verbose
            )
            if new_cfg is not current_cfg:
                current_cfg = new_cfg
            if new_df is not None:
                current_df = new_df
            trades_since_reflection = 0

        # Check for new entry
        if open_pos is None:
            allowed, reason = rm.can_trade(current_ts)
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
                            if verbose:
                                print(f"  [trade] OPEN long @ {current_price:.2f} "
                                      f"SL={pos.stop_loss:.2f} TP1={pos.tp1:.2f}")
                    elif row.get("short_signal", 0) > 0:
                        pos = rm.open_position(
                            entry_price=current_price, atr=atr_val,
                            direction="short", entry_time=current_ts, entry_idx=i,
                        )
                        if pos:
                            open_pos = pos
                            if verbose:
                                print(f"  [trade] OPEN short @ {current_price:.2f} "
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


def current_cfg_to_version(cfg: MCTFBConfig) -> str:
    """Best-effort version display."""
    return getattr(cfg, "_version", "01")


def config_to_dict(cfg: MCTFBConfig) -> dict:
    from dataclasses import asdict
    return asdict(cfg)


def _do_reflection(
    trades: list[dict],
    cfg: MCTFBConfig,
    original_df: pd.DataFrame,
    state_dir: Path,
    goal: dict,
    current_candle: int,
    reflection_count: int,
    verbose: bool,
    window: int = 25,
) -> tuple[MCTFBConfig, pd.DataFrame]:
    """Hill-climb reflection: test candidate configs, keep only if better.

    1. Score the last `window` closed trades under the current config (baseline)
    2. Generate K=5 candidate configs, each nudging ONE random variable
    3. Re-backtest each candidate on the same `window` trades
    4. Keep the candidate with the best score IF it beats baseline
    5. Otherwise keep the current config (no change)
    """
    import random
    from dataclasses import replace
    from .backtest import backtest as _bt

    closed = [t for t in trades if t.get("status") == "closed"]
    if len(closed) < 5:
        if verbose:
            print(f"\n  [reflect #{reflection_count}] only {len(closed)} closed trades — skipping\n")
        return cfg, None

    # Use only the last `window` trades for evaluation
    recent = closed[-window:]
    baseline_score = score(recent, goal)

    if verbose:
        print(f"\n  [reflect #{reflection_count}] baseline_score={baseline_score:.4f} "
              f"on last {len(recent)} trades")

    # Extract the slice of original_df that covers these trades
    # We need the candle range from first trade entry to current_candle
    first_trade_ts = recent[0].get("entry_time", 0)
    # Find the candle index in original_df closest to first_trade_ts
    # We'll use a window of 500 candles before current to ensure enough lookback
    eval_start = max(0, current_candle - 500)
    eval_df = original_df.iloc[eval_start:current_candle + 1].copy()

    # Generate K candidates
    search_space = MCTFBConfig.search_space()
    candidates = []
    seen_keys = set()

    for _k in range(8):
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

        # Skip if no actual change
        if new_val == old_val:
            continue

        cand_cfg = replace(cfg, **{var_name: new_val})
        candidates.append((cand_cfg, var_name, old_val, new_val))

    if not candidates:
        if verbose:
            print(f"  [reflect #{reflection_count}] no valid candidates — keeping config\n")
        return cfg, None

    # Backtest each candidate on the eval window
    best_score = baseline_score
    best_cfg = cfg
    best_var = None
    best_old = None
    best_new = None

    for cand_cfg, var_name, old_val, new_val in candidates:
        try:
            result = _bt(eval_df, cand_cfg, verbose=False)
            cand_trades = result["trades"]
            cand_closed = [t for t in cand_trades if t.get("status") == "closed"]
            if len(cand_closed) < 3:
                continue
            cand_recent = cand_closed[-window:]
            cand_score = score(cand_recent, goal)
        except Exception:
            continue

        if cand_score > best_score:
            best_score = cand_score
            best_cfg = cand_cfg
            best_var = var_name
            best_old = old_val
            best_new = new_val

    # Re-compute indicators if config changed
    if best_cfg is not cfg:
        new_df = compute_all_indicators(original_df, best_cfg)
        new_df = generate_entry_signals(new_df, best_cfg)
        if verbose:
            improvement = best_score - baseline_score
            print(f"  [reflect #{reflection_count}] KEEP {best_var}: {best_old} -> {best_new} "
                  f"| score {baseline_score:.4f} -> {best_score:.4f} (+{improvement:.4f})")
            print(f"  [reflect #{reflection_count}] re-computing indicators...\n")
        return best_cfg, new_df
    else:
        if verbose:
            print(f"  [reflect #{reflection_count}] no improvement found — keeping config\n")
        return cfg, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterative backtest with reflection")
    parser.add_argument("--asset", default=None, help="Yahoo Finance ticker")
    parser.add_argument("--period", default="730d", help="History period")
    parser.add_argument("--interval", default="1h", help="Candle interval")
    parser.add_argument("--reflect-interval", type=int, default=15,
                        help="Trades between reflections")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    state_dir = STATE_DIR
    strategy_path = state_dir / "strategy.yaml"

    if strategy_path.exists():
        cfg = load_config_from_yaml(strategy_path)
    else:
        cfg = MCTFBConfig()

    if args.asset:
        from dataclasses import replace
        cfg = replace(cfg, asset=args.asset, asset_label=args.asset)

    print("MCTFB Iterative Backtest (Self-Improving)")
    print(f"  asset:    {cfg.asset}")
    print(f"  period:   {args.period}")
    print(f"  interval: {args.interval}")
    print(f"  reflect:  every {args.reflect_interval} trades")

    # Load data
    adapter = YFinanceAdapter()

    async def load_data():
        return await adapter.fetch_history(cfg.asset, interval=args.interval, period=args.period)

    df = asyncio.run(load_data())
    if df.empty:
        print("ERROR: No data loaded.")
        return

    print(f"  candles:  {len(df)}")

    # Run iterative backtest
    result = iterative_backtest(df, cfg, reflect_interval=args.reflect_interval, verbose=args.verbose)

    # Print results
    print_metrics(result["metrics"])
    print(f"  Reflections:     {result['metrics']['reflections']}")
    print(f"  Final config:    {yaml.dump(result['final_config'], default_flow_style=False)}")

    # Save trades
    trades_path = state_dir / "trades.jsonl"
    with open(trades_path, "w") as f:
        for t in result["trades"]:
            f.write(json.dumps(t, default=str) + "\n")
    print(f"\nTrades saved to {trades_path}")


if __name__ == "__main__":
    main()
