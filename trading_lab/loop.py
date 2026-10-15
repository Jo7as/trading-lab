"""Live execution loop for MCTFB strategy on Railway.

Every tick:
  1. Fetch latest OHLCV from Yahoo Finance
  2. Compute indicators + entry signals
  3. Manage open position (SL, TP1, TP2, time stop)
  4. Check for new entry if no open position
  5. Write heartbeat
  6. Reflect after N closed trades (from goal.yaml)
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .config import MCTFBConfig, load_config_from_yaml
from .adapters.yfinance_adapter import YFinanceAdapter
from .indicators import compute_all_indicators
from .signals import generate_entry_signals
from .risk import RiskManager, Position
from .reflect import reflect_fallback, load_trades

MAX_CONSECUTIVE_FAILURES = 5


class TradingLoop:
    def __init__(
        self,
        asset: str,
        goal: dict,
        state_dir: Path,
        interval_seconds: int = 300,
        mode: str = "paper",
    ):
        self.asset = asset
        self.goal = goal
        self.state_dir = Path(state_dir)
        self.interval = interval_seconds
        self.mode = mode
        self.consecutive_failures = 0
        self.trades_since_reflection = 0
        self.reflect_interval = int(goal.get("reflection_every", 25))

        self.adapter = YFinanceAdapter()
        self.rm = RiskManager(MCTFBConfig())
        self.open_trade: Position | None = None

    async def run_forever(self) -> None:
        while True:
            try:
                await self._tick()
                self.consecutive_failures = 0
            except Exception as exc:
                self.consecutive_failures += 1
                print(f"[loop] error: {exc}")
                if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print("[loop] circuit-breaker tripped")
                    return
            await asyncio.sleep(self.interval)

    async def _tick(self) -> None:
        cfg = load_config_from_yaml(self.state_dir / "strategy.yaml")
        self.rm.cfg = cfg

        # Fetch data
        result = await self.adapter.fetch(cfg.asset, interval="1h", period="30d")
        if result.get("price") is None:
            print("[loop] no data from yfinance")
            return

        df = result.get("df")
        if df is None or df.empty:
            return

        # Compute indicators + signals
        df = compute_all_indicators(df, cfg)
        df = generate_entry_signals(df, cfg)

        last = df.iloc[-1]
        current_ts = time.time()
        current_price = float(last["close"])

        # Manage open position
        if self.open_trade:
            self._manage_open(last, current_ts, current_price, cfg)
            if self.open_trade.status == "closed":
                self._log_trade(self.open_trade)
                self.trades_since_reflection += 1
                self.open_trade = None

                if self.trades_since_reflection >= self.reflect_interval:
                    self._reflect()

        # Check for new entry
        if self.open_trade is None:
            allowed, reason = self.rm.can_trade(current_ts)
            if allowed:
                atr_val = float(last["atr"]) if last.get("atr") else 0.0
                if atr_val > 0:
                    if last.get("long_signal", 0) > 0:
                        pos = self.rm.open_position(
                            current_price, atr_val, "long", current_ts, len(df) - 1
                        )
                        if pos:
                            self.open_trade = pos
                            self._log_trade(pos)
                            print(f"[trade] OPEN long {cfg.asset} @ {current_price:.2f} "
                                  f"SL={pos.stop_loss:.2f} TP1={pos.tp1:.2f}")
                    elif last.get("short_signal", 0) > 0:
                        pos = self.rm.open_position(
                            current_price, atr_val, "short", current_ts, len(df) - 1
                        )
                        if pos:
                            self.open_trade = pos
                            self._log_trade(pos)
                            print(f"[trade] OPEN short {cfg.asset} @ {current_price:.2f} "
                                  f"SL={pos.stop_loss:.2f} TP1={pos.tp1:.2f}")

        self._write_heartbeat(last, cfg)

    def _manage_open(
        self, row: pd.Series, current_ts: float, current_price: float, cfg: MCTFBConfig
    ) -> None:
        """Check SL, TP1, TP2, time stop for the open position."""
        pos = self.open_trade
        if pos is None or pos.status == "closed":
            return

        current_high = float(row["high"])
        current_low = float(row["low"])

        # Stop loss
        if pos.direction == "long":
            if current_low <= pos.stop_loss:
                self.rm.close_position(pos, pos.stop_loss, "stop_loss", current_ts)
                return
        else:
            if current_high >= pos.stop_loss:
                self.rm.close_position(pos, pos.stop_loss, "stop_loss", current_ts)
                return

        # TP1 (partial close)
        if not pos.tp1_hit:
            if pos.direction == "long" and current_high >= pos.tp1:
                self.rm.close_position(pos, pos.tp1, "tp1_partial", current_ts, partial=True)
                print(f"[trade] TP1 hit @ {pos.tp1:.2f} — 50% closed, SL→breakeven")
                return
            elif pos.direction == "short" and current_low <= pos.tp1:
                self.rm.close_position(pos, pos.tp1, "tp1_partial", current_ts, partial=True)
                print(f"[trade] TP1 hit @ {pos.tp1:.2f} — 50% closed, SL→breakeven")
                return

        # TP2 (supertrend flip)
        if pos.tp1_hit:
            st_dir = float(row.get("st_dir", 0))
            if pos.direction == "long" and st_dir < 0:
                self.rm.close_position(pos, current_price, "supertrend_exit", current_ts)
                print(f"[trade] Supertrend exit @ {current_price:.2f}")
                return
            elif pos.direction == "short" and st_dir > 0:
                self.rm.close_position(pos, current_price, "supertrend_exit", current_ts)
                print(f"[trade] Supertrend exit @ {current_price:.2f}")
                return

        # Time stop
        candles_held = len(row) - pos.entry_idx  # rough approximation
        if candles_held >= cfg.max_hold_hours:
            self.rm.close_position(pos, current_price, "time_stop", current_ts)
            print(f"[trade] Time stop exit @ {current_price:.2f}")
            return

    def _reflect(self) -> None:
        """Hill-climb reflection: test candidates, keep only if better."""
        print(f"[reflect] {self.trades_since_reflection} trades — hill-climbing...")
        try:
            from .iterative_backtest import _do_reflection
            from .config import load_config_from_yaml
            from .adapters.yfinance_adapter import YFinanceAdapter
            from .score import score as _score
            from .reflect import load_trades, load_yaml
            from dataclasses import replace as _replace

            # Load current config and goal
            strategy_path = self.state_dir / "strategy.yaml"
            goal_path = self.state_dir / "goal.yaml"
            cfg = load_config_from_yaml(strategy_path)
            goal = load_yaml(goal_path)

            # Load recent trades
            trades = load_trades(50)
            if len(trades) < 5:
                print("[reflect] not enough trades yet — skipping")
                self.trades_since_reflection = 0
                return

            # Fetch recent OHLCV for candidate backtesting
            adapter = YFinanceAdapter()
            import asyncio as _aio
            loop = _aio.new_event_loop()
            df = loop.run_until_complete(
                adapter.fetch_history(cfg.asset, interval="1h", period="60d")
            )
            loop.close()

            if df is None or df.empty:
                print("[reflect] no data for backtest — using random fallback")
                reflect_fallback()
                self.trades_since_reflection = 0
                return

            # Run hill-climb reflection
            new_cfg, new_df = _do_reflection(
                trades, cfg, df, self.state_dir, goal,
                len(df) - 1, 0, True, window=25
            )

            # Save updated config to strategy.yaml if changed
            if new_cfg is not cfg:
                from .config import config_to_yaml, config_to_dict
                import yaml as _yaml
                # Read current version, increment
                raw = _yaml.safe_load(strategy_path.read_text())
                old_version = raw.get("version", "01")
                new_version = str(int(old_version) + 1).zfill(2)
                # Write new config
                new_dict = {k: v for k, v in config_to_dict(new_cfg).items()}
                new_dict["version"] = new_version
                with open(strategy_path, "w") as f:
                    _yaml.dump(new_dict, f, default_flow_style=False, sort_keys=False)
                print(f"[reflect] strategy updated to v{new_version}")
            else:
                print("[reflect] no improvement — keeping config")

            self.trades_since_reflection = 0
        except Exception as exc:
            print(f"[reflect] FAILED: {exc}")
            # Fallback to random search
            try:
                reflect_fallback()
            except Exception:
                pass
            self.trades_since_reflection = 0

    def _log_trade(self, pos: Position) -> None:
        path = self.state_dir / "trades.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(asdict(pos), default=str) + "\n")

    def _write_heartbeat(self, row: pd.Series, cfg: MCTFBConfig) -> None:
        hb = {
            "ts": time.time(),
            "asset": cfg.asset,
            "mode": self.mode,
            "price": float(row.get("close", 0)),
            "rsi": float(row.get("rsi", 0)),
            "atr": float(row.get("atr", 0)),
            "trend_up": bool(row.get("trend_up", 0)),
            "squeeze": bool(row.get("squeeze", 0)),
            "open_trade": asdict(self.open_trade) if self.open_trade else None,
            "risk_state": self.rm.get_stats(),
            "consecutive_failures": self.consecutive_failures,
            "trades_since_reflection": self.trades_since_reflection,
            "reflect_interval": self.reflect_interval,
        }
        path = self.state_dir / "heartbeat.json"
        with open(path, "w") as f:
            json.dump(hb, f, indent=2, default=str)


# Keep pd import at top level
import pandas as pd
