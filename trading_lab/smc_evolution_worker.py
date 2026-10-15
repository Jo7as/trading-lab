"""SMC/ICT Evolution Worker — standalone, ST-principle outer loop.

Batch hill-climb per round (MTF backtests are expensive):
  load 4h/15m/1m → train/OOS split → score train → mutate 1 var →
  score forward slice → anneal → OOS validate → save strategy.yaml

State lives in smc-state/ (or LAB_STATE).
"""
from __future__ import annotations
from .env import env

import asyncio
import dataclasses
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .backtest import print_metrics
from .exploration import SMC_RESEARCH_SEEDS, random_smc_config
from .score import score, trades_per_week
from .significance import promotion_gate
from .strategies.smc_ict import SmcIctConfig, run_backtest

_MAX_ANNEAL_DROP = 0.05
_MAX_ANNEAL_P = 0.25

_DEFAULT_STATE = Path(__file__).resolve().parent.parent / "smc-state"


def _period_days(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 2:
        return 1.0
    delta = df.index[-1] - df.index[0]
    return max(delta.total_seconds() / 86400.0, 1.0)


def _slice_mtf(
    df_4h: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_1m: pd.DataFrame,
    start: Any,
    end: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align TF slices to [start, end)."""
    s = pd.Timestamp(start)
    if s.tzinfo is None:
        s = s.tz_localize("UTC")
    if end is None:
        e = None
    else:
        e = pd.Timestamp(end)
        if e.tzinfo is None:
            e = e.tz_localize("UTC")

    def _cut(df: pd.DataFrame) -> pd.DataFrame:
        out = df[df.index >= s]
        if e is not None:
            out = out[out.index < e]
        return out.copy()

    return _cut(df_4h), _cut(df_15m), _cut(df_1m)


def _forward_mtf_slice(
    df_4h: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_1m: pd.DataFrame,
    *,
    start_ts: Any,
    days: int,
    min_1m_bars: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Forward window of `days` calendar days from start_ts."""
    s = pd.Timestamp(start_ts)
    if s.tzinfo is None:
        s = s.tz_localize("UTC")
    e = s + pd.Timedelta(days=days)
    a, b, c = _slice_mtf(df_4h, df_15m, df_1m, s, e)
    if len(c) < min_1m_bars or a.empty or b.empty:
        return None
    return a, b, c


class SmcEvolutionWorker:
    """Batch hill-climb evolution for SMC/ICT nested MTF strategy."""

    def __init__(
        self,
        state_dir: Path | None = None,
        *,
        year_start: int = 2020,
        year_end: int | None = None,
        oos_split: float = 0.30,
        num_candidates: int = 10,
        forward_eval_days: int = 90,
        explore_every: int = 8,
        explore_holdout_split: float = 0.30,
        max_rounds: int | None = None,
        verbose: bool = True,
    ):
        env_state = env("LAB_STATE", "").strip()
        if state_dir is not None:
            self.state_dir = Path(state_dir)
        elif env_state:
            self.state_dir = Path(env_state)
        else:
            self.state_dir = _DEFAULT_STATE
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.year_start = year_start
        self.year_end = year_end
        self.oos_split = oos_split
        self.num_candidates = num_candidates
        self.forward_eval_days = forward_eval_days
        self.explore_every = explore_every
        self.explore_holdout_split = explore_holdout_split
        self.max_rounds = max_rounds
        self.verbose = verbose

        self.goal = self._load_goal()
        self.evolution_round = 0
        self.stagnation_rounds = 0

        # Override candidates / forward days from goal when present
        if "candidates_per_round" in self.goal:
            self.num_candidates = int(self.goal["candidates_per_round"])
        if "forward_eval_days" in self.goal:
            self.forward_eval_days = int(self.goal["forward_eval_days"])

    def _load_goal(self) -> dict:
        path = self.state_dir / "goal.yaml"
        if not path.exists():
            return {
                "asset": "EUR/USD",
                "asset_label": "EUR/USD",
                "target_return_30d": 0.04,
                "max_drawdown": 0.08,
                "min_sharpe": 0.5,
                "min_trades_per_week": 1.0,
                "target_trades_per_week": 3.0,
            }
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def _goal_asset(self) -> tuple[str, str]:
        asset = str(self.goal.get("asset", "EUR/USD"))
        label = str(self.goal.get("asset_label", asset))
        return asset, label

    def _pin_goal_asset(self, cfg: SmcIctConfig) -> SmcIctConfig:
        asset, label = self._goal_asset()
        if cfg.asset == asset and cfg.asset_label == label:
            pinned = cfg
        else:
            pinned = replace(cfg, asset=asset, asset_label=label)
        return SmcIctConfig.pin_strategy_locks(pinned)

    @staticmethod
    def _config_snapshot(cfg: SmcIctConfig) -> tuple[tuple[str, Any], ...]:
        return tuple(sorted(asdict(cfg).items()))

    def _baseline_path(self) -> Path:
        return self.state_dir / "smc_baseline.yaml"

    def _load_strategy(self, path: Path | None = None) -> SmcIctConfig:
        path = path or (self.state_dir / "strategy.yaml")
        if not path.exists():
            return self._pin_goal_asset(SmcIctConfig(sl_buffer=0.0008))
        raw = yaml.safe_load(path.read_text()) or {}
        valid = {f.name for f in dataclasses.fields(SmcIctConfig)}
        kwargs = {k: v for k, v in raw.items() if k in valid}
        try:
            return self._pin_goal_asset(SmcIctConfig(**kwargs))
        except Exception:
            return self._pin_goal_asset(SmcIctConfig(sl_buffer=0.0008))

    def _save_prior(self, strategy: dict) -> None:
        hist = self.state_dir / "history"
        hist.mkdir(parents=True, exist_ok=True)
        version = strategy.get("version", "01")
        path = hist / f"v{version}.yaml"
        with open(path, "w") as f:
            yaml.dump(strategy, f, default_flow_style=False, sort_keys=False)

    def _save_strategy(self, cfg: SmcIctConfig, *, increment: bool = False) -> None:
        path = self.state_dir / "strategy.yaml"
        raw = yaml.safe_load(path.read_text()) if path.exists() else {}
        old_ver = str(raw.get("version", "01"))
        if increment and raw:
            self._save_prior(raw)
        if increment:
            try:
                new_ver = f"{int(old_ver, 10) + 1:02d}"
            except (TypeError, ValueError):
                new_ver = "02"
        else:
            new_ver = old_ver
        d = asdict(self._pin_goal_asset(cfg))
        d["version"] = new_ver
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)
        if increment and self.verbose:
            print(f"    [strategy] saved v{new_ver}")

    def _append_hypothesis(self, hyp: dict) -> None:
        path = self.state_dir / "hypotheses.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(hyp, default=str) + "\n")

    def _save_trades(self, trades: list[dict]) -> None:
        path = self.state_dir / "trades.jsonl"
        with open(path, "w") as f:
            for t in trades:
                f.write(json.dumps(t, default=str) + "\n")

    def _enforce_goal_asset(self) -> tuple[SmcIctConfig, bool]:
        strategy_path = self.state_dir / "strategy.yaml"
        goal_asset, _ = self._goal_asset()
        if not strategy_path.exists():
            baseline = self._baseline_path()
            if baseline.exists():
                shutil.copy2(baseline, strategy_path)
            return self._load_strategy(strategy_path), True

        raw = yaml.safe_load(strategy_path.read_text()) or {}
        if str(raw.get("asset", "")) == goal_asset:
            return self._load_strategy(strategy_path), False

        print(
            f"  [asset] DRIFT detected: {raw.get('asset')!r} != {goal_asset!r} "
            f"— restoring {self._baseline_path().name}"
        )
        self._save_prior(raw)
        baseline = self._baseline_path()
        if baseline.exists():
            shutil.copy2(baseline, strategy_path)
        cfg = self._load_strategy(strategy_path)
        self._save_strategy(cfg, increment=False)
        self.stagnation_rounds = 0
        return cfg, True

    async def _fetch_mtf(
        self, asset: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        from .adapters.dukascopy_adapter import DukascopyAdapter

        start = datetime(self.year_start, 1, 1, tzinfo=timezone.utc)
        if self.year_end is not None:
            end = datetime(self.year_end, 1, 1, tzinfo=timezone.utc)
        else:
            end = datetime.now(timezone.utc)

        ad = DukascopyAdapter()
        print(f"  Fetching {asset} 4h/15m/1m {start.date()} → {end.date()}...")
        df_4h = await ad.fetch_history(asset, "4h", start=start, end=end)
        df_15m = await ad.fetch_history(asset, "15m", start=start, end=end)
        df_1m = await ad.fetch_history(asset, "1m", start=start, end=end)
        print(f"  Loaded 4h={len(df_4h)} 15m={len(df_15m)} 1m={len(df_1m)}")
        return df_4h, df_15m, df_1m

    def _score_mtf(
        self,
        df_4h: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_1m: pd.DataFrame,
        cfg: SmcIctConfig,
    ) -> tuple[float, int, dict]:
        result = run_backtest(df_4h, df_15m, df_1m, cfg, verbose=False)
        closed = [t for t in result["trades"] if t.get("status", "closed") == "closed"]
        # run_backtest trades are already closed entries
        if not closed:
            closed = list(result["trades"])
        days = _period_days(df_1m)
        if len(closed) < 2:
            return -1.0, len(closed), result
        return score(closed, self.goal, period_days=days), len(closed), result

    def _mutate_one(self, cfg: SmcIctConfig) -> tuple[SmcIctConfig, str, Any, Any] | None:
        """Mutate exactly one numeric or categorical knob."""
        numeric = SmcIctConfig.search_space()
        categorical = SmcIctConfig.categorical_space()
        use_cat = random.random() < 0.35 and categorical

        if use_cat:
            var_name, choices = random.choice(categorical)
            old_val = getattr(cfg, var_name)
            options = [c for c in choices if c != old_val]
            if not options:
                return None
            new_val = random.choice(options)
        else:
            var_name, lo, hi, step = random.choice(numeric)
            old_val = getattr(cfg, var_name)
            direction = random.choice([-1, 1])
            new_val = float(old_val) + direction * step
            new_val = max(lo, min(hi, new_val))
            if var_name in (
                "swing_length_15m",
                "swing_length_1m",
                "ob_lookback",
                "entry_timeout_bars",
                "choch_lookback_bars",
                "choch_refine_bars",
                "max_hold_bars",
            ):
                new_val = int(round(new_val))
            else:
                new_val = round(float(new_val), 6)
            if new_val == old_val:
                return None

        # Guard: never turn off all liquidity sources
        trial = SmcIctConfig.pin_strategy_locks(replace(cfg, **{var_name: new_val}))
        # If the only mutation was a locked field (shouldn't happen), skip
        if getattr(trial, var_name) != new_val and var_name in SmcIctConfig.STRATEGY_LOCKS:
            return None
        if not any(
            getattr(trial, k) for k in ("use_asia", "use_london", "use_pd", "use_pw")
        ):
            return None
        return trial, var_name, old_val, getattr(trial, var_name)

    def _hill_climb_round(
        self,
        cfg: SmcIctConfig,
        df_4h: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_1m: pd.DataFrame,
    ) -> SmcIctConfig:
        """One-variable hill-climb on a forward slice inside train."""
        # Forward window: start ~40% into train so there's history + room ahead
        n = len(df_1m)
        start_i = int(n * 0.40)
        start_ts = df_1m.index[start_i]
        fwd = _forward_mtf_slice(
            df_4h,
            df_15m,
            df_1m,
            start_ts=start_ts,
            days=self.forward_eval_days,
        )
        if fwd is None:
            # Fallback: last 30% of train
            cut = int(n * 0.70)
            start_ts = df_1m.index[cut]
            fwd = _slice_mtf(df_4h, df_15m, df_1m, start_ts)
            if fwd[2].empty or len(fwd[2]) < 2000:
                if self.verbose:
                    print("    [hill] skip — insufficient forward bars")
                return cfg

        f4, f15, f1 = fwd
        baseline_score, baseline_n, _ = self._score_mtf(f4, f15, f1, cfg)
        if baseline_n < 2:
            if self.verbose:
                print(f"    [hill] skip — baseline trades={baseline_n}")
            return cfg

        cycle = self.evolution_round % 100
        temperature = max(0.08, 0.6 * (0.98**cycle))
        if self.stagnation_rounds >= 5:
            temperature = max(temperature, 0.25)

        if self.verbose:
            print(
                f"    [hill] forward baseline={baseline_score:.4f} "
                f"trades={baseline_n} temp={temperature:.3f} "
                f"bars_1m={len(f1)} days={self.forward_eval_days}"
            )

        best_score = baseline_score
        best_cfg = cfg
        best_meta: tuple[str, Any, Any] | None = None
        wildcards: list[tuple] = []

        seen: set[str] = set()
        for _ in range(self.num_candidates):
            mut = self._mutate_one(cfg)
            if mut is None:
                continue
            cand_cfg, var_name, old_val, new_val = mut
            key = f"{var_name}={new_val}"
            if key in seen:
                continue
            seen.add(key)
            try:
                cand_score, cand_n, _ = self._score_mtf(f4, f15, f1, cand_cfg)
            except Exception as exc:
                if self.verbose:
                    print(f"    [hill] candidate error {var_name}: {exc}")
                continue
            if cand_n < 2:
                continue
            delta = cand_score - baseline_score
            if cand_score > best_score:
                best_score = cand_score
                best_cfg = cand_cfg
                best_meta = (var_name, old_val, new_val)
            elif delta > -_MAX_ANNEAL_DROP:
                # Annealing pool
                p = min(
                    _MAX_ANNEAL_P,
                    math.exp(delta / max(temperature, 1e-6)) if delta < 0 else 0.0,
                )
                wildcards.append((cand_cfg, cand_score, var_name, old_val, new_val, p))

        if best_cfg is not cfg and best_meta is not None:
            var_name, old_val, new_val = best_meta
            if self.verbose:
                print(
                    f"    [hill] KEEP {var_name}: {old_val} → {new_val} "
                    f"| {baseline_score:.4f} → {best_score:.4f}"
                )
            self._log_hypothesis(
                baseline_score, best_score, var_name, old_val, new_val, baseline_n
            )
            self._save_strategy(best_cfg, increment=True)
            return best_cfg

        # Mild anneal accept
        if wildcards:
            wildcards.sort(key=lambda x: x[1], reverse=True)
            cand_cfg, cand_score, var_name, old_val, new_val, p = wildcards[0]
            if random.random() < p:
                if self.verbose:
                    print(
                        f"    [hill] ANNEAL {var_name}: {old_val} → {new_val} "
                        f"| score={cand_score:.4f} p={p:.3f}"
                    )
                self._log_hypothesis(
                    baseline_score, cand_score, f"anneal:{var_name}", old_val, new_val, baseline_n
                )
                self._save_strategy(cand_cfg, increment=True)
                return cand_cfg

        if self.verbose:
            print("    [hill] no improvement — keeping config")
        return cfg

    def _log_hypothesis(
        self,
        old_score: float,
        new_score: float,
        var: str,
        old_val: Any,
        new_val: Any,
        trades_n: int,
    ) -> None:
        path = self.state_dir / "strategy.yaml"
        strategy = yaml.safe_load(path.read_text()) if path.exists() else {}
        old_ver = strategy.get("version", "01")
        hyp = {
            "ts": time.time(),
            "mode": "smc_evolution",
            "round": self.evolution_round,
            "version": old_ver,
            "baseline_score": round(old_score, 4),
            "new_score": round(new_score, 4),
            "improvement": round(new_score - old_score, 4),
            "trades_examined": trades_n,
            "variable_changed": var,
            "old_value": old_val,
            "new_value": new_val,
        }
        self._append_hypothesis(hyp)

    def _exploration_jump(
        self,
        cfg: SmcIctConfig,
        df_4h: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_1m: pd.DataFrame,
    ) -> SmcIctConfig:
        if self.verbose:
            print(
                f"\n  [explore] stagnation={self.stagnation_rounds} — "
                f"testing research seeds + random configs"
            )
        n = len(df_1m)
        cut = int(n * (1 - self.explore_holdout_split))
        start_ts = df_1m.index[cut]
        e4, e15, e1 = _slice_mtf(df_4h, df_15m, df_1m, start_ts)
        if len(e1) < 2000:
            return cfg

        baseline_score, baseline_n, _ = self._score_mtf(e4, e15, e1, cfg)
        best_score = baseline_score
        best_cfg = cfg
        best_label = "current"

        candidates: list[tuple[str, SmcIctConfig]] = list(SMC_RESEARCH_SEEDS)
        for i in range(4):
            candidates.append((f"random_{i}", random_smc_config()))

        for label, cand in candidates:
            cand = self._pin_goal_asset(cand)
            try:
                cand_score, cand_n, _ = self._score_mtf(e4, e15, e1, cand)
            except Exception:
                continue
            if cand_n < 3:
                continue
            if cand_score > best_score:
                best_score = cand_score
                best_cfg = cand
                best_label = label

        if best_cfg is not cfg:
            if self.verbose:
                print(
                    f"  [explore] ADOPT {best_label} | "
                    f"{baseline_score:.4f} → {best_score:.4f}"
                )
            self._log_hypothesis(
                baseline_score,
                best_score,
                f"explore:{best_label}",
                "seed",
                best_label,
                baseline_n,
            )
            self._save_strategy(best_cfg, increment=True)
            return best_cfg

        if self.verbose:
            print(f"  [explore] no seed beat current (baseline={baseline_score:.4f})")
        return cfg

    def _print_trades_per_week(
        self, trades: list[dict], label: str, period_days: float
    ) -> None:
        tpw = trades_per_week(trades, period_days)
        target = self.goal.get("target_trades_per_week", 3)
        min_tpw = self.goal.get("min_trades_per_week", 1)
        status = "OK" if tpw >= min_tpw else "LOW"
        print(
            f"  Trades/week ({label}): {tpw:.1f} "
            f"(target {min_tpw}–{target}) [{status}]"
        )

    def _check_goal(self, metrics: dict, label: str, period_days: float) -> bool:
        target = self.goal.get("target_return_30d", 0.04) * 100
        max_dd = self.goal.get("max_drawdown", 0.08) * 100
        min_sharpe = self.goal.get("min_sharpe", 0.5)
        total_pnl = metrics.get("total_pnl_pct", 0)
        est_30d = total_pnl / (period_days / 30) if period_days > 0 else 0
        actual_dd = abs(metrics.get("max_drawdown_pct", 0))
        actual_sharpe = metrics.get("sharpe", 0)
        print(f"\n  Goal check ({label}):")
        print(
            f"    Return/30d:  {est_30d:.2f}% (target: {target:.0f}%) "
            f"{'PASS' if est_30d >= target else 'FAIL'}"
        )
        print(
            f"    Max DD:      {actual_dd:.2f}% (limit: {max_dd:.0f}%) "
            f"{'PASS' if actual_dd <= max_dd else 'FAIL'}"
        )
        print(
            f"    Sharpe:      {actual_sharpe:.3f} (min: {min_sharpe}) "
            f"{'PASS' if actual_sharpe >= min_sharpe else 'FAIL'}"
        )
        return est_30d >= target and actual_dd <= max_dd and actual_sharpe >= min_sharpe

    def _update_best_oos(
        self, cfg: SmcIctConfig, oos_pnl: float, trades: list[dict] | None = None
    ) -> None:
        ok, sig = promotion_gate(trades or [])
        if not ok:
            print(f"  [best-oos] blocked — significance gate: {sig.summary()}")
            return
        meta_path = self.state_dir / "best_oos_meta.json"
        strat_path = self.state_dir / "best_oos.yaml"
        prev = -1e18
        if meta_path.exists():
            try:
                prev = float(json.loads(meta_path.read_text()).get("oos_pnl_pct", prev))
            except Exception:
                pass
        if oos_pnl <= prev:
            return
        d = asdict(cfg)
        path = self.state_dir / "strategy.yaml"
        if path.exists():
            d["version"] = (yaml.safe_load(path.read_text()) or {}).get("version", "01")
        with open(strat_path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)
        meta_path.write_text(
            json.dumps(
                {
                    "oos_pnl_pct": oos_pnl,
                    "evolution_round": self.evolution_round,
                    "version": d.get("version"),
                },
                indent=2,
            )
        )
        if self.verbose:
            print(f"  [best-oos] new record: {oos_pnl:.2f}% OOS")

    async def run_forever(self) -> None:
        asset, _ = self._goal_asset()
        df_4h, df_15m, df_1m = await self._fetch_mtf(asset)
        if df_1m.empty:
            print("  ERROR: no 1m data")
            return

        split_i = int(len(df_1m) * (1 - self.oos_split))
        split_ts = df_1m.index[split_i]
        train_4h, train_15m, train_1m = _slice_mtf(df_4h, df_15m, df_1m, df_1m.index[0], split_ts)
        oos_4h, oos_15m, oos_1m = _slice_mtf(df_4h, df_15m, df_1m, split_ts)
        print(
            f"  Train 1m={len(train_1m)} | OOS 1m={len(oos_1m)} "
            f"(split {split_ts})"
        )

        while True:
            self.evolution_round += 1
            if self.max_rounds is not None and self.evolution_round > self.max_rounds:
                print(f"  Reached max_rounds={self.max_rounds} — stopping")
                break

            print(f"\n{'=' * 60}")
            print(f"SMC EVOLUTION ROUND #{self.evolution_round}")
            print(f"{'=' * 60}")

            cfg, _ = self._enforce_goal_asset()
            round_start = self._config_snapshot(cfg)
            print(
                f"  asset={cfg.asset} lookback={cfg.ob_lookback} "
                f"limit={cfg.entry_limit_mode} sl_buffer={cfg.sl_buffer} "
                f"liq=A{int(cfg.use_asia)}L{int(cfg.use_london)}"
                f"PD{int(cfg.use_pd)}PW{int(cfg.use_pw)}"
            )

            if self.stagnation_rounds >= self.explore_every:
                cfg = self._exploration_jump(cfg, train_4h, train_15m, train_1m)

            # Full train metrics only every N rounds (expensive); hill-climb always
            do_full_train = (
                self.evolution_round == 1
                or self.evolution_round % 3 == 0
            )
            if do_full_train:
                print("\n  Running train backtest...")
                t0 = time.time()
                train_score, train_n, train_result = self._score_mtf(
                    train_4h, train_15m, train_1m, cfg
                )
                print(
                    f"  Train done in {time.time() - t0:.1f}s | score={train_score:.4f}"
                )
                print_metrics(train_result["metrics"], label="IN-SAMPLE (train)")
                self._print_trades_per_week(
                    train_result["trades"], "In-sample", _period_days(train_1m)
                )
                self._check_goal(
                    train_result["metrics"], "In-sample", _period_days(train_1m)
                )
                self._save_trades(train_result["trades"])
            else:
                print("\n  Skipping full train (every 3rd round) — hill-climb only")

            # Hill-climb on forward slice
            print("\n  Hill-climb...")
            cfg = self._hill_climb_round(cfg, train_4h, train_15m, train_1m)

            # OOS with frozen evolved config
            evolved = self._load_strategy()
            print("\n  Running OOS backtest...")
            t1 = time.time()
            oos_score, oos_n, oos_result = self._score_mtf(
                oos_4h, oos_15m, oos_1m, evolved
            )
            print(f"  OOS done in {time.time() - t1:.1f}s | score={oos_score:.4f}")
            print_metrics(oos_result["metrics"], label="OUT-OF-SAMPLE (holdout)")
            self._print_trades_per_week(
                oos_result["trades"], "OOS", _period_days(oos_1m)
            )
            self._check_goal(oos_result["metrics"], "OOS", _period_days(oos_1m))
            self._update_best_oos(
                evolved,
                float(oos_result["metrics"].get("total_pnl_pct", 0)),
                oos_result.get("trades"),
            )

            round_end = self._config_snapshot(evolved)
            if round_end != round_start:
                self.stagnation_rounds = 0
                print("  Config changed this round")
            else:
                self.stagnation_rounds += 1
                print(f"  Stagnation: {self.stagnation_rounds} rounds")

            print("  Starting next round...")

    async def run_rounds(self, n: int) -> None:
        self.max_rounds = n
        await self.run_forever()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SMC/ICT Evolution Worker")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="State directory (default: smc-state/)",
    )
    parser.add_argument("--year-start", type=int, default=2020)
    parser.add_argument(
        "--year-end",
        type=int,
        default=None,
        help="Exclusive end year (default: now)",
    )
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--forward-days", type=int, default=None)
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Stop after N rounds (default: forever)",
    )
    parser.add_argument("--oos-split", type=float, default=0.30)
    args = parser.parse_args()

    worker = SmcEvolutionWorker(
        state_dir=args.state_dir,
        year_start=args.year_start,
        year_end=args.year_end,
        oos_split=args.oos_split,
        num_candidates=args.candidates or 10,
        forward_eval_days=args.forward_days or 90,
        max_rounds=args.rounds,
        verbose=True,
    )
    if args.candidates:
        worker.num_candidates = args.candidates
    if args.forward_days:
        worker.forward_eval_days = args.forward_days

    print("SMC/ICT Evolution Worker")
    print(f"  state:      {worker.state_dir}")
    print(f"  years:      {worker.year_start} → {worker.year_end or 'now'}")
    print(f"  candidates: {worker.num_candidates}")
    print(f"  forward:    {worker.forward_eval_days}d")
    print(f"  rounds:     {worker.max_rounds or 'forever'}")
    print()
    asyncio.run(worker.run_forever())


if __name__ == "__main__":
    main()
