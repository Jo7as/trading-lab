"""Shared evolution loop for the mean-reversion candidates.

Same round structure as ``orb_evolution_worker``: train score → mutate one
param (hill-climb + simulated annealing) → OOS validate. Costs are already in
the fills (bid/ask + slippage), so there is no separate cost haircut.

``hill_climb`` / ``evolve`` never touch disk and are reused by the walk-forward
report; ``run_forever`` adds the state-volume persistence around them.
"""
from __future__ import annotations
from .env import env

import json
import math
import os
import random
import time
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import reflect
from .backtest import print_metrics
from .mr_common import load_bid_ask, period_days, resample_ohlc
from .score import score, trades_per_week
from .significance import promotion_gate

_MAX_ANNEAL_DROP = 0.05
_MAX_ANNEAL_P = 0.25

Frames = dict[str, pd.DataFrame]


def slice_frames(frames: Frames, start: pd.Timestamp | None, end: pd.Timestamp | None) -> Frames:
    """[start, end) of every timeframe."""
    out: Frames = {}
    for tf, df in frames.items():
        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= df.index >= start
        if end is not None:
            mask &= df.index < end
        out[tf] = df[mask.to_numpy()]
    return out


def frames_span(frames: Frames) -> tuple[pd.Timestamp, pd.Timestamp]:
    df = next(iter(frames.values()))
    return df.index[0], df.index[-1]


class MrEvolutionBase:
    """Hill-climb evolution over a frozen strategy dataclass."""

    MODE = "mr_evolution"
    VERSION_PREFIX = "mr"
    BASELINE_FILE = "baseline.yaml"
    DEFAULT_ASSET = "XAU/USD"
    BASE_TIMEFRAME = "15min"           # what gets downloaded
    TIMEFRAMES: tuple[str, ...] = ("15min",)
    config_cls: Any = None

    def __init__(
        self,
        state_dir: Path | None = None,
        *,
        year_start: int = 2020,
        oos_split: float = 0.30,
        num_candidates: int = 12,
        forward_eval_days: int = 365,
        explore_every: int = 15,
        max_rounds: int | None = None,
        asset: str | None = None,
        verbose: bool = True,
    ) -> None:
        self.state_dir = state_dir or reflect.STATE_DIR
        self.year_start = year_start
        self.oos_split = oos_split
        self.num_candidates = num_candidates
        self.forward_eval_days = forward_eval_days
        self.explore_every = explore_every
        self.max_rounds = max_rounds
        self.verbose = verbose
        self.goal = reflect.load_yaml(self.state_dir / "goal.yaml") or {}
        if asset:
            self.goal["asset"] = asset
            self.goal["asset_label"] = asset
        self.evolution_round = 0
        self.stagnation_rounds = 0

    # ── Strategy hooks (subclasses) ────────────────────────────────

    def _backtest(self, df: pd.DataFrame, cfg: Any) -> dict[str, Any]:
        raise NotImplementedError

    def _extra_mutation(self, cfg: Any) -> tuple[Any, str, Any, Any] | None:
        """Optional non-numeric mutation (flags, timeframe); None = use search_space."""
        return None

    def _valid(self, cfg: Any) -> bool:
        return True

    def _random_seed(self, base: Any) -> Any:
        raise NotImplementedError

    def _describe(self, cfg: Any) -> str:
        return ""

    # ── Config / state ─────────────────────────────────────────────

    def _goal_asset(self) -> tuple[str, str]:
        asset = str(self.goal.get("asset", self.DEFAULT_ASSET))
        return asset, str(self.goal.get("asset_label", asset))

    def _pin(self, cfg: Any) -> Any:
        asset, label = self._goal_asset()
        return replace(cfg, asset=asset, asset_label=label)

    def _load_config(self, path: Path) -> Any:
        raw = (yaml.safe_load(path.read_text()) if path.exists() else {}) or {}
        valid = {f.name for f in fields(self.config_cls)}
        return self.config_cls(**{k: v for k, v in raw.items() if k in valid})

    def _load_strategy(self) -> Any:
        for name in ("strategy.yaml", self.BASELINE_FILE):
            path = self.state_dir / name
            if path.exists():
                return self._pin(self._load_config(path))
        return self._pin(self.config_cls())

    def _current_version(self) -> str:
        path = self.state_dir / "strategy.yaml"
        default = f"{self.VERSION_PREFIX}_01"
        if not path.exists():
            return default
        return str((yaml.safe_load(path.read_text()) or {}).get("version", default))

    def _next_version(self, ver: str) -> str:
        tail = ver.rsplit("_", 1)[-1]
        if tail.isdigit():
            return f"{self.VERSION_PREFIX}_{int(tail) + 1:02d}"
        return f"{self.VERSION_PREFIX}_02"

    def _owns_global_state(self) -> bool:
        # reflect.* helpers write to reflect.STATE_DIR — only use them when
        # that is this worker's own volume, never another strategy's.
        try:
            return self.state_dir.resolve() == reflect.STATE_DIR.resolve()
        except OSError:
            return False

    def _save_prior(self, strategy: dict) -> None:
        if self._owns_global_state():
            reflect.save_prior(strategy)
            return
        history = self.state_dir / "history"
        history.mkdir(parents=True, exist_ok=True)
        path = history / f"v{strategy.get('version', '01')}.yaml"
        path.write_text(yaml.dump(strategy, default_flow_style=False, sort_keys=False))

    def _append_hypothesis(self, hyp: dict) -> None:
        if self._owns_global_state():
            reflect.append_hypothesis(hyp)
            return
        with open(self.state_dir / "hypotheses.jsonl", "a") as f:
            f.write(json.dumps(hyp, default=str) + "\n")

    def _save_strategy(self, cfg: Any) -> None:
        path = self.state_dir / "strategy.yaml"
        ver = self._current_version()
        if path.exists():
            self._save_prior(yaml.safe_load(path.read_text()) or {})
        ver = self._next_version(ver)
        d = asdict(self._pin(cfg))
        d["version"] = ver
        path.write_text(yaml.dump(d, default_flow_style=False, sort_keys=False))
        if self.verbose:
            print(f"  [strategy] saved {ver}")

    @staticmethod
    def _config_snapshot(cfg: Any) -> tuple:
        return tuple(sorted(asdict(cfg).items()))

    # ── Scoring ────────────────────────────────────────────────────

    def run(self, frames: Frames, cfg: Any) -> dict[str, Any]:
        return self._backtest(frames[cfg.timeframe], cfg)

    def score_frames(self, frames: Frames, cfg: Any) -> tuple[float, int, dict[str, Any]]:
        """(score, closed trades, backtest result) for one config on one window."""
        result = self.run(frames, cfg)
        closed = [t for t in result["trades"] if t.get("status") == "closed"]
        days = period_days(frames[cfg.timeframe])
        # goal target is per 30 days — scale it to the window being scored
        goal = dict(self.goal)
        goal["target_return_30d"] = float(goal.get("target_return_30d", 0.02)) * days / 30.0
        return score(closed, goal, period_days=days), len(closed), result

    def rank_frames(self, frames: Frames, cfg: Any) -> tuple[tuple[float, float], int]:
        """((score, PnL %), closed trades) — PnL breaks ties where score() saturates."""
        sc, n, result = self.score_frames(frames, cfg)
        return (sc, round(float(result["metrics"].get("total_pnl_pct", 0.0)), 1)), n

    def _forward_slice(self, frames: Frames) -> Frames:
        start, end = frames_span(frames)
        cut = end - pd.Timedelta(days=self.forward_eval_days)
        return frames if cut <= start else slice_frames(frames, cut, None)

    # ── Search ─────────────────────────────────────────────────────

    def _mutate_one(self, cfg: Any) -> tuple[Any, str, Any, Any] | None:
        extra = self._extra_mutation(cfg)
        if extra is not None:
            return extra if self._valid(extra[0]) else None

        name, lo, hi, step = random.choice(list(self.config_cls.search_space()))
        old = getattr(cfg, name)
        if isinstance(old, int) and not isinstance(old, bool):
            choices = [s for s in range(int(lo), int(hi) + 1, int(step)) if s != old]
            if not choices:
                return None
            new: Any = random.choice(choices)
        else:
            direction = random.choice([-1, 1])
            new = float(old) + direction * float(step)
            new = round((new - float(lo)) / float(step)) * float(step) + float(lo)
            new = max(float(lo), min(float(hi), round(new, 6)))
            if new == float(old):
                return None
        cand = replace(cfg, **{name: new})
        return (cand, name, old, new) if self._valid(cand) else None

    def hill_climb(self, cfg: Any, frames: Frames) -> tuple[Any, dict[str, Any] | None]:
        """One round on ``frames``. Returns (config, change meta or None)."""
        baseline_rank, baseline_n = self.rank_frames(frames, cfg)
        baseline_score = baseline_rank[0]
        if baseline_n < 3:
            if self.verbose:
                print(f"    [hill] skip — baseline trades={baseline_n}")
            return cfg, None

        temperature = max(0.08, 0.5 * (0.98 ** (self.evolution_round % 100)))
        if self.stagnation_rounds >= 5:
            temperature = max(temperature, 0.25)
        if self.verbose:
            print(
                f"    [hill] baseline={baseline_score:.4f} trades={baseline_n} "
                f"temp={temperature:.3f}"
            )

        best_rank, best_cfg = baseline_rank, cfg
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
                cand_rank, cand_n = self.rank_frames(frames, cand_cfg)
            except Exception as exc:
                if self.verbose:
                    print(f"    [hill] error {var_name}: {exc}")
                continue
            if cand_n < 3:
                continue
            cand_score = cand_rank[0]
            delta = cand_score - baseline_score
            if cand_rank > best_rank:
                best_rank, best_cfg = cand_rank, cand_cfg
                best_meta = (var_name, old_val, new_val)
            elif -_MAX_ANNEAL_DROP < delta < 0:
                p = min(_MAX_ANNEAL_P, math.exp(delta / max(temperature, 1e-6)))
                wildcards.append((cand_cfg, cand_score, var_name, old_val, new_val, p))

        def meta(kind: str, var: str, old: Any, new: Any, new_score: float) -> dict[str, Any]:
            return {
                "kind": kind, "variable_changed": var, "old_value": old, "new_value": new,
                "baseline_score": round(baseline_score, 4), "new_score": round(new_score, 4),
                "trades_examined": baseline_n,
            }

        if best_meta is not None:
            var_name, old_val, new_val = best_meta
            if self.verbose:
                print(
                    f"    [hill] KEEP {var_name}: {old_val} → {new_val} "
                    f"| score {baseline_score:.4f} → {best_rank[0]:.4f} "
                    f"| pnl {baseline_rank[1]:+.1f}% → {best_rank[1]:+.1f}%"
                )
            return best_cfg, meta("keep", var_name, old_val, new_val, best_rank[0])

        if wildcards:
            wildcards.sort(key=lambda x: x[1], reverse=True)
            cand_cfg, cand_score, var_name, old_val, new_val, p = wildcards[0]
            if random.random() < p:
                if self.verbose:
                    print(
                        f"    [hill] ANNEAL {var_name}: {old_val} → {new_val} "
                        f"| score={cand_score:.4f} p={p:.3f}"
                    )
                return cand_cfg, meta("anneal", f"anneal:{var_name}", old_val, new_val, cand_score)

        if self.verbose:
            print("    [hill] no improvement — keeping config")
        return cfg, None

    def exploration_jump(self, cfg: Any, frames: Frames) -> Any:
        """Random restarts when the climb stagnates; keeps the best seed if it beats cfg."""
        if self.verbose:
            print("  [explore] stagnation jump — trying random seeds")
        best_cfg = cfg
        best_rank, _ = self.rank_frames(frames, cfg)
        for i in range(6):
            cand = self._pin(self._random_seed(cfg))
            if not self._valid(cand):
                continue
            rank, n = self.rank_frames(frames, cand)
            if n >= 5 and rank > best_rank:
                best_rank, best_cfg = rank, cand
                if self.verbose:
                    print(f"    seed#{i} score={rank[0]:.4f} pnl={rank[1]:+.1f}% trades={n}")
        return best_cfg

    def evolve(self, cfg: Any, frames_train: Frames, rounds: int) -> Any:
        """In-memory evolution on a train window (used by the walk-forward report)."""
        fwd = self._forward_slice(frames_train)
        saved = (self.evolution_round, self.stagnation_rounds)
        self.evolution_round = self.stagnation_rounds = 0
        try:
            for _ in range(rounds):
                self.evolution_round += 1
                if self.stagnation_rounds >= self.explore_every:
                    jumped = self.exploration_jump(cfg, fwd)
                    if jumped is not cfg:
                        cfg, self.stagnation_rounds = jumped, 0
                new_cfg, change = self.hill_climb(cfg, fwd)
                self.stagnation_rounds = 0 if change else self.stagnation_rounds + 1
                cfg = new_cfg
        finally:
            self.evolution_round, self.stagnation_rounds = saved
        return cfg

    # ── Reporting / persistence ────────────────────────────────────

    def _log_hypothesis(self, change: dict[str, Any]) -> None:
        hyp = {
            "ts": time.time(),
            "mode": self.MODE,
            "round": self.evolution_round,
            "version": self._current_version(),
            "improvement": round(change["new_score"] - change["baseline_score"], 4),
            **{k: v for k, v in change.items() if k != "kind"},
        }
        self._append_hypothesis(hyp)

    def _update_best_oos(self, cfg: Any, oos_pnl: float, trades: list[dict] | None = None) -> None:
        allowed, sig, evaluated = promotion_gate(trades or [])
        if evaluated and not sig.passed:
            print(
                f"  [best-oos] {'flagged' if allowed else 'blocked'} — "
                f"significance gate: {sig.summary()}"
            )
        if not allowed:
            return
        meta_path = self.state_dir / "best_oos_meta.json"
        prev = -1e18
        if meta_path.exists():
            try:
                prev = float(json.loads(meta_path.read_text()).get("oos_pnl_pct", prev))
            except Exception:
                pass
        if oos_pnl <= prev:
            return
        d = asdict(cfg)
        d["version"] = self._current_version()
        (self.state_dir / "best_oos.yaml").write_text(
            yaml.dump(d, default_flow_style=False, sort_keys=False)
        )
        meta_path.write_text(
            json.dumps(
                {
                    "oos_pnl_pct": round(oos_pnl, 4),
                    "evolution_round": self.evolution_round,
                    "version": d["version"],
                    "significant": sig.passed if evaluated else None,
                    "significance": sig.as_dict() if evaluated else None,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n"
        )
        if self.verbose:
            print(f"  [best-oos] new record: {oos_pnl:.2f}% OOS (v{d['version']})")

    def _check_goal(self, result: dict[str, Any], label: str, days: float) -> None:
        metrics = result["metrics"]
        target = self.goal.get("target_return_30d", 0.02) * 100
        max_dd = self.goal.get("max_drawdown", 0.10) * 100
        min_sharpe = self.goal.get("min_sharpe", 0.5)
        min_tpw = self.goal.get("min_trades_per_week", 2)
        est_30d = float(metrics.get("total_pnl_pct", 0)) / (days / 30) if days > 0 else 0.0
        dd = abs(float(metrics.get("max_drawdown_pct", 0)))
        sharpe = float(metrics.get("sharpe", 0) or 0)
        tpw = trades_per_week(result["trades"], days)

        def verdict(ok: bool) -> str:
            return "PASS" if ok else "FAIL"

        print(f"\n  Goal check ({label}, net of spread + slippage):")
        print(f"    Return/30d:  {est_30d:.2f}% (target: {target:.1f}%) {verdict(est_30d >= target)}")
        print(f"    Max DD:      {dd:.2f}% (limit: {max_dd:.0f}%) {verdict(dd <= max_dd)}")
        print(f"    Sharpe:      {sharpe:.3f} (min: {min_sharpe}) {verdict(sharpe >= min_sharpe)}")
        print(f"    Trades/week: {tpw:.2f} (min: {min_tpw}) {verdict(tpw >= min_tpw)}")

    def _save_trades(self, trades: list[dict]) -> None:
        with open(self.state_dir / "trades.jsonl", "w") as f:
            for t in trades:
                f.write(json.dumps(t, default=str) + "\n")

    # ── Data ───────────────────────────────────────────────────────

    async def load_frames(self) -> Frames:
        asset, _ = self._goal_asset()
        start = datetime(self.year_start, 1, 1, tzinfo=timezone.utc)
        print(f"\n  Loading {asset} {self.BASE_TIMEFRAME} (bid+ask) from {start.date()} ...")
        t0 = time.time()
        base = await load_bid_ask(asset, self.BASE_TIMEFRAME, start)
        print(f"  Loaded {len(base)} bars in {time.time() - t0:.0f}s")
        if base.empty:
            return {}
        if "ask_close" not in base.columns:
            print("  WARN: no ask side — using the fixed fallback spread")
        return {
            tf: base if tf == self.BASE_TIMEFRAME else resample_ohlc(base, tf)
            for tf in self.TIMEFRAMES
        }

    def split(self, frames: Frames) -> tuple[Frames, Frames, pd.Timestamp]:
        start, end = frames_span(frames)
        split_ts = start + (end - start) * (1 - self.oos_split)
        return slice_frames(frames, None, split_ts), slice_frames(frames, split_ts, None), split_ts

    # ── Main loop ──────────────────────────────────────────────────

    async def run_forever(self) -> None:
        if self._owns_global_state():
            pruned = reflect.prune_state_volume()
            if any(pruned.values()):
                print(f"  [volume] pruned {pruned}")

        frames = await self.load_frames()
        if not frames:
            print("  ERROR: no data")
            return
        train, oos, split_ts = self.split(frames)
        base_tf = self.TIMEFRAMES[0]
        train_days = period_days(train[base_tf])
        oos_days = period_days(oos[base_tf])
        print(
            f"  Train={len(train[base_tf])} | OOS={len(oos[base_tf])} {base_tf} bars "
            f"(split {split_ts})"
        )
        fwd = self._forward_slice(train)

        while True:
            self.evolution_round += 1
            if self.max_rounds is not None and self.evolution_round > self.max_rounds:
                print(f"  Reached max_rounds={self.max_rounds}")
                break

            print(f"\n{'=' * 60}")
            print(f"{self.MODE.upper()} ROUND #{self.evolution_round}")
            print(f"{'=' * 60}")

            cfg = self._load_strategy()
            round_start = self._config_snapshot(cfg)
            print(f"  asset={cfg.asset} {self._describe(cfg)}")

            if self.stagnation_rounds >= self.explore_every:
                jumped = self.exploration_jump(cfg, fwd)
                if jumped is not cfg:
                    cfg = jumped
                    self._save_strategy(cfg)
                    self.stagnation_rounds = 0

            if self.evolution_round == 1 or self.evolution_round % 3 == 0:
                print("\n  Running train backtest...")
                t1 = time.time()
                train_sc, _, train_res = self.score_frames(train, cfg)
                print(f"  Train done in {time.time() - t1:.1f}s | score={train_sc:.4f}")
                print_metrics(train_res["metrics"], label="IN-SAMPLE")
                self._check_goal(train_res, "In-sample", train_days)
                self._save_trades(train_res["trades"])
            else:
                print("\n  Skipping full train (every 3rd round) — hill-climb only")

            print("\n  Hill-climb...")
            new_cfg, change = self.hill_climb(cfg, fwd)
            if change is not None:
                self._log_hypothesis(change)
                self._save_strategy(new_cfg)

            evolved = self._load_strategy()
            print("\n  Running OOS backtest...")
            oos_sc, _, oos_res = self.score_frames(oos, evolved)
            print(f"  OOS score={oos_sc:.4f}")
            print_metrics(oos_res["metrics"], label="OUT-OF-SAMPLE")
            self._check_goal(oos_res, "OOS", oos_days)
            self._update_best_oos(
                evolved,
                float(oos_res["metrics"].get("total_pnl_pct", 0)),
                oos_res.get("trades"),
            )

            if self._config_snapshot(evolved) != round_start:
                self.stagnation_rounds = 0
                print("  Config changed this round")
            else:
                self.stagnation_rounds += 1
                print(f"  Stagnation: {self.stagnation_rounds} rounds")

            print("  Starting next round...")


def env_int(name: str, default: int) -> int:
    return int(env(name, str(default)))
