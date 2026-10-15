"""Evolution worker for Nasdaq Opening Range Breakout (5m).

Round-based hill-climb (like SMC): train score → mutate one param → OOS validate.
Fitness uses net-of-cost trades (default 2pt round-trip) so evolution does not
chase zero-cost illusions.
"""
from __future__ import annotations
from .env import env

import asyncio
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

from .adapters.market_data import get_market_adapter
from .backtest import print_metrics
from .reflect import (
    STATE_DIR,
    append_hypothesis,
    load_yaml,
    next_version,
    prune_state_volume,
    save_prior,
)
from .score import score, trades_per_week
from .significance import promotion_gate
from .strategies.opening_range_breakout import (
    OpeningRangeBreakoutConfig,
    run_backtest,
)

_MAX_ANNEAL_DROP = 0.05
_MAX_ANNEAL_P = 0.25

_BOOL_FLIP = ("long_only",)
_CUTOFF_CHOICES = ("10:00", "10:30", "11:00", "11:30", "12:00")


def _period_days(df: pd.DataFrame, bar_minutes: int = 5) -> float:
    if df.empty:
        return 1.0
    return max(len(df) * bar_minutes / (60 * 24), 1.0)


def _apply_rt_cost(
    trades: list[dict],
    rt_points: float,
    risk_pct: float = 0.01,
) -> list[dict]:
    """Haircut account pnl_pct by round-trip points / stop distance."""
    if rt_points <= 0:
        return trades
    out: list[dict] = []
    for t in trades:
        t2 = dict(t)
        if t2.get("status") != "closed":
            out.append(t2)
            continue
        entry = float(t2.get("entry_price") or 0)
        sl = float(t2.get("stop_loss") or 0)
        dist = abs(entry - sl)
        meta = t2.get("meta") or {}
        if dist <= 0:
            dist = float(meta.get("orb_width") or 0) + 1.0
        if dist <= 0:
            out.append(t2)
            continue
        haircut = (rt_points / dist) * risk_pct * 100.0
        if t2.get("pnl_pct") is not None:
            t2["pnl_pct"] = float(t2["pnl_pct"]) - haircut
        out.append(t2)
    return out


def _load_orb_config(path: Path) -> OpeningRangeBreakoutConfig:
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    raw = raw or {}
    valid = {f.name for f in fields(OpeningRangeBreakoutConfig)}
    kwargs = {k: v for k, v in raw.items() if k in valid}
    return OpeningRangeBreakoutConfig(**kwargs)


def _save_orb_yaml(path: Path, cfg: OpeningRangeBreakoutConfig, version: str) -> None:
    d = asdict(cfg)
    d["version"] = version
    path.write_text(yaml.dump(d, default_flow_style=False, sort_keys=False))


class OrbEvolutionWorker:
    """ORB evolution on NQ/NAS100 5m bars."""

    def __init__(
        self,
        state_dir: Path | None = None,
        *,
        year_start: int = 2020,
        oos_split: float = 0.30,
        num_candidates: int = 12,
        forward_eval_days: int = 120,
        explore_every: int = 15,
        cost_rt_points: float | None = None,
        max_rounds: int | None = None,
        verbose: bool = True,
    ) -> None:
        self.state_dir = state_dir or STATE_DIR
        self.year_start = year_start
        self.oos_split = oos_split
        self.num_candidates = num_candidates
        self.forward_eval_days = forward_eval_days
        self.explore_every = explore_every
        if cost_rt_points is None:
            cost_rt_points = float(env("ORB_COST_RT_POINTS", "2.0"))
        self.cost_rt_points = cost_rt_points
        self.max_rounds = max_rounds
        self.verbose = verbose
        self.adapter = get_market_adapter(
            env("LAB_DATA_SOURCE", "dukascopy")
        )
        self.goal = load_yaml(self.state_dir / "goal.yaml")
        self.evolution_round = 0
        self.stagnation_rounds = 0

    def _goal_asset(self) -> tuple[str, str]:
        asset = str(self.goal.get("asset", "NQ"))
        label = str(self.goal.get("asset_label", asset))
        return asset, label

    def _pin(self, cfg: OpeningRangeBreakoutConfig) -> OpeningRangeBreakoutConfig:
        asset, label = self._goal_asset()
        return replace(cfg, asset=asset, asset_label=label, bar_minutes=5)

    def _load_strategy(self) -> OpeningRangeBreakoutConfig:
        path = self.state_dir / "strategy.yaml"
        if not path.exists():
            baseline = self.state_dir / "orb_baseline.yaml"
            if baseline.exists():
                return self._pin(_load_orb_config(baseline))
            return self._pin(OpeningRangeBreakoutConfig())
        return self._pin(_load_orb_config(path))

    def _current_version(self) -> str:
        path = self.state_dir / "strategy.yaml"
        if not path.exists():
            return "orb_01"
        raw = yaml.safe_load(path.read_text()) or {}
        return str(raw.get("version", "orb_01"))

    def _save_strategy(self, cfg: OpeningRangeBreakoutConfig, *, increment: bool) -> None:
        path = self.state_dir / "strategy.yaml"
        ver = self._current_version()
        if path.exists():
            save_prior(yaml.safe_load(path.read_text()) or {})
        if increment:
            # numeric tail if present, else bump suffix
            try:
                if ver.startswith("orb_"):
                    tail = ver[4:]
                    if tail.isdigit():
                        ver = f"orb_{int(tail) + 1:02d}"
                    else:
                        ver = next_version("01")
                        ver = f"orb_{ver}"
                else:
                    ver = next_version(ver)
            except Exception:
                ver = f"orb_{int(time.time()) % 100000}"
        _save_orb_yaml(path, self._pin(cfg), ver)
        if self.verbose:
            print(f"  [strategy] saved {ver}")

    @staticmethod
    def _config_snapshot(cfg: OpeningRangeBreakoutConfig) -> tuple:
        d = asdict(cfg)
        d.pop("version", None)
        return tuple(sorted(d.items()))

    def _net_trades(self, trades: list[dict]) -> list[dict]:
        closed = [t for t in trades if t.get("status") == "closed"]
        return _apply_rt_cost(closed, self.cost_rt_points)

    def _score_df(
        self, df: pd.DataFrame, cfg: OpeningRangeBreakoutConfig
    ) -> tuple[float, int, dict[str, Any]]:
        result = run_backtest(df, cfg, verbose=False)
        net = self._net_trades(result["trades"])
        days = _period_days(df, cfg.bar_minutes)
        sc = score(net, self.goal, period_days=days)
        # Patch metrics with net pnl for display / best_oos
        if net:
            result = dict(result)
            m = dict(result.get("metrics") or {})
            m["total_pnl_pct_net"] = round(sum(t.get("pnl_pct") or 0 for t in net), 4)
            m["cost_rt_points"] = self.cost_rt_points
            result["metrics"] = m
            result["trades_net"] = net
        return sc, len(net), result

    def _mutate_one(
        self, cfg: OpeningRangeBreakoutConfig
    ) -> tuple[OpeningRangeBreakoutConfig, str, Any, Any] | None:
        space = list(OpeningRangeBreakoutConfig.search_space())
        roll = random.random()
        if roll < 0.15:
            var = "long_only"
            old = cfg.long_only
            new = not old
            return replace(cfg, long_only=new, short_only=False if new else cfg.short_only), var, old, new
        if roll < 0.25:
            var = "entry_cutoff"
            old = cfg.entry_cutoff
            choices = [c for c in _CUTOFF_CHOICES if c != old]
            new = random.choice(choices)
            return replace(cfg, entry_cutoff=new), var, old, new

        name, lo, hi, step = random.choice(space)
        old = getattr(cfg, name)
        # discrete steps
        if isinstance(old, int) or name in ("orb_minutes",):
            steps = list(range(int(lo), int(hi) + 1, int(step)))
            steps = [s for s in steps if s != int(old)]
            if not steps:
                return None
            new = random.choice(steps)
            return replace(cfg, **{name: int(new)}), name, old, new

        direction = random.choice([-1, 1])
        new = float(old) + direction * float(step)
        new = max(float(lo), min(float(hi), new))
        # snap to step grid
        grid = round((new - float(lo)) / float(step)) * float(step) + float(lo)
        grid = max(float(lo), min(float(hi), round(grid, 6)))
        if grid == float(old):
            grid = float(old) + direction * float(step)
            grid = max(float(lo), min(float(hi), round(grid, 6)))
        if grid == float(old):
            return None
        return replace(cfg, **{name: grid}), name, old, grid

    def _random_seed(self) -> OpeningRangeBreakoutConfig:
        base = self._load_strategy()
        return replace(
            base,
            orb_minutes=random.choice([10, 15, 20, 30]),
            tp_range_mult=random.choice([0.5, 0.75, 1.0, 1.25, 1.5]),
            max_orb_pct=random.choice([0.006, 0.008, 0.01, 0.012]),
            sl_buffer=random.choice([0.0, 1.0, 2.0]),
            breakout_buffer=random.choice([0.0, 1.0, 2.0]),
            long_only=random.choice([True, False]),
            entry_cutoff=random.choice(_CUTOFF_CHOICES),
            short_only=False,
        )

    def _exploration_jump(self, df_train: pd.DataFrame, cfg: OpeningRangeBreakoutConfig) -> OpeningRangeBreakoutConfig:
        if self.verbose:
            print("  [explore] stagnation jump — trying random ORB seeds")
        best_cfg = cfg
        best_sc, _, _ = self._score_df(df_train.iloc[int(len(df_train) * 0.7) :], cfg)
        for i in range(6):
            cand = self._pin(self._random_seed())
            sc, n, _ = self._score_df(df_train.iloc[int(len(df_train) * 0.7) :], cand)
            if n < 5:
                continue
            if sc > best_sc:
                best_sc = sc
                best_cfg = cand
                if self.verbose:
                    print(f"    seed#{i} score={sc:.4f} trades={n}")
        if best_cfg is not cfg:
            self._save_strategy(best_cfg, increment=True)
            self.stagnation_rounds = 0
        return best_cfg

    def _hill_climb(self, cfg: OpeningRangeBreakoutConfig, df_train: pd.DataFrame) -> OpeningRangeBreakoutConfig:
        # Forward slice ≈ last forward_eval_days of train
        bars = max(int(self.forward_eval_days * 24 * 60 / 5), 500)
        if len(df_train) <= bars + 100:
            fwd = df_train.copy()
        else:
            fwd = df_train.iloc[-bars:].copy()

        baseline_score, baseline_n, _ = self._score_df(fwd, cfg)
        if baseline_n < 3:
            if self.verbose:
                print(f"    [hill] skip — baseline trades={baseline_n}")
            return cfg

        temperature = max(0.08, 0.5 * (0.98 ** (self.evolution_round % 100)))
        if self.stagnation_rounds >= 5:
            temperature = max(temperature, 0.25)

        if self.verbose:
            print(
                f"    [hill] forward baseline={baseline_score:.4f} "
                f"trades={baseline_n} temp={temperature:.3f} bars={len(fwd)} "
                f"cost_rt={self.cost_rt_points}pt"
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
                cand_score, cand_n, _ = self._score_df(fwd, cand_cfg)
            except Exception as exc:
                if self.verbose:
                    print(f"    [hill] error {var_name}: {exc}")
                continue
            if cand_n < 3:
                continue
            delta = cand_score - baseline_score
            if cand_score > best_score:
                best_score = cand_score
                best_cfg = cand_cfg
                best_meta = (var_name, old_val, new_val)
            elif delta > -_MAX_ANNEAL_DROP:
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
                    baseline_score,
                    cand_score,
                    f"anneal:{var_name}",
                    old_val,
                    new_val,
                    baseline_n,
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
        hyp = {
            "ts": time.time(),
            "mode": "orb_evolution",
            "round": self.evolution_round,
            "version": self._current_version(),
            "baseline_score": round(old_score, 4),
            "new_score": round(new_score, 4),
            "improvement": round(new_score - old_score, 4),
            "trades_examined": trades_n,
            "variable_changed": var,
            "old_value": old_val,
            "new_value": new_val,
            "cost_rt_points": self.cost_rt_points,
        }
        try:
            append_hypothesis(hyp)
        except Exception:
            path = self.state_dir / "hypotheses.jsonl"
            with open(path, "a") as f:
                f.write(json.dumps(hyp, default=str) + "\n")

    def _update_best_oos(
        self, cfg: OpeningRangeBreakoutConfig, oos_pnl_net: float, trades: list[dict] | None = None
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
        if oos_pnl_net <= prev:
            return
        d = asdict(cfg)
        d["version"] = self._current_version()
        strat_path.write_text(yaml.dump(d, default_flow_style=False, sort_keys=False))
        meta_path.write_text(
            json.dumps(
                {
                    "oos_pnl_pct": round(oos_pnl_net, 4),
                    "evolution_round": self.evolution_round,
                    "version": d["version"],
                    "cost_rt_points": self.cost_rt_points,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n"
        )
        if self.verbose:
            print(f"  [best-oos] new record: {oos_pnl_net:.2f}% OOS net (v{d['version']})")

    def _print_tpw(self, trades: list[dict], label: str, days: float) -> None:
        tpw = trades_per_week(trades, days)
        print(f"  {label} trades/week: {tpw:.2f}")

    def _check_goal(self, metrics: dict, label: str, days: float) -> None:
        target = self.goal.get("target_return_30d", 0.02) * 100
        max_dd = self.goal.get("max_drawdown", 0.10) * 100
        min_sharpe = self.goal.get("min_sharpe", 0.5)
        total = float(metrics.get("total_pnl_pct_net", metrics.get("total_pnl_pct", 0)))
        est_30d = total / (days / 30) if days > 0 else 0
        dd = abs(float(metrics.get("max_drawdown_pct", 0)))
        sharpe = float(metrics.get("sharpe", 0) or 0)
        print(f"\n  Goal check ({label}, net of {self.cost_rt_points}pt):")
        print(
            f"    Return/30d:  {est_30d:.2f}% (target: {target:.0f}%) "
            f"{'PASS' if est_30d >= target else 'FAIL'}"
        )
        print(
            f"    Max DD:      {dd:.2f}% (limit: {max_dd:.0f}%) "
            f"{'PASS' if dd <= max_dd else 'FAIL'}"
        )
        print(
            f"    Sharpe:      {sharpe:.3f} (min: {min_sharpe}) "
            f"{'PASS' if sharpe >= min_sharpe else 'FAIL'}"
        )

    def _save_trades(self, trades: list[dict]) -> None:
        path = self.state_dir / "trades.jsonl"
        with open(path, "w") as f:
            for t in trades:
                f.write(json.dumps(t, default=str) + "\n")

    async def run_forever(self) -> None:
        pruned = prune_state_volume()
        if any(pruned.values()):
            print(
                f"  [volume] pruned history={pruned.get('history_removed', 0)} "
                f"hypotheses_lines={pruned.get('hypotheses_lines_removed', 0)} "
                f"trades_lines={pruned.get('trades_lines_removed', 0)}"
            )

        asset, _ = self._goal_asset()
        start = datetime(self.year_start, 1, 1, tzinfo=timezone.utc)
        print(f"\n  Loading {asset} 5m from {start.date()} ...")
        t0 = time.time()
        df = await self.adapter.fetch_history(asset, interval="5m", start=start)
        print(f"  Loaded {len(df)} bars in {time.time() - t0:.0f}s")
        if df.empty:
            print("  ERROR: no data")
            return

        split_i = int(len(df) * (1 - self.oos_split))
        df_train = df.iloc[:split_i].copy()
        df_oos = df.iloc[split_i:].copy()
        print(
            f"  Train={len(df_train)} | OOS={len(df_oos)} "
            f"(split {df_oos.index[0]})"
        )
        print(f"  Cost model: {self.cost_rt_points}pt round-trip")

        while True:
            self.evolution_round += 1
            if self.max_rounds is not None and self.evolution_round > self.max_rounds:
                print(f"  Reached max_rounds={self.max_rounds}")
                break

            print(f"\n{'=' * 60}")
            print(f"ORB EVOLUTION ROUND #{self.evolution_round}")
            print(f"{'=' * 60}")

            cfg = self._load_strategy()
            round_start = self._config_snapshot(cfg)
            print(
                f"  asset={cfg.asset} orb={cfg.orb_minutes}m tp={cfg.tp_range_mult}R "
                f"max_orb={cfg.max_orb_pct} long_only={cfg.long_only} "
                f"cutoff={cfg.entry_cutoff}"
            )

            if self.stagnation_rounds >= self.explore_every:
                cfg = self._exploration_jump(df_train, cfg)

            do_full = self.evolution_round == 1 or self.evolution_round % 3 == 0
            if do_full:
                print("\n  Running train backtest (net)...")
                t1 = time.time()
                train_sc, train_n, train_res = self._score_df(df_train, cfg)
                print(f"  Train done in {time.time() - t1:.1f}s | score={train_sc:.4f}")
                # Show gross engine metrics + net pnl
                print_metrics(train_res["metrics"], label="IN-SAMPLE (gross engine)")
                net_pnl = train_res["metrics"].get("total_pnl_pct_net")
                print(f"  Net PnL ({self.cost_rt_points}pt): {net_pnl}% | trades={train_n}")
                self._print_tpw(
                    train_res.get("trades_net") or [],
                    "In-sample",
                    _period_days(df_train),
                )
                self._check_goal(
                    train_res["metrics"], "In-sample", _period_days(df_train)
                )
                self._save_trades(train_res.get("trades") or [])
            else:
                print("\n  Skipping full train (every 3rd round) — hill-climb only")

            print("\n  Hill-climb...")
            cfg = self._hill_climb(cfg, df_train)

            evolved = self._load_strategy()
            print("\n  Running OOS backtest (net)...")
            t2 = time.time()
            oos_sc, oos_n, oos_res = self._score_df(df_oos, evolved)
            print(f"  OOS done in {time.time() - t2:.1f}s | score={oos_sc:.4f}")
            print_metrics(oos_res["metrics"], label="OUT-OF-SAMPLE (gross engine)")
            net_oos = float(oos_res["metrics"].get("total_pnl_pct_net") or 0)
            print(f"  Net PnL ({self.cost_rt_points}pt): {net_oos}% | trades={oos_n}")
            self._print_tpw(
                oos_res.get("trades_net") or [], "OOS", _period_days(df_oos)
            )
            self._check_goal(oos_res["metrics"], "OOS", _period_days(df_oos))
            self._update_best_oos(
                evolved, net_oos, oos_res.get("trades_net") or oos_res.get("trades")
            )

            if self._config_snapshot(evolved) != round_start:
                self.stagnation_rounds = 0
                print("  Config changed this round")
            else:
                self.stagnation_rounds += 1
                print(f"  Stagnation: {self.stagnation_rounds} rounds")

            print("  Starting next round...")


def main() -> None:
    worker = OrbEvolutionWorker(
        year_start=int(env("ORB_EVOLUTION_YEAR_START", "2020")),
        num_candidates=int(env("ORB_EVOLUTION_CANDIDATES", "12")),
        forward_eval_days=int(env("ORB_EVOLUTION_FORWARD_DAYS", "120")),
        cost_rt_points=float(env("ORB_COST_RT_POINTS", "2.0")),
        verbose=True,
    )
    print("ORB Evolution Worker")
    print(f"  cost_rt_points: {worker.cost_rt_points}")
    asyncio.run(worker.run_forever())


if __name__ == "__main__":
    main()
