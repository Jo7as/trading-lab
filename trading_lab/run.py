"""Entrypoint for the Trading Lab worker (MCTFB strategy).

Reads config from /app/state/goal.yaml and /app/state/strategy.yaml
and starts the async trading loop.

Data source: Dukascopy (XAU/USD spot) by default for signal worker.
Default asset: Gold (XAU/USD)
"""
from __future__ import annotations
from .env import env

import argparse
import asyncio
import os
from pathlib import Path

import yaml

from .loop import TradingLoop
from .config import load_config_from_yaml

STATE_DIR = Path(env("LAB_STATE", "/app/state"))
if not STATE_DIR.exists():
    STATE_DIR = Path.home() / "trading-lab" / "state"


def load_goal() -> dict:
    goal_path = STATE_DIR / "goal.yaml"
    with open(goal_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading Lab MCTFB worker")
    parser.add_argument("--asset", default=None, help="Override asset from strategy.yaml")
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Loop interval in seconds (default 300 = 5min)",
    )
    args = parser.parse_args()

    mode = env("LAB_MODE", "paper")
    worker_mode = env("LAB_WORKER", "evolution")

    print(f"Booting trading-lab worker")
    print(f"  mode:     {mode}")
    print(f"  state:    {STATE_DIR}")
    print(f"  worker:   {worker_mode}")

    if worker_mode == "signal":
        from .signal_loop import SignalLoop
        from .notify.telegram import TelegramNotifier

        interval = int(
            env("SIGNAL_MANAGE_INTERVAL_SECONDS")
            or env("SIGNAL_INTERVAL_SECONDS", "60")
        )
        hour_aligned = env("SIGNAL_HOUR_ALIGNED", "1").lower() in (
            "1",
            "true",
            "yes",
        )
        entry_offset = int(
            env("SIGNAL_ENTRY_POLL_OFFSET_SECONDS", "60")
        )
        entry_retry = int(
            env("SIGNAL_ENTRY_RETRY_INTERVAL_SECONDS", "20")
        )
        entry_retry_window = int(
            env("SIGNAL_ENTRY_RETRY_WINDOW_SECONDS", "900")
        )
        if hour_aligned:
            print(
                f"  entry_poll: UTC :00+{entry_offset}s "
                f"(retry every {entry_retry}s for {entry_retry_window}s)"
            )
        data_src = env("LAB_DATA_SOURCE", "dukascopy")
        print(f"  data_source: {data_src}")

        def _make_loop(
            strategy_path: Path,
            *,
            state_suffix: str = "",
            log_tag: str = "signal",
        ) -> SignalLoop:
            return SignalLoop(
                state_dir=STATE_DIR,
                strategy_path=strategy_path,
                interval_seconds=interval,
                hour_aligned=hour_aligned,
                entry_poll_offset_seconds=entry_offset,
                entry_retry_interval_seconds=entry_retry,
                entry_retry_window_seconds=entry_retry_window,
                mode=mode,
                notifier=TelegramNotifier(),
                state_suffix=state_suffix,
                log_tag=log_tag,
            )

        async def _run_signals() -> None:
            assets = {
                a.strip().lower()
                for a in env("SIGNAL_ASSETS", "gold").split(",")
                if a.strip()
            }
            tasks = []

            gold_path = Path(
                env(
                    "LAB_STRATEGY_PATH",
                    str(STATE_DIR / "hf_baseline.yaml"),
                )
            )
            print(f"  gold strategy: {gold_path}")
            tasks.append(_make_loop(gold_path, log_tag="signal:gold").run_forever())

            if "jpy" in assets:
                jpy_path = Path(
                    env(
                        "SIGNAL_JPY_STRATEGY_PATH",
                        str(STATE_DIR / "jpy_strategy.yaml"),
                    )
                )
                if jpy_path.exists():
                    print(f"  jpy strategy: {jpy_path}")
                    tasks.append(
                        _make_loop(
                            jpy_path,
                            state_suffix="_jpy",
                            log_tag="signal:jpy",
                        ).run_forever()
                    )
                else:
                    print(
                        f"  jpy skipped: {jpy_path} not found "
                        "(enable after JPY evolution winner)"
                    )

            await asyncio.gather(*tasks)

        asyncio.run(_run_signals())
        return

    if worker_mode == "smc_evolution":
        from .smc_evolution_worker import SmcEvolutionWorker

        smc_state = Path(
            env(
                "LAB_STATE",
                str(Path(__file__).resolve().parent.parent / "smc-state"),
            )
        )
        goal_path = smc_state / "goal.yaml"
        goal = yaml.safe_load(goal_path.read_text()) if goal_path.exists() else {}
        asset = args.asset or goal.get("asset", "EUR/USD")
        print(f"  asset:     {asset}")
        print(f"  smc state: {smc_state}")
        worker = SmcEvolutionWorker(
            state_dir=smc_state,
            year_start=int(env("SMC_EVOLUTION_YEAR_START", "2020")),
            year_end=(
                int(os.environ["SMC_EVOLUTION_YEAR_END"])
                if env("SMC_EVOLUTION_YEAR_END")
                else None
            ),
            num_candidates=int(
                env(
                    "SMC_EVOLUTION_CANDIDATES",
                    str(goal.get("candidates_per_round", 10)),
                )
            ),
            forward_eval_days=int(
                env(
                    "SMC_EVOLUTION_FORWARD_DAYS",
                    str(goal.get("forward_eval_days", 90)),
                )
            ),
            max_rounds=(
                int(os.environ["SMC_EVOLUTION_ROUNDS"])
                if env("SMC_EVOLUTION_ROUNDS")
                else None
            ),
            verbose=True,
        )
        asyncio.run(worker.run_forever())
        return

    if worker_mode == "orb_evolution":
        from .orb_evolution_worker import OrbEvolutionWorker

        orb_state = Path(
            env(
                "LAB_STATE",
                str(Path(__file__).resolve().parent.parent / "orb-state"),
            )
        )
        goal_path = orb_state / "goal.yaml"
        goal = yaml.safe_load(goal_path.read_text()) if goal_path.exists() else {}
        asset = args.asset or goal.get("asset", "NQ")
        cost_rt = float(env("ORB_COST_RT_POINTS", "2.0"))
        print(f"  asset:     {asset}")
        print(f"  orb state: {orb_state}")
        print(f"  cost RT:   {cost_rt}pt")
        worker = OrbEvolutionWorker(
            state_dir=orb_state,
            year_start=int(env("ORB_EVOLUTION_YEAR_START", "2020")),
            num_candidates=int(
                env(
                    "ORB_EVOLUTION_CANDIDATES",
                    str(goal.get("candidates_per_round", 12)),
                )
            ),
            forward_eval_days=int(
                env(
                    "ORB_EVOLUTION_FORWARD_DAYS",
                    str(goal.get("forward_eval_days", 120)),
                )
            ),
            cost_rt_points=cost_rt,
            max_rounds=(
                int(os.environ["ORB_EVOLUTION_ROUNDS"])
                if env("ORB_EVOLUTION_ROUNDS")
                else None
            ),
            verbose=True,
        )
        asyncio.run(worker.run_forever())
        return

    if worker_mode == "session_range_evolution":
        from .session_range_evolution_worker import SessionRangeEvolutionWorker

        srf_state = Path(
            env(
                "LAB_STATE",
                str(Path(__file__).resolve().parent.parent / "session-range-state"),
            )
        )
        goal_path = srf_state / "goal.yaml"
        goal = yaml.safe_load(goal_path.read_text()) if goal_path.exists() else {}
        print(f"  asset:     {goal.get('asset', 'XAU/USD')}")
        print(f"  srf state: {srf_state}")
        worker = SessionRangeEvolutionWorker(
            state_dir=srf_state,
            year_start=int(env("SRF_EVOLUTION_YEAR_START", "2020")),
            num_candidates=int(
                env(
                    "SRF_EVOLUTION_CANDIDATES",
                    str(goal.get("candidates_per_round", 12)),
                )
            ),
            forward_eval_days=int(
                env(
                    "SRF_EVOLUTION_FORWARD_DAYS",
                    str(goal.get("forward_eval_days", 365)),
                )
            ),
            max_rounds=(
                int(os.environ["SRF_EVOLUTION_ROUNDS"])
                if env("SRF_EVOLUTION_ROUNDS")
                else None
            ),
            verbose=True,
        )
        asyncio.run(worker.run_forever())
        return

    if worker_mode == "mr_evolution":
        from .mean_reversion_evolution_worker import MeanReversionEvolutionWorker

        mr_state = Path(
            env(
                "LAB_STATE",
                str(Path(__file__).resolve().parent.parent / "mean-reversion-state"),
            )
        )
        goal_path = mr_state / "goal.yaml"
        goal = yaml.safe_load(goal_path.read_text()) if goal_path.exists() else {}
        # One asset per service: Gold from goal.yaml, USD/JPY via override
        asset = args.asset or env("MR_EVOLUTION_ASSET") or None
        print(f"  asset:     {asset or goal.get('asset', 'GC=F')}")
        print(f"  mr state:  {mr_state}")
        worker = MeanReversionEvolutionWorker(
            state_dir=mr_state,
            year_start=int(env("MR_EVOLUTION_YEAR_START", "2020")),
            num_candidates=int(
                env(
                    "MR_EVOLUTION_CANDIDATES",
                    str(goal.get("candidates_per_round", 12)),
                )
            ),
            forward_eval_days=int(
                env(
                    "MR_EVOLUTION_FORWARD_DAYS",
                    str(goal.get("forward_eval_days", 365)),
                )
            ),
            max_rounds=(
                int(os.environ["MR_EVOLUTION_ROUNDS"])
                if env("MR_EVOLUTION_ROUNDS")
                else None
            ),
            asset=asset,
            verbose=True,
        )
        asyncio.run(worker.run_forever())
        return

    if worker_mode == "orb2r_evolution":
        from .orb2r_evolution_worker import Orb2rEvolutionWorker

        orb2r_state = Path(
            env(
                "LAB_STATE",
                str(Path(__file__).resolve().parent.parent / "orb2r-state"),
            )
        )
        goal_path = orb2r_state / "goal.yaml"
        goal = yaml.safe_load(goal_path.read_text()) if goal_path.exists() else {}
        # One market per service: US100 from goal.yaml, GER40 / XAUUSD via override
        asset = args.asset or env("ORB2R_EVOLUTION_ASSET") or None
        print(f"  asset:       {asset or goal.get('asset', 'US100')}")
        print(f"  orb2r state: {orb2r_state}")
        worker = Orb2rEvolutionWorker(
            state_dir=orb2r_state,
            year_start=int(env("ORB2R_EVOLUTION_YEAR_START", "2016")),
            num_candidates=int(
                env(
                    "ORB2R_EVOLUTION_CANDIDATES",
                    str(goal.get("candidates_per_round", 12)),
                )
            ),
            forward_eval_days=int(
                env(
                    "ORB2R_EVOLUTION_FORWARD_DAYS",
                    str(goal.get("forward_eval_days", 730)),
                )
            ),
            max_rounds=(
                int(os.environ["ORB2R_EVOLUTION_ROUNDS"])
                if env("ORB2R_EVOLUTION_ROUNDS")
                else None
            ),
            asset=asset,
            verbose=True,
        )
        asyncio.run(worker.run_forever())
        return

    goal = load_goal()
    strategy_path = STATE_DIR / "strategy.yaml"
    cfg = load_config_from_yaml(strategy_path) if strategy_path.exists() else None
    asset = args.asset or (cfg.asset if cfg else goal.get("asset", "GC=F"))

    print(f"  asset:    {asset}")
    if worker_mode == "evolution":
        hf_flag = env("LAB_FORCE_HF", "")
        pause_env = env("LAB_NO_REFLECT_ROUNDS", "")
        baseline_env = env("LAB_BASELINE_PATH", "")
        tpl_env = env("LAB_STATE_TEMPLATE", "")
        print(f"  LAB_FORCE_HF: {hf_flag or 'off'}")
        print(f"  LAB_NO_REFLECT_ROUNDS: {pause_env or 'off'}")
        if baseline_env:
            print(f"  LAB_BASELINE_PATH: {baseline_env}")
        if tpl_env:
            print(f"  LAB_STATE_TEMPLATE: {tpl_env}")
    if worker_mode == "evolution":
        # Supertrend+Volume evolution worker
        from .st_evolution_worker import STEvolutionWorker
        worker = STEvolutionWorker(
            state_dir=STATE_DIR,
            period=env("EVOLUTION_PERIOD", "730d"),
            interval=env("EVOLUTION_INTERVAL", "1h"),
            reflect_interval=int(goal.get("reflection_every", 15)),
            verbose=True,
        )
        asyncio.run(worker.run_forever())
    else:
        # Live trading worker
        print(f"  interval: {args.interval}s")
        loop = TradingLoop(
            asset=asset,
            goal=goal,
            state_dir=STATE_DIR,
            interval_seconds=args.interval,
            mode=mode,
        )
        asyncio.run(loop.run_forever())


if __name__ == "__main__":
    main()
