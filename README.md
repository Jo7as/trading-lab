# trading-lab

A research lab for **testing, backtesting and evolving intraday trading strategies**.
It is a paper-trading / signal system — no live broker execution.

The loop is simple:

1. **Strategies** live in `trading_lab/strategies/` and plug into a shared engine.
2. **Backtests** and exploratory studies run them over historical 1-minute
   bid+ask data (Dukascopy) or daily/hourly bars (yfinance).
3. **Evolution workers** search the parameter space (hill-climb + simulated
   annealing) with a walk-forward out-of-sample holdout, and persist a
   versioned state on disk. A worker can be deployed as a container (Railway).
4. A **signal worker** replays the current best strategy on closed bars and
   posts signals (Telegram).

## Layout

```
trading_lab/          library + workers (the code you import)
  run.py              entrypoint; picks a worker from LAB_WORKER
  config.py indicators.py signals.py risk.py score.py backtest.py
  minute_data.py      Dukascopy 1m bid+ask loader with per-year cache
  mr_common.py        shared helpers (bid+ask frames, sizing, exits)
  loop.py             live paper loop
  signal_loop.py      signal worker (Telegram)
  reflect.py exploration.py candidate_leaderboard.py iterative_backtest.py
  mr_evolution_base.py
  *_evolution_worker.py   one worker per strategy
  adapters/           data layer (dukascopy, yfinance, market_data, …)
  notify/             Telegram
  strategies/         one module per strategy (+ strategies/smc/)
backtests/            runnable backtest / study runners (use the library)
studies/              exploratory measurement harnesses
configs/              strategy presets: baselines/ and evolved/
deploy/state-templates/  seed state for fresh deployments (Railway volumes)
docs/                 strategy specs, research notes, ADD_STRATEGY.md, DEPLOY_RAILWAY.md
AGENTS.md             onboarding guide for AI assistants in your IDE
scripts/              ops helpers (monitoring, inspections)
mt5/                  MetaTrader 5 demo EA
tests/                pytest suite
```

## Install

```bash
uv sync                 # or: pip install -e .
```

## Quickstart

```bash
# Backtest an opening-range breakout config over 1m bid+ask data
python backtests/backtest_orb_2r.py --markets US100

# Run the pre-registered day-filter study
python backtests/backtest_orb_2r_filters.py --markets US100,GER40

# A stage-1 measurement harness (no trading): US100 NY midday reversal
python studies/research_c1_midday_reversal.py
```

Backtests and studies write their reports to `artifacts/` (git-ignored).

## Add your own strategy

A new idea is one module + one runner + one test. Hand this to your AI assistant
in the IDE along with **[docs/ADD_STRATEGY.md](docs/ADD_STRATEGY.md)**:
add `trading_lab/strategies/my_idea.py` (a config dataclass + a
`run_backtest(df, cfg) -> dict`), a runner in `backtests/`, and a test in
`tests/`. To run it 24/7, plug it into an evolution worker (Level 3 in the doc).

## Starting a worker

The entrypoint dispatches on `LAB_WORKER`:

```bash
LAB_WORKER=evolution       LAB_STATE=state python -m trading_lab.run
LAB_WORKER=orb2r_evolution LAB_STATE=orb2r-state ORB2R_EVOLUTION_ASSET=JP225 \
    python -m trading_lab.run
LAB_WORKER=signal          LAB_STATE=live-state python -m trading_lab.run
```

Per-strategy workers and their knobs (e.g. `ORB2R_EVOLUTION_*`,
`SMC_EVOLUTION_*`, `MR_EVOLUTION_*`) are read from the environment. See
`trading_lab/run.py`.

## Configuration

Copy `.env.example` to `.env` and adjust. Canonical variables use the `LAB_`
prefix (`.env.example` lists them); the historical `HERMES_*` names are still
accepted as a fallback (`trading_lab/env.py`).

## Deploy (optional)

Run the container locally:

```bash
docker build -t trading-lab .
docker run --env-file .env trading-lab
```

Or deploy to **Railway** to run an always-on evolution worker and/or a Telegram
signal worker — the repo ships a ready-made setup (`Dockerfile`,
`railway.toml`, `deploy/state-templates/`). One repo serves all worker types;
you only change `LAB_WORKER` and the volume per service. Full walkthrough:
**[docs/DEPLOY_RAILWAY.md](docs/DEPLOY_RAILWAY.md)**.

On first boot an empty `/app/state` is seeded from
`deploy/state-templates/${LAB_STATE_TEMPLATE:-state-template}`, and
`/app/live-state` from `live-state-template`.

## Tests

```bash
python -m pytest -q
```

## License

MIT — see [LICENSE](LICENSE).
