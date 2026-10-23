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
docs/                 strategy specs, research notes, ADD_STRATEGY.md, ACCEPTANCE.md, DEPLOY_RAILWAY.md
AGENTS.md             onboarding guide for AI assistants in your IDE
scripts/              ops helpers (monitoring, inspections)
mt5/                  MetaTrader 5 demo EA
tests/                pytest suite
```

## Install

```bash
uv sync                 # or: pip install -e .
```

`uv sync` creates a local `.venv`. Activate it once before running any of the
commands below, **or** prefix each command with `uv run`:

```bash
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

> Running a `python ...` command without activating (or without `uv run`) uses
> your system interpreter, which does not have the dependencies — you will get
> `ModuleNotFoundError`.

## Quickstart

```bash
# Backtest an opening-range breakout config over 1m bid+ask data
uv run python backtests/backtest_orb_2r.py --markets US100

# Run the pre-registered day-filter study
uv run python backtests/backtest_orb_2r_filters.py --markets US100,GER40

# A stage-1 measurement harness (no trading): US100 NY midday reversal
uv run python studies/research_c1_midday_reversal.py
```

Backtests and studies write their reports to `artifacts/` (git-ignored). The
first run of a backtest or study fetches its data on demand, so it needs network
access (later runs use the local cache under `artifacts/cache/`).

## Data

Two free sources, switchable with `LAB_DATA_SOURCE=dukascopy|yfinance` (default
is `dukascopy` for intraday workers, `yfinance` for the higher-timeframe ones):

| Source | What you get | Intervals | Instruments |
|---|---|---|---|
| **Dukascopy** | OHLCV, and **1-minute bid+ask** with the real historical spread | 1m bid+ask (`minute_data`); the adapter also serves 5m, 10m, 15m, 30m, 1h, 4h, 1d | any instrument code of the `dukascopy_python` library for 1m bid+ask (FX, metals, indices, commodities, …); the historical adapter covers the mapped symbols + any `X/Y` FX pair |
| **yfinance** | OHLCV bars | any interval yfinance itself supports (1m … 1mo) | any yfinance ticker (stocks, ETFs, indices, futures, forex, crypto) — passed straight through |

- The 1-minute bid+ask series is Dukascopy, cached per calendar year under
  `artifacts/cache/m1/` (finished years are never re-downloaded).
- Add a new symbol with one line: `INSTRUMENTS` in `trading_lab/minute_data.py`
  (1m aliases, e.g. `"JP225": "E_N225Jap"`) and/or `INSTRUMENT_MAP` in
  `trading_lab/adapters/dukascopy_adapter.py`. yfinance needs no map — any
  ticker works as-is.
- Everything is fetched on demand and cached; tests never touch the network.

## Research findings

Read **[docs/FINDINGS.md](docs/FINDINGS.md)** first: this repo's research log.
No strategy here has a validated, cost-surviving edge — five directions were
tested and rejected, and the recurring cause is the same (the average move per
trade is the same order as the spread). The log states the two rules that follow
(measure `move/cost` before building; a missing cost model is a bug) and what is
durable: `trading_lab/prop_sim.py` (FTMO/Lucid/Instant evaluation + payout
simulation), the acceptance/significance path, the report/dashboard tooling, and
the per-strategy dossiers under **[docs/strategies/](docs/strategies/README.md)**.

## Built-in strategies

The engine is strategy-agnostic; a few complete strategies ship as reference
implementations (note: none of them is validated net of costs — see the findings
above):

**Costs column matters:** a strategy without a cost model reports *gross* P&L —
treat those numbers as unverified, not as profit (see `docs/FINDINGS.md`).

| Module | Strategy | Notes | Costs modelled |
|---|---|---|---|
| `strategies/orb_2r.py` | ORB, fixed 2R target | 1m bid+ask; `US100`/`GER40`/`XAUUSD`/`JP225` | yes (bid/ask + slippage + commission) |
| `strategies/session_range_fade.py` | Session-Range-Fade | fade failed London breakouts (Gold) | yes (bid/ask + slippage) |
| `strategies/mean_reversion.py` | Z-score mean reversion | 15m fade back to the SMA | yes (slippage) |
| `strategies/pair_spread.py` | Pair spread (stat-arb) | rejected candidate, kept as a guard | yes (bid/ask both legs) |
| `strategies/orb_rejection_ifvg.py` | ORB rejection fade | rejected candidate, kept for its ablation | yes (bid/ask + ticks) |
| `strategies/opening_range_breakout.py` | Opening-Range Breakout | NY 09:30, 5m | **no — gross only** |
| `strategies/supertrend_volume.py` | Supertrend + Volume | Gold, hourly (default worker) | **no — gross only** |
| `strategies/regime_hybrid.py` | ADX regime hybrid | Supertrend in trends, BB reversion in ranges | **no — gross only** |
| `strategies/bb_squeeze_breakout.py` | BB Squeeze Breakout | Forex H1 | **no — gross only** |
| `strategies/trend_range_rejection.py` | Trend Range Rejection | CB reversal timing | **no — gross only** |
| `strategies/smc_ict.py` (+ `strategies/smc/`) | SMC / ICT multi-timeframe | nested entry | **no — gross only** |
| `strategies/silver_bullet.py` | ICT Silver Bullet | time-boxed SMC scalp | **no — gross only** |

Evolution workers (`trading_lab/*_evolution_worker.py`) exist for Supertrend,
SMC, ORB, ORB-2R, Session-Range and Mean-Reversion. Every result must pass the
shared **acceptance path** — a null/significance test, not just positive PnL —
before it counts, and evolution promotion is gated on it: see
**[docs/ACCEPTANCE.md](docs/ACCEPTANCE.md)**. **Any strategy can be
implemented** against the same engine — module + runner + test, and optionally an
evolution worker: see **[docs/ADD_STRATEGY.md](docs/ADD_STRATEGY.md)**.

## Add your own strategy

A new idea is one module + one runner + one test. Hand this to your AI assistant
in the IDE along with **[docs/ADD_STRATEGY.md](docs/ADD_STRATEGY.md)**:
add `trading_lab/strategies/my_idea.py` (a config dataclass + a
`run_backtest(df, cfg) -> dict`), a runner in `backtests/`, and a test in
`tests/`. To run it 24/7, plug it into an evolution worker (Level 3 in the doc).

## Starting a worker

The entrypoint dispatches on `LAB_WORKER`:

```bash
LAB_WORKER=evolution       LAB_STATE=state uv run python -m trading_lab.run
LAB_WORKER=orb2r_evolution LAB_STATE=orb2r-state ORB2R_EVOLUTION_ASSET=JP225 \
    uv run python -m trading_lab.run
LAB_WORKER=signal          LAB_STATE=live-state uv run python -m trading_lab.run
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
uv run python -m pytest -q     # or, with the venv activated: python -m pytest -q
```

## License

MIT — see [LICENSE](LICENSE).
