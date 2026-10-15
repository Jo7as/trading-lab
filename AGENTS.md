# AGENTS.md

Guidance for AI coding assistants (Cursor, VS Code Copilot, Kilo, Claude Code,
…) working in this repository.

## What this is

`trading-lab` — a research lab for **testing, backtesting and evolving intraday
trading strategies**. Paper/signal only, no live broker execution. Python 3.11.

## Setup & checks

- Install: `uv sync` (or `pip install -e .`).
- Tests: `python -m pytest -q` — **keep the suite green**; add a test for any
  behaviour change. Never hit the network from tests.
- There is no separate lint/typecheck command configured; match the style of the
  surrounding modules instead.

## Layout

| Path | What goes here |
|---|---|
| `trading_lab/` | Library + workers (importable code) |
| `trading_lab/strategies/` | One module per strategy (config dataclass + `run_backtest`) |
| `trading_lab/adapters/` | Data sources (dukascopy, yfinance, market_data, …) |
| `trading_lab/*_evolution_worker.py` | Evolution workers (one per strategy) |
| `backtests/` | Runnable backtest / study runners that use the library |
| `studies/` | Exploratory measurement harnesses (no trading) |
| `configs/` | Strategy presets: `baselines/`, `evolved/` |
| `deploy/state-templates/` | Seed state for fresh deployments |
| `docs/` | Strategy specs, research notes, deployment guide |
| `scripts/`, `mt5/`, `tests/` | Ops helpers, MT5 demo EA, test suite |
| `artifacts/` | Generated reports/caches — **git-ignored, don't commit** |

## Adding a strategy

Follow **[docs/ADD_STRATEGY.md](docs/ADD_STRATEGY.md)**. In short: copy a module
in `trading_lab/strategies/`, keep the `run_backtest(df, cfg) -> dict` contract,
add a runner in `backtests/`, add a test in `tests/`.

Every strategy must pass the shared **acceptance path** (a null/significance
test, not just positive PnL) — see **[docs/ACCEPTANCE.md](docs/ACCEPTANCE.md)**.
Evolution promotion to `best_oos` is gated on it.

## Conventions

- Environment variables are read via `trading_lab.env.env` with the `LAB_`
  prefix (legacy `HERMES_*` names still work).
- Reports/results go to `artifacts/`; never commit data, `.env`, or live state.
- Don't commit secrets. `.env` is git-ignored; use `.env.example` for keys.
- Deployment (Railway: evolution worker + Telegram signal worker) is optional —
  see `docs/DEPLOY_RAILWAY.md`. It is never required to run or test strategies.
- Match the existing module style: `from __future__ import annotations`,
  type hints, dataclasses, small focused functions.
