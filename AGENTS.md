# AGENTS.md

Guidance for AI coding assistants (Cursor, VS Code Copilot, Kilo, Claude Code,
…) working in this repository.

## What this is

`trading-lab` — a research lab for **testing, backtesting and evolving intraday
trading strategies**. Paper/signal only, no live broker execution. Python 3.11.

## Setup & checks

- Install: `uv sync` (or `pip install -e .`).
- **Tell the user about the venv on first run.** `uv sync` creates `.venv`; run
  every Python command through it, e.g. `uv run python ...` (or activate it first
  with `source .venv/bin/activate`, Windows: `.venv\Scripts\activate`). A bare
  `python ...` uses the system interpreter and fails with `ModuleNotFoundError`.
  If `import trading_lab` ever reports "No module named", re-run `uv sync`.
  This is the single most common setup mistake — point it out proactively.
- Tests: `uv run python -m pytest -q` — **keep the suite green**; add a test for
  any behaviour change. Never hit the network from tests.
- Backtests/studies download their data on demand (network) on the first run;
  results are cached under `artifacts/`. Workers read config from `.env`
  (copy `.env.example`); nothing there is required just to run or test code.
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
| `docs/strategies/` | One dossier per strategy (index in `README.md`): status, numbers, caveats |
| `scripts/`, `mt5/`, `tests/` | Ops helpers, MT5 demo EA, test suite |
| `artifacts/` | Generated reports/caches — **git-ignored, don't commit** |

## Adding a strategy

Follow **[docs/ADD_STRATEGY.md](docs/ADD_STRATEGY.md)** — it has a copy-paste
skeleton and five levels. The short version:

1. `trading_lab/strategies/my_idea.py` — a config dataclass (+ `search_space()`)
   and `run_backtest(df, cfg, *, verbose=False) -> dict` returning
   `{"trades", "metrics", "equity_curve", "signals"}`. Reuse the shared blocks —
   `RiskManager` (`trading_lab.risk`), `_compute_metrics`/`_pos_to_dict`
   (`trading_lab.backtest`), the indicators — don't re-implement them.
2. `backtests/backtest_my_idea.py` — load data, call `run_backtest`, write a
   report to `artifacts/` (see `backtests/backtest_orb_2r.py` for a full example).
3. `tests/test_my_idea.py` — a tiny synthetic frame, assert the expected trade;
   never hit the network.
4. Run the **acceptance path** (`trading_lab.acceptance.run_acceptance`): the
   shared null/significance test, not just positive PnL — see
   **[docs/ACCEPTANCE.md](docs/ACCEPTANCE.md)**. Evolution promotion to `best_oos`
   is gated on it.
5. Optional: an evolution worker for 24/7 parameter search — Level 3 in
   ADD_STRATEGY.md, reference `trading_lab/orb2r_evolution_worker.py`.

To hand this to your IDE assistant, point it at `docs/ADD_STRATEGY.md` together
with the idea.

## Conventions

- Environment variables are read via `trading_lab.env.env` with the `LAB_`
  prefix (legacy `HERMES_*` names still work).
- Reports/results go to `artifacts/`; never commit data, `.env`, or live state.
- Don't commit secrets. `.env` is git-ignored; use `.env.example` for keys.
- Deployment (Railway: evolution worker + Telegram signal worker) is optional —
  see `docs/DEPLOY_RAILWAY.md`. It is never required to run or test strategies.
- Match the existing module style: `from __future__ import annotations`,
  type hints, dataclasses, small focused functions.
