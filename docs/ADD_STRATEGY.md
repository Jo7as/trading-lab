# Adding your own strategy

trading-lab is built so a new idea is **one module + one runner + one test**.
You can hand an idea to an AI assistant in your IDE (Cursor, VS Code, Copilot,
Kilo, …) and point it at this file.

There are two levels. Start at **Level 1**; only touch Level 3 if you want the
idea to run 24/7 on a server.

## Level 1 — the strategy module

Create `trading_lab/strategies/my_idea.py`. It needs exactly two things:

1. A **config dataclass**. The risk fields (`risk_per_trade`, `sl_atr_mult`,
   `tp1_r`, `tp1_close_pct`, `max_daily_loss`, `cooldown_hours`) are required by
   `RiskManager`; add your own parameters on top. Add a `search_space()` so the
   evolution worker can search it later.
2. A **`run_backtest(df, cfg, *, verbose=False) -> dict`** function that walks
   the bars causally and returns
   `{"trades": [...], "metrics": {...}, "equity_curve": [...], "signals": [...]}`.

Use the shared building blocks — don't re-implement them:

| Tool | From | Purpose |
|---|---|---|
| `RiskManager(cfg, account_size, backtest_mode=True)` | `trading_lab.risk` | sizing, SL/TP, daily-loss/cooldown gate |
| `rm.can_trade(ts)` / `rm.open_position(...)` / `rm.close_position(...)` | `trading_lab.risk` | position lifecycle |
| `_pos_to_dict(pos)` | `trading_lab.backtest` | trade → dict for the trade log |
| `_compute_metrics(trades, equity_curve, cfg)` | `trading_lab.backtest` | win rate, DD, Sharpe, expectancy |
| `atr`, `sma`, … | `trading_lab.indicators` | indicators |

### Minimal skeleton

```python
"""Donchian-channel breakout (copy this as a starting point)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..backtest import _compute_metrics, _pos_to_dict
from ..indicators import atr as atr_indicator
from ..risk import RiskManager


@dataclass(frozen=True)
class DonchianConfig:
    # risk (required by RiskManager)
    risk_per_trade: float = 0.01
    sl_atr_mult: float = 2.0
    tp1_r: float = 1.0
    tp1_close_pct: float = 1.0
    max_daily_loss: float = 0.03
    cooldown_hours: int = 0
    # strategy
    lookback: int = 20
    atr_period: int = 14
    account_size: float = 10000.0
    asset: str = "US100"
    asset_label: str = "US100"

    @staticmethod
    def search_space() -> list[tuple[str, float, float, float]]:
        return [("lookback", 10, 60, 5), ("sl_atr_mult", 1.0, 3.0, 0.5)]


def run_backtest(df: pd.DataFrame, cfg: DonchianConfig, *, verbose: bool = False) -> dict[str, Any]:
    """Long-only Donchian breakout: ATR stop, 1R target, flat at session end."""
    out = df.copy()
    if "volume" not in out.columns:
        out["volume"] = 0.0
    high, low, close = (out[c].to_numpy(float) for c in ("high", "low", "close"))
    atr = atr_indicator(out["high"], out["low"], out["close"], cfg.atr_period).to_numpy(float)

    rm = RiskManager(cfg, account_size=cfg.account_size, backtest_mode=True)
    trades: list[dict] = []
    equity: list[float] = []
    pos = None

    for i in range(len(out)):
        ts = float(out.index[i].timestamp())
        if pos is None:
            broke_out = i >= cfg.lookback and close[i] > high[i - cfg.lookback:i].max()
            allowed, _ = rm.can_trade(ts)
            if broke_out and allowed and atr[i] == atr[i] and atr[i] > 0:
                pos = rm.open_position(
                    entry_price=close[i], atr=float(atr[i]), direction="long",
                    entry_time=ts, entry_idx=i,
                )
        elif low[i] <= pos.stop_loss:
            rm.close_position(pos, pos.stop_loss, "stop_loss", ts)
            trades.append(_pos_to_dict(pos)); pos = None
        elif high[i] >= pos.tp1:
            rm.close_position(pos, pos.tp1, "tp", ts)
            trades.append(_pos_to_dict(pos)); pos = None
        equity.append(rm.current_equity)

    if pos is not None:
        rm.close_position(pos, float(close[-1]), "backtest_end", float(out.index[-1].timestamp()))
        trades.append(_pos_to_dict(pos))

    return {
        "trades": trades,
        "metrics": _compute_metrics(trades, equity, cfg),  # type: ignore[arg-type]
        "equity_curve": equity,
        "signals": [],
    }
```

> **Before writing any code — three mandatory gates.**
> **(0b) Hypothesis card.** Any candidate behaviour is written down first as
> `docs/HYPOTHESIS_TEMPLATE.yaml` → `docs/hypotheses/<id>.yaml`, and
> `trading_lab.hypothesis` must accept it (claim with a number, mechanism,
> placebo, kill criteria, `kosten_ratio` from the cost gate). See
> `docs/PATTERN_RESEARCH.md` for how to look for recurring behaviour without
> chart-shape scanning.
> **(0) Cost gate.** Run `studies/cost_gate.py` for the instrument and your
> intended horizon. It reports `E|move| / round-trip cost`; a ratio below ~2
> means **nothing built on that horizon can work** (no parameter search fixes
> it). On our CFD data 5-minute horizons fail almost everywhere and 60-minute
> horizons pass on the indices.
> **(3) Calendar invariants.** Any session-based strategy must pass
> `trading_lab.invariants.assert_invariants` (entries/exits inside the session,
> no position spanning two days) and keep a test on a **full 24 h frame**
> (`tests/test_invariants.py`) — this is the bug class that made an ORB strategy
> look profitable.
>
> **Watch out — index spaces.** If your strategy trades session/day slices (RTH,
> opening range, session hours) built from a boolean mask, the slices must index
> the **full** frame. Open-coding `np.flatnonzero(mask)` + `unique(...)` and then
> slicing the full frame silently walks windows out of phase with the clock —
> this once made an entire strategy look profitable. Use
> `trading_lab.mr_common.day_slices(local, mask)` and keep a test on a **full
> 24 h frame** (`tests/test_day_slices.py`); an RTH-only synthetic frame will not
> catch it. Also: a result that looks too good is a bug signal — check the
> session mapping, the per-year split and the random-direction baseline first.

Good references to read next: `trading_lab/strategies/orb_2r.py` (1m bid+ask,
costs, filters) and `trading_lab/strategies/opening_range_breakout.py` (5m,
RiskManager lifecycle).

## Level 2 — a runner in `backtests/`

Add `backtests/backtest_my_idea.py` that loads data, calls `run_backtest` and
writes a report to `artifacts/`:

```python
from dataclasses import replace
from trading_lab.minute_data import load_minute_bid_ask
from trading_lab.strategies.my_idea import DonchianConfig, run_backtest

df = load_minute_bid_ask("US100", 2016)          # 1m bid+ask, cached per year
res = run_backtest(df, replace(DonchianConfig(), asset="US100"))
# res["metrics"] -> total_trades, win_rate, sharpe, expectancy_r, max_drawdown_pct …
```

Run it: `uv run python backtests/backtest_my_idea.py`. See `backtests/backtest_orb_2r.py`
for a full example with a train/test/confirm split and a markdown report.

### Data

- **1-minute bid+ask (Dukascopy):** `trading_lab.minute_data.load_minute_bid_ask(asset, year)`.
  Add new symbols to `INSTRUMENTS` in `trading_lab/minute_data.py`
  (e.g. `"JP225": "E_N225Jap"`).
- **Other timeframes / daily:** resample with `trading_lab.mr_common.resample_ohlc`,
  or use an adapter from `trading_lab/adapters/` (yfinance, …).
- Everything is cached under `artifacts/`; never hit the network from tests.

## Level 3 — plug into evolution (optional)

To let a worker search your parameters around the clock, subclass
`MrEvolutionBase` in a new `trading_lab/my_idea_evolution_worker.py`:

```python
class MyIdeaEvolutionWorker(MrEvolutionBase):
    MODE = "my_idea_evolution"
    VERSION_PREFIX = "mid"
    BASELINE_FILE = "my_idea_baseline.yaml"
    DEFAULT_ASSET = "US100"
    BASE_TIMEFRAME = "1min"
    TIMEFRAMES = ("1min",)
    config_cls = DonchianConfig

    def _backtest(self, df, cfg): ...        # call run_backtest, return {"trades","metrics"}
    def _random_seed(self, base): ...        # return a random config
    async def load_frames(self): ...         # load data (see orb2r_evolution_worker.py)
```

Then:
1. Add a seed template `deploy/state-templates/my-idea-state-template/`
   (`goal.yaml`, `strategy.yaml`, `my_idea_baseline.yaml`).
2. Register the mode in `trading_lab/run.py` (`if worker_mode == "my_idea_evolution": ...`).
3. Deploy it — see `docs/DEPLOY_RAILWAY.md`.

`trading_lab/orb2r_evolution_worker.py` is the reference implementation.

## Level 4 — a test

Add `tests/test_my_idea.py`. Tests never touch the network — build a tiny
synthetic OHLCV frame and assert the trade you expect:

```python
def test_breakout_opens_a_long():
    df = _flat_day(...)                       # small hand-made frame
    trades = run_backtest(df, DonchianConfig(lookback=3))["trades"]
    assert trades[0]["direction"] == "long"
```

Run the suite: `uv run python -m pytest -q`. Keep it green.

## Level 5 — acceptance (required for every strategy)

A positive PnL is not enough. Every strategy runs the **same acceptance path**
before it counts: the shared significance test plus any strategy-specific
criteria (period sums, cost stress, neighbours). Wire it into your runner:

```python
from trading_lab.acceptance import run_acceptance

report = run_acceptance(trades, extra={"oos_positive": ..., "double_cost_positive": ...})
print(report.summary())
assert report.passed
```

The evolution path enforces this automatically — a worker only promotes a new
`best_oos` record when its OOS trades pass the gate. Full details, the null
models and the tunable thresholds: **[ACCEPTANCE.md](ACCEPTANCE.md)**.

