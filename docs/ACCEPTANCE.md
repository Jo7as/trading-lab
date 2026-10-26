# The acceptance path

Every strategy — written by you or by someone else — runs through the **same
acceptance path**. A result only counts as real if it is distinguishable from a
no-edge ("null") benchmark, not merely because its total PnL is positive.

The whole thing is two small modules:

- `trading_lab/significance.py` — the null tests and the single threshold set.
- `trading_lab/acceptance.py` — `run_acceptance(...)`, one report, one decision.

## What it checks

`run_acceptance(trades, ..., extra={...})` combines:

1. **Significance** (mandatory) — the shared null test below.
2. **Extra criteria** (optional, strategy-specific) — passed in as a dict, e.g.
   `{"oos_positive": True, "double_cost_positive": True, "neighbours_positive": ...}`.

`passed` is `True` only when the significance test **and** every extra criterion
pass. The report (`AcceptanceReport.summary()`) prints all of it.

## The null test

Two models, chosen automatically:

- **Generic (default).** When the trade log carries a clock, the decision is made
  on the **daily** aggregate — `t_daily >= 3` on the calendar-day sums
  (`significance.assess_trades`, the same criterion as funnel stage S4). The
  daily series is used because same-day trades are not independent and an account
  experiences days - it does **not** add power: with independent trades its t
  equals the per-trade t (checked over 2,000 seeds, `tests/test_screening.py::
  test_daily_aggregation_does_not_add_power`). The real fix for "rejected for lack
  of power" is a bar that knows how much was searched: one pre-registered variant
  needs t >= 1.645, the best of a broad search is judged by its **Deflated
  Sharpe** (`screening.deflated_sharpe`, fed by `trading_lab/optimizer.py`), and
  `t >= 3` is only the default when the number of trials is unknown. A **sign-flip permutation** on the per-trade
  returns (one-sided p-value for `mean > 0`) is kept as the fallback for a log
  with **no** clock at all — a series that cannot be aggregated cannot do better.
  Both need no strategy cooperation (`significance.returns_from_trades()`, using
  `r`, else `pnl_pct`).
- **Random-entry (percentile).** The caller supplies a null sample of the same
  statistic (`null_samples`) and the test reports the percentile the strategy
  beats. Used by the ORB machine and the C1 harness, which build a true
  random-entry null from their own bar/session data.

Both also report a **95 % bootstrap CI** of the mean return.

## Thresholds and knobs

Defaults live in one place (`significance.py`): `MIN_PERCENTILE = 95`,
`MAX_PVALUE = 0.05`, `MIN_TRADES = 10` (a sample below this is not trusted, so a
single lucky trade never counts).

The evolution gate reads them from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `LAB_SIGNIFICANCE` | `1` | `0` disables the gate (debug only) |
| `LAB_SIGNIFICANCE_MODE` | `enforce` | `enforce` blocks a non-significant promotion; `flag` still promotes but records the result |
| `LAB_SIGNIFICANCE_MIN_PERCENTILE` | `95` | random-entry threshold |
| `LAB_SIGNIFICANCE_MAX_PVALUE` | `0.05` | per-trade sign-flip threshold (clockless fallback) |
| `LAB_SIGNIFICANCE_MIN_TRADES` | `10` | minimum trades to trust |
| `LAB_SIGNIFICANCE_T_DAILY` | `3.0` | S4 bar on the **daily** t (clocked log) |

**Fail-open, but only for an empty log.** Three outcomes, and the difference
between them matters:

| OOS log | `evaluated` | `enforce` mode | why |
|---|---|---|---|
| no closed trade at all | `False` | **allowed** (fail-open) | a missing/empty log must not become a silent, permanent block; the meta records `significant: null` |
| 1 … 9 usable returns | `True` | **blocked**, `reason="n<10"` | the test has no power there. This is an *evidence* verdict, not a finding against the strategy: the fix is more OOS trades (history, instruments, less selective rules), never a lower `MIN_TRADES` — that guard is what stops a one-lucky-trade promotion |
| closed trades, none of `r`/`pnl_pct`/`equity_pnl_pct` | `True` | **blocked**, `reason="no return series …"` | a logging/schema bug is not "nothing to evaluate"; a mis-keyed log must never promote unchallenged |

Note the second row: a low-frequency strategy can be permanently unable to promote
if its OOS window holds fewer than `MIN_TRADES` trades. That is deliberate — the
remedy is a larger OOS sample, not a weaker test. `studies/pipeline_power.py`
measures the power the gate actually has at a given sample size.

## Where it is enforced

- **Evolution promotion (default on).** No worker promotes a new
  `best_oos` record unless the OOS trades pass `promotion_gate(...)`. Wired into
  every promotion path: `MrEvolutionBase` (mean-reversion, ORB-2R, session-range),
  `orb_evolution_worker`, `smc_evolution_worker` and `st_evolution_worker`.
  A failed test logs `[best-oos] blocked — significance gate: …` (or `flagged`
  in `flag` mode), and every promotion writes the result into
  `best_oos_meta.json`:
  ```json
  { "oos_pnl_pct": -4.21, "significant": false,
    "significance": { "method": "sign-flip", "n": 201, "p_value": 0.73, "passed": false, ... } }
  ```

- **The ORB Phase-1 machine** (`backtests/backtest_orb_2r.py`) and the
  **C1 harness** (`studies/research_c1_midday_reversal.py`) compute their
  percentile through the shared `significance.percentile_of`.

## Using it in a runner

```python
from trading_lab.acceptance import run_acceptance

report = run_acceptance(
    trades,                                  # your trade log (closed trades)
    extra={
        "oos_positive": bool(sum_r_2022_now > 0),
        "double_cost_positive": bool(sum_r_2x > 0),
    },
)
print(report.summary())
if not report.passed:
    raise SystemExit("rejected by the acceptance path")
```

Strategies that can build a random-entry null should pass it for the stricter
test:

```python
run_acceptance(trades, observed=sum_r, null_samples=random_sums_1d)
```
