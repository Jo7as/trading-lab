# Strategies — index

One file per strategy. **Status** is the honest verdict after backtesting,
consistency checks and the prop-firm simulation; numbers are at 0.5 % risk per
trade on Dukascopy 1-minute CFDs (US100 ≈ NQ, SP500 ≈ ES), spread + slippage
included.

## Overview

| Strategy | Status | Instruments tested | Headline | File |
|---|---|---|---|---|
| **ORB Rejection Fade** (`orb_fade`) | **REJECTED** | US100, SP500, GER40 (2019–2026) | earlier positives were a day-slicing bug; corrected: −0.25…−0.56 R, 8/8 negative years | [orb_rejection_fade.md](orb_rejection_fade.md) |
| Fair Pricing (JJ Simon) | rejected | US100 (2016–2026) | −0.15…−0.30 R, acceptance FAIL | [fair_pricing_REJECTED.md](fair_pricing_REJECTED.md) |
| NY-Open EMA12 | rejected | US100 (2016–2026) | −0.15…−0.30 R, all 20 parameter combos negative | [ny_open_ema12_REJECTED.md](ny_open_ema12_REJECTED.md) |

There is currently **no candidate.** The one that looked strongest was rejected
after a day-slicing bug in its engine was found and fixed.

## How the verdict is produced

| step | command |
|---|---|
| single-instrument headline | `studies/strategy_report.py --strategy orb_fade --asset US100 --firm lucid` |
| ablation + baselines | `backtests/backtest_orb_rejection_ifvg.py --runs 20` |
| monthly dashboard (HTML/CSV) | `studies/strategy_dashboard.py --risk-pct 0.5` → `artifacts/dashboards/` |
| prop-firm pass/payout | `studies/run_prop_sim.py --firm ftmo --asset US100` |

Prop-firm presets live in `trading_lab/prop_sim.py` (FTMO 2-step, 1-step, Lucid,
Instant 25K/100K) and apply to any strategy through the same trade log — new
strategy = one loader in `studies/*.py`'s `STRATEGIES` registry.

## Status definitions

- **GO candidate** — passes the shared acceptance path, beats the random-direction
  and classic-breakout baselines, positive in every instrument-year, fits the
  prop-firm risk rules. Not live; robustness/diversification still open.
- **rejected** — tested and dropped; kept as a written negative result so the
  idea is not rebuilt. Implementation removed from `main` (or archived on a
  branch) where noted in the file.
