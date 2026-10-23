# Strategies — index

One file per strategy. **Status** is the honest verdict after backtesting,
consistency checks and the prop-firm simulation; numbers are at 0.5 % risk per
trade on Dukascopy 1-minute CFDs (US100 ≈ NQ, SP500 ≈ ES), spread + slippage
included.

## Overview

| Strategy | Status | Instruments tested | Headline | File |
|---|---|---|---|---|
| **ORB Rejection Fade** (`orb_fade`) | **REJECTED** | US100, SP500, GER40 + 8 more (2019–2026) | earlier positives were a day-slicing bug; corrected: −0.29…−1.18 R on all 11 markets, 7–8/8 negative years | [orb_rejection_fade.md](orb_rejection_fade.md) · [source spec](orb_rejection_ifvg_SPEC.yaml) |
| Pair spreads (ES/NQ, gold/silver, Brent/WTI, DAX/FTSE) | rejected | US100/SP500, US30/SP500, GER40/UK100, XAU/XAG (2019–2026) | gross PF ≈ 1.0 (no edge), net −141…−691 % with real spreads | [pair_spreads_REJECTED.md](pair_spreads_REJECTED.md) |
| Fair Pricing (JJ Simon) | rejected | US100 (2016–2026) | −0.15…−0.30 R, acceptance FAIL | [fair_pricing_REJECTED.md](fair_pricing_REJECTED.md) |
| NY-Open EMA12 | rejected | US100 (2016–2026) | −0.15…−0.30 R, all 20 parameter combos negative | [ny_open_ema12_REJECTED.md](ny_open_ema12_REJECTED.md) |

There is currently **no candidate.** The one that looked strongest was rejected
after a day-slicing bug in its engine was found and fixed.

**Next plan:** [MULTI_STRATEGY_DUKASCOPY_PLAN.md](../MULTI_STRATEGY_DUKASCOPY_PLAN.md)
— a four-module design (noise-area momentum, order-flow absorption at levels,
mean reversion inside the noise area, FX strength) with a regime and portfolio
layer. Not started; build module by module with a go/no-go after each.

## Audit of the pre-existing strategies

See [EXISTING_STRATEGIES_AUDIT.md](EXISTING_STRATEGIES_AUDIT.md): nothing in the
repo passed its own gates (`orb_2r.json` → `passed: []`,
`mean_reversion_candidates.json` → `all_go: []`), the two evolution promotions
are flagged unvalidated (one from an asset-drift bug), and the one positive gross
lead (gold Supertrend `hf_ls01`, +59 % gross) turns **negative** once the gold
spread is applied (avg gross move 0.29 USD vs 0.37 USD spread).

## Latest step (Oct 2026)

[../STRATEGY_HYPOTHESES_STEP2.md](../STRATEGY_HYPOTHESES_STEP2.md): cleaned re-runs of every
headline, three pre-registered hypotheses on ≥60m horizons (all nine cells
FAIL), and the pre-committed decision to change the data foundation (crypto L1 or
real futures) instead of searching more patterns on these CFDs.

## Method (for the next candidates)

- `docs/PATTERN_RESEARCH.md` — measurable-state catalogue and protocol (no shape scanning).
- `docs/HYPOTHESIS_TEMPLATE.yaml` + `trading_lab/hypothesis.py` — the S2 card gate.
- `studies/cost_gate.py` — S0 pre-flight; `trading_lab/data_quality.py` + `studies/data_audit.py` — data gate.
- `trading_lab/research.py` + `studies/observe.py` — observation tables and event studies (WIP).

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
