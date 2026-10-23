# Findings — research log (to Oct 2026)

Short version: **no strategy in this project has a validated, cost-surviving
edge.** Five directions were tested and rejected, an audit of the pre-existing
strategies found nothing that passed its own gates, and the recurring failure
mode is identical every time. What remains valuable is the tooling and the
negative results — both are documented so nothing gets rebuilt.

## What was tested, and what happened

| direction | result | evidence |
|---|---|---|
| Fair pricing (JJ Simon) | rejected — −0.15…−0.30 R, acceptance FAIL | [strategies/fair_pricing_REJECTED.md](strategies/fair_pricing_REJECTED.md) |
| NY-open first candle vs EMA12 | rejected — all 20 parameter combos negative | [strategies/ny_open_ema12_REJECTED.md](strategies/ny_open_ema12_REJECTED.md) |
| ORB rejection fade + inverse FVG | rejected — the early positives were a **day-slicing bug**; corrected: −0.29…−1.18 R on 11 markets | [strategies/orb_rejection_fade.md](strategies/orb_rejection_fade.md) |
| Gold Supertrend baseline (`hf_ls01`) | **no edge** — +59 % gross, but net is −2 %…+28 % depending on stop slippage (+13 % at 0.5×, 4/8 negative years): indistinguishable from zero |
| `opening_range_breakout` / `silver_bullet` (gross-only modules) | **no edge** — negative or zero *before* costs (−9.8 %, +0.3 %) | [strategies/EXISTING_STRATEGIES_AUDIT.md](strategies/EXISTING_STRATEGIES_AUDIT.md) |
| Pair spreads (stat-arb, market-neutral) | rejected — gross PF ≈ 1.0, net −141…−691 % | [strategies/pair_spreads_REJECTED.md](strategies/pair_spreads_REJECTED.md) |

Also audited: the repo's own stored verdicts already said nothing passed
(`orb_2r.json` → `passed: []`; `mean_reversion_candidates.json` →
`all_go: []`), and both evolution "promotions" are flagged in their own metadata
as unvalidated (one from an *asset-drift bug*).

## The data layer was dirty (found last, and it touches everything)

A data audit (`trading_lab/data_quality.py`, `studies/data_audit.py`) of the
cached 1-minute bid+ask found **corrupt bars on all 11 instruments**: 33–741 bars
per instrument move more than 50× the median, with the worst clustered in the
daily roll window (21:00–01:00 UTC) — JP225 logged **+178 % in one minute**,
Brent +21.9 %, copper +14 %. A single such bar fabricates a "profitable pattern"
(one of them produced a fake "+4.5 %/day first-hour momentum" in the research
engine).

Measured impact: the gold Supertrend baseline's gross P&L falls from **+59.3 % to
+47.1 %** once the corrupt bars and roll hours are masked — about a fifth of what
was reported was data artifact. **Every earlier result in this repo ran on
uncleaned data** and must be re-checked with `clean()` before being quoted.

Consequence for the process: the data layer (plan step 1) is a *prerequisite*,
not a formality — `clean()` (corrupt bars + roll window) is now applied in the
research engine, and the spread-by-hour table shows US-hours spreads are 2–3×
tighter than Asia (part of why trading Asia hours never worked).

## The structural lesson

Every rejection has the same cause: **the average move per trade is of the same
order as, or smaller than, the round-trip cost** (spread + slippage).

- gold Supertrend: 0.72 $ average move vs 0.37 $ spread (ratio 1.94 — right at
  the threshold, so it survives but barely);
- pair spreads: gross expectancy ≈ 0, spreads then dominate;
- ORB fade: gross-positive-looking only because of a bug; correct edges ≈ random.

Two rules follow, and they are now enforced in `docs/ADD_STRATEGY.md` and the
review process:

1. **Measure `avg |move| / round-trip cost` before building anything**, and take
   the cost from the strategy's **real sizing and order types** — not from the
   nominal risk rule. (An earlier estimate here was 2.6× too high because the
   `RiskManager` caps the position at 1× notional and a limit exit pays no
   spread.) Below ~2–3 it cannot work at that frequency; no parameter search
   fixes that.
2. **A strategy module without a cost model is a bug, not a simplification.**
   Anything reported "gross only" is unreported, not profitable. **7 of the 12
   modules in this repo have no cost model** (see the table in the README and in
   `strategies/EXISTING_STRATEGIES_AUDIT.md`) — results from those are gross.

Two process lessons, learned the hard way:

3. **"Too good to be true" is a bug signal.** A Sharpe near 5 and 100 % positive
   rolling windows appeared in this project — and were a slicing artifact. A
   backtest must be tested on the *real frame shape* (a full 24 h frame), not on
   RTH-only synthetic bars. Guarded by `mr_common.day_slices` +
   `tests/test_day_slices.py`.
4. **Write down what would kill the idea before starting.** The pre-committed
   kill criteria in [RESEARCH_BACKLOG.md](RESEARCH_BACKLOG.md) are what ended
   this round rather than another month of parameter tuning.

## What is durable here

- **Prop-firm simulator** (`trading_lab/prop_sim.py`): FTMO 2-step, 1-step,
  Lucid, Instant 25K/100K — evaluation pass rates, funded payout simulation,
  block-bootstrap Monte Carlo. Strategy-agnostic: feed it any trade log.
- **Acceptance/significance** (`trading_lab/acceptance.py`, `significance.py`)
  and the daily/monthly **report + dashboard** tooling (`studies/`).
- **The verified slice helper** and the 24 h-frame guard (the bug above).
- **Market data**: Dukascopy 1-minute bid+ask for 11 instruments (2016/2019–
  2026) plus tick-level quotes, cached under `artifacts/` (git-ignored).
- **Written negative results** so the same ideas are not re-tried.

## Open options (not started)

1. **Execution cost, not pattern search.** Everything died on CFD spreads. CME
   futures have ~1-tick spreads and ~$1 commission — roughly 5–10× lower
   relative cost. Testable with a small Databento MBP-1 sample (ES/NQ) or on a
   funded futures account.
2. **Longer horizons** (daily/multi-day), where moves are many spreads — but
   most prop firms forbid overnight/weekend holds; swing account types allow it.
3. **The multi-module plan** ([MULTI_STRATEGY_DUKASCOPY_PLAN.md](MULTI_STRATEGY_DUKASCOPY_PLAN.md)):
   its own gate framework is sound, but apply lesson (1) to each module first.

Priors for "another intraday pattern on CFD data": low (~10 %). The honest
levers are execution cost or horizon, not more patterns.
