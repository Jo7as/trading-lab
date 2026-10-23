# Pattern & state research — how to look for recurring behaviour the right way

This replaces "how to find chart patterns". Chart *shapes* (flags, wedges, SMC
structures, "looks like a rejection") are the wrong instrument: they are
high-dimensional, discretionary and endlessly re-fittable, so scanning them for
significance manufactures findings. It is how this project once produced a
profitable-looking strategy out of a bug.

What we do instead: define **measurable states**, test the *forward return in the
state's direction* against the round-trip cost, a placebo and a dose-response —
**before** any strategy code exists. The value is in the question, not the scan.

## 0. Prerequisites (both are mandatory)

1. **Cost gate (`studies/cost_gate.py`).** If `E|move| / round-trip cost < ~2` at
   your horizon, nothing built there can work. On our CFD data 5-minute horizons
   fail almost everywhere; 60-minute horizons pass on the indices. Choose the
   horizon first, then the state.
2. **Clean data (`trading_lab.data_quality`).** The cached series carry corrupt
   bars on every instrument (33–741 bars > 50× the median move, worst at the
   daily roll). Run `studies/data_audit.py` and feed studies the `clean()` frame.

## 1. The catalogue: measurable states (not shapes)

Everything below is a number, computable causally, with no eyeballing. Grouped by
mechanism that has a reason to exist:

| state | how it is measured | plausible mechanism |
|---|---|---|
| level break | close beyond yesterday's high/low, in σ and spread units | stop/limit clusters at levels (Osler) |
| distance to level | \|level − price\| / σ and / spread | same |
| volatility regime | efficiency ratio / ATR percentile over 20 days | dealer gamma, margin, flow regimes |
| compression → expansion | yesterday's range vs its own distribution | volatility clustering |
| time-of-day seasonality | move and spread per session hour (see `observe.py`) | session participants differ |
| gap / overnight behaviour | open vs prior close, fill rate | auction imbalance |
| intraday momentum | first-hour sign → rest-of-day (and last-hour) | leveraged-ETF / gamma rebalancing |
| mean reversion | \|z\| vs a 20/60-bar mean → forward return | liquidity provision |
| order-flow proxy | Dukascopy tick bid/ask volumes (indicative) | imbalance (weak — indicative only) |
| cross-asset divergence | one instrument vs a correlated one (ES/NQ, gold/silver, DAX/FTSE) | lead-lag, index vs basket |
| carry / roll | term structure or swap for the instrument | financing |
| event conditioning | FOMC/CPI days vs others (`calendar_events.py`) | scheduled flow |

Websites: none. Each state must be expressible as a rule a computer evaluates
without interpretation.

## 2. The protocol (per candidate state)

1. **Pre-register** the hypothesis card (`docs/HYPOTHESIS_TEMPLATE.yaml`): claim
   with a number, mechanism, dose-response, placebo, kill criteria — **before**
   looking at results. `trading_lab.hypothesis` refuses a card that has no number,
   no mechanism or no kill criteria, and refuses one without a pre-computed
   `kosten_ratio` (the S0 gate) — so the cost question is answered first.
2. **Measure** on train data only, with `trading_lab.research`: mean forward
   return in the state's direction, t-statistic (non-overlapping sample), hit
   rate, **placebo** (random bars from the full series), **dose-response**
   (magnitude quintiles), and **effect/cost**.
3. **Gate (S2b):** t ≥ 3, effect ≥ 2× cost, placebo clearly smaller, dose-response
   visible. Fail → the state is **dropped, not tuned**.
4. Only then write the strategy (S3/S4), and it must pass the calendar invariants
   (`trading_lab.invariants`) and the cost-model rules (real bid/ask, taker vs
   limit, sizing from the actual position cap).

## 3. Search budget (the part that keeps it honest)

- **A scan is exploratory, not evidence.** If you scan K states, the significance
  threshold must be adjusted for K (or the scan is repeated on untouched data).
- **Log every trial** (state, parameters, data window, result) in an append-only
  registry; the deflated Sharpe is computed from that count, not from the winner.
- **Cap iterations:** 3 per hypothesis, then it is dead. No "improving" a state
  that failed its placebo.

## 4. What we do not do (learned the hard way)

- No shape scanning / no discretionary structures / no "looks like".
- No parameter fine-tuning after seeing the result; a parameter set is chosen
  once from a plateau, then frozen, and the freeze is hashed.
- No per-instrument tuning of a shared rule.
- No conclusions from gross P&L: a strategy module without a cost model is
  unreported, not profitable.
- No results from uncleaned data: corrupt bars fabricate patterns.
