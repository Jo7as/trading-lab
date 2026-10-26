# Findings — research log (to Oct 2026)

Short version: **nothing in this project has a validated, cost-surviving edge.**
Six directions were tested and rejected, an audit of the pre-existing strategies
found nothing that passed its own gates, and the recurring failure mode is
identical every time. The one candidate that did pass its gates — a structural FX
carry factor — later turned out to be a **look-ahead bug** (see below), which is
itself the most instructive result in the file. What remains valuable is the
tooling and the negative results — both are documented so nothing gets rebuilt.

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

Three rules follow, and they are now enforced in `docs/ADD_STRATEGY.md` and the
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

3. **Check that the exit is doing any work before tuning the reward/risk.**
   `studies/exit_attribution.py` groups trades by exit reason. On every ORB market
   the fixed 2R target fires in **6-13 %** of trades while **46-67 %** are closed at
   the session end, and those time exits cover a median of only **6-22 %** of the
   entry-to-target distance. Sweeping `target_r` there tunes a parameter that is
   almost absent from the P&L. If the target fires in under ~20 % of trades, fix the
   horizon/target pair - do not optimise the ratio.

Two process lessons, learned the hard way:

4. **"Too good to be true" is a bug signal.** A Sharpe near 5 and 100 % positive
   rolling windows appeared in this project — and were a slicing artifact. A
   backtest must be tested on the *real frame shape* (a full 24 h frame), not on
   RTH-only synthetic bars. Guarded by `mr_common.day_slices` +
   `tests/test_day_slices.py`.
5. **Write down what would kill the idea before starting.** The pre-committed
   kill criteria in [RESEARCH_BACKLOG.md](RESEARCH_BACKLOG.md) are what ended
   this round rather than another month of parameter tuning.
6. **Force every result into a second, independent representation.** The H-009
   look-ahead (below) survived months of review and only broke when the same
   strategy had to be rebuilt as a daily series for the prop simulator. Monthly
   vs daily, per-trade vs per-month, R vs %: if two representations of the same
   rule disagree, one of them is wrong. It is the cheapest test we have, and it
   has now caught both of this project's serious bugs.

## Method layer adopted (Oct 2026)

The research method now comes from the reviewed drop-in (integrated):
`docs/DATA_AUDIT_SPEC.md` (A1–A6 classification — real market events are **kept**,
not deleted), `docs/PATTERN_RESEARCH.md` (Z1–Z10 measurable-state catalogue and
protocol), `docs/hypotheses/` (template + three registered cards),
`trading_lab/hypothesis_card.py` + `scripts/validate_hypotheses.py` (the S2 gate:
numbered claim, mechanism, placebo, kill criteria, S0 ratio below 0.10,
instruments must have `audit_passed`).

Outcome of the first pass under that protocol: 9 of 11 instruments pass the data
audit (JP225 thin days, UST10Y wrong data); the strict S0 limit excludes 60-minute
horizons on our CFDs (only ≥240m qualifies); three pre-registered hypotheses fail
their gates; **H-001's high-volatility variant is the largest effect measured
(+0.183 % median, eff/cost 9.4) but fails on z and an inverted dose-response**.

## FX carry looked like the one survivor — then it turned out to be look-ahead

Everything above is a *pattern* search, and all of it died on costs. The direction
that briefly survived was structural, not predictive: cross-sectional **FX carry**
— long the 4 highest rate-differential pairs, short the 4 lowest, monthly,
equal-weight, gated by a VIX calm filter (`docs/hypotheses/H-009.yaml`).

It is now **rejected**, and the way it was caught is the useful part. The prop-firm
simulator needs a *daily* P&L path, so the carry basket had to be rebuilt from
scratch as a daily series. Comparing the two exposed the bug: the VIX filter was
evaluated with the VIX at the **end** of the month it was supposed to gate, while the
return runs from the start of that month. The VIX is correlated with the month's
outcome, so the filter was selecting the good months after the fact.

| evaluation (26 pairs, FRED rates, measured swap markup, 92 months) | CAGR | Sharpe | MDD |
|---|---|---|---|
| VIX filter at month **end** (the bug) | +13.8 % | 1.69 | −5.9 % |
| control: VIX-end **≥ 70** | −15.9 % | −1.48 | −29.8 % |
| **VIX filter at month start (correct)** | **+3.3 %** | **0.37** | **−20.2 %** |
| control: VIX-start ≥ 70 | +4.9 % | 0.41 | −11.5 % |
| no filter at all | +3.8 % | 0.39 | −16.5 % |

The control rows are the proof: with the end-of-month VIX the filter separates the
groups by 30 percentage points; with the start-of-month VIX it separates them by
0.1. There is no filter edge. And the price leg alone is −0.6 %/yr (Sharpe −0.07) —
the financing carried all of it, so there is no carry *price* alpha here either.

Every earlier carry number is void, including the measured-markup re-run
(+14.4 %/1.71, train 1.38, holdout 3.03, placebo −5.1 %) — same biased evaluation.
What survives is the *measurement* work, which is independent of the filter:
`trading_lab/duka_swaps.py` and its 349-instrument table still describe the broker's
financing correctly, and `trading_lab/rates.py` still gives the historical
differentials. The strategy conclusion is gone.

Lesson worth keeping: **the bug was not found by staring at the carry code — it was
found by forcing the same strategy into a second, independent representation**
(monthly → daily) for an unrelated reason (prop rules). Two representations that
must agree are the cheapest test we have. It is the same shape as the earlier
day-slicing bug, which only showed up when the frame shape changed.

## What is durable here## What is durable here

- **Prop-firm simulator** (`trading_lab/prop_sim.py`): FTMO 2-step, 1-step,
  Lucid, Instant 25K/100K — evaluation pass rates, funded payout simulation,
  block-bootstrap Monte Carlo. Strategy-agnostic: feed it any trade log.
- **Acceptance/significance** (`trading_lab/acceptance.py`, `significance.py`)
  and the daily/monthly **report + dashboard** tooling (`studies/`).
- **The verified slice helper** and the 24 h-frame guard (the bug above).
- **Market data**: Dukascopy 1-minute bid+ask for 11 instruments (2016/2019–
  2026) plus tick-level quotes, cached under `artifacts/` (git-ignored).
- **Written negative results** so the same ideas are not re-tried.
- **Real rates and real broker swaps** (`trading_lab/rates.py`,
  `trading_lab/duka_swaps.py`): monthly 3-month interbank rates from FRED and the
  broker's published overnight-swap table, so carry financing is measured instead
  of assumed. Note FRED's OECD series are stale for EUR/GBP (last value
  2026-01-31) — the carry ranking uses the forward-filled value and flags it.

## What the cost gate actually says (`studies/cost_move_screen.py`)

The S0 gate was required by `AGENTS.md` but only ever applied pointwise. Run
across all 21 cached instruments and four horizons it is worth stating carefully,
because the obvious reading of it is wrong.

**The gate is a threshold, not a death sentence.** With `ratio = E|move| / cost`,
a signal must capture more than `q_min = 1/ratio` of the average absolute move
before costs. Only `q_min > 100 %` (i.e. `ratio < 1`) is genuinely impossible -
there even a perfect signal loses. Everything else is a statement about how good
the signal has to be:

| horizon | q_min (median) | range across instruments | really dead |
|---|---|---|---|
| 15 min | **41.3 %** | 14.5 % - 151.5 % | COPPER, XAGUSD |
| 60 min | 20.7 % | 7.2 % - 73.6 % | - |
| 240 min | **9.8 %** | 3.4 % - 33.3 % | - |
| 1 day | **3.4 %** | 1.1 % - 12.0 % | - |

So the honest statement is not "intraday is dead" but: **at 15 minutes a signal
has to capture ~41 % of the average move to break even; at 4 hours ~10 %; at a
day ~3 %.** A directional signal that captures 41 % of the average absolute move
is extraordinarily strong - that is why nothing intraday here ever worked - but
the gate does not *prove* it cannot exist.

Two corrections to how this was first written down:

1. Only **2 of 21 instruments** fail even the impossible test, and only at 15
   minutes. The earlier table's "FAIL" column was a convention (`ratio < 3`, the
   loose `cost_gate.DEFAULT_K`), not a result, and calling it "dead on arrival"
   was wrong.
2. The eight dead ideas were **not excluded by this gate**. They died because
   their signal captured too little of the move - the gate tells you the bar, it
   does not tell you that you missed it. What the gate does legitimately support
   is a *prior*: the bar falls by roughly 4x per horizon step (41 % -> 21 % -> 10
   % -> 3 %), so the same signal quality goes much further at longer horizons.

Practical consequence, and it survives the correction: **the 4-hour horizon is the
one that both lowers the bar to a plausible level (~10 %) and still fits inside a
trading day.** The daily horizon is easier still but needs an account that allows
overnight and weekend holds. That is where the untested ground is.

## The multi-day horizons are empty too (`studies/state_scan.py`)

All nine registered cards were intraday, so the horizons where the S0 threshold
drops from ~70 % to ~52 % hit rate had never been looked at. The scan covers 5
index/metal markets x 4 holding periods (1/2/5/10 days) x 5 states (volatility
regime, overnight gap, range compression, distance to prior high/low) x both
tails = **200 cells**:

**0 cells beat the unconditional drift and the cost.** Not one is even
significant on its own: no cell reaches `|t| >= 3`, and the excess-return
distribution is centred on zero (median +0.02 %, p5 -0.22 %, p95 +0.39 %).

The scan also produced two mistakes that are worth more than the result, because
both would have manufactured false candidates:

1. **Overlapping forward windows.** With daily observations and an h-day forward
   return, consecutive windows share (h-1)/h of their data, so the t-statistic is
   inflated by roughly `sqrt(h)`. Taking every h-th day fixes it.
2. **No placebo.** The first version compared the conditional return with *zero*.
   Over 2016-2026 the indices drift up, so in a rising market every condition
   looks profitable - that version reported 61 "hits". The effect has to be
   measured against the *unconditional* drift of the same sample.

With both fixed: 61 -> 0. Rule of thumb for any future event study here: forward
windows must not overlap, and the benchmark is the sample's own drift, never zero.

## Is the pipeline too strict? No - it is underpowered (`studies/pipeline_power.py`)

After nine rejections the fair question is whether the gates would have recognised
a real edge. Measured directly: synthesise trade series with a *known* Sharpe, run
the real acceptance path, count how often it passes.

| true Sharpe | 150 trades | 400 | 1000 | 2500 |
|---|---|---|---|---|
| 0.0 (false positive) | 5 % | 4 % | 6 % | 4 % |
| 0.5 | 7 % | 19 % | 26 % | 42 % |
| 1.0 | 17 % | 36 % | 63 % | 94 % |
| 1.5 | 32 % | 59 % | 89 % | 100 % |
| 2.0 | 44 % | 84 % | 100 % | 100 % |

**The calibration is correct** - the false-positive rate sits at the nominal 5 %.
The pipeline is not rejecting everything out of hand; it rejects at exactly the
rate it advertises.

**The power, though, is the problem.** At the sample sizes this repo actually
produces (150-1500 trades) a realistic candidate (Sharpe 0.5-1.5) is detected only
7-89 % of the time, and a Sharpe-0.5 edge with 1000 trades just **26 %** of the
time - three real edges in four get thrown away. The 80 % detection threshold is
Sharpe >= 2.0 at 400 trades, 1.5 at 1000, 1.0 at 2500.

So the honest framing is not "the gates are too strict" but **"the gates have a
detection threshold, and it sits above what these data can supply."** Three
consequences:

1. Do **not** loosen the thresholds. Lowering p would raise the false-positive
   rate and the whole point of the nine rejections is that they are trustworthy.
2. The only honest lever is **n**: more years of history, more instruments, or more
   trades per year. The last one collides with the cost wall - higher frequency
   means a smaller move per trade, which is exactly what the S0 screen rejects.
3. Every card should state its power up front. A card with 400 trades and an
   expected Sharpe of 1.0 has 36 % power: it will probably fail *even if the edge
   is real*, and that is worth knowing before spending a week on it.

## 2026-10-26 — pipeline audit + target change

A full read of the pipeline (acceptance → cost model → data path → evolution/OOS →
runners) asked whether the strategies are checked **too strictly, or wrongly**.
The tests themselves are sound, but four classes of real defect turned up. All are
fixed and covered by tests.

| defect | where | effect |
|---|---|---|
| a closed log carrying none of `r`/`pnl_pct`/`equity_pnl_pct` **promoted unchallenged** | `significance.py` | the gate could be bypassed by a logging/schema bug |
| three modules could not run at all: `NEAR_K` undefined, `env = env(...)` shadowing (×2) | `cost_move_screen.py`, `st_evolution_worker.py`, `compare_live_sim.py` | the screen died *after* printing its matrix; the worker and the live-compare helper raised `UnboundLocalError` |
| a missing `goal.yaml` raised `FileNotFoundError` (or `None.get`) | every evolution worker | any local state dir without a goal crashed the worker |
| `pullback.py` resolved fills and exits on the wrong side of the book | `strategies/pullback.py` | see below |

`pullback.py` (H-010) was the substantive one. Its resting entry was tested with
the bid *close*, its short-side stop and target with the bid high/low, and its
time exit left at the mid — so shorts were systematically flattered and an exit
could print at a price the market never traded. Every order is now resolved on the
side it really trades at (long: the limit fills on an ask touch, stop/target on
the bid; short mirrored) and the spread sits inside the fills instead of being
charged on top. Fill *rate* and adverse selection are still not modelled — a
touched limit is assumed to fill at its own price, which is exactly what H-010 has
to measure next (it also has no hypothesis card yet; the protocol wants S2 before
the test).

**Deliberately not changed.** The significance test is calibrated — 5 % false
positives at n≥100 and the same power across payoff shapes, verified by
simulation, so mean-reversion shapes are not disadvantaged — and `MIN_TRADES = 10`
still *blocks* rather than waves through. A low-frequency strategy whose OOS window
holds 1–9 trades is therefore stuck; that is an evidence gap, and the remedy is a
larger sample, not a smaller threshold. `docs/ACCEPTANCE.md` now spells out the
three gate outcomes.

**Target change.** The band is now **1–2 %/month** (was 2–4 %): the old band needed
a net Sharpe of 2.7–6.0 and made every measurement a failure by construction. The
new band is the one this file's own power analysis arrived at. The numbers live in
`trading_lab/goal.py` only — before, the target sat in ten places with five
different values (0.01/0.02/0.04/0.06/0.08 as a 30-day target, plus a 0.27 annual
gate in `run_h004.py`). The deploy templates moved with it, and their drawdown
limit was aligned to the 20 % gate this repo already documents (0.08, as the
templates had it, is unsatisfiable at any sensible volatility). Side effect to
watch: `score.py` normalises returns by the target, so a smaller target lets the
return component saturate earlier. If the hill-climb starts accepting noise, raise
`target_return_30d` back to 0.02 in the state dir's `goal.yaml` — one line, one
file.

**Environment finding (not a repo bug).** `uv run python backtests/backtest_orb_2r.py`
— the invocation the README documents — failed with `ModuleNotFoundError:
trading_lab`, because the three `.pth` files in `.venv/…/site-packages/` carry the
macOS `hidden` flag and CPython skips hidden `.pth` files, which makes the
editable install of `trading_lab` inert. `python -m …` and anything run from the
repo root still work. `chflags nohidden` on those files fixes it but was reverted
by something watching the Desktop path, so four entry points now add
`sys.path.insert(0, ".")` themselves (ORB machine, its filter study, the C1
harness, the dashboard). The remaining ~36 scripts need the same one-line insert
before they can be run as `python <path>` again.

**Still open from the audit** (each needs a decision, none touched): the fixed OOS
window with `best_oos` = max over unbounded rounds (selection on the holdout); the
ST worker's OOS-driven short gate (real leakage); forward windows running over
session boundaries in the H-010 studies; `mfe_mae`/`first_passage_grid` reading
uncleaned data; `data_quality.clean(tz=…)` masking the roll in local time while
`data_audit.build_study_frame` does it in UTC; and the two event studies that
charge 2 spreads round-trip where `cost_gate` assumes 1 — which, per
`AEP_STAGE0_K3.md`, is enough to flip 4/26 candidates into 15/26.

## Open options (not started)

1. **Execution cost, not pattern search.** Everything died on CFD spreads. CME
   futures have ~1-tick spreads and ~$1 commission — roughly 5–10× lower
   relative cost. Testable with a small Databento MBP-1 sample (ES/NQ) or on a
   funded futures account.
2. **Longer horizons** (4 h and daily), where moves are many spreads. The screen
   above says 4 h is the only horizon that both clears the gate and fits a
   trading day; daily clears it everywhere but needs an account that allows
   overnight/weekend holds. Most prop firms forbid both, so this is the own-capital
   route.
3. **The multi-module plan** ([MULTI_STRATEGY_DUKASCOPY_PLAN.md](MULTI_STRATEGY_DUKASCOPY_PLAN.md)):
   its own gate framework is sound, but apply lesson (1) to each module first.

Priors for "another intraday pattern on CFD data": low (~10 %). The honest
levers are execution cost or horizon, not more patterns.
