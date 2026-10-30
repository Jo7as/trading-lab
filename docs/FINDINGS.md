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

## 2026-10-26 (later) — Phase 1/2: clocks, cost space, OOS discipline

### Nothing reaches the new band either — and now the bar is the right one

Every stored candidate, re-checked against **1–2 %/month** (band lower edge
+12.7 %/yr):

| candidate | source | annualised, net | vs band |
|---|---|---|---|
| gold Supertrend `hf_ls01` / XAUUSD 1h | `net_of_costs`, 1 582 trades | +3.7 %/yr (0× slippage), **+1.7 %/yr (0.5×)**, −0.3 %/yr (1×) | FAIL, 3.5–7× short |
| `opening_range_breakout` / US100 | −9.8 % gross over 7.8 y | −2.6 %/yr, negative *before* costs | FAIL |
| `silver_bullet` / XAUUSD | +0.3 % gross, 269 trades | −2.1 %/yr (was −5.2 % before the limit-entry fix) | FAIL |
| `orb_2r` US100/GER40/XAUUSD | its own six-check gate | `passed: []` | FAIL |
| mean-reversion candidates | `all_go: []`; 0–1 of 10 walk-forward folds positive | negative | FAIL |
| H-009 carry | corrected, no look-ahead | +2.4 %/yr, Sharpe 0.27 | FAIL |
| H-004 Asian range | `studies/run_h004.py` | −1.6 %/yr | FAIL |

Lowering the target did **not** turn anything into a pass. It changed the size of
the gap from 10–30× to ~3.5–7× and made the band reachable *in principle*
(Sharpe 0.8–1.8) — the honest outcome, since nothing in the repo is there.

### Clocks (T1.1/T1.4/T1.7)

* `data_quality.clean`/`audit` mask the daily roll in **UTC**
  (`ROLL_HOURS_UTC`, `roll_mask`). The old mask applied the hours in the *asset*
  timezone: for a New York asset it blanked 01:00–04:59 UTC and left the real
  21:00 UTC roll in the data — i.e. exactly the fabricated jumps the gate exists
  to remove.
* `smc_ict` reads `weekday`/`hour` from the market calendar (New York), not
  Berlin; the Berlin weekday pushed every trade after 18:00 NY onto the next day.
* `docs/TIME_CONVENTIONS.md` writes the convention down: three clocks (UTC index,
  exchange-local sessions, plan documents in local wall-clock notation), the
  instrument→session map, the roll window, the cost space and a study checklist.
* `recheck_cleaned`'s per-year branch was dead (`"cost" in t` could never be true,
  so `pos_years` was always 0); the cost column is attached now and the metric
  printed.

### Cost space (T1.5)

One rule, in code and tested: `cost_gate.leg_cost` / `round_trip_spread`
(limit/TP leg = 0, market/stop/time leg = 0.5 full spreads → 0.0 / 0.5 / 1.0 for
the three round-trip shapes). `cost_gate` derives its two reported levels from the
helper; `net_of_costs` takes `entry_leg` per strategy — it charged half a spread on
*every* entry, which over-charged `silver_bullet`'s resting limit (its net(0×)
moves from −5.2 % to −2.1 %) — and `gold_baseline_net_costs`/`gold_baseline_v2`
use it too.

### OOS discipline (T2.1/T2.2/T2.3)

* `trading_lab/walk_forward.py`: 5 rolling folds (20 % test, step capped so the
  first fold still trains on ≥50 % of the history) instead of one fixed 30 %
  holdout, chosen per round — so no single window is maximised continuously. Every
  worker that had a split now splits per round; `LAB_LEGACY_SPLIT=1` restores the
  old behaviour for old artefacts.
* `trading_lab/promotion.py`: a promotion must also clear trades ≥ 10, Sharpe ≥ 0.5
  and drawdown ≤ 20 % (env-tunable), and the meta record carries the fold plus
  `selection: holdout-max`, so it cannot be read as a frozen-holdout claim. The
  leaderboard's `prev` starts at −inf now (before, a negative OOS PnL could never
  become a record).
* The ST worker's short gate reads the **train** window; too few train shorts keeps
  shorts on instead of silently forcing long-only.

### Fill side (T1.6)

`studies/h010_fill_rate.py` measures what a backtest cannot. First result (US100,
2024): the resting limit fills **97 %** (ny) / **94 %** (london) of the time,
median 1 bar, with a median **1.25–1.36 spread adverse excursion before the fill**
— and the filled signals are *better* than the counterfactual (R +0.03 vs −0.45 in
ny). So the fill assumption is not a free lunch, but the price of a limit entry is
real and shows up as that excursion, which `_resolve` only partly captures (it
checks the fill bar's own extreme). Queue position and partial fills remain
unmodelled: the study's number is an upper bound.

The fill side is **asset-dependent**, and gold is the bad case: XAUUSD flags
adverse selection in both sessions (R_fill −0.15 vs R_nofill +0.02 in ny, −0.05 vs
+0.05 in london — the *unfilled* signals would have done better), while US100
(+0.03 vs −0.45) and GER40 (+0.09, +0.11) do not. Fill rates are 93–97 %
everywhere, median 1 bar. So "a touched limit fills at its own price" is a
reasonable assumption for the indices and a generous one for gold.

**The session-bounded forward window mattered more than the cost fix.** The event
studies used to read the response at `mid[i + horizon]` on a *session-filtered*
frame, so a window that ran past the close booked the **overnight gap** as the
setup's reaction. `h010_variants` (H=240) showed 9 of 40 cells with |t| ≥ 2.5 and a
positive net; bounded to the session day (479 926 / 482 371 samples dropped) it
collapses to **1 of 40**, and London's 1.91 bp cell falls to 0.11 bp. For US100 ny
the one surviving hit is exactly the 1.0 expected by chance among 20 cells, and
`run_h010_pullback` finds **0 of 4** cells with |t| ≥ 2.5 and a positive net.
H-010's apparent edge was largely the night.

The cost fix on its own: `run_h001` 2 → 1 spread (eff/cost 3.3 → 6.5, verdict
unchanged), `mfe_mae` a few pullback cells turn positive (E[R] −0.56 → +0.03),
`first_passage_grid` 0 → 7 of 480 positive E[R] cells, `asymmetric_rr` numerically
unchanged (it already used 0.5/1.0 through its own constants, now through the
helper).

## Open options (not started)

1. **Execution cost, not pattern search.** Everything died on CFD spreads. CME
   futures have ~1-tick spreads and ~$1 commission. *Correction 2026-10-27
   (`trading_lab/costs.py`):* that is **~1.5x**, not 5-10x, cheaper than an index CFD
   on a prop account - MNQ ~0.5-0.6 bp round trip vs US100 at FTMO ~0.9 bp (no index
   commission there). The repo also charged **no commission at all** until then,
   which made FX majors look 3-4x cheaper than they are and AUD/NZD/gold (wide
   Dukascopy feed spreads) look ~2x more expensive than at a raw-ECN venue. Testable with a small Databento MBP-1 sample (ES/NQ) or on a
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

## 2026-10-27 — real costs, H-011 (relative value, limit) and H-012 (multi-market trend)

**Cost model fixed (`trading_lab/costs.py`).** Until now only the Dukascopy feed
spread was charged and no commission anywhere. Checked against FTMO (FX $2.50/lot/side
since 2025-09-29, indices commission-free), IC Markets Raw and Dukascopy: FX majors were
3-4x too cheap (EURUSD 0.19 bp spread, but ~0.6-0.7 bp with commission), AUD/NZD/gold
~2x too expensive (the feed's 0.9-1.0 pip vs 0.02-0.4 pip raw ECN), index CFDs about
right. Default profile is now `ftmo`; `ecn_raw`, `dukascopy`, `cme_micro` and the old
`data` are selectable. ORB-2R slippage is relative now (0.5 bp per fill) - the old
absolute 1.0 point was ~2 bp per fill on SP500.

**H-011 - H-006 with limit entries** (`studies/run_h011_relval_limit.py`, train only).
Limits do not rescue it: both legs fill only 49-57 % (ftmo spreads) / 83-88 % (ecn_raw)
of the time, the chased second leg loses the fast part of the reversion, and the
signals that did NOT fill were the best ones (+7..+12 bp gross) - adverse selection.
Best cell (ecn_raw, market, z >= 2.5, 08-11 NY): net +0.84 bp/event, 4/6 years, t 0.69
-> **kill at S3**. All ftmo cells are net negative. Holdout untouched.

**H-012 - a crude always-in TSMOM book on 26 markets** (`studies/run_h012_trend_multi.py`).
**Withdrawn as evidence (2026-10-27, review with the user).** The design was not fit for
the question: no stop-loss and no take-profit (always long or short), risk parity that
ignored correlation (14 FX pairs share the dollar factor -> 20 % realised vol instead of
10 %, 67 % drawdown), a flat financing guess instead of real swap rates, CFD oil/copper
without roll handling, no unit tests and no sanity check against a published trend
index. Its "kill at S1" says nothing about trend following or swing trading in general.
The one number that stands: an always-in book at ~5x gross leverage pays ~6-7 %/yr
financing on CFDs - continuous-exposure designs must model swaps; stop-based swing
trades with ~0.5 % risk carry a fraction of that. Holdout untouched.

**Rerun with the `ftmo` cost profile (2026-10-27, all 39 runs, holdout locked).** Index
results are unchanged (FTMO charges no index commission; spreads from the data). FX gets
worse by 0.01-0.05 R per trade (commission): pullback EURUSD/USDJPY, session_range_fade
EURUSD/USDJPY, mean_reversion EURUSD and supertrend EURUSD drop from S2/S3 to S1/S0
(net negative). orb_2r improves by +0.015..+0.034 R (relative instead of absolute
slippage, most on SP500). Funnel: S1 161/242 -> S2 142 -> S3 69 -> **S4 0**, as before.
Practical reading for a prop account: intraday edges have to live on indices (no
commission) or be large enough per trade to carry ~0.6-1 bp FX round trips.

## 2026-10-27 — S-001: SMC/ICT 2022, fairly re-tested on the new structure framework

Spec agreed with the user (`docs/specs/smc_ict2022.yaml`), every rule tested
(`tests/test_smc_ict2022.py`), 96-variant grid (timeframe, CE/edge, min RR, liquidity vs
fixed 3 R, FVG/IFVG, bias on/off), 5 markets, train 2019-2024, ftmo costs, holdout locked.
The reference check caught three implementation gaps before the grid ran (FVG confirmed
one bar after the MSS, one trade per killzone = filled trades, internal swing sweeps) and
one spec question (MSS = CHoCH only) that the user changed to "first internal break,
CHoCH or BOS".

Funnel: S1 25/30 -> S2 18 -> S3 3 -> **S4 0**. Best: **GER40, 4h/15m, CE, min RR 2,
fixed 3 R, no bias** - 129 trades (1.8/month), 47 % hits, +0.29 R/trade, 5/6 years,
neighbours 86 %, placebo +0.32 R, PBO 0.13, but DSR 0.49; for 1.5 %/month it would need
2.9 % risk per trade and a 17 % drawdown. Per-axis averages: the HTF bias does not help
(4 of 5 markets better without it), 4h/15m beats 1h/5m, the liquidity target does not
beat a fixed 3 R, FVG beats IFVG on indices. Verdict: no demonstrable SMC edge in train;
GER40 is a lead, too thin to prove or to trade on a prop account.

## 2026-10-28 — S-002: Fibonacci retracement with a wrong-level control

Spec agreed with the user (`docs/specs/fib_retracement.yaml`), rules tested
(`tests/test_fib_retracement.py`; the tests caught a look-ahead - one mutable impulse
object shared across bars - and intraday entries after the session end), 384-variant
grid, 5 markets, train 2019-2024, ftmo costs + swaps, holdout locked.

Funnel: S1 20/32 -> S2 16 -> S3 2 -> **S4 0**; best US100 intraday 0.5 + confirmation,
+0.04 R/trade, DSR 0.08. **The control decides it:** pooled over markets and variants,
intraday 0.5 / 0.618 = -0.057 R, control (random level in [0.40, 0.75] away from the fib
numbers) = -0.057 R, 0.705 = -0.085 R - fib levels are exactly as good as wrong levels,
as the Expert Systems 2021 study found. Swing: 0.5 +0.04 R / 0.618 +0.01 R vs control
-0.06 R, but only ~10-25 trades per market in six years and the pooled figure counts
shared trades several times - a hint, not evidence. Swing variants cannot reach S1's
100 trades on any single market.

## 2026-10-29 — Step A for S-003: what happens after a sweep? (measurement, no strategy)

`studies/sweep_continuation_study.py`, train 2019-2024, 4 FX majors (Mahadzva's set) +
US100/SP500/GER40/XAUUSD, 15m and 1h, three level types (external / internal swing
target, EQH/EQL pool), two kinds (reclaim = wick through and close back; through = close
beyond), with/against the structure trend, horizons 1/4/12/24 bars, excess over the
unconditional drift in ATR, events thinned so windows do not overlap. 768 cells.

* **Classic sweeps do not reverse.** After a wick through a level with the close back
  inside, price drifts slightly further IN the sweep direction: +0.01..+0.04 ATR on
  average, 60-88 % of cells positive, t >= 2 in ~6 % of reclaim cells (2.3 % expected by
  chance). Direction consistent with Mahadzva's continuation claim; size far below the
  ~0.06-0.27 ATR round trip (ftmo).
* **Closes THROUGH a level mean-revert short term, strongest on FX:** internal swing
  break in trend direction, next 4 bars: EURUSD 15m -0.09 ATR (t -4.1), 1h -0.12 ATR
  (t -2.7), GBPUSD/AUDUSD negative too, USDJPY not. So "wait for a close beyond the swept
  level, then enter with it" (the confirmation rule in the pasted S-003 plan) selects
  exactly the events that pull back.
* Only EURUSD 1h shows a fade effect larger than its cost (0.12 vs 0.065 ATR) - a lead,
  one cell among 768, needs a spec, the pipeline and the (unspent) holdout.

## 2026-10-29 — S-004: Breakout-Fade FX (spec `docs/specs/breakout_fade_fx.yaml`) - rejected at S2

Fade an internal swing break in the trend direction (short a close above the internal
swing high in an uptrend, long mirrored), stop = signal-bar extreme + x ATR, time exit
2/4/8 bars or retest of the broken level; 1m bid/ask fills, ftmo costs, 0.5 bp slippage
per fill. Grid 96 per pair, EURUSD/GBPUSD/AUDUSD/USDJPY (control), train 2019-2024.

* **Reference check passed - no implementation bug.** Base variant (1h, trend on, 1.5 ATR,
  4 bars) on mid prices: EURUSD +0.051 R (t 2.4), GBPUSD +0.040 R, AUDUSD +0.024 R,
  USDJPY -0.028 R - the step-A effect (EURUSD +0.09..0.115 ATR) at an average risk of
  1.8 ATR. The spec's "+0.08 R" assumed 1.5 ATR risk and was too optimistic.
* **Costs ~0.09 R per trade eat it:** spread 0.02-0.05 R, slippage 0.03-0.05 R (largest
  item), commission 0.02 R.
* Fixed RR targets (1:1 / 2:1 / 3:1, diagnostic only, not in the grid) do not help: best
  +0.06 R before costs; costs in R depend on the stop distance, not on the RR.
* **Grid: best variant per pair net -0.030 (EURUSD, GBPUSD), -0.048 (AUDUSD), -0.071 R
  (USDJPY); none passes S2.** Holdout untouched. Lesson: a fade effect of ~0.1 ATR is too
  small for market entries on FX; a limit-entry variant would be a new spec.

## 2026-10-29 — Step B: big moves after time-based liquidity levels? (measurement)

`studies/liquidity_levels_study.py` on the new `trading_lab/liquidity.py` (pd / pw / asia /
or), 8 markets, 15m + 1h, train 2019-2024. Bracket from the event bar's close: stop 1 ATR
(or, against the exceed, beyond the bar's extreme + 0.1 ATR), target 1/2/3/5 x stop, at most
8 h (15m) / 24 h (1h); excess over the same bracket from every 4th bar. 1536 cells, 0 with
net > 0 and t >= 2 individually.

* **Classic sweep reversal (reclaim, trade against) does not produce big moves:** with the
  tight SMC stop a small excess at 1R (+0.05 R, t >= 2 in 5-6 of 16 cells for pd/or), gone
  at 3-5R, and the tiny stops make costs 0.2-0.4 R per trade. Against a close THROUGH a
  level with a tight stop is clearly negative (-0.12..-0.20 R excess at 1R).
* **The one lead: continuation after the previous WEEK's high/low is taken** (`pw`, trade
  with the exceed, 1 ATR stop, 3-5 R target): excess +0.10..+0.15 R pooled, positive in 13
  of 16 market x tf cells for closes through at 3R (indices GER40/SP500/US100 and USDJPY
  strongest, GBPUSD/AUDUSD 15m negative), net ~+0.02..+0.04 R pooled, up to +0.2..+0.3 R
  on indices 1h. Only ~2 events per market and month (n ~ 150 per cell) - single cells are
  t 1-2; it needs a pooled spec (several markets, one rule) to be testable. Fits Osler's
  stop cascades and classic weekly breakout / time-series momentum.
* pd / asia / or continuation: same sign, roughly half the size - below costs.

## 2026-10-29 — S-006: Noise-area intraday momentum (Zarattini/Aziz/Barbon) - FIRST S4 survivors

Spec `docs/specs/noise_area_momentum.yaml` (rules from the paper, no tuning), index CFDs,
1m bid/ask fills, ftmo costs + 0.5 bp per fill, 1 R = one 14-day daily sigma. Train 2019-2024.

* Reference check: base variant (30 min, trailing band/VWAP) on mid prices SP500 Sharpe
  1.52, US100 1.38 (paper: SPY 1.33 net); net 0.65 / 0.69. US30 ~0, GER40 negative net -
  the effect does not carry to every index.
* Funnel (4 variants per market, 16 in total): **3 rows pass S1-S4** - SP500 30min/band
  (net +0.027 R, t_daily 1.86, DSR 0.974), SP500 30min/band_vwap (+0.020 R, DSR 0.951),
  US100 30min/band (+0.026 R, t_daily 1.95, DSR 0.978); 5/6 positive years, all neighbours
  positive. 15 min checks are weaker (more trades, more cost). PBO is not computable with 4
  trials (NaN passes by design) - selection risk is small, the rule comes from the paper.
* Size: ~20 trades/month, +0.02..0.03 R net each -> ~2.5 % p.a. at 0.5 %/R. SP500 and US100
  are one bet (correlated). A basket component, not the 1 %/month on its own.
* Holdout (2025+) not opened yet - needs the user's release.
* **Holdout (2025-01 .. 2026-10, opened once with the user's release 2026-10-29):**
  SP500 30min/band n 405, net -0.030 R (t -1.13), 27 % positive months, max DD 16.8 R -
  FAILED; SP500 30min/band_vwap -0.029 R - FAILED; US100 30min/band n 358, net +0.012 R
  (t 0.41), 50 % positive months, max DD 11.2 R - formally passes S5 (mean > 0) but is not
  distinguishable from zero. Verdict: the train edge (Sharpe ~0.65 net) did not carry over
  convincingly; the SP500+US100 block (1/2 R each) lost -3.75 R in the holdout with a
  max DD of 11.2 R - at 1 % per R that is an 11 % drawdown, beyond the prop limit. Not tradable as is; the holdout is spent for S-006 - no
  re-tuning on it.

## 2026-10-29 — S-007: Time-series momentum over 26 markets / 7 classes - real, but not past S3/S4

Spec `docs/specs/tsmom_multi_asset.yaml` (Moskowitz/Ooi/Pedersen rules), long daily data
2004-2024 (`trading_lab/long_daily.py`, yfinance; FX re-dated - the yfinance "=X" close is
the 00:00 UTC price of the label day, fixed and tested), costs measured on the CFD data
(0.9-1.9 bp per side on indices/FX), swaps by the CFD formula (rate + 2.5 % / FX
differential - 1 %). Portfolio = unit (agreed), 8 % target vol.

* Reference passed: monthly correlation with the AQR TSMOM factor 0.71 (252 months),
  Sharpe 0.79 before costs, 0.51 net (0.45 with double swap markup); big years 2013/2014/
  2017/2022, losing years 2011/2012/2016/2018/2023 - the textbook pattern.
* Swaps are the main cost (~-2.2 % p.a. vs -0.35 % p.a. spread+slippage); UST10Y loses
  -14 R (low vol -> large notional -> financing); BTC brings +31 R from 12 positions.
* Funnel: 12m/no stop n 1642 episodes, +0.150 R gross / +0.094 R net, placebo +0.004 R,
  t 2.37, DSR 0.906, 13/21 positive years (62 %) -> **fails S3** (>= 70 % years) and would
  fail S4 (DSR < 0.95) too. 12m + 3-ATR stop weaker (DSR 0.85); the blend with the 1- and
  3-month signals much weaker (Sharpe 0.15/0.11) - faster signals do not help here.
* Verdict: the effect is real (placebo edge +0.15 R, AQR match) but at CFD costs and a 8 %
  vol it is not significant enough for our bar. Holdout untouched.
* **Literature track + holdout (both released by the user 2026-10-29):** with the new track
  (`docs/PRUEFABLAUF.md`) the paper base (12m, no stop) passes S3 (13/21 >= 60 %, holding
  ~111 days) and S4 (t 2.37 >= 2.0, placebo +0.15 R); the 3-ATR stop variant fails S4
  (normal bar). **Holdout 2025-01 .. 2026-10: +13.5 % net (gross +23.9 %, swaps -9.6 %),
  Sharpe 0.87, t 1.20, max DD -6.3 % at 8 % vol - PASSED S5.** Contributors: metals +26.5 R,
  equity indices +23.4 R, crypto +11.8 R; bonds -10 R (financing again).
* First strategy of the lab through S1-S5. Caveats: holdout t 1.2 (22 months prove little on
  their own); train+holdout max DD -20 % at 8 % vol. On FundedNext Instant (6 % trailing):
  3 % vol -> ~1.7 % p.a., 95 % 1-year survival, ~2.5 % of the account paid out per year;
  6 % vol -> 49 % survival. A basket pillar, not the 1 %/month on its own; real swap rates
  (bonds, futures-based CFDs) are the biggest open cost question.

## 2026-10-29 — S-008: Turn of the Month (McConnell/Xu) - the effect is gone, fails S1

Spec `docs/specs/turn_of_month.yaml` (paper rule: long 6 indices from the close of day -2 to
the close of day +3, shared risk budget, no variants), long daily data 2004-2024.

* Reference only partly: window vs other days (bp/day) SP500 +5.3 vs +4.5, US100 +6.8 vs
  +5.8, GER40 +5.7 vs +3.4, UK100 +5.8 vs +0.8, but US30 +2.8 vs +3.4 and JP225 -0.4 vs +4.6.
* Portfolio: 1506 positions, +0.022 R gross / +0.005 R net, Sharpe 0.28 / 0.07, 12/21 years.
  **Placebo (4 random mid-month days, same sizing) +0.026 R - the window is no better than
  any other 4 days** -> fails S1. Holdout untouched. Consistent with the post-2006 decline
  reported for developed markets.
* Bug found on the way (before any verdict): the shared budget was divided by the number of
  indices entering the SAME day - scattered placebo entries got 6x the size; now divided by
  the active indices, with a test.

## 2026-10-29 — Quick checks before a spec: Pre-FOMC drift alive, FX carry dead after the markup

`studies/satellite_check.py` (measurement, no strategy, TRAIN only). FOMC dates from
federalreserve.gov in `trading_lab/fomc_meetings.json` (191 scheduled meetings 2004-2027,
press-conference flag).

* **Pre-FOMC drift** (Lucca/Moench), 14:00 NY the day before -> 13:55 statement day, 1m CFD
  2019-2024, 47 meetings: SP500 +27.6 bp vs +5.5 bp on other days (t 1.91), US100 +44.8 vs
  +7.7 (t 2.22), US30 +16.8 vs +4.1 (t 1.36), GER40 +39.5 vs +3.7 (t 2.38); 64-72 % positive.
  Round trip on indices ~3-5 bp incl. one night of financing - the effect is 5-10x the cost.
  -> worth a spec (S-009).
* **FX carry G10** (13 pairs, vol-scaled, monthly, 2004-2024): carry income +5.2 % p.a. before
  the swap markup, +1.9 % after 1 % per side; price -3.4 % p.a. (2008 -37 %, 2015 -26 %);
  Sharpe 0.12 before / -0.10 after the markup. Dead on CFDs - no spec.

## 2026-10-29 — S-009: Pre-FOMC drift (Lucca/Moench) - real, but fails S3 by one year

Spec `docs/specs/pre_fomc.yaml` (long SP500/US100/US30/GER40 14:00 NY day before -> 13:50
statement day, shared budget, no variants), 1m bid/ask, train 2019-2024, 47 meetings.

* 184 positions, move (mid) GER40 +41 bp, US100 +42, SP500 +26, US30 +15 per meeting;
  +0.062 R gross / +0.053 R net per position; placebo (same window, random non-FOMC days,
  same weekdays, 100 runs) +0.006 R -> edge +0.056 R; t 2.11; reference 4/4.
* Years 4/6 (2019 -1.3 R, 2021 -0.4 R) = 67 % < 70 % -> **fails S3** (1-day holds keep the
  70 % bar on the literature track). S4 would pass (t >= 2). Holdout untouched.
* Size: +1.6 R per year ~ +0.8 % p.a. at 0.5 % per R - a small basket piece even if it held.

## 2026-10-29 — Real FundedNext swaps (demo, MT5 cost logger): S-007 net Sharpe 0.51 -> 0.58

`mt5/TradingLabCostLogger.mq5` on the FundedNext demo (FundedNext-Server3, netting). The demo
**feed is frozen** (EURUSD 1.11921 for 20+ minutes, most symbols without quotes) - spreads and
slippage cannot be measured there; swaps are account parameters and can.

* FX swaps (points mode) are close to zero: EURUSD -0.2 % / -0.3 % p.a. long / short, USDJPY
  -0.0 / -0.1, majors mostly within +-0.6 %; AUD pairs: long +2 to +3 %, short -4 to -8 %.
  Metals: gold -1.1 % / -0.4 %, silver -1.5 % / +0.1 %. Our formula assumed rate differential
  - 1 % (FX) and rate + 2.5 % (metals: -6.4 % long).
* Index swaps are quoted as money per lot and the demo's contract data disagree without live
  quotes - not used (UK100 points to ~-1.7 % long, model -6.2 %).
* S-007 base with the measured FX/metal swaps (`--broker artifacts/broker_costs/fundednext_demo.json`):
  financing -32 % instead of -46 % over 21 years, **net Sharpe 0.58 (was 0.51)**, metals
  +36.8 R (was +25.7 R). Indices/bonds/commodities still on the formula - likely too
  pessimistic as well.
* The demo server lacks US30, oil, copper, bonds and crypto; FundedNext's help lists US30,
  USOUSD/UKOUSD and BTCUSD for the paid Stellar accounts.

## 2026-10-29 — Crypto order flow, step 1: Kim & Hansen quarter-hour effect replicated on 2024 (measurement)

Plan: `docs/research/CRYPTO_ORDERFLOW_PLAN.md` (phase 4, step 1). `studies/crypto_kim_hansen_replication.py`,
Binance USD-M perps, 10-s taker-signed bars (`trading_lab/adapters/crypto_trades.py`), opening
imbalance in the first 10 s after :00/:15/:30/:45 (base-asset sizes, as in the paper), slope of the
forward log return on it, Newey-West; placebo on :02/:17/:32/:47. Train only (2024-01..2024-10,
the overlap with the paper's 2021-01..2024-10 sample); holdout untouched. Checked line by line
against the paper PDF; remaining gaps: no reversal controls W_t, openings-only regression, 10 months.

| bp per unit OI | 4 h | 8 h | 12 h | placebo 4 h |
|---|---|---|---|---|
| BTC (paper A.4) | +2.4 (3.8), t 1.5 | +2.6 (5.6), t 1.0 | +3.0 (6.4), t 0.9 | -0.1 |
| SOL (paper A.4) | +4.5 (3.3), t 1.9 | +7.9 (2.4), t 2.2 | +8.4 (6.7), t 1.9 | -3.1 (t -1.6) |

* Same sign and order of magnitude as the paper; BTC about half the paper's size and not
  significant on 10 months alone; SOL in line, its placebo mildly negative.
* The first quarter alone (Jan-Apr) showed 2-3x larger slopes - a strong bull quarter, not the norm.
* Economics, as the theory predicted (`CRYPTO_MICROSTRUCTURE_THEORY.md` ch. 6): sign(OI) x return over
  the top 10 % |OI| openings is +4.5..+8.9 bp gross against ~10 bp taker round trip - **not tradable
  on its own**. Next: longer sample (2021-2023) and the hypotheses H1-H7 (cumulation, venue
  aggregation, extremes, state dependence, reversal-then-continuation entry).
* Index swaps from the MT5 symbol specification (user, 2026-10-29): SP500m futures calc,
  -41.41 / -35 USD per lot per night, 1 lot = 100 x index -> **-1.9 % / -1.6 % p.a.** (model
  -6.4 % / +1.4 %). With SP500 + UK100 measured and US100/US30/GER40/JP225 estimated as their
  mean: financing -22 % over 21 years, **S-007 net Sharpe 0.64** (14/21 positive years,
  indices +44.9 R instead of +29.3 R). Bonds/commodities/crypto still on the formula.

## 2026-10-29 — Crypto order flow, H3 + H5: extreme openings, market vs limit entry - no robust edge

`studies/crypto_opening_entry_study.py` (+ `tests/test_crypto_opening_entry.py`), 2024, Binance prices,
Bybit costs (taker 11.1-11.8 bp round trip, limit entry 7.5-7.9 bp), non-overlapping trades, limit
fills only on trade-through, thresholds and offsets fixed before the run (2 x 2 x 4 cells per market).

* 64 cells: a handful net positive (best BTC |OI| >= 0.8, 8 h, limit 5 bp: +5.4 bp net, t 2.4;
  SOL |OI| >= 0.6, 8 h, limit 10 bp: +10.7 bp, t 2.3), but scattered and inconsistent - SOL's
  stricter threshold is worse, neighbouring offsets flip sign. With 64 cells, t ~ 2 is what chance
  gives. No robust pattern beyond "8 h holds look better than 4 h".
* **Adverse selection is clear:** the signals whose limit order was NOT filled would have earned
  +12..+75 bp gross - the limit fills exactly when the move goes the wrong way first.
* Verdict: the quarter-hour opening signal stays a component, not a strategy. Next: H8 (levels x flow).

## 2026-10-29 — Basket scan: OpEx, close momentum, FX fix - effects too small for CFD costs

`studies/baustein_scan.py` (measurements, train only):

* **Options expiration week** (indices 2004-2024): OpEx week +2.5..+3.0 bp/day vs +3.3..+5.6 on
  other days (t -0.1..-0.7) - no effect.
* **Intraday momentum into the close** (Gao/Baltussen, 2019-2024 1m): correlation +0.05..+0.08,
  trading the sign +1.0..+1.5 bp (t 1.1-1.7) vs ~1.7-2.2 bp round trip - below costs; big days no better.
* **FX fix reversal** (Evans): the move 15:30->16:00 London reverses 16:00->16:30 - statistically
  real (t 3.3-5.6 on EURUSD, GBPUSD, AUDUSD, EURGBP) but only +0.7..+1.1 bp vs 1.7-3.2 bp cost;
  month end GBPUSD +3.4 bp / EURGBP +2.6 bp (t ~2.5, n 72) ~ cost. Not tradable with market orders.
* Lesson: intraday flow effects exist but are 1-3 bp; retail CFD costs are 2-3 bp. Basket
  blocks have to come from multi-day holding periods.

## 2026-10-29 — Crypto order flow, H8: SMC levels x order flow - no edge, levels = random levels

`studies/crypto_level_flow_study.py` on `trading_lab/level_flow.py` (+ `tests/test_level_flow.py`).
BTC + SOL perps 2024 (train), Binance flow and prices, 10-s bars. Levels: previous day / week
high-low (UTC), external swings, EQH/EQL, order blocks (15-min `market_structure`), previous-day
POC/VAH/VAL. First touch per level, 5-min reaction window, labels with thresholds fixed before
the run (flow into the level >= 0.15, through = close >= 0.25 ATR beyond): absorption / reclaim
(fade) and cascade / break (follow). Placebo: the same levels moved 0.15-0.5 %, 3 seeds.
12,820 real events, 38,139 control events.

* **No label is positive after costs** (~11.4 bp taker round trip) in the pooled tables; real and
  control levels behave alike - the SMC levels carry no information a random nearby price
  does not (as with fib levels in S-002).
* **Flow does not improve SMC's reading:** absorption (strong flow in, close back) faded is
  *worse* than reclaim without flow (4 h: -8.1 vs +1.4 bp both markets; BTC -10.8 bp, t -2.2;
  order blocks -18.9 bp, t -2.0). Read the other way it says the flow direction tends to win even
  when the first 5 minutes reject - in line with Kim & Hansen's flow continuation - but this is a
  post-hoc reading of one cell family and needs its own pre-registered test on other data.
* "Break" without flow mean-reverts strongly at real AND control levels (4 h -12.9 vs -15.1 bp):
  a generic reversal after a 0.25-ATR push, not a level effect.
* Killzones: no zone turns a label positive net with a control that differs (Asia reclaim +12.9
  gross vs control +10.9).
* Limits: one year, Binance flow only (Bybit download still running), one set of thresholds,
  5-min window, taker costs. Verdict: H8 as specified rejected; not worth a spec.

## 2026-10-29 — Quick check FX value (Asness/Moskowitz/Pedersen): decayed, does not help the basket

`studies/fx_value_check.py`, FRED spot + CPI 1980-2024, 7 currencies vs USD, monthly, ~8 % vol,
price only (FundedNext FX swaps ~0).

* Time-series value (long cheap, short expensive currency): Sharpe 0.05 - nothing.
* Cross-sectional value (rank-weighted, dollar neutral): Sharpe 0.50 over 1980-2024 (t 3.35)
  but decaying by decade: 0.68 / 0.86 / 0.46 / 0.32 / **-0.40 (2020s)**; 2004-2024 only 0.20,
  11/21 positive years. Correlation with S-007 0.00.
* With S-007 at Sharpe ~0.63, an uncorrelated partner only helps above ~0.26 (equal risk):
  value at 0.20 lowers the basket. Not a building block - another published effect that
  faded after publication.

## 2026-10-29 — Crypto: VPIN as a risk filter - adds nothing beyond realised volatility

`studies/crypto_vpin_risk_filter.py` (+ `tests/test_crypto_vpin_filter.py`), BTC + SOL perps 2024
(train), Binance. VPIN with the paper's defaults (V = January's average daily notional / 50,
n = 50), taker-side and BVC classification, sampled hourly; targets: |return| and realised
volatility over the next 1 h / 4 h, jump = next-hour |return| > 3 x 7-day hourly std.

* Unconditionally VPIN sorts risk: top vs bottom quintile, 1-h jump rate 1.0 -> 3.2 % (BTC, BVC),
  0.9 -> 2.4 % (SOL, BVC).
* **Controlling for past 1-h and 24-h realised volatility, VPIN adds nothing**: R² gain 0.0-0.4
  points, and the coefficient is *negative* (t -1.5 .. -4.2) - at a given volatility, high VPIN
  is followed by slightly calmer hours. Within volatility terciles the jump rates of the top-VPIN
  hours are higher in some cells and lower in others (few jumps per cell).
* Matches the known critique (Andersen & Bondarenko 2014, J. Financial Markets): VPIN's
  forecasting power is mostly volatility and trading intensity. For a prop account's daily-loss
  protection a plain realised-volatility filter does the same job without trade data.

## 2026-10-29 — Crypto order flow, H2: Binance + Bybit aggregated - no gain; line closed

Kim & Hansen measurement (`studies/crypto_kim_hansen_replication.py --venues ...`), 2024-01..10,
opening imbalance from Binance alone, Bybit alone, and both summed (base-asset sizes):

| 4 h slope, bp per unit OI | Binance | Bybit | Binance + Bybit |
|---|---|---|---|
| BTC | +2.4 (t 1.5) | +0.8 (t 0.7) | +1.8 (t 1.1) |
| SOL | +4.5 (t 1.9) | +3.4 (t 1.8) | +4.8 (t 2.0) |

* Bybit's flow is weaker than Binance's, and summing the two does not beat Binance alone
  (BTC worse, SOL equal) - consistent with Binance as the price leader (Albers et al. 2021).
  The top-10 % |OI| gross stays below costs in every variant (BTC combined +2.5..+6.4 bp).
* **Line closed** (dossier `docs/strategies/crypto_orderflow_REJECTED.md`): the quarter-hour
  effect is real but too small (2-9 bp vs ~11 bp), extreme/limit entries are not robust and
  suffer adverse selection, SMC levels x flow behave like random levels, VPIN is volatility,
  multi-venue aggregation adds nothing. Hyperliquid TWAPs (E5) not tested - no free history,
  and a public signal on a venue no prop firm executes on.

## 2026-10-29 — S-005: Weekly level break (own find from step B) - fails S4

Spec `docs/specs/weekly_level_break.yaml` (normal track), 8 markets pooled, 8 variants, train
2019-2024, ftmo costs + 0.5 bp, measured FundedNext swaps.

* The continuation is real before costs: gross +0.10..+0.19 R per trade vs placebo (same
  bracket, random bars at the same hour, random direction) +0.01..+0.03 R.
* Costs eat most of it: 15m (1 ATR on 15m is small) net -0.05..-0.07 R; 1h 'durch' net
  +0.052 R (3R) / +0.060 R (5R), 5/6 positive years but 2023 -35..-52 R; t 0.77 / 0.50,
  DSR 0.24 / 0.16, PBO 0.36 -> **0 of 8 pass S4**. Holdout untouched.
* Same pattern as S-004: a real but small effect; the 1-ATR stop on intraday bars makes the
  round trip 0.08-0.2 R.

## 2026-10-29 — Wider trend universe (49 instead of 26 markets): worse, not better

`studies/tsmom_universe_check.py`: S-007 rules on the 26 markets plus 23 further FundedNext
markets (AUS200, CAC, STOXX 50, IBEX, Hang Seng, platinum, palladium, 14 FX crosses, USDSEK,
USDNOK), train 2004-2024: Sharpe gross 0.79 -> 0.71, net 0.64 -> 0.46, 14/21 -> 12/21 years;
FX crosses +37 R -> -1.5 R. With a fixed budget per asset class, more markets of the same
class only dilute it ("Trends Everywhere" gains come from NEW classes). S-007 stays at 26.

## 2026-10-29 — Quick check cross-sectional momentum (26 markets): weak and too close to S-007

`studies/xsmom_check.py`: 12-1 month rank momentum within each class, vol-scaled, monthly,
2004-2024: Sharpe 0.30 gross / 0.27 net (t 1.26), 12/21 years; equities +0.42, real assets
+0.45, FX crosses +0.29, FX vs USD -0.06. **Correlation with S-007 +0.69** - a 25 % mix lowers
the basket Sharpe from 0.63 to 0.57. Not a building block.

## 2026-10-29 — Step C: round numbers (Osler) on 1m-4h - no range edge; break effect only where costs are 0.4-1.7 R

`studies/round_number_study.py` (measurement, train 2019-2024): 7 markets (5 FX, USDJPY, gold),
round grid (x.xx00/x.xx50, JPY 0.50, gold 5) vs a control grid of the same density shifted
by 37 %, first touch per level and day, bracket from the bar close (stop 1 ATR, target k ATR).

* **Range fade at round numbers ("abprall"): no effect.** Round minus control -0.03..+0.03 R
  on every timeframe; from 15m up slightly WORSE than control (1-3 of 7 markets positive).
* **Continuation after a close through a round number ("durch"): small, real, short-lived** -
  +0.09 R (1m), +0.05 R (5m, 15m; 15m 7/7 markets) vs control, t >= 2 in up to 3 of 7
  markets - Osler's stop clustering behind round numbers. Gone from 30m up (1h-4h negative).
* Round-trip cost per 1-ATR stop: 1.7 R (1m), 0.68 (5m), 0.37 (15m), 0.25 (30m), 0.18 (1h),
  0.12 (2h), 0.09 (4h). The effect lives exactly where costs are 4-20x larger. Not tradable.

## 2026-10-29 — Crypto H9: follow the flow after absorption at SMC levels - PRE-REGISTRATION

Written before any 2023 data was downloaded or looked at. Origin: a post-hoc reading of H8
(above): fading "absorption" lost money at 4 h (BTC -10.8 bp, t -2.2; order blocks -18.9 bp), so
the flow direction may win even when the first 5 minutes reject the level (Kim & Hansen flow
continuation). H9 tests that reading as its own hypothesis on data H8 never saw.

* **Data (decisive sample):** BTCUSDT + SOLUSDT perps, **2023-01-01 .. 2023-12-31**, Binance USD-M
  aggTrades -> 10-s taker-signed bars (`trading_lab/adapters/crypto_trades.py`), Binance flow AND
  prices as in H8. Holdout (2025+) untouched. 2024 (H8's sample, cached) is reported as a second,
  separate sample only - it generated the hypothesis and cannot confirm it.
* **Levels (unchanged from H8, `trading_lab/level_flow.py`):** previous UTC day / week high-low,
  external swings, EQH/EQL pools, order-block edges (15-min `market_structure`), previous-day
  POC/VAH/VAL (bins BTC 25, SOL 0.10). Each level from `known_at` on, first touch within 3 days.
* **Event = absorption (H8 definition, thresholds unchanged):** reaction window = the 30 10-s bars
  (5 min) from the touch bar; approach side d = +1 if the level was above price when it became
  known, else -1; flow = taker imbalance (buy - sell) / (buy + sell) notional over the window;
  absorption iff d x flow >= 0.15 AND the window's last close is not beyond the level
  (beyond_atr <= 0, ATR(14) of 15-min bars known at the touch).
* **Trade: follow the flow = +d** (into/through the level, against the 5-min rejection), entry at
  the close of the window's last bar, exit after **4 h** (primary); 1 h secondary.
* **Costs:** net = gross - `costs.crypto_round_trip_bp(market, "bybit_perp")` per event
  (taker both legs + half spreads). Funding is not modelled (a 4 h hold crosses at most one 8-h
  fixing, typically ~1 bp, either sign).
* **Statistics:** mean signed bp per event, t clustered by UTC day (per-day means), as in H8.
* **Controls:** (a) random levels - the same levels moved by +-0.15..0.5 % (`control_levels`,
  seeds 1-3), same event rule and trade; difference real - control with a day-clustered t
  (OLS on a real/control dummy, cluster = day). (b) random direction - the same real events with
  a random +-1 direction, 2000 draws; one-sided p = share of draws whose mean >= the observed one.
* **Verdict "confirmed: candidate for a spec (user must agree)" only if ALL hold on 2023, pooled
  BTC + SOL, 4 h:** C1 gross > 0 with day-clustered t >= 2.0; C2 net > 0; C3 real - random-level
  control > 0 with t >= 1.65; C4 random-direction p < 0.05; C5 gross > 0 in BTC and SOL separately.
  Anything else: **rejected**. If C1/C2/C4 hold but C3 fails, the result is "flow continuation,
  not a level effect" - rejected as a levels x flow hypothesis.
* **Reported, no weight in the verdict:** 1 h horizon; order blocks alone; per quarter 2023; 2024
  second sample; Bybit flow and prices for 2023 (if downloaded); a non-overlapping version (per
  market, skip events within 4 h of the last kept one); comparison rows "follow +d at every
  touch" and "cascade" (flow in, close through).
* Code: `studies/crypto_absorption_follow_study.py` + `tests/test_absorption_follow.py`; no
  threshold, window, horizon or level is changed after the first look at 2023.

## 2026-10-29 — Quick check: Krohn/Mueller/Whelan long fix windows on our data - every window negative net

`studies/fx_fix_windows.py`, 7 USD pairs, 2019-2024 (train), 1m bid/ask + 0.5 bp per side. Windows as in
the paper's Section V.E (USD long 02:00 NY -> ECB fix; foreign long London fix -> 17:00 NY; Tokyo pair).
* Mid: the pattern is partly visible (+0.4..+1.7 bp per window; Tokyo windows t up to 2.9), weaker than
  the paper's ~2.5 bp/day.
* Net: **all 28 pair-windows negative**, portfolio -1.1 (pre-ECB) to -4.3 bp/day (pre-Tokyo). Windows that
  start or end at 17:00 NY pay the rollover spread spike. Matches the paper (full spreads -> negative);
  not tradable for a retail CFD account.

## 2026-10-29 — Step D: own range determination (time-at-price, POC / value area) - no better than plain range levels

`studies/value_area_study.py` (measurement, train 2019-2024, 8 markets, 15m + 1h): previous day's
time-at-price profile (50 bins, 70 % value area) -> POC / VAH / VAL; control = previous day's range at
50 / 85 / 15 %. First touch per level and day, bracket from the bar close (stop 1 ATR, target k ATR).
* Range fade at VAH/VAL ("abprall"): profile minus control -0.05..+0.04 R, no market with t >= 2 more than
  once; the value-area edges are not better turning points than plain range fractions.
* Only lead: break of the previous day's POC on 1h, +0.06 R vs control, 6/8 markets, t >= 2 in 1/8 -
  still net -0.04..-0.07 R after costs.
* Net on every cell negative (-0.04..-0.34 R). Time-at-price adds no tradable information here.

## 2026-10-29 — Crypto H10: volume-profile levels (magnet / hold / SMC + flow) - no better than random levels

Pre-registration `docs/specs/h10_volume_profile_PREREG.md` (commit 11f74bb, before any data was
read). `trading_lab/volume_levels.py` (+ `tests/test_volume_levels.py`), `studies/crypto_volume_profile_study.py`;
report `artifacts/crypto/vp_2024.md`. BTC + SOL Binance perps, 1-min bars from the 10-s taker flow.
9 windows (last 96/288/672 15-min bars, UTC day/week, Tokyo/London/NY session, leg between external
swings) x POC/VAH/VAL/HVN/LVN (50 rows, 70 % VA) + naked daily POC; 90k levels, 71k magnet events,
45k hold events (+138k control). Context: there is no academic specification of profile settings;
Kavajecz & Odders-White (2004) tie support/resistance to order-book depth, Osler (2003) to stop/TP
clustering - neither says traded volume marks those prices. Step D (above, CFDs) found the same
for time-at-price.

* **Magnet (24 h, vs a level at the same ATR distance and side at a random other time):** 45/46 cells
  within +-5 pp or |t| < 3; most effects are slightly NEGATIVE (POC/VA reached 1-5 pp less often than
  the control, the mirror level is reached as often or more). **The "80 % return to the prior POC" is
  not there:** prior-day POC reached within 24 h 48.9 % (control 52.3 %), weekly POC 41.3 % (43.0 %),
  naked POC 30.0 % (33.2 %). High rates (70-80 % for leg / session levels) come from the short
  distance (median 0.6-1.2 ATR1h) and are matched by the control.
* **Hold (bounce 0.5 ATR before break 0.5 ATR, vs shifted levels):** no level type holds better than
  random. Leg POC/VAH/VAL break MORE often (-3.5..-3.8 pp, t -2.4..-3.9, < 5 pp so not a candidate).
  Fade 4 h net negative in almost all large cells.
* **Interactions:** SMC confluence (0.1 ATR15, 71-77 % of levels flagged - the SMC level set is dense)
  changes nothing (-1..+1 pp). Taker flow into the level on the approach lowers the hold rate by
  4-8 pp (t -2.8..-4.0) - but the same split on the CONTROL levels gives the same -5..-9 pp: flow
  continuation (Kim & Hansen), not a level effect.
* **2024 discovery candidates (|effect| >= 5 pp, |t| >= 3):** magnet `week:hvn` +13.4 pp (t 3.47,
  n 173, BTC and SOL positive); hold `asia:val` -9.0 pp (t -3.07, holds LESS); interactions
  `poc:flow` -7.5 pp / `vah:flow` -7.3 pp (both mirrored by the control). 4 of 97 cells at roughly a
  Bonferroni threshold - about what chance plus the generic flow effect give.
* **2023 confirmation (run once, unchanged code; `artifacts/crypto/vp_2023.md`, `vp_verdict.md`;
  87k levels, 68k magnet, 44k hold events):** `week:hvn` magnet +5.9 pp, t 1.58 - fails;
  `asia:val` -3.0 pp, t -0.39 - fails; `leg:vah` breaks more again (-3.4 pp, t -2.94, both markets
  -3.4 pp) but stays below the 5 pp bar in both years. Prior-day POC within 24 h 51.1 % vs control
  51.8 %; naked POC 27.3 % vs 30.3 %. Confluence again 0 pp.
* **Interactions `poc:flow` (-10.5 pp, t -3.70) and `vah:flow` (-11.4 pp, t -5.18) pass the rule
  formally - but the identical split on the shifted CONTROL levels gives -11.3 / -12.7 pp. Flow into
  any price lowers the bounce rate (flow continuation, as in H8/H9 and Kim & Hansen); the profile level
  adds nothing. Not a volume-profile effect.**
* **Verdict (pre-registered rule): 0 of 87 magnet/hold cells confirmed - volume-profile levels on
  BTC/SOL perps are no better than random levels**, for every window (fixed bars, sessions, legs) and
  every node type; no "magnet", no "hold", no SMC synergy. Together with H8 (previous-day POC/VA) and
  step D (CFD time-at-price) the volume-profile line is closed. Leads worth nothing on their own: leg
  value-area levels break ~3.5 pp more often than random (both years, < 5 pp); generic taker-flow
  continuation at touches (already the H9 question).

## 2026-10-29 — H11-B quick check: NY opening range + untouched 1h level as target - level = random level

User idea (chart, NQ 5m): the first 30 min of New York draw a range 0 / 0.5 / 1, an untouched 1h
swing high/low beyond it is the target, and two different paths lead there - "deep" (retrace to 0.5,
limit there, stop 0) or "shallow" (break of 1, stop order, stop 0.5). Quick measurement, no spec:
`studies/h11_target_first_quick.py` (+ `tests/test_h11_target_first.py`), report
`artifacts/h11_target_first_quick.json`. US100, mid space, `build_study_frame`, NY clock, entries
10:00-11:59, flat 15:55, stop wins ties, target only from the bar after the fill. **2016-2021 only**;
2022-2024 untouched for a confirmation, holdout closed. Placebo: same days and paths, target distance
(in range units) permuted across the days of the group, 1000 shuffles.

| dir | path | n | target hit | driftless p0 | R gross | placebo | p | R net |
|---|---|---|---|---|---|---|---|---|
| long | deep | 161 | 27 % | 31 % | -0.13 | -0.09 | 0.82 | -0.17 |
| long | shallow | 197 | 59 % | 57 % | +0.04 | +0.05 | 0.57 | -0.01 |
| short | deep | 240 | 26 % | 29 % | +0.05 | +0.04 | 0.44 | +0.01 |
| short | shallow | 166 | 55 % | 52 % | +0.07 | +0.03 | 0.24 | +0.02 |

* Hit rates sit on the driftless first-passage value for the given stop/target geometry; the real 1h
  level beats its shuffle in no group. Net within +-0.02 R, sign flips year to year.
* Same verdict as H8/H10 and step C/D: a "liquidity" level is reached as often as any level at the
  same distance. The path classes (deep vs shallow) only change the geometry, not the odds.
* Not tested: token-learned path classes (H11 token stage). Expectation lowered - the setup itself
  carries no effect, so tokens would have to supply all of it.

## 2026-10-29 — H11-A quick check: a big London/NY move is predictable from the pre-session range (the bar for tokens)

`studies/h11_move_forecast_quick.py`, report `artifacts/h11_move_forecast_quick.json`. Mid space,
`build_study_frame`, exchange-local clocks, **2016-2021 only**. y = range of the first 4 h of the
session / its 20-session median (prior only); "big" = top quartile of y. Predictors: x_pre = 6 h
pre-session range / own 20-session median, x_rel = pre range / session median, x_prev = yesterday's y,
x_reg = 20/100-session median ratio.

| asset:session | n | AUC x_pre | AUC x_rel | P(big) top quintile x_rel | room (median range / spread) |
|---|---|---|---|---|---|
| US100 ny | 1389 | 0.70 | 0.71 | 53 % | 66 |
| SP500 ny | 1249 | 0.76 | 0.76 | 59 % | 41 |
| US30 ny | 1390 | 0.74 | 0.74 | 57 % | 85 |
| GER40 london | 766 | 0.67 | 0.67 | 47 % | 80 |
| UK100 london | 616 | 0.72 | 0.70 | 55 % | 50 |
| XAUUSD london | 1374 | 0.63 | 0.66 | 45 % | 19 |
| XAUUSD ny | 1403 | 0.65 | 0.67 | 44 % | 28 |

* Base rate 25 %; bottom quintile 8-17 %. AUC > 0.5 in every year for the US indices (0.58-0.84).
  This is volatility clustering, not news timing (P(big) by weekday flat, 20-33 %). GER40 starts
  2018 here (pre-session coverage before 2018 below 80 %).
* x_reg < 0.5 is mostly the normaliser (a rising median shrinks y), not a signal.
* Meaning for H11: a token reading of the pre-session must beat AUC ~0.70 (indices) to add anything.
  A predicted big range is not yet a trade - direction is the open question (stage 2: tokens /
  two-sided breakout / EMA trend / 15m BOS as a fixed list).

## 2026-10-30 — H11 stage 1: chart tokens add nothing beyond recent volatility - H11 closed

Pre-registration `docs/specs/h11_tokens_PREREG.md` (released 2026-10-30, code committed before the
run, 666c4c3). `trading_lab/chart_tokens.py`, `studies/h11_tokens_study.py`, report
`artifacts/h11_tokens_discovery.json`. 7 asset-sessions, 2016-2021, 200 readings (SAX N/W/A and
candles 1-5 x TF 5/15/30/60 x volume letter), stratified AUC inside the pre-session-range quintiles,
leave-one-year-out, max-statistic over the grid (1000 shuffles).

* **Verdict per prereg: no reading has p_fw <= 0.05 -> H11 closed, no confirmation run, stage 2
  not run.** Best: K1@15m+v 0.583 (p_fw 0.19), S12w3a3@60m+v 0.581, K1@30m+v 0.578. Median of all
  readings 0.549. Null maximum: median 0.539, 95 % 0.639.
* **Design weakness, reported:** readings with 12-16 hourly bars need 12-16 h of gap-free pre-open
  data, which the daily CFD break rarely allows (S16@60m: n = 109). Those tiny readings swing
  wildly (real AUC down to 0.34 - leave-one-year-out on sparse tokens) and inflate the null maximum;
  a single large reading's own null 95 % is only ~0.52. A minimum n per reading should have been in
  the prereg. **It does not change the conclusion**, see next point.
* **Diagnostic (discovery years, already seen, no new data): the shape carries nothing.** Splitting
  the best readings into their parts, each with its own null (300 shuffles):

  | reading | full token | size + volume letters only | shape only |
  |---|---|---|---|
  | K1@15m+v | 0.583 | 0.580 | 0.507 (p 0.22) |
  | K1@30m+v | 0.578 | 0.583 | 0.490 (p 0.82) |
  | K1@15m | 0.566 | 0.554 | 0.507 (p 0.23) |

  The real, stable part (every asset-session 0.54-0.63, 5/6 years > 0.5) is **how big and how
  active the last 15-30 min before the open were** - finer-grained volatility clustering on top of
  the 6 h range. The candle / path shape adds 0.00-0.01. "Patterns" in the shape sense: none.
* Lessons for H12 (fuzzy frequent patterns): a minimum n per reading, and a **size/volume-only
  control** in every comparison - otherwise a pattern search rediscovers volatility.

## 2026-10-30 — H12: token-combination mining - one directional candidate, about the size of costs

Pre-registration `docs/specs/h12_pattern_mining_PREREG.md` with its amendments; `trading_lab/pattern_mining.py`,
`studies/h12_mining_study.py`, reports `artifacts/h12/validation.json`, `artifacts/h12/confirmation.json`.
6 markets, decision every 15 min London 08:00 - NY 15:00, move = first passage of a top-decile hour
(1.6-1.9 ATR1h) within 60 min. Mining 2018-05..2020 (gold from 2016), validation 2021, confirmation
2022-2024, run once. Holdout closed.

* **Track B (session sentences Asia -> London -> NY):** 143 units met the frequency rule, 0 mining
  candidates. Closed.
* **Track A:** 6268 units met the frequency rule, 300 mining candidates, 173 validated, top 10 sent on.
  Theme: new session low / sharp drop in the last 5-60 min -> more big moves.
* **Most of this is volatility, not direction.** Up-moves rise too (2021: down 15 % vs control 10 %,
  up 11 % vs 7 %). The volatility control (market x clock x last-1-h range quintile) was too coarse,
  so the written confirmation rule could be passed by volatility alone. **Amendment after seeing
  validation, before touching 2022-2024, stricter, decided by the user:** confirmed only on
  directional excess `(y_D - y_O) - (p_D - p_O)`, z >= 2.58, > 0 in >= 4 markets.
* **Confirmation: 1 of 10 passes** (the old rule alone would have passed 4 of 10).

  **5-min word `cca` family (sharp drop over the last 30 min, large range, high volume) + 1-h trend up
  -> short side** (5 variants: cca/l/v3, bca/l/v3, cba/l/v3, cca/m/v3, cca/l/v2).

  | period | n | P(down) / control | P(up) / control | directional excess | z | per week per market |
  |---|---|---|---|---|---|---|
  | mining | 6992 | 9.6 / 7.0 % | 4.3 / 5.3 % | +3.7 pp | 6.65 | 5.2 |
  | validation 2021 | 2333 | 11.1 / 7.6 % | 7.5 / 5.5 % | +1.5 pp | 1.46 | 5.0 |
  | confirmation | 6626 | 10.3 / 8.1 % | 6.8 / 6.7 % | +2.0 pp | 3.24 | 4.9 |

  Confirmation by year +1.7 / +2.2 / +2.1 pp; by market SP500 +3.3, XAUUSD +2.9, GER40 +2.5, US30 +2.5,
  US100 +0.3, UK100 -0.2. Its 15-min sister (`b15+trend`) misses: +1.5 pp, z 1.84. Direction is
  continuation of the drop, against the 1-h trend - "buy the dip" is the wrong side here.
* **Economics, rough:** 2.0 pp x ~1.75 ATR1h = ~0.035 ATR1h per signal, counting only the tail events.
  Round trip on the Dukascopy spread: 0.033-0.036 ATR (US100, US30, GER40), 0.050 (SP500), 0.07 (UK100,
  XAUUSD). So the effect is **about the size of the cost**; no trade was simulated. The rest of the
  return distribution (the ~83 % of signals without a move) is unmeasured.
* Caveats: the verdict rule was changed after validation (stricter, documented); the top 10 were chosen
  by the old, volatility-contaminated rule; validation alone had z 1.46. Not a Baustein yet.

**Correction to H11 (2026-10-30):** `studies/h11_tokens_study.py::_ratio` voided a token for 20 days after
every gap (rolling median with `min_periods=NORM` on a series with NaN) - the cause of n = 109 on S16@60m
and of lost rows elsewhere. Found and fixed for H12 (`prior_median`), not rerun for H11. Its conclusion
rests on the large readings (~7900 rows, shape alone 0.49-0.51), but the bug was in the run.

## 2026-10-30 — Multi-day scan: correlation filter confirms Hurst, vol-managed equity premium helps the basket, reversal does not

`studies/multiday_scan.py` (measurements, train 2004-2024, measured FundedNext swaps).
* **Correlation filter for S-007:** Sharpe by quintile of lagged market correlation 1.04 / 0.46 / 1.40 /
  0.24 / -0.13 (low -> high); scaled risk 0.66 -> 0.74, DD unchanged. Confirms Hurst et al., but S-007's
  holdout is spent, so a filter version has no clean test.
* **Vol-managed equity premium** (6 indices long, 21-day vol sizing, weekly): Sharpe 0.50 (t 2.33), 15/21
  years, correlation with S-007 +0.14; 2008 -36 %, 2022 -17 % on this (too large) scale. **Basket with S-007
  (equal risk): Sharpe 0.75 vs 0.64.** On FundedNext Instant at 4 % vol: 3.0 % p.a. vs 2.5 %, DD -8.3 % vs
  -9.0 %, survival 91 % vs 90 %. A modest gain; the holdout of this block is untouched.
* **Index reversal after 3-day drops (< -2 sigma):** +24.7 bp vs +19.4 bp unconditional over 5 days,
  Sharpe 0.19 - lowers the basket.
