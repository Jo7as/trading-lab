# H11: Chart tokens as precursors of big London / NY moves - PRE-REGISTRATION

Written 2026-10-30, before any token was computed on data. **Status: DRAFT - waits for the
user's release.** Measurement only, no strategy. Planned code: `trading_lab/chart_tokens.py`
(+ `tests/test_chart_tokens.py`), `studies/h11_tokens_study.py`. Nothing below (token
definitions, timeframes, target, baseline, score, thresholds, verdict rule) is changed after the
first run on the discovery years.

## Why

The user's idea: translate the chart into tokens (letters for small pieces of price action),
then look for "sentences" - single tokens or short sequences - that announce a big move in the
London or New York session, and let several different sentences lead to the same outcome
(synonyms). Same shape at a different price must be the same token; size and volume must be
similar for two pieces to count as the same.

Two quick checks came first (`docs/FINDINGS.md`, 2026-10-29, commit bbc2cc9):

* **H11-B** (NY opening range + untouched 1h level as target, "deep" vs "shallow" path): the real
  level is hit no more often than a level at the same distance (all shuffle p >= 0.24), net
  +-0.02 R. **Not repeated here.**
* **H11-A** (no tokens): the 6 h pre-session range already predicts a big session move,
  AUC 0.67-0.76 on the indices, 0.63-0.67 on gold, stable in every year. **This is the bar.** A
  token reading only counts if it adds information *beyond* that range - otherwise it is a
  complicated way of measuring volatility clustering.

Literature context: SAX (Lin, Keogh et al. 2003/2007) is a representation, not a forecasting
claim; candlestick-pattern studies on liquid markets mostly find no net edge after costs (e.g.
Marshall, Young & Rose 2006 on DJIA stocks). Prior on a net-tradable result: low.

## Data

* Dukascopy 1-min bid/ask, **mid** prices, `data_audit.build_study_frame`, exchange-local clocks
  (`docs/TIME_CONVENTIONS.md`).
* Asset-sessions (as in H11-A): US100, SP500, US30 - `ny` (America/New_York 09:30);
  GER40, UK100 - `london` (Europe/London 08:00); XAUUSD - `london` and `ny`. 7 asset-sessions.
* **Discovery: 2016-2021** (GER40 from 2018, UK100 from 2019 - pre-session coverage).
  **Confirmation: 2022-2024**, run once, reported separately. **Holdout 2025+ is not opened.**
* Volume: Dukascopy's own volume (no exchange volume). Checked 2026-10-29: no gaps, plausible
  intraday profile, rank correlation with the bar range 0.71-0.80 (1 min) and 0.55-0.65 relative
  to the same clock (15 min) - it is mostly range, partly something else of unknown meaning.

## Target (stage 1)

Per asset-session and day: `y` = range (mid high - low) of the first **4 h** after the open /
median of the same window over the previous 20 sessions. **big** = `y` >= the 75th percentile of
`y` in that asset-session's discovery years (number frozen and reused for confirmation).

## Baseline (the bar)

`x_rel` = range of the 6 h before the open / 20-session median of the session window (H11-A).
Quintile edges of `x_rel` per asset-session from the discovery years, frozen.

## Tokens - a fixed grid of **200 readings**

Each reading uses **one timeframe only**: TF in {**5, 15, 30, 60 min**} (mid bars resampled from
1 min, aligned to the session open). 1 min (noise, moves below the spread) and 2 h / 4 h (2-3
tokens per session) are left out on purpose. All tokens use only bars that **close at or before
the open**.

The user wants to try many window lengths and timeframes. That is allowed **inside this grid,
written down now**; the multiple testing is paid for by a max-statistic permutation (below), so a
bigger grid automatically means a higher bar. Nothing outside the grid is tried.

**Family S - shape of the path (SAX).** The last N TF bars before the open, N in
{4, 6, 8, 12, 16}: z-normalise their closes (std 0 -> all middle letters), PAA to W segments,
W in {3, 4} (W <= N), alphabet A in {3, 4} with the normal breakpoints (A=3: -0.4307 / +0.4307;
A=4: -0.6745 / 0 / +0.6745) -> a word like `acbc`. Same shape at any price and any size -> same
word. 5 x 2 x 2 = 20 settings.

**Family K - candles.** The last C TF bars, C in {1, 2, 3, 4, 5}, each one of 5 shapes, from its
own range only:
`D` doji (|close - open| < 0.1 x range or range 0), `U` strong up (open in the lower third and
close in the upper third of the bar), `N` strong down (mirror), `u` other up, `n` other down ->
a word like `UnD` (5^C words). Same candle at any price -> same letter. 5 settings.

**Grid:** (20 S + 5 K settings) x 4 TF x volume on/off = **200 readings**.

**Size letter (both families, always on).** Range of the bars the word covers / median of the
same bars' range (same clock) over the previous 20 sessions -> `s` / `m` / `l` by the discovery
tertiles of that ratio (per asset-session, frozen).

**Volume letter (the "with volume" readings).** Sum of volume over the same bars / its median
over the previous 20 sessions at the same clock -> `v1` / `v2` / `v3` by discovery tertiles. The
same-clock normalisation keeps the letter from measuring the time of day.

A token = word + size letter (+ volume letter). Large vocabularies (e.g. K with C = 5: 9375
tokens with volume) are sparse by design; the shrinkage below pulls rare tokens to the base rate,
so they cannot win by luck on a handful of days.

## Score

The token score is the shrunk big-move rate of the token, **pooled over all 7 asset-sessions**
(tokens are price- and size-normalised, so the same sentence is allowed to mean the same thing
on every market):

    score(token) = (big_count + 20 x base) / (n + 20)        base = pooled big rate (~0.25)

Synonyms are handled here: every token that leads to big moves gets a high score, no matter how
many different ones there are.

* **Discovery:** leave-one-year-out - the score for year Y is learned on the other discovery
  years only, so the discovery AUC is out of sample.
* **Confirmation:** the score is learned on all discovery years, then applied to 2022-2024 once.

## Statistic (stage 1)

**Primary: stratified AUC** of the token score for `big`, inside the `x_rel` quintiles of each
asset-session (AUC per stratum, averaged with stratum sizes as weights). A token reading that
only re-measures volatility has a stratified AUC of ~0.50.

* **Family-wise p (max statistic):** the `big` labels are shuffled inside (asset-session x year x
  `x_rel` quintile) strata, 1000 times; for each shuffle the scores of **all 200 readings** are
  re-learned (leave-one-year-out, as for the real data) and the **largest** stratified AUC over
  the grid is kept. p_fw(reading) = share of shuffles whose maximum >= the reading's real
  stratified AUC. This is the "best of 200 by luck" bar - trying more settings raises it.
* Reported, no weight: the plain AUC of the score, the AUC of `x_rel` alone, per asset-session
  and per year, and the 10 tokens with the highest discovery score (n >= 30), each with its
  count and big rate.

## Verdict stage 1 (fixed now)

1. **Discovery** (leave-one-year-out): readings with p_fw <= 0.05 are candidates. At most the
   **3 best** (highest stratified AUC) go on. No candidate -> H11 is closed.
2. **Confirmation 2022-2024**, run once, only on those <= 3 readings with the score learned on
   all discovery years. A reading **passes** iff stratified AUC > 0.50 with a permutation p
   <= 0.05 / (number of readings sent) (labels shuffled inside asset-session x year x quintile,
   2000 times), and its stratified AUC > 0.50 in at least 5 of the 7 asset-sessions.

The confirmation bar is deliberately low ("real and on most markets"), and it is **not moved
after the discovery result is known**. Loosening is only possible by changing this document
before the discovery run.

**No reading passes -> H11 is closed** ("chart tokens add nothing beyond the pre-session range").
Stage 2 is then not run.

## Stage 2 - direction (only if at least one reading passes stage 1)

Run on the best passing reading only (highest confirmation stratified AUC). **Flagged day**:
predicted P(big) >= 0.40 from a logistic regression of `big` on `x_rel` and the token score,
fitted on the discovery years.

Fixed list of **4 direction rules**, identical mechanics: one trade per flagged day and
asset-session, stop = 1.0 x ATR(14) of 1-h bars (`indicators.atr`, known at the open), time exit
at the end of the 4 h window, no target. Cost: spread of the entry bar x
`cost_gate.round_trip_spread(entry, exit)` (market or stop entry; stop / time exit).

| rule | direction | entry |
|---|---|---|
| D1 tokens | sign of the shrunk P(up-move | token), up-move = session close > open; no trade if \|P - 0.5\| < 0.05 | market at the open |
| D2 breakout | the side that breaks first | stop orders at the high / low of the first 30 min, OCO, until window end |
| D3 EMA trend | sign(EMA20 - EMA50) on 1-h bars at the open (`indicators.ema`) | market at the open |
| D4 15-min structure | direction of the last BOS event of `market_structure.analyze` on 15-min mid bars before the open | market at the open |

Reported for every rule: the same rule on **non-flagged** days (does the flag help?) and a
**random-direction placebo** on the flagged days (same entry, stop, exit).

**Verdict stage 2.** A rule passes iff
1. discovery: mean net R > 0 with t >= 2.5 (Newey-West, 5 lags, over the daily series), and
   flagged mean net R > non-flagged mean net R,
2. confirmation: mean net R > 0 with t >= 2.0, positive in at least 2 of the 3 years.

A passing rule is a **candidate for a strategy spec** (`docs/specs/<name>.yaml`, the user sets
`status: abgestimmt`), then tests, then `studies/pruefen.py`. It is never a strategy by itself,
and the holdout is opened only through that pipeline.

## Cells (counted up front)

Stage 1: 200 readings (25 settings x 4 TF x volume on/off), pooled, family-wise corrected;
<= 3 readings to confirmation; 7 asset-sessions reported.
Stage 2: 4 rules on 1 reading. Nothing else is tried; extra ideas get their own pre-registration.

## Not in this test

Path tokens inside the session toward a level (H11-B: the setup itself is null); 1-min and
2 h / 4 h token readings; anything outside the grid above; exchange (CME) volume; FX pairs (data audit not done for FX yet).
