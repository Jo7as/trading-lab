# ORB Rejection Fade — **REJECTED** (the edge was an artifact)

Status: **rejected.** Every positive number this strategy ever produced came from a
bug in its own day-slicing code, not from a real edge.

## Root cause

`prepare()` built the per-day slices from the **RTH-filtered** position index
(`np.flatnonzero(rth)` + `np.unique(day_arr, return_index=True)`) but then applied
those slices to the **full 24-hour frame** (`p.bid[k][sl]`). So instead of each
day's 09:30–16:00 session, the engine walked a rolling 390-bar window
(`slice(0,390)`, `slice(390,780)`, …) that drifted out of phase with the clock —
e.g. "day 1" became 18:00–00:29 ET, "day 2" 00:30–06:59 ET. The opening range,
the touches, the entry window and the exit all operated on misaligned bars.

Synthetic unit tests never caught it because they used RTH-only frames, where
`idx_rth == all rows` and the slices happen to be correct. The real 1-minute
caches are ~24 h, so every real backtest was affected.

Fix: map RTH positions back through `idx_rth` (`lo = idx_rth[s]`,
`hi = idx_rth[s + c - 1] + 1`) and slice the full frame; regression test
`test_full_day_frame_still_slices_the_session` builds a 24 h frame and asserts the
slice is 09:30–15:59 and the entry lands inside 09:45–11:30.

## Corrected result (fixed engine, 2019–2026, price-scaled stops, next-open entry)

| market | trades | expectancy | win rate | profit factor | negative years |
|---|---|---|---|---|---|
| US100 | 2 605 | **−0.249 R** | 12 % | 0.81 | 8 of 8 |
| SP500 | 2 811 | **−0.562 R** | 15 % | 0.58 | 8 of 8 |
| GER40 | 2 811 | **−0.390 R** | 14 % | 0.69 | 8 of 8 |

The naive fade — enter the first touch of the opening-range edge, stop just
beyond the touch, target the opposite edge — loses on every market and in every
year once the sessions are correct. The 12–15 % win rate shows most trades stop
out; the far target is rarely reached in the real session.

## What was void

Everything documented for this strategy before the fix is invalid: the ablation
(naive/touches/engulf/full), the random-direction and breakout baselines, the
2016–2018 holdout, the cost stress, the multi-market comparison, the
diversification portfolio, the prop-firm pass rates and payouts, and the
pass-speed surface. They measured a misaligned-window artifact.

## What is still valid

- The strategy-agnostic tooling: `trading_lab/prop_sim.py` (FTMO 1-/2-step,
  Lucid, Instant; eval pass rates and funded payouts), `significance`/`acceptance`,
  `studies/strategy_report.py`, `studies/strategy_dashboard.py`.
- The other strategies' negative results (fair pricing, NY-open EMA12).
- The process lessons: synthetic RTH-only tests are not enough (test the real
  frame shape); a "too good" result is a bug signal; per-day slicing must be
  validated against the actual clock.

## Completeness check across 11 markets (fixed engine)

`studies/multimarket_check.py`, 2019–2026, stops and costs scaled to each
instrument's median 1-minute range, next-open entry:

| market | trades | expectancy | WR | PF | negative years |
|---|---|---|---|---|---|
| US100 | 2 515 | −0.285 R | 19.1 % | 0.72 | 7 |
| SP500 | 2 531 | −0.386 R | 19.7 % | 0.62 | 8 |
| US30 | 2 619 | −0.355 R | 17.7 % | 0.66 | 8 |
| GER40 | 2 735 | −0.323 R | 19.7 % | 0.68 | 8 |
| UK100 | 2 800 | −0.460 R | 15.8 % | 0.57 | 8 |
| XAUUSD | 2 751 | −0.539 R | 17.8 % | 0.49 | 8 |
| XAGUSD | 2 496 | −0.988 R | 9.4 % | 0.22 | 8 |
| COPPER | 2 313 | −1.180 R | 8.2 % | 0.15 | 8 |
| BRENT | 2 755 | −0.755 R | 13.7 % | 0.34 | 8 |
| UST10Y | 2 765 | −0.848 R | 13.3 % | 0.29 | 8 |
| JP225 | 2 605 | −0.469 R | 17.0 % | 0.56 | 8 |

**Every market is negative, none is close.** On US100 the fade's expectancy
(−0.249 R) equals the random-direction baseline (−0.248 R): no information. There
are therefore no uncorrelated markets "where the strategy works" — the rule
itself does not work.

## Next

The rule as specified does not work on the three major indices with correct
sessions, so searching more instruments would be a search for a pattern that is
not there in this form. A follow-up would need a materially different hypothesis
(e.g. the fade only after a genuine failed breakout with acceptance, or a
different exit), not another market sweep.
