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

## Next

The rule as specified does not work on the three major indices with correct
sessions, so searching more instruments would be a search for a pattern that is
not there in this form. A follow-up would need a materially different hypothesis
(e.g. the fade only after a genuine failed breakout with acceptance, or a
different exit), not another market sweep.
