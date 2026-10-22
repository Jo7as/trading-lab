# NY-Open EMA12 momentum — rejected

Tested as a hypothesis, not built out. The source was a social-media reel with
three curated trades and no statistics; nearly every rule was an assumption.

## Setup

- Instrument: US100 (Dukascopy `E_NQ-100`) 1m bid+ask → 5m signals, 1m execution.
- Rule: first 5m candle after the NY open vs a 5m EMA12 — close above → long,
  below → short; one trade per day at 09:35; first-candle-extreme stop; ATR
  chandelier trailing stop armed at +1R; no take-profit; flat at 15:55.
- Costs: historical bid/ask spread + 2-tick slippage. Fixed-fractional risk with
  a leverage cap and a minimum stop distance.

## Result

| Period | Trades | Win rate | Expectancy | Profit factor | Total R | Buy & hold |
|---|---|---|---|---|---|---|
| IS 2016–2021 | 1,497 | 34.9 % | −0.302 R | 0.53 | −644 | +260 % |
| OOS 2022–now | 1,194 | 36.1 % | −0.145 R | 0.77 | −240 | +79 % |

Acceptance path FAIL in both periods.

- The signal **does** beat a random-direction baseline with the same exit model
  (strategy percentile 96 % OOS, 100 % IS) — so the EMA carries weak directional
  information relative to a coin flip. Costs and the exit model consume it.
- Sensitivity grid EMA 9–20 × trail 1.5–3.0: **all 20 combinations negative**
  (best −0.073 R). No profitable region, hence no plateau to optimise.
- Overnight-hold variants (the source's claimed source of profit, with trailing)
  are also negative: −0.083 R to −0.149 R.
- Buy & hold dominates by a wide margin.

## Why no evolution worker

The failure is structural (cost/exit), not parametric: there is no maximum to
evolve toward, and a broad search over a uniformly negative neighbourhood would
only fit noise (multiple testing). A targeted, hypothesis-driven search (e.g.
VIX-regime split, RTH direction filter, or the same signal on SP500/Gold) would
be the only defensible follow-up — not a parameter sweep.

## Implementation

The strategy module, runner and tests were removed after the verdict. This note
is the record; re-implement only with a materially different hypothesis and the
frozen-protocol discipline (no re-optimisation), then run the acceptance path.
