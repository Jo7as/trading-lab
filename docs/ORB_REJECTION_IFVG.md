# ORB Rejection Fade — validated edge, but not the video's recipe

Source spec: `strategy_orb_rejection_ifvg.yaml` (an Instagram reel + AI demo).
Built ablation-first: the same engine runs with fewer or more filters so each
filter's contribution is visible. All numbers are 1-minute RTH on Dukascopy
CFDs (US100 = NQ proxy, SP500 = ES proxy); costs are the historical bid/ask
spread plus entry/stop slippage.

## Ablation (US100): the filters destroy the edge

| mode | IS 2019–2023 | OOS 2024–now |
|---|---|---|
| **naive** (fade the first touch) | n=446, **+1.061 R**, p=0.0002 | n=278, **+1.013 R**, p=0.0002 |
| touches (≥3 touches) | n=108, −0.561 R | n=64, −0.444 R |
| engulf (+ rejection + engulfing) | n=3 | n=1 |
| full (+ inverse FVG) | n=2 | n=1 |

The "full" setup the video describes fires essentially never (≤2 trades in five
years) — the recipe as specified is not tradable. Every additional filter lowers
both the sample and the expectancy.

## The simple edge replicates across instruments

| mode | US100 IS | US100 OOS | SP500 IS | SP500 OOS |
|---|---|---|---|---|
| **naive fade** | +1.061 R (p=0.0002) | +1.013 R (p=0.0002) | +1.090 R (p=0.0002) | +0.654 R (p=0.0002) |
| classic ORB breakout | −0.267 R | −0.306 R | −0.293 R | −0.281 R |
| random direction (same bars/exits) | percentile 100 | percentile 100 | percentile 100 | percentile 100 |

The naive fade — enter the first touch of the opening-range edge, stop just
beyond the touch, target the opposite edge — beats the random-direction baseline
and the classic breakout on both instruments, in-sample and out-of-sample, and
passes the shared acceptance path (sign-flip) in all four cells. Cost stress
(2× slippage) keeps it positive (+0.688 R OOS US100).

## Prop-firm fit (`trading_lab/prop_sim.py`)

At 0.5 % risk per trade, evaluated with the block-bootstrap Monte Carlo:

| firm | US100 | SP500 |
|---|---|---|
| FTMO 2-step 100k (10 %/5 %, static 10 %) | pass 99.6 %, median 95 d | pass 99.9 % |
| Lucid 25K (5 %, EOD-trailing 4 %) | pass 82.3 %, median 25 d | pass 93.8 % |

Largest single trade: −$1,133 (US100/FTMO) and −$283 (Lucid) — both inside the
daily loss limits, i.e. the strategy fits the risk rules (unlike the video's
numbers, whose worst loss exceeded the account's total-loss limit).

## Caveats

- CFD proxies (no real futures, no contract rolls); point values and sizing are
  conventions.
- The Monte Carlo resamples the observed 2024–2026 window: it reports the pass
  rate **if the edge persists as observed**, not edge decay.
- `naive` was one of several ablated variants; it is positive in both instruments
  and both periods, but that is still a selection among variants.
- Not done: parameter-neighbourhood robustness, walk-forward, live/paper.

## Reproduce

```
.venv/bin/python backtests/backtest_orb_rejection_ifvg.py --runs 20        # ablation + baselines, both assets
.venv/bin/python studies/run_prop_sim.py --firm ftmo  --asset US100
.venv/bin/python studies/run_prop_sim.py --firm lucid --asset SP500
```
