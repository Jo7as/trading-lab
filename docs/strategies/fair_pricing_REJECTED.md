# Fair Pricing (JJ Simon) — rejected

The mechanical transcription of the fair-pricing interview spec was implemented
and tested on US100 1m Dukascopy data (`E_NQ-100`, the Nasdaq-100 CFD proxy for
NQ/MNQ), 2016–2026, through the shared backtest and acceptance pipeline.

## Result

| Period | Trades | Win rate | Expectancy | Acceptance |
|---|---|---|---|---|
| dev 2016–2021 | 3,320 | 30.8 % | −0.212 R | FAIL |
| test 2022–now | 5,339 | 33.7 % | −0.152 R | FAIL |

Setups A (opening continuation) and B (reversion to the fair price) are negative
in both periods and dominate the sample. Setup C (pre-news reversion) has only a
handful of trades because the test calendar covered 2024 only. The sign-flip
significance test rejects the edge in both periods.

The loss is not a cost artifact: it persists with the historical bid/ask spread
and slippage applied, and the fixed-risk sizing is not leverage-capped.

## Scope of the verdict

Implemented: setups A, B, C, D1, D2, E; sessions NY AM, NY PM, Asia, London; the
fair-price state machine (§4.1–4.4, 4.6 incl. FP2); an offline economic calendar;
swing/BOS, displacement, consolidation and shock primitives.

Not implemented, and therefore **not** rejected by this result:

- the prop-firm account model, routing and offline optimizer (§6.3–6.5, §7, §9);
- the fully discretionary entries and wick-rejection variants (§11);
- §4.5 (fair-price structural adjust).

So this rejects the **mechanical transcription** on an index CFD, not necessarily
the whole discretionary concept. Do not re-add it to `main` without a materially
different hypothesis (e.g. a real news feed with full history, futures data, or
an execution model that captures the discretionary timing).

## Where the code is

Code, tests, calendar loader, example calendar and the source spec are preserved
on the `archive/fair-pricing` branch:

```
git show archive/fair-pricing:trading_lab/strategies/fair_pricing.py
git checkout archive/fair-pricing
```
