# ORB Rejection Fade — **candidate** (validated, not yet live)

Status: **GO candidate** for prop trading at 0.5 % risk per trade. One rule,
tested on two correlated US-index instruments. Not a portfolio, not a combo.

| | |
|---|---|
| Source | `strategy_orb_rejection_ifvg.yaml` (Instagram reel + AI demo) |
| Code | `trading_lab/strategies/orb_rejection_ifvg.py` (`mode="naive"`) |
| Tests | `tests/test_orb_rejection_ifvg.py` (7) |
| Backtest | `backtests/backtest_orb_rejection_ifvg.py` (ablation + baselines) |
| Report | `studies/strategy_report.py --strategy orb_fade --firm lucid` |
| Dashboard | `studies/strategy_dashboard.py` → `artifacts/dashboards/orb_fade_*.{html,md,csv}` |

## The rule (as tested)

1. Opening range = first 15 min (09:30–09:45 ET); `orh`/`orl` from the 1m bars.
2. In the window **09:45–11:30**, count "touches" of an edge: `high ≥ orh − 2 ticks`
   and `close ≤ orh + 0.05·range` (mirrored for the low edge), ≥3 bars apart.
3. **First** such touch with the close still **inside** the range → enter at that
   bar's close (short at the high edge / long at the low edge).
4. Stop: just beyond the touch extreme (+2 ticks), clipped to 2–10 points.
5. Target: the **opposite** edge, front-run by 2 ticks. Require reward/risk ≥ 1.5.
6. Flat by 15:55; max 2 trades/day; stop after 2 consecutive losses in a day.
7. A momentum breakout (3 closes beyond an edge by 0.1·range) locks that side.

Discarded filters (they make it worse): ≥N touches, rejection candles,
engulfing trigger, inverse FVG. The "full" video setup fires ≤2× in 5 years.

## Results (Dukascopy CFDs, 1-min RTH, spread + entry/stop slippage)

| metric | US100 IS 2019–23 | US100 OOS 2024–26 | SP500 IS | SP500 OOS |
|---|---|---|---|---|
| trades | 446 | 278 | 308 | 215 |
| trades/week | 1.71 | 1.92 | 1.18 | 1.50 |
| win rate | 42.8 % | 35.6 % | 54.9 % | 45.1 % |
| expectancy | +1.06 R | +1.01 R | +1.09 R | +0.65 R |
| profit factor | 2.56 | 2.36 | 3.00 | 1.99 |
| monthly mean | +4.02 % | +3.99 % | +2.79 % | +2.11 % |
| max drawdown | −5.2 % | −6.4 % | −4.8 % | −4.4 % |
| acceptance (sign-flip) | PASS p=0.0002 | PASS p=0.0002 | PASS p=0.0002 | PASS p=0.0002 |
| vs random-direction baseline | percentile 100 | 100 | 100 | 100 |
| classic ORB breakout | −0.27 R | −0.31 R | −0.29 R | −0.28 R |

All at 0.5 % risk/trade, fixed (not compounded).

## Consistency audit (read-only, 2019–2026)

**Every one of the 16 instrument-years is positive** (expectancy +0.52 … +1.73 R,
profit factor 1.68 … 4.46). First vs second half: US100 +1.09 R → +0.96 R,
SP500 +1.13 R → +1.00 R (no decay). No concentration: best month = 8 % (US100) /
10 % (SP500) of total profit, top-3 months 20 % / 18 %, best day 5 % / 2 %.
Long vs short: US100 longs +1.97 R (65 % WR) carry ~84 % of the profit, shorts
only +0.29 R (25 % WR); SP500 is balanced (+1.33 R vs +0.89 R).

## Prop-firm outcomes (Monte Carlo, 2019–2026)

| firm | US100 | SP500 |
|---|---|---|
| FTMO 2-step 100k | pass 99.9 %, median 98 d | 100 %, 134 d |
| 1-step 100k | 100 %, 48 d | 100 %, 69 d |
| Lucid 25K (EOD trailing 4 %) | 91.3 %, 28 d | 98.2 %, 44 d |
| Instant 25K (4 % static) | survives 73 %, mean payout $8.3k | 94 %, $6.3k |
| Instant 100K (8 % static) | survives 99 %, mean payout $37k | 100 %, $26k |

Largest single trade: −2.2 R (US100) / −2.5 R (SP500) → within daily limits.

## What to watch

1. **Two correlated instruments** (both US indices) — this is scale, not diversification.
2. **US100 shorts are marginal** (+0.29 R, 25 % WR); dropping them is a post-hoc
   choice that needs its own out-of-sample test.
3. **Entry at the touch bar's close** — the most optimistic assumption; must be
   checked against next-bar-open.
4. **Parameters untested** (touch tolerance, min R/R, front-run, stop bounds,
   OR length, breakout buffer) — no neighbourhood/plateau check yet.
5. **Selection risk**: `naive` was chosen after the filters failed (though it was
   the spec's pre-registered baseline); 16/16 positive years argue against luck.
6. **Model limits**: CFD proxies (no futures roll), SP500-2018 missing, prop daily
   loss modelled on realised (not intrabar) P&L, Monte Carlo resamples 2019–2026.

## Improvement backlog (not done; needs approval)

1. **Robustness** (no tuning): next-bar-open entry, parameter neighbourhood,
   cost stress on both assets, holdout US100 2016–2018.
2. **Diversification** (biggest lever): same rule on uncorrelated Dukascopy
   markets — Gold (XAU/USD), DAX (GER40), Nikkei (JP225) — no per-market tuning.
3. **Structural variants** (one at a time, frozen, OOS-tested): long-only on US100;
   ATR-based stop; "mid then edge" scaling; FOMC/CPI news filter.
4. **Portfolio layer**: inverse-vol allocation + a correlation-aware joint prop sim.

## Reproduce

```
.venv/bin/python backtests/backtest_orb_rejection_ifvg.py --runs 20
.venv/bin/python studies/strategy_report.py --asset US100 --start-year 2024 --firm lucid
.venv/bin/python studies/strategy_dashboard.py --risk-pct 0.5
```
