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

## Robustness sprint results

**A1 — Entry timing (2019–2026).** Next-bar-open entry (with 1-tick slippage) is
*not* worse than the same-bar close:
US100 +1.025 R (close) → **+1.238 R** (next open), n 708→698;
SP500 +1.064 R → **+1.113 R**, n 476→469. The edge is not an artifact of
entering on the touch bar's close.

**A2 — Parameter neighbourhood (2019–2026, next-open).** Broad positive plateau,
no cliff. Longer opening range helps (US100 +0.96 R at 10 min → +1.70 R at
30 min; SP500 +1.07 → +1.37) and a stricter R/R filter helps (US100 +1.07 at
1.0 → +1.44 at 2.0; SP500 +0.88 → +1.19). Touch tolerance, front-run and the stop
bounds are flat (US100 1.06–1.30 R, SP500 0.88–1.16 R). The defaults are
**conservative, not overfit**.

**A3 — Holdout US100 2016–2018** (never used before): +0.35 R, PF 1.41,
WR 35–37 % (n≈250). Positive but clearly weaker than 2019–2026 — the edge existed
earlier too, at a lower magnitude.

**A4 — Cost stress (next-open, 2019–2026).**

| | default | 2× slippage | +0.3 pt commission | 3× slippage + commission |
|---|---|---|---|---|
| US100 | +1.24 R | +1.05 R | +1.10 R | **+0.70 R** (PF 1.73) |
| SP500 | +1.11 R | +0.91 R | +0.86 R | **+0.32 R** (PF 1.33) |

Break-even slippage is beyond 3× the default on both.

**B1 — Diversification (same rule, other markets, 2019–2026, price-scaled stops).**

| market | trades | expectancy | WR | PF | verdict |
|---|---|---|---|---|---|
| GER40 | 905 | **+1.20 R** | 48.0 % | 2.65 | works (non-US market) |
| US100 | 721 | +1.12 R | 43.7 % | 2.39 | works |
| SP500 | 711 | +0.86 R | 45.0 % | 2.00 | works |
| XAUUSD | 1003 | +0.14 R | 31.6 % | 1.13 | marginal |
| JP225 | 880 | **−0.14 R** | 27.7 % | 0.88 | fails |

Monthly-return correlations are low (US100↔SP500 0.03, US100↔GER40 0.01,
SP500↔JP225 0.53, others ≈0), so US100 + SP500 + GER40 is a genuine
diversification candidate — not the correlated pair it looked like from the
underlying markets. Caveat: the SP500 figures here use price-scaled stops and so
differ from the 2/10-point runs above; a portfolio needs one consistent config.

**Sprint verdict:** the candidate survives the cheap falsification tests
(A1–A4). The pre-2019 era is weaker, the edge is not universal (Nikkei fails),
and diversification across US100/SP500/GER40 looks promising.

## Portfolio (B2) + Monte Carlo (A6)

Same rule on US100 + SP500 + GER40, capital split with weights summing to 1
(0.5 % risk per trade of each allocation), 2019–2026, 1 994 trading days (1 206
with a trade). Run: `studies/portfolio_orb_fade.py`.

| variant | %/yr | vol | Sharpe | max DD | positive months | longest negative streak |
|---|---|---|---|---|---|---|
| US100 | +51.0 | 18.4 | 2.77 | −10.0 | 77.1 % | 3 |
| SP500 | +38.6 | 16.7 | 2.32 | −8.4 | 70.8 % | 4 |
| GER40 | +65.6 | 18.2 | 3.60 | −13.5 | 79.2 % | 4 |
| equal weight | +51.7 | 10.1 | **5.14** | −3.3 | 92.7 % | 1 |
| inverse vol | +51.3 | 10.0 | 5.12 | −3.0 | 92.7 % | 1 |
| inv-vol, 12 % vol target | +61.4 | 12.0 | 5.12 | −3.6 | 92.7 % | 1 |

Daily-R correlations are ~0 (US100↔SP500 0.01, US100↔GER40 −0.05,
SP500↔GER40 −0.09) — that is where the drawdown reduction comes from.
Prop on the vol-targeted book: FTMO 100 % pass (median 66 d), Lucid 99 %
(22 d), Instant 25K survive 98 % (mean payout $12 k), Instant 100K survive 100 %
($49 k). Block-bootstrap Monte Carlo (20-day blocks, 3 000 runs): annual
+61 % mean, p5 +41 %, P(negative year) 0 %, P(DD > 20 %) 0 %.

**Read this sceptically.** A Sharpe near 5 and P(negative year) = 0 are not
credible as a forecast: the Monte Carlo resamples one favourable 2019–2026
sample (no regime change, no edge decay), costs are backtest assumptions, the
12 % target implies leverage 1.2×, and the rule was selected within the project.
The **relative** finding is the solid one — near-zero correlation cuts the
drawdown roughly in three and lifts the Sharpe well above any single market —
not the absolute level. Expect live numbers far below these.

## Walk-forward / stability (A5)

Parameters are frozen, so this is out-of-sample by construction: the same rule
per calendar year and in rolling windows. `studies/walk_forward_orb_fade.py`.

Per market × year (expectancy R), 2019–2026: US100 0 negative years
(+0.50…+1.72 R), SP500 0 negative years (+0.36…+1.85 R), GER40 0 negative years
(+0.65…+1.68 R). **XAUUSD 2 negative years** (−0.36 in 2019, −0.04 in 2022),
**JP225 5 negative years** (−0.83, −0.14, −0.70, −0.11, −0.41).

Pre-committed selection rule — keep a market only if it is positive in
2019–2022 (in-sample), then judge 2023–2026 — selects exactly
**US100 + SP500 + GER40** and drops gold and Nikkei. That book: every year
positive (2019–2026 +49 %…+102 %), OOS 2023–2026 **+58.8 %/yr compounded**
(linear-sum +47.4 %/yr) at 9.5 % vol, Sharpe 4.97; 1 743 rolling 252-day windows
all positive (min +43 %, median +66 %), worst window drawdown −3.0 %.

Return conventions: earlier tables used the linear sum ×252/365; compounding
gives higher figures (e.g. +47 %/yr linear vs +59 %/yr compounded here). Same
caveat as the portfolio section: one favourable sample, backtest costs, selection
within the project — treat the levels as upper bounds.

## Additional instruments (in progress)

Dukascopy offers silver (`XAG/USD`), copper, Brent/WTI, the Dow, FTSE 100, the
10Y Treasury CFD and FX majors; mappings were added to
`trading_lab/minute_data.py`. Downloads for silver, Brent, 10Y, copper, FTSE 100
and Dow are running. Dow/FTSE overlap the existing equity indices (low expected
diversification); silver, energy and rates have different drivers and are the
interesting ones. They will be judged with the same pre-committed in-sample rule
above — not added because they happen to look good.

## Improvement backlog (partly done; needs approval)

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
