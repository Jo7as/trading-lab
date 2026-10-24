# Design principles — where an edge can actually live (from our own measurements)

Written after 8 tested-and-rejected directions (fair pricing, NY-open EMA12, ORB
fade, pair spreads, gold Supertrend, FX Supertrend, Asian-range breakout) and the
audit of the pre-existing repo strategies. Read with `docs/GOAL.md` (2–4 %/month).

## 1. Is "everything fails" normal? Yes — and our sample makes it likely

- **Base rate.** Most plausible ideas do not survive realistic costs
  out-of-sample; publication decay alone (McLean/Pontiff) removes ~half. Seeing
  0 of 5 on the first pass is unremarkable.
- **We tested publicly-circulated ideas.** Reels, interviews, viral strategies
  are exactly the ones most likely to be arbitraged. That is a biased sample
  *against* us.
- **Our data class is the expensive one.** CFD spreads are 3–7× wider than
  futures, with 24 h sessions, a dirty daily roll, tick-volume only (no tape),
  and no term structure. Microstructure edges are simply not visible here.
- **Our instrumentation is deliberately harsh** (stop-first intrabar, taker on
  both legs, audit-correct frames, per-year, placebo, dose-response). Most retail
  backtests are lenient in exactly these places — that is why they "work" and
  live trading does not. A harsh bar produces fewer hits, but the hits are real.

**Conclusion:** the failures are information, not a bug. What *is* a bug signal
is a result that looks too good (we caught one: a Sharpe-5 "edge" that was a
day-slicing artifact).

## 2. Where edges are demonstrably *not*

- Anything **below ~15 minutes** on CFD data: the average move barely exceeds the
  spread (`studies/cost_gate.py`; S0 ratios of 0.7–2).
- **Single-instrument intraday patterns** (fade the edge, break the range, flip
  the trend): 15 m/30 m/1 h versions are negative and often *worse than a
  random direction* — paying the spread for noise.
- **Levering a weak edge** to hit the return target: raises ruin risk, not Sharpe.

## 3. What a strategy should look like here

| Dimension | Choice | Why (evidence) |
|---|---|---|
| Horizon | **≥ 1 h, ideally 1 h–days** | S0 passes from 15 m (FX) / 60 m (indices) upward; 4 h has the best ratios (13–28) |
| Frequency | **1–10 trades/week** | cost per trade is fixed; more trades = more spread paid for the same edge |
| Instrument | tight relative spread: **FX majors**, index futures | EURUSD 0.28 bp vs AUDUSD 1.3 bp vs silver 12 bp |
| Session | **London/NY overlap (08:00–11:00 NY)** | S0 up to 38 there (tightest spreads, largest moves) |
| Edge class | **structural beats predictive**: carry/basis/funding, calendar/forced flow (fix, rebalancing, CB days), cross-sectional breadth, slow session seasonality | not a prediction game; moves span many spreads |
| Portfolio | **breadth, not leverage**: N low-correlation edges | Sharpe ≈ s·√N — to reach 2.0 from s≈0.7 needs ~8 edges |

## 4. The honest route to 2–4 %/month

A single retail edge with net Sharpe 2–4 is not realistic. A **basket of ~6–10
low-correlation edges at net Sharpe 0.5–1.0** reaches portfolio Sharpe ~2, which
is what the target requires. That changes the research question from "find the
strategy" to "collect small, cost-surviving, uncorrelated edges and combine
them" — carry/basis, calendar flow, cross-sectional FX, slow trend, and
session-conditional mean reversion are the candidate families (in that order).

## 5. What we do not do

No sub-15 m CFDs. No single-instrument intraday pattern hunting. No
parameter-tuning after seeing results. No gross-only numbers. No leverage to
reach a return target. No conclusions from one favourable sample.
