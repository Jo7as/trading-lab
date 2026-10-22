# Research backlog — what to test next, and why

Working notes. Evidence-based priors, not promises. Base rate so far: **0 of 3**
intraday ideas survived (fair pricing, NY-open EMA12, ORB fade) — all three died
on costs or on a bug. That shapes the priors below.

## A. What our own work already suggests (usable signals)

1. **Costs, not signals, are the recurring killer.** Every rejection was
   "signal exists, but smaller than spread + slippage". Structural consequence:
   prefer *fewer, larger* trades, markets with tight relative spreads, and
   predefined exits — over 1-minute scalping.
2. **Volatility-driven continuation** is the best-documented intraday effect we
   touched (fair-pricing Setup D was the only piece that looked positive in
   small samples; the noise-area/SPY literature is the futures version).
   Candidate: *shock continuation* with ATR-relative thresholds.
3. **Regime conditioning is cheap and literature-backed** (the SPY mechanism's
   Sharpe rises with VIX). We can proxy VIX with Dukascopy `VOL.IDX/USD`.
4. **Long/short asymmetry** showed up repeatedly (long side carries the
   profit); test sides separately, never pick after the fact.
5. **The repo's own strategies were never audited here** — `orb_2r`,
   `mean_reversion`/`session_range_fade`, the SMC family have dossiers
   (`docs/STRATEGY_RESEARCH.md`, `docs/SMC_*`) and evolution outputs. That is the
   cheapest place to look for something already positive.

## B. Not yet tested, worth it and not off-the-shelf

Ordered by expected value / effort:

1. **Audit the existing strategies with the new rigour** (24 h frames, per-year,
   random-direction baseline, cost stress, then the prop sim). Zero new code.
2. **Market-neutral spreads** — ES/NQ, gold/silver, Brent/WTI, DAX/FTSE (all
   cached). Mean-reverting cointegration spreads are what non-institutional
   professional desks actually trade; low beta and small drawdowns suit prop
   rules. Highest-value untested direction.
3. **Overnight vs intraday decomposition** — the overnight drift is a robust,
   documented anomaly and our 1 m caches have 24 h sessions. Constraint: many
   prop firms forbid overnight/weekend holds; check per firm, or use it as a
   filter rather than a position.
4. **Cross-sectional relative momentum** in a small basket (rank the cached
   markets, long top / short bottom, daily or multi-hour rebalance) — documented,
   cost-manageable because it is not 1-minute.
5. **Compression/expansion** (NR7 / inside-day → next-day range expansion) —
   volatility clustering, cheap to code on 1m.
6. **Calendar/event conditioning** of any existing signal (FOMC/CPI) — we have
   the calendar loader; often shows where an edge actually lives.
7. **Cross-asset lead-lag** at the European open (US index futures vs DAX/FTSE)
   and gold vs real yields (UST10Y).

## C. How professional (non-institutional) algos are actually built

Not secret patterns — mostly classic effects plus discipline:

- **Mean reversion with strict cost control** (bread and butter; often
  spread/pairs reversion).
- **Momentum/breakout gated by a volatility regime** (trend days only).
- **Statistical arbitrage / pairs / index-vs-basket** — market-neutral, the
  professional staple.
- **Overnight, gap and calendar anomalies** — small, robust, capacity-limited.
- **Order-flow/microstructure** — institutional/HFT territory; retail cannot
  compete (see the M2 skepticism below).
- **The portfolio/execution layer carries most of the value**: risk budgeting,
  correlation-aware sizing, cost-aware execution, staged scaling. Pros usually
  run *many* low-correlation signals (each Sharpe ~0.2–0.4), not one big alpha.

Realistic path to prop EV: **breadth of small edges + risk control**, not one
great signal. The multi-strategy plan
(`docs/MULTI_STRATEGY_DUKASCOPY_PLAN.md`) is built on that idea.

## D. Module priors for the multi-strategy plan (and kill criteria)

| module | net-edge chance | kill criterion |
|---|---|---|
| M1 noise-area momentum | 30–40 % | not net-positive after 2024 on US100+SP500+GER40 |
| M2 OFI at levels | 15–25 % (signal 40–60 %) | \|IC\|≈0 or no absorption/breakout separation in the pre-test |
| M3 mean reversion in the noise area | 25–35 % | not net-positive on two independent markets |
| M4 FX strength | 15–20 % | — (lowest priority) |

Portfolio reaching a prop-viable level (Sharpe ~1.5 net): **10–15 %**.
All four working: **< 5 %**. A module counts as surviving only if it is
net-positive, per-year stable (or beats the random baseline), plateau-shaped in
its neighbourhood, and positive on **two** independent markets.

Order of work: **M2 pre-test** (cheap, decides feasibility) → **M1** → M3 →
M4 → regime/portfolio only with ≥2 survivors. Time-box: if M1 and the M2
pre-test produce nothing, stop the intraday-alpha hunt and reconsider the goal.
