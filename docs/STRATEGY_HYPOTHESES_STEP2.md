# Steps 1–3 (Oct 2026): cleaned re-runs, three hypotheses, the decision

## Step 1 — every headline re-run on CLEANED data

`studies/recheck_cleaned.py` (cleaning = corrupt bars >50× median **and** the
daily roll window, rows dropped).

**S0 cost gate changes materially.** On dirty data 15m looked hopeless; on clean
data the indices pass at 15m already, because the corrupt bars had inflated the
spread estimate:

| instrument | 15m | 60m | 240m |
|---|---|---|---|
| US100 | 3.50 PASS | 7.58 PASS | 17.4 PASS |
| US30 | 4.23 PASS | 9.05 PASS | 20.3 PASS |
| SP500 | 3.69 PASS | 7.93 PASS | 17.7 PASS |
| GER40 | 3.43 PASS | 7.23 PASS | 15.8 PASS |
| UK100 | 2.05 MARGINAL | 4.26 PASS | 9.2 PASS |
| XAUUSD | 2.58 MARGINAL | 5.31 PASS | 11.6 PASS |
| BRENT | 1.75 FAIL | 3.80 PASS | 8.3 PASS |
| UST10Y | 1.42 FAIL | 2.82 MARGINAL | 6.1 PASS |
| JP225 | 2.67 MARGINAL | 5.73 PASS | 13.5 PASS |
| XAGUSD / COPPER | 0.75 / 0.70 FAIL | 1.53 / 1.48 FAIL | 3.3 PASS |

**Rejections stand on clean data.** ORB-fade (naive, next-open, cleaned):
US100 −0.307 R (WR 19 %, PF 0.69), SP500 −0.359 R, GER40 −0.401 R. Pair spreads
(15m, cleaned): US100/SP500 real −125 % vs mid −12.5 % (PF 0.96) → gross ≈ 0,
net negative; XAU/XAG identical in shape.

**The one positive lead shrinks.** Gold Supertrend `hf_ls01`, cleaned 1h:
1 289 trades, gross **+47.1 %** (was +59.3 % on dirty data), spread cost 25.3 %,
net **+21.9 %** at zero slippage and **+9.2 %** at 0.5× slippage over 7.8 years
(≈1.2 %/yr) — too thin for a prop evaluation.

## Step 2 — three pre-registered hypotheses (≥60 minute horizons)

`studies/hypotheses_60m.py` computes everything on the identical minutes clock
(no resampling), on cleaned data. Gates pre-registered before running: sign-test
z ≥ 3, |median move| ≥ 2× round-trip cost, placebo clearly smaller, monotone
dose-response.

| hypothesis | US100 | SP500 | GER40 |
|---|---|---|---|
| H1 first-hour momentum (09:30→10:00 sign, move to 15:00) | med +0.040 %, z 2.14, eff/cost 2.4 | +0.022 %, z 1.71, 1.8 | +0.007 %, z 0.49, 0.5 |
| H2 last-hour momentum (open→15:00 sign, 15:00→15:55) | +0.009 %, z 1.14, 0.5 | +0.002 %, z 0.52, 0.2 | −0.001 %, z −0.36, 0.0 |
| H3 overnight gap fade (open vs prior close → close) | −0.012 %, z −0.46 | −0.007 %, z −0.52 | +0.005 %, z 0.31 |

**All nine cells FAIL.** H1 US100 is the closest thing to a signal in this project
since the bug (+0.04 % median per day, effect 2.4× cost) but it fails on
significance (z 2.14) *and* has no dose-response, and the placebo is not clearly
smaller. H2 and H3 are indistinguishable from zero.

Method note: the `research.py` event-study engine was **wrong** (it reported a
median of +2.38 %/day for H1 where the direct clock computation gives +0.08 %);
it is deprecated in favour of the direct-clock study, which is what every number
above uses.

## Step 3 — the decision (pre-committed)

The rule was: if the ≥60m hypotheses produce nothing, **do not keep searching
patterns on this data — change the foundation.** That is where we are. Three
concrete changes, in order of expected value:

1. **Asset class with a cheaper, cleaner data foundation: crypto.** Free *true*
   level-1 history (bookTicker + aggTrades), 24/7 sessions (no roll corruption),
   relative spreads at 1h+ are small. Our entire failure mode — cost, corrupt
   roll bars, session boundaries — largely disappears.
2. **Real futures data/execution** (Databento MBP-1, ES/NQ): ~1 tick spread,
   ~$1 commission, i.e. 5–10× lower relative cost than the CFDs.
3. **Longer horizons** (daily/multi-day), where moves are many spreads — subject
   to the prop firms' overnight rules.

What we will *not* do: another intraday pattern scan on the same CFD caches.

## What stays valid

The measurement apparatus (cost gate, data-quality gate, calendar invariants,
hypothesis-card gate, prop simulator, acceptance, reporting) and the documented
negative results. The apparatus is what made these three hypotheses cheap to kill
honestly — that is its job.
