# Pair spreads (statistical arbitrage) — REJECTED

Tested as the highest-value untested direction (market-neutral, prop-friendly).
Rejected: **the reversion has no edge before costs, and costs are the whole story.**

## Setup

Dollar-neutral pair: long 1 $ of A / short 1 $ of B (or the reverse) when
`log(A) − log(B)` deviates from its 60-bar rolling mean by ≥ 2σ; exit at |z| ≤ 0.5,
stop at |z| ≥ 3.5, or after 24 bars. 15-minute bars, one bar of execution delay.
Costs are explicit: the long leg buys the ask / sells the bid, the short leg the
mirror — a full spread per leg per round trip, from each leg's real bid/ask.
Code: `trading_lab/strategies/pair_spread.py`.

## Result (2019–2026, 15-minute bars)

| pair | net (real spread) | gross (mid prices) | random entry | gross PF | net PF |
|---|---|---|---|---|---|
| US100/SP500 | **−165.7 %** | −5.4 % | −422 % | 0.99 | 0.67 |
| US30/SP500 | −141.1 % | −10.3 % | −320 % | 0.97 | 0.63 |
| GER40/UK100 | −156.5 % | **+21.1 %** | −617 % | 1.04 | 0.74 |
| XAUUSD/XAGUSD | **−691.2 %** | **+52.4 %** | −1763 % | 1.06 | 0.46 |

The gross (cost-free) expectancy is ≈ 0: PF 0.97–1.06, i.e. the average trade
captures essentially nothing. With real spreads every pair is deeply negative
(PF 0.46–0.74). Random entry is far worse — the z-score does order the bars, but
the ordered part is worth less than zero after costs.

## Why my pre-test was wrong (correction to `studies/spread_pairs_gate.py`)

The gate compared `2·σ_deviation / cost` and reported 8–54 — apparently a wide
margin. That is the **wrong quantity**: σ of the deviation is not the tradable
move. The tradable move is the *realized* reversion from entry to exit, which
here is ≈ 0 (PF ≈ 1.0 gross). Correct gate: measure the **conditional expected
P&L from entry to exit** (gross), and require it to exceed the round-trip cost of
both legs by a wide factor — not the volatility of the deviation. Same lesson as
the single-instrument tests, one level deeper.

## Verdict

Fifth rejected direction in this project (fair pricing, NY-open EMA12, ORB fade,
gold Supertrend baseline, pair spreads). The recurring, structural finding is
that on CFD data with realistic spreads, any signal whose per-trade move is of
the same order as the spread does not survive — and the honest responses are
fewer/larger trades, cheaper (real futures) execution, or accepting the evidence.

Not tested here: pairs on *real futures* (CME) where the spread is ~1 tick and
commissions are ~$1, or on a *slower* horizon where moves are many spreads — both
require different data/accounts, not more backtesting on CFDs.
