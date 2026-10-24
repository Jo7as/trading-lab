# Goal

**Target: 2–4 % per month.** Written down so every candidate is measured against
it, not against "is it positive".

## What that target actually requires

| Ziel | p. a. (compounded) | nötige **Netto-Sharpe** bei 15 % Vola | bei 12 % |
|---|---|---|---|
| 2 %/Monat | 26.8 % | **1.8** | 2.2 |
| 3 %/Monat | 42.6 % | 2.8 | 3.5 |
| 4 %/Monat | 60.1 % | **4.0** | 5.0 |

(Sharpe = CAGR / Vola. Expected drawdown at Sharpe ~2 and 15 % vol is roughly
1.5–2.5 × vol ≈ 22–38 %.)

**Consequence for selection:** a strategy that is merely "significant after
costs" is not enough. It must produce ≈27–60 % p.a. at its volatility, i.e. a
net Sharpe of ~1.8–4.0. Everything measured in this repo so far is ≈0 (best
finding: 4-hour Supertrend on EURUSD, +7.3 % net over 7.8 years ≈ 0.9 %/yr).

## Two different games (do not mix them up)

1. **Own capital, 2–4 %/month** — needs a genuinely strong edge (net Sharpe
   1.8–4.0). This is beyond almost everything in retail reach.
2. **Prop-firm EV** — the objective is not a % return on capital but
   `pass_rate × expected payout − fees`, with the firm's drawdown/consistency
   rules. A much weaker edge (net Sharpe ~0.8–1.2, small drawdowns, many
   attempts) can still be EV-positive. `trading_lab/prop_sim.py` computes this.

The user's target is (1); the prop route is a vehicle for it. If a candidate
passes the significance and cost gates but delivers, say, 5 % p.a., it is a
**fail against the goal**, not a partial success.

## Gates that follow from this

Add to every card, next to `kill_kriterien`:

- **Goal gate:** annualised net return at the intended volatility target must be
  in the 27–60 % band; otherwise the card is rejected *even if significant*.
- **Drawdown gate:** net max drawdown ≤ 20 % (and ≤ 1.5 × the in-sample value).
- **Capacity/cost gate:** the edge must survive 2× the assumed spread (cost
  break-even ≥ 2× default) — otherwise it is a rounding-error edge that cannot
  hold real size.
- **Breadth fallback:** if no single edge reaches the band, the honest path is
  several low-correlation edges whose *portfolio* Sharpe reaches ~2 — not one
  weaker edge levered up (leverage raises ruin risk, not Sharpe).
