# Goal

**Target: 1–2 % per month** (+12.7 … +26.8 % p.a. compounded). Written down so
every candidate is measured against it, not against "is it positive".

Decided 2026-10-26; the previous 2–4 %/month band is retired (see *Why 2–4 % was
dropped* below). The numbers live in exactly one place — `trading_lab/goal.py` —
and every consumer (evolution workers, `score.py`, the study gates, the
`deploy/state-templates/*/goal.yaml`) reads them from there. The target had drifted
into ten places with five different values before that.

**The band is not evidence.** It says what counts as *enough*; it never replaces
the significance, cost or data gates (`docs/ACCEPTANCE.md`). A candidate that
clears the band by luck is still rejected.

## What the target actually requires

Net Sharpe = annual return / annual volatility:

| Ziel | p. a. (compounded) | bei 10 % Vola | bei 15 % Vola | bei 20 % Vola |
|---|---|---|---|---|
| 1 %/Monat | 12.7 % | 1.27 | **0.85** | **0.63** |
| 1.5 %/Monat | 19.6 % | 1.96 | 1.30 | 0.98 |
| 2 %/Monat | 26.8 % | 2.68 | 1.79 | **1.34** |

Sharpe 0.5–1.0 is a good, tradeable strategy; 1.3–1.8 at 15–20 % vol is
demanding but not off the scale. Expected drawdown is ~1.5–2.5 × vol, i.e.
**22–38 % at 15–20 % vol** — so the *drawdown*, not the return, is what constrains
the band:

| Vola | erwarteter MDD | DD-Gate ≤ 20 % |
|---|---|---|
| 10 % | ~18 % | hält |
| 15 % | ~27 % | **reisst** |
| 20 % | ~36 % | **reisst** |

The band's **lower** edge (1 %/month) is reachable at ~10–12 % vol with Sharpe
~1.0–1.3. Its **upper** edge (2 %/month) needs Sharpe ~1.8–2.5 *and* a drawdown
above the gate — treat 2 %/month as the aspiration and 1 %/month as the bar.

## Two different games (do not mix them up)

1. **Own capital, 1–2 %/month** — needs a net Sharpe of ~0.8–1.8. This is the
   target. It is the honest ceiling the repo's own measurements point at.
2. **Prop-firm EV** — the objective is not a % return on capital but
   `pass_rate × expected payout − fees`, with the firm's drawdown/consistency
   rules. A 10 % total drawdown limit supports roughly 5 % vol, and at 5 % vol a
   Sharpe of 1.0 makes ~5 %/yr = **0.4 %/month**. The band is therefore *not*
   reachable under prop rules — a prop account is a vehicle for volume and
   capital, not for the band. `trading_lab/prop_sim.py` computes this.

A candidate that passes the significance and cost gates but delivers, say,
5 % p.a., is a **fail against the goal**, not a partial success.

## Gates that follow from this

Add to every card, next to `kill_kriterien`:

- **Goal gate:** annualised net return inside the band (+12.7 … +26.8 %/yr);
  the gate sits at the **lower** edge, +12.7 %/yr — otherwise the card is
  rejected *even if significant*. (`studies/run_h004.py` applies it.)
- **Drawdown gate:** net max drawdown ≤ 20 % (and ≤ 1.5 × the in-sample value).
  With 15–20 % vol this, not the return, is the binding constraint.
- **Capacity/cost gate:** the edge must survive 2× the assumed spread (cost
  break-even ≥ 2× default) — otherwise it is a rounding-error edge that cannot
  hold real size.
- **Breadth fallback:** if no single edge reaches the band, the honest path is
  several low-correlation edges whose *portfolio* Sharpe reaches ~1.3 — not one
  weaker edge levered up (leverage raises ruin risk, not Sharpe).

## Why 2–4 % was dropped

The retired band needed +27 … +60 %/yr, i.e. a net Sharpe of **2.7–6.0** at
10–20 % vol. Sharpe 2 is world class and does not persist outside a handful of
closed funds, so the old band was not "ambitious" but off the scale — it turned
every measurement in the repo into a failure by construction, including the one
finding that looked real. The measurements did not change with the band; only the
threshold did, and the new band is the one this repo's own analysis arrived at
(see the status note below).

## Status (2026-10-25/26): no direction has cleared the band yet

Eight directions have been tested and all eight are dead. The last candidate
(FX carry, H-009) looked like it cleared the *old* band at +14.4 %/yr / Sharpe
1.71 — until a second, independent representation of the same rule (a *daily*
path, built for the prop simulator) exposed a look-ahead in its VIX filter.
Correctly evaluated it is +2.4 %/yr at Sharpe 0.27, and its own random-ranking
placebo beats it. See `FINDINGS.md`.

The best measured result in the repo is ~0.9 %/yr gross (4-hour Supertrend on
EURUSD) — i.e. *below* the new band as well, so the band does not turn that into
a pass. What the band changes is the size of the remaining gap: it is now a
factor ~2–5, not ~10–30.

Two things decide whether the band is reachable, and they are measurement
problems, not threshold problems:

1. **Costs and horizon.** A directional signal must capture `q_min = 1/ratio` of
   the average move; below ~4 hours almost no CFD passes that hurdle
   (`studies/cost_move_screen.py`, `docs/AEP_STAGE0_K3.md`). Futures have 5–6×
   smaller spreads.
2. **Power.** With 400–1500 trades the acceptance path detects a true Sharpe 1.0
   edge only 36–63 % of the time (`studies/pipeline_power.py`). More history and
   instruments, not a looser gate, is the fix.

**What would actually move the needle** (in order): a genuinely higher Sharpe
(nothing here has produced one), a lower cost base (CME futures), or more
independent instruments to buy power with.
