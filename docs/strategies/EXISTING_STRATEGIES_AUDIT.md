# Audit: do the repo's existing strategies have an edge?

Read-only review of what the repo already contains, re-checked with the current
gates (per-year split, cost reality, published criteria). Base rate before this
audit: 3 intraday ideas rejected in this project (fair pricing, NY-open EMA12,
ORB fade).

## What the repo contains

12 strategy modules (`orb_2r`, `mean_reversion`, `session_range_fade`,
`opening_range_breakout`, `bb_squeeze_breakout`, `supertrend_volume`,
`regime_hybrid`, `trend_range_rejection`, `silver_bullet`, `smc_ict`, `smc/*`),
4 evolution workers, a candidate leaderboard, and stored reports under
`artifacts/`.

## Documented verdicts found in the repo

| item | what the stored evidence says |
|---|---|
| `artifacts/orb_2r.json` | criteria min avg R 0.05, random percentile ≥ 95, ≤1 losing year → **`passed: []`** (US100, GER40, XAUUSD all failed) |
| `artifacts/mean_reversion_candidates.json` | go criteria (OOS pnl>0, Sharpe≥0.5, DD≥−10 %, ≥2 trades/wk) → **`summary.all_go: []`**; every candidate negative OOS and in walk-forward (A_gold 1/10 positive folds, Sharpe −1.99; B_gold 0/10; B_jpy 1/10) |
| `configs/evolved/gold_evolved_v12119.yaml` | promoted, but its own meta says **"asset drift bug — evolved on Gold data"**, *not validated* |
| `configs/evolved/jpy_evolved_v48831.yaml` | promoted, meta says **"independent validation pending"** |
| `artifacts/accidental_gold_best_oos.yaml` | filename records the asset-drift accident |

So the repo's own stored verdicts are: **nothing passed**, and the two promoted
configs are explicitly flagged as unvalidated/faulty.

## The one lead with positive gross numbers — and why it dies

`configs/baselines/hf_baseline.yaml` (`hf_ls01`, XAU/USD, Supertrend 7/1.5 +
volume, long+short) advertises "~7.2 % return/30 d" and the gold meta gives
"OOS +43.4 %, Sharpe 1.81, DD −11 %".

Re-run on our cached Dukascopy XAUUSD 1-minute data, 2019–2026, gross of costs:

| timeframe | trades | per week | gross pnl | Sharpe | max DD | WR | PF | per year |
|---|---|---|---|---|---|---|---|---|
| **1h** | 1 582 | 3.9 | **+59.4 %** | 0.78 | −9.3 % | 36 % | 1.14 | 7 of 8 positive (2022 −6.3 %) |
| 15m | 6 458 | 15.9 | +36.9 % | 0.48 | −12.1 % | 34 % | 1.04 | 7 of 8 |

It does **not** decay (2024 +8.6 %, 2025 +13.4 %, 2026 +22.0 %). But
`supertrend_volume.py` contains **no cost model at all** — no spread, slippage or
commission. Netted for the gold spread:

- median price 1 911, median 1-minute spread **0.374 USD** (0.0196 %);
- 1h ATR(14) = 5.09 → stop = 1.5·ATR = 7.6 = 0.40 % of price → the 1 %-risk
  sizing implies **2.5× notional** per trade;
- cost ≈ 0.0196 % × 2.5 = **0.049 % of account per round trip**; ×1 582 trades =
  **77.5 %** over 7.8 years.

**Net ≈ −18 % (−2.3 %/yr)**, and the sizing-independent statement is worse: the
average gross move per trade is 0.0376 % of the account ÷ 2.5 = **0.29 USD**,
against a **0.37 USD** spread — *the average trade captures less than one spread*.
No position sizing fixes that.

## Verdict

**No validated edge exists in the repo** — same conclusion as this project's own
three rejections. The recurring failure mode is identical every time: a signal
with positive gross expectancy whose per-trade move is the same order as (or
smaller than) the spread. Two structural lessons for anything built next:

1. **Measure the gross move per trade against the spread before anything else.**
   If `avg |move| / spread < ~2`, stop — it cannot work at that frequency.
2. **A missing cost model is a bug, not a simplification.** Any strategy module
   without spread/slippage inputs should be treated as unreported, not as
   profitable. (The same applies to a result that is "too good": here it was
   gross-only.)

Follow-up candidates are listed in `docs/RESEARCH_BACKLOG.md`; the highest-value
untested direction is market-neutral spreads (ES/NQ, gold/silver, Brent/WTI,
DAX/FTSE), where the tradable object *is* the spread and costs are smaller
relative to the moves.
