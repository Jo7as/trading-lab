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

## Repo-wide: which modules even have a cost model?

| modules | cost model |
|---|---|
| `orb_2r`, `session_range_fade`, `mean_reversion`, `pair_spread`, `orb_rejection_ifvg` | bid/ask and/or slippage/commission |
| `opening_range_breakout`, `supertrend_volume`, `regime_hybrid`, `bb_squeeze_breakout`, `trend_range_rejection`, `smc_ict` (+`smc/`), `silver_bullet` | **none — every number from these is gross** |

That is 7 of 12 modules. Any result attributed to them (README claims, artifacts
reports, leaderboard entries) is unverified until the spread is applied. The
README table now carries a "Costs modelled" column.

Spreads in the cached data are realistic (median ask−bid: US100 0.022 % /
0.0098 % at the 25th percentile, SP500 0.012 %, GER40 0.017 %, XAUUSD 0.020 %,
XAGUSD 0.123 %, JP225 0.024 %), and the strategies that do model costs price it
from each bar's actual bid/ask rather than a median — so their costs are correct
per fill.

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
commission. Netted for the gold spread, using the strategy's **actual sizing**:

- median price 1 911, median 1-minute spread **0.374 USD** (0.0195 %);
- `RiskManager.calculate_position_size` **caps the position at 1× account
  notional** (`min(size, 1.0)`), and 97 % of trades hit that cap → real risk per
  trade ≈ **0.59 %** (not the nominal 1 %), notional = 1×;
- round-trip cost = 0.0195 % × 1 = **0.0182 % of account**; ×1 582 trades =
  **28.8 %** over 7.8 years (≈3.7 %/yr);
- **net ≈ +30.5 %** (≈+3.5 %/yr). Average gross move per trade **0.72 USD** against
  a **0.374 USD** spread → ratio **1.94**.

> Correction: an earlier version of this note computed −18 % net by taking the
> *nominal* 1 % risk rule at face value and assuming a 2.5× notional. The sizing
> cap makes the real cost ~2.6× smaller. The honest verdict is the one above.

**So this baseline is *not* rejected** — it survives costs at ≈+3.5 %/yr with a
−9 % drawdown, but only just (move/spread 1.94, right at the ~2 threshold). It
deserves a proper cost-aware validation (per-year after costs, random baseline,
tick-level fills with explicit order types, then the prop sim) before any
conclusion. It is currently the only candidate in the repo that clears the
spread at all.

## Verdict

**No *validated* edge exists in the repo** — the stored gate results (`passed: []`,
`all_go: []`) stand, and the pair-spread direction was rejected on its gross
result. But the gold Supertrend baseline is **not** rejected: it survives the
spread at ≈+3.5 %/yr after its real (capped) sizing, at the margin (move/spread
1.94) — it needs cost-aware validation before any conclusion.

Two structural lessons for anything built next:

1. **Measure the gross move per trade against the spread before anything else**,
   and compute the cost from the strategy's **actual sizing and order types**
   (a `RiskManager` position is capped at 1× notional; a limit exit pays no
   spread, a stop/time exit does). If `avg |move| / spread < ~2`, stop — it
   cannot work at that frequency.
2. **A missing cost model is a bug, not a simplification.** Any strategy module
   without spread/slippage inputs should be treated as unreported, not as
   profitable. (The same applies to a result that is "too good": here it was
   gross-only.)

Follow-up candidates are listed in `docs/RESEARCH_BACKLOG.md`; the highest-value
untested direction is market-neutral spreads (ES/NQ, gold/silver, Brent/WTI,
DAX/FTSE), where the tradable object *is* the spread and costs are smaller
relative to the moves.
