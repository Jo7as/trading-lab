# SMC Flow Evolution — Stages A–C (Spec)

Reference playbook (unchanged baseline): **4h bias → liquidity sweep → 15m align/ChoCh → 1m nested ChoCh → 1m FVG limit (`fvg_edge`)**.

This document defines **optional flow gates** (A–C) for ablation backtests and evolution. All flags default to **off** so the live baseline matches the Jul 2026 reference tune.

Implementation: `trading_lab/strategies/smc/flow_filters.py` (pure functions) + `SmcIctConfig` flags. Wiring into `run_backtest` / `StateMachine` is **incremental** — one stage at a time.

---

## State machine (target)

### Baseline (today)

```text
IDLE → HTF_ALIGNED → SWEPT → WAIT_FOR_CHOCH → ENTRY_PENDING → IN_TRADE
                              ↑ on_15m_align (bias OR 15m ChoCh)
```

### With Stages A–C enabled

```text
IDLE
  → HTF_ALIGNED
  → SWEEP_TOUCHED     (optional: wick through level, not yet confirmed)
  → SWEPT             (Stage A: close back inside / reject)
  → [Stage B: price in correct premium/discount half of dealing range]
  → M15_MSS           (Stage C: 15m MSS/ChoCh + optional displacement)
  → WAIT_FOR_CHOCH    (watch 1m nested ChoCh)
  → ENTRY_PENDING     (1m FVG limit)
  → IN_TRADE
```

`IN_POI` remains for Stage D (POI retest gate) — not in A–C.

New enum values (when wired): `SWEEP_TOUCHED`, `M15_MSS`. Until wired, gates can run **inside** existing transitions without new states (see Integration).

---

## Context fields (`SmcContext`)

| Field | Set when | Used for |
|-------|----------|----------|
| `sweep_level` | Sweep detected | A: close-back vs level |
| `sweep_extreme` | Wick extreme on sweep bar | Invalidation / SL anchor |
| `dealing_range_low/high` | After A confirmed | B: premium/discount |
| `m15_mss_bar_i` | Stage C pass | C: FVG must come from displacement leg after this bar |

---

## Stage A — Sweep quality

**ICT rule:** Sweep = raid + **rejection**; do not treat breakout closes as sweeps.

| Config | Type | Default | Meaning |
|--------|------|---------|---------|
| `flow_a_sweep_close_back` | bool | `false` | Require confirming bar close back inside range relative to level |
| `flow_a_sweep_close_mode` | `"wick"` \| `"body"` | `"body"` | `body`: close must be on safe side of level; `wick`: same but allow wick-only path when close_back off |

**Logic** (`flow_filters.sweep_close_back_ok`):

- **Bullish** (sell-side sweep / `*_low`, `pdl`, `pwl`): `low < level` and `close > level` (or `>=` with buffer).
- **Bearish** (buy-side sweep / `*_high`, `pdh`, `pwh`): `high > level` and `close < level`.

**Integration point:** After `LiquidityEngine.detect_sweeps` returns a hit on bar `i`, if `flow_a_sweep_close_back`, call filter with `(open, high, low, close, level, is_high_pool)` before `sm.on_sweep`.

**Invalidation (existing + A):** If price later trades beyond `sweep_extreme`, expire setup (`reset_setup` or `on_sweep_invalidation`).

**Ablation metric:** Trades/week ↓, WR ↑, OOS PnL; compare Jul 2026 replay still fires.

---

## Stage B — Premium / discount

**ICT rule:** After sweep, define a **dealing range**; longs prefer **discount** (below equilibrium), shorts **premium** (above).

| Config | Type | Default | Meaning |
|--------|------|---------|---------|
| `flow_b_premium_discount` | bool | `false` | Gate 15m align / 1m watch on location |
| `flow_b_range_mode` | `"15m_swing"` \| `"sweep_bar"` | `"15m_swing"` | How to set dealing range |
| `flow_b_equilibrium` | float | `0.5` | Fib mid of range (0.5 = classic EQ) |

**Range modes:**

- `15m_swing`: `dealing_range_low` = last confirmed 15m swing low before sweep; `high` = last swing high (causal at sweep bar).
- `sweep_bar`: min/max of sweep bar + prior N 15m bars (N configurable later; start with sweep bar wick extremes).

**Logic** (`flow_filters.premium_discount_ok`):

- `eq = low + (high - low) * flow_b_equilibrium`
- Bullish bias: `price <= eq` (discount) at gate check (use sweep bar **close** or current 15m close).
- Bearish bias: `price >= eq` (premium).

**Integration point:** Before `on_15m_align` → `WAIT_FOR_CHOCH`, or when arming 1m watch.

**Ablation:** Enable only after A (or alone) on 2024→now.

---

## Stage C — 15m MSS + displacement

**ICT rule:** After sweep, need **lower-TF structure shift** with **displacement** (impulsive leg), not only “15m bias == 4h”.

| Config | Type | Default | Meaning |
|--------|------|---------|---------|
| `flow_c_require_15m_mss` | bool | `false` | Replace loose align with MSS requirement |
| `flow_c_displacement` | bool | `false` | Require displacement candle on 15m at/after ChoCh |
| `flow_c_displacement_atr_mult` | float | `1.0` | Min \|body\| / ATR on 15m for displacement bar |
| `flow_c_align_fallback` | bool | `true` | If `require_15m_mss` off, keep current `aligned OR choch_ok` |

**MSS satisfied when** (`flow_filters.m15_mss_ok`):

1. `snap_15m.last_choch_dir == int(4h_bias)` (body-close ChoCh already in engine), **and**
2. ChoCh bar index `>= sweep_bar_15m_i` (causal: shift after sweep), **and**
3. If `flow_c_displacement`: `abs(close-open) >= atr * flow_c_displacement_atr_mult` on that bar.

**1m entry:** Unchanged nested ChoCh + FVG; optional later: `fvg.created_idx >= m15_mss_bar_i`.

**Integration point:** Replace or narrow `sm.on_15m_align` success path when `flow_c_require_15m_mss`.

---

## Evolution (flow — separate from exit evolution)

Use **one flag per round** (same as `one_variable_only` in goal.yaml).

```python
SmcIctConfig.flow_evolution_space()  # categorical only, stages A–C
```

Suggested order for hill-climb / ablation:

1. `flow_a_sweep_close_back` on/off  
2. `flow_b_premium_discount` on/off (+ `flow_b_range_mode`)  
3. `flow_c_require_15m_mss` on/off (+ `flow_c_displacement`)  

Do **not** mutate `STRATEGY_LOCKS` (nested ChoCh, `fvg_edge`, no swing liq, 4h L=10).

**Score:** Keep composite score; log WR and trades/week per variant. Prefer configs that improve **OOS** on 2024→now without breaking reference replay.

---

## Test plan

| Test | File |
|------|------|
| A: close back inside for high/low pools | `tests/test_smc_flow_filters.py` |
| B: EQ discount/premium | same |
| C: MSS + displacement threshold | same |
| Reference short Jul 2026 | `research_smc_replay_ref_short.py` (must still pass with all flags off) |

---

## Wiring checklist (implementation PRs)

- [x] **PR1:** A only — sweep confirmation in `smc_ict.run_backtest` 1m loop  
- [x] **PR2:** B — populate dealing range on sweep confirm  
- [x] **PR3:** C — gate `on_15m_align`; store `m15_mss_bar_i`  
- [ ] **PR4:** Evolution worker includes `flow_evolution_space()`  
- [ ] **PR5:** Funnel script `research_smc_funnel.py` logs drop-off per stage  
