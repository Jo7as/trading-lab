# SMC/ICT — PnL expectations, correct execution, gap vs trading-lab

Research summary (Jul 2026) for bringing the EUR/USD nested SMC system to a **2–4%/month** target at ~1% risk/trade.

Sources: ICT 2022 model guides ([ictkillzone.com](https://www.ictkillzone.com/ict-2022-model), [innercircletrader.net](https://innercircletrader.net/tutorials/complete-ict-trading-strategy-2022/)), prop-firm expectancy math ([PropJournal](https://propjournal.net/strategies/ict-smart-money)), forward-test commentary ([TGSignals SMC](https://tgsignals.com/tools/smart-money-concepts/)), public backtest claims (treat marketing carefully).

---

## 1. What “good” SMC actually claims (vs marketing)

### Credible mechanical picture

Serious ICT/prop writeups converge on **asymmetric R**, not high win rate:

| Metric | Typical “disciplined SMC” range |
|--------|----------------------------------|
| Win rate | **40–55%** (often ~45%) |
| Avg win | **2R–5R** (to opposing liquidity / ERL) |
| Avg loss | **~1R** (beyond sweep / OB) |
| Frequency | **1–3 quality setups / day max**, often **few / week** on one pair |
| Profit factor | **~1.5–2.0** if filters work |

PropJournal example (illustrative math, not a guarantee):

- 40% WR × 3R win − 60% × 1R loss = **+0.6R per trade**
- At **1% risk** → **+0.6% expectancy / trade**
- **~20 trades** → **~12%** (≈ a few months of selective trading, not “every month automatic”)

That path **can** produce **2–4%/month** *if*:

1. Expectancy stays ~**0.3–0.6R/trade**, and  
2. You get **~6–12 trades/month** (not 50 mediocre ones).

### Unreliable claims

- EA ads: **“20%/month, 5% DD”** — treat as marketing.
- Medium “2600 trades, 61% WR, PF 2.17” — often indicator-promotion; verify methodology before believing.
- Huge compounded % on futures (RTY ICT macros) often use aggressive sizing / different instruments — not transferable 1:1 to EURUSD 1% risk.

### Session reality ([TGSignals](https://tgsignals.com/tools/smart-money-concepts/))

- Same pattern **≠** same edge: expectancy varies strongly by **asset × NY hour**.
- **Killzone confluence ≈ ~3×** expectancy of standalone OB/FVG.
- Standalone FVG without killzone / sweep confluence often **net-negative** in their forward data.

---

## 2. ICT 2022 model — the execution sequence (must all be true)

From the 2022 Mentorship / Killzone guides, a valid trade is a **sentence**, not a loose state machine:

1. **Daily / HTF bias** before the session (structure + **named Draw on Liquidity / ERL**).
2. **Killzone** active: London ~02:00–05:00 ET, NY Open ~08:30–11:00 ET, (optional PM).
3. **Liquidity sweep** (Judas) of a pool **opposite** the intended delivery (SSL for longs, BSL for shorts).
4. **MSS + displacement** on LTF (5m/1m): impulsive break that leaves a clean **FVG**.
5. Entry at **FVG CE (50%)** or **OTE 62–79%** overlapping the FVG — not chase.
6. **Stop beyond the sweep extreme** (manipulation wick), not a tiny OB buffer if that sits inside the sweep.
7. **T1 = nearest IRL** (internal opposing pool); **T2 = ERL** (the HTF draw). Partial at T1, BE, runner to T2.

**Any NO → no trade.** Frequency drops; quality rises.

---

## 3. Math: why trading-lab ~+1% total ≠ 2–4%/month

Current default path (B + sell-side, 2024→now):

- ~**1.5 trades/week**, WR ~**49%**, total PnL ~**+1.3%** over ~2.5 years  
- Exit mix historically: **~54% full SL**, **~23% TP1+BE (~0.75R)**, **~21% TP2 (~2.25R)**  
- Net expectancy ≈ **near zero per trade** → months of noise, not 2–4%.

### What 2%/month needs at 1% risk

| Trades/month | Required expectancy / trade |
|--------------|-----------------------------|
| 4 | **0.50R** |
| 6 | **0.33R** |
| 8 | **0.25R** |
| 12 | **0.17R** |

Example that hits ~2–3%/month:

- 6 trades/month, **45% WR**, wins average **2.5R** (many runners to ERL), losses **1R**  
- E = 0.45×2.5 − 0.55×1 = **0.575R** → **~3.5%/month**

Our bottleneck is **not** “need higher WR” — it is **avg win too small / too many full SLs**.

---

## 4. Gap analysis — trading-lab vs textbook ICT

| ICT requirement | trading-lab today | Impact |
|-----------------|--------------|--------|
| Named **ERL / Draw** before entry | Missing (fixed `tp1_r`/`tp2_r`) | Caps wins; no “run to liquidity” |
| **T1 = IRL**, **T2 = ERL** | Fixed 1.5R / 3R + 50% off | Many “wins” stop at ~0.75R (TP1+BE) |
| SL beyond **sweep wick** | Often **1m OB ± tiny buffer** | Early full SL (54%) |
| Hard **Killzone** for entry | Optional / off by default | Asia/London fills dilute edge |
| **MSS + displacement** after sweep | Soft 15m align/ChoCh; C optional | Many weak confirms |
| HTF bias + **premium/discount** | B on (good) | Keep |
| Sell-side / long quality | Sell-side default (good) | Keep |
| One model, high confluence | Still many session/PD/PW mixes | Noise |

**Verdict:** The playbook shape (4h → sweep → 15m → 1m FVG) is directionally right, but **risk geometry and targets** are closer to a **scalp RR engine** than the **2022 ICT delivery model**. That alone explains sub-1% yearly-scale PnL.

---

## 5. Implementation roadmap (to make it “run”)

Prioritize **expectancy**, not parameter hill-climb on the current exit model.

### P0 — Exit / invalidation (largest PnL lever) ✅ shipped (strategy.yaml v04)

1. **`sl_mode=sweep`**: SL = `sweep_extreme ± buffer`; if wider than **`max_sl_dist`** (~12 pips), fall back to OB.  
2. **`tp_mode=liquidity`**: T1 = nearest opposing pool (IRL); T2 = dealing-range / nearest HTF (ERL), capped by **`max_irl_r` / `max_erl_r`** (avoids stale multi-month PDH). RR fallback when no pool.  
3. **`move_sl_to_be_on_tp1=false`** by default so ERL runners are not killed at entry.

**2024→now (B + sell-side):** baseline OB+RR+BE ≈ **+1.27%** (PF 1.43); P0 ≈ **+1.15%** (PF 1.34). Competitive but not a PnL upgrade.

### P1 — Killzone / 15m MSS ✅ ablated (does not beat ctrl)

`flow_killzone` modes: `off|ny|ny_am|london|london_ny`. Script: `research_smc_p1_ablation.py` → `research/smc_p1_ablation_2024.json`.

| Rank | Variant | PnL | PF | n |
|------|---------|-----|----|---|
| 1 | **baseline\|ctrl** | **+1.27%** | 1.43 | 193 |
| 2 | p0\|ctrl | +1.15% | 1.34 | 174 |
| 3 | baseline\|kz_london_ny | +0.65% | 1.24 | 146 |
| 4–6 | baseline MSS / KZ+MSS | ≤ +0.57% | high PF | 30–84 |
| — | killzone-only / mss_disp | weak or negative | — | — |

MSS alone improves risk metrics (Sharpe/PF) but **cuts volume** and total PnL. Killzones over-filter. **Do not enable P1 by default.**

### P2 — Selectivity / next

7. Keep **B + sell-side**.  
8. Evolution may still mutate P0/P1 knobs; production defaults stay baseline exits.  
9. Monthly 2–4% target still unmet — needs better expectancy source (not more gates on this sample).

### Success criteria (honest)

- Expectancy **≥ 0.25R/trade** on 2024→OOS  
- **≥ 2%/month** annualized on fixed 1% risk (not compounding fantasy)  
- Max DD within goal (e.g. 8%)  
- Still pass Jul 2026 reference short replay (or document intentional divergence)

---

## 6. Default config (set Jul 2026)

In `smc-state/strategy.yaml` **v05** (post P0/P1 ablation):

- Flow: B + sell-side; killzone/MSS **off**  
- Exits: **`sl_mode=ob`**, **`tp_mode=rr`**, **`move_sl_to_be_on_tp1=true`** (~+1.27% / 2.5y)  
- P0 liquidity / P1 killzone remain implemented and evolvable, not default

---

## 7. Key links

- [ICT 2022 Model](https://www.ictkillzone.com/ict-2022-model)  
- [ICT Kill Zones](https://www.ictkillzone.com/ict-kill-zones)  
- [ICT OTE](https://www.ictkillzone.com/ict-ote)  
- [2022 Mentorship strategy overview](https://innercircletrader.net/tutorials/complete-ict-trading-strategy-2022/)  
- [PropJournal ICT expectancy sketch](https://propjournal.net/strategies/ict-smart-money)  
- [TGSignals SMC forward-test framing](https://tgsignals.com/tools/smart-money-concepts/)  
