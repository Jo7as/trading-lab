# ICT Silver Bullet — trading-lab implementation notes

Mechanical encoding of the ICT Silver Bullet checklist in
`trading_lab/strategies/silver_bullet.py`.

## Rules encoded

1. Windows (NY local): London 03–04, NY AM 10–11, NY PM 14–15  
2. At window open: nearest BSL/SSL from 15m session/PD/PW/swings  
3. In-window wick sweep of one side → bias toward opposing pool  
4. 1m ChoCh toward draw + FVG on that bar  
5. Optional: FVG in premium/discount vs BSL–SSL; displacement body ≥ ATR  
6. Limit at FVG edge; SL beyond FVG candle / sweep; TP = fixed `tp_r` (3R)  
7. One trade per window; management may extend past the clock  

## Backtest XAU/USD 2024-01 → 2026-07 (1% risk)

| Variant | n | WR | PnL | PF |
|---------|---|----|-----|----|
| all windows + PD | 312 | 29.5% | −0.11% | 0.99 |
| NY AM + PD | 91 | 31.9% | +0.04% | 1.01 |
| NY AM no PD | 220 | 30.5% | −3.1% | 0.84 |
| all no PD | 707 | 29.7% | −3.0% | 0.93 |

Claimed discretionary band (~55–65% WR @ 1:3) **not reproduced**. Edge ≈ 0.

## Takeaway

Silver Bullet is clean as a *checklist*, but a strict causal automation on spot gold
does not unlock the nested-MTF SMC problem either. Further discretion (named draw,
session context, skipping ugly windows) is exactly what we cannot encode without
overfitting.

Research artifact: `research/silver_bullet_xauusd_2024.json`  
Runner: `research_silver_bullet.py`
