Erstelle ein modulares, hochperformantes Trading-Strategie-Skript basierend auf Smart Money Concepts (SMC).

**DEINE ROLLE:**
Du bist ein Senior Quantitative Developer und Algorithmic Trading Experte mit jahrelanger Erfahrung in C++, Python, und MQL5/PineScript (passe dich dem gewünschten Framework des Nutzers an). Du schreibst Enterprise-Grade, fehlertoleranten, extrem schnellen (O(N) oder besser) und hochgradig modularen Code. Du verabscheust Spaghetti-Code, berücksichtigst Edge Cases (z.B. fehlende Daten, Repainting-Gefahren, Look-Ahead-Bias) und hältst dich strikt an Best Practices für objektorientiertes oder funktionales Design.

**ZIEL DER ARCHITEKTUR:**
Das System handelt eine Multi-Timeframe-Strategie (4h, 15m, 1m) als Nested Entry. Nach 4h-Bias und Liquidity-Sweep wartet es auf 15m-Angleichung (Bias oder ChoCh Richtung 4h), beobachtet dann den **1m-Chart jede Minute** bis zum 1m-ChoCh, und entered auf dem **1m-Pullback** (1m OB/FVG Limit). Ein 15m-POI-Retest ist *nicht* Pflicht vor dem 1m-ChoCh (kann später per Evolution getestet werden). HTF-Analysen nur bei Bar-Close / Cache.

**IMPLEMENTIERE DEN FOLGENDEN DETAILLIERTEN ABLAUFPLAN / REGELWERK:**

### 1. Trend Filter (Market Structure) — primär 4h
- **Primary Bias:** 4h External Swing Structure (Body Close für BoS; Wick ≠ Break).
- **15m:** Muss NICHT schon vor dem Sweep gleichgerichtet sein. Nach dem Sweep: 15m Bias == 4h **oder** 15m ChoCh in 4h-Richtung → dann 1m-Watch starten.
- **Performance-Tipp:** Caching für HTF-Strukturpunkte.

### 2. Location & Liquidity
- **Liquiditäts-Zonen:** Session Highs/Lows (Asia, London), PDH/PDL, PWH/PWL.
- **Sweep:** Wick durch Level in Setup-Richtung (Longs: Lows; Shorts: Highs).
- **15m POI (OB+FVG):** optionaler Kontext / für Research & spätere Evolution-Filter — kein harter Gate vor 1m-ChoCh.
- Untested FVG/OBs in Queues; nach Mitigation entfernen.

### 3. Confirmation (1m Timeframe)
- Sobald 15m aligned/ChoCh: State `WAIT_FOR_CHOCH` — **sofort** 1m Struktur jede Minute prüfen.
- 1m ChoCh in Richtung **4h** Bias (Body Close Pflicht).
- Danach: 1m Pullback-Entry (siehe §4). Der 1m-ChoCh muss nicht im 15m-POI liegen.

### 4. Entry Trigger
- Nach 1m ChoCh (`ENTRY_PENDING`): **1m Orderblock (inkl. 1m FVG)** des Impulses.
- Long: Buy-Limit an Oberkante; Short: Sell-Limit an Unterkante.
- Timeout/Invalidation wenn Struktur bricht ohne Fill.

### 5. Risk Management
- **SL:** Long unter 1m Swing Low (+ Buffer); Short über 1m Swing High (+ Buffer).
- **TP (Phase 1):** Fixed RR — TP1 = 1.5R (50%), TP2 = 3R (Runner).

**STATE MACHINE:**
`IDLE` → `HTF_ALIGNED` (4h) → `SWEPT` → `WAIT_FOR_CHOCH` (15m align; watch 1m) → `ENTRY_PENDING` (1m ChoCh + pullback limit) → `IN_TRADE`

Optional/Evolution später: `IN_POI` Retest-Gate, strengere OB/FVG-Filter.

**STRIKTE ANFORDERUNGEN AN DEN CODE & FEHLERVERMEIDUNG:**
1. State-Machine wie oben.
2. Modular: MarketStructureEngine, LiquidityEngine, SupplyDemandEngine, manage_position.
3. Kein Look-Ahead Bias.
4. Caching / keine O(n²)-Rescans pro Tick.
5. Configurability: Timeframes, Sessions, SL-Buffer.
6. Error Handling: fehlende Zonen, Division by zero, Disconnects.
