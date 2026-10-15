# MT5 Demo — trading-lab ST `hf_ls01`

## 1. Datei kopieren
1. MetaTrader 5 öffnen  
2. Menü **Datei → Datenordner öffnen**  
3. Ordner `MQL5/Experts/`  
4. Datei legen:  
   `trading-lab/mt5/trading-labST_hf_ls01.mq5`

## 2. Kompilieren
1. **MetaEditor** öffnen (F4 in MT5)  
2. Datei öffnen → **Compile** (F7)  
3. Fehlerfrei → erscheint unter Navigator → Expert Advisors

## 3. Demo-Chart
1. Symbol wählen: z.B. `NAS100`, `US100`, `USTEC` (oder `XAUUSD` zum Vergleich)  
2. Timeframe: **H1**  
3. EA per Drag & Drop auf Chart  
4. Haken: **Algo Trading** / Automated Trading erlauben  
5. Inputs prüfen:
   - `InpRiskPercent` = **2.0** (Default; Apex-Übung)  
   - `InpLongOnly` = **false**  
   - `InpOneTradePerDay` = **true**

## 4. Was der EA macht
- Supertrend **7 / 1.5** Flip + Volume-Median-Filter  
- Long + Short  
- SL = 1.5×ATR, TP1 = 2R (60% schließen → SL auf BE), Rest per Supertrend-Trail  
- Max Hold 48 H1-Bars  
- Max 1 Position, optional 1 Trade/Tag  

## 5. Hinweise
- CFD-Symbole ≠ Apex Futures (Punktwert/Spread anders) — Demo nur zum Flow-Test  
- Für Apex später: gleiche Logik auf NQ/MNQ in Tradovate/Rithmic portieren oder Signale manuell  
- Zuerst **Strategy Tester** (H1, 1–2 Jahre) laufen lassen, bevor Live-Demo-Geld
