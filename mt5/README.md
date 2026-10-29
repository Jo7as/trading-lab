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

---

# Kosten-Logger `TradingLabCostLogger.mq5` (echte Spreads, Swaps, Slippage)

Misst auf einem Demo-/Prop-Konto die Kosten aller 26 Märkte der Trendfolge S-007.
Standard: **nur lesen** (keine Orders). Mit `InpProbe = true` sendet er einmal pro
Handelstag um 15:00 New York je Markt eine Mini-Order (kleinste Lotgröße), schließt sie nach
5 Sekunden und protokolliert Fill, Slippage und Kommission - **nur auf Demokonten** (der EA
bricht sonst ab) und schließt nur seine eigenen Positionen (Magic 20261029).

1. Datei nach `MQL5/Experts/` kopieren, im MetaEditor kompilieren (F7).
2. EA auf **einen** beliebigen Chart ziehen (er liest alle Symbole selbst), Algo-Trading an.
3. Beim ersten Start schreibt er `tl_allsymbols_<login>.csv` mit allen Symbolnamen des
   Brokers. Stimmen Namen nicht (z.B. `US500` vs. `SPX500`, `UKOIL` vs. `BRENT`), in den
   Eingaben `InpMap1..4` anpassen (`Labor-Name=Broker-Name`). Fehlende Symbole stehen im
   Journal ("symbol not found").
4. **Kurz-Sitzung** (reicht für Swaps + eine Slippage-Messung): `InpProbe = true` und `InpProbeNow = true`,
   ~5 Minuten laufen lassen, bis im Journal "one-shot session finished" steht, dann den EA entfernen.
   Für Spreads über den Tag: optional länger laufen lassen (Spreads schwanken nach Uhrzeit).
5. Dateien liegen unter `Datei -> Datenordner öffnen -> MQL5/Files`. Auswertung:

       uv run python studies/broker_costs_report.py "<Pfad zu MQL5/Files>" <Login> --name fundednext_demo
       uv run python studies/tsmom_check.py --referenz --broker artifacts/broker_costs/fundednext_demo.json

Der EA verändert keine Einstellungen des Kontos und speichert keine Zugangsdaten.
