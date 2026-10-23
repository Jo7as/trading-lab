# Strategie-Agent: Plan für die Backtest-Umgebung

Grundlage: der 10-stufige Prozess aus dem QNTFRAME-Video („Lab Notes 001 – From ideas to systems“), übersetzt in einen **Agenten mit festen Stufen, Datenformaten und Prüfschranken**. Er arbeitet die Hypothesen aus dem [Multi-Strategie-Plan](Multi_Strategie_Algo_Dukascopy.md) ab und kann später eigene Hypothesen ergänzen.

---

## 0. Inhalt des Videos (Transkript der Untertitel und Folien)

| # | Folie | Untertitel | Kernaussage auf der Folie |
|---|---|---|---|
| 01 | „A trading idea is only the beginning.“ | „A trading idea is only the beginning.“ | Eine Beobachtung ist nur eine Frage („Why?“), die noch getestet werden muss |
| 02 | „Observe. Question. Define.“ (Research) | „At QNTFRAME, we turn market observations and research into testable hypotheses.“ | 01 Market Observations: wiederkehrendes Verhalten finden → 02 Research Papers: fragen, was es erklären könnte → 03 Testable Hypothesis: eine prüfbare Behauptung formulieren |
| 03 | „Context changes behaviour.“ (Regime Filters) | „We define the rules, then study how they behave as trend and volatility change.“ | Regime: Trend / Range / Volatilität |
| 04 | „One history. Many possible paths.“ (Monte Carlo) | „Monte Carlo simulations shuffle trades, resample blocks, and vary execution costs, to explore a range of possible outcomes.“ | Shuffle · Block Resample · Vary Costs |
| 05 | „Test beyond the data you fit.“ (Overfitting Control) | „To reduce overfitting, we keep rules simple, run walk-forward tests, and reserve data the strategy has never seen.“ | Development (Train + Validate) gegen Holdout („untouched, reserve until the final check“); Walk-Forward Fit → Test; „Simple rules / fewer free parameters“ |
| 06 | „Make every rule explicit.“ (Development) | „If the evidence holds up, we translate those rules into code.“ | Pseudocode: `signal = evaluate(market_data)`, `risk = check_limits(account)`, `if signal and risk.allowed: size = calculate_position(); order = execute(size); monitor(order)` |
| 07 | „Rules become actions.“ (Execution) | „The algorithm identifies signals, sizes positions, and opens and manages trades within predefined risk limits.“ | Signal → Position Size → Order → Risk Control |
| 08 | „Automate. Monitor. Refine.“ (Monitoring) | „We forward-test, log what happens, and keep monitoring.“ | Forward Test / Paper Execution · Event Log / Signal Received · Risk Check / Limits Applied · Health Check / Connection Active |
| 09 | „Keep perspective.“ (Research / Risk) | „Backtests do not guarantee future performance.“ | Backtests beschreiben einen historischen Test, die Zukunft bleibt unsicher |
| 10 | „From ideas to systems.“ | „This is QNTFRAME. From ideas to systems. Follow the research.“ | Abspann |

**Bewertung:** Der Prozess im Video ist fachlich sauber und deckt sich mit unserem bisherigen Vorgehen. Das Video bleibt auf Überschriften-Ebene. Dieser Plan ergänzt, was für einen **automatisierten Agenten** fehlt: konkrete Datenformate, Prüfschranken mit Zahlen, Rechenverfahren und Regeln, die der Agent nicht umgehen darf.

---

## 1. Architektur des Agenten

```
                ┌──────────────────────────────────────────────────────────┐
                │                 ORCHESTRATOR (Zustandsmaschine)           │
                │  verwaltet Hypothesen-Warteschlange, Gates, Versuchszähler │
                └──────────────────────────────────────────────────────────┘
   S1 Research ─► S2 Hypothese ─► S3 Spezifikation ─► S4 Code ─► S5 Backtest (Train/Validate)
                                                                      │
        ┌─────────────────────────────────────────────────────────────┘
        ▼
   S6 Regime-Analyse ─► S7 Monte Carlo ─► S8 Overfitting-Kontrolle (Walk-Forward, Quer-Validierung)
                                                                      │
                                                     Gate bestanden?  ▼
                                    S9 Holdout (einmalig) ─► S10 Portfolio-Einbindung
                                                                      ▼
                                    S11 Paper/Forward-Test ─► S12 Monitoring & Refine (zurück zu S1)
```

**Grundregeln für den Agenten:**
1. **Jede Stufe erzeugt ein Artefakt** (JSON/YAML + Report). Die nächste Stufe arbeitet nur mit diesem Artefakt.
2. **Jede Stufe hat ein Gate.** Wer es nicht besteht, geht mit Begründung zurück oder wird verworfen. Ein Gate darf nie übersprungen werden.
3. **Der Holdout-Datensatz ist für den Agenten technisch gesperrt.** Nur der Orchestrator gibt ihn in S9 frei, einmal pro Strategie-Version.
4. **Jeder Backtest-Lauf wird im Versuchsregister gezählt** (Grundlage der Deflated Sharpe).
5. **Einfachheit:** höchstens 5 freie Parameter pro Strategie, davon höchstens 3 optimiert.

---

## 2. Datenformate (Artefakte)

### 2.1 Hypothesen-Karte (Ergebnis von S2)
```yaml
id: H-2026-001
titel: "Absorption an Order-Cluster-Marken führt zur Umkehr"
beobachtung: "An Vortageshoch/-tief kehrt der Kurs trotz starkem Kaufdruck oft um"
mechanismus: "Take-Profit-Cluster an Marken (Osler 2000/2001); Kursreaktion auf OFI linear (Cont et al. 2014)"
quellen: ["Osler 2000 FRBNY EPR", "Cont/Kukanov/Stoikov 2014 JFEC"]
behauptung: "Bei |OFI_z| >= 2 und AQ >= 0,7 in einer Markenzone ist die mittlere Mid-Rendite der nächsten 30 Min gegen die Druckrichtung > 0"
dosis_wirkung: "Effekt steigt mit AQ-Quintil"
placebo: "Gleiche Bedingung an Zufallsniveaus ohne Marke zeigt keinen Effekt"
instrumente_universum: "gemäß Kostenkennzahl CR <= 0,10"
erwarteter_horizont_min: 30
kill_kriterien:
  - "t-Wert < 3 in der Event-Studie"
  - "kein Unterschied zum Placebo"
  - "Effekt pro Trade < 2 x Round-Trip-Kosten"
status: offen
```

### 2.2 Strategie-Spezifikation (Ergebnis von S3)
```yaml
id: S-M2-v0
hypothese: H-2026-001
daytrading: true                # Pipeline-Fehler bei Übernacht-Positionen
instrumente: auto_cr_filter
session: siehe Multi-Strategie-Plan 1.3
signal:
  inputs: [mid, bid, ask, bid_vol, ask_vol, ofi, sigma_1m, levels]
  regeln: "… exakt formuliert, keine Interpretationsspielräume …"
risiko:
  stop: "k_stop * sigma_1m * sqrt(15) jenseits der Marke"
  ziel: "min(TWAP, 2R)"
  zeit_stop_min: 60
  max_trades_je_instrument_tag: 2
sizing: "Vola-Ziel, Cluster-Gewichtung (Multi-Plan 10.7)"
parameter:                      # max. 5 frei, max. 3 optimiert
  - {name: ofi_z_min, default: 2.0, raster: [1.5, 2.0, 2.5, 3.0], optimiert: true}
  - {name: aq_fade_min, default: 0.7, raster: [0.5, 0.6, 0.7, 0.8], optimiert: true}
  - {name: k_stop, default: 1.5, raster: [1.0, 1.5, 2.0], optimiert: true}
  - {name: ofi_window_s, default: 60, optimiert: false}
  - {name: zone_width, default: 1.0, optimiert: false}
kosten_modell: EXEC-v1
```

### 2.3 Backtest-Ergebnis (Ergebnis von S5–S8)
```yaml
lauf_id: R-000123
strategie: S-M2-v0
parameter: {ofi_z_min: 2.0, aq_fade_min: 0.7, k_stop: 1.5}
zeitraum: {train: "2014-01..2019-12", validate: "2020-01..2023-12"}
kennzahlen_netto: {sharpe, sortino, cagr, mdd, monatsrendite_median, trefferquote, profit_factor, trades_pro_tag, kosten_anteil, breakeven_slippage}
je_instrument: [...]
je_regime: {trend: {...}, range: {...}, hohe_vola: {...}, niedrige_vola: {...}}
monte_carlo: {p05_cagr, p50_cagr, p95_mdd, p_monat_unter_minus5, p_mdd_ueber_30}
walk_forward: {wfe, oos_sharpe_je_fenster: [...]}
quer_validierung: {median_ratio_b_a, anteil_b_positiv}
versuchsregister: {n_trials_gesamt: 412, deflated_sharpe_p: 0.97}
gate: {bestanden: true, gruende: [...]}
```

### 2.4 Versuchsregister (zentral, nur anfügbar)
Jede Zeile: Zeitstempel, Strategie-ID, Parameter, Datenabschnitt, Instrumente, Sharpe, Anzahl Trades. **Der Agent kann Einträge weder löschen noch ändern.**

---

## 3. Die Stufen im Detail

### S1 Research: Beobachten (Video 02, Schritt 01–02)
**Aufgabe:** Wiederkehrendes Marktverhalten finden und eine mögliche Erklärung aus der Literatur zuordnen.
**Eingaben:** Dukascopy-1-Min-Daten (nur Train-Zeitraum!), Literaturliste, bestehende Hypothesen.
**Werkzeuge:** deskriptive Statistik je Instrument und Session-Stunde (Volatilität, Spread, Autokorrelation auf 1/5/15/30 Min, Häufigkeit von Noise-Area-Ausbrüchen, Verhalten an Marken).
**Ausgabe:** Liste von Beobachtungen, jeweils mit Kandidat für einen Mechanismus.
**Gate S1:** Zu jeder Beobachtung gibt es einen **ökonomischen Mechanismus** (wer handelt warum so?). Beobachtungen ohne Mechanismus werden nicht weiterverfolgt. Keine Chartmuster.

### S2 Hypothese definieren (Video 02, Schritt 03)
**Aufgabe:** Eine Hypothesen-Karte (2.1) schreiben, mit messbarer Behauptung, Dosis-Wirkung, Placebo und Kill-Kriterien.
**Gate S2:** Alle Felder ausgefüllt, Behauptung ist mit einer Zahl prüfbar, Kill-Kriterien stehen **vor** dem Test fest.

### S2b Event-Studie (Vortest ohne Handelsregel)
**Aufgabe:** Mittleren Kursverlauf nach Eintreten der Bedingung messen, mit Konfidenzband, Dosis-Wirkung (Quintile) und Placebo. Nur Train-Zeitraum.
**Gate S2b:** t ≥ 3, Dosis-Wirkung sichtbar, Placebo deutlich schwächer, Effekt ≥ 2 × Round-Trip-Kosten. **Sonst: Hypothese verworfen** (kein Nachbessern).

### S3 Spezifikation: jede Regel explizit (Video 06)
**Aufgabe:** Strategie-Spezifikation (2.2) schreiben. Jede Regel ist eindeutig, ohne Ermessensspielraum.
**Gate S3:** Höchstens 5 freie Parameter (3 optimiert). Alle Parameter dimensionslos (σ-Einheiten, Spread-Einheiten, Session-Minuten). Daytrading-Flag gesetzt.

### S4 Code (Video 06, Pseudocode erweitert)
Feste Struktur, die jede Strategie implementieren muss:
```python
class Strategy:
    def prepare(self, date, history) -> State: ...            # nur Daten bis Vortag
    def evaluate(self, market_data, state) -> Signal | None: ...   # nur Daten bis zum aktuellen Tick

def on_market_event(market_data, account, strategy, state):
    signal = strategy.evaluate(market_data, state)              # 1 SIGNAL
    risk = check_limits(account, signal)                         # 4 RISK CONTROL (vorab)
    if signal and risk.allowed:
        size = calculate_position(signal, account, risk)         # 2 POSITION SIZE (Vola-Ziel, Exposure-Grenzen)
        order = execute(size, signal.order_type)                 # 3 ORDER (Market/Limit/Stop, Kostenmodell)
        monitor(order)                                           # Stops, Ziele, Zeit-Stop, Session-Ende
    log_event(market_data.ts, signal, risk, order_state)        # Event-Log (Video 08)
```
**Pflicht-Tests (automatisch):**
- **Look-ahead-Test:** Wenn man die Zukunftsdaten abschneidet, dürfen sich die Signale der Vergangenheit nicht ändern.
- **Daytrading-Test:** null Positionen über das Session-Ende.
- **Determinismus:** Gleiche Eingabe ergibt identische Trades.
- **Kostentest:** Bei Kosten = 0 muss der Gewinn ≥ dem Gewinn mit Kosten sein. Bei doppelten Kosten muss er sinken.
- **Einheitentest:** Kein Parameter in absoluten Einheiten.

**Gate S4:** alle Tests grün.

### S5 Backtest: Train + Validate (Video 05, linker Block)
**Datenaufteilung (fest, vom Orchestrator vorgegeben):**
| Abschnitt | Zeitraum (Beispiel) | Zweck |
|---|---|---|
| Train | 2012–2019 | Parameter-Raster, Plateau-Suche |
| Validate | 2020 bis 24 Monate vor heute | Prüfung der gewählten Parameter |
| **Holdout** | letzte 24 Monate | **gesperrt bis S9** |

**Ablauf:** Default-Parameter → Raster auf Train (gepoolt über alle Instrumente, geglättete Zielfunktion, siehe Multi-Plan 10.3) → **eine** gewählte Einstellung auf Validate.
**Gate S5:**
- Validate-Netto-Sharpe ≥ 0,6 und ≥ 50 % der Train-Sharpe
- ≥ 60 % der Instrumente netto positiv
- Zufalls-Benchmark (100 Läufe mit zufälliger Richtung, gleiches Risikomanagement) geschlagen: über dem 95-%-Quantil
- Break-even-Zusatzslippage ≥ 2 × Default

### S6 Regime-Analyse: Kontext ändert Verhalten (Video 03)
**Regime-Definitionen** (je Tag und Instrument, nur Daten bis Vortag):
| Regime | Definition |
|---|---|
| Trend | Efficiency Ratio der letzten 20 Tage (Nettobewegung / Summe der Einzelbewegungen) im oberen Drittel |
| Range | Efficiency Ratio im unteren Drittel |
| Hohe Volatilität | 20-Tage-Volatilität über dem 70. Perzentil (1 Jahr) oder VIX über dem 70. Perzentil |
| Niedrige Volatilität | unter dem 30. Perzentil |

**Aufgabe:** Alle Kennzahlen aus S5 getrennt je Regime (und je Kombination Trend/Range × Vola hoch/niedrig) berichten.
**Erlaubte Folgerung:** **höchstens ein** Regime-Filter je Strategie (z. B. „M3 nur im Range-Regime“), und nur wenn er einen Mechanismus hat und in Train **und** Validate wirkt.
**Gate S6:** Die Strategie verliert in keinem Regime mit mehr als 20 % Zeitanteil deutlich (Sharpe < −0,5), oder ein begründeter Regime-Filter schaltet dieses Regime ab.

### S7 Monte Carlo: eine Historie, viele mögliche Pfade (Video 04)
Drei Verfahren, jeweils 5.000 Läufe auf den Validate-Trades bzw. -Tagesrenditen:
| Verfahren | Vorgehen | Was es zeigt |
|---|---|---|
| **Shuffle** | Reihenfolge der Trades zufällig mischen | Drawdown-Verteilung, wenn Verluste ungünstig gebündelt auftreten |
| **Block Resample** | Tagesrenditen in Blöcken à 20 Tagen mit Zurücklegen ziehen | erhält Phasen (Regime-Cluster), realistischere Pfade |
| **Vary Costs** | je Lauf Spread-Aufschlag zufällig aus [0; +100 %], Zusatzslippage aus [0; 2 × Default], Latenz aus [100; 1.000 ms] | Robustheit gegen schlechtere Ausführung |

**Ausgabe:** 5., 50., 95. Perzentil von CAGR, Monatsrendite und MDD; Wahrscheinlichkeit eines Monats < −5 %; Wahrscheinlichkeit eines MDD > 30 %; Wahrscheinlichkeit, nach 12 Monaten im Minus zu sein.
**Gate S7:** 5.-Perzentil-CAGR > 0 bei variierten Kosten; P(MDD > 30 %) ≤ 5 %; P(12 Monate negativ) ≤ 15 %.

### S8 Overfitting-Kontrolle: Test jenseits der angepassten Daten (Video 05)
| Prüfung | Verfahren | Schwelle |
|---|---|---|
| **Walk-Forward** | 3 Jahre Fit → 1 Jahr Test, rollierend über Train + Validate | Walk-Forward-Effizienz (OOS-Sharpe / IS-Sharpe) ≥ 0,5; ≥ 70 % der Testfenster positiv |
| **Parameter-Stabilität** | Sharpe bei ±1 Rasterschritt um die gewählten Werte | Abfall ≤ 30 % (Plateau) |
| **Quer-Validierung über Instrumente** | 20 zufällige A/B-Teilungen je Cluster + Leave-one-cluster-out (Multi-Plan 10.5) | Median Sharpe_B / Sharpe_A ≥ 0,5; Sharpe_B > 0 in ≥ 80 % |
| **Deflated Sharpe** | mit allen Versuchen aus dem Versuchsregister | P(echte Sharpe > 0) ≥ 95 % |
| **Einfachheit** | Anzahl optimierter Parameter | ≤ 3 |

**Gate S8:** alle Schwellen erfüllt. Danach wird die Strategie-Version **eingefroren** (Hash von Code + Parametern).

### S9 Holdout: der finale Check (Video 05, „Reserve until the final check“)
- Der Orchestrator gibt den Holdout **einmal** für die eingefrorene Version frei.
- **Gate S9:** Netto-Sharpe im Holdout ≥ 0,5 × Validate-Sharpe und > 0; MDD ≤ 1,5 × Validate-MDD; Daytrading-Check bestanden.
- **Durchgefallen:** Diese Version ist endgültig verworfen. Eine neue Version muss wieder bei S3 beginnen und bekommt **keinen** neuen Holdout. Der Holdout zählt dann als „verbraucht“, und der Orchestrator verlängert die Sperrzone um die Zeit, die seither neu hinzugekommen ist.

### S10 Portfolio-Einbindung
- Bestandene Strategien werden Module im Multi-Strategie-Portfolio (Regime-Steuerung R, Portfolio-Schicht P, Multi-Plan Abschnitte 8–9).
- **Portfolio-Gate (Vorgabe 2–4 % pro Monat):** Holdout-Netto-Sharpe des Portfolios ≥ 2,0, Median-Monatsrendite ≥ 2 % beim gewählten Vola-Ziel, MDD ≤ 20 %, Monte Carlo wie S7 auf Portfolio-Ebene.

### S11 Forward-Test / Paper-Execution (Video 07–08)
- Ausführung über Demo-Konto (z. B. Dukascopy JForex), alle Module gleichzeitig, mindestens 3 Monate.
- **Ablauf je Marktereignis** wie Pseudocode in S4: Signal → Risiko-Check → Positionsgröße → Order → Überwachung.
- **Abgleich mit dem Backtest:** Jeder Live-Signalzeitpunkt wird parallel im Backtest-Simulator nachgespielt.

**Gate S11:**
- Abweichung der Fill-Preise zum Simulator im Median ≤ Default-Slippage
- Signalübereinstimmung Live gegen Simulator ≥ 95 %
- Netto-Sharpe im Forward-Test innerhalb des 90-%-Bands aus der Monte-Carlo-Simulation

### S12 Monitoring und Refine (Video 08)
**System-Monitor (laufend):**
| Kanal | Inhalt | Alarm |
|---|---|---|
| Event-Log | jedes Signal, jede Order, jeder Fill, jede Ablehnung mit Zeitstempel und Grund | – |
| Risk Check | angewandte Grenzen (Exposure je Währung, Tagesverlust, Drawdown-Stufe) | jede ausgelöste Grenze |
| Health Check | Datenverbindung, Tick-Lücken > 30 s, Order-Bestätigungen, Uhrzeit-Drift | sofort; neue Orders sperren |
| Performance | rollierende Sharpe und Drawdown je Modul gegen das Monte-Carlo-Band | Unterschreitung des 5. Perzentils |
| Ausführung | Slippage je Instrument und Stunde gegen Annahme | Median > 1,5 × Annahme |

**Refine-Regeln (vorab festgelegt):**
- Modul unter dem 5. Monte-Carlo-Perzentil → Budget halbieren (Multi-Plan 9.5).
- Modul 6 Monate mit Sharpe < 0 und DD > 2 × erwartet → aussetzen, zurück zu S1 mit neuer Beobachtung.
- **Parameter werden live nie ad hoc geändert.** Jede Änderung ist eine neue Version und durchläuft S3–S9.

### S13 Perspektive (Video 09)
Jeder Report endet mit einem festen Hinweis: Backtests beschreiben einen historischen Test, die künftige Performance bleibt unsicher. Dazu kommen die Planungsannahmen: Live-Sharpe ≈ 0,5 × Backtest-Sharpe bis zum Gegenbeweis und das 5.-Perzentil-Szenario aus S7.

---

## 4. Orchestrator: Regeln und Warteschlange

### 4.1 Startwarteschlange (aus dem Multi-Strategie-Plan)
| Priorität | Hypothese | Modul | Erster Schritt |
|---|---|---|---|
| 1 | Noise-Area-Momentum auf FX, Metallen, Energie, Indizes | M1 | S2b (Häufigkeit und Folgerendite von Noise-Area-Ausbrüchen je Cluster) |
| 2 | Level-1-OFI taugt als Orderflow-Maß (Vortest) | M2 | Regression ΔMid gegen OFI (Multi-Plan 5.3), Kill bei R² < 0,15 |
| 3 | Absorption vs. Durchbruch an Marken | M2 | S2b mit AQ-Quintilen und Placebo |
| 4 | Mean Reversion innerhalb der Noise Area, spread-gefiltert | M3 | S2b: Verlauf nach \|z\| ≥ 2 innerhalb gegen außerhalb der Noise Area |
| 5 | Intraday-Währungsstärke (Momentum gegen Umkehr) | M4 | S2b |

### 4.2 Harte Regeln (vom Agenten nicht änderbar)
1. Holdout-Zugriff nur über den Orchestrator in S9, einmal pro eingefrorener Version.
2. Versuchsregister ist nur anfügbar.
3. Maximal 3 Iterationen S3 → S8 pro Hypothese. Danach gilt die Hypothese als verworfen.
4. Instrumente werden nur über die Kostenkennzahl und Datenkriterien ausgewählt, nie nach Performance.
5. Keine Chartmuster, keine Übernacht-Positionen, keine absoluten Parameter.
6. Zielgröße bei der Optimierung ist **immer** die Netto-Sharpe (gepoolt), nie Rendite, Trefferquote oder Profit-Factor allein.
7. Das Renditeziel wird nie über zusätzlichen Hebel erreicht. Hebel ergibt sich ausschließlich aus dem Vola-Ziel der Portfolio-Schicht.

### 4.3 Rollen (falls der Agent aus mehreren KI-Agenten besteht)
| Rolle | Stufen | Darf | Darf nicht |
|---|---|---|---|
| Researcher | S1, S2, S2b | Train-Daten lesen, Literatur, Event-Studien | Strategie-Code schreiben, Validate/Holdout sehen |
| Engineer | S3, S4 | Spezifikation und Code, Tests | Parameter wählen |
| Tester | S5–S8 | Backtests, Monte Carlo, Walk-Forward, Train + Validate | Holdout sehen, Code ändern |
| Auditor | Gates, S9 | Gate-Prüfung, Holdout-Freigabe, Versuchsregister prüfen | Strategien ändern |
| Operator | S11, S12 | Paper-Execution, Monitoring, Alarme | Parameter live ändern |

Die Trennung verhindert, dass dieselbe Instanz Regeln an Ergebnisse anpasst, die sie bereits gesehen hat.

---

## 5. Report-Vorlage je Strategie-Version

1. Hypothesen-Karte (Kurzfassung) und Mechanismus
2. Event-Studie (Plot, t-Werte, Dosis-Wirkung, Placebo)
3. Spezifikation (Regeln, Parameter, Code-Hash)
4. Backtest Train/Validate: Kennzahlen gepoolt, je Instrument, je Cluster, Kostenaufschlüsselung, Zufalls-Benchmark
5. Regime-Tabelle
6. Monte Carlo (drei Verfahren, Perzentile, Wahrscheinlichkeiten)
7. Overfitting-Kontrollen (Walk-Forward, Plateau, Quer-Validierung, Deflated Sharpe, Anzahl Versuche)
8. Gate-Entscheidungen mit Begründung
9. (falls freigegeben) Holdout-Ergebnis
10. Hinweis zur Perspektive (S13)

---

## 6. Umsetzungsreihenfolge für die Backtest-Umgebung

| Schritt | Baustein | Abhängig von |
|---|---|---|
| 1 | Daten-Layer: Dukascopy-Ticks → bereinigte 1-Min-Kerzen mit Bid/Ask/Spread/Ticks/OFI; yfinance-Kontext | – |
| 2 | Datenaufteilung Train/Validate/Holdout mit technischer Sperre | 1 |
| 3 | Versuchsregister (nur anfügbar) | – |
| 4 | Ausführungs- und Kostenmodell auf Tick-Basis (Multi-Plan Abschnitt 2) | 1 |
| 5 | Strategie-Schnittstelle + Pflicht-Tests (S4) | 4 |
| 6 | Event-Studien-Engine mit Placebo und Quintilen (S2b) | 1 |
| 7 | Backtest-Runner mit Rastersuche, gepoolter Zielfunktion, Zufalls-Benchmark (S5) | 4, 5 |
| 8 | Regime-Klassifikation (S6) | 1 |
| 9 | Monte-Carlo-Modul: Shuffle, Block-Resample, Vary Costs (S7) | 7 |
| 10 | Walk-Forward, Quer-Validierung, Deflated Sharpe (S8) | 3, 7 |
| 11 | Orchestrator mit Gates und Warteschlange | 2–10 |
| 12 | Report-Generator | 7–10 |
| 13 | Paper-Execution-Anbindung + System-Monitor (S11, S12) | 5, 11 |

*Keine Anlageberatung. Backtests beschreiben einen historischen Test, die künftige Performance bleibt unsicher.*
