# Multi-Strategie-Daytrading-Algo (Datenbasis: Dukascopy + yfinance)

**Ein Algo, vier Strategien, eine Portfolio-Steuerung.** Ohne Chartmuster und ohne ereignisbasierte Zeitpunkte, ausschließlich intraday.

| Modul | Strategie | Verdient an … | Instrumente |
|---|---|---|---|
| **M1** | Noise-Area-Momentum (Trendfolge) | Trendtagen | erweitertes Universum (Abschnitt 10.2), Aufnahme nach Kostenkennzahl |
| **M2** | Absorption vs. Durchbruch an Order-Cluster-Marken (Level-1-Orderflow) | Wendepunkten und Ausbrüchen an Marken | erweitertes Universum, strengere Kostengrenze |
| **M3** | Mean Reversion *innerhalb* der Noise Area, spread-gefiltert | ruhigen Range-Tagen | erweitertes Universum, strengste Kostengrenze |
| **M4** (optional) | Intraday-Währungsstärke (relativ, dollarneutral) | Divergenz zwischen Währungen | 8 Währungen über ihre liquidesten Paare |
| **R** | Regime-Steuerung | – | gewichtet M1 gegen M3 nach Marktphase |
| **P** | Portfolio- und Risiko-Schicht | – | Risikobudget, Dollar-Exposure, Verlustgrenzen |

---

## 0. Ziel, Prinzip, Vorgaben

### 0.1 Warum mehrere Strategien
- Die Sharpe eines Portfolios aus N gering korrelierten Strategien liegt bei etwa Einzel-Sharpe × √N (bei Korrelation 0). Beispiel: 4 Module mit je 0,9 und geringer Korrelation → ca. 1,6–1,8.
- **Die Module sind bewusst als Gegenspieler gebaut:**
  - M1 verdient, wenn der Kurs die Noise Area verlässt (Trend).
  - M3 handelt nur, solange der Kurs *innerhalb* der Noise Area bleibt (Range).
  - M2 entscheidet an Marken, ob die Range hält oder bricht.
  - M4 ist dollarneutral und damit unabhängig von der allgemeinen Dollar-Richtung.
- An einem Tag, an dem eine Strategie verliert, hat die Gegenspieler-Strategie meist gute Bedingungen. Dadurch wird die Kapitalkurve glatter, *ohne* dass Rendite verloren geht.

### 0.2 Harte Vorgaben
- **Daytrading:** Alle Positionen werden spätestens zum Ende der Instrument-Session geschlossen. FX spätestens **16:45 New York** (vor dem Tageswechsel um 17:00 NY, an dem die Spreads stark steigen und Swap fällig wird). Die Pipeline prüft das automatisch.
- **Keine Chartmuster.** Marken (Vortageshoch und -tief, Asien-Range, runde Zahlen) gelten nicht als Muster, sondern als Orte, an denen sich nachweislich Orders häufen (Osler, Fed New York).
- **Renditeziel 2–4 % pro Monat:** nur erreichbar mit einer Portfolio-Sharpe nach Kosten von etwa 2 (siehe Gate in Abschnitt 11). Das Ziel wird **nicht** über zusätzlichen Hebel erzwungen.

---

## 1. Daten

### 1.1 Dukascopy (Hauptquelle, intraday)
| Feld | Inhalt |
|---|---|
| Zeitstempel | UTC, Millisekunden |
| Bid, Ask | bester Geld- und Briefkurs |
| BidVolume, AskVolume | angebotene Menge am besten Kurs in Mio. (nur Dukascopy-Pool) |

**Instrumente:** das gesamte Kandidaten-Universum aus Abschnitt 10.2 (FX-Majors und -Crosses, Metalle, Energie, Aktienindex-CFDs). Welche Instrumente tatsächlich gehandelt werden, entscheidet allein die Kostenkennzahl in 10.2. Die genauen Symbolnamen sind im Dukascopy-Katalog nachzuschlagen.
**Zeitraum:** so weit verfügbar, mindestens 2012 bis heute. **Holdout:** die letzten 24 Monate sind gesperrt (Abschnitt 11).

**Aufbereitung (Pflicht, in dieser Reihenfolge):**
1. Tick-Dateien je Stunde laden, nach UTC sortieren, Duplikate entfernen.
2. **Fehlerhafte Ticks entfernen:** Ask ≤ Bid; Sprung des Mid-Kurses um > 20 × die 1-Min-Standardabweichung, der innerhalb von 3 Ticks zurückkehrt; Spread > 50 × Median-Spread.
3. **Wochenenden und Feiertage:** Ticks zwischen Freitag 17:00 NY und Sonntag 17:00 NY verwerfen. Lücken > 5 Min innerhalb der Handelszeit protokollieren.
4. **1-Min-Kerzen** erzeugen, getrennt für Bid, Ask und Mid (OHLC), dazu: Anzahl Ticks, mittlerer Spread, maximaler Spread, mittlere BidVolume und AskVolume, OFI-Summe (Abschnitt 5.2).
5. Den **Tick-Stream** zusätzlich gespeichert lassen. M2 und das Ausführungsmodell brauchen Ticks.
6. Zeitzonen: Alle Regeln werden in Ortszeit formuliert (London, New York, Frankfurt) und über einen Zeitzonen-Kalender in UTC umgerechnet (Sommerzeitwechsel in Europa und USA liegen 2–3 Wochen auseinander!).

### 1.2 yfinance (nur Tagesdaten als Kontext)
| Symbol | Zweck |
|---|---|
| `^VIX` | Volatilitätsregime |
| `DX-Y.NYB` | Dollar-Index-Trend |
| `^GSPC`, `^GDAXI` | Aktienmarkt-Trend |
| `^TNX` | US-Zinsniveau (10 Jahre) |

- **Nur Schlusskurse vom Vortag verwenden** (kein Look-ahead).
- Fehlende Tage mit dem letzten gültigen Wert füllen und protokollieren.
- Optional: ein tägliches Skript, das ab sofort 1-Min-Daten von yfinance archiviert (für spätere Tests).

### 1.3 Sessions je Instrument (Parameter)
| Instrument | Session-Start (Noise-Area-Anker) | Handelsende (flat) | Asien-Range |
|---|---|---|---|
| FX-Majors, Crosses | 08:00 London | 16:45 New York | 00:00–07:00 London |
| Gold, Silber, Platin | 08:00 London | 16:45 New York | 00:00–07:00 London |
| Kupfer | 08:00 London | 13:55 New York | – |
| Brent, WTI | 09:00 New York | 14:25 New York | – |
| US500, US-Tech-100, US30 | 09:30 New York | 15:55 New York | – |
| DAX, Euro Stoxx 50 | 09:00 Frankfurt | 17:25 Frankfurt | – |
| FTSE 100 | 08:00 London | 16:25 London | – |
| CAC 40 | 09:00 Paris | 17:25 Paris | – |
| SMI | 09:00 Zürich | 17:15 Zürich | – |
| Nikkei 225 | 09:00 Tokio | 15:25 Tokio | – |
| Hang Seng | 09:30 Hongkong | 15:55 Hongkong | – |
| ASX 200 | 10:00 Sydney | 15:55 Sydney | – |

Die CFDs werden bei Dukascopy oft fast rund um die Uhr gequotet. Der Algo handelt trotzdem nur in der **Kernsession des Heimatmarkts**, weil dort Liquidität und Spreads am besten sind. Die Session-Zeiten sind Definitionen und werden **nicht** pro Instrument nach Ergebnis optimiert (siehe 10.8).

**Handelstag (für Vortageshoch und -tief):** von 17:00 NY bis 17:00 NY (FX-Konvention).

---

## 2. Gemeinsames Ausführungs- und Kostenmodell

Alle Module nutzen dasselbe Modell, damit die Ergebnisse vergleichbar sind.

| Element | Regel |
|---|---|
| Signalpreis | Mid-Kurs |
| Market-Order | Kauf zum **Ask**, Verkauf zum **Bid** des ersten Ticks **nach** Signalzeitpunkt + Latenz |
| Latenz | Default 300 ms; Testwerte 100 / 300 / 1000 ms |
| Stop-Order | ausgelöst, sobald Bid (Long-Stop) bzw. Ask (Short-Stop) den Stopp berührt; Fill zum nächsten Tick + `stop_slippage` |
| Limit-Order | **konservativ:** Kauf-Limit nur gefüllt, wenn der Ask das Limit um mindestens 1 Tick *unterschreitet* (Durchhandeln). Keine Annahme über die Position in der Warteschlange |
| Kommission | Parameter je Mio. gehandeltem Volumen und Seite (aus der Preisliste des Brokers); bei CFDs entsprechend der Preisliste |
| Zusatz-Slippage | Default 0; Stress-Tests mit +0,1 / +0,3 / +0,5 Pip bzw. entsprechend in bp |
| Swap | entfällt (keine Übernacht-Positionen) |
| Mindest- und Schrittgröße | gemäß Broker (z. B. 1.000 Einheiten in FX) |

**Kosten-Report je Modul:** Bruttogewinn, Spread-Kosten, Kommission, Slippage, Nettogewinn, Kosten in % des Bruttogewinns, **Break-even-Zusatzslippage** (ab welcher Zusatzslippage das Modul bei null landet).

---

## 3. Gemeinsame Bausteine

**3.1 Intraday-Sigma je Minute (für M1 und M3):**
```
move[d, m] = | Mid[d, m] / Mid[d, session_start] - 1 |   für Minute m seit Session-Start
sigma[t, m] = Mittelwert über d = t-1 … t-L von move[d, m]          (L = 14, Variante 30/60/90)
```

**3.2 Noise Area (für M1 und M3):**
```
Anker_hoch = max(Mid[t, session_start], Close_Vortag)     # Close_Vortag = Mid zum Handelsende des Vortags
Anker_tief = min(Mid[t, session_start], Close_Vortag)
UB[t, m] = Anker_hoch × (1 + VM × sigma[t, m])
LB[t, m] = Anker_tief × (1 − VM × sigma[t, m])
```

**3.3 Ersatz für VWAP:** Für FX und CFDs gibt es kein echtes Handelsvolumen.
- Default: **TWAP** des Mid-Kurses seit Session-Start.
- Variante: **tickgewichteter Durchschnittspreis** (jede Minute mit der Anzahl ihrer Ticks gewichtet; Ticks sind ein Aktivitätsmaß).

**3.4 Marken (für M2):**
| Marke | Definition |
|---|---|
| PDH / PDL | Hoch und Tief des vorigen Handelstags (17:00–17:00 NY), Mid |
| AH / AL | Hoch und Tief der Asien-Range (00:00–07:00 London) des aktuellen Tages |
| RN | runde Zahlen: FX auf ,xx00 und ,xx50 (JPY-Paare: xx,00 und xx,50); Gold auf 10 $; US500 auf 25 Punkten; DAX auf 100 Punkten |

Marken innerhalb von 0,1 × ATR(14) zueinander werden zu einer Zone zusammengefasst (Clusterbildung verstärkt die Wirkung laut Hypothese → Variable M2-9).

**3.5 Volatilitätsmaße:**
- ATR(14) täglich aus Mid-Kerzen.
- σ_1m: Standardabweichung der 1-Min-Mid-Renditen der letzten 20 Tage in derselben Session-Stunde.

---

## 4. Modul M1: Noise-Area-Momentum

**Idee:** Bewegt sich der Kurs seit Session-Start stärker als für diese Uhrzeit üblich, ist ein Trendtag wahrscheinlich. Grundlage: Zarattini, Aziz & Barbon (2024), dort für SPY und laut FAQ für 33 Futures getestet. **Neu hier:** Anwendung auf FX-Spot, Gold und Index-CFDs mit echten Bid/Ask-Kosten.

**Regeln `M1-v0`:**
```
Prüfzeitpunkte: alle 30 Minuten ab session_start + 30 min, letzte Prüfung Handelsende − 30 min
Long  wenn Mid > max(UB, TWAP)
Short wenn Mid < min(LB, TWAP)
sonst flat (bestehende Position wird geschlossen, wenn ihre Bedingung nicht mehr gilt)
Exit spätestens zum Handelsende
Sizing: exposure = Kapital_M1_Instrument × min(L_max, sigma_target / sigma_daily)
        sigma_daily = Std. der Tagesrenditen (Close-to-Close, 14 Tage); sigma_target = 1 % täglich (Default)
```

**Testvariablen:**
| # | Variable | Default | Testwerte |
|---|---|---|---|
| M1-1 | `lookback_sigma` | 14 | 14, 30, 60, 90 |
| M1-2 | `VM` | 1,0 | 0,75, 1,0, 1,25, 1,5, 2,0 |
| M1-3 | `check_interval` | 30 Min | 15, 30, 60 |
| M1-4 | `trailing_ref` | max(Band, TWAP) | nur Band; Band + tickgewichteter Durchschnitt |
| M1-5 | `gap_anchor` | an | an/aus (Vortages-Close als Anker) |
| M1-6 | `session_start` FX | 08:00 London | 07:00 London; 08:00 London; 08:30 New York |
| M1-7 | `sigma_target` | 1 % | 0,5 / 1 / 1,5 % (nur Risikoskalierung) |
| M1-8 | `L_max` je Instrument | 10× (FX), 5× (Gold, Indizes) | je nach Broker-Limit |
| M1-9 | `instrument_set` | alle | nur FX; nur Gold + Indizes; einzeln |

---

## 5. Modul M2: Absorption vs. Durchbruch

### 5.1 Idee
- Laut Cont, Kukanov & Stoikov (2014) bewegt Orderflow-Druck (OFI) den Kurs **normalerweise linear**: ΔMid ≈ β × OFI.
- **Absorption:** Liegt an einer Marke ungewöhnlich viel passive Liquidität (versteckte Orders, Take-Profit-Cluster laut Osler), reagiert der Kurs dort auf starken Druck **schwächer als erwartet**. Die Marke hält wahrscheinlich → **Gegenposition**.
- **Durchbruch:** Bricht der Kurs mit starkem Druck und normaler oder überdurchschnittlicher Reaktion durch die Marke, werden die dahinter liegenden Stop-Loss-Orders ausgelöst (Osler) → **Mitgehen**.

### 5.2 OFI aus Level-1-Ticks (Formel nach Cont et al.)
Für jeden Tick n mit Bid b_n, Ask a_n, BidVolume q^b_n, AskVolume q^a_n:
```
e_n = 1{b_n ≥ b_(n-1)} · q^b_n  −  1{b_n ≤ b_(n-1)} · q^b_(n-1)
    − 1{a_n ≤ a_(n-1)} · q^a_n  +  1{a_n ≥ a_(n-1)} · q^a_(n-1)
OFI(Fenster) = Σ e_n über alle Ticks im Fenster
```
Positive OFI = Kaufdruck, negative OFI = Verkaufsdruck.

### 5.3 Pflicht-Vortest (entscheidet, ob M2 mit Dukascopy-Daten überhaupt sinnvoll ist)
```
Für jedes Instrument: Buckets à 10 Sekunden
Regression: ΔMid_bucket = α + β · OFI_bucket + ε     (rollierend, je Session-Stunde)
```
- **Erwartung laut Literatur:** deutlicher linearer Zusammenhang (hohes R²).
- **Kill-Kriterium:** R² im Median < 0,15 → Die Dukascopy-Mengen sind als Orderflow-Proxy unbrauchbar, M2 wird **gestrichen**. Die Daten stammen nur aus einem Handelsplatz, die Mengen können geglättet oder begrenzt sein. Deshalb ist dieser Test zwingend.

### 5.4 Signale
```
Zone = Marke ± z_width × σ_1m × sqrt(5)        (Default z_width = 1)
Fenster für Druckmessung: letzte W = 60 Sekunden, solange der Kurs in der Zone ist
OFI_z   = OFI(W) / Std(OFI(W)) der letzten 20 Tage, gleiche Session-Stunde
Erwartet = β_hat · OFI(W)                      (β_hat aus 5.3, nur Vergangenheitsdaten)
Absorptionsquote AQ = 1 − ΔMid(W) / Erwartet  (nur wenn |OFI_z| ≥ 2)
     AQ nahe 0  → normale Reaktion
     AQ nahe 1  → Druck wird vollständig "geschluckt"
     AQ < 0     → überproportionale Reaktion
```

**A) Absorptions-Fade:** Kurs nähert sich der Marke von unten (Widerstand) bzw. von oben (Unterstützung):
```
Bedingung: |OFI_z| ≥ 2 in Richtung Marke  UND  AQ ≥ 0,7  UND  Marke noch nicht um > 1 × σ_1m durchbrochen
Entry:  Limit-Order gegen die Druckrichtung, 1 Tick vor der Marke
Stop:   Marke + k_stop × σ_1m × sqrt(15)  auf der anderen Seite   (Default k_stop = 1,5)
Ziel:   TWAP der Session oder 2R, je nachdem, was zuerst erreicht wird
Zeit-Stop: 60 Minuten
```

**B) Durchbruch:**
```
Bedingung: Kurs schließt eine 1-Min-Kerze ≥ 0,5 × σ_1m jenseits der Marke  UND  |OFI_z| ≥ 2 in Ausbruchsrichtung  UND  AQ ≤ 0,2
Entry:  Market-Order in Ausbruchsrichtung
Stop:   zurück auf der anderen Seite der Marke − 0,5 × σ_1m
Ziel:   3R oder Trailing-Stop (Hoch bzw. Tief der letzten 15 Minuten)
Zeit-Stop: 90 Minuten
```
**Regeln für beide:** Pro Marke und Tag höchstens ein Fade und ein Durchbruch. Nach einem verlorenen Fade ist ein Durchbruch-Signal an derselben Marke erlaubt (natürliche Umkehr der Hypothese).

### 5.5 Pflicht-Event-Studie vor der Handelsregel
- Für alle Zeitpunkte mit |OFI_z| ≥ 2 in einer Zone: mittlerer Kursverlauf der nächsten 1–60 Minuten, **getrennt nach AQ-Quintil**.
- **Erwartung:** hohe AQ → Kurs kehrt um; niedrige AQ → Kurs läuft weiter. Diese „Dosis-Wirkung“ muss sichtbar sein.
- **Placebo:** dieselbe Analyse an zufälligen Kursniveaus, die *keine* Marke sind. An Marken muss der Effekt deutlich stärker sein als an Zufallsniveaus, sonst ist die Marken-Hypothese widerlegt.

### 5.6 Testvariablen
| # | Variable | Default | Testwerte |
|---|---|---|---|
| M2-1 | `ofi_window` | 60 s | 15, 30, 60, 120, 300 s |
| M2-2 | `ofi_z_min` | 2,0 | 1,5, 2,0, 2,5, 3,0 |
| M2-3 | `aq_fade_min` | 0,7 | 0,5, 0,6, 0,7, 0,8 |
| M2-4 | `aq_break_max` | 0,2 | 0, 0,2, 0,4 |
| M2-5 | `level_types` | alle | nur PDH/PDL; nur AH/AL; nur RN; Kombinationen |
| M2-6 | `zone_width` | 1 σ | 0,5, 1, 1,5 |
| M2-7 | `k_stop` | 1,5 | 1,0, 1,5, 2,0, 3,0 |
| M2-8 | `target` | TWAP oder 2R | 1R, 2R, 3R; TWAP; Gegenseite der Asien-Range |
| M2-9 | `cluster_bonus` | aus | nur Zonen mit ≥ 2 zusammenfallenden Marken |
| M2-10 | `session_filter` | London + NY | nur London; nur NY; Überschneidung 13:00–16:00 London |
| M2-11 | `mode` | Fade + Durchbruch | nur Fade; nur Durchbruch |

---

## 6. Modul M3: Mean Reversion innerhalb der Noise Area

**Idee:** Solange der Kurs *innerhalb* der Noise Area bleibt, ist der Tag statistisch eher ein Range-Tag. Kurzfristige Ausschläge vom Session-Durchschnitt kehren dann eher zurück. Die Forschung zu FX-Mean-Reversion nach Kosten ist **negativ** (offener EUR/USD-Backtest: −0,33 R pro Trade). Deshalb zwei strenge Filter: **Nur innerhalb der Noise Area** und **nur bei günstigem Spread**.

**Regeln `M3-v0`:**
```
z = (Mid − TWAP_session) / (σ_1m × sqrt(Minuten seit Session-Start))
Bedingungen (alle):
   LB < Mid < UB                          (innerhalb der Noise Area, also kein M1-Signal)
   |z| ≥ 2,0
   aktueller Spread ≤ Median-Spread dieser Session-Stunde (letzte 20 Tage)
   Regime R erlaubt M3 (Abschnitt 8)
Entry: Limit-Order gegen den Ausschlag zum aktuellen Mid ± 0,5 Spread
Exit:  z kehrt auf 0 zurück (Ziel) | z erreicht 3,5 (Stop) | Kurs verlässt die Noise Area (Stop, M1 übernimmt) | Zeit-Stop 45 Min
Max. 3 Trades je Instrument und Tag
```

**Testvariablen:**
| # | Variable | Default | Testwerte |
|---|---|---|---|
| M3-1 | `z_entry` | 2,0 | 1,5, 2,0, 2,5 |
| M3-2 | `z_exit` | 0 | 0, 0,5 |
| M3-3 | `z_stop` | 3,5 | 3,0, 3,5, 4,5 |
| M3-4 | `spread_filter` | ≤ Median | ≤ 25. Perzentil; ≤ Median; aus (zeigt den Wert des Filters) |
| M3-5 | `noise_area_filter` | an | an/aus (zeigt den Wert der Kopplung an M1) |
| M3-6 | `time_stop` | 45 Min | 15, 30, 45, 90 |
| M3-7 | `session` | London + NY | nur Asien (Range-Hypothese, Spreads breiter!); nur London; nur NY |
| M3-8 | `entry_type` | Limit | Limit; Market (zeigt den Kostenvorteil) |

---

## 7. Modul M4 (optional): Intraday-Währungsstärke

**Idee:** Statt einzelne Paare zu handeln, wird die Stärke jeder der 8 Hauptwährungen (USD, EUR, JPY, GBP, CHF, AUD, CAD, NZD) gemessen. Gehandelt wird **stärkste gegen schwächste** Währung. Das ist automatisch **dollarneutral**, außer der Dollar selbst ist die stärkste oder schwächste Währung. **Status:** Auf Monatsbasis ist Währungs-Momentum gut dokumentiert. Intraday ist das eine **eigene Hypothese**, beide Richtungen (Momentum und Umkehr) werden getestet.

```
Für jede Währung c: Stärke S_c = Mittelwert der vol-normierten Renditen seit Session-Start
    über alle verfügbaren Paare mit c (Vorzeichen so, dass "c steigt" positiv ist)
Prüfzeitpunkte: alle 60 Min ab 09:00 London
Momentum-Variante: Long stärkste / Short schwächste Währung über das direkt gehandelte Paar
Umkehr-Variante:   umgekehrt
Nur handeln, wenn S_max − S_min ≥ Schwelle (Default 2 Standardabweichungen der Differenz)
Exit: nächste Prüfung (Neuberechnung) oder Handelsende
```
| # | Variable | Default | Testwerte |
|---|---|---|---|
| M4-1 | `direction` | Momentum | Momentum; Umkehr |
| M4-2 | `check_interval` | 60 Min | 30, 60, 120 |
| M4-3 | `threshold` | 2 σ | 1, 1,5, 2, 3 σ |
| M4-4 | `pairs_used` | nur Paare, die die Kostengrenze für M4 erfüllen (CR ≤ 0,12, siehe 10.2) | alle 28; nur Majors + EUR/JPY, EUR/GBP |
| M4-5 | `n_legs` | 1 Paar | Top-2 gegen Bottom-2 (bis zu 4 Paare) |

---

## 8. Regime-Steuerung R

**Ziel:** Trend-Modul M1 und Range-Modul M3 je nach Marktphase unterschiedlich gewichten. **Muss sich gegenüber festen Gewichten im Test beweisen** (Variable P-6).

```
Täglich vor Session-Start, nur Daten bis Vortag:
   VIX_pct      = Perzentil des VIX-Schlusskurses über 1 Jahr                  (yfinance)
   RV_pct_i     = Perzentil der 20-Tage-Volatilität von Instrument i über 1 Jahr  (Dukascopy)
   Trendtage_i  = Anteil der letzten 20 Tage, an denen M1 bei Instrument i ein Signal hatte
Regime_i = "Trend" wenn (VIX_pct ≥ 70 ODER RV_pct_i ≥ 70) UND Trendtage_i ≥ 40 %
         = "Range" wenn VIX_pct ≤ 40 UND RV_pct_i ≤ 40
         = "Neutral" sonst
Gewichte:  Trend:   M1 × 1,5, M3 × 0,25
           Range:   M1 × 0,75, M3 × 1,5
           Neutral: M1 × 1,0, M3 × 1,0
M2 und M4: nicht regimegesteuert (Default), Variante: M2-Durchbruch im Trend-Regime × 1,5, M2-Fade im Range-Regime × 1,5
```
| # | Variable | Default | Testwerte |
|---|---|---|---|
| R-1 | `regime_inputs` | VIX + RV + Trendtage | nur VIX; nur RV; nur Trendtage |
| R-2 | Schwellen | 70 / 40 | 60/40; 80/30 |
| R-3 | Gewichte | s. o. | 2,0/0; 1,25/0,5 |
| R-4 | `regime_on` | an | an/aus |

---

## 9. Portfolio- und Risiko-Schicht P

### 9.1 Risikobudget
```
Innerhalb jedes Moduls: hierarchische Risikoverteilung (gleiches Risiko je Cluster, dann je Instrument, siehe 10.7)
Zwischen den Modulen: Basis-Allokation nach inverser Volatilität der Modul-Tagesrenditen (60 Tage), wöchentlich angepasst
Danach Regime-Gewichte (Abschnitt 8) anwenden und auf 100 % normieren
Portfolio-Vola-Ziel: vol_target_annual (Default 15 %) → Skalierungsfaktor auf alle Module, max. 1,5
```
Weil M2 und M4 nicht jeden Tag handeln, wird ihre Volatilität als **Vola pro Trade × Handelsfrequenz** geschätzt.

### 9.2 Dollar- und Währungs-Exposure (wichtig bei FX!)
Alle FX-Majors hängen am US-Dollar. Ohne Begrenzung wären 6 Positionen eine einzige Dollar-Wette.
```
Für jede Währung c: Netto-Exposure E_c = Σ über alle offenen Positionen (Nominal in USD × Vorzeichen für c)
Grenze: |E_USD| ≤ 3 × Kapital,  |E_c| ≤ 2 × Kapital für jede andere Währung
Neue Order, die eine Grenze verletzen würde: Größe reduzieren, bis die Grenze eingehalten ist; sonst verwerfen
Metalle zählen je als eigene "Währung" (XAU, XAG, XPT, Kupfer); Brent und WTI zusammen als Ölrisiko (Grenze 1,5 × Kapital)
Aktienindizes: Netto-Aktienrisiko je Region (USA, Europa, Asien-Pazifik) ≤ 1,5 × Kapital, gesamt ≤ 3 × Kapital
```

### 9.3 Konflikte zwischen Modulen
- Mehrere Module dürfen dasselbe Instrument gleichzeitig handeln. **Ausgeführt wird die Nettoposition** (spart Kosten).
- Gewinn und Verlust werden je Modul **virtuell** auf Basis der eigenen Signale abgerechnet, damit die Modul-Statistik sauber bleibt.
- Da M1 und M3 sich über die Noise-Area-Bedingung gegenseitig ausschließen, gibt es keinen direkten Konflikt. M2 kann M3 widersprechen, das ist erlaubt (Netting).

### 9.4 Verlustgrenzen
| Kontrolle | Default | Aktion |
|---|---|---|
| Tagesverlust Portfolio | −2,5 % | alles schließen, Handel bis zum nächsten Tag stoppen |
| Tagesverlust je Modul | −1,5 % des Modulkapitals | Modul für den Tag stoppen |
| Drawdown-Stufen | −8 % → Risiko × 0,75; −12 % → × 0,5 | volle Größe erst nach neuem Hoch |
| Drawdown-Stopp | −20 % | Handel komplett stoppen, Analyse |
| Spread-Notbremse | Spread > 5 × Median | keine neuen Orders für dieses Instrument |
| Datenausfall | > 30 s kein Tick in aktiver Session | keine neuen Orders, offene Positionen mit Stops absichern |

### 9.5 Modul-Gesundheit (vorab festgelegt, im Backtest mitsimuliert)
| Zustand | Bedingung (rollierend 6 Monate) | Aktion |
|---|---|---|
| gesund | Sharpe ≥ 0,5 | volles Budget |
| schwach | 0 ≤ Sharpe < 0,5 | Budget × 0,5 |
| ausgesetzt | Sharpe < 0 und DD > 2 × erwartet | nur Paper-Mitlauf; Wiederaufnahme nach 3 Monaten Sharpe ≥ 0,5 |

---

## 10. Instrumentübergreifende Kalibrierung

**Ziel:** Eine Logik, die auf vielen Instrumenten mehrmals täglich feuert und über die Anzahl der Instrumente eine hohe Portfolio-Sharpe erreicht, **ohne** jedes Instrument einzeln „passend zu optimieren“.
**Warum das so streng geregelt ist:** Wer für jedes Instrument die Parameter dreht, bis es nach Kosten im Plus ist, findet bei genug Versuchen **immer** profitable Einstellungen, auch ohne echten Edge. Out-of-sample wird daraus typischerweise ein Verlust. Dieser Abschnitt legt fest, was angepasst werden darf und was nicht.

### 10.1 Eine Regel, dimensionslose Parameter
Alle Parameter werden in **relativen Einheiten** angegeben, damit derselbe Wert auf jedem Instrument dasselbe bedeutet:

| Größe | Einheit | Beispiele |
|---|---|---|
| Kursabstände | Vielfache von σ (σ_1m, Noise-Area-σ, ATR) | `z_entry`, `zone_width`, `k_stop`, `VM` |
| Kosten | Vielfache des aktuellen bzw. Median-Spreads; Kostenkennzahl CR | `spread_filter`, Aufnahmegrenze |
| Zeit | Minuten seit Session-Start; Anteil der Session | `check_interval`, `time_stop` |
| Orderflow | z-Werte (OFI_z), Quoten (AQ) | `ofi_z_min`, `aq_fade_min` |
| Positionsgröße | Volatilitäts-Ziel | `sigma_target` |

Die Pipeline prüft beim Start, dass **kein Parameter in absoluten Einheiten** (Pips, Punkte, Dollar) vorliegt. Ausnahme sind reine Definitionen, die nicht optimiert werden (runde Zahlen in 3.4, Session-Zeiten in 1.3).

### 10.2 Kandidaten-Universum und Aufnahme nach Kostenkennzahl

**Kandidaten (alle bei Dukascopy verfügbar, Symbolnamen prüfen):**
| Cluster | Instrumente |
|---|---|
| USD-Majors | EURUSD, GBPUSD, AUDUSD, NZDUSD, USDJPY, USDCAD, USDCHF |
| Crosses | EURJPY, GBPJPY, AUDJPY, CADJPY, EURGBP, EURCHF, EURAUD, AUDNZD |
| Edelmetalle | XAUUSD, XAGUSD, XPTUSD |
| Industriemetall | Kupfer |
| Energie | Brent, WTI |
| US-Indizes | US500, US-Tech-100, US30 |
| Europa-Indizes | DAX, Euro Stoxx 50, FTSE 100, CAC 40, SMI |
| Asien-Pazifik-Indizes | Nikkei 225, Hang Seng, ASX 200 |

**Kostenkennzahl CR (vorab, ohne Blick auf Strategieergebnisse):**
```
CR_i = Median( Round-Trip-Kosten_i ) / Median( |Mid-Bewegung über 30 Minuten|_i )
       Round-Trip-Kosten = Spread beim Einstieg + Spread beim Ausstieg (je halber Spread pro Seite) + Kommission
       beide Mediane nur über die Kernsession aus 1.3, nur aus dem Kalibrierungszeitraum
```
CR sagt, welcher Anteil einer typischen Bewegung durch Kosten verloren geht.

| Modul | Aufnahme, wenn | Begründung |
|---|---|---|
| M1 | CR ≤ 0,15 | wenige, große Bewegungen; verträgt höhere Kosten |
| M2 | CR ≤ 0,10 | mittlere Ziele (1–3R) |
| M3 | CR ≤ 0,08 | häufig, kleine Ziele, sehr kostenempfindlich |
| M4 | CR ≤ 0,12 | stündliche Neugewichtung |

**Zusätzliche Datenkriterien:** mindestens 5 Jahre Historie, Ticks in mindestens 95 % der Session-Minuten.
**Jährliche Neubewertung:** CR wird jedes Jahr nur mit Daten *vor* diesem Jahr neu berechnet. Ein Instrument kann also in späteren Jahren hinzukommen oder herausfallen, aber **nie wegen seiner Performance**.

### 10.3 Globale Kalibrierung auf gepoolten Daten
```
Für jede Parameter-Kombination θ im Raster:
    Für jedes aufgenommene Instrument i: Netto-Tagesrenditen r_i(θ), mit GLEICHEM Risiko je Instrument (Vola-Ziel)
    Gepoolte Kennzahl: Sharpe des gleichgewichteten Korbs (Cluster-gewichtet, siehe 10.7)
Geglättete Zielfunktion: Für jedes θ den Mittelwert der Korb-Sharpe über θ und seine direkten Nachbarn im Raster
θ_global = argmax der geglätteten Zielfunktion        (Plateau statt Spitze)
```
**Zusatzbedingung:** Bei θ_global müssen **mindestens 60 % der Instrumente** in-sample netto positiv sein. Liefert ein einzelnes Cluster mehr als 50 % des Korb-Gewinns, gilt der Edge als nicht breit genug und wird gesondert geprüft.

### 10.4 Instrumentspezifische Anpassung mit Schrumpfung
Pro Modul darf **höchstens ein Parameter** je Instrument abweichen, und zwar nur dieser:

| Modul | anpassbarer Parameter | warum dieser |
|---|---|---|
| M1 | `VM` (Breite der Noise Area) | Instrumente unterscheiden sich darin, wie viel „normales Rauschen“ sie haben |
| M2 | `ofi_z_min` | Orderflow-Daten unterscheiden sich je Instrument in der Qualität |
| M3 | `z_entry` | Mean Reversion ist je Instrument unterschiedlich stark |
| M4 | keiner | Strategie ist von Natur aus instrumentübergreifend |

```
θ_i*    = Plateau-Optimum für Instrument i (wie in 10.3, aber nur auf Instrument i)
w_i     = n_i / (n_i + n_0)          n_i = Anzahl Trades von i im Kalibrierungszeitraum, n_0 = 500 (Default)
θ_i     = w_i × θ_i* + (1 − w_i) × θ_global
Begrenzung: θ_i liegt höchstens ±25 % von θ_global entfernt
```
- **Wenig Trades → nahe am globalen Wert.** Erst bei sehr vielen Trades darf das Instrument wirklich abweichen.
- **Bewährungsprobe:** Die geschrumpften Parameter werden nur übernommen, wenn sie im Walk-Forward (T6) **besser** abschneiden als θ_global auf demselben Instrument. Sonst gilt θ_global.

### 10.5 Quer-Validierung über Instrumente
Prüft, ob die Regel auch auf Instrumenten funktioniert, auf denen sie **nicht** kalibriert wurde.
```
Wiederhole 20-mal:
    Teile die Instrumente zufällig in Gruppe A und B, getrennt je Cluster (jedes Cluster zur Hälfte in A und B)
    Kalibriere θ_global nur auf A (In-Sample-Zeitraum)
    Teste mit diesem θ auf B (gleicher Zeitraum)
Kennzahlen: Sharpe_B / Sharpe_A (Median), Anteil der Läufe mit Sharpe_B > 0
Zusätzlich: "Leave-one-cluster-out" (auf allen Clustern außer einem kalibrieren, auf dem ausgelassenen testen)
```
**Bestanden, wenn:** Median von Sharpe_B / Sharpe_A ≥ 0,5 **und** Sharpe_B > 0 in ≥ 80 % der Läufe **und** kein ausgelassenes Cluster mit deutlich negativer Sharpe (< −0,3).

### 10.6 Zählung aller Versuche und Korrektur
- **Jede** ausgewertete Parameter-Kombination wird protokolliert: Modul, θ, Instrumentmenge, Zeitraum. Daraus ergibt sich die Zahl der Versuche N_T.
- **Deflated Sharpe Ratio** (Bailey & López de Prado) mit N_T, Schiefe, Kurtosis und Anzahl Beobachtungen. Anforderung: Die Wahrscheinlichkeit, dass die echte Sharpe > 0 ist, liegt bei ≥ 95 %.
- **Planungsannahme:** Bis zum bestandenen Holdout gilt **erwartete Live-Sharpe = 0,5 × In-Sample-Sharpe**.

### 10.7 Korrelation, effektive Anzahl, Risikoverteilung
```
ρ̄_m   = mittlere paarweise Korrelation der Instrument-Tagesrenditen in Modul m
N_eff  = N / (1 + (N − 1) × ρ̄_m)                    # "effektive" Anzahl unabhängiger Instrumente
erwartete Modul-Sharpe ≈ Ø Instrument-Sharpe × √N_eff
```
- **Hierarchische Risikoverteilung:** Zuerst gleiches Risiko **je Cluster** (10.2), dann gleiches Risiko je Instrument innerhalb des Clusters. So dominieren die 7 USD-Majors nicht das Portfolio, nur weil es viele sind.
- Die Pipeline berichtet N_eff je Modul und für das Gesamtportfolio. **Faustregel für das Ziel Sharpe 2:** Ø Instrument-Sharpe (out-of-sample) × √N_eff ≈ 2. Zum Beispiel 0,5 × √16 oder 0,7 × √8.

### 10.8 Verboten (automatisch geprüft, wo möglich)
| Verbot | Warum |
|---|---|
| mehr als einen Parameter je Instrument anpassen | Überanpassung |
| Instrumente wegen schlechter Ergebnisse ausschließen | Ausschluss nur über CR und Datenkriterien (10.2) |
| Session-Zeiten, Marken-Definitionen oder Handelsstunden je Instrument nach Ergebnis wählen | versteckte Optimierung |
| θ_global nach Blick auf den Holdout ändern | Holdout wäre verbraucht |
| Instrumente einzeln mit eigenem Vola-Ziel „hochhebeln“, weil sie gut liefen | Performance-Chasing |

---

## 11. Testprotokoll

| Phase | Inhalt | Ergebnis |
|---|---|---|
| **T0 Daten** | Aufbereitung 1.1, Plausibilität (Spreads je Stunde, Tickzahl, Lücken), Stichproben gegen Charts | Daten-Freigabe |
| **T1 Vortests** | M2: OFI-Regression (5.3). M1/M3: Verteilung der Noise-Area-Ausbrüche je Instrument | Kill oder Weiter je Modul |
| **T2 Event-Studien** | M2: AQ-Quintile mit Placebo (5.5). M3: Kursverlauf nach \|z\| ≥ 2 innerhalb gegen außerhalb der Noise Area. M4: Momentum gegen Umkehr | Hypothese bestätigt oder verworfen |
| **T2b Universum** | Kostenkennzahl CR und Datenkriterien je Instrument und Modul (10.2), **vor** allen Strategietests | Liste der aufgenommenen Instrumente |
| **T3 `v0` je Modul** | Default-Regeln mit vollem Kostenmodell auf allen aufgenommenen Instrumenten, In-Sample bis 24 Monate vor heute | Basis-Kennzahlen je Instrument und gepoolt |
| **T4 Globale Kalibrierung** | Einzelvariablen und Raster auf **gepoolten** Daten, geglättete Zielfunktion (10.3) | θ_global je Modul |
| **T4b Quer-Validierung** | 20 zufällige A/B-Teilungen + Leave-one-cluster-out (10.5) | bestanden oder Modul zurück auf T2 |
| **T5 Einfrieren** | θ_global festschreiben; optional geschrumpfte Instrument-Parameter (10.4), höchstens einer je Instrument | finale Modulversionen |
| **T5b Versuchszählung** | Deflated Sharpe mit allen protokollierten Versuchen (10.6) | Wahrscheinlichkeit echte Sharpe > 0 ≥ 95 % |
| **T6 Walk-Forward** | 3 Jahre Kalibrierung → 1 Jahr Test, rollierend (nur zur Bestätigung) | Stabilität |
| **T7 Kombination** | Module + R + P zusammen; Korrelationsmatrix; Effekt von R und P ein/aus | Portfolio-Kennzahlen |
| **T8 Holdout** | letzte 24 Monate, **einmalig** | Go/No-Go |
| **T9 Monte Carlo** | Block-Bootstrap der Portfolio-Tagesrenditen (20-Tage-Blöcke, 5.000 Läufe) | Verteilung von Monatsrendite und MDD |
| **T10 Demo** | Dukascopy-Demokonto (JForex-API), mind. 3 Monate, alle Module live | Abgleich Fill gegen Backtest |
| **T11 Live gestuft** | 25 % → 50 % → 100 % des Ziel-Risikos über 6 Monate | – |

**Aufnahme-Gate je Modul** (allein, nach Kosten, out-of-sample):
- Sharpe ≥ 0,6
- Break-even-Zusatzslippage ≥ 2 × Default-Annahme
- Zufalls-Benchmark klar geschlagen (über dem 95-%-Quantil von 100 Läufen mit zufälliger Richtung)
- in ≥ 70 % der Jahre positiv
- Korrelation zu den bereits aufgenommenen Modulen ≤ 0,5
- Quer-Validierung über Instrumente bestanden (10.5)
- mindestens 60 % der aufgenommenen Instrumente out-of-sample netto positiv

**Portfolio-Gate (Vorgabe 2–4 % pro Monat):**
| Kriterium | Mindestwert |
|---|---|
| Holdout-Netto-Sharpe | ≥ 2,0 |
| Median-Monatsrendite beim gewählten Vola-Ziel | ≥ 2 % |
| Holdout-MDD | ≤ 20 % |
| positive Monate | ≥ 70 % |
| Monte Carlo: P(Monat < −5 %) | ≤ 10 % |
| Monte Carlo: P(MDD > 30 %) | ≤ 5 % |
| Daytrading-Check | 0 Übernacht-Positionen |

**Pflicht-Reports:**
- Je Modul und gesamt: Sharpe, Sortino, CAGR, Monatsrenditen (Verteilung, Anteil ≥ 2 %), MDD (Höhe und Dauer), Trefferquote, Ø Gewinn/Verlust, Profit-Factor, Trades pro Tag, Kostenaufschlüsselung, Break-even-Slippage.
- Ergebnisse je Jahr, Instrument, Session-Stunde und Regime.
- Korrelationsmatrix der Module (gesamt, rollierend, in Stressphasen).
- Netto-Exposure je Währung über die Zeit.
- Je Modul: Liste der aufgenommenen Instrumente mit CR, Ergebnis je Instrument und je Cluster, ρ̄ und N_eff (10.7), Zahl der Versuche N_T und Deflated Sharpe (10.6), Ergebnisse der Quer-Validierung (10.5).

---

## 12. Versions-Fahrplan

| Schritt | Version | Inhalt |
|---|---|---|
| 1 | `DATA-v1` | Dukascopy-Downloader, Bereinigung, 1-Min-Kerzen inkl. Spread-, Tick- und OFI-Spalten; yfinance-Kontext |
| 2 | `EXEC-v1` | gemeinsames Ausführungs- und Kostenmodell (Abschnitt 2) mit Tick-Fills |
| 3 | `UNIV-v1` | Kostenkennzahl CR und Datenkriterien für alle Kandidaten (10.2), Versuchsprotokoll (10.6) einrichten |
| 4 | `M1-v0` | Noise Area auf allen für M1 aufgenommenen Instrumenten, Kosten-Report je Instrument und gepoolt |
| 5 | `M2-PRE` | OFI-Regression (5.3) je Instrument → Kill oder Weiter (auch einzelne Instrumente können hier herausfallen, weil ihre Daten ungeeignet sind) |
| 6 | `M2-ES` | Event-Studie AQ-Quintile + Placebo (5.5), gepoolt |
| 7 | `M3-v0` | Mean Reversion mit beiden Filtern, plus Varianten ohne Filter zum Vergleich |
| 8 | `M2-v0` | nur wenn 5 und 6 bestanden |
| 9 | `M4-ES`, `M4-v0` | optional |
| 10 | `Mx-GLOBAL` | globale Kalibrierung je Modul auf gepoolten Daten (10.3) |
| 11 | `Mx-XVAL` | Quer-Validierung über Instrumente (10.5) |
| 12 | `Mx-SHRINK` | optional: geschrumpfte Instrument-Parameter (10.4), nur wenn im Walk-Forward besser |
| 13 | `PORT-v0` | Kombination mit hierarchischer Risikoverteilung (Cluster → Instrument → Modul), ohne R |
| 14 | `PORT-v1` | + Regime R + Exposure-Grenzen + Verlustgrenzen |
| 15 | `PORT-v1-WF/HOLDOUT/MC` | Validierung, Deflated Sharpe, Portfolio-Gate |
| 16 | `PORT-v1-DEMO` | Dukascopy-Demo, 3 Monate |

**Begründung der Reihenfolge:**
- M1 kommt zuerst. Es hat die stärkste Vorgeschichte (Paper plus Nachbauten), ist einfach zu bauen und liefert gleichzeitig die Noise Area, die M3 braucht.
- M2 hat einen frühen Kill-Test. So verlierst du keine Zeit, falls die Dukascopy-Mengen als Orderflow-Maß nicht taugen.

---

## Quellen (Grundlagen der Module)
- Zarattini, Aziz, Barbon (2024/2025): Beat the Market – An Effective Intraday Momentum Strategy for SPY, SSRN 4824172 (M1)
- Cont, Kukanov, Stoikov (2014): The Price Impact of Order Book Events, *Journal of Financial Econometrics* 12(1) (M2, OFI-Formel)
- Osler (2000): Support for Resistance, *FRBNY Economic Policy Review*; Osler (2001): Currency Orders and Exchange-Rate Dynamics, FRBNY Staff Report 125 (M2, Marken und Order-Cluster)
- Kolm, Turiel, Westray (2021): Deep Order Flow Imbalance, SSRN 3900141 (Horizont von Orderbuch-Signalen)
- Wiśniewska (2014): Mean Reversion im 1-Min-EUR/USD, *Folia Oeconomica Stetinensia* (M3)
- Moskowitz, Ooi, Pedersen (2012): Time Series Momentum, *JFE* (Trend-Evidenz in FX)

*Keine Anlageberatung. Alle Module sind Hypothesen, bis die Pipeline sie getestet hat.*
