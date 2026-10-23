# PATTERN_RESEARCH: messbare Marktzustände statt Muster

**Grundsatz:** Wir suchen keine Formen im Chart. Wir definieren **messbare Marktzustände**, formulieren vorab eine prüfbare Behauptung über die Vorwärtsrendite nach diesem Zustand und verwerfen sie, wenn sie die Kosten nicht schlägt oder auf unberührten Daten nicht hält.

> „Observe. Question. Define.“ Der Engpass ist das **Question**, nicht das **Observe**.

Eine Hypothese hat immer diese Form:
> **Zustand X** (messbar) **an Ort Y** (Abstand zu einem Level in σ- bzw. Spread-Einheiten) **zur Zeit Z** (Session-Phase) → **Verteilung der Vorwärtsrendite über Horizont H**, verglichen mit einem **Placebo**, mit **Dosis-Wirkung** und **Kill-Kriterien**, **vor** dem Test festgelegt.

---

## 1. Lehren aus bisherigen Fehlern (warum dieses Dokument existiert)

| Fehler | Was passiert ist | Gegenmaßnahme in diesem Protokoll |
|---|---|---|
| Bewegung ≤ Spread | Muster existierte, aber die Bewegung pro Trade war nicht größer als die Kosten | **S0-Gate (Effekt/Kosten) vor jedem Code** |
| Bug-Artefakt | Das „Muster“ war ein Daten- oder Code-Fehler | **Daten-Audit zuerst** (DATA_AUDIT_SPEC), Pflicht-Tests (Look-ahead, Determinismus) |
| Öffentlich bekannt und abgegolten | Der Effekt war real, aber bereits weg | **Post-Publication-Split** und Holdout auf jüngsten Daten |
| Multiple Testing (ORB-„Edge“) | Viele Varianten durchsucht, eine sah gut aus | **Trial-Budget, Versuchsregister, Deflated Sharpe, Vorregistrierung** |

**Bekannter Befund des Kostengates:** Auf 5-Min-Horizont fällt praktisch jedes Instrument durch. Auf 60-Min-Horizont bestehen die Index-CFDs. **Daraus folgt:** Neue Hypothesen haben einen Zielhorizont **H ≥ 60 Minuten**, außer eine S0-Rechnung zeigt vorab, dass ein kürzerer Horizont auf einem bestimmten Instrument die Kosten schlagen kann.

---

## 2. Katalog messbarer Zustände

Alle Größen sind **dimensionslos** (σ-, Spread-, Perzentil- oder Session-Einheiten) und werden **nur aus Daten bis zum Entscheidungszeitpunkt** berechnet.

| # | Zustandsfamilie | Messgröße(n) | Einheit | Mögliche Mechanismen |
|---|---|---|---|---|
| Z1 | **Volatilitätsregime** | Efficiency Ratio (20 Tage), ATR-Perzentil (1 Jahr), realisierte Vola des Vortags in Perzentil | Perzentil | Trendtage häufen sich bei hoher Vola (Absicherung mit negativem Gamma, Stop-Kaskaden) |
| Z2 | **Abstand zu Levels** | (Kurs − Level) / σ_H; Levels: Vortageshoch/-tief, Vortagesschluss, Session-Eröffnung, runde Zahlen | σ_H, Spread | Order-Cluster an bekannten Marken (Osler) |
| Z3 | **Intraday-Fortschritt** | Rendite seit Session-Start / σ der gleichen Uhrzeit (Noise-Area-Logik) | σ | ungewöhnliches Ungleichgewicht zwischen Angebot und Nachfrage → Trendtag (Zarattini et al.) |
| Z4 | **Tageszeit-Saisonalität** | Session-Phase (erste Stunde, Mittag, letzte Stunde), je Instrument | Session-Minuten | Liquiditätszyklen, Absicherungsflüsse vor Schluss (Baltussen et al.) |
| Z5 | **Gap / Overnight** | (Session-Eröffnung − Vortagesschluss) / σ_Tag; Overnight-Rendite gegen Intraday-Rendite der Vortage | σ_Tag | Tug of War zwischen Overnight- und Intraday-Investoren (Lou, Polk & Skouras) |
| Z6 | **Kompression → Expansion** | Spanne der ersten Stunde / Median-Spanne erste Stunde (60 Tage); Vortages-Spanne in Perzentil | Verhältnis, Perzentil | aufgestaute Orders, Volatilitäts-Clustering |
| Z7 | **Orderflow-Proxy** | Level-1-OFI aus Dukascopy-Ticks, z-normiert; Absorptionsquote (Kursreaktion / erwartete Reaktion) | z, Quote | Kursreaktion auf Orderflow linear zur Markttiefe (Cont et al.) |
| Z8 | **Cross-Asset-Divergenz** | Rendite Instrument A minus β × Rendite Instrument B seit Session-Start, in σ (z. B. DAX gegen Euro Stoxx, US500 gegen US-Tech-100, Gold gegen USD) | σ | Arbitrage zwischen eng verbundenen Märkten, verzögerte Informationsverarbeitung |
| Z9 | **Carry / Roll** | Zinsdifferenz (FX), Roll-Kosten (CFD-Finanzierung) | bp pro Tag | nur als Kosten- bzw. Kontextgröße, da intraday kein Swap anfällt |
| Z10 | **Autokorrelation auf dem Zielhorizont** | Korrelation der Rendite über H mit der vorangehenden Rendite über H, rollierend | Korrelation | Momentum vs. Mean Reversion je Instrument und Horizont |

**Kombinationen:** Eine Hypothese darf **höchstens zwei Zustandsfamilien** kombinieren (z. B. Z3 + Z1). Mehr Kombinationen vervielfachen die Zahl impliziter Tests.

---

## 3. Protokoll

### 3.1 Vorregistrierung (vor jedem Blick auf Ergebnisse)
- Jede Hypothese wird als **Hypothesen-Karte** (YAML, siehe `hypotheses/TEMPLATE.yaml`) angelegt.
- Der **Validator** (`tools/validate_hypothesis.py`) lehnt unvollständige Karten ab. Ohne gültige Karte darf keine Event-Studie laufen.
- Die Karte wird **vor** der Event-Studie committet. Der Commit-Zeitstempel ist der Nachweis der Vorregistrierung.
- **Maximal 3 offene Hypothesen gleichzeitig.** Eine neue kommt erst dazu, wenn eine abgeschlossen (bestätigt oder verworfen) ist.

### 3.2 S0-Gate: Effekt/Kosten (vor jedem Code)
```
Kosten_RT(Instrument, Uhrzeit) = Spread_Einstieg + Spread_Ausstieg (je Median der 15-Min-Fenster, DATA_AUDIT A4) + Kommission
Typische Bewegung(H)           = Median |Rendite über H| im Zielzeitfenster, Train-Daten
S0-Quote                       = Kosten_RT / Typische Bewegung(H)
```
- Die Karte enthält die **vorab berechnete S0-Quote** und den **minimalen Effekt**, ab dem die Hypothese wirtschaftlich relevant ist: `min_effekt = 2 × Kosten_RT`.
- **Gate:** S0-Quote ≤ 0,10 (Default; strengere Grenze für Hypothesen mit vielen Trades). Fällt die Hypothese durch, wird sie **nicht** getestet.

### 3.3 Event-Studie (statt Backtest als erster Test)
- Ereignis = Zustand X an Ort Y zur Zeit Z ist erfüllt (nicht überlappend: höchstens ein Ereignis je Instrument innerhalb von H).
- Gemessen wird die Verteilung der Mid-Vorwärtsrendite über H in σ-Einheiten **und** netto nach Kosten_RT.
- **Placebo:** gleiche Uhrzeit und gleiches Volatilitätsregime, aber Zustand X nicht erfüllt (bzw. an Zufallsniveaus statt Levels). Mindestens 20 Placebo-Ereignisse je echtem Ereignis.
- **Dosis-Wirkung:** Ergebnis je Quintil der Zustandsstärke (z. B. Abstand in σ). Der Effekt muss **monoton** mit der Stärke wachsen (Spearman-Rang-Korrelation der Quintil-Mittelwerte ≥ 0,8).
- **Statistik:** t-Wert mit Newey-West-Korrektur oder Block-Bootstrap über Tage (Ereignisse am selben Tag sind nicht unabhängig).

### 3.4 Kill-Kriterien (Default, in der Karte verschärfbar, nie lockerbar)
| Kriterium | Kill, wenn |
|---|---|
| Signifikanz | \|t\| < 3,0 |
| Placebo | Effekt echt − Effekt Placebo < 50 % des echten Effekts |
| Dosis-Wirkung | nicht monoton (Rangkorrelation < 0,8) |
| Wirtschaftlichkeit | mittlerer Netto-Effekt pro Ereignis < 0 oder Brutto-Effekt < `min_effekt` |
| Stabilität | Vorzeichen des Effekts in < 70 % der Jahre gleich |
| Breite | Effekt nur in einem einzigen Instrument (bei Hypothesen über mehrere Instrumente) |

**Kill heißt tot.** Kein Nachbessern derselben Karte. Eine neue Karte mit geänderter Idee zählt als neue Hypothese gegen das Trial-Budget.

### 3.5 Trial-Budget und Versuchsregister
- **Budget je Forschungsrunde:** 3 Hypothesen × höchstens 10 vorregistrierte Varianten (z. B. Schwellen-Raster) = **30 Versuche**.
- Jede ausgewertete Variante wird im **Versuchsregister** gezählt (nur anfügbar).
- Die Deflated Sharpe bzw. die Signifikanzschwelle wird mit der **Gesamtzahl der Versuche aller Runden** berechnet, nicht nur der aktuellen.
- **Explorative Scans** (z. B. „alle Zustände × alle Instrumente durchsuchen“) sind erlaubt, aber:
  1. Sie laufen nur auf einem **separaten Explorationsabschnitt** der Train-Daten.
  2. Jeder gescannte Zustand zählt als Versuch.
  3. Ein Fund aus dem Scan ist **keine Hypothese**, sondern nur eine Beobachtung (S1). Er braucht eine eigene Karte und muss auf **anderen, unberührten** Daten bestätigt werden.

### 3.6 Datenabschnitte
| Abschnitt | Zweck | Zugriff |
|---|---|---|
| Exploration (z. B. 2012–2015) | Scans, Beobachtungen | frei |
| Train (2016–2019) | Event-Studien der vorregistrierten Karten | nach gültiger Karte |
| Validate (2020 bis 24 Monate vor heute) | Bestätigung der überlebenden Karten, erst dann Handelsregel | nach bestandener Event-Studie |
| Holdout (letzte 24 Monate) | finaler Check, einmalig | nur Orchestrator |

---

## 4. Was wir nicht tun

| Nicht tun | Warum |
|---|---|
| Formen-Scanning (Flaggen, Keile, Kopf-Schulter, Dreiecke, Kerzenmuster) | hochdimensional, diskretionär, beliebig nachjustierbar → stärkster Generator für Zufallsfunde |
| Diskretionäre Strukturen (SMC, Order Blocks, Fair Value Gaps, Liquidity Sweeps als Bilderkennung) | nicht eindeutig definierbar; wenn eine Idee darin steckt, muss sie als messbarer Zustand aus Abschnitt 2 formuliert werden |
| Per-Instrument-Tuning | findet bei genug Versuchen immer Gewinner (siehe Multi-Plan Abschnitt 10) |
| Hypothesen mit H < 60 Min ohne vorab bestandenes S0-Gate | Kostengate zeigt: 5-Min-Horizont scheitert praktisch überall |
| Karten nach dem Test ändern | macht die Vorregistrierung wertlos |
| Daten ohne `audit_passed` verwenden | Ausreißer erzeugen Schein-Effekte |
| Optimieren auf Rendite, Trefferquote oder Profit-Factor | nur Netto-Sharpe bzw. Netto-Effekt in σ zählt |
| Ergebnisse ohne Placebo berichten | ohne Vergleich ist ein Effekt nicht interpretierbar |

---

## 5. Ablauf einer Forschungsrunde

```
1. Beobachtung (Exploration oder Literatur)           → Notiz, kein Test
2. Hypothesen-Karte schreiben                          → Validator muss grün sein
3. S0-Gate berechnen und in die Karte eintragen       → Kill, wenn S0 > Grenze
4. Karte committen (Vorregistrierung)
5. Event-Studie auf Train mit Placebo und Dosis-Wirkung → Kill-Kriterien prüfen
6. Überlebt: Bestätigung auf Validate (gleiche Definition, keine Änderung)
7. Überlebt: einfache Handelsregel ableiten (Strategie-Spezifikation, Agent-Plan S3 ff.)
8. Ergebnis (auch Kill) im Karten-Status und im Versuchsregister festhalten
```
