# AEP Stufe 0 — trägt die Startbibliothek ihr eigenes Kostengate?

Vorabprüfung des Plans `Neue Idee pläne/Adaptives_Edge_Portfolio_Agentenplan.md`,
**bevor** dessen Schritte 3–14 begonnen werden. Reproduzierbar mit

```bash
.venv/bin/python studies/aep_k3_screen.py --csv artifacts/aep_k3.csv
```

## Was geprüft wurde

Der AEP-Plan listet in Abschnitt 10 eine Startbibliothek von „etwa 22
Kandidaten-Einheiten" und macht **K3 zur Pflicht**:

> K3 | S0-Kostengate bestanden | Kosten / typische Bewegung über den Zielhorizont
> ≤ 0,10 für das Cluster

Das braucht **keinen Backtest** — nur Bid/Ask-Daten und die Haltedauer, die die
Regeln in Anhang A vorgeben (Zeit-Stop bzw. Sessionende). Geprüft wurde jede
Einheit aus den Tabellen 10, Anhang A und Teil XI, und zwar **jedes Bein** des
Clusters (der Plan handelt alle Instrumente eines Clusters gleichgewichtet).

Horizonte: aus den Regeln gelesen. Wo eine Position bis zum Sessionende läuft,
sind zwei plausible Einstiegszeitpunkte angesetzt (konservativ der längere).

Datenbasis: Dukascopy 1-Minuten Bid+Ask ab 2019, 21 Instrumente, `cost_gate.horizon_gate`.

## Korrektur (wichtig): das Kostenmodell entscheidet das Ergebnis

Die erste Fassung dieser Prüfung rechnete mit **Market-Einstieg** (beide Beine
zahlen den Spread). Der AEP-Plan beschreibt seine Einstiege aber überwiegend als
**Limit-Orders** (K-01/K-02 „Limit-Order am Schlusskurs", K-03 „Limit-Order vor der
Marke", K-04/K-05/K-06 mit Stop-Orders bzw. Limit). Eine Limit-Order liegt am Bid
(Kauf) bzw. Ask (Verkauf) und zahlt damit **keinen halben Spread**.

| Kostenmodell | bestehen K3 vollständig | ohne bereits widerlegte Karte |
|---|---|---|
| Market-Einstieg (beide Beine zahlen den Spread) | 4 von 26 | **1** |
| **Limit-Einstieg (ein Bein zahlt keinen Spread)** | **15 von 26** | **10** |

Über alle Instrumente und Horizonte hinweg: Market 17 von 84 Zellen, **Limit 45
von 84**. Bei 60 Minuten steigt die Zahl bestehender Instrumente von 2 auf 8, bei
120 Minuten von 6 auf 14.

**Die erste Fassung dieses Berichts war damit zu pessimistisch.** Die Aussage war
„die Startbibliothek trägt nicht" — richtig ist: sie trägt bei Limit-Einstieg
gerade eben (10 Einheiten ohne tote Karte gegen die geforderten 6).

Zu beachten bleibt: der Gate nimmt an, dass die Limit-Order **gefüllt** wird. Zwei
Risiken stecken darin und müssen in der Event-Studie gemessen werden:

- **Fill-Rate** — eine ruhende Order wird nur bei Berührung gefüllt. Die
  Signalzählung muss die nicht gefüllten Signale mitzählen, sonst überschätzt man
  die Frequenz.
- **Adverse Selection** — gefüllt wird vor allem dann, wenn der Kurs durchläuft.
  Genau bei einer Pullback-Strategie ist das der teure Fall. Die Event-Studie muss
  gefüllte gegen nicht gefüllte Signale vergleichen.

## Ergebnis (Market-Einstieg, zur Referenz)

| | |
|---|---|
| Einheiten geprüft | 26 |
| bestehen K3 vollständig | **4** |
| bestehen nur teilweise (einzelne Beine) | 13 |
| bestehen gar nicht | 9 |
| **davon ohne bereits widerlegte Repo-Karte** | **15 Einheiten, davon 1 mit K3 bestanden** |

**Die vier, die bestehen:**

| Einheit | Horizonte | engstes Bein | Repo-Status |
|---|---|---|---|
| K-06 London-Breakout × GBP/EUR | 240/420 min | GBPUSD 0,072 | **H-004 `kill`** |
| K-06 London-Breakout × JPY | 240/420 min | GBPJPY 0,098 | **H-004 `kill`** |
| K-07 Noise-Area × USD-Majors | 390/600 min | NZDUSD 0,099 | **H-001 `kill`** |
| K-08 Gold-COMEX × Gold | 240/390 min | XAUUSD 0,096 | — |

Der Plan selbst verlangt in Abschnitt 24.1 **mindestens 6** Einheiten mit positivem
Netto-Ergebnis, sonst „**Abbruch**". Nach dem eigenen K3-Gate bleiben davon **1**
(K-08 Gold-COMEX, laut Plan „Evidenz: schwach").

## Die Kostenwand schlägt dort zu, wo der Plan seine Priorität 1 setzt

| Einheit | bestes Bein | Faktor über der Grenze |
|---|---|---|
| K-01 F-Asien × AUD/NZD/CAD | 0,486 | 4,9× |
| K-01 F-Asien × EUR-Crosses | 0,329 | 3,3× |
| K-04 Paar-Spread × AUD/NZD/CAD | 0,237 | 2,4× |
| K-15 Index-Regime × US-Indizes | 0,149 | 1,5× |
| K-16 Schlussstunde × US-Indizes | 0,149 | 1,5× |
| K-17 Bitcoin × Krypto | 0,738 | 7,4× |

K-01 und K-02 sind die **Priorität-1-Kandidaten** des Plans und bestehen ihn am
schlechtesten. K-15 bis K-18 stammen aus dem Strukturbruch-Radar (Teil XI) und
bestehen ihr eigenes Aufnahme-Gate nicht. K-17 (Bitcoin) liegt um Faktor 7
darüber — konsistent mit dem, was der Plan in Abschnitt 39 selbst über
Krypto-CFD-Spreads schreibt.

Ein Muster über alle 26 Einheiten: **nur Horizonte ab ~240 Minuten bestehen.**
Unter 120 Minuten besteht praktisch kein Bein außer den liquidesten Majors
(EURUSD, USDJPY). Das deckt sich exakt mit dem Gate-Screen über 21 Instrumente
(`studies/cost_move_screen.py`): bei 15 Min besteht 0 von 21, bei 60 Min 2 von 21,
bei 240 Min 11 von 21.

## Kandidaten, die im Repo schon widerlegt sind

| AEP-Kandidat | Repo-Karte | Status |
|---|---|---|
| K-06 London-Breakout | H-004 Asian-Range-Breakout | `kill` |
| K-07 Noise-Area-Trend | H-001 | `kill` |
| K-11a Fixing London | H-005 London-16:00-Fix (WMR) | `kill` |
| K-11b Fixing Tokio | verwandt mit H-005 | `kill` |
| K-12 Monatsende | verwandt mit H-005 (Melvin & Prins vs. WMR) | `kill` |
| K-16 Schlussstunden-Momentum | H-003 | `kill` |
| K-10 Währungsstärke | verwandt mit H-006 | `kill` |
| K-02 F-Tag / K-01 F-Asien | Mean Reversion (M3) | verworfen |
| K-04 Paar-Spread-Umkehr | Pair-Spreads getestet: gross PF ≈ 1,0 | verworfen |

Die vier Einheiten, die K3 bestehen, sind damit fast vollständig schon getestet.
Nach dem Protokoll (`PATTERN_RESEARCH.md`) ist ein Re-Test mit geänderten Regeln
**kein** Re-Test derselben Hypothese, sondern eine neue Karte — aber es ist
derselbe Mechanismus, und die Evidenz dagegen ist bereits vorhanden.

## Empfehlung (revidiert)

Mit Limit-Einstieg überleben **10 Einheiten ohne tote Karte** und damit mehr als
die vom Plan geforderten 6 — die Bibliothek ist also nicht von vornherein tot,
sie war nur unter dem falschen Kostenmodell geprüft. Der Engpass verschiebt sich
damit vom Kostengate auf die Frage, die es **nicht** beantwortet: **Fill-Rate und
Adverse Selection**. Beides ist in der Event-Studie messbar und muss dort stehen.

Reihenfolge:

1. **Limit-Einstieg als Standard-Kostenmodell** in `cost_gate` und in den Karten.
   Ein Gate, das den Market-Fall prüft, verwirft Limit-Strategien zu Unrecht.
2. **Fill-Rate und Adverse Selection als Pflichtkennzahlen** in der Event-Studie:
   Anteil gefüllter Signale, mittlere Rendite gefüllt vs. nicht gefüllt.
3. Dann die 10 überlebenden Einheiten als Karten testen — nicht als Portfolio.

Der ursprüngliche Befund bleibt für den Market-Fall richtig (4 von 26) und für
die Priorität-1-Kandidaten K-01/K-04/K-17, die auch mit Limit nicht bestehen.

Wenn die AEP-Idee weiterverfolgt werden soll, dann in dieser Reihenfolge:

1. **Zuerst Bibliothek aufbauen, nicht verwalten.** Die Portfolio-Logik
   (Bewertung, Schrumpfung, Kovarianz, CUSUM, Meta-Raster) löst ein Problem, das
   erst ab ~6 lebenden Einheiten existiert.
2. **Der Horizont muss ≥ 240 Minuten sein** (Gate-Befund), sonst ist der
   Kandidat vor dem ersten Backtest tot. Der Plan setzt seine Priorität 1 bei
   M5-Strategien mit 60-Minuten-Stop — dort ist nichts zu holen.
3. **K-08 (Gold-COMEX) und K-09 (Kompression) sind die einzigen nicht
   vorgetesteten Kandidaten mit Gate-Potenzial** — aber K-09 besteht nur
   teilweise (JPY-Beine) und K-08 hat laut Plan schwache Evidenz. Beide sollten
   als normale Hypothesen-Karten nach dem bestehenden Protokoll laufen, nicht als
   Teil eines Portfolios.
4. **Die vier widerlegten Mechanismus-Familien** (Noise-Area, Asian-Range,
   Fixing, Schlussstunde) nicht unter neuem Namen erneut testen.
