# Nischen-Ideen für Daytrading (Weg 3)

Stand: 2026-10-13 · **Alles eigene, ungeprüfte Hypothesen.** Zu keiner dieser Ideen kenne ich eine saubere Studie. Genau deshalb könnten sie noch funktionieren, und genau deshalb ist der Test hier strenger als beim ORB.

**Was eine gute Nische ausmacht:**
1. Ein Grund, warum sich dort ein Muster bildet (feste Termine, wenig Teilnehmer, Pflichthandel).
2. Für große Fonds zu klein oder zu unbequem (kurzes Zeitfenster, wenig Volumen, Randmarkt).
3. Bei einer Prop Firm als CFD handelbar und mit Kosten, die in R klein bleiben.

## A. Kleinere Indizes zu ihrer Eröffnung
Viele ORB- und Momentum-Studien gibt es nur für US-Indizes. Andere Indizes werden von weniger Systematikern bearbeitet.

| # | Idee | Markt | Uhrzeit (deutsch) | Begründung |
|---|---|---|---|---|
| A1 | ORB / Eröffnungsmomentum | **JP225** (Nikkei) | 01:00–03:00 (Sommerzeit 02:00) | Tokio-Eröffnung, starke Reaktion auf US-Vortag und Yen. Wenige europäische Händler um diese Zeit. Für einen Bot egal. |
| A2 | Eröffnung nach der Mittagspause | JP225 | 05:30 (Tokio 12:30) | Tokio hat eine Mittagspause. Was in der Pause passiert (China-Daten, Yen), wird zur Wiedereröffnung eingepreist. |
| A3 | ORB / Gap Fill | **UK100** | 09:00 (London 08:00) | Eine Stunde vor dem DAX, andere Teilnehmer (Rohstoffe, Banken). Kaum mit US-Eröffnung verbunden. |
| A4 | ORB | **AUS200** | 01:00 / 00:00 | Rohstofflastig, eigene Nachrichtenlage. Nur wenn der Spread der Prop Firm passt. |
| A5 | ORB | **HK50** | 03:30 | Stark von China-Nachrichten getrieben, sehr eigene Dynamik. Spreads oft hoch, Kosten zuerst prüfen. |

Vorteil für dich: Diese Märkte öffnen zu Zeiten, an denen US100/GER40 ruhen. Sie füllen also Lücken im Tag und hängen an anderen Nachrichten.

## B. Feste Termine mit Spezialisten-Flüssen
| # | Idee | Markt | Termin | Begründung |
|---|---|---|---|---|
| B1 | Reaktion auf **US-Öllagerdaten** (EIA) | WTI/Brent-CFD | Mittwoch 16:30 | Wöchentlich, feste Uhrzeit, Raffinerien und Händler müssen sich anpassen. Hypothese: Die erste Bewegung nach der Zahl setzt sich 30–60 min fort, oder sie wird gefadet. Beides testen. |
| B2 | Reaktion auf **US-Gaslagerdaten** | NatGas-CFD | Donnerstag 16:30 | Wie B1, noch nischiger. Spreads prüfen. |
| B3 | **Eurex-Verfall** (3. Freitag, DAX-Abrechnung um 13:00 per Intraday-Auktion) | GER40 | 3. Freitag, 12:30–14:00 | Optionsschreiber sichern bis zur Abrechnung ab, danach fällt die Absicherung weg. Hypothese: Vor 13:00 klebt der DAX an runden Strikes, danach größere Bewegung. |
| B4 | **DAX-Index-Anpassung** | GER40 | Quartalsverfall, Schlussauktion 17:30 | Indexfonds müssen zur Schlussauktion handeln. Hypothese: Bewegung in der letzten Stunde vor der Auktion. |
| B5 | **Ifo / ZEW** | GER40, EURUSD | 10:00 / 11:00 | Deutsche Daten, die US-Systematiker ignorieren. Hypothese: Die erste Reaktion wird innerhalb einer Stunde teilweise zurückgenommen. |
| B6 | **US-Feiertage, Europa offen** | GER40, UK100 | ~9 Tage im Jahr | Ohne US-Impuls kaum Volumen. Hypothese: Mean Reversion um VWAP funktioniert an diesen Tagen besser, Ausbrüche schlechter. Als Schalter zwischen Strategien. |

## C. Tageszeit-Eigenheiten
| # | Idee | Markt | Uhrzeit (deutsch) | Begründung |
|---|---|---|---|---|
| C1 | **Mittags-Umkehr US** | US500 / US100 | 18:00–19:30 | Mittagsflaute in New York, wenig Volumen. Hypothese: Übertreibungen vom Vormittag laufen zurück Richtung VWAP. Gegenpol zu ORB und Noise Area. |
| C2 | **DAX nach Xetra-Schluss** | GER40 | 17:30–22:00 | Kassamarkt zu, CFD läuft weiter und folgt nur noch den USA. Hypothese: Der DAX-CFD hinkt US-Bewegungen nach 17:30 hinterher. |
| C3 | **Asien-Range bei Indizes** | US500 / GER40 | 01:00–08:00 | Nachts handeln fast nur Algorithmen bei geringer Schwankung. Hypothese: Range-Handel (Fade an Asien-Hoch/-Tief) funktioniert, solange keine Nachricht kommt. Kosten nachts sind oft höher. |

## D. Was **nicht** geht: Eigenheiten von CFD-Feeds ausnutzen
Fehlerhafte Kurse, verzögerte Feeds, Spread-Sprünge zum Tageswechsel oder Lücken zwischen CFD und Future auszunutzen (Latenz-Arbitrage) ist bei FTMO und fast allen Prop Firms **ausdrücklich verboten** und führt zur Kontoschließung. Das kommt nicht in Frage, auch wenn ein Backtest es zeigt. Wichtig für den Bot: Er darf nicht unbemerkt eine solche Kante „finden“. Deshalb jede Strategie auf Futures-Daten gegenprüfen. Funktioniert sie nur auf dem CFD-Feed, fliegt sie raus.

## Zwei Stufen: erst messen, dann handeln
**Stufe 1, Muster messen (ohne Trades):** Für jedes Ereignis (z. B. jeden Mittwoch 16:30) die Kursbewegung in festen Fenstern messen, etwa 0–5 min und 5–60 min danach. Prüfen, ob die zweite Bewegung häufiger in Richtung der ersten läuft (Fortsetzung) oder dagegen (Umkehr), wie oft und wie groß. Gegenprobe mit derselben Messung zu zufälligen Uhrzeiten. Weiter nur, wenn das Muster deutlich ist (z. B. ≥ 58 % statt 50 %) und in mehreren Jahren auftritt.

**Stufe 2, Regel handeln:** Aus dem Muster eine feste Regel bauen (Einstieg, Stop, Ziel, Ausstieg) und den Testplan unten durchlaufen. Erst wenn die Regel besser als Zufall **und** nach Kosten im Plus ist, ist sie eine Strategie.

## Testplan für Nischen (strenger als beim ORB)
Weil es keine Studien gibt, ist das Risiko groß, ein Zufallsmuster zu finden.

1. **Hypothese vorher aufschreiben:** Richtung, Uhrzeit, Ein- und Ausstieg, *bevor* die Daten angeschaut werden. Pro Idee höchstens 3 Varianten.
2. **Datensplit:** 2016–2021 Entwicklung, 2022–2024 Prüfung (einmal), 2025–heute Bestätigung.
3. **Strengere Zufallsschwelle:** Wer 20 Ideen testet, findet eine per Zufall mit 95 %. Deshalb **besser als 99 % der Zufallsläufe** verlangen.
4. **Kosten pro Trade ≤ 0,04 R** mit den echten Spreads der Prop Firm *zu dieser Uhrzeit* (nachts und zu Nachrichten sind sie breiter).
5. **Gegenprobe auf Futures-Daten** (siehe D).
6. **Mindestens 200 Trades** im Testzeitraum, sonst ist das Ergebnis nicht aussagekräftig. Ideen mit 12 Trades im Jahr (B3, B4) nur als Filter, nie allein.
7. Erst danach Schattenbetrieb wie beim ORB.

## Meine Reihenfolge
1. **C1 Mittags-Umkehr US**: Gleiche Daten wie ORB/Noise Area, Gegenpol, viele Trades.
2. **A1/A3 JP225 und UK100 zur Eröffnung**: Gleiche Testmaschine wie der ORB, nur andere Uhrzeiten. Verteilt das Risiko über den ganzen Tag.
3. **B1 Öllagerdaten**: Feste Uhrzeit, klarer Pflichthandel, 50 Trades im Jahr.
4. Rest als Filter oder später.

**Datenbedarf zusätzlich:** 1-min-Bid/Ask für JP225, UK100, AUS200, HK50, WTI (Dukascopy hat die meisten), Termine für EIA, Eurex-Verfall, Ifo/ZEW, US-Feiertage.
