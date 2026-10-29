# Paper-Notizen (vollständig gelesen, mit Konsequenzen für das Labor)

Jeder Eintrag: Daten, genaue Definitionen, Ergebnisse mit Zahlen, Grenzen, was es für uns heißt.
Texte lokal aus den PDFs extrahiert (pdftotext); Zitate kurz gehalten.

## Osler (2001/2003): Currency Orders and Exchange-Rate Dynamics - NY Fed SR 125 (J. of Finance 2003)

* **Daten:** 9.667 Stop-Loss- und Take-Profit-Orders einer großen FX-Bank, 1.9.1999-11.4.2000;
  43 % USD/JPY, 33 % EUR/USD, 24 % GBP/USD; Ø 5,8 Mio. $, Median 3,0 Mio. $; ~92 neue Orders/Tag;
  Median-Laufzeit 0,4 Tage; ~1/3 ausgeführt. Bedingte Orders = 5-10 % des Handelsvolumens der Bank.
* **Definitionen:** Take-Profit = kaufen (verkaufen), wenn der Kurs auf ein Level FÄLLT (STEIGT);
  Stop-Loss = kaufen (verkaufen), wenn er STEIGT (FÄLLT). Nur Richtung + Lage zum Marktkurs zählt.
* **Clustering:** stärkstes Cluster bei Endziffern 00 (~8,7 % aller Orders, statt 1 % bei Gleichverteilung),
  dann 50, dann übrige 0, dann 5; 25/75 nicht besonders. Anderson-Darling: 65,8 / 137,6 / 63,5 (krit. 8,0).
* **Unterschiede (Tabelle I):** Take-Profit genau bei 00: 9,3 % vs. Stop-Loss 4,4 % (p 1,1 %).
  Stop-Loss-**Käufe**: 14,4 % bei Endziffern 01-10 (knapp ÜBER 00) vs. 7,4 % bei 90-99 (p 0,029).
  Stop-Loss-**Verkäufe** spiegelbildlich knapp unter 00 (p 0,064). Um 50 herum: Stop-Käufe 17,3 % bei
  40-49 vs. 6,3 % bei 51-60, Stop-Verkäufe 16,9 % bei 51-60 (p 0,002 / 0,004). Take-Profit: keine Asymmetrie.
* **Mechanismus:** am runden Level überwiegen Take-Profit-Orders -> Kurs prallt eher ab (Vorhersage 1);
  knapp dahinter überwiegen Stop-Orders -> nach dem Durchbruch beschleunigt sich der Kurs (Vorhersage 2).
* **Größe in ECHTEN Kursen (zitiert aus Osler 2001, Minutendaten DEM/JPY/GBP, NY-Handelszeit
  1996-1998):** Abprall-Häufigkeit an runden Zahlen +4,6 Prozentpunkte gegenüber beliebigen Levels;
  Bewegung 30 Minuten nach einem Durchbruch nur 0,0014 % größer als an beliebigen Levels (0,0063 %)
  - also **~0,14 bp zusätzliche Bewegung**. Simulationen (Abschnitt III) sind bewusst übertrieben.
* **Grenzen:** eine Bank, 7 Monate, 1999-2000; das Paper sagt ausdrücklich nichts über Profitabilität
  ("no attempt is made to evaluate whether ... risk-adjusted profits").
* **Für uns:** bestätigt unseren Schritt C: der Effekt ist real, aber Bruchteile eines bp bzw.
  ein paar Prozentpunkte Trefferquote - weit unter 2-3 bp CFD-Kosten. Wichtiges Detail für eine eigene
  Range-Bestimmung: Take-Profit sitzt GENAU am Level, Stops 1-10 Pips DAHINTER - ein Range-Fade müsste
  am Level einsteigen und den Stop jenseits der Stop-Zone setzen (>10 Pips), ein Breakout erst nach
  dem Durchlaufen der Stop-Zone handeln.

## Osler (2002/2005): Stop-Loss Orders and Price Cascades in Currency Markets - NY Fed SR 150 (JIMF 2005)

* **Daten:** Reuters-Minutenquotes DEM, JPY, GBP gegen USD, **nur New-York-Zeit 9-16 Uhr**, Jan 1996 -
  Apr 1998; Orderbuch der Royal Bank of Scotland (9.655 Orders, > 55 Mrd. $; 42,6 % Stop-Loss).
* **Methode:** "Erreichen" = Kurs kommt auf 0,01 % an ein Level heran; runde Zahlen = Endziffern 00 und 50;
  "Durchbruch" = 15 Minuten später jenseits des Levels, sonst "Umkehr". Vergleich mit 10.000 Sätzen von je
  30 zufälligen Levels, in 58 Blöcken à 10 Handelstagen (Vorzeichentest, Binomial).
* **Ergebnisse:**
  - Fortsetzung nach Durchbruch, 15 Min.: DEM **0,061 %** gegenüber 0,054 % an zufälligen Levels
    (**+0,7 bp**), in 51 von 58 Blöcken größer (p < 0,001 %). USD/JPY: Durchbruch minus Umkehr +0,013 %.
  - Umkehr-Häufigkeit an runden Zahlen DEM **59,3 % vs. 54,8 %** (+4,5 Prozentpunkte); GBP nicht signifikant.
  - **Dauer:** Umkehr-Effekt (Take-Profit) < 30 Minuten signifikant; Fortsetzung (Stop-Loss) **mindestens
    2 Stunden**, beim Yen teils 1 Tag, nicht 2 Tage.
  - Stärker am **NY-Nachmittag** (weniger Liquidität). Kein Unterschied Kauf/Verkauf.
  - Sehr große Stop-Orders (> 50 Mio. $): 62 % ihres Volumens bei Endziffern 90-99/01-09 (Take-Profit: 28 %).
  - Kurs kreuzt runde Zahlen ~5-mal pro Tag; große Kaskaden laut Händlern höchstens ~1-mal pro Woche.
* **Grenzen:** Kausalität nicht bewiesen (Notenbank-Eingriffe und Chaos als Alternativen geprüft und
  verworfen); nur Quotes, keine Transaktionen; 1996-1998.
* **Für uns:** deckt sich mit Schritt C: Fortsetzung real (+0,7 bp nach 15 Min.), Umkehr nur Trefferquote.
  Neu und nutzbar: (1) die Wirkung hält bei Stops **Stunden**, bei Take-Profit nur Minuten - eine
  Fortsetzungs-Idee braucht Haltedauern von Stunden (dort sind die Kosten tragbar), eine Range-Fade-Idee
  hätte nur Minuten (Kosten zu hoch). (2) Stärker bei **wenig Liquidität** (NY-Nachmittag) - Filter-Idee.
  (3) Große Kaskaden sind selten (~1/Woche) - ein Filter auf "großer Bruch" (z. B. Volumen/Range-Expansion)
  könnte die seltenen, großen Fälle von den 5 täglichen Kleinst-Fällen trennen.

## Tengelin & Sopasakis (2020): Tick based clustering ... support and resistance levels - National Accounting Review 2(4)

* **Daten:** EUR/USD Tickdaten (Zeit in ms, Bid, Ask), Preis = Mitte; Test Jan-Mai 2019 (5 Monate).
* **Methode:** 1-dimensionales Clustering des PREISES, gewichtet mit Tick-Aktivität ("Masse" =
  Anzahl Ticks am Preis): (a) modifizierter K-means/Lloyd, (b) Gaussian-Mixture (EM). Anzahl Cluster
  über Silhouette, Calinski-Harabasz, Davies-Bouldin, ELBO bzw. Shapiro-Wilk/KS gewählt (5-8 Cluster
  aus 1 Stunde Daten). Clustergrenzen = Unterstützung/Widerstand. Tick-Aktivität korreliert mit der
  Preisvarianz (Pearson 0,75, Spearman 0,83, ein Tag im Jan 2019).
* **Test (Scheintrades):** Fenster ~12 h; Kurs trifft ein Level: "Ziel" wenn er 10 Pips abprallt,
  "Stop" wenn er 10 Pips durchläuft; sonst null. **GMM: 249 Ziel / 211 Stop / 28 null (51,0 % / 43,2 %),
  p = 0,042 gesamt**, je Monat meist nicht signifikant (Jan 0,014, Mrz 0,943). K-means: 48,8 % / 45,8 %,
  p = 0,276 - Zufall. **Keine Kosten berücksichtigt** (ausdrücklich).
* **Grenzen:** 5 Monate, ein Markt, symmetrisches 10/10-Pip-Ziel, keine Kosten, keine Kontroll-Levels.
* **Für uns:** schwacher Beleg. Rechnerisch (249-211)/488 x 10 Pips ≈ **+0,8 Pip pro Trade vor Kosten**
  - bei ~1 Pip Kosten bleibt nichts. Nützlich ist die IDEE der Range-Bestimmung: Levels aus der
  **Verteilung von Zeit bzw. Aktivität am Preis** (Volume-/Time-at-Price-Profil, Cluster), nicht aus
  Linien. Das ist ein Kandidat für unsere eigene Range-Bestimmung - mit Kontroll-Levels und Kosten testen.

## Krohn, Mueller, Whelan (2021/2024): Foreign Exchange Fixings and Returns Around the Clock - BoC SWP 2021-48 (J. of Finance 2024)

* **Daten:** G9-Währungen gegen USD (AUD, CAD, EUR, GBP, JPY, NZD, NOK, SEK, CHF; ~75 % des Spot-Umsatzes),
  Intraday-Mittelkurse Jan 1999 - Dez 2019; Orderflow 2006-2019 von Refinitiv FX Matching (Interbank) und
  CME-Futures.
* **Befund:** der USD steigt vor den drei großen Fixings und fällt danach - V-Form um **Tokio 9:55 Ortszeit,
  EZB 14:15 Frankfurt, London 16:00** (W-Form über 24 h). Dollar-Portfolio (gleichgewichtet): vor Tokio
  -5,0 % p.a., danach +5,3 %; vor EZB -4,2 %, ab London-Fix bis NY-Schluss +4,6 % (Vorzeichen aus Sicht
  "Fremdwährung long"); t 4,5-11,7. Täglich **~2,5 bp** Aufwertung vor, gleich viel Abwertung nach dem Fix.
  Robust über 21 Jahre, alle Währungen (Yen fällt nach London weiter), auch Forwards/Futures.
* **Mechanismus:** Händler sichern die vorhersehbare Dollar-Nachfrage am Fix vorab ab ("pre-fix hedging",
  in Kundenverträgen offengelegt); Lagerrisiko der wenigen großen Liquiditätsgeber. Interbank-Orderflow
  erklärt die Umkehr (1 Std.-Abw. Orderflow vor dem Fix -> 2,5-3,5 bp Umkehr danach); CME-Orderflow nicht.
  Umkehr-Renditen höher bei hoher Volatilität (VIX sagt sie voraus).
* **Kosten (Abschnitt V.E):** Strategie (USD long 2:00-8:15 ET bis EZB-Fix, USD short 11:00-17:00 ET nach
  London-Fix; bzw. Tokio 17:00-20:55 / 20:55-2:00 ET) ohne Kosten 8-12 % p.a.; **mit vollen Spreads
  negativ**; mit 50 % der Spreads EUR 2,65 % / GBP 0,51 % / JPY 2,73 % (Tokio), Europa EUR 8,64 % /
  GBP 5,46 %; Sharpe 0,12-1,03. Tokio-Strategie seit ~2013 netto negativ (EUR, GBP).
* **Für uns:** passt zu unserer Messung (FX-Fixing-Umkehr 0,7-1 bp im 30-Min.-Fenster). Neu ist die
  **lange Variante**: die Umkehr läuft STUNDEN (London-Fix 11:00 ET bis NY-Schluss 17:00 ET, EZB-Fenster
  2:00-8:15 ET). Das sind ~2,5 bp pro Tag bei 1,7-2,5 bp Round-Trip für uns - höchstens grenzwertig,
  am ehesten EUR/USD und GBP/USD im Europa-Fenster und in Phasen hoher Volatilität. Kurzmessung lohnt.

## Moskowitz, Ooi, Pedersen (2012): Time Series Momentum - JFE 104, 228-250 (Grundlage von S-007)

* **Daten:** 58 liquide Futures/Forwards (24 Rohstoffe, 12 Währungspaare aus 9 Währungen, 9 Aktienindizes,
  13 Staatsanleihen), 1965-2009, Hauptergebnisse ab 1985. Futures-Renditen = Überschussrenditen.
* **Regel (TSMOM):** Vorzeichen der **12-Monats-Überschussrendite**, Haltedauer 1 Monat, monatlich neu.
  Größe je Instrument = **40 % / ex-ante-Vola** -> jedes Instrument gleich viel Risiko; Portfolio =
  gleichgewichteter Durchschnitt aller Instrumente (~12 % Vola p.a.). Ex-ante-Vola: EWMA der quadrierten
  TÄGLICHEN Renditen (um den EWMA-Mittelwert), Schwerpunkt 60 Tage, x 261, **Stand t-1** (kein Look-ahead).
  Margin-Bedarf 5-20 %.
* **Ergebnisse:** 12-Monats-TSMOM positiv für **jedes** der 58 Instrumente; Fortsetzung über 1-12 Monate,
  teilweise Umkehr über 1-5 Jahre. Diversifiziertes Portfolio: **Sharpe > 1 (brutto)**, ~2,5x Aktienmarkt;
  Alpha ~1,58 %/Monat gegenüber MSCI World + Fama-French; kein Beta zu Markt/SMB/HML, positiv zu UMD.
  Am besten in extremen Marktphasen (groß rauf und runter). Spekulanten (CFTC) verdienen auf Kosten der
  Absicherer. Korrelationen von TSMOM ÜBER Anlageklassen sind höher als die der Klassen selbst.
* **Grenzen:** brutto (keine Kosten im Hauptergebnis); 1985-2009 - die schwächeren 2010er fehlen.
* **Abgleich mit S-007:** gleiche Signal-Idee und gleiche Vola-Schätzung (EWMA, Schwerpunkt 60, Stand t-1).
  Unterschiede (alle in der Spec so gewollt): (1) wir gewichten **je Anlageklasse** gleich statt je Instrument
  (Paper: 13 Anleihen + 24 Rohstoffe dominieren; wir: 7 USD-Paare zählen nicht 7-fach); (2) wöchentliche
  Umschichtung mit 25 %-Puffer statt monatlich; (3) Signal auf Kursrendite (CFD/Spot) statt Überschussrendite
  - bei FX fehlt damit der Zinsanteil im Signal; (4) 26 statt 58 Märkte, kaum Rohstoffe/Anleihen.
  Unsere Brutto-Sharpe 0,79 (2004-2024) vs. > 1 (1985-2009) passt zur bekannten Schwäche nach 2009 und zum
  kleineren Universum.

## Hurst, Ooi, Pedersen (2017): A Century of Evidence on Trend-Following Investing - J. Portfolio Mgmt 44(1)

* **Daten:** 67 Märkte (29 Rohstoffe, 11 Aktienindizes, 15 Anleihen, 12 Währungspaare), Monatsrenditen
  **1880-2016**; Futures wo vorhanden (Roll des liquidesten Kontrakts), sonst Kassa-Renditen finanziert zum
  lokalen Geldmarktzins; CBOT-Berichte 1877-1951 von Hand abgeschrieben (doppelt kontrolliert).
* **Regel:** gleichgewichtete Kombination aus **1-, 3- und 12-Monats-TSMOM** (Vorzeichen der Überschussrendite),
  jede Position auf gleiche Vola, Gesamtportfolio auf **10 % ex-ante-Vola**, monatlich. 3 Jahre Daten für
  die Vola-Schätzung. Kosten aus AQR-Schätzungen 2012, für 1993-2002 doppelt, für 1880-1992 sechsfach;
  zusätzlich 2/20-Gebühren simuliert; Roll-Kosten NICHT enthalten.
* **Ergebnisse (Zahlen stehen in den Grafiken, im Text):** positiv in **jedem Jahrzehnt** seit 1880, auch
  netto; jede der drei Signallängen in jedem Jahrzehnt positiv; mit einem Monat Verzögerung schwächer,
  v. a. die kurzen Signale. Ø Sharpe je Einzelmarkt ~0,4. Positiv in 8 der 10 schlimmsten 60/40-Drawdowns
  ("Smile"). Eigene Drawdowns bis ~25 %, lang. 20 % Beimischung zu 60/40 senkt Drawdown und Vola und hebt
  die Rendite.
* **Wann gut/schlecht:** kaum Unterschied nach Rezession/Boom, Inflation, Krieg, Zinsniveau, Vola-Änderung.
  **Einziger klarer Zusammenhang: Korrelation zwischen den Märkten** - niedrige Korrelation = bessere
  Rendite. Von Ende 2008 bis Mitte 2014 hohe "Risk-on/Risk-off"-Korrelation -> schwache Phase.
* **Fußnote wichtig:** Trendfolge auf KASSA-Preisen ist nicht umsetzbar, wenn Futures den Trend über die
  Roll-Rendite vorwegnehmen - realistisch ist nur die Rechnung mit Finanzierung/Roll.
* **Für S-007:** (1) unsere Finanzierungs-Modellierung (Swaps) ist genau die richtige Korrektur zu dieser
  Fußnote; (2) unsere schwachen Jahre 2011-2018 passen zur dokumentierten Hoch-Korrelationsphase;
  (3) offene Frage: im Paper hilft die 1/3/12-Mischung, bei uns war sie viel schwächer (netto Sharpe 0,15) -
  wahrscheinlich Kosten/Umschlag der kurzen Signale bei wöchentlicher Umschichtung auf CFDs (Paper: monatlich,
  institutionelle Kosten). Nicht nachträglich ändern; ggf. eigene Spec mit monatlicher Umschichtung.
