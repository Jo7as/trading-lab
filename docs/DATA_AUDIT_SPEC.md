# Daten-Audit (Schritt 1, Voraussetzung für jede Event-Studie)

**Ziel:** Bevor irgendeine Hypothese getestet wird, müssen die Daten nachweislich sauber sein. Bekannter offener Defekt: **1.644 Ausreißer-Bars, schlimmster 104 × Median.** Solange dieser Defekt besteht, sind alle Event-Studien nicht zitierfähig.

**Ergebnis des Audits:** ein Audit-Report pro Instrument plus eine **Freigabe-Flagge** je Instrument (`audit_passed: true/false`). Event-Studien und Backtests dürfen nur auf freigegebenen Instrumenten laufen. Der Hypothesen-Validator prüft das.

---

## A1 Ausreißer-Bars

**Definition** (je Instrument, auf 1-Min-Mid-Kerzen):
```
range_t      = High_t − Low_t
med_range_h  = Median(range) derselben Session-Stunde über die letzten 60 Handelstage (rollierend, nur Vergangenheit)
ratio_t      = range_t / med_range_h
Ausreißer-Kandidat, wenn ratio_t ≥ 20   ODER   |Mid_close_t − Mid_close_(t−1)| ≥ 15 × σ_1m(h)
```

**Klassifikation jedes Kandidaten (Reihenfolge der Prüfung):**
| Klasse | Erkennung | Behandlung |
|---|---|---|
| **Tick-Fehler (Spike)** | Extremwert stammt aus ≤ 3 Ticks und der Kurs kehrt innerhalb von ≤ 5 Ticks / 10 s auf das alte Niveau zurück; oder Ask ≤ Bid | Ticks entfernen, Bar aus den restlichen Ticks neu bauen |
| **Spread-Explosion** | Bid/Ask-Spread im Bar > 20 × Median-Spread der Stunde; der Mid-Kurs bleibt plausibel | Bar markieren (`spread_event`), für Signale nutzbar, aber **Ausführung im Simulator zu echten Bid/Ask** (Kosten werden sichtbar) |
| **Session-Rand / Rollover** | Bar liegt in den ersten bzw. letzten 5 Min der Instrument-Session oder um 17:00 New York | markieren (`session_edge`), von Signalen ausschließen |
| **Echtes Marktereignis** | Bewegung wird von nachfolgenden Bars bestätigt (kein Rücksprung), Tickzahl im Bar ≥ 3 × Median, gleiche Bewegung in korrelierten Instrumenten (z. B. US500 und US-Tech-100) | **behalten** und markieren (`news_like`) |
| **Unklar** | nichts davon trifft eindeutig zu | markieren (`unclear`), aus Event-Studien ausschließen, im Report einzeln auflisten |

**Wichtig:** Echte Marktereignisse werden **nicht** gelöscht. Sonst würde man genau die Tage entfernen, an denen Trendstrategien verdienen bzw. Stops reißen, und der Backtest würde zu gut.

**Report A1:** Anzahl Kandidaten je Klasse, die 50 extremsten Bars mit Tick-Ansicht (Plot), Anteil der Bars je Instrument und Jahr. **Zielwert nach Bereinigung: keine Bars mehr mit `ratio_t ≥ 20` in den Klassen „Tick-Fehler“ oder „unklar“ ohne Markierung.**

## A2 Session- und Tagesgrenzen

| Prüfung | Regel | Fehler, wenn |
|---|---|---|
| Session-Definition | Kernsession je Instrument aus der Session-Tabelle (Multi-Plan 1.3), in Ortszeit des Heimatmarkts | Bars außerhalb der Session werden als Session-Bars gezählt |
| Handelstag FX | 17:00 bis 17:00 New York | Vortageshoch und -tief über falsche Grenzen berechnet |
| Feiertage | Börsenkalender je Heimatmarkt (US, DE, UK, JP, HK, AU, CH, FR) | Handel an Feiertagen mit dünnen Quotes |
| Halbtage | verkürzte Sessions (z. B. US nach Thanksgiving) | Session-Ende falsch |
| Wochenende | Ticks zwischen Fr 17:00 und So 17:00 New York | vorhanden |
| Lücken | Minuten ohne Tick innerhalb der Kernsession | > 5 % der Session-Minuten an einem Tag → Tag markieren (`thin_day`) |

## A3 Sommerzeit (DST)

- Alle Session-Regeln in **Ortszeit** formulieren und je Datum in UTC umrechnen (Zeitzonen-Datenbank, z. B. `zoneinfo`).
- **Testfälle (Pflicht):** die Wochen zwischen dem US- und dem EU-Sommerzeitwechsel (März und Oktober/November) jedes Jahres. In diesen Wochen verschiebt sich der Abstand London–New York um eine Stunde.
- **Prüfung:** Der Session-Start jedes Index-CFDs fällt in jedem Jahr auf die lokale Eröffnungsminute. Prüfung über den typischen Volatilitätssprung zur Eröffnung: Die Minute mit der höchsten mittleren Tickzahl muss die lokale Eröffnungsminute sein.

## A4 Spread je Uhrzeit

- Für jedes Instrument: Median-, 75.- und 95.-Perzentil-Spread je 15-Min-Fenster des Tages, je Jahr.
- Diese Tabelle ist **Eingabe für das Kostenmodell und das Effekt/Kosten-Gate (S0)**. Kosten werden je Uhrzeit, nicht als Tagesmittel, angesetzt.
- **Plausibilität:** Spread-Spitzen müssen an Session-Rändern, beim Rollover 17:00 New York und an bekannten News-Zeitpunkten liegen. Spitzen mitten in ruhigen Phasen deuten auf Datenfehler.

## A5 Konsistenz-Checks

| Check | Fehler, wenn |
|---|---|
| Ask > Bid | in irgendeinem Tick verletzt (nach A1) |
| OHLC-Logik | Low ≤ Open, Close ≤ High verletzt |
| Monotone Zeitstempel | Zeitstempel springen zurück |
| Duplikate | gleicher Zeitstempel und gleiche Kurse mehrfach |
| Cross-Check Tageskerzen | Tages-Close aus Dukascopy weicht vom Tages-Close aus yfinance (gleiches Underlying, z. B. US500 gegen `^GSPC`) im Median um > 0,3 % ab → Session- oder Zeitzonenfehler (Basis-Differenz CFD vs. Index beachten) |

## A6 Freigabe

Ein Instrument bekommt `audit_passed: true`, wenn:
- A1: keine unmarkierten Ausreißer mit `ratio ≥ 20` mehr vorhanden
- A2: < 1 % der Handelstage als `thin_day` markiert (sonst diese Tage ausschließen und dokumentieren)
- A3: DST-Prüfung in allen Jahren bestanden
- A4: Spread-Tabelle erzeugt, ohne unerklärte Spitzen
- A5: alle Konsistenz-Checks bestanden

Der Audit-Report wird versioniert (Daten-Hash). **Jede spätere Event-Studie nennt die Audit-Version, auf der sie läuft.**
