# Der Prüfablauf - eine Strategie Schritt für Schritt testen

```bash
uv sync                                   # einmal: legt die .venv an
uv run python studies/pruefen.py          # geführt, fragt alles ab
uv run python studies/pruefen.py --strategie momentum --instrument US100
uv run python studies/pruefen.py --liste  # Strategien, Instrumente, die 12 Schritte
```

**Jeder Durchlauf ist gleich**, egal welche Strategie: dieselben 12 Schritte in derselben
Reihenfolge, dieselben Schwellen. Nach jedem Schritt zeigt der Ablauf, was passiert ist,
das Zwischenergebnis und was es bedeutet, und hält an. Mit `a` kann man jederzeit
abbrechen. Das Protokoll wird trotzdem geschrieben.

| # | Schritt | Was passiert | Entscheidest du |
|---|---|---|---|
| 1 | Umgebung prüfen | Python, .venv; Ziel des Ablaufs erklärt | - |
| 2 | Strategie und Instrument | Auswahl aus `trading_lab/registry.py` | ja |
| 3 | Daten laden und aufteilen | 1-Minuten Bid/Ask aus dem Cache; **Train** bis 2024, **Holdout** ab 2025 (verschlossen) | Download, falls nichts im Cache |
| 4 | Basis-Backtest | Standard-Parameter auf Train: n, brutto/netto R, Jahre | - |
| 5 | Suchraum und Budget | Grid anzeigen, einzelne Parameter fixieren, schnell/normal/gründlich | ja |
| 6 | Optimierung | nur Train; Ziel = untere Konfidenzgrenze des Netto-Erwartungswerts je Trade; jede Variante wird gezählt | - |
| 7 | S1 Brutto + Placebo | brutto > 0, besser als Gegenrichtung (Replay), n >= 100 | - |
| 8 | S2 Kosten | gemessen netto > 0 (echte Bid/Ask-Fills) | - |
| 9 | S3 Robustheit | >= 70 % der Jahre positiv, Nachbar-Einstellungen >= 50 % positiv | - |
| 10 | S4 Signifikanz | Deflated Sharpe >= 0.95 (rechnet den Suchaufwand heraus), Placebo-Vorsprung, PBO <= 0.25 | - |
| 11 | S5 Holdout | nur für S4-Überlebende, nur nach Eintippen von `HOLDOUT`, wird im Ledger vermerkt | ja |
| 12 | Prop-Firm + Bericht | Risiko/Trade für 1,5 %/Monat, Drawdown, schlechtester Tag; `bericht.md` | - |

## Die Regeln dahinter

* **Locker vorne, streng hinten.** Eine Strategie scheitert nur an einem *Ergebnis*, nie an
  einer *Prognose*. Kosten kommen in S2, nicht vor dem ersten Trade.
* **Breit suchen ist erlaubt - es kostet Beweiskraft.** Jede unterschiedliche Variante ist
  ein Versuch; die S4-Hürde steigt mit der *effektiven* Zahl unabhängiger Versuche.
* **R:R ist ein Parameter, kein Ziel.** Optimiert wird der Erwartungswert je Trade nach
  Kosten, abzüglich seiner Unsicherheit.
* **Fehlende Werte**: in S1-S3 läuft eine Variante weiter ("nicht beurteilbar"), in S4/S5
  nicht - ohne Nachweis kein Bestehen.
* **Der Holdout zählt einmal.** Ein zweites Öffnen für dieselbe Strategie x Instrument wird
  angezeigt und im Bericht vermerkt (`artifacts/pruefung/holdout_ledger.json`).

## Ausgabe

`artifacts/pruefung/<datum_zeit>_<strategie>_<instrument>/`:
`schritt_01.json` ... `schritt_12.json` (jedes Zwischenergebnis), `varianten.json`,
`bericht.md` (Tabelle aller Varianten mit erreichter Stufe + Protokoll jeder Entscheidung).

## Eine neue Strategie in den Ablauf bringen

1. Strategie-Modul nach [ADD_STRATEGY.md](ADD_STRATEGY.md) (Config-Dataclass + `run_backtest`).
2. In `trading_lab/registry.py` einen Adapter `_my_idea(asset) -> Spec` schreiben (Grid,
   `prepare`, `run`), in `REGISTRY` und `DESCRIPTIONS` eintragen.
3. `uv run python studies/pruefen.py --strategie my_idea` - fertig, derselbe Ablauf.

Für viele Strategien auf einmal (ohne Rückfragen): `studies/optimize_all.py`, danach
`studies/optimize_report.py`. Gleiche Stufen, gleiche Schwellen, gleicher Code.

Kombinationen fester Varianten (Portfolios) werden als eine Hypothese getestet:
[PORTFOLIOS.md](PORTFOLIOS.md) (`studies/portfolio_holdout.py`).

*Forschungswerkzeug auf historischen Daten - keine Anlageberatung.*
