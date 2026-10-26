# Eingefrorene Portfolios

Ein Portfolio ist eine **feste** Kombination von Strategie-Varianten mit festem Risiko und
**vorab festgelegten** Bestehenskriterien. Es wird als *eine* Hypothese einmal gegen den
Holdout (ab 2025-01-01) getestet.

| ID | Name | Rolle | Komponenten | Risiko/Trade | Train %/Monat (in-sample) | Train max. DD |
|---|---|---|---|---|---|---|
| **TK6** | Trendkorb 6 | **primär (der Test)** | Momentum US100 + SP500, Supertrend USDJPY / GER40 / SP500, klassisches ORB US100 | 0,25 % | 2,11 % (Median 1,49 %) | 8,0 % |
| US2 | US100-Duo | sekundär (Info) | Momentum US100, klassisches ORB US100 | 0,36 % | 1,39 % (Median 1,02 %) | 7,9 % |
| M1 | Momentum solo | sekundär (Info) | Momentum US100 | 0,35 % | 1,17 % (Median 0,64 %) | 7,9 % |

Jede Komponente enthält: `params` (optimiert), `voll_config` (ALLE Felder der Strategie),
`regeln` (Entry, Stop-Loss, Take-Profit/Ausstieg, Filter, Session, Kosten - in Worten mit den
echten Zahlen, `trading_lab/rules.py`), `kosten_modell` und `herkunft`.

**Verlauf:** Am 2026-10-26 zuerst als TK6/US3 eingefroren. Vor jedem Holdout-Blick fiel auf,
dass Supertrend, BB-Squeeze und klassisches ORB auf dem Bid ohne Spread rechnen; nach der
Kosten-Korrektur erfüllen nur noch 6 Varianten die Auswahlregel -> **TK6 / US2 / M1**
(TK6/US3 gelöscht, nie geöffnet; Ledger leer).

**Einschränkung:** Der Holdout 2025-2026 ist für diese eingefrorenen Varianten ungesehen, aber
frühere Explorations-Studies im Repo (z. B. `studies/momentum_sweep.py`) liefen über die volle
Historie inkl. 2025-2026. Die Strategie-*Familie* Momentum ist dort also schon einmal
angeschaut worden - ein Rest von Vorwissen bleibt.

Dateien: `configs/portfolios/<ID>.yaml` (Parameter, Kriterien, Herkunft, Train-Ergebnis) und
`<ID>.lock` (SHA-256 der YAML **und** aller Quelldateien, von denen das Ergebnis abhängt:
Strategie-Module, Fills/Größe/Kosten). Wird danach Datei oder Code geändert, verweigert der Test den Lauf.

**Kriterien (alle Portfolios):** im Holdout Tages-t >= 1,645, Monatsmittel > 0,
max. Drawdown <= 10 %, schlechtester Tag <= 5 % - beim festen Risiko.

**Bestätigend ist nur TK6.** US2 und M1 sind Teilmengen von TK6; wer nach TK6 noch US2 oder
M1 öffnet, testet nichts Neues mehr (der Ablauf vermerkt das über das Ledger).

```bash
uv run python studies/portfolio_holdout.py --liste
uv run python studies/portfolio_holdout.py TK6 --train   # nur ansehen, Holdout bleibt zu
uv run python studies/portfolio_holdout.py TK6           # der eine Test: 'HOLDOUT' eintippen
```

Bauen/Einfrieren: `studies/portfolio_build.py` (`--neu` nur solange ungeöffnet).
Code: `trading_lab/portfolio.py`, `trading_lab/rules.py`; Tests: `tests/test_portfolio.py`.
Die Train-Zahlen sind in-sample und daher optimistisch - erfahrungsgemäß bleibt außerhalb
die Hälfte oder weniger. Keine Anlageberatung.

## Ergebnis des Holdout-Tests TK6 (2026-10-26, 23:18) - NICHT BESTANDEN

Vorregistriert in Commit `16893b3`. Holdout 2025-01-01 .. 2026-10-26 (21,8 Monate), Risiko 0,25 %/Trade:

| Kennzahl | Train (in-sample) | Holdout |
|---|---|---|
| % pro Monat (Mittel) | 2,11 | **0,20** (9 % des Trainingswerts) |
| % pro Monat (Median) | 1,49 | -0,05 |
| positive Monate | 0,71 | 0,50 |
| max. Drawdown % | 8,0 | **17,7** (Kriterium <= 10: verfehlt) |
| t (Tagesreihe) | - | **0,25** (Kriterium >= 1,645: verfehlt) |

Komponenten im Holdout (Summe R): momentum US100 **-22,0**, momentum SP500 +16,6,
supertrend GER40 +20,5, supertrend USDJPY +3,5, supertrend SP500 +2,3, orb_classic US100 -3,7.

Der Holdout ist für alle sechs Komponenten verbraucht (Ledger). US2 und M1 sind Teilmengen
und würden nichts Unabhängiges mehr testen. Bericht: `artifacts/pruefung/20261026_231806_portfolio_TK6/`.
