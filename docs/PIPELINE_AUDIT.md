# Audit der Strategie-Prüflogik — und die neue Reihenfolge

Erstellt 2026-10-26 nach der Rückmeldung, dass Kandidaten **an Prognosen statt an
Ergebnissen** gestorben sind. Das Audit betrifft die Prüflogik selbst, nicht die
Strategien.

## Der Kernfehler

Das Repo prüfte wie eine **Mauer**: alle Kriterien gleichzeitig, und die
Kostenschätzung **vor** jedem Test.

```python
# hypothesis_card.py, heute
if _ist_zahl(grenze) and quote > grenze:
    f.append(f"S0-Gate nicht bestanden: s0_quote {quote} > grenze {grenze}"
             f" → Hypothese nicht testen")
```

Eine **Vorhersage** verhinderte also den Test. Und diese Vorhersage war obendrein
auf den falschen Größen gebaut:

| Die Quote nutzt | Richtig wäre |
|---|---|
| `typische_bewegung_bp` = **mittlere absolute** Bewegung | die Bewegung **in Trendrichtung, bis zum Ziel** (First-Passage/MFE) |
| **Market**-Kosten auf beiden Beinen | die tatsächlichen Fills der Strategie (Limit-Einstieg → ein halber Spread) |
| einen **Zeitpunkt** (kein Fenster) | die Verteilung über die Haltedauer |

## Die zehn Befunde

| # | Befund | Ort | Wirkung |
|---|---|---|---|
| F1 | S0-Quote **blockiert die Registrierung** | `hypothesis_card.py:167` | Prognose tötet vor dem ersten Trade |
| F2 | S0 auf **Mittelwert-Bewegung** | `hypothesis_card.py` | systematisch zu pessimistisch |
| F3 | S0 mit **Market-Kosten** | Karten | Limit-Strategien doppelt bestraft |
| F4 | **`MAX_VARIANTEN = 10`** | `hypothesis_card.py:46` | Exploration auf 10 Varianten gekappt |
| F5 | Alle Gates **gleichzeitig (AND)** | `acceptance.py` | kein Trichter, kein Stufenfortschritt |
| F6 | `t_min = 3.0` auf **Trade-Ebene** | `hypothesis_card.py:37` | bei `sd(R)=4,5` sind ~20.000 Trades nötig → unerreichbar |
| F7 | **Keine Power-/Varianz-Prüfung** | überall | Tests unterpowert, Ergebnis nicht unterscheidbar von Rauschen |
| F8 | Kill-Kriterien **nur verschärfbar** | `hypothesis_card.py:175` | ein falsch gesetzter Wert ist nicht korrigierbar |
| F9 | `iterative_backtest.py` optimiert **während** des Backtests | `trading_lab/` | Overfitting-Maschine |
| F10 | Evolutions-Promotions **ohne Validierung** | Worker | laut eigener Metadaten „unvalidated" |

F6 und F7 sind die teuersten: Sie haben dazu geführt, dass Kandidaten **wegen
fehlender Teststärke** verworfen wurden, nicht wegen fehlender Kante.

## Die neue Reihenfolge (`trading_lab/screening.py`)

```
S1 Exploration   brutto > 0  UND  besser als Placebo        hunderte–tausende Varianten
S2 Kosten        netto > 0 bei realen Kosten               dutzende
S3 Robustheit    Vorzeichen in >= 70 % der Jahre,
                 Nachbar-Varianten positiv                 ~10
S4 Signifikanz   t >= 3 auf der TAGES-Reihe,
                 Placebo-Vorsprung, PBO <= 0.25            3–5
S5 Out-of-Sample Holdout, einmalig                         1–2
```

Regeln, die ab jetzt gelten:

1. **Kein Kriterium darf eine Stufe überspringen.** Kosten sind Stufe 2, nicht
   Stufe 0. Signifikanz ist Stufe 4, nicht Stufe 1.
2. **Nichts wird still verworfen.** Jede Stufe berichtet, was sie entfernt hat;
   `Funnel.summary()` warnt, wenn eine Stufe gar nichts cuttet.
3. **Erst Brutto, dann Netto.** Die Kosten werden auf ein *Ergebnis* angewendet,
   nicht auf eine *Erwartung*.
4. **Signifikanz auf der Tagesreihe.** Die Trade-Reihe ist für kleine Edges
   unbrauchbar (`sd(R)` bestimmt die Nachweisgrenze).
5. **Varianten-Budget statt Varianten-Deckel.** Exploration darf breit sein; die
   Disziplin kommt über das Versuchsregister und die Stufen, nicht über ein Limit
   von 10.

## Was am Werkzeug noch zu ändern ist

| Datei | Änderung | Stand |
|---|---|---|
| `hypothesis_card.py` | `S0`-Quote von *Fehler* auf *Warnung* umstellen; `MAX_VARIANTEN` durch ein Stufen-Budget ersetzen | erledigt (A1/A2) |
| `hypothesis_card.py` | Kill-Kriterien je **Stufe** statt global (S4 streng, S1 nicht) | erledigt (A3) |
| `cost_gate.py` | `typische_bewegung` → First-Passage-Variante als zweites Feld `p_reach` | erledigt (Feld `p_reach`, limit-bewusstes `verdict`) |
| `acceptance.py` + `significance.py` | Signifikanz auf der **Tages**-Reihe (`t_daily`), per-Trade nur noch Fallback ohne Uhr | erledigt |
| `iterative_backtest.py` | als Overfitting-Antipattern kennzeichnen (oder entfernen) | **offen** |

## Was zuerst läuft

Die Bestandsaufnahme der schon vorhandenen Kandidaten durch denselben Trichter
(S1 ohne Kosten), damit sichtbar wird, welche der früheren „Tode" durch **Power**
zustande kamen und welche durch **fehlende Kante**. Erst danach neue Strategien.

---

# Umsetzungsstand (2026-10-26)

## Was geaendert ist

| Schritt | Datei | Aenderung |
|---|---|---|
| A6 | `studies/research_smc_funnel.py` | `run_funnel` -> `count_entry_gates` (Namenskollision mit dem Pruef-Trichter aufgeloest) |
| A1 | `trading_lab/hypothesis_card.py` | S0-Quote ist **Warnung** statt Fehler: eine Prognose darf den Test nicht verhindern |
| A2 | `trading_lab/hypothesis_card.py` | `MAX_VARIANTEN = 10` -> `VARIANTEN_BUDGET` je Stufe (S1/S2 unbegrenzt) |
| A3 | `trading_lab/hypothesis_card.py` | `KILL_DEFAULTS` -> `KILL_JE_STUFE` (S1 locker, S4 streng) |
| A4 | `trading_lab/hypothesis_card.py` | Lockerung erlaubt mit `abweichung_begruendung` (vorher: nur verschaerfbar) |
| A5 | `trading_lab/acceptance.py` | ist jetzt explizit **Stufe S4**, meldet `t_daily` neben dem Trade-t |
| B1 | `trading_lab/screening.py` | `daily_series`, `t_stat`, `pbo` ergaenzt |
| B1 | `trading_lab/screening.py` | `Stage.skipped`: drei Ausgaenge statt zwei - fehlende Werte laufen weiter |
| B3 | `trading_lab/screening.py` | Holdout gesperrt ohne `holdout_freigabe`; Default endet bei S4 |
| B2 | `studies/run_screening.py` | Runner: Tabelle -> Stufen -> Bericht |
| C1 | `docs/hypotheses/H-*.yaml` | Feld `skript` (Zuordnung) und `stufe` ergaenzt |
| C2 | `studies/cards_to_variants.py` | Inventar der 9 Karten im Varianten-Format, Luecken als `null` |
| D1/D2 | `tests/test_screening.py` | 24 Tests inkl. Meta-Test: echter Edge ueberlebt, Rauschen stirbt |

## Die wichtigste neue Regel

**Drei Ausgaenge statt zwei.** Jede Stufe liefert `kept` / `dropped` / `skipped`:

* `dropped` = an dieser Stufe gescheitert, und das ist ein Befund (zu kleines t,
  Vorzeichen kippt ueber die Jahre, ...).
* `skipped` = nicht beurteilbar, weil ein Wert fehlt. **Diese Zeilen laufen weiter.**

Vorher gab es nur bestanden/durchgefallen, also hat ein fehlender Wert eine Karte
entfernt. Genau daran sind die alten Karten gestorben.

## Zuordnung Karte -> Study (verifiziert)

| Karte | Study | Stand |
|---|---|---|
| H-001 | `studies/run_h001.py` | laeuft, braucht `--json` |
| H-002 | `studies/hypotheses_60m.py` (H3 Gap-Fill) | laeuft, braucht `--json` |
| H-003 | `studies/hypotheses_60m.py` (H2 Last-Hour) | laeuft, braucht `--json` |
| H-004 | `studies/multimarket_check.py` | zu verifizieren |
| H-005 | **UNGEKLAERT** | Study fehlt |
| H-006 | **UNGEKLAERT** | Study fehlt |
| H-007 | `trading_lab/st_evolution_worker.py` | kein Trade-Log |
| H-008 | `studies/carry_final.py` | Portfolio, monatlich |
| H-009 | `studies/carry_real_swaps.py` | Portfolio, monatlich |

Inventar: `studies/cards_to_variants.py` -> 7 von 9 reproduzierbar, H-005/H-006 offen.

## Was noch fehlt

1. **`--json` je Study** (C2): `n`, `pos_years`, `t_daily` stehen in keiner Karte
   strukturiert. Ohne sie kann der Trichter S1/S3/S4 nicht beurteilen und
   **ueberspringt** sie - sichtbar im Runner-Bericht, aber ohne Aussage.
2. **H-005/H-006**: Study-Zuordnung klaeren.
3. **H-007**: der Evolutionsworker braucht `--dump-trades`.
4. **`iterative_backtest.py`**: optimiert *waehrend* des Backtests; als Antipattern
   kennzeichnen oder entfernen.

## Pruefbare Abnahmekriterien

```bash
.venv/bin/python -m pytest -q                                  # 284 Tests
.venv/bin/python studies/run_screening.py artifacts/cards_variants.json --up-to S3
.venv/bin/python -m trading_lab.hypothesis_card docs/hypotheses/H-001.yaml
```

Erwartet: eine Karte mit schlechter S0-Quote ist **gueltig** (Warnung), eine mit
mehr als 10 Varianten ist **gueltig**, fehlende Stufen-Kriterien warnen statt zu
blockieren, und der Runner zeigt fuer die alten Karten "nicht beurteilbar" statt
sie zu verwerfen.

---

# Nachtrag: die Verdrahtung (2026-10-26)

Der Trichter existierte, aber die **ausfuehrende** Leitung entschied weiter auf der
per-Trade-Reihe. Das ist jetzt angeglichen:

| Was | Vorher | Jetzt |
|---|---|---|
| `significance.promotion_gate` | per-Trade Sign-Flip, `p<=0.05` | `assess_trades`: **Tages-t >= 3** bei vorhandener Uhr; per-Trade nur Fallback ohne Uhr |
| `acceptance.run_acceptance` | dieselbe per-Trade-Pruefung, `t_daily` nur gedruckt | `assess_trades` — dieselbe Entscheidung wie Stufe S4 |
| `cost_gate.verdict()` | immer Taker-Ratio (beide Beine Market) | `verdict(limit=True)` liest `ratio_limit`; `p_reach` = Anteil der Fenster, deren eigene Bewegung den Spread deckt |
| Studies -> Trichter | keine strukturierten Zahlen (C2 offen) | `screening.funnel_row_from_trades(...)` + `--json` in `run_h001.py` und `hypotheses_60m.py` |

Neue Umgebungsvariable: `LAB_SIGNIFICANCE_T_DAILY` (Default `3.0`).

`studies/pipeline_power.py` misst jetzt dieselbe Schranke (Tages-t >= 3) auf
Kanten mit bekannter Sharpe — statt der per-Trade-Permutation.

**Weiterhin offen:** `--json` fuer `multimarket_check.py`, `carry_final.py`,
`carry_real_swaps.py` (deren Trade-Logs sind bereits **netto** — eine Brutto-Zeile
fuer S1 wuerde in S2 doppelt kosten, deshalb bewusst nicht verdrahtet);
`iterative_backtest.py` als Antipattern kennzeichnen; H-005/H-006 Study-Zuordnung.

---

# Review 2026-10-26 (zweite Runde): Trichter nachgebessert, Optimierer gebaut

Ausloeser: Rueckmeldung "vom Strategie-Tester zum Strategie-Zerstoerer". Bilanz der
neun Karten vorher: nur **H-005/H-006** sind wirklich an einer Prognose gestorben
(nie getestet); H-002/H-004/H-007 waren gemessen negativ bzw. schlechter als Zufall,
H-001 scheiterte am Placebo (71 % des Effekts), H-009 war Look-ahead.

## Befunde im neuen Trichter und was geaendert wurde

| # | Befund | Fix |
|---|---|---|
| R1 | Behauptung "Tages-Aggregation erhoeht die Teststaerke" ist **falsch**: bei unabhaengigen Trades ist t(Tag) = t(Trade) (2.000 Seeds: 0.55 vs 0.56). Der alte Test bestand nur per Seed-Glueck (28 %). | Docstrings in `screening`, `significance`, `acceptance`, `docs/ACCEPTANCE.md` korrigiert; Test `test_daily_aggregation_does_not_add_power` ersetzt ihn. Die Tagesreihe bleibt (ehrlich bei korrelierten Trades), bringt aber keine Power. |
| R2 | S4 `t >= 3` unabhaengig davon, wie viel gesucht wurde: zu streng fuer 1 vorregistrierte Variante, zu locker fuer das Beste aus 1.000. | `screening.required_t(n_trials)` (Bonferroni, 1 -> 1.645) und `screening.deflated_sharpe` (Bailey/Lopez de Prado). S4 nimmt `dsr >= 0.95` wenn vorhanden, sonst `t >= required_t(n_trials)`, ohne `n_trials` 3.0. |
| R3 | S1 pruefte nur `brutto > Placebo`, nicht `brutto > 0`: eine negative Variante kam durch, wenn der Placebo noch negativer war; fehlender Placebo = -inf = immer bestanden. | S1 = `n >= 100` UND `brutto > 0` UND (Placebo fehlt ODER brutto > Placebo). |
| R4 | `cards_to_variants.py` setzte `gross_bp = typische_bewegung_bp` -> S2 verglich wieder eine **Prognose** mit den Kosten. | `gross_bp = None`, Prognose nur noch als Info-Feld `s0_bewegung_prognose_bp`. |
| R5 | S2 kannte nur `gross_bp - cost_bp`. | S2 bevorzugt ein **gemessenes** `net_mean` (Backtest mit Bid/Ask). |
| R6 | Spalten nur aus `None` (JSON `null`) liessen S1/S2 mit TypeError abstuerzen. | `pd.to_numeric(..., errors="coerce")`. |
| R7 | `score.py` (Evolution) gewichtet Trade-Frequenz 25 %, Sharpe per Trade mit Pauschalfaktor - der Hill-Climb sucht "3-5 Trades/Woche", nicht ein Edge. | Unveraendert fuer die Deploy-Worker; fuer Forschung ersetzt durch `trading_lab/optimizer.py`. |
| R8 | `iterative_backtest.py` optimiert waehrend des Backtests (Audit F9). | Als Antipattern im Modul-Docstring markiert. |

## Neu: `trading_lab/optimizer.py` + `studies/optimize_all.py`

* Suche nur auf **Train** (2019-2024), Holdout 2025-01-01..heute.
* Ziel: **untere Konfidenzgrenze des Netto-Erwartungswerts je Trade** (`mean R - 1 se`),
  nicht R:R (R:R ist ein Parameter) und nicht der Evolutions-Score.
* Zufallsstart + Koordinatenabstieg ueber ein diskretes Grid (Entry, Filter, Stop, Ziel,
  Haltedauer, Session).
* Jede **unterschiedliche** Ergebniszeile ist ein Trial; `n_eff` = effektive Zahl
  unabhaengiger Trials aus der mittleren Korrelation -> DSR.
* Shortlist -> Grid-Nachbarn (S3), Jahre (S3), Replay-Placebo (Gegenrichtung, gleiche
  Einstiegszeit/Stop/Ziel, feste Haltedauer = Median), PBO ueber alle Trials.
* Holdout nur mit Freigabe und nur fuer S4-Ueberlebende.
* Prop-Profil je Kandidat (`prop_profile`) + `studies/optimize_report.py` (FTMO/1-Step/
  Instant via `trading_lab.prop_sim`).

## H-005 / H-006 erstmals getestet (`studies/run_h005_h006.py`)

* H-005 (16:00-Fix): brutto EURUSD/GBPUSD leicht positiv, netto ~0 bis negativ, kein
  Fall besteht S3 -> kill (gemessen).
* H-006 (Relative Value): AUDUSD/NZDUSD mit **echtem Brutto-Effekt** (+1.7..+3.1 bp,
  t 2.8..3.5, Placebo ~0), netto negativ wegen zwei Legs Kosten -> kill an S2.
  Offene Spur: Limit-Einstieg.

## Ergebnis des ersten Gesamtlaufs (2026-10-26)

S1 151/249 -> S2 137 -> S3 77 -> **S4 0** -> S5 nicht geöffnet. Bester Kandidat:
momentum US100 (DSR 0.937, 6/6 Jahre, alle Nachbarn positiv, PBO 0.17). Details:
`docs/HANDOFF_2026-10-26.md`, Bericht `artifacts/optimize/report.md`.
