# Mean-Reversion-Kandidaten: Vollständiger Bauplan

**Status:** Planungsdokument, noch keine Implementierung begonnen
**Erstellt:** 2026-10-13
**Bezug:** Baut auf `STRATEGY_RESEARCH.md`, der bestehenden `trading_lab/strategies/regime_hybrid.py` (getestet, schwach) und dem Regelwerk `Gold_Mean-Reversion_Strategie_Regelwerk.md` (30.09.2026, bisher nicht implementiert) auf.
**Zielleser:** Eine Claude-Code-Session in VS Code, die dieses Dokument als Arbeitsgrundlage nimmt und Schritt für Schritt umsetzt.

---

## 0. Kontext — warum dieses Dokument existiert

Ziel ist ein zweites, strukturell unkorreliertes Strategie-Standbein neben dem bestehenden Supertrend+Volume-Bot, mit realistischem Ertragsziel **1-3% pro Monat** (bewusst niedriger als das ursprüngliche 6%-Ziel aus `STRATEGY_RESEARCH.md`, das für Supertrend gewählt wurde). Mean Reversion passt zu diesem niedrigeren Ziel besser als zur ursprünglichen 6%-Vorgabe, weil weniger Trade-Frequenz nötig ist (siehe Abschnitt 7, Frequenz-Rechnung).

Ein erster Versuch existiert bereits im Code: `regime_hybrid.py` schaltet per ADX zwischen Supertrend (Trend) und Bollinger-Band-Mean-Reversion (Range) um. Echtes Backtest-Ergebnis (`research/regime_forex_backtest.json`, 14.07.2026):

| | USD/JPY | Gold (GC=F) |
|---|---|---|
| MR-Trades (von Gesamt) | 15 von 1100 | 52 von 1091 |
| OOS PnL (Hybrid) | +4,31% | +37,28% |
| Reiner Supertrend (ohne MR) | +12,91% | +43,74% |
| MR-Effekt | **-8,6 Punkte** | **-6,46 Punkte** |

Fazit aus diesem Test: Das MR-Modul hat in beiden Fällen die Performance verschlechtert, feuerte aber auch viel zu selten (15 Trades bei JPY über 40.000+ Kerzen), um daraus überhaupt eine verlässliche Aussage abzuleiten. Das ist der Ausgangspunkt für dieses Dokument — wir bauen zwei neue, bessere Kandidaten und lassen sie gegeneinander und gegen diese Baseline antreten.

---

## 1. Architektur-Überblick (bestehendes System, zum Verständnis nötig)

Beide neuen Kandidaten müssen sich in dieses bestehende Muster einfügen, nicht etwas Eigenständiges daneben bauen.

```
trading_lab/
├── run.py                        # Entrypoint, dispatcht per LAB_TRADING_WORKER env var
├── score.py                      # Generische Zielfunktion score(trades, goal) -> [-1, +1] — WIEDERVERWENDEN
├── reflect.py                    # State/Hypothesen/Versionierung (load_yaml, save_prior, append_hypothesis,
│                                  #   write_strategy, next_version, prune_state_volume) — WIEDERVERWENDEN
├── risk.py                       # Position, RiskManager — WIEDERVERWENDEN
├── backtest.py                   # _compute_metrics, _pos_to_dict, print_metrics — WIEDERVERWENDEN
├── exploration.py                # RESEARCH_SEEDS, random_config (bisher Supertrend-spezifisch)
├── adapters/market_data.py       # get_market_adapter() — Dukascopy/yfinance Abstraktion, WIEDERVERWENDEN
├── strategies/
│   ├── supertrend_volume.py      # Bestehende Trendfolge-Strategie
│   ├── regime_hybrid.py          # Bestehender (schwacher) MR-Versuch — NICHT ANFASSEN
│   ├── bb_squeeze_breakout.py
│   ├── opening_range_breakout.py
│   ├── trend_range_rejection.py  # TRR — anderes Konzept (SMC-basiert), nicht verwechseln
│   └── smc_ict.py / silver_bullet.py
├── st_evolution_worker.py        # Hill-Climb + Simulated Annealing für Supertrend (937 Zeilen)
├── smc_evolution_worker.py       # Gleiches Muster für SMC (755 Zeilen)
├── orb_evolution_worker.py       # Gleiches Muster für ORB (581 Zeilen) — KÜRZESTE VORLAGE, hier orientieren
└── signal_loop.py                # LIVE-Signal-Loop (Telegram), separat von Evolution
```

**Dispatch-Muster in `run.py`** (`LAB_TRADING_WORKER` env var):
- `evolution` → `STEvolutionWorker` (Supertrend, Standard)
- `smc_evolution` → `SmcEvolutionWorker`
- `orb_evolution` → `OrbEvolutionWorker`
- `signal` → `SignalLoop` (live, Telegram-Benachrichtigungen)

**Deployment:** Jedes Asset/jede Strategie läuft als eigener Railway-Service mit eigenem State-Volume (z. B. `jpy-state` vs. `state`), geseedet über `LAB_STATE_TEMPLATE` beim ersten Boot (siehe `Dockerfile`).

**Wichtige Phasentrennung, die für beide neuen Kandidaten gilt:**
1. **Phase Research/Evolution** (jetzt): Backtest + Hill-Climb-Suche nach guten Parametern, kein echtes Geld, kein Live-Signal
2. **Phase Live-Signal** (später, nur falls Go-Kriterien erfüllt): Eigener `signal_loop`-Zweig mit Telegram, State-Persistenz gegen Railway-Neustarts — analog zu dem, was für Supertrend/Gold bereits existiert

Dieses Dokument behandelt nur Phase 1. Phase 2 wird erst angegangen, wenn ein Kandidat die Go-Kriterien (Abschnitt 8) erfüllt.

---

## 2. Kandidat A — Session-Range-Fade (Gold, "reine" Mean Reversion)

Basiert vollständig auf dem Regelwerk vom 30.09.2026. Vollständige Regeln hier noch einmal zusammengefasst, mit Code-Mapping.

### 2.1 Konzept
Während der ruhigen Asien-Session (00:00-07:00 UTC) bildet Gold eine Range. Bricht der Preis in der volatileren London-Session (07:00-11:00 UTC) aus dieser Range aus, ohne dass sich ein Trend etabliert, wird der Ausbruch gefadet — Trade läuft zurück Richtung Range-Mitte/gegenüberliegende Grenze.

### 2.2 Instrument & Daten
- **Instrument:** XAU/USD (Gold), Dukascopy, über bestehenden `adapters/market_data.py`
- **Timeframe:** M5 oder M15 (nicht Tick-für-Tick — erhöht nur Overfitting-Risiko)
- **Alle Zeiten in UTC**, explizit mit `zoneinfo`/`pytz` normalisiert, niemals manueller Offset

### 2.3 Range-Bildung
- Fenster: `>= 00:00 UTC` und `< 07:00 UTC` (exakt so, nicht `<=`)
- `Range_High` = höchstes High im Fenster (Ask-Preise)
- `Range_Low` = niedrigstes Low im Fenster (Bid-Preise)
- Bid/Ask konsequent auseinanderhalten, nicht mit Mid-Preisen rechnen
- **Range-Size-Filter:** `Range_Size = Range_High - Range_Low` muss zwischen `1.5× ATR(14, Tagesbasis)` und `4× ATR(14, Tagesbasis)` liegen — sonst kein Setup an diesem Tag (verhindert Noise-Ranges und bereits laufende Trendtage)

### 2.4 Entry-Logik (nur im London-Fenster 07:00-11:00 UTC)
**Short-Setup (Fade am Range-Top):**
1. Eine vollständig **geschlossene** Kerze bricht mit ihrem High über `Range_High` aus
2. Die darauffolgende, ebenfalls vollständig geschlossene Kerze schließt wieder **unterhalb** von `Range_High` zurück
3. Short-Signal, Entry zum Open der nächsten Kerze (nicht zum Close der Bestätigungskerze)

**Long-Setup:** spiegelbildlich am `Range_Low`

**Max. 1 Trade pro Session** — sobald ein Signal ausgelöst wurde, kein weiteres mehr bis 00:00 UTC

### 2.5 Exit
- **SL:** Short: Ausbruchs-Extrempunkt (höchstes High) + Puffer (0.3× ATR(14) oder fixer Wert, z. B. 1.50 USD). Long spiegelbildlich.
- **TP1** (50% der Position): Range-Mitte `(Range_High + Range_Low) / 2`, danach SL auf Break-even nachziehen
- **TP2** (Rest): gegenüberliegende Range-Grenze
- **Zeit-Exit:** Falls bis 11:00 UTC (Ende London-Fenster) weder SL noch TP2 erreicht, Position zum Marktpreis schließen

### 2.6 Risk-Management
- Risiko pro Trade: 0,5-1% des Kontostands (fixed fractional), Formel: `Lotgröße = (Kontostand × Risiko%) / (|Entry - SL| × Kontraktgröße)`
- Max. 1 Trade/Tag deckelt das Tagesrisiko ohnehin
- Circuit Breaker: nach -4% Drawdown in einer Kalenderwoche pausiert die Strategie bis zur nächsten Woche

### 2.7 Filter / No-Trade-Bedingungen
- **Trendtag-Filter:** Falls Preis in den letzten 24h vor Range-Bildung bereits > 2× ATR(14) Richtungsbewegung gemacht hat → Setup an diesem Tag überspringen
- **Monatsultimo/Rollover:** 21:00-22:00 UTC Rollover, letzter Handelstag im Monat → keine neuen Entries
- News-Blackout (NFP/FOMC/CPI) und Wochentag-Ausschlüsse (Montagvormittag, Freitagnachmittag) sind im Regelwerk vorgesehen, **werden aber in Phase 1 (Backtest) zunächst NICHT eingebaut** — sie brauchen einen externen Calendar-Feed bzw. sollen laut Regelwerk erst empirisch validiert werden, bevor sie fest codiert werden. Das ist ein bewusster Scope-Cut für die erste Version, kein Vergessen.

### 2.8 Code-Mapping
Neue Datei: `trading_lab/strategies/session_range_fade.py`

```python
@dataclass(frozen=True)
class SessionRangeFadeConfig:
    asset: str = "XAU/USD"
    asset_label: str = "XAU/USD"
    timeframe: str = "15min"          # M5 oder M15, konfigurierbar für Evolution-Suche

    asia_start_utc_hour: int = 0
    asia_end_utc_hour: int = 7
    london_start_utc_hour: int = 7
    london_end_utc_hour: int = 11

    atr_period_daily: int = 14
    range_min_atr_mult: float = 1.5
    range_max_atr_mult: float = 4.0

    sl_buffer_atr_mult: float = 0.3
    tp1_close_pct: float = 0.5        # Range-Mitte, Teilverkauf
    # TP2 = gegenüberliegende Range-Grenze (keine separate Config nötig)

    trend_day_atr_mult: float = 2.0   # Kill-Switch-Schwelle

    account_size: float = 10000.0
    risk_per_trade: float = 0.0075    # Mitte von 0.5-1%
    max_trades_per_day: int = 1

    weekly_circuit_breaker_pct: float = -0.04
```

Funktionen (Stil wie `regime_hybrid.py`): `compute_asia_range(df, cfg)`, `generate_signals(df, cfg)` (Breakout+Reversal-Erkennung, **ausschließlich auf vollständig geschlossenen Kerzen**), `run_backtest(df, cfg)` (nutzt `RiskManager`/`Position` aus `risk.py`, `_compute_metrics`/`_pos_to_dict` aus `backtest.py`).

### 2.9 Guards (aus dem Regelwerk, zwingend umzusetzen — Tabelle zur Prüfung beim Code-Review)

| Fehler | Guard |
|---|---|
| Lookahead-Bias | Signal nur auf letzter vollständig geschlossener Kerze; `df.iloc[-1]` explizit prüfen, nicht blind als "fertig" annehmen |
| Off-by-one Session-Grenzen | Grenzen exakt `>= Start` und `< Ende`, nie `<=` |
| Timezone-Bug | Jeden Timestamp explizit nach UTC normalisieren (zoneinfo), kein manueller Offset |
| Bid/Ask-Inkonsistenz | Ask für Käufe, Bid für Verkäufe — identisch in Backtest und später Live |
| Datenlücken | Zeitlücken > X Minuten erkennen, diese Perioden von Range-Berechnung ausschließen |
| Division durch Null | Sanity-Check: `Range_Size` muss über Mindestgröße liegen, sonst Tag überspringen |
| Unrealistischer Backtest | Historische Spreads verwenden, sonst bewusst konservativen Fixwert ansetzen |
| Overfitting | Walk-Forward-Validierung (siehe Abschnitt 7), keine Nachjustierung an OOS-Daten |

Die Live-spezifischen Guards (State-Persistenz gegen Railway-Neustart, doppelte Telegram-Nachrichten, Race Conditions bei Cron) sind erst in **Phase 2 (Live-Signal)** relevant, nicht beim Backtest/Evolution-Worker.

---

## 3. Kandidat B — Z-Score Mean Reversion (Gold + USD/JPY, mechanisch, höhere Frequenz)

Mein Vorschlag von vorhin, hier vollständig spezifiziert mit gleicher Genauigkeit wie Kandidat A.

### 3.1 Konzept
Statt eines seltenen, session-strukturellen Events (Kandidat A) ein mechanischeres, häufigeres Signal: Preisabweichung vom gleitenden Mittelwert in Standardabweichungen gemessen (Z-Score), mit zwei Stärke-Stufen statt einer harten Schwelle. Höhere Frequenz by design, auf 15m statt 1h.

### 3.2 Instrument & Daten
- **Phase 1 (dieses Dokument):** Gold (GC=F) und USD/JPY **getrennt**, je eigener Lauf — konsistent mit dem bestehenden Pro-Asset-Muster des Systems (siehe Abschnitt 1). Eine kombinierte Basket-Auswertung über mehrere JPY-Paare ist bewusst **nicht** Teil von Phase 1 (höheres Engineering-Risiko, passt nicht ins bestehende State-pro-Asset-Muster) — kommt nur als Ausbauschritt, falls B einzeln überzeugt.
- **Timeframe:** 15m
- **Daten:** Dukascopy, gleicher Adapter wie Kandidat A und `regime_hybrid.py`

### 3.3 Signal-Logik
```
SMA20 = rolling_mean(close, 20)        # auf 15m
STD20 = rolling_std(close, 20)
z = (close - SMA20) / STD20

LONG Tier A:  z <= -2.0   → volle Risiko-Size
LONG Tier B:  -2.0 < z <= -1.5  → halbe Risiko-Size
SHORT Tier A: z >= 2.0    → volle Risiko-Size
SHORT Tier B: 1.5 <= z < 2.0 → halbe Risiko-Size
```
Bei gleichzeitigem Zutreffen von Tier A und B-Bedingung gilt Tier A (strengere Schwelle hat Vorrang, keine Doppelzählung).

**Trendfilter (lockerer als bei `regime_hybrid.py`):** ADX(14) auf 15m. Kein Entry (weder Tier A noch B), wenn `ADX >= adx_block_threshold` (Default 32 — bewusst höher als die 25 aus `regime_hybrid.py`, da wir Reversion auch in moderatem Trend als Pullback zulassen wollen, nur bei starkem Trend aussetzen).

**Volatilitäts-Mindestfilter (Division-durch-Null-Guard, analog Kandidat A Abschnitt 2.9):** `STD20 / close` muss über `min_std_pct` (Default 0.05%) liegen — sonst kein Signal (verhindert künstlich aufgeblähte Z-Scores in Totphasen).

**Optionaler Reversal-Bestätigungsfilter:** `require_reversal_candle: bool` (Default `False`). Falls `True`: aktuelle Kerze muss Richtung Mittelwert geschlossen haben (wie im alten `regime_hybrid.py`-Modul). Wird bewusst als Config-Flag (nicht hart codiert) gebaut, damit der Evolution-Worker empirisch herausfindet, ob diese Bedingung hilft oder (wie im alten Modul vermutet) die Frequenz unnötig killt.

**Optionaler Session-Filter:** `session_filter_enabled: bool` (Default `True`), Fenster 07:00-20:00 UTC (London+NY, schließt dünne Asia-Only-Stunden aus) — konfigurierbar, nicht hart in die Entry-Logik gegossen.

### 3.4 Exit-Logik
- **TP1** (50%, konfigurierbar `tp1_close_pct`): Rückkehr bis `SMA20 - 0.5×STD20` (bei Long) bzw. `SMA20 + 0.5×STD20` (bei Short) — nicht erst beim vollen Mittelwert, nimmt früher Teilgewinn
- **TP2** (Rest): volle Rückkehr zu `SMA20`
- Nach TP1: SL auf Break-even
- **SL:** `entry ± sl_atr_mult × ATR(14, 15m)`, Default `sl_atr_mult = 1.5` (gleicher Wert wie in `regime_hybrid.py`, konsistent)
- **Zeit-Stop:** `max_hold_bars` (Default 16 = 4h auf 15m), danach Schließung zum Marktpreis

### 3.5 Code-Mapping
Neue Datei: `trading_lab/strategies/mean_reversion.py`

```python
@dataclass(frozen=True)
class MeanReversionConfig:
    asset: str = "GC=F"
    asset_label: str = "XAU/USD"
    timeframe: str = "15min"

    z_period: int = 20
    z_entry_tier_a: float = 2.0
    z_entry_tier_b: float = 1.5
    tier_b_size_mult: float = 0.5

    adx_period: int = 14
    adx_block_threshold: float = 32.0
    min_std_pct: float = 0.0005

    require_reversal_candle: bool = False

    atr_period: int = 14
    sl_atr_mult: float = 1.5
    tp1_std_mult: float = 0.5
    tp1_close_pct: float = 0.5
    max_hold_bars: int = 16

    session_filter_enabled: bool = True
    session_start_utc_hour: int = 7
    session_end_utc_hour: int = 20

    account_size: float = 10000.0
    risk_per_trade: float = 0.01
    max_concurrent_positions: int = 1
    long_only: bool = False
    cooldown_bars: int = 0
```

Funktionen (gleicher Stil wie `regime_hybrid.py`): `prepare_dataframe(df, cfg)` (Z-Score, ADX, STD-Filter berechnen, nur auf abgeschlossenen Kerzen), `generate_signals(...)`, `manage_position(...)`, `run_backtest(df, cfg)`.

### 3.6 Guards (zusätzlich zu den generischen aus Abschnitt 2.9, spezifisch für diese Strategie)

| Fehler | Guard |
|---|---|
| Division durch Near-Zero STD | `min_std_pct`-Filter vor Z-Score-Berechnung |
| Tier A/B Doppelzählung | Tier-Priorität explizit im Code: erst Tier-A-Bedingung prüfen, sonst Tier-B |
| ADX NaN am Serienanfang | `dropna(subset=[...])` vor Backtest-Loop, wie in `regime_hybrid.py` bereits gehandhabt |
| 15m-Datenlücken (Wochenende, dünne Asia-Session bei JPY) | Gleicher Gap-Guard wie Kandidat A |
| Z-Score auf laufender Kerze | Nur geschlossene Kerzen verwenden, identischer Guard wie Kandidat A |

---

## 4. Gemeinsame Infrastruktur (von beiden Kandidaten wiederverwendet, nicht neu bauen)

- `score.py::score(trades, goal)` — generische Zielfunktion, direkt wiederverwendbar
- `reflect.py` — `load_yaml`, `save_prior`, `append_hypothesis`, `write_strategy`, `next_version`, `prune_state_volume`
- `risk.py` — `Position`, `RiskManager`
- `backtest.py` — `_compute_metrics`, `_pos_to_dict`, `print_metrics`
- `adapters/market_data.py::get_market_adapter()` — Dukascopy-Zugriff

---

## 5. Evolution-Worker für beide Kandidaten

Je ein neuer Worker, modelliert nach `orb_evolution_worker.py` (kürzeste, einfachste bestehende Vorlage, 581 Zeilen) — Hill-Climb + Simulated Annealing, Reflexion nach jeweils N Trades, Out-of-Sample-Forward-Eval, Leaderboard.

- `trading_lab/session_range_evolution_worker.py` — Parametersuche über `range_min_atr_mult`, `range_max_atr_mult`, `sl_buffer_atr_mult`, `trend_day_atr_mult`, `timeframe` (M5 vs M15)
- `trading_lab/mean_reversion_evolution_worker.py` — Parametersuche über `z_entry_tier_a/b`, `adx_block_threshold`, `sl_atr_mult`, `max_hold_bars`, `require_reversal_candle` (True/False), `session_filter_enabled`

Neue State-Templates: `session-range-state-template/`, `mean-reversion-state-template/` (je mit `goal.yaml`, Seed-Strategy-YAML) — Struktur analog zu `orb-state-template/`.

**`goal.yaml`-Kalibrierung für beide** (angepasst an 1-3%/Monat statt der bisherigen höheren Ziele):
```yaml
target_return_30d: 0.02       # 2% Mitte der Zielspanne
max_drawdown: 0.10
min_sharpe: 0.5                # niedriger als die bisherigen 0.8/1.2, da MR naturgemäß geringere Sharpe pro Trade hat
min_trades_per_week: 2
target_trades_per_week: 5
reflection_every: 15
```

**Einbindung in `run.py`:** Neue `worker_mode`-Zweige `session_range_evolution` und `mr_evolution`, analog zu den bestehenden `smc_evolution`/`orb_evolution`-Blöcken.

---

## 6. Implementierungsreihenfolge (konkrete Dateiliste, in Bau-Reihenfolge)

1. `trading_lab/strategies/session_range_fade.py` (Kandidat A, Strategie-Logik)
2. `trading_lab/strategies/mean_reversion.py` (Kandidat B, Strategie-Logik)
3. Sanity-Check-Skripte (einmalige Einzel-Backtests, kein Evolution-Loop): `backtest_session_range_sanity.py`, `backtest_mean_reversion_sanity.py` — nur prüfen: laufen beide ohne Crash, plausible Trade-Anzahl, keine offensichtlichen Bugs
4. `trading_lab/session_range_evolution_worker.py`
5. `trading_lab/mean_reversion_evolution_worker.py`
6. `session-range-state-template/`, `mean-reversion-state-template/` (je `goal.yaml` + Seed-YAML)
7. `run.py` erweitern (zwei neue `worker_mode`-Zweige)
8. Lokaler Dry-Run beider Worker (begrenzte Rundenzahl, **nicht** `run_forever()`)
9. Vergleichs-Report (Abschnitt 8)

---

## 7. Frequenz-Rechnung (warum 1-3%/Monat realistisch weniger Trades braucht als das alte 6%-Ziel)

Bei 1% Risiko pro Trade, Ziel 2%/Monat (Mitte der Spanne):

| Erwartungswert/Trade | Trades/Monat nötig | Trades/Woche |
|---|---|---|
| +0.10R | 20 | ~5 |
| +0.15R | 13 | ~3 |
| +0.20R | 10 | ~2-3 |

Überschlag für Kandidat A (max. 1 Trade/Tag, zusätzlich durch Range-Size-Filter + Breakout-Reversal-Muster gefiltert): geschätzt nur ~15-25% der Tage liefern überhaupt ein gültiges Signal → ~4-7 Trades/Monat. Das liegt **unter** der obigen Tabelle selbst bei +0.15R Erwartungswert — reine Spekulation, die der Backtest (Abschnitt 8) klären muss. Falls die reale Frequenz zu niedrig ausfällt, ist Kandidat A trotz guter Logik für das 1-3%-Ziel allein nicht ausreichend und bräuchte entweder zusätzliche Setups oder bleibt bewusst ein kleiner Diversifikations-Baustein statt Haupttreiber.

Kandidat B ist für höhere Frequenz designt (15m statt 1h, zwei Tiers statt einer harten Schwelle, lockererer ADX-Cutoff) — soll näher an die 10-20 Trades/Monat-Zielspanne kommen.

---

## 8. Validierung & Go-Kriterien

**Methodik (für beide Kandidaten identisch, über den jeweiligen Evolution-Worker):**
- Walk-Forward statt einfachem Split: rollierende Fenster, mehrere Marktregime abgedeckt
- Mindeststichprobe: 100-150+ Trades in-sample, bevor überhaupt eine Aussage getroffen wird
- Parameter-Sensitivität: Kernparameter ±20% variieren — bricht der Edge weg, ist es Overfitting
- Realistische Kosten: Spread/Slippage einrechnen (Dukascopy-Spreads, kein 0-Spread-Backtest)
- Monte-Carlo-Bootstrap der Trade-Reihenfolge für Drawdown-/Consistency-Rule-Risiko
- **Korrelationscheck gegen den bestehenden Supertrend-Bot** — gemeinsame Equity-Kurve simulieren, nicht getrennt betrachten. Das ist der eigentliche Test, ob "Diversifikation" hier wirklich funktioniert, nicht nur die Standalone-Metrik des neuen Kandidaten.

**Go-Kriterien (Vorschlag, angepasst an 1-3%/Monat-Ziel, analog zum Format in `research/regime_forex_backtest.json`):**
```yaml
oos_pnl_pct: "> 0"
oos_sharpe: ">= 0.5"
oos_dd_pct: "<= -10.0"   # nicht schlechter als -10%
oos_tpw: ">= 2.0"        # mind. 2 Trades/Woche
```

**Vergleichstabelle am Ende (Format wie bisherige Backtests):**

| | Kandidat A (Session-Range) | Kandidat B (Z-Score) | Baseline: regime_hybrid (alt) | Baseline: reiner Supertrend |
|---|---|---|---|---|
| OOS Trades/Woche | | | 3.06 (JPY) / 3.18 (Gold) | — |
| OOS PnL % | | | +4.31 (JPY) / +37.28 (Gold) | +12.91 (JPY) / +43.74 (Gold) |
| OOS Sharpe | | | 0.47 / 1.91 | — |
| OOS Max DD % | | | -9.45 / -8.65 | — |
| Go? | | | Nein (JPY) / Ja (Gold, nur wegen Trend-Modul) | — |

---

## 9. Offene Punkte / bekannte Lücken (bewusst nicht in Phase 1 gelöst)

- **Portfolio-weites Risikolimit** über alle Bots hinweg (Regelwerk Abschnitt 6: max. 3% gleichzeitig offenes Risiko kombiniert) — existiert im System noch nicht, muss gebaut werden, sobald mehr als eine Strategie gleichzeitig live läuft. Nicht Teil dieses Dokuments.
- **News-Blackout-Filter** für Kandidat A braucht einen externen Economic-Calendar-Feed — aus Scope-Gründen auf später verschoben
- **Basket-Erweiterung** für Kandidat B (mehrere JPY-Paare kombiniert) — nur falls Einzel-Ergebnis für USD/JPY überzeugt
- **Railway-Deployment-Methode:** Kein `railway`-CLI auf dem lokalen Rechner gefunden/authentifiziert. Wird erst relevant nach Abschnitt 8 (Vergleichs-Report), nicht vorher. Entweder CLI einrichten oder manuelles Anlegen des Railway-Service über das Dashboard.

## 10. Explizit NICHT Teil dieser Runde

- Kein Live-Trading, keine Verbindung zu MT5/Broker/Prop-Firm-Konto
- Keine Änderung an `regime_hybrid.py` oder anderen laufenden Strategien
- Keine Entscheidung über ein Instant-Funded-Konto — das kommt erst, wenn ein Kandidat die Go-Kriterien erfüllt UND der Korrelationscheck gegen den bestehenden Bot positiv ausfällt
