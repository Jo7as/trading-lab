# Multi-Asset Recherche: höhere Frequenz mit hf_ls01

**Datum:** 2026-07-14  
**Strategie:** `hf_ls01` (ST 7/1.5, no EMA, long+short, SL 1.5×ATR, TP1 2R, 48h hold)  
**Daten:** Dukascopy, Start 2020-01-01, OOS = 30 % Holdout  
**Rohdaten:** [`research/multi_asset_backtest_1h.json`](research/multi_asset_backtest_1h.json), [`research/multi_asset_backtest_15m.json`](research/multi_asset_backtest_15m.json)

---

## Executive Summary

Ziel war mehr Trades durch zusätzliche Instrumente (Silber, Forex), ohne die Strategie zu ändern.

**Kernbefund:** Auf **1h** liefern fast alle getesteten Märkte **ähnliche Frequenz** (~3,8–4,6 Trades/Woche) — deutlich mehr als Gold allein bringt nur **Gold M15** (~15,6/Woche OOS). Die meisten Forex-Paare sind mit `hf_ls01` **OOS negativ**.

| Empfehlung | Warum |
|------------|-------|
| **#1 Gold 1h + USD/JPY 1h** | ~8 OOS-Trades/Woche (~35/Monat), JPY OOS +12,9 %, Korrelation zu Gold **−0,33** |
| **#2 Gold M15 allein** | ~16 OOS-Trades/Woche (~68/Monat), aber OOS nur +6,8 %, Sharpe 0,14 |
| **Nicht: Silber** | Hohe Korrelation (0,80), Shorts verlieren massiv, Full-Period −43 % (1h) / −95 % (15m) |
| **Nicht: EUR/GBP/AUD** | OOS negativ (−5 % bis −9 %) |

---

## 1. Warum mehr Instrumente?

Aktuell (Signal-Bot): nur **XAU/USD 1h** → ~**4 Trades/Woche** (~17/Monat).

Ziel laut Anfrage: **≥15 Trades/Monat** bei 20 Handelstagen (~0,75/Tag).

Mögliche Hebel:

1. **Neues Instrument** (Silber, Forex) — gleiche Logik, anderer Markt
2. **Kürzerer Timeframe** (M15 statt 1h) — gleiches Instrument
3. **Kombination** — Portfolio aus uncorrelated Assets

---

## 2. Instrument-Profile (Recherche)

### Silber (XAG/USD)

- **Volatilität:** ~32 % p.a. vs. Gold ~19 % ([Gale Finance 2025](https://www.gale.finance/compare/xau-vs-xag-2025/))
- **Korrelation zu Gold:** ~0,66–0,85 (unsere Messung: **0,798** wöchentlich)
- **Eigenschaft:** 2–3× stärkere %-Moves, Short-Seite riskanter
- **Für ST+Volume:** Mehr Flips möglich, aber **nicht unabhängig** von Gold — viele parallele Signale

### USD/JPY

- Klassischer Trend-Market, Zins-/Risk-Sentiment-getrieben
- **Korrelation zu Gold:** **−0,326** (negativ = echte Diversifikation)
- Forex-Volume oft sparse → Strategie nutzt **ATR-Fallback** für Volume-Filter

### EUR/USD, GBP/USD, AUD/USD

- Interne Forschung ([STRATEGY_RESEARCH.md](STRATEGY_RESEARCH.md) §9.2) hatte EUR bereits als schwach identifiziert
- Höhere Frequenz als Gold, aber **negative Expectancy** mit `hf_ls01` auf 1h (siehe Backtests)

---

## 3. Backtest-Ergebnisse — 1h (Hauptvergleich)

Konfiguration: exakt `hf_baseline.yaml` / `hf_ls01`, unverändert.

| Asset | Full Trades | Full /Woche | OOS Trades | OOS /Woche | OOS /Monat | OOS PnL | OOS Sharpe | OOS DD |
|-------|-------------|-------------|------------|------------|------------|---------|------------|--------|
| **XAU/USD (GC=F)** | 1341 | 3,95 | 396 | 3,91 | 16,8 | **+43,4 %** | **1,81** | −9,1 % |
| XAG/USD (Silber) | 1284 | 3,77 | 370 | 3,63 | 15,7 | +16,4 % | 0,50 | −15,6 % |
| **USD/JPY** | 1413 | 4,16 | 419 | 4,12 | 17,8 | **+12,9 %** | 1,06 | −7,4 % |
| GBP/USD | 1490 | 4,38 | 439 | 4,30 | 18,7 | −7,2 % | −0,81 | −13,0 % |
| EUR/USD | 1551 | 4,56 | 456 | 4,47 | 19,4 | −9,3 % | −1,04 | −12,9 % |
| AUD/USD | 1440 | 4,23 | 428 | 4,28 | 18,6 | −5,5 % | −0,46 | −9,2 % |

**Interpretation:**

- **Frequenz:** Alle Märkte erfüllen ≥15 Trades/Monat OOS einzeln — das war auf 1h **kein Engpass**.
- **Qualität:** Nur **Gold** und **USD/JPY** sind OOS klar profitabel. Silber OOS +16 %, aber Full-Period katastrophal (Shorts −46 %).
- **Forex-Majors** (EUR, GBP, AUD): mehr Trades, aber **Geldverlust** OOS mit Gold-optimierten Parametern.

### Silber — Warnung Full vs. OOS

| | 1h Full | 1h OOS | Short-PnL Full |
|--|---------|--------|------------------|
| XAG/USD | **−43,4 %** | +16,4 % | **−46,2 %** |

Silber mit `long_only=false` und gleichen SL/TP wie Gold **zerstört** die Short-Seite über den Gesamtzeitraum. OOS-Positivität ist kein robustes Signal für Live-Trading ohne Re-Optimierung oder `long_only=true`.

---

## 4. Backtest-Ergebnisse — 15m (Referenz Gold + Silber)

| Asset | OOS Trades | OOS /Woche | OOS /Monat | OOS PnL | OOS Sharpe | OOS DD |
|-------|------------|------------|------------|---------|------------|--------|
| **XAU/USD M15** | 1591 | **15,58** | **67,7** | +6,8 % | 0,14 | −17,0 % |
| XAG/USD M15 | 1574 | 15,43 | 67,0 | **−18,6 %** | −0,21 | −60,8 % |

**Gold M15** erfüllt das Frequenz-Ziel allein (~16/Woche), aber mit **schwachem OOS-Edge** (+6,8 % gesamt, ~0,29 %/Monat).

**Silber M15:** Frequenz ja, Performance nein.

---

## 5. Korrelations-Matrix (wöchentliche Returns, 1h, 341 Wochen)

| | Gold | Silber | USD/JPY | GBP | EUR | AUD |
|--|------|--------|---------|-----|-----|-----|
| **Gold** | 1,00 | **0,80** | **−0,33** | 0,40 | 0,39 | 0,48 |
| Silber | 0,80 | 1,00 | −0,25 | 0,41 | 0,40 | 0,53 |
| USD/JPY | −0,33 | −0,25 | 1,00 | −0,43 | −0,51 | −0,36 |
| GBP | 0,40 | 0,41 | −0,43 | 1,00 | 0,77 | 0,72 |
| EUR | 0,39 | 0,40 | −0,51 | 0,77 | 1,00 | 0,65 |
| AUD | 0,48 | 0,53 | −0,36 | 0,72 | 0,65 | 1,00 |

**Portfolio-Frequenz (brutto, OOS):**

| Portfolio | Summe OOS Trades/Woche | ~Trades/Monat | Diversifikation |
|-----------|------------------------|---------------|-----------------|
| Nur Gold 1h | 3,91 | 16,8 | — |
| Gold + Silber 1h | 7,54 | 32,5 | Schlecht (ρ=0,80) |
| **Gold + USD/JPY 1h** | **8,03** | **34,4** | **Gut (ρ=−0,33)** |
| Alle 6 Assets 1h | 24,71 | ~106 | Forex-Cluster korreliert |

Bei Gold+Silber sind ~80 % der Zeit die Signale **nicht unabhängig** — effektive Frequenz deutlich unter der Summe.

---

## 6. Empfehlung

### Für höhere Frequenz **mit** positivem OOS-Edge

**Portfolio: XAU/USD 1h + USD/JPY 1h**

- Kombiniert ~**8 Trades/Woche** OOS (~**34/Monat**) — Ziel 15+/Monat erfüllt
- Beide Märkte OOS profitabel mit gleichen Parametern
- Niedrige Kreuzkorrelation → weniger „doppelte Wetten“
- Risiko: 0,5–0,75 % pro Trade **pro Asset** (nicht 1 % auf beiden gleichzeitig)

### Für maximale Frequenz (Qualität zweitrangig)

**Gold auf M15** statt zusätzliches Asset

- ~16 Trades/Woche, aber schwacher OOS-Edge
- Einfacher operativ (ein Symbol, ein Timeframe)
- Bereits im Leaderboard als `hf_st7_m15` Seed validiert (mit Pullback)

### Nicht empfohlen (mit hf_ls01 unverändert)

| Asset | Grund |
|-------|-------|
| **Silber 1h/15m** | Shorts bluten; Full-DD −82 % bis −135 %; hoch korreliert mit Gold |
| **EUR/USD, GBP/USD, AUD/USD** | OOS negativ; Parameter für Gold nicht übertragbar |
| **Alle 6 Assets parallel** | Hohe Brutto-Frequenz, aber Forex-Block zieht PnL runter |

---

## 7. Szenario-Vergleich (Ziel: ≥15 Trades/Monat)

| Szenario | OOS Trades/Monat | OOS PnL/Monat (geschätzt) | Umsetzbarkeit |
|----------|------------------|---------------------------|---------------|
| Gold 1h (Status quo) | ~17 | ~1,85 % | Läuft |
| Gold 1h + USD/JPY 1h | ~34 | ~2,4 % (Summe, nicht portfolio-adjusted) | Mittel |
| Gold M15 allein | ~68 | ~0,29 % | Signal-Bot-TF-Wechsel |
| Gold 1h + Silber 1h | ~32 | Unklar (Silber Short-Risiko) | Nicht empfohlen |
| Gold + EUR + GBP + AUD | ~90+ | Negativ OOS | Nein |

---

## 8. Umsetzungs-Roadmap (später — nicht Teil dieser Recherche)

1. **USD/JPY** in Dukascopy-Adapter ✅ (bereits gemappt)
2. Separater Backtest mit `long_only=true` nur auf Silber (optional)
3. Multi-Asset Signal-Bot: pro Asset `position.json`, Telegram-Label `XAU` / `JPY`
4. Portfolio-Risk-Cap: max 1 gleichzeitige Position **gesamt** oder max 1 pro Asset
5. Evolution **nicht** auf Forex ausweiten ohne asset-spezifische Parameter

---

## 9. Reproduktion

```bash
# 1h — alle Kandidaten
uv run python backtest_multi_asset.py \
  --data-source dukascopy \
  --start 2020-01-01 \
  --interval 1h \
  --output research/multi_asset_backtest_1h.json

# 15m — Gold + Silber Referenz
uv run python backtest_multi_asset.py \
  --interval 15m \
  --assets "GC=F,XAG/USD" \
  --output research/multi_asset_backtest_15m.json
```

---

## 10. Quellen

- Intern: [STRATEGY_RESEARCH.md](STRATEGY_RESEARCH.md) §9.2 (EUR/USD schwach)
- Intern: [hf_baseline.yaml](hf_baseline.yaml), [state/candidate_leaderboard.json](state/candidate_leaderboard.json)
- [Gale Finance — XAU vs XAG 2025](https://www.gale.finance/compare/xau-vs-xag-2025/) — Volatilität, Korrelation
- [World Gold Council — Gold vs Silver](https://www.gold.org/goldhub/research/gold-safe-haven-versus-silver-wildcard) — Liquidität, Industrie-Anteil Silber
- Dukascopy Spot-Daten via `dukascopy-python` (2020–2026)

---

## Anhang: Geänderte Dateien (nur Research-Infrastruktur)

| Datei | Änderung |
|-------|----------|
| `trading_lab/adapters/dukascopy_adapter.py` | Mapping XAG/USD, USD/JPY, AUD/USD |
| `backtest_multi_asset.py` | Neues Research-Skript |
| `research/multi_asset_backtest_*.json` | Backtest-Rohdaten |

**Unverändert:** Signal-Bot, Evolution, `hf_baseline.yaml`, Railway-Deploys.

---

## 11. Portfolio-Backtest: Gold + Silber (long only) + USD/JPY

**Setup:** `hf_ls01` auf 1h, Dukascopy 2020+, gemeinsames $10k-Konto, 1 % Risiko/Trade, max. 1 Position pro Asset.

Skript: [`backtest_portfolio.py`](backtest_portfolio.py) → [`research/portfolio_gold_xag_jpy.json`](research/portfolio_gold_xag_jpy.json)

### Einzelassets (Full)

| Asset | Modus | Trades/Woche | Full PnL | Full DD |
|-------|-------|--------------|----------|---------|
| Gold | L+S | 3,95 | +60,0 % | −11,0 % |
| Silber | **long only** | 2,19 | −4,1 % | −55,3 % |
| USD/JPY | L+S | 4,16 | +16,8 % | −9,6 % |

### Portfolio kombiniert

| | Full (2020–heute) | OOS (30 % Holdout) |
|--|-------------------|---------------------|
| Trades/Woche | **10,3** | **9,9** |
| Trades/Monat | ~45 | ~43 |
| Total PnL | +72,6 % | **+94,2 %** |
| **$/Monat** | ~0,93 % | **~4,01 %** |
| Sharpe | 0,40 | **1,58** |
| Max DD | −47,3 % | −17,8 % |
| Win Rate | 35,5 % | 39,0 % |

### OOS PnL-Beitrag pro Asset

| Asset | OOS PnL | OOS Trades |
|-------|---------|------------|
| Gold | +43,4 % | 396 |
| Silber (long only) | +37,6 % | 195 |
| USD/JPY | +13,3 % | 418 |
| **Summe** | **+94,2 %** | 1009 |

### Overlap

- Max. **3 Assets gleichzeitig** offen (häufig 2+)
- Gold/Silber oft gleichzeitig long (ρ≈0,80) → bis zu **3 % effektives Risiko** wenn alle 3 offen

### Fazit Portfolio

- **OOS ~4 %/Monat** — deutlich über Gold allein (~1,85 %/Monat)
- Frequenz **~10 Trades/Woche** — Ziel 15+/Monat weit übertroffen
- **Full-Period DD −47 %** wegen Silber-Historie (2020–2022) — OOS-DD nur −18 %
- Silber long only trägt im OOS **+37,6 %** bei (fast so viel wie Gold)
- Risiko: korrelierte Long-Cluster Gold+Silber; Portfolio-Risk-Cap empfohlen (z. B. max 1,5 % Gesamtexposure)
