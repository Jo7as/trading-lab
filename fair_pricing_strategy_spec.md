# Fair Pricing Theory – Algorithmische Strategie-Spezifikation

Version 1.0 · Quelle: Transkript des Interviews mit JJ Simon (Chart Fanatics)

## Herkunfts-Tags

Jede Regel trägt einen Tag, der sagt, woher sie stammt:

- **[E]** – explizit im Transkript genannt.
- **[A]** – aus den Live-Trades bzw. Beispielen abgeleitet.
- **[Ann]** – Annahme, die nötig ist, um die Regel codierbar zu machen. Sie wird als Parameter geführt und sollte im Backtest variiert werden.

Alle Zeiten gelten in `America/New_York` (ET). Alle Signale werden **auf Kerzenschluss** ausgewertet.

---

## 0. Konventionen

| Symbol | Bedeutung |
|---|---|
| `bar[i]` | 1-Minuten-Kerze mit `O,H,L,C,V`. `i` ist die zuletzt **geschlossene** Kerze. |
| `body(i)` | `abs(C[i] - O[i])` |
| `color(i)` | `+1`, wenn `C>O` (grün), `-1`, wenn `C<O` (rot), `0`, wenn `C==O` (Doji) |
| `d` | Handelsrichtung: `+1` = long, `-1` = short |
| `FP` | aktueller Fair Price (ein Preislevel) |
| `D` | Abstand in Punkten vom Entry zum FP: `abs(FP - entry_price)` |
| `pv` | Punktwert: NQ = 20 $/Pkt, MNQ = 2 $/Pkt. Tick = 0,25 Pkt. |
| `snap()` | Rundung auf das Tick-Raster (0,25) |

Daten-Anforderungen:

- 1-Minuten-OHLCV (Pflicht).
- 5-Minuten-OHLCV (optional für den HTF-Bias, kann aus 1 Minute aggregiert werden).
- Wirtschaftskalender mit Zeit und Impact (High Impact = Red Folder).
- Optional ein Feed mit geplanten Reden (Fed usw.).

---

## 1. Parameter (Defaults)

| Parameter | Default | Tag | Bemerkung |
|---|---|---|---|
| `INSTRUMENT` | NQ / MNQ | E | nur NASDAQ-Futures |
| `TF_EXEC` | 1m | E | alle Entries |
| `TF_BIAS` | 5m | E | nur HTF-Bias |
| `SESSION_LEN_MIN` | 90 | E | erste 90 Minuten jeder Session |
| `SESSIONS` | NY_AM 09:30, NY_PM 14:00, ASIA 18:00, LONDON 03:00 | E / Ann | Asia- und London-Startzeit nicht genannt; 18:00 = Globex-Reopen (er erwähnt „6 p.m.“ als Level) |
| `LONDON_ENABLED` | false | E | „works for London“, er nennt es aber nicht als Kern |
| `NEWS_TIME` | 08:30 | E | geplante Red-Folder-News |
| `NEWS_WINDOW_END` | 11:00 | A | News-Reversion lief bis zum Ende von NY AM |
| `HTF_LOOKBACK_H` | 12 | E (6–12) | Spanne im Backtest testen: 6–12 |
| `HTF_NEUTRAL_PTS` | 0 | Ann | Totzone für den Bias |
| `CONT_SL_PTS / CONT_TP_PTS` | 25 / 38 | E | Opening Continuation |
| `CONT_BIG_CANDLE_PTS` | 25 | E | darüber: SL 50 / TP 76 mit halber Größe |
| `CONT_BIG_CANDLE_MEASURE` | `range` (H−L) | Ann | alternativ `body` |
| `CONT_EXTRA_WINDOW_MIN` | 5 | E | zusätzliche Continuations nur in den ersten 5 Minuten |
| `CONT_EXTRA_MAX` | 1 | A / Ann | |
| `EVAL_SL_PTS / EVAL_TP_PTS` | 25 / 38 | E | 1 NQ = −500 $ / +760 $ |
| `EVAL_ALT_SL / ALT_TP` | 50 / 76 | E | halbe Größe, gleicher Dollarbetrag |
| `MIN_SL_PTS_NY` | 25 | E | „never less than 25 in NY“ |
| `TP_REACH_RATIO` | 0.80 | E | 0.75 = „greedy“ |
| `TP_GRID_PTS` | [25, 38, 50, 75, 100, 125, 150, 175, 200] | E / A | |
| `SL_GRID_PTS` | [25, 37.5, 50, 75] | E | häufigste Stops aus seiner Optimierung |
| `SESSION_MAX_CONSEC_REV_LOSSES` | 3 | E | |
| `FUNDED_SAME_DIR_EXPOSURE_CAP` | 2 | Ann | darüber gehen Trades nur noch an Evals |
| `COPY_TRADE_APLUS_N` | 2 | A / Ann | nur der erste BOS nach dem News-Extrem |
| `SWING_STRICT` | true | Ann | `<` statt `<=` |
| `DISP_LOOKBACK` | 1 | E | Close jenseits des Dochts von N Vorkerzen |
| `MOVE_AWAY_MIN_PTS` | `TP_REACH_RATIO * EVAL_TP_PTS` (≈ 30) | Ann | Mindestabstand vom FP, bevor eine Reversion erlaubt ist |
| `CONSOL_MIN_BARS` | 10 | A | er zählt „10 Dochte“ |
| `CONSOL_MAX_RANGE_PTS` | 30 | Ann | alternativ `k * ATR(20)` |
| `CONSOL_LEVEL` | `midpoint` | Ann | Alternativen: `last_close`, `vwap`, `poc` |
| `SHOCK_RANGE_ATR_MULT` | 4.0 | Ann | Proxy für unerwartete News |
| `SHOCK_MIN_PTS` | 60 | Ann | |
| `BE_ON_FP_TOUCH` | false | A | optional |
| `BE_AT_0930_FOR_PREOPEN_TRADES` | true | E | „make sure you're break even when market opens“ |
| `FILL_MODEL` | Market am Open der nächsten Kerze | A | er steigt in der ersten Sekunde der neuen Minute ein |

---

## 2. Primitive (exakte Definitionen)

### 2.1 Displacement-Kerze [E]

Short-Displacement an `bar[i]`:

```
color(i)   == -1                             # rote Kerze
color(i-1) == +1                             # verdrängt Gegenfarbe [E]
body(i)    >  body(i-1)                      # [E]
C[i] < min(L[i-1 .. i-DISP_LOOKBACK])        # Close unter dem Docht [E]
```

Long-Displacement: gespiegelt (grün, Vorkerze rot, `C[i] > max(H[...])`).

Doji als Vorkerze (`color == 0`) → **kein** Displacement [Ann].

Optionale Erweiterung [A, Default off]: Zwei Kerzen der Gegenfarbe hintereinander gelten als Displacement („double displacement“).

### 2.2 Swing-Punkte (Struktur) [E]

- **Swing Low an `j`:** `L[j] < L[j-1]` und `L[j] < L[j+1]`. Gilt erst als bestätigt nach dem Schluss von `j+1`.
- **Swing High an `j`:** `H[j] > H[j-1]` und `H[j] > H[j+1]`.

Ein Swing gilt als **outstanding** (ungebrochen), solange keine Kerze jenseits davon **geschlossen** hat. Ein Docht durch das Level bricht ihn nicht [E].

### 2.3 Break of Structure (BOS) [E]

Short-BOS an `bar[i]`:

1. Es gibt einen bestätigten, ungebrochenen Swing Low `j` mit `j+1 <= i-1`.
2. `C[i] < L[j]`.

Referenz-Swing ist der **jüngste** ungebrochene Swing Low. Werden mehrere Swings gleichzeitig gebrochen, zählt das als **ein** Signal. Die Stärke steigt mit der Anzahl gebrochener Swings bzw. Dochte (nur für das Ranking, [A]).

Long-BOS: gespiegelt mit Swing Highs.

### 2.4 Signalstärke [E]

```
BOS          -> STRONG
DISPLACEMENT -> WEAK
```

Erfüllt eine Kerze beides, gilt sie als STRONG.

### 2.5 Konsolidierung [Ann, aus A]

Ein Fenster von mindestens `CONSOL_MIN_BARS` aufeinanderfolgenden Kerzen mit:

```
max(H) - min(L) <= CONSOL_MAX_RANGE_PTS
```

`level` = Mittelpunkt der Range (siehe `CONSOL_LEVEL`). Gesucht wird immer die **jüngste** Konsolidierung.

### 2.6 Unerwarteter News-Schock [Ann]

Ein Schock an `bar[i]` liegt vor, wenn:

- `(H-L)[i] >= max(SHOCK_MIN_PTS, SHOCK_RANGE_ATR_MULT * ATR20)`,
- und `i` liegt **nicht** in `[NEWS_TIME, NEWS_TIME+2min]` eines geplanten Events,
- und `i` ist **nicht** die erste Kerze einer Session.

`shock_dir = color(i)`.

Hinweis: Er handelt hier diskretionär auf Tweets bzw. Reden. Für einen Algo braucht es diesen Preis-Proxy oder einen News-Feed.

---

## 3. HTF-Bias [E]

```
delta    = close_now - close(now - HTF_LOOKBACK_H)
htf_bias = -sign(delta)     # Reversion: Bewegung der letzten 6–12 h umkehren
```

Ist `abs(delta) <= HTF_NEUTRAL_PTS`, ist der Bias `0` (neutral).

Der HTF-Bias filtert **nur** Setup A (Opening Continuation). Für Reversions ist der Bias immer „Richtung FP“.

---

## 4. Fair-Price-Engine (State Machine pro Session)

### 4.1 Session-Start

```
FP          = O[first bar of session]   # [E] Eröffnungskurs
FP_source   = "SESSION_OPEN"
session_open_price = FP
```

### 4.2 Tag mit geplanter High-Impact-News um 08:30 [E]

1. `FP_news = C[08:29 bar]`, also der Preis direkt vor der News.
2. Ab 08:30 gilt `FP = FP_news` mit `FP_source = "PRE_NEWS"`. Die News-Reversion (Setup C) ist ab der Kerze 08:31 aktiv.
3. Beim NY-AM-Open (09:30) wird entschieden [E / Ann]:

```
news_dir = sign(C[09:29] - FP_news)   # Seite, auf der der Preis nach der News steht
if color(09:30 bar) == news_dir and C[09:30] jenseits des Post-News-Extrems (08:30..09:29) in news_dir:
    FP = O[09:30]; FP_source = "SESSION_OPEN"    # News wichtiger als erwartet, Open ist fair
else:
    FP = FP_news   (bleibt)                       # [E] „news price is stronger than the open“
```

Sobald der Preis nach 09:30 `FP_news` berührt, wird das Open-Level verworfen [E].

### 4.3 Pre-Open-Konsolidierung [A, Default on]

Liegt eine Konsolidierung `K` aus dem Fenster `[Session-Start − 90 min, Session-Start − 1]` vor, und durchläuft der Preis innerhalb der ersten 15 Kerzen `K.level` von der Open-Seite aus, dann gilt `FP = K.level`.

### 4.4 Unerwarteter Schock [E]

1. Bei einem Schock: `state = SHOCK`, `shock_extreme` wird mitgeführt.
2. Bildet sich danach eine Konsolidierung `K` (Abschnitt 2.5): `FP = K.level`, `FP_source = "POST_SHOCK_CONSOL"`.
3. Setzt der Preis danach erneut in Schock-Richtung fort, ist der nächste FP die jüngste Konsolidierung [E].

### 4.5 Struktureller Bruch gegen den FP [E]

Ist der Preis in einer Konsolidierung `K` zwischen FP und der Gegenseite, und es kommt zu einem BOS **weg** vom FP aus `K` heraus:

```
FP = K.level   # [E] „adjust fair price to the most recent consolidation“
```

### 4.6 Session-Ende und Übergänge [E]

- Am Session-Ende verfällt der FP. Nach 11:00 ist das 09:30-Level kein Ziel mehr.
- **NY PM (14:00):** Das primäre Ziel ist `O[14:00]` für Continuation und Reversion. Öffnet 14:00 weg vom 09:30-Level, ist ein **sekundäres Reversion-Ziel** `FP2 = O[09:30]` erlaubt [E].
- Wurde NY AM durch die Drei-Verluste-Regel gestoppt, gilt `FP2` nicht; stattdessen die zuletzt etablierte Konsolidierung [E / Ann].

---

## 5. Setups

### Setup A – Opening Continuation [E]

| | |
|---|---|
| **Wann** | Schluss der ersten Kerze jeder aktiven Session |
| **Richtung** | `d = color(first bar)`. Bei Doji kein Trade. |
| **Filter 1** | `d == htf_bias` [E] |
| **Filter 2** | Die erste Kerze schließt jenseits des jüngsten ungebrochenen Swings in Richtung `d` aus dem Pre-Open-Fenster. Fallback: Close jenseits des Extrems der Vorkerze [Ann]. |
| **Größe** | `size_measure = range` oder `body` der ersten Kerze. Ist sie `<= 25`: SL 25 / TP 38, volle Größe. Sonst: SL 50 / TP 76, **halbe Größe** [E]. |
| **Entry** | Market am Open der nächsten Kerze |
| **Account** | Eval. Upgrade auf Funded, wenn die Continuation-Richtung zugleich Richtung eines anderen FP zeigt (z. B. `FP_news` ≠ Open) [A]. |
| **Extra-Continuations** | Bis `CONT_EXTRA_MAX` weitere in den ersten `CONT_EXTRA_WINDOW_MIN` Minuten, ausgelöst durch Displacement oder BOS in Richtung `d`. Optional als Limit am Close der Vorkerze [A]. |
| **Invalidierung** | Displacement oder BOS gegen `d` → Continuation-Modus aus, ab dann nur Reversion [A]. |

### Setup B – Reversion zum FP [E]

| | |
|---|---|
| **Wann** | Ab der zweiten Kerze der Session bis Session-Ende, solange die Session nicht gestoppt ist |
| **Richtung** | `d = sign(FP - C[i])` |
| **Move-away** | Seit der letzten FP-Berührung muss sich der Preis mindestens `MOVE_AWAY_MIN_PTS` entfernt haben [A] |
| **Trigger** | Displacement oder BOS in Richtung `d` an `bar[i]` |
| **Ausschluss** | `C[i]` hat FP schon erreicht oder gekreuzt [E] |
| **Abstand** | `D = abs(FP - entry_est)` mit `entry_est = C[i]` |
| **Routing** | siehe Abschnitt 6 (Eval vs. Funded, 80-%-Regel) |

### Setup C – Reversion nach erwarteten News (A+) [E]

Wie Setup B, mit `FP = FP_news`. Fenster: 08:31–11:00.

- Höchste Priorität beim Routing auf Funded Accounts, auch No-Consistency-Accounts mit großem TP.
- Erster **BOS** nach dem Post-News-Extrem: Copy-Trade auf `COPY_TRADE_APLUS_N` Accounts erlaubt [A].
- Vor 09:30 eröffnete Trades: Mit `BE_AT_0930_FOR_PREOPEN_TRADES` den SL beim Open auf Entry ziehen [E].
- Continuation direkt auf die News: **nicht** im Default [E: „mostly stay away“], optional per Flag.

### Setup D – Unerwartete News [E]

**D1 – Drift-Continuation**

| | |
|---|---|
| **Entry 1** | Am Schluss der Schock-Kerze in `shock_dir` |
| **Entry 2** | Nach der ersten Konsolidierung nach dem Schock, BOS in `shock_dir` |
| **SL** | Hinter der gegenüberliegenden Seite der jüngsten Konsolidierung vor dem Entry („largest possible stop“) [E] |
| **TP** | keiner |
| **Exit** | Erster BOS **gegen** die Position (auf Kerzenschluss), nur mit Swings, die nach dem Entry entstanden sind [E] |
| **Account** | Funded bevorzugt. Positionsgröße so, dass der SL-Dollarbetrag dem `risk_usd` des Accounts entspricht. |

**D2 – Reversion** zur Post-Schock-Konsolidierung, wenn der Preis danach weiter überschießt. Läuft wie Setup B.

### Setup E – NY PM [E]

Setups A und B mit dem FP aus Abschnitt 4.6, inklusive des sekundären Ziels `FP2`.

---

## 6. Risiko, Take-Profit und Routing (Kernlogik)

### 6.1 Prinzipien [E]

1. **Statische Dollarbeträge pro Account-Typ.** Sie stammen aus dem Offline-Optimizer (Abschnitt 9), nicht aus dem Chart.
2. **Zuerst den TP bestimmen, der SL folgt daraus:**
   ```
   SL_pts = risk_usd / (qty * pv)
   ```
   Der SL ist kein Invalidierungslevel.
3. **80-%-Regel:** Ein Trade ist nur zulässig, wenn `D >= TP_REACH_RATIO * TP_pts`.
4. **Liegt `D` zwischen zwei Rasterstufen, die niedrigere Stufe nehmen.**
5. **`SL_pts >= MIN_SL_PTS_NY`** in NY-Sessions. Sonst wird der Trade für diesen Account verworfen.
6. **Gleicher Dollarbetrag bei mehr Punkten:** weniger Kontrakte (NQ ↔ MNQ, 1 NQ = 10 MNQ).

### 6.2 Evaluation [E]

```
TP_pts, SL_pts = 38, 25    (Standard, 1 NQ bzw. 10 MNQ → +760 $ / −500 $ bei 50k)
alt:              76, 50    mit halber Kontraktzahl (gleicher Dollarbetrag)
zulässig, wenn D >= 0.8 * TP_pts  (also D >= ~30.4 bzw. ~60.8)
Signale: WEAK (Displacement) und STRONG; Continuations (Setup A) gehen hierher.
```

Für andere Kontogrößen wird `risk_usd : reward_usd = 1 : 1.5` skaliert, passend zum Max Loss / Profit Target der Firma [E].

### 6.3 Funded Accounts [E]

Jeder Account hat aus dem Optimizer: `target_usd` (Gewinnziel dieses Trades bzw. Tages), `risk_usd`, `rr_min`, `rr_max`.

```
for qty in allowed_qty_ascending (in MNQ-Schritten):
    TP_pts = snap(target_usd / (qty * pv_mnq))
    if TP_pts <= D / TP_REACH_RATIO:
        break                              # kleinste Größe → TP am nächsten an D
SL_pts = snap(risk_usd / (qty * pv_mnq))
reject if SL_pts < MIN_SL_PTS_NY
reject if not (rr_min <= TP_pts / SL_pts <= rr_max)
```

Beispiele aus dem Transkript:

- `target_usd = 6000`, `D = 50` → 6 NQ, TP 50 Pkt.
- `target_usd = 6000`, `D = 100` → 3 NQ.
- `target_usd = 3210` mit mindestens 80 % des nötigen Wegs → TP so setzen, dass genau 3.210 $ erreicht werden.

R:R-Bänder nach Account-Typ [E]:

| Typ | `rr_min` | `rr_max` | Signalpräferenz |
|---|---|---|---|
| FUNDED_CONSISTENCY | 1.0 | 1.5 (max 2.0) | STRONG bevorzugt |
| FUNDED_NO_CONSISTENCY | 1.0 | 6.0 | nur STRONG, Setup C, Setup D |
| BUFFER (wie Funded) | 1.0 | 2.0 | Ziel fix, z. B. 1.000 $ / Tag [A] |
| LIVE | per Optimizer | per Optimizer | Bonus-Guard (Abschnitt 7) |

### 6.4 Account-Auswahl pro Signal

```
candidates = accounts with status ACTIVE, no open position, not daily_done,
             signal_strength allowed, setup allowed
if signal is FUNDED-eligible and funded_open_same_dir < FUNDED_SAME_DIR_EXPOSURE_CAP:
    pick funded account whose computed TP_pts is closest to (and <=) D / TP_REACH_RATIO,
    priority: Setup C > STRONG > WEAK
else:
    pick next eval account (round robin)
1 Signal = 1 Account   (Ausnahme: COPY_TRADE_APLUS_N bei Setup C, erster BOS)
```

### 6.5 Layering [E]

Weitere Signale in dieselbe Richtung, während Positionen offen sind, gehen an **andere** Accounts. Es gibt keinen Pyramiden-Aufbau innerhalb eines Accounts.

### 6.6 Re-Trading eines Accounts am selben Tag [A]

Ein Account bleibt aktiv, bis das Tagesziel erreicht ist oder der Account verloren ist.

```
next_target_usd = daily_target_usd + abs(realized_pnl_today if < 0 else 0)
                  - (realized_pnl_today if > 0 else 0)
daily_done = realized_pnl_today >= daily_target_usd
```

---

## 7. Account-Modell und Guards

```
Account {
  firm, type ∈ {EVAL, FUNDED_CONSISTENCY, FUNDED_NO_CONSISTENCY, BUFFER, LIVE},
  size, balance, start_balance,
  drawdown_type ∈ {EOD_TRAILING, INTRADAY_TRAILING, STATIC}, max_loss, dd_floor,
  profit_target, consistency_pct, min_winning_days, payout_rules,
  daily_target_usd, risk_usd, rr_min, rr_max,
  live_transfer_threshold, live_account_ev,
  bonus_threshold, bonus_pending,
  status ∈ {ACTIVE, DONE_TODAY, PASSED, BLOWN, PAUSED}
}
```

Guards:

1. **Live-Transfer-Guard [E]:** Wenn `balance - start_balance + next_target_usd > live_transfer_threshold - buffer`, dann den Trade nicht nehmen oder das Ziel kappen. Beispiel: Schwelle 15k, Balance zwischen 13k und 14k halten.
2. **Bonus-Guard [E]:** Auf einem Live-Account mit `pnl >= bonus_threshold` (z. B. 4.500 $ auf einem 150k-Live) → `PAUSED`, bis der Bonus gutgeschrieben ist.
3. **Neue Firma [E]:** Maximal 1 aktiver Account, bis der erste Payout erhalten ist. Danach bis zu 5.
4. **Drawdown:** Kein Trade, wenn `balance - SL_usd < dd_floor` (harte Firmenregel).
5. **Consistency:** Der Trade wird verworfen, wenn ein voller TP-Treffer `consistency_pct` verletzen würde (größter Tag > x % des Gesamtprofits).

---

## 8. Sessions, Stopps und Ausführung

### 8.1 Session-Steuerung

- Nur Entries im Fenster `[session_start, session_start + 90 min)` [E].
- **Drei-Verluste-Regel [E]:** Drei Reversion-Trades (Setups B, C, D2) in Folge verloren, gezählt in der Reihenfolge ihres Schließens → keine neuen Entries mehr in dieser Session. Continuation-Verluste zählen nicht mit.
- **Geplante Rede bzw. Event innerhalb der Session [A]:** Ab dem Event-Zeitpunkt keine neuen Entries.
- **Offene Positionen nach Session-Ende:** bis TP oder SL halten (er hält) [A]. Optional `FLATTEN_AT`.
- **Override [A]:** Ist die Mehrheit der Accounts noch nicht `DONE_TODAY`, Entries bis `SESSION_EXTEND_UNTIL` erlauben. Default aus.

### 8.2 Order-Management

- Bracket-Order (OCO) mit TP und SL bei Entry.
- Break-even: optional `BE_ON_FP_TOUCH` (SL auf Entry, sobald `H/L` den FP berührt und der TP jenseits des FP liegt) [A].
- Kein manuelles frühes Schließen vor dem TP [E].
- Teilfüllungen bei Limits loggen; das Ausführungsmodell sollte Slippage beim Open berücksichtigen [A].

---

## 9. Offline-Optimizer (Prop-Firm-Simulation) [E]

Der Optimizer liefert pro Firma und Account-Typ die statischen Werte `risk_usd`, `target_usd` und `daily_target_usd`.

1. Die Trade-Liste der Strategie (aus dem Backtest) auf **N = 10 parallele Accounts** mit den exakten Firmenregeln verteilen (Trailing DD, Consistency, Winning Days, Payout-Caps, Live-Transfer).
2. **Grid-Suche** über statische `(SL_usd, TP_usd)`-Kombinationen (Raster aus `SL_GRID_PTS` × `TP_GRID_PTS` × Kontraktzahl).
3. Für Evals:
   ```
   pass_rate          = passed / attempted       # Stichprobe >= 50 Evals
   cost_per_funded    = eval_price / pass_rate
   ```
4. Für Funded Accounts:
   ```
   EV_funded = mean(sum(payouts über die Lebensdauer))   # nur Payouts zählen
   ```
   Live-Übergang einrechnen: `EV_live ≈ Startkapital des Live-Kontos` [E], plus Bonus.
5. **Auswahl:** die Kombination mit maximalem `EV_funded`.
6. **Gate:** Die Strategie ist nur zulässig, wenn `EV_funded > cost_per_funded`.
7. **Risk of Ruin:**
   ```
   p_payout = pass_rate * payout_rate
   RoR      = (1 - p_payout) ^ floor(bankroll / eval_price)
   ```
   Ziel ist `RoR < 5 %` [E].

Bekannte Firmenwerte aus dem Transkript (zur Validierung):

| Wert | Quelle |
|---|---|
| 50k-Eval: Max Loss 2.000 $, Ziel 3.000 $, 50 % Consistency, Kosten ~100 $ | E |
| Pass-Rate in seinem Beispiel: ~25–33 % | E |
| Live-Bonus Lucid / Tradeify: +4.500 $ auf 150k → 4.500 $ Bonus | E |

---

## 10. Hauptschleife (Pseudocode)

```python
on_bar_close(i):
    update_swings(i); update_consolidations(i); update_atr(i)
    sess = active_session(t[i])
    if sess is None: manage_open_positions(i); return

    if is_first_bar(sess, i):
        init_fp(sess, i)                      # 4.1 / 4.2 / 4.3
        try_setup_A(sess, i)                  # Continuation
        return

    fp_engine.update(i)                       # 4.2 (09:30-Entscheid), 4.4, 4.5
    if detect_shock(i): try_setup_D1(i)

    if sess.stopped or not in_entry_window(sess, i) or event_blackout(i):
        manage_open_positions(i); return

    if cont_mode_active(sess, i): try_extra_continuation(sess, i)

    sig = signal(i, d=sign(fp - C[i]))        # BOS -> STRONG, Displacement -> WEAK
    if sig and moved_away(fp, i) and not crossed(fp, C[i]):
        setup = 'C' if fp_source == 'PRE_NEWS' else 'B'
        route_and_submit(sig, setup, D=abs(fp - C[i]))   # Abschnitt 6

    manage_open_positions(i)                  # D1-Exit bei Gegen-BOS, BE-Regeln

on_trade_closed(tr):
    update_account(tr)
    if tr.setup in {'B','C','D2'}:
        sess.consec_rev_losses = sess.consec_rev_losses + 1 if tr.pnl < 0 else 0
        if sess.consec_rev_losses >= 3: sess.stopped = True
```

---

## 11. Offene Punkte / nicht 1:1 codierbar

| Punkt | Problem | Umsetzung |
|---|---|---|
| Docht-Rejection-Entries, „halbes Displacement“ (Live-Tag 1) | diskretionär | optional als `ENTRY_WICK_REJECTION` mit Docht/Range-Schwelle; Default aus |
| „24 after“-Volumenkerze als FP | diskretionär | optional: Kerze mit `V > k * avg_V` in den ersten 10 Minuten → FP = deren Open |
| Tweets / Reden | brauchen einen News-Feed | Preis-Proxy (Abschnitt 2.6) |
| Trend-Tag-Erkennung („nicht weiter gegen den Trend“) | teilweise diskretionär | durch die Drei-Verluste-Regel und Abschnitt 4.5 abgedeckt |
| Asia- und London-Startzeiten | nicht genannt | Parameter |
| Opening-Candle-Größe: Range oder Body? | nicht eindeutig | Parameter |
| Optimale TP/SL-Werte pro Firma | sein Dashboard ist nicht öffentlich | Optimizer (Abschnitt 9) |
| PM-Ziel 09:30 vs. „nach 11 Uhr kein Ziel“ | widersprüchlich | `FP2` als Flag |
| Ein Trade pro Account vs. Re-Trading bis zum Tagesziel | beides genannt | sequentiell erlaubt (6.6), nie zwei Positionen gleichzeitig |

---

## 12. Empfohlene Backtest-Varianten

1. **Basis:** Setups A, B, C; Evals 25/38; Drei-Verluste-Stop; nur NY AM.
2. **+ NY PM** mit und ohne `FP2`.
3. **+ Setup D** (Schock-Proxy).
4. **Sensitivitäten:**
   - `HTF_LOOKBACK_H` ∈ {6, 8, 10, 12}
   - `TP_REACH_RATIO` ∈ {0.75, 0.8, 0.9}
   - `CONSOL_*`
   - `CONT_BIG_CANDLE_MEASURE`
   - `SWING_STRICT`
5. **Bewertung immer über die Prop-Simulation** (Pass-Rate, `EV_funded`, `cost_per_funded`). **Nicht** über eine Live-Equity-Kurve [E].
