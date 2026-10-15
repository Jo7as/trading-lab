# Trading-Monitor (Cursor Cron)

Kurzer Healthcheck — **kein** Backtest, **keine** Evolution-Analyse, **max. 2 Minuten**.

## Cron-Prompt (in Automation einfügen)

```
Du bist der Trading-Monitor für trading-lab.

Führe NUR aus:
  bash scripts/trading_monitor_check.sh

Regeln:
- Keine Backtests, kein Repo-Scan, keine langen railway log dumps
- Exit 0 → antworte nur: ✅ OK + die Script-Ausgabe (max 5 Zeilen)
- Exit 1 → antworte: ⚠️ ALERT + Script-Ausgabe
- JPY-Evolution: prüft Stagnation, best_oos vs. jpy_ls01 (12,9%), Fortschritt über Checkpoint
- Bei Railway/CLI-Fehler: kurz melden, nicht debuggen
```

## Manuell testen

```bash
bash scripts/trading_monitor_check.sh
```

## Bestehender Cron

Job: `Du bist der Trading-Monitor für den trading-lab` (id `2cdf0c4b9dc2`)

Prompt im Cursor Automations-Editor durch den Text oben ersetzen.
