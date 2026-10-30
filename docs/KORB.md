# Der Baustein-Korb

Ziel: viele kleine, echte, wenig korrelierte Kanten statt einer großen. Maschinenlesbar in
[`configs/korb/KORB.yaml`](../configs/korb/KORB.yaml); ein Baustein kommt nur hinein, wenn er alle fünf
Stufen inklusive Holdout bestanden hat. Stand 2026-10-30. Forschung, keine Anlageberatung.

| Baustein | Idee | Rhythmus | Sharpe netto (Training) | Holdout ab 2025 |
|---|---|---|---|---|
| **S-007** Trendfolge (`tsmom_multi_asset`) | 26 Märkte, long/short nach 12-Monats-Rendite, Größe nach Vola, Budget je Anlageklasse | wöchentlich, Mi 15:00 NY | 0,64 | +13,5 % (Sharpe 0,87, t 1,2) bei 8 % Vola |
| **S-010** Aktienprämie (`vol_managed_equity`, Variante `vola`) | 6 Indizes long, Größe umgekehrt zur Vormonats-Schwankung | monatlich, 1. Handelstag | 0,54 | +22,3 % (Sharpe 1,30, t 1,8) |

**Korb (gleiches Risiko):** Training Sharpe ~0,78, Korrelation +0,18. Bei 4 % Ziel-Schwankung im Training
~3 % pro Jahr (~0,25 % pro Monat), max. Drawdown ~-8 %; Holdout +11,5 % in 22 Monaten, max. Drawdown -5,1 %.
Rund 20 Orders pro Monat. Das Ziel von 1-2 % pro Monat erreicht der Korb nicht.

**Grenzen:** beide bestanden über die Literatur-Spur (`docs/PRUEFABLAUF.md`), nach den normalen Hürden nicht;
Holdouts einzeln nicht signifikant (t < 2), 2025/26 waren günstige Jahre; vier von sechs Index-Swaps sind
geschätzt, FundedNext-Spreads und -Slippage ungemessen (S-010 fällt mit teuren Swaps auf Sharpe 0,27);
noch kein Forward-Test.

**Nachrechnen:**

    uv run python studies/tsmom_check.py --referenz --broker artifacts/broker_costs/fundednext_demo.json
    uv run python studies/vol_managed_check.py

Details: Specs in `docs/specs/`, Ergebnisse in `docs/FINDINGS.md`, Paper in `docs/PAPERS.md`.
