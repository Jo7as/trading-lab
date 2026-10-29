# S-007 Trendfolge über alle Anlageklassen (TSMOM) — BESTÄTIGT, Baustein im KORB

* Spec: [`docs/specs/tsmom_multi_asset.yaml`](../specs/tsmom_multi_asset.yaml) (Moskowitz/Ooi/Pedersen 2012),
  Code `trading_lab/strategies/tsmom_multi_asset.py`, Tests `tests/test_tsmom_multi_asset.py`,
  Lauf `studies/tsmom_check.py`, Korb `configs/korb/KORB.yaml`.
* Regel: je Markt Vorzeichen der 12-Monats-Rendite, Größe nach EWMA-Vola, gleiches Risiko je
  Anlageklasse (Krypto halb), Portfolio auf Ziel-Vola skaliert, wöchentlich (Mi 15:00 NY), 25 %-Puffer.
* Prüfung (Literatur-Spur, `docs/PRUEFABLAUF.md`): Training 2004-2024 Sharpe 0,79 brutto / 0,51 netto
  (Formel-Swaps), t 2,37, Placebo +0,004 R vs. +0,150 R, AQR-Korrelation 0,71; **Holdout
  2025-01..2026-10: +13,5 % netto, Sharpe 0,87 - bestanden.**
* Echte FundedNext-Swaps (MT5-Logger, 2026-10-29): netto Sharpe **0,64**.
* Instant-Konto 6 % mitlaufend: 3 % Ziel-Vola -> ~2,1 % p.a., max DD -7,4 %, 96 % Überleben 1. Jahr,
  Ø Auszahlung ~2,8 % des Kontos pro Jahr (Training).
* Offen: Spreads/Slippage auf einem Konto mit Live-Feed (Demo-Feed eingefroren); Swaps für
  Anleihen/Rohstoffe/Krypto; Papier-/Demo-Forwardtest.
