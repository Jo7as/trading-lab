# Crypto order flow (BTC / SOL perps) - REJECTED (2026-10-29)

**Idea:** trade intraday on order flow aggregated over several exchanges (taker-signed trades),
grounded in microstructure research, optionally combined with SMC levels and killzones; target
a prop account (1-2 % per month).

**Verdict:** the information is real but too small for our costs and speed. No variant comes
close to the ~11 bp taker round trip (Bybit), let alone to Sharpe ~1.5.

| Test (2024, train only) | Result |
|---|---|
| Kim & Hansen quarter-hour opening imbalance (replication) | same sign as the paper; BTC +2.4 / SOL +4.5 bp per unit OI at 4 h, not significant on 10 months; placebo clean on BTC |
| H3/H5 extreme openings, market vs limit entry | 64 cells, scattered t ~ 2 (chance level); unfilled limits would have been the winners (adverse selection) |
| H8 SMC levels x flow (day/week H-L, swings, EQH/EQL, order blocks, volume profile, killzones) | nothing positive net; real levels = random levels; flow does not improve the SMC reading |
| VPIN as a risk filter | sorts jumps only through volatility; adds nothing beyond realised vol |
| H2 Binance + Bybit aggregation | not better than Binance alone |

**Reusable pieces** (tested, kept): `trading_lab/adapters/crypto_trades.py` (Binance/Bybit dumps ->
10-s flow bars), `trading_lab/orderflow.py` (imbalance, CVD, VPIN incl. BVC, Corwin-Schultz,
Kyle lambda, volume profile; provenance table in the research notes), `trading_lab/level_flow.py`,
`trading_lab/hac.py`, crypto cost profiles in `trading_lab/costs.py`.

**Not tested:** Hyperliquid's publicly announced TWAPs (no free history; public signal on a
venue no prop firm executes on). Longer history (2021-2023) was not loaded - more data would
narrow the estimates, not lift them above costs.

**Details:** `docs/FINDINGS.md` (2026-10-29 crypto entries); research notes in `docs/research/`
(git-ignored: plan, theory, papers).
