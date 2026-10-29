# Prop-firm rules used by the simulator (research 2026-10-29)

Presets in `trading_lab/prop_sim.py` (`FIRMS`), checked per strategy with
`studies/prop_firms.py <strategie> <markt> '<params>'` (train only) and in
`studies/optimize_report.py`. Rules change - **check the firm's own page before buying**;
only FundedNext's numbers come from the official page (`verified=True`). Fees and payout
splits depend on the account size and are NOT filled in: EV = payout - fee.

| Preset | Type | Max loss | Daily | Other | Source quality |
|---|---|---|---|---|---|
| `fundednext_instant` | instant, no target | 6 % trailing, stops at the initial balance | none | no consistency (one review says 40 %) | official page |
| `the5ers_hyper_growth` | 1 step, 10 % target | 6 % static | 3 % = pause until the next day (one review: termination) | split 50 % at start | reviews |
| `fundingpips_zero` | instant | 5 % trailing (mechanics unclear) | 3 % hard | 15 % consistency, 7 trading days, weekend maybe banned | reviews, contradictory |
| `ftmo_1step` | 1 step, 10 % target | 10 % EOD trailing | 3 % hard | best day <= 50 % of profit | reviews |
| `ftmo` | 2 step 10 % + 5 % | 10 % static | 5 % | 4 trading days per phase | reviews |

Modelling limits: daily P&L only (an intraday trailing floor is approximated end of day);
a daily "pause" caps the day at the limit; consistency = best day of the cycle <= share x
cycle profit, otherwise the payout waits; payouts every 14 days reset the balance (and the
trailing peak) to the start.

Consequences for strategy design: instant accounts allow ~0.25-0.5 % risk per trade, not
1 %; consistency rules punish strategies that earn their money on a few big days (trend
following) - FundingPips' 15 % rule blocked 92 % of the payouts of S-006 (train).

Sources: fundednext.com/cfds/stellar-instant; propfirmbriefing.com, tradeventure.app,
propvator.com (FundedNext); proptradingvibes.com, propjournal.net, propfirmbridge.com,
lunefi.com (The5ers); proptradingvibes.com, marketsxplora.com, lunefi.com (FundingPips);
tradetanto.com, propnavi.io, myforexfirms.com (FTMO).

## Crypto prop firms (research 2026-10-29, for the order-flow plan)

Context: `docs/research/CRYPTO_ORDERFLOW_PLAN.md`. Order-flow signals over hours carry only a few
bp (see `CRYPTO_MICROSTRUCTURE_THEORY.md` ch. 5-6), so **where the order executes** matters more
than the headline rules. **All sources are reviews/affiliate sites, none verified on the firm's
own page; no presets in `prop_sim` yet.** Almost every funded account is **simulated**: the
firm mirrors prices, and the question is only whose prices and spreads.

| Firm | Instant? | Execution / prices | Drawdown / daily | Split / payout | Notes |
|---|---|---|---|---|---|
| HyroTrader | **no** (Zero-Step = 1 phase, 10 % target, 1 day minimum) | own **Bybit sub-account** via API, real Bybit book (USDT perps, BTC + SOL); funded stage simulated, real capital for selected traders | static max loss 5-10 %; daily 3 % (Zero-Step), reviews contradict: "trailing intraday" in Standard mode, Swing option | 80 % -> 90 %, USDT/USDC within 24 h, min. 100 $ | best fit for order flow: the same book that we measure; bots allowed with your own strategy; company is being restructured (per one review) |
| Breakout (Kraken) | no (1-step) | firm routing, "order may never reach an exchange" | static 3-6 %, daily 3 % | 80-90 %, USDC on demand | serious owner; for order flow the execution prices are unclear |
| FundedNext Stellar Instant | **yes** | **CFD** (9 crypto pairs) | 6 % trailing, no daily | 70-80 %, on demand / 14 d | already a preset (`fundednext_instant`); CFD spread kills hours-scale order flow |
| FXIFY Instant | yes | CFD-like (80+ crypto symbols) | not stated | up to 90 %, 14 d | same CFD problem |
| HydraFunding | yes | unclear | unclear | unclear | fee non-refundable; only mentioned in Kraken's comparison |
| Crypto Fund Trader | yes (instant route) | Bybit interface, simulated + MT5 CFDs | trailing until +6 %, then static 6-10 % | 50-90 % | **on FINMA's warning list since 2024-08 -> avoid** |

Conclusion: **no instant firm executes on a real crypto order book.** For order flow that means
either (a) HyroTrader (not instant, but Bybit book, Bybit fees approx. 5.5 bp taker / 2 bp maker)
or (b) an instant CFD firm, only if the edge survives that firm's spread (measure it then; it
is not assumed). We start without money: research on free data, firm choice once the study
shows an edge. Before buying: check the official pages, then add a `verified=True` preset.

Sources: cryptoslate.com/prop-firms/crypto-trading, cryptoslate.com/prop-firms/hyrotrader-review,
proptradingvibes.com/prop-firms/hyrotrader, cryptoslate.com/prop-firms/breakout-review,
kraken.com/learn/breakout-vs-hydrafunding, coincodecap.com/best-crypto-prop-firms-with-instant-funding.
