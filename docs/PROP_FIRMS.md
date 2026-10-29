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
