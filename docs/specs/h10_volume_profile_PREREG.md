# H10: Volume-profile levels on crypto perps - PRE-REGISTRATION

Written 2026-10-29, before any profile level was computed on data. Measurement only, no
strategy. Code: `trading_lab/volume_levels.py`, `studies/crypto_volume_profile_study.py`,
`tests/test_volume_levels.py`. Nothing below (window, bin count, threshold, horizon, control,
verdict rule) is changed after the first run on 2024.

## Why

Volume profile (POC / value area / HVN / LVN) has **no academic specification** - window, row
size and value-area share are practitioner conventions (Steidlmayer / CBOT Market Profile,
Dalton). The often quoted "price returns to the prior POC ~80 % of the time" has no source.
What the literature does have: support/resistance levels coincide with order-book depth peaks
(Kavajecz & Odders-White 2004, RFS) and stop / take-profit orders cluster at salient prices
(Osler, NY Fed Staff Report 125, 2003). If high-volume prices mark where size rests, they could
attract price (magnet) or stop it (hold). Crypto perps are the testbed because their volume is
real and taker-signed (Dukascopy FX/CFD volume is tick count). H8 already showed the previous
UTC day's POC/VAH/VAL do not hold better than random levels at first touch - **that test is not
repeated** (no HOLD cells for the day window).

## Data

* BTCUSDT + SOLUSDT perps, Binance USD-M aggTrades -> 10-s taker-signed bars
  (`adapters/crypto_trades.load_bars`, `allow_download=False`), resampled to **1-min** bars
  (`orderflow.resample`) for profiles, touches and outcomes. Profiles place each 1-min bar's
  notional at its VWAP (`orderflow.volume_profile`); the error is below one row at the row sizes
  used (window range / 50).
* **Discovery: 2024** (cached). **Confirmation: 2023**, run once its cache is complete, reported
  separately. Holdout 2025+ is not opened.
* Missing days stay empty on the grid; events whose window or horizon touches a missing day are
  dropped.
* Costs (reported fade returns only): `costs.crypto_round_trip_bp(market, "bybit_perp")`.

## Profile windows (9) - each instance gives levels known at its end

| window | definition | instances |
|---|---|---|
| `n96` / `n288` / `n672` | last 96 / 288 / 672 bars of 15 min (1 / 3 / 7 days) | rolling, a new instance every 1/4 of the window (6 h / 18 h / 42 h) on a UTC grid |
| `day` | UTC day | daily, known 00:00 UTC next day |
| `week` | Monday 00:00 - Monday 00:00 UTC | weekly |
| `asia` | Tokyo 09:00-15:00 JST | daily, known at session end |
| `london` | London 08:00-16:30 local (DST-aware) | daily |
| `ny` | New York 09:30-16:00 local (DST-aware) | daily |
| `leg` | between two consecutive confirmed **external** swings of `market_structure.analyze` on 15-min bars (low -> high = up leg, high -> low = down leg, pooled) | known at the end of the 15-min bar that confirms the second swing |

Sessions are built with `mr_common.day_slices` on the full frame.

## Levels (per instance)

* Rows: 50 equal rows over the window's [low, high]; `orderflow.profile_levels` for **POC, VAH,
  VAL** (70 % value area, CBOT growth rule); level price = row centre.
* **HVN**: local maximum of the profile smoothed with weights [1,2,3,2,1]/9, strict maximum over
  +-2 rows, smoothed volume >= the mean row volume, not the POC row.
* **LVN**: local minimum of the smoothed profile over +-2 rows, inside [VAL, VAH],
  smoothed volume <= 0.5 x mean row volume.
* **Naked POC** (`naked`): the POC of a UTC day that was **not traded through during the next
  UTC day** (low > POC or high < POC all day) - naked from 00:00 UTC two days after its own day;
  one event per naked POC, at that moment.

## Distance unit and filter

ATR(14) of 1-h bars (`indicators.atr`), known at the bar's end. A level is used only if
0.25 <= |level - price| / ATR1h <= 6 at the moment it becomes known (price = last 1-min close);
levels inside that bar's range are skipped.

## Test (a) MAGNET - primary

From the moment the level is known (`t0`, close price `p0`), side `s = sign(level - p0)`:
hit within h in {4 h, 12 h, **24 h** (primary)} iff the 1-min high (s = +1) / low (s = -1)
reaches the level in (t0, t0 + h].

* **Control (primary): random level at the same distance and side, at another time.** For each
  real event draw R = 5 times `t'` from the same market's 1-min closes of the same calendar
  month, at least 24 h away from `t0`; control level = p(t') + s x dist_ATR x ATR1h(t'); the same
  hit rule. Effect = hit_real - mean(hit_control), per event.
* Control (secondary, reported): **mirror** - same time, same distance, opposite side.
* The same-time / same-side level would be the same price, so it cannot be a control.

## Test (b) HOLD

Levels from all windows **except `day`** (H8), plus `naked`. First touch within 24 h of `t0`
(approach side d = s). From the touch bar: **break** = high (d = +1) / low (d = -1) goes
>= 0.5 ATR1h beyond the level, checked from the touch bar on; **bounce** = price comes back
>= 0.5 ATR1h on the approach side, checked from the bar AFTER the touch; whichever comes
first within 4 h. Both in the same bar or neither in 4 h -> unresolved, excluded.
hold rate = bounce / (bounce + break).

* **Control:** the same levels moved by a random +-0.15..0.5 % (`level_flow.control_levels`,
  seeds 1-3), same timing and rule. Effect = hold_real - hold_control (cluster-by-day OLS,
  `level_flow.cluster_diff_t`).
* Reported, no weight: fade return (-d) from the touch bar's close to 4 h later, gross and net.

## Test (c) Interactions (on HOLD events, levels pooled across windows, per level type)

* **SMC confluence:** a profile level is confluent if, at `t0`, an active order-block edge
  (facing edge, as in `level_flow.structure_levels`), an untaken EQH/EQL pool, or an unbroken
  external swing (15-min `market_structure`) lies within **0.1 x ATR(14) of 15-min bars**.
  Effect = hold(confluent) - hold(not confluent), real; and the same split on the controls.
* **Flow at the touch:** taker imbalance over the 5 one-minute bars ending with the touch bar;
  "into the level" iff d x flow >= 0.15 (H8 threshold). Effect = hold(flow in) - hold(not),
  real vs control.

## Statistics

Per cell: mean effect, t clustered by UTC day (`level_flow.day_clustered_t`) and a Newey-West
t of the mean of the per-day effects (Bartlett, 5 lags, the estimator of `hac.newey_west_ols`) - windows and horizons overlap across days.
The verdict uses the **smaller |t|** of the two.

## Cells (counted up front)

* MAGNET: 9 windows x 5 level types + naked = 46 cells x 3 horizons = 138 (46 at 24 h decide).
* HOLD: 8 windows x 5 types + naked = 41 cells.
* Interactions: 5 types x 2 (confluence, flow) = 10 cells.
* **97 deciding cells per market-pool**; BTC and SOL are pooled for the verdict, reported apart.

## Verdict (fixed now)

A cell **"confirmed"** iff
1. 2024 (discovery): |effect| >= 5 percentage points and |t| >= 3.0 (roughly Bonferroni for
   ~100 cells),
2. 2023 (confirmation): same sign, |t| >= 2.0, and effect >= half the 2024 effect,
3. same sign in BTC and SOL separately on 2023.

Signs both ways count (a level may repel). Verdict on the question: **volume-profile levels
matter on crypto** iff at least one MAGNET or HOLD cell is confirmed; otherwise "no better than
random levels". A confirmed cell is a candidate for a separate spec (user decides), never a
strategy by itself. Raw base rates (e.g. "prior POC reached within 24 h") are reported next to
the control so the practitioner's 80 % claim can be read directly.
