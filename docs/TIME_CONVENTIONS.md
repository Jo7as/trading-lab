# Time and cost conventions

Two measurement spaces decide whether a study's numbers mean anything: the
**clock** (which bar belongs to which session/day) and the **cost space** (which
price series the move is measured on, and what a leg costs). Both are written down
once here and enforced by helpers; a study that does not declare them is not
comparable to the others.

## 1. Three clocks — and only one of them is Berlin

| layer | convention | enforced by |
|---|---|---|
| data index | **UTC**, bar-open timestamp | `trading_lab/minute_data.py` (bid+ask join), `mr_common.ensure_utc` |
| session windows | **exchange-local** time of the home market, DST-correct | `orb_2r.MARKETS`, `data_audit.SESSIONS`, `pullback.SESSIONS`, `momentum.SESSIONS` |
| plan documents | written in local wall clock (London/New York/Frankfurt/Sydney) | `docs/MULTI_STRATEGY_DUKASCOPY_PLAN.md` §1.1/§1.3 |

The plan's German-time session table is a *notation*; `orb_2r.py` says so itself
("Exchange-local equivalents of the German times in the plan"). Nothing is tested
or ranked in Berlin time — except GER40/Xetra, where Berlin **is** the home market.
On a Berlin clock the windows look like this: London 08:00 local = 09:00 Berlin,
New York 09:30 local = 15:30 Berlin.

Why local anchoring matters: the EU and US switch to daylight saving two to three
weeks apart (`MULTI_STRATEGY_DUKASCOPY_PLAN.md` §1.1). A Berlin-anchored US session
would shift by an hour twice a year; an exchange-local one never does.

## 2. Instrument → session map

| instrument | timezone | session | flat |
|---|---|---|---|
| US100, SP500, US30 | `America/New_York` | 09:30 | 15:55 |
| XAUUSD, XAGUSD, COPPER, BRENT | `America/New_York` | 08:20 | 16:55 |
| GER40 | `Europe/Berlin` | 09:00 | 17:25 |
| UK100 | `Europe/London` | 08:00 | 16:30 |
| JP225 | `Asia/Tokyo` | 09:00 | 14:55 |
| FX majors (SMC) | `America/New_York` (session logic) | killzones by NY hour | 16:45 |

Newer strategies expose the session as a choice: `pullback`/`momentum` use
`london` = `Europe/London` 08:00–16:30 and `ny` = `America/New_York` 09:30–15:55.
Adding a session means adding it to the module's `SESSIONS` dict — never a hard
coded UTC offset.

## 3. The daily roll

The CFD/FX roll is **17:00 New York = 21:00 UTC (EDT) / 22:00 UTC (EST)**. It is a
UTC event, so every roll mask is built from UTC hours:
`data_quality.ROLL_HOURS_UTC` + `data_quality.roll_mask()` (used by
`clean` and `audit`). Applying the hours in the *asset* timezone — which the code
did until 2026-10-26 — blanks 01:00–04:59 UTC for a New York asset and leaves the
real roll in the data, which is how a fabricated jump survived the quality gate.

Two known deviations, both deliberate for now:

* `data_quality.clean` masks a **generous** window (21:00–00:59 UTC) so no roll
  bar slips through; narrowing it to `(21, 22)` needs a re-measurement of where
  the corrupt bars actually sit.
* `data_audit.build_study_frame` masks a **narrow** NY window (17:00–17:05) plus
  the session close. Both paths are correct in *space* (UTC/NY), they differ in
  width — a study should use `build_study_frame` and say so.

## 4. Cost space

Everything is measured in the **mid** space: the move (the R numerator, `E|Δmid|`)
is mid-to-mid, and a leg costs half a spread when it crosses the book and nothing
when it rests in it:

| entry \ exit | limit / TP | market / stop / time |
|---|---|---|
| **limit** | 0.0 | 0.5 |
| **market** | 0.5 | 1.0 |

(units of the **full** spread; `trading_lab/cost_gate.py: leg_cost` /
`round_trip_spread`, tests in `tests/test_cost_gate.py`.)

Consequences:

* A resting entry **and** a resting exit are free in this model — which is exactly
  why a limit strategy must report its **fill rate** and adverse selection
  (`studies/h010_fill_rate.py`); the spread it saves is paid for in fills.
* **Do not mix the spaces.** If a study measures the move on the **bid** series
  instead, the same trade needs a different charge: a long pays 1.0 spread for a
  market entry and 0 for the exit, because both of its exits are already bid-side.
  Charging a mid-space cost on a bid-space move double-counts. `run_h001`,
  `mfe_mae` and `first_passage_grid` used to charge two full spreads for one round
  trip; per `docs/AEP_STAGE0_K3.md` that is enough to flip 4/26 candidates into
  15/26.
* Strategy engines that model bid/ask themselves (`orb_2r`, `mean_reversion`,
  `session_range_fade`, `pair_spread`, `pullback`) must **not** be netted again —
  the spread is already in the fills.

## 5. Checklist for a new study

1. State the **space**: mid or bid? Which spread (per bar, per session, median)?
2. Use `data_audit.build_study_frame` for the frame (roll + session close masked),
   or say why not.
3. Use `cost_gate.round_trip_spread(entry, exit_)` — never a hand-written
   multiple of the spread.
4. Anchor sessions in the **exchange** timezone via the module's `SESSIONS`.
5. Per-day/session slices must index the **full** frame
   (`mr_common.day_slices`); keep the test on a **full 24 h frame**.
6. A forward window must not run over the session end into the night — the gap is
   not "the market's reaction to the setup".
7. Declare the power: with 400–1500 trades a real Sharpe-1.0 edge is detected only
   36–63 % of the time (`studies/pipeline_power.py`).
