# H12: Mining token combinations before London / NY moves - PRE-REGISTRATION

Written 2026-10-30. **Status: released by the user in chat on 2026-10-30 (defaults of the open
points accepted: top-decile hour, 60 min, Asia = Tokyo 09:00, one mismatch, thresholds as written).** Measurement only, no
strategy. Planned code: `trading_lab/pattern_mining.py` (+ tests), `studies/h12_mining_study.py`.
Nothing below is changed after the first run on the mining years.

## Why

The user's idea, after H11: do not look at one moment and one move type, but **search on all
levels** - collect every move during London / New York, look at **everything from the start of
the Asia session up to the move**, on several timeframes at once, and keep the token
**combinations** that show up conspicuously often before moves. Combinations may be similar
rather than identical, and the useful ones are **frequent** (several times a day to 2-5 times a
week), so they can become a block of the basket (`configs/korb/KORB.yaml`).

Lessons carried over from H11 (`docs/FINDINGS.md`, 2026-10-30): (1) the shape of the last bars
before the London / NY open carried nothing, the signal was recent volatility - so every
combination is compared with a **volatility- and clock-matched control**, not with the overall
base rate; (2) a minimum count per combination; (3) a wide search finds hundreds of
"conspicuous" combinations by chance - so mining, validation and confirmation are separate years.

Two tracks, both measured, verdicts separate:

* **Track A - intraday moves:** at every 15-min point in London / NY, which combinations of tokens
  from Asia start up to now come before a 60-min move? (sections below up to "Track B")
* **Track B - session sentences (DNA search):** every day is a sentence of session words
  Asia -> London -> NY (and the days before). Which words / motifs in Asia say what London or
  New York will do, and which Asia + London words say what New York will do - similar, not
  identical. (section "Track B")

## Data

* Dukascopy 1-min bid/ask, mid prices, `data_audit.build_study_frame`, exchange-local clocks.
* Markets (audited): US100, SP500, US30, GER40, UK100, XAUUSD.
* **Mining 2016-2019, validation 2020-2021, confirmation 2022-2024** (GER40 from 2018, UK100
  from 2019 - they only contribute where they have data). **Holdout 2025+ not opened.**

## Decision points and the move (the outcome)

* **Decision points:** every 15-min bar close from **08:00 London to 15:00 New York** (the London
  session, the overlap and New York up to one hour before the cash close), every market.
* **Move up / move down:** from the decision close `p_t`, within the next **60 min** (mid,
  1-min bars), price reaches `p_t + m` before `p_t - m` (up) or `p_t - m` before `p_t + m` (down);
  stop wins nothing here, a bar touching both counts as neither. `m` = **threshold x ATR(14) of
  1-h bars** at t, threshold = the 90th percentile of max(|60-min excursion|) / ATR1h on the
  mining years of that market (frozen). So "a move" is a top-decile hour, the same rarity on
  every market.
* Reversal and breakout are not separate outcomes: they are combinations of context tokens
  (`new high/low`, `quiet`) with a move, and the search finds them like any other combination.

## Items - the tokens describing the state at t ("all levels")

**Phase tokens - from the start of Asia up to t** (the user's window). Asia starts Tokyo 09:00
JST = 00:00 UTC (no DST in Japan). Every **completed** phase before t gives one token, the
running phase gives one for its part so far:

| phase | window |
|---|---|
| Asia | Tokyo 09:00-15:00 JST |
| pre-London | Tokyo 15:00 JST - London 08:00 |
| London-AM | London 08:00 - New York 09:30 |
| NY | New York 09:30 - t |

Phase token = SAX shape of the phase path (W = 3, A = 3) + size letter (phase range / its
20-session median, tercile) + **position letter** (close at t below / inside / above the phase
range).

**Last-bars tokens** at t: 5 min (last 6 bars), 15 min (last 4), 1 h (last 4): SAX W = 3, A = 3,
+ size letter + volume letter (same-clock 20-session median, tercile).

**Context tokens:** 1-h trend (EMA20 vs EMA50 on 1-h bars: up / down, `indicators.ema`);
position versus the previous day's range (below / inside / above); `new_high` / `new_low` (t's
bar made a new session high / low); `quiet` (last 2 h range in the lowest tercile of its
same-clock median ratio); clock bucket (London-AM / overlap / NY-PM).

All tokens use only bars that close at or before t. Size and volume letters are frozen on the
mining years. That is ~14 items per decision point.

## Combinations and similarity

* A **combination** = a set of **2 or 3 items** from different groups (e.g. `Asia:acb/s/above` +
  `15m:bca/l/v3` + `trend:up`). Single items are reported but not candidates.
* **Similar, not identical (families):** two combinations are in the same family if their items
  match after letting **one SAX letter differ by one step** in at most one item (`acb` ~ `bcb`),
  or a size / volume letter differ by one step. Families are formed on the mining years by
  linking similar frequent combinations (connected components); the family is then counted as
  one combination. Order inside a combination does not matter (a set).
* **Frequency rule (the user's):** a combination / family must occur on average **>= 2 times per
  week per market** in the mining years, counted as non-overlapping occurrences (>= 60 min apart).

## Score - conspicuous before moves, beyond volatility

For every occurrence the **expected** move probability comes from the control: the move rate
(up and down separately) in the same stratum of (market x clock bucket x quintile of the last-1-h
range / ATR1h), mining years. Per combination:

    lift_up = observed up-moves / expected up-moves     (same for down)
    z       = (observed - expected) / sqrt(sum p(1-p)), day-clustered (per-day sums)

A combination that only says "it is already volatile" has lift ~1. **Direction:** a combination
is "up" if lift_up > 1 and lift_up > lift_down (else "down").

## Selection and verdict (fixed now)

1. **Mining 2016-2019:** frequent combinations / families with lift >= 1.3 and z >= 4.0 in their
   direction (a high bar because the number of combinations tried is in the thousands; the count
   is reported). Ranked by z.
2. **Validation 2020-2021:** each mining candidate, unchanged, must have lift > 1.1 and z >= 2.0 in
   the same direction, and lift > 1 in at least 4 of the markets where it occurs. At most the
   **10 best** (by validation z) go on.
3. **Confirmation 2022-2024**, run once on those <= 10: lift > 1 with one-sided p <= 0.05 / 10
   (z >= 2.58), same direction, lift > 1 in at least 4 markets.

**Nothing survives step 2 -> H12 closed.** A confirmed combination is a candidate for stage 2
(below), never a strategy by itself.

## Track B - session sentences (DNA-style motif search)

**Session words.** For each market and day, one word per session: Asia (Tokyo 09:00-15:00 JST),
London (08:00-16:30 London), New York (09:30-16:00 New York). Word = SAX shape of the session path
(W = 4, A = 3) + size letter (session range / its 20-session median, tercile) + position letter
(session close below / inside / above the previous session's range). E.g. `abcc/l/above`.

**Outcome classes of a target session** (London or New York), fixed now:
* direction: up / down / flat (|close - open| < 0.25 x the session's 20-session median range),
* size: small / normal / big (tercile of range / 20-session median, mining years),
* reversal: yes if the session first trades beyond one side of its first hour's range and then
  closes beyond the other side.
-> 9 direction x size classes + reversal yes / no, each tested as its own binary outcome (10).

**Motifs (the "sentences").** For a target session the motif is an ordered sequence of the
session words before it:

| question | motif | target |
|---|---|---|
| Asia -> London | Asia word of the same day (+ optional: the previous day's NY word) | London |
| Asia -> New York | Asia word (+ optional previous NY word) | New York |
| Asia + London -> New York | Asia word, London word (+ optional previous NY word) | New York |
| multi-day | the last 2 or 3 complete days as session-word sequences | next London / NY |

Motif length 1-4 words. **Similar, not identical (like DNA motif search with mismatches):** a
motif matches a day if at most **one** letter of one word differs by one step (SAX letter, size or
position); motif families are formed as in track A (connected components of frequent motifs on the
mining years). Order matters inside a motif (it is a sequence in time).

**Frequency:** a motif family must occur on average >= 1 time per week per market in the mining
years (one session per day, so 2-5 per week is only possible for short motifs).

**Control (no volatility rediscovery):** the expected rate of each outcome class comes from the
same market x quintile of the previous session's range / its 20-session median, mining years.
Lift and day-clustered z as in track A (one observation per day, so z is per day).

**Selection and verdict:** the same three steps as track A (mining lift >= 1.3, z >= 4.0;
validation lift > 1.1, z >= 2.0, lift > 1 in >= 4 markets, at most 10 go on; confirmation
z >= 2.58, same direction, >= 4 markets). Nothing survives validation -> track B closed.

## Stage 2 - make it tradeable (only for confirmed combinations, separate prereg)

Net return of trading the confirmed combination in its direction (market entry at t, stop /
target from the move definition, 60-min time exit, `cost_gate.round_trip_spread`), and the
user's idea of **upgrading a frequent signal with filters** (trend, 15-min structure, session) -
as a fixed list in its own pre-registration, written after stage 1.

## Reported, no weight

Track B: for every question the top 30 motifs with their outcome-class table (how often each
class followed vs the control). Top 30 combinations of the mining years with counts, lifts, examples; single-item lifts; the
volatility-only control (lift of the size / volume letters alone); per-year and per-market tables.

## Not in this test

FX (no audit yet), exchange volume, 1-min tokens, horizons other than 60 min, combinations of
more than 3 items.

## Amendments before the first run (2026-10-30, decided by the user, nothing measured yet)

Found while counting frequencies only (no outcome was looked at):

1. **Asia data gap.** Dukascopy index CFDs have no Asia-hour bars before 2018-04 (UK100 before
   2019-01); XAUUSD has them from 2016. Years therefore: **mining 2018-05 - 2020-12** (XAUUSD also
   2016-01 - 2018-04), **validation 2021**, **confirmation 2022-2024** (unchanged). UK100 enters
   from 2019. User's choice.
2. **Track B frequency.** One session per day makes ">= 1 per week per market" reachable only for
   a few single-word families (upper bound measured: best family 1292 pooled occurrences without
   the position letter vs 592 needed). New rule: a motif family needs **>= 150 occurrences pooled
   over all markets** in the mining years. User's choice.

Technical clarifications (no threshold changed):

3. **Families are radius-1 stars**, not connected components: family(C) = C plus every combination
   that differs from C by one allowed one-step mutation. Connected components would chain
   (acb ~ bcb ~ bcc ~ ...) until a family covers almost every word of an item group.
4. **Control rates per period:** the expected move / class rate of a stratum is computed on the
   same period that is being evaluated (mining, validation or confirmation), so a year with more
   moves everywhere does not lift every combination.
5. **Track B multi-day motifs** = the last 4 session words before the target, contiguous in time
   (e.g. London(d-1), NY(d-1), Asia(d) -> London(d) uses 3; the 4-word version reaches back one
   more session). Motif length stays within 1-4 words.

## Amendment before the confirmation run (2026-10-30, decided by the user, 2022-2024 untouched)

Validation showed that the selected combinations raise up-moves as well as down-moves: the
volatility control (market x clock x last-1-h range quintile) is too coarse, so the lift of one
direction also measures volatility. The written confirmation rule could therefore be passed by
volatility alone. This amendment was made **after seeing mining and validation results**, it is
stricter, and the user decided it.

**New confirmation verdict (track A).** For a unit with direction D (other direction O), per
decision point: `s = (y_D - y_O) - (p_D - p_O)`. A unit is confirmed iff mean(s) > 0, day-clustered
z >= 2.58, and mean(s) > 0 in at least 4 markets. The selected units stay as chosen (top 10 by
validation z, including the one duplicate pair). The old rule (lift > 1, z >= 2.58, >= 4 markets)
is reported next to it, without weight.
