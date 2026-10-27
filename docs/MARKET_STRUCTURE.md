# Market structure framework (`trading_lab/market_structure.py`)

The professional replacement for `strategies/smc/structure.py` (which stays untouched for
the legacy SMC strategies). Written 2026-10-27 after a review found that the legacy engine
reported a single BOS/CHoCH as a *state* for 4-6 bars, treated every 3-bar pivot as a swing
(~1 in 5 bars), had no external/internal split and defined CHoCH against any opposite
swing instead of the protected one.

Every object carries the bar at which it becomes **known** (`confirmed_idx`). Nothing is
ever used before that bar - `tests/test_market_structure.py` checks this property on random
data by cutting the series and comparing.

## Definitions

**Swing (pivot).** A swing high at bar `i` needs `left` bars before it with highs `<=` and
`right` bars after it with highs `<` `high[i]`. It is confirmed at `i + right`. Two filters
turn raw pivots into structure:

* *alternation* - swings alternate high/low/high. A new high following a high replaces it
  only if it is higher (the more extreme one is the swing);
* *significance* - a new high is accepted only if it is at least `min_swing_atr x ATR`
  above the last swing low (zig-zag rule). Smaller wiggles are noise.

**Two levels.** *External* (swing) structure: larger `left/right` and ATR threshold - the
trend a higher-timeframe trader sees. *Internal* structure: smaller settings - the legs
inside one external swing. Each level runs its own state machine.

**Trend state, BOS and CHoCH (events, one bar each).**

* Uptrend: the *target* is the last confirmed swing high that has not been broken. The
  first **close** above it is a **BOS up** (event on that bar only). The target is then
  cleared until the next swing high confirms - so the same break can never fire twice.
* The *protected low* of an uptrend is the low of the leg that produced the last BOS
  (lowest low between the broken high's pivot and the BOS bar). The first **close** below
  it is a **CHoCH down**: the trend flips, and the protected high becomes the high of the
  leg that broke it.
* Breaking an internal low that is *not* the protected low is not a CHoCH.
* Downtrend: mirrored. From the neutral start, the first close beyond the last swing
  high/low sets the trend (labelled CHoCH).
* A wick through the target with the close back inside is a **sweep**, not a break.

**Liquidity: equal highs / lows.** When a swing high confirms within `eq_tol_atr x ATR` of
an earlier, still unbroken swing high, the pair forms an **EQH** pool at the higher of the
two (EQL mirrored). A wick through the pool with the close back inside is a **sweep** of
the pool; a close through it **takes** it.

**Fair value gap (FVG).** Three candles `i-2, i-1, i`, known at the close of `i`. Bullish:
`low[i] > high[i-2]`, zone `[high[i-2], low[i]]`; bearish mirrored. Kept only if the gap is
at least `fvg_min_atr x ATR` (and, optionally, the middle candle's body at least
`displacement_atr x ATR`). Lifecycle from bar `i+1` on:

* *touched* - price trades into the zone;
* *filled* - price reaches the far edge (by wick or by close, `fvg_fill`);
* *inverted* - a **close** beyond the far edge. The zone becomes an **inverse FVG (IFVG)**
  in the opposite direction: a broken bullish FVG is now resistance.

**Inverse FVG (IFVG).** Born at the inverting close, opposite direction, same bounds.
*Retest*: price trades back into it. *Invalidated*: a close back through its far side.

**Order block (OB).** Tied to a structure break: at a bullish BOS/CHoCH, the last bearish
candle (close < open) of the leg that produced the break, zone `[low, high]` of that candle
(bearish mirrored). Known at the break bar. *Mitigated* when price returns into it;
*invalidated* by a close beyond its far edge.

## Use

```python
from trading_lab.market_structure import StructureConfig, analyze
res = analyze(df, StructureConfig())          # df: open/high/low/close (mid recommended)
res.bars[["ext_trend", "ext_event", "int_event", "ext_protected_low"]]
res.active_zones(bar_idx, kinds=("fvg", "ifvg", "ob"))
```

`ext_event` / `int_event`: `+1` BOS up, `-1` BOS down, `+2` CHoCH up, `-2` CHoCH down,
`0` none. `sweep_high` / `sweep_low`: a swing high/low or EQH/EQL pool was swept on this bar.
