"""15m (and later 1m) POI respect / rejection measurement.

Keeps OB and FVG bounds separate so we can measure *where* reactions
and lower-TF ChoChs occur (OB vs FVG vs overlap vs outside).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Literal

import numpy as np
import pandas as pd

from ...indicators import atr as atr_indicator
from .structure import MarketStructureEngine
from .supply_demand import detect_fvg_at, order_block_for_fvg
from .types import TrendBias, Zone

ObRangeMode = Literal["wick", "body"]
Location = Literal[
    "ob_only",
    "fvg_only",
    "overlap",
    "merged_gap",  # inside merged POI but neither pure OB nor FVG (shouldn't happen often)
    "outside",
    "no_choch",
]


@dataclass(frozen=True)
class PoiPair:
    """OB + FVG kept separate for location analytics."""

    direction: int
    ob: Zone
    fvg: Zone
    created_idx: int
    created_ts: Any

    @property
    def merged_top(self) -> float:
        return max(self.ob.top, self.fvg.top)

    @property
    def merged_bottom(self) -> float:
        return min(self.ob.bottom, self.fvg.bottom)

    def touches_merged(self, high: float, low: float) -> bool:
        return low <= self.merged_top and high >= self.merged_bottom

    def touches_ob(self, high: float, low: float) -> bool:
        return self.ob.touched(high, low)

    def touches_fvg(self, high: float, low: float) -> bool:
        return self.fvg.touched(high, low)


@dataclass
class TouchEvent:
    poi_idx: int
    touch_idx: int
    touch_ts: Any
    direction: int
    respected: bool
    mitigated_first: bool
    reaction_atr: float
    first_touch_part: Location  # which part was touched first
    choch_location: Location = "no_choch"
    choch_idx_1m: int | None = None


@dataclass
class RespectParams:
    ob_lookback: int = 10
    ob_range: ObRangeMode = "wick"
    require_fvg: bool = True
    horizon_bars: int = 20
    min_reaction_atr: float = 0.35
    atr_period: int = 14
    min_fvg_atr: float = 0.05  # skip tiny gaps
    max_pois: int = 5000


@dataclass
class RespectReport:
    params: RespectParams
    n_pois: int = 0
    n_touched: int = 0
    n_respected: int = 0
    n_mitigated_first: int = 0
    respect_rate: float = 0.0
    touch_part_counts: dict[str, int] = field(default_factory=dict)
    choch_location_counts: dict[str, int] = field(default_factory=dict)
    events: list[TouchEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": {
                "ob_lookback": self.params.ob_lookback,
                "ob_range": self.params.ob_range,
                "horizon_bars": self.params.horizon_bars,
                "min_reaction_atr": self.params.min_reaction_atr,
                "min_fvg_atr": self.params.min_fvg_atr,
            },
            "n_pois": self.n_pois,
            "n_touched": self.n_touched,
            "n_respected": self.n_respected,
            "n_mitigated_first": self.n_mitigated_first,
            "respect_rate": round(self.respect_rate, 4),
            "touch_part_counts": self.touch_part_counts,
            "choch_location_counts": self.choch_location_counts,
        }


def _ob_with_range_mode(
    df: pd.DataFrame, fvg: Zone, *, lookback: int, mode: ObRangeMode
) -> Zone | None:
    ob = order_block_for_fvg(df, fvg, lookback=lookback)
    if ob is None:
        return None
    if mode == "wick":
        return ob
    # body-only: recompute from candle open/close
    i = ob.created_idx
    o = float(df["open"].iloc[i])
    c = float(df["close"].iloc[i])
    return Zone(
        direction=ob.direction,
        top=max(o, c),
        bottom=min(o, c),
        created_idx=i,
        created_ts=ob.created_ts,
        kind="ob",
    )


def detect_poi_pairs(df: pd.DataFrame, params: RespectParams) -> list[PoiPair]:
    """Causal OB+FVG pairs on a single timeframe."""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    atr_s = atr_indicator(df["high"], df["low"], df["close"], params.atr_period)
    out: list[PoiPair] = []
    for i in range(2, len(df)):
        fvg = detect_fvg_at(high, low, df.index, i)
        if fvg is None:
            continue
        atr_i = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 0.0
        gap = abs(fvg.top - fvg.bottom)
        if atr_i > 0 and gap < params.min_fvg_atr * atr_i:
            continue
        if params.require_fvg:
            ob = _ob_with_range_mode(
                df, fvg, lookback=params.ob_lookback, mode=params.ob_range
            )
            if ob is None:
                continue
            out.append(
                PoiPair(
                    direction=fvg.direction,
                    ob=ob,
                    fvg=fvg,
                    created_idx=i,
                    created_ts=df.index[i],
                )
            )
        if len(out) >= params.max_pois:
            break
    return out


def _classify_price(price: float, poi: PoiPair) -> Location:
    in_ob = poi.ob.contains(price)
    in_fvg = poi.fvg.contains(price)
    if in_ob and in_fvg:
        return "overlap"
    if in_ob:
        return "ob_only"
    if in_fvg:
        return "fvg_only"
    if poi.merged_bottom <= price <= poi.merged_top:
        return "merged_gap"
    return "outside"


def _classify_touch_bar(high: float, low: float, poi: PoiPair) -> Location:
    t_ob = poi.touches_ob(high, low)
    t_fvg = poi.touches_fvg(high, low)
    if t_ob and t_fvg:
        return "overlap"
    if t_ob:
        return "ob_only"
    if t_fvg:
        return "fvg_only"
    if poi.touches_merged(high, low):
        return "merged_gap"
    return "outside"


def _fully_mitigated(poi: PoiPair, high: float, low: float) -> bool:
    if poi.direction > 0:
        return low <= poi.merged_bottom
    return high >= poi.merged_top


def _favorable_excursion(
    direction: int,
    touch_extreme: float,
    highs: np.ndarray,
    lows: np.ndarray,
    start: int,
    end: int,
) -> float:
    if start >= end:
        return 0.0
    if direction > 0:
        return float(np.max(highs[start:end]) - touch_extreme)
    return float(touch_extreme - np.min(lows[start:end]))


def evaluate_respect_15m(
    df_15m: pd.DataFrame,
    params: RespectParams | None = None,
    *,
    df_1m: pd.DataFrame | None = None,
    swing_length_1m: int = 5,
) -> RespectReport:
    """Measure first-touch rejection rate of 15m OB+FVG pairs.

    If df_1m is provided, also locate the first 1m ChoCh after the touch
    relative to OB / FVG / overlap.
    """
    params = params or RespectParams()
    if df_15m.index.tz is None:
        df_15m = df_15m.copy()
        df_15m.index = df_15m.index.tz_localize("UTC")

    atr_s = atr_indicator(
        df_15m["high"], df_15m["low"], df_15m["close"], params.atr_period
    )
    highs = df_15m["high"].to_numpy(dtype=float)
    lows = df_15m["low"].to_numpy(dtype=float)
    closes = df_15m["close"].to_numpy(dtype=float)

    pois = detect_poi_pairs(df_15m, params)
    eng_1m: MarketStructureEngine | None = None
    choch_1m: np.ndarray | None = None
    closes_1m: np.ndarray | None = None
    if df_1m is not None and not df_1m.empty:
        if df_1m.index.tz is None:
            df_1m = df_1m.copy()
            df_1m.index = df_1m.index.tz_localize("UTC")
        eng_1m = MarketStructureEngine(swing_length_1m)
        eng_1m.precompute(df_1m)
        closes_1m = df_1m["close"].to_numpy(dtype=float)
        choch_1m = np.zeros(len(df_1m), dtype=np.int8)
        # Sample every bar would be slow; step=1 needed for first ChoCh accuracy.
        # Use lightweight break check only (cached swings).
        for i in range(len(df_1m)):
            choch_1m[i] = eng_1m.snapshot_at(df_1m, i).last_choch_dir


    events: list[TouchEvent] = []
    touch_parts: dict[str, int] = {}
    choch_locs: dict[str, int] = {}

    for pi, poi in enumerate(pois):
        # first touch after creation (+1 bar so displacement bar doesn't count)
        touch_i: int | None = None
        for j in range(poi.created_idx + 1, len(df_15m)):
            if poi.touches_merged(float(highs[j]), float(lows[j])):
                touch_i = j
                break
        if touch_i is None:
            continue

        part = _classify_touch_bar(float(highs[touch_i]), float(lows[touch_i]), poi)
        touch_parts[part] = touch_parts.get(part, 0) + 1

        atr_i = float(atr_s.iloc[touch_i]) if not np.isnan(atr_s.iloc[touch_i]) else 0.0
        end = min(len(df_15m), touch_i + 1 + params.horizon_bars)
        touch_ext = float(lows[touch_i]) if poi.direction > 0 else float(highs[touch_i])

        mitigated_first = False
        respected = False
        reaction = 0.0

        for j in range(touch_i, end):
            if _fully_mitigated(poi, float(highs[j]), float(lows[j])):
                # same bar as touch: only fail if close also confirms deep break
                if j == touch_i:
                    if poi.direction > 0 and closes[j] < poi.merged_bottom:
                        mitigated_first = True
                        break
                    if poi.direction < 0 and closes[j] > poi.merged_top:
                        mitigated_first = True
                        break
                else:
                    mitigated_first = True
                    break
            reaction = _favorable_excursion(
                poi.direction, touch_ext, highs, lows, touch_i, j + 1
            )
            if atr_i > 0 and reaction >= params.min_reaction_atr * atr_i:
                respected = True
                break
            # body rejection: close back outside zone in bias direction
            if poi.direction > 0 and closes[j] > poi.merged_top:
                respected = True
                reaction = max(reaction, closes[j] - touch_ext)
                break
            if poi.direction < 0 and closes[j] < poi.merged_bottom:
                respected = True
                reaction = max(reaction, touch_ext - closes[j])
                break

        choch_loc: Location = "no_choch"
        choch_i1m: int | None = None
        if choch_1m is not None and df_1m is not None and closes_1m is not None:
            choch_loc, choch_i1m = _find_1m_choch_location_fast(
                df_1m, choch_1m, closes_1m, poi, df_15m.index[touch_i], params.horizon_bars
            )
        choch_locs[choch_loc] = choch_locs.get(choch_loc, 0) + 1

        events.append(
            TouchEvent(
                poi_idx=pi,
                touch_idx=touch_i,
                touch_ts=df_15m.index[touch_i],
                direction=poi.direction,
                respected=respected and not mitigated_first,
                mitigated_first=mitigated_first,
                reaction_atr=(reaction / atr_i) if atr_i > 0 else 0.0,
                first_touch_part=part,
                choch_location=choch_loc,
                choch_idx_1m=choch_i1m,
            )
        )

    n_touched = len(events)
    n_respected = sum(1 for e in events if e.respected)
    n_mit = sum(1 for e in events if e.mitigated_first)
    rate = n_respected / n_touched if n_touched else 0.0
    return RespectReport(
        params=params,
        n_pois=len(pois),
        n_touched=n_touched,
        n_respected=n_respected,
        n_mitigated_first=n_mit,
        respect_rate=rate,
        touch_part_counts=touch_parts,
        choch_location_counts=choch_locs,
        events=events,
    )


def _find_1m_choch_location_fast(
    df_1m: pd.DataFrame,
    choch_dirs: np.ndarray,
    closes: np.ndarray,
    poi: PoiPair,
    touch_ts: Any,
    horizon_15m_bars: int,
) -> tuple[Location, int | None]:
    touch_ts = pd.Timestamp(touch_ts)
    horizon_1m = max(30, horizon_15m_bars * 15)
    start = int(df_1m.index.searchsorted(touch_ts))
    end = min(len(df_1m), start + horizon_1m)
    want = poi.direction
    for i in range(start, end):
        if int(choch_dirs[i]) == want:
            return _classify_price(float(closes[i]), poi), i
    return "no_choch", None


def _find_1m_choch_location(
    df_1m: pd.DataFrame,
    eng: MarketStructureEngine,
    poi: PoiPair,
    touch_ts: Any,
    horizon_15m_bars: int,
) -> tuple[Location, int | None]:
    """Legacy wrapper."""
    closes = df_1m["close"].to_numpy(dtype=float)
    dirs = np.array(
        [eng.snapshot_at(df_1m, i).last_choch_dir for i in range(len(df_1m))],
        dtype=np.int8,
    )
    return _find_1m_choch_location_fast(
        df_1m, dirs, closes, poi, touch_ts, horizon_15m_bars
    )


def grid_search_respect(
    df_15m: pd.DataFrame,
    *,
    df_1m: pd.DataFrame | None = None,
    lookbacks: list[int] | None = None,
    ranges: list[ObRangeMode] | None = None,
    horizons: list[int] | None = None,
    reactions: list[float] | None = None,
) -> list[RespectReport]:
    """Small grid over OB detection / respect definition knobs."""
    lookbacks = lookbacks or [5, 10, 15]
    ranges = ranges or ["wick", "body"]
    horizons = horizons or [12, 20, 32]
    reactions = reactions or [0.25, 0.35, 0.5]
    reports: list[RespectReport] = []
    for lb, rg, hz, rx in product(lookbacks, ranges, horizons, reactions):
        params = RespectParams(
            ob_lookback=lb,
            ob_range=rg,
            horizon_bars=hz,
            min_reaction_atr=rx,
        )
        reports.append(evaluate_respect_15m(df_15m, params, df_1m=df_1m))
    reports.sort(key=lambda r: (r.respect_rate, r.n_touched), reverse=True)
    return reports


def entry_poi_near_choch(
    df: pd.DataFrame,
    choch_idx: int,
    direction: int,
    params: RespectParams,
) -> PoiPair | Zone | None:
    """1m entry zone used after ChoCh — prefer OB+FVG, else OB-only Zone."""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)

    for i in range(choch_idx, max(1, choch_idx - 5) - 1, -1):
        fvg = detect_fvg_at(high, low, df.index, i)
        if fvg is None or fvg.direction != direction:
            continue
        ob = _ob_with_range_mode(
            df, fvg, lookback=params.ob_lookback, mode=params.ob_range
        )
        if ob is None:
            continue
        return PoiPair(
            direction=direction,
            ob=ob,
            fvg=fvg,
            created_idx=i,
            created_ts=df.index[i],
        )

    # OB-only fallback (wrap as degenerate pair with fvg=ob)
    for j in range(choch_idx - 1, max(-1, choch_idx - params.ob_lookback) - 1, -1):
        if j < 0:
            break
        if direction > 0 and closes[j] < opens[j]:
            z = Zone(
                direction=1,
                top=float(high[j]) if params.ob_range == "wick" else max(opens[j], closes[j]),
                bottom=float(low[j]) if params.ob_range == "wick" else min(opens[j], closes[j]),
                created_idx=j,
                created_ts=df.index[j],
                kind="ob",
            )
            return PoiPair(direction=1, ob=z, fvg=z, created_idx=j, created_ts=z.created_ts)
        if direction < 0 and closes[j] > opens[j]:
            z = Zone(
                direction=-1,
                top=float(high[j]) if params.ob_range == "wick" else max(opens[j], closes[j]),
                bottom=float(low[j]) if params.ob_range == "wick" else min(opens[j], closes[j]),
                created_idx=j,
                created_ts=df.index[j],
                kind="ob",
            )
            return PoiPair(direction=-1, ob=z, fvg=z, created_idx=j, created_ts=z.created_ts)
    return None


def evaluate_respect_1m_after_choch(
    df_1m: pd.DataFrame,
    params: RespectParams | None = None,
    *,
    swing_length_1m: int = 5,
    max_chochs: int = 8000,
    choch_events: list[tuple[int, int]] | None = None,
) -> RespectReport:
    """After each 1m ChoCh, measure pullback respect of the 1m entry OB(+FVG).

    This matches the live entry model: ChoCh first, then limit on pullback.
    Pass `choch_events` to reuse across a grid (expensive scan once).
    """
    params = params or RespectParams(
        horizon_bars=30,
        min_reaction_atr=0.5,
        min_fvg_atr=0.0,
    )
    if df_1m.index.tz is None:
        df_1m = df_1m.copy()
        df_1m.index = df_1m.index.tz_localize("UTC")

    atr_s = atr_indicator(df_1m["high"], df_1m["low"], df_1m["close"], params.atr_period)
    highs = df_1m["high"].to_numpy(dtype=float)
    lows = df_1m["low"].to_numpy(dtype=float)
    closes = df_1m["close"].to_numpy(dtype=float)

    if choch_events is None:
        eng = MarketStructureEngine(swing_length_1m)
        eng.precompute(df_1m)
        choch_events = []
        last_fire = -999
        last_dir = 0
        for i in range(len(df_1m)):
            d = int(eng.snapshot_at(df_1m, i).last_choch_dir)
            if d == 0:
                continue
            if i - last_fire <= 10 and d == last_dir:
                continue
            choch_events.append((i, d))
            last_fire = i
            last_dir = d
            if len(choch_events) >= max_chochs:
                break

    events: list[TouchEvent] = []
    touch_parts: dict[str, int] = {}
    kind_counts: dict[str, int] = {"ob_fvg": 0, "ob_only": 0, "no_poi": 0}

    n_pois = 0
    for ci, (choch_i, direction) in enumerate(choch_events):
        poi = entry_poi_near_choch(df_1m, choch_i, direction, params)
        if poi is None:
            kind_counts["no_poi"] += 1
            continue
        n_pois += 1
        if (
            poi.ob.kind == "ob"
            and poi.fvg.kind == "ob"
            and poi.ob.created_idx == poi.fvg.created_idx
        ):
            kind_counts["ob_only"] += 1
        else:
            kind_counts["ob_fvg"] += 1

        # Pullback touch: after ChoCh bar
        touch_i: int | None = None
        search_end = min(len(df_1m), choch_i + 1 + params.horizon_bars * 2)
        for j in range(choch_i + 1, search_end):
            if poi.touches_merged(float(highs[j]), float(lows[j])):
                touch_i = j
                break
        if touch_i is None:
            continue

        part = _classify_touch_bar(float(highs[touch_i]), float(lows[touch_i]), poi)
        touch_parts[part] = touch_parts.get(part, 0) + 1

        atr_i = float(atr_s.iloc[touch_i]) if not np.isnan(atr_s.iloc[touch_i]) else 0.0
        end = min(len(df_1m), touch_i + 1 + params.horizon_bars)
        touch_ext = float(lows[touch_i]) if direction > 0 else float(highs[touch_i])

        mitigated_first = False
        respected = False
        reaction = 0.0
        for j in range(touch_i, end):
            if _fully_mitigated(poi, float(highs[j]), float(lows[j])):
                if j == touch_i:
                    if direction > 0 and closes[j] < poi.merged_bottom:
                        mitigated_first = True
                        break
                    if direction < 0 and closes[j] > poi.merged_top:
                        mitigated_first = True
                        break
                else:
                    mitigated_first = True
                    break
            reaction = _favorable_excursion(
                direction, touch_ext, highs, lows, touch_i, j + 1
            )
            if atr_i > 0 and reaction >= params.min_reaction_atr * atr_i:
                respected = True
                break
            if direction > 0 and closes[j] > poi.merged_top:
                respected = True
                break
            if direction < 0 and closes[j] < poi.merged_bottom:
                respected = True
                break

        events.append(
            TouchEvent(
                poi_idx=ci,
                touch_idx=touch_i,
                touch_ts=df_1m.index[touch_i],
                direction=direction,
                respected=respected and not mitigated_first,
                mitigated_first=mitigated_first,
                reaction_atr=(reaction / atr_i) if atr_i > 0 else 0.0,
                first_touch_part=part,
                choch_location="ob_only",
                choch_idx_1m=choch_i,
            )
        )

    n_touched = len(events)
    n_respected = sum(1 for e in events if e.respected)
    n_mit = sum(1 for e in events if e.mitigated_first)
    rate = n_respected / n_touched if n_touched else 0.0
    return RespectReport(
        params=params,
        n_pois=n_pois,
        n_touched=n_touched,
        n_respected=n_respected,
        n_mitigated_first=n_mit,
        respect_rate=rate,
        touch_part_counts=touch_parts,
        choch_location_counts=kind_counts,
        events=events,
    )


def collect_choch_events(
    df_1m: pd.DataFrame,
    *,
    swing_length_1m: int = 5,
    max_chochs: int = 8000,
) -> list[tuple[int, int]]:
    """Precompute ChoCh indices once for grid reuse."""
    if df_1m.index.tz is None:
        df_1m = df_1m.copy()
        df_1m.index = df_1m.index.tz_localize("UTC")
    eng = MarketStructureEngine(swing_length_1m)
    eng.precompute(df_1m)
    choch_events: list[tuple[int, int]] = []
    last_fire = -999
    last_dir = 0
    for i in range(len(df_1m)):
        d = int(eng.snapshot_at(df_1m, i).last_choch_dir)
        if d == 0:
            continue
        if i - last_fire <= 10 and d == last_dir:
            continue
        choch_events.append((i, d))
        last_fire = i
        last_dir = d
        if len(choch_events) >= max_chochs:
            break
    return choch_events


def grid_search_1m_entry(
    df_1m: pd.DataFrame,
    *,
    lookbacks: list[int] | None = None,
    ranges: list[ObRangeMode] | None = None,
    horizons: list[int] | None = None,
    reactions: list[float] | None = None,
    swing_length_1m: int = 5,
) -> list[RespectReport]:
    lookbacks = lookbacks or [5, 10, 15]
    ranges = ranges or ["wick", "body"]
    horizons = horizons or [15, 30, 45]
    reactions = reactions or [0.35, 0.5, 0.75]
    print("  precomputing ChoCh events once...", flush=True)
    choch_events = collect_choch_events(df_1m, swing_length_1m=swing_length_1m)
    print(f"  choch_events={len(choch_events)}", flush=True)
    reports: list[RespectReport] = []
    combos = list(product(lookbacks, ranges, horizons, reactions))
    for k, (lb, rg, hz, rx) in enumerate(combos, 1):
        params = RespectParams(
            ob_lookback=lb,
            ob_range=rg,
            horizon_bars=hz,
            min_reaction_atr=rx,
            min_fvg_atr=0.0,
        )
        reports.append(
            evaluate_respect_1m_after_choch(
                df_1m,
                params,
                swing_length_1m=swing_length_1m,
                choch_events=choch_events,
            )
        )
        if k % 9 == 0 or k == len(combos):
            print(f"  grid {k}/{len(combos)} best_so_far={max(r.respect_rate for r in reports):.1%}", flush=True)
    reports.sort(key=lambda r: (r.respect_rate, r.n_touched), reverse=True)
    return reports
