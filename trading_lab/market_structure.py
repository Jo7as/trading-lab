"""Causal market structure: swings, BOS/CHoCH events, liquidity, FVG / IFVG, order blocks.

Definitions in ``docs/MARKET_STRUCTURE.md``. Every rule has a test on a hand-made price
path in ``tests/test_market_structure.py``, plus a no-look-ahead property test.

Design rules:
* everything is known at a bar's CLOSE and stored with that bar (``confirmed_idx``);
* BOS and CHoCH are EVENTS (one bar each), never states;
* swings are filtered (alternation + ATR significance), at two levels (external/internal);
* CHoCH is a close beyond the PROTECTED level, not beyond any opposite swing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .indicators import atr as _atr

BOS_UP, BOS_DN, CHOCH_UP, CHOCH_DN = 1, -1, 2, -2


@dataclass(frozen=True)
class StructureConfig:
    atr_period: int = 14
    # external (swing) structure
    ext_left: int = 5
    ext_right: int = 5
    ext_min_swing_atr: float = 1.5
    # internal structure
    int_left: int = 2
    int_right: int = 2
    int_min_swing_atr: float = 0.5
    # liquidity (equal highs / lows)
    eq_tol_atr: float = 0.10             # band around a cluster, in ATR
    eq_lookback: int = 300               # bars back a swing may find its equal
    eq_min_sep: int = 3                  # bars between two touches of one pool
    eq_levels: tuple[str, ...] = ("ext", "int")
    # fair value gaps
    fvg_min_atr: float = 0.25
    displacement_atr: float = 0.0        # 0 = no body filter on the middle candle
    fvg_fill: str = "wick"               # wick | close
    zone_max_age: int = 500              # bars a zone stays tracked


@dataclass
class Swing:
    idx: int
    confirmed_idx: int
    price: float
    kind: int                            # +1 high, -1 low
    broken_idx: int | None = None


@dataclass
class Zone:
    kind: str                            # fvg | ifvg | ob
    direction: int                       # +1 bullish (support), -1 bearish (resistance)
    top: float
    bottom: float
    created_idx: int                     # known at the close of this bar
    origin_idx: int                      # for ifvg: the fvg it came from; ob: the candle
    touched_idx: int | None = None
    filled_idx: int | None = None
    inverted_idx: int | None = None      # fvg -> ifvg
    retest_idx: int | None = None        # ifvg / ob: first return into the zone
    invalidated_idx: int | None = None

    def active_at(self, i: int) -> bool:
        if i < self.created_idx:
            return False
        for end in (self.filled_idx if self.kind == "fvg" else None, self.inverted_idx, self.invalidated_idx):
            if end is not None and end <= i:
                return False
        return True


@dataclass
class Pool:
    """Equal highs (EQH, kind +1) / equal lows (EQL, kind -1): a cluster of >= 2 swing
    points within ``eq_tol_atr x ATR`` of each other. Stops sit beyond the cluster, so
    ``level`` is its outer edge (highest high / lowest low)."""
    kind: int
    level: float
    created_idx: int
    zone_low: float = np.nan
    zone_high: float = np.nan
    touches: list[int] = field(default_factory=list)   # pivot bars of the swings in the cluster
    level_name: str = "ext"
    updated_idx: int = -1                # last bar a touch joined
    swept_idx: list[int] = field(default_factory=list)
    taken_idx: int | None = None

    @property
    def n_touches(self) -> int:
        return len(self.touches)


class _LiquidityBook:
    """Builds EQH/EQL pools from confirmed swings of one level (causal)."""

    def __init__(self, name: str, cfg: "StructureConfig"):
        self.name, self.cfg = name, cfg
        self.cands: dict[int, list[Swing]] = {1: [], -1: []}   # unbroken swings, not yet pooled
        self.open: list[Pool] = []

    def on_bar_close(self, i: int, c: float) -> None:
        """Drop candidates that a close has broken or that are too old."""
        lb = self.cfg.eq_lookback
        self.cands[1] = [s for s in self.cands[1] if c <= s.price and i - s.idx <= lb]
        self.cands[-1] = [s for s in self.cands[-1] if c >= s.price and i - s.idx <= lb]

    def on_swing(self, s: Swing, i: int, atr_i: float, pools: list[Pool]) -> None:
        if not np.isfinite(atr_i):
            self.cands[s.kind].append(s)
            return
        tol = self.cfg.eq_tol_atr * atr_i
        # 1) joins an existing open pool?
        for p in self.open:
            if p.kind != s.kind or p.taken_idx is not None:
                continue
            # the whole cluster must stay within the band (no creeping), and the candle
            # that SWEPT the pool is the sweep, not another equal high
            width = max(p.zone_high, s.price) - min(p.zone_low, s.price)
            if width <= tol and s.idx not in p.swept_idx and s.idx - max(p.touches) >= self.cfg.eq_min_sep:
                p.touches.append(s.idx)
                p.zone_low, p.zone_high = min(p.zone_low, s.price), max(p.zone_high, s.price)
                p.level = p.zone_high if p.kind > 0 else p.zone_low
                p.updated_idx = i
                return
        # 2) forms a new pool with an earlier, unbroken swing at the same level?
        match = [q for q in self.cands[s.kind]
                 if q is not s and abs(q.price - s.price) <= tol and s.idx - q.idx >= self.cfg.eq_min_sep]
        if match:
            q = max(match, key=lambda x: x.idx)
            lo, hi = min(q.price, s.price), max(q.price, s.price)
            p = Pool(s.kind, hi if s.kind > 0 else lo, created_idx=i, zone_low=lo, zone_high=hi,
                     touches=[q.idx, s.idx], level_name=self.name, updated_idx=i)
            pools.append(p)
            self.open.append(p)
            self.cands[s.kind] = [x for x in self.cands[s.kind] if x is not q]
            return
        self.cands[s.kind].append(s)

    def update(self, i: int, h: float, l: float, c: float) -> tuple[int, int, int, int]:
        """(eqh_swept, eql_swept, eqh_taken, eql_taken) on bar i for pools known before i."""
        sh = sl = th = tl = 0
        for p in self.open:
            if p.taken_idx is not None or p.created_idx >= i:
                continue
            if p.kind > 0:
                if c > p.level:
                    p.taken_idx, th = i, 1
                elif h > p.level:
                    p.swept_idx.append(i)
                    sh = 1
            else:
                if c < p.level:
                    p.taken_idx, tl = i, 1
                elif l < p.level:
                    p.swept_idx.append(i)
                    sl = 1
        self.open = [p for p in self.open if p.taken_idx is None]
        return sh, sl, th, tl


@dataclass
class StructureResult:
    bars: pd.DataFrame
    swings: dict[str, list[Swing]]
    zones: list[Zone]
    pools: list[Pool]

    def active_zones(self, bar_idx: int, kinds: tuple[str, ...] = ("fvg", "ifvg", "ob"),
                     direction: int | None = None) -> list[Zone]:
        return [z for z in self.zones if z.kind in kinds and z.active_at(bar_idx)
                and (direction is None or z.direction == direction)]


# ── swings ──────────────────────────────────────────────────────────────

def raw_pivots(high: np.ndarray, low: np.ndarray, left: int, right: int) -> list[Swing]:
    """Fractal pivots, ordered by confirmation bar. Left side <=, right side < (so of
    two equal highs the later one is not a pivot of the earlier - each pivot once)."""
    n = len(high)
    out: list[Swing] = []
    for i in range(left, n - right):
        c = i + right
        if high[i] >= high[i - left:i].max(initial=-np.inf) and high[i] > high[i + 1:c + 1].max(initial=-np.inf):
            out.append(Swing(i, c, float(high[i]), 1))
        if low[i] <= low[i - left:i].min(initial=np.inf) and low[i] < low[i + 1:c + 1].min(initial=np.inf):
            out.append(Swing(i, c, float(low[i]), -1))
    out.sort(key=lambda s: (s.confirmed_idx, s.idx))
    return out


class _SwingFilter:
    """Alternation + ATR significance, applied as pivots confirm (causal)."""

    def __init__(self, min_atr: float):
        self.min_atr = min_atr
        self.swings: list[Swing] = []

    def add(self, p: Swing, atr_now: float) -> Swing | None:
        """Returns the swing that is now the latest (new or replaced), or None."""
        if not self.swings:
            self.swings.append(p)
            return p
        last = self.swings[-1]
        if p.kind == last.kind:
            if (p.kind > 0 and p.price > last.price) or (p.kind < 0 and p.price < last.price):
                self.swings[-1] = p             # the more extreme one is the swing
                return p
            return None
        size = abs(p.price - last.price)
        if np.isfinite(atr_now) and size < self.min_atr * atr_now:
            return None                          # noise
        self.swings.append(p)
        return p

    def last(self, kind: int) -> Swing | None:
        for s in reversed(self.swings):
            if s.kind == kind:
                return s
        return None


# ── one level of structure ─────────────────────────────────────────────

class _Level:
    def __init__(self, name: str, left: int, right: int, min_atr: float):
        self.name, self.left, self.right = name, left, right
        self.filter = _SwingFilter(min_atr)
        self.trend = 0
        self.target_high: Swing | None = None    # unbroken swing high a BOS up must close beyond
        self.target_low: Swing | None = None
        self.protected_low: float = np.nan
        self.protected_low_idx: int = -1
        self.protected_high: float = np.nan
        self.protected_high_idx: int = -1
        self.leg_start_high_idx = 0              # pivot of the broken high (start of the leg)
        self.leg_start_low_idx = 0

    def on_swing(self, s: Swing) -> None:
        if s.kind > 0:
            self.target_high = s
        else:
            self.target_low = s

    def on_bar(self, i: int, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> tuple[int, int, int]:
        """(event, sweep_high, sweep_low) for bar i."""
        ev = sw_h = sw_l = 0
        th, tl = self.target_high, self.target_low
        up_break = th is not None and c[i] > th.price
        dn_break = tl is not None and c[i] < tl.price
        prot_dn = np.isfinite(self.protected_low) and c[i] < self.protected_low
        prot_up = np.isfinite(self.protected_high) and c[i] > self.protected_high

        if self.trend >= 0 and (prot_dn if self.trend > 0 else dn_break):
            # CHoCH down (or the first direction from neutral)
            start = self.protected_low_idx if self.trend > 0 else (tl.idx if tl else i)
            ev = CHOCH_DN
            self._flip_down(i, start, h)
        elif self.trend <= 0 and (prot_up if self.trend < 0 else up_break):
            start = self.protected_high_idx if self.trend < 0 else (th.idx if th else i)
            ev = CHOCH_UP
            self._flip_up(i, start, l)
        elif self.trend > 0 and up_break:
            ev = BOS_UP
            seg = l[th.idx:i + 1]
            self.protected_low = float(seg.min())
            self.protected_low_idx = th.idx + int(seg.argmin())
            th.broken_idx = i
            self.target_high = None
        elif self.trend < 0 and dn_break:
            ev = BOS_DN
            seg = h[tl.idx:i + 1]
            self.protected_high = float(seg.max())
            self.protected_high_idx = tl.idx + int(seg.argmax())
            tl.broken_idx = i
            self.target_low = None
        # sweeps of the current targets (wick through, close back)
        th, tl = self.target_high, self.target_low
        if th is not None and h[i] > th.price and c[i] <= th.price:
            sw_h = 1
        if tl is not None and l[i] < tl.price and c[i] >= tl.price:
            sw_l = 1
        return ev, sw_h, sw_l

    def _flip_down(self, i: int, start: int, h: np.ndarray) -> None:
        self.trend = -1
        start = max(0, min(start, i))
        seg = h[start:i + 1]
        self.protected_high = float(seg.max())
        self.protected_high_idx = start + int(seg.argmax())
        self.protected_low, self.protected_low_idx = np.nan, -1
        if self.target_low is not None:
            self.target_low.broken_idx = i
        self.target_low = None

    def _flip_up(self, i: int, start: int, l: np.ndarray) -> None:
        self.trend = 1
        start = max(0, min(start, i))
        seg = l[start:i + 1]
        self.protected_low = float(seg.min())
        self.protected_low_idx = start + int(seg.argmin())
        self.protected_high, self.protected_high_idx = np.nan, -1
        if self.target_high is not None:
            self.target_high.broken_idx = i
        self.target_high = None


# ── the full analysis ──────────────────────────────────────────────────

def analyze(df: pd.DataFrame, cfg: StructureConfig = StructureConfig()) -> StructureResult:
    """Run every component over ``df`` bar by bar. Causal by construction."""
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(df)
    a = _atr(df["high"], df["low"], df["close"], cfg.atr_period).to_numpy()

    levels = {"ext": _Level("ext", cfg.ext_left, cfg.ext_right, cfg.ext_min_swing_atr),
              "int": _Level("int", cfg.int_left, cfg.int_right, cfg.int_min_swing_atr)}
    piv = {k: raw_pivots(h, l, lv.left, lv.right) for k, lv in levels.items()}
    ptr = {k: 0 for k in levels}
    cols = {f"{k}_{x}": np.zeros(n) for k in levels for x in ("trend", "event")}
    for k in levels:
        cols[f"{k}_swing_sweep_high"] = np.zeros(n)
        cols[f"{k}_swing_sweep_low"] = np.zeros(n)
        cols[f"{k}_protected_high"] = np.full(n, np.nan)
        cols[f"{k}_protected_low"] = np.full(n, np.nan)
        cols[f"{k}_swing_high"] = np.full(n, np.nan)
        cols[f"{k}_swing_low"] = np.full(n, np.nan)
    sweep_h, sweep_l = np.zeros(n), np.zeros(n)
    zones: list[Zone] = []
    pools: list[Pool] = []
    books = {k: _LiquidityBook(k, cfg) for k in cfg.eq_levels if k in levels}
    for k in books:
        for x in ("eqh_sweep", "eql_sweep", "eqh_taken", "eql_taken"):
            cols[f"{k}_{x}"] = np.zeros(n)
    active: list[Zone] = []

    for i in range(n):
        atr_i = a[i - 1] if i > 0 and np.isfinite(a[i - 1]) else a[i]
        # 1) swings confirmed at this bar -> filters, targets, liquidity pools
        for k, lv in levels.items():
            lst = piv[k]
            while ptr[k] < len(lst) and lst[ptr[k]].confirmed_idx <= i:
                p = lst[ptr[k]]
                ptr[k] += 1
                s = lv.filter.add(p, atr_i)
                if s is None:
                    continue
                lv.on_swing(s)
                if k in books:
                    books[k].on_swing(s, i, atr_i, pools)
        # 2) structure events on this close
        for k, lv in levels.items():
            ev, sh, sl = lv.on_bar(i, h, l, c)
            cols[f"{k}_event"][i] = ev
            cols[f"{k}_trend"][i] = lv.trend
            cols[f"{k}_protected_high"][i] = lv.protected_high
            cols[f"{k}_protected_low"][i] = lv.protected_low
            lh, ll = lv.filter.last(1), lv.filter.last(-1)
            cols[f"{k}_swing_high"][i] = lh.price if lh else np.nan
            cols[f"{k}_swing_low"][i] = ll.price if ll else np.nan
            # per-level sweep of the current swing target (wick through, close back)
            cols[f"{k}_swing_sweep_high"][i], cols[f"{k}_swing_sweep_low"][i] = sh, sl
            if k == "ext":
                sweep_h[i], sweep_l[i] = sh, sl
                if ev in (BOS_UP, CHOCH_UP, BOS_DN, CHOCH_DN):
                    ob = _order_block(ev, i, o, h, l, c, lv)
                    if ob is not None:
                        zones.append(ob)
                        active.append(ob)
        # 3) liquidity pools: sweeps / takes of pools known before this bar, then
        #    drop candidate swings this close has broken
        for k, bk in books.items():
            esh, esl, eth, etl = bk.update(i, h[i], l[i], c[i])
            cols[f"{k}_eqh_sweep"][i], cols[f"{k}_eql_sweep"][i] = esh, esl
            cols[f"{k}_eqh_taken"][i], cols[f"{k}_eql_taken"][i] = eth, etl
            sweep_h[i] = max(sweep_h[i], esh)
            sweep_l[i] = max(sweep_l[i], esl)
            bk.on_bar_close(i, c[i])
        # 4) zone lifecycles (zones created at an EARLIER bar)
        still: list[Zone] = []
        for z in active:
            if z.created_idx >= i:
                still.append(z)
                continue
            _update_zone(z, i, h, l, c, cfg, zones, still)
            # an FVG stays TRACKED after it was filled (it can still invert by a close
            # through it); it is only no longer an entry zone (see Zone.active_at)
            tracked = z.inverted_idx is None if z.kind == "fvg" else z.invalidated_idx is None
            if tracked and i - z.created_idx <= cfg.zone_max_age:
                still.append(z)
        active = still
        # 5) new FVG at this close (from i+1 on it can be touched)
        if i >= 2:
            f = _fvg_at(i, o, h, l, c, atr_i, cfg)
            if f is not None:
                zones.append(f)
                active.append(f)

    bars = pd.DataFrame(cols, index=df.index)
    bars["sweep_high"], bars["sweep_low"] = sweep_h, sweep_l
    for k in [c for c in bars.columns if c.endswith(("_trend", "_event", "_sweep", "_taken"))] + ["sweep_high", "sweep_low"]:
        bars[k] = bars[k].astype(int)
    return StructureResult(bars=bars, swings={k: lv.filter.swings for k, lv in levels.items()},
                           zones=zones, pools=pools)


def _fvg_at(i: int, o, h, l, c, atr_i: float, cfg: StructureConfig) -> Zone | None:
    if not np.isfinite(atr_i):
        return None
    body_ok = cfg.displacement_atr <= 0 or abs(c[i - 1] - o[i - 1]) >= cfg.displacement_atr * atr_i
    if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= cfg.fvg_min_atr * atr_i and body_ok:
        return Zone("fvg", 1, top=float(l[i]), bottom=float(h[i - 2]), created_idx=i, origin_idx=i - 1)
    if h[i] < l[i - 2] and (l[i - 2] - h[i]) >= cfg.fvg_min_atr * atr_i and body_ok:
        return Zone("fvg", -1, top=float(l[i - 2]), bottom=float(h[i]), created_idx=i, origin_idx=i - 1)
    return None


def _update_zone(z: Zone, i: int, h, l, c, cfg: StructureConfig, zones: list[Zone], still: list[Zone]) -> None:
    inside = l[i] <= z.top and h[i] >= z.bottom
    if z.kind == "fvg":
        if inside and z.touched_idx is None:
            z.touched_idx = i
        if z.direction > 0:
            reached = (l[i] if cfg.fvg_fill == "wick" else c[i]) <= z.bottom
            through = c[i] < z.bottom
        else:
            reached = (h[i] if cfg.fvg_fill == "wick" else c[i]) >= z.top
            through = c[i] > z.top
        if reached and z.filled_idx is None:
            z.filled_idx = i
        if through and z.inverted_idx is None:
            z.inverted_idx = i
            inv = Zone("ifvg", -z.direction, top=z.top, bottom=z.bottom, created_idx=i, origin_idx=z.created_idx)
            zones.append(inv)
            still.append(inv)
        return
    # ifvg and ob: support (+1) / resistance (-1)
    if inside and z.retest_idx is None:
        z.retest_idx = i
    if z.direction > 0 and c[i] < z.bottom:
        z.invalidated_idx = i
    elif z.direction < 0 and c[i] > z.top:
        z.invalidated_idx = i


def _order_block(ev: int, i: int, o, h, l, c, lv: _Level) -> Zone | None:
    """Last opposite candle of the leg that produced the break."""
    if ev in (BOS_UP, CHOCH_UP):
        start = lv.protected_low_idx if lv.protected_low_idx >= 0 else max(0, i - 20)
        for j in range(i - 1, start - 1, -1):
            if c[j] < o[j]:
                return Zone("ob", 1, top=float(h[j]), bottom=float(l[j]), created_idx=i, origin_idx=j)
    else:
        start = lv.protected_high_idx if lv.protected_high_idx >= 0 else max(0, i - 20)
        for j in range(i - 1, start - 1, -1):
            if c[j] > o[j]:
                return Zone("ob", -1, top=float(h[j]), bottom=float(l[j]), created_idx=i, origin_idx=j)
    return None


def mid_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Mid OHLC from a bid/ask frame - structure should not be drawn on one side of the book."""
    if all(f"ask_{k}" in df for k in ("open", "high", "low", "close")):
        return pd.DataFrame({k: (df[k] + df[f"ask_{k}"]) / 2 for k in ("open", "high", "low", "close")},
                            index=df.index)
    return df[["open", "high", "low", "close"]]


__all__ = ["StructureConfig", "StructureResult", "Swing", "Zone", "Pool", "analyze", "raw_pivots",
           "mid_ohlc", "BOS_UP", "BOS_DN", "CHOCH_UP", "CHOCH_DN"]
