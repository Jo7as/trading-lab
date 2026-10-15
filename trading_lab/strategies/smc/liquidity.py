"""Session / daily / weekly liquidity pools and wick sweeps."""
from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .types import LiquidityLevel

_NY = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


def _parse_hhmm(s: str) -> time:
    h, m = s.strip().split(":")
    return time(int(h), int(m))


def _to_ny(ts: Any) -> datetime:
    if isinstance(ts, pd.Timestamp):
        dt = ts.to_pydatetime()
    else:
        dt = ts
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_UTC)
    return dt.astimezone(_NY)


def _in_session(ts: Any, start: time, end: time) -> bool:
    """True if NY local time is inside [start, end) — supports overnight."""
    t = _to_ny(ts).time()
    if start <= end:
        return start <= t < end
    # overnight e.g. 19:00 → 00:00
    return t >= start or t < end


def _session_date_key(ts: Any, start: time, end: time) -> str:
    """Stable key for which session instance a bar belongs to."""
    ny = _to_ny(ts)
    if start <= end:
        return ny.date().isoformat()
    # overnight: bars before `end` belong to previous local date's session
    if ny.time() < end:
        return (ny.date() - timedelta(days=1)).isoformat()
    return ny.date().isoformat()


class LiquidityEngine:
    """Build and query liquidity levels from OHLCV (typically 15m or 1m)."""

    def __init__(
        self,
        *,
        asia_start: str = "19:00",
        asia_end: str = "00:00",
        london_start: str = "02:00",
        london_end: str = "05:00",
        use_asia: bool = True,
        use_london: bool = True,
        use_pd: bool = True,
        use_pw: bool = True,
        use_swing_liq: bool = False,
        swing_liq_length: int = 5,
    ):
        self.asia_start = _parse_hhmm(asia_start)
        self.asia_end = _parse_hhmm(asia_end)
        self.london_start = _parse_hhmm(london_start)
        self.london_end = _parse_hhmm(london_end)
        self.use_asia = use_asia
        self.use_london = use_london
        self.use_pd = use_pd
        self.use_pw = use_pw
        self.use_swing_liq = use_swing_liq
        self.swing_liq_length = max(1, int(swing_liq_length))
        self.levels: list[LiquidityLevel] = []
        self._unswept: list[LiquidityLevel] = []
        self._created_ts: list[pd.Timestamp] = []
        self._built = False

    def _kind_allowed(self, kind: str) -> bool:
        if kind.startswith("asia_"):
            return self.use_asia
        if kind.startswith("london_"):
            return self.use_london
        if kind in ("pdh", "pdl"):
            return self.use_pd
        if kind in ("pwh", "pwl"):
            return self.use_pw
        if kind in ("swing_high", "swing_low"):
            return self.use_swing_liq
        return True

    def build(self, df: pd.DataFrame) -> None:
        """Precompute Asia/London ranges + PDH/PDL/PWH/PWL (causal per day)."""
        if df.empty:
            self.levels = []
            self._all_levels = []
            self._next_level_i = 0
            self._unswept = []
            self._built = True
            return

        idx = df.index
        highs = df["high"]
        lows = df["low"]

        levels: list[LiquidityLevel] = []

        # Session ranges
        sessions: list[tuple[str, time, time]] = []
        if self.use_asia:
            sessions.append(("asia", self.asia_start, self.asia_end))
        if self.use_london:
            sessions.append(("london", self.london_start, self.london_end))
        for name, start, end in sessions:
            buckets: dict[str, list[int]] = {}
            for i, ts in enumerate(idx):
                if _in_session(ts, start, end):
                    key = _session_date_key(ts, start, end)
                    buckets.setdefault(key, []).append(i)
            for key, ixs in buckets.items():
                if not ixs:
                    continue
                hi = float(highs.iloc[ixs].max())
                lo = float(lows.iloc[ixs].min())
                end_ts = idx[ixs[-1]]
                levels.append(
                    LiquidityLevel(hi, f"{name}_high", end_ts, swept=False)
                )
                levels.append(
                    LiquidityLevel(lo, f"{name}_low", end_ts, swept=False)
                )

        # Previous day / week — use calendar NY dates
        by_day: dict[str, list[int]] = {}
        by_week: dict[str, list[int]] = {}
        for i, ts in enumerate(idx):
            ny = _to_ny(ts)
            dkey = ny.date().isoformat()
            # ISO week key
            iso = ny.isocalendar()
            wkey = f"{iso.year}-W{iso.week:02d}"
            by_day.setdefault(dkey, []).append(i)
            by_week.setdefault(wkey, []).append(i)

        if self.use_pd:
            day_keys = sorted(by_day.keys())
            for di, dkey in enumerate(day_keys):
                if di == 0:
                    continue
                prev_ixs = by_day[day_keys[di - 1]]
                # Level becomes available at first bar of current day
                first_ts = idx[by_day[dkey][0]]
                levels.append(
                    LiquidityLevel(
                        float(highs.iloc[prev_ixs].max()),
                        "pdh",
                        first_ts,
                        swept=False,
                    )
                )
                levels.append(
                    LiquidityLevel(
                        float(lows.iloc[prev_ixs].min()),
                        "pdl",
                        first_ts,
                        swept=False,
                    )
                )

        if self.use_pw:
            week_keys = sorted(by_week.keys())
            for wi, wkey in enumerate(week_keys):
                if wi == 0:
                    continue
                prev_ixs = by_week[week_keys[wi - 1]]
                first_ts = idx[by_week[wkey][0]]
                levels.append(
                    LiquidityLevel(
                        float(highs.iloc[prev_ixs].max()),
                        "pwh",
                        first_ts,
                        swept=False,
                    )
                )
                levels.append(
                    LiquidityLevel(
                        float(lows.iloc[prev_ixs].min()),
                        "pwl",
                        first_ts,
                        swept=False,
                    )
                )

        # Local swing highs/lows (discretionary liquidity / equal highs)
        if self.use_swing_liq:
            from .structure import detect_swings

            swings = detect_swings(
                highs.to_numpy(dtype=float),
                lows.to_numpy(dtype=float),
                idx,
                self.swing_liq_length,
            )
            for s in swings:
                kind = "swing_high" if s.kind == 1 else "swing_low"
                levels.append(
                    LiquidityLevel(
                        float(s.price),
                        kind,
                        idx[s.confirmed_idx],
                        swept=False,
                    )
                )

        self.levels = [lv for lv in levels if self._kind_allowed(lv.kind)]
        for lv in self.levels:
            lv.created_unix = float(pd.Timestamp(lv.created_ts).timestamp())
        self.levels.sort(key=lambda lv: lv.created_unix)
        self._created_ts = [pd.Timestamp(lv.created_ts) for lv in self.levels]
        # Monotonic activation: future levels stay off _unswept until asof catches up
        self._all_levels: list[LiquidityLevel] = list(self.levels)
        self._next_level_i = 0
        self._unswept: list[LiquidityLevel] = []
        self._built = True

    def _activate_through(self, asof_unix: float) -> None:
        while (
            self._next_level_i < len(self._all_levels)
            and self._all_levels[self._next_level_i].created_unix <= asof_unix
        ):
            self._unswept.append(self._all_levels[self._next_level_i])
            self._next_level_i += 1

    @staticmethod
    def _asof_unix(asof_ts: Any) -> float:
        if isinstance(asof_ts, (int, float)):
            return float(asof_ts)
        return float(pd.Timestamp(asof_ts).timestamp())

    def active_levels(self, asof_ts: Any) -> list[LiquidityLevel]:
        """Levels whose created_ts <= asof (known at this time)."""
        if not self._built:
            return []
        asof = self._asof_unix(asof_ts)
        return [lv for lv in self.levels if lv.created_unix <= asof]

    def detect_sweeps(
        self,
        high: float,
        low: float,
        asof_ts: Any,
        *,
        allow_kind: Any | None = None,
    ) -> list[LiquidityLevel]:
        """Wick sweeps of unswept levels known at asof (mutate swept flags).

        ``allow_kind``: optional ``callable(kind) -> bool``. Disallowed kinds
        stay unswept without mark/restore thrash (important for filters).
        """
        swept_now: list[LiquidityLevel] = []
        asof = self._asof_unix(asof_ts)
        self._activate_through(asof)
        remaining: list[LiquidityLevel] = []
        for lv in self._unswept:
            if allow_kind is not None and not allow_kind(lv.kind):
                remaining.append(lv)
                continue
            is_high = lv.kind.endswith("_high") or lv.kind in ("pdh", "pwh")
            hit = (is_high and high > lv.price) or ((not is_high) and low < lv.price)
            if hit:
                lv.swept = True
                swept_now.append(lv)
            else:
                remaining.append(lv)
        self._unswept = remaining
        return swept_now

    def restore_level(self, lv: LiquidityLevel) -> None:
        """Undo a false-positive wick hit (e.g. Stage A close-back failed)."""
        lv.swept = False
        if lv not in self._unswept:
            self._unswept.append(lv)

    @staticmethod
    def is_buy_side(kind: str) -> bool:
        return kind.endswith("_high") or kind in ("pdh", "pwh", "swing_high")

    @staticmethod
    def is_sell_side(kind: str) -> bool:
        return kind.endswith("_low") or kind in ("pdl", "pwl", "swing_low")

    # HTF external draws preferred for ERL (beyond nearest IRL)
    _ERL_LONG = frozenset({"pdh", "pwh"})
    _ERL_SHORT = frozenset({"pdl", "pwl"})

    def trade_targets(
        self,
        entry: float,
        direction: str,
        asof_ts: Any,
        *,
        dealing_high: float | None = None,
        dealing_low: float | None = None,
        min_dist: float = 0.0,
        max_dist: float | None = None,
    ) -> tuple[float | None, float | None]:
        """Nearest IRL and next ERL in the trade direction.

        Long → buy-side pools above entry; short → sell-side below.
        ERL prefers the dealing-range extreme, else the *nearest* HTF
        (PDH/PWH or PDL/PWL) beyond IRL — never the farthest historical pool.
        ``max_dist`` caps how far candidates may sit from entry (absolute).
        """
        levels = self.active_levels(asof_ts)
        cands: list[tuple[float, str]] = []

        def _in_band(px: float, *, above: bool) -> bool:
            if above:
                if px <= entry + min_dist:
                    return False
                if max_dist is not None and px > entry + max_dist:
                    return False
                return True
            if px >= entry - min_dist:
                return False
            if max_dist is not None and px < entry - max_dist:
                return False
            return True

        if direction == "long":
            for lv in levels:
                if not self.is_buy_side(lv.kind):
                    continue
                if _in_band(lv.price, above=True):
                    cands.append((float(lv.price), lv.kind))
            if dealing_high is not None and _in_band(dealing_high, above=True):
                cands.append((float(dealing_high), "dealing_high"))
            if not cands:
                return None, None
            cands.sort(key=lambda x: x[0])
            irl = cands[0][0]
            # ERL: dealing extreme first, else nearest HTF beyond IRL, else next pool
            erl: float | None = None
            if dealing_high is not None and dealing_high > irl + 1e-12:
                if max_dist is None or dealing_high <= entry + max_dist:
                    erl = float(dealing_high)
            if erl is None:
                htf = sorted(
                    px
                    for px, kind in cands
                    if kind in self._ERL_LONG and px > irl + 1e-12
                )
                if htf:
                    erl = htf[0]
            if erl is None:
                nxt = [px for px, _ in cands if px > irl + 1e-12]
                if nxt:
                    erl = nxt[0]
            return irl, erl

        for lv in levels:
            if not self.is_sell_side(lv.kind):
                continue
            if _in_band(lv.price, above=False):
                cands.append((float(lv.price), lv.kind))
        if dealing_low is not None and _in_band(dealing_low, above=False):
            cands.append((float(dealing_low), "dealing_low"))
        if not cands:
            return None, None
        cands.sort(key=lambda x: x[0], reverse=True)
        irl = cands[0][0]
        erl = None
        if dealing_low is not None and dealing_low < irl - 1e-12:
            if max_dist is None or dealing_low >= entry - max_dist:
                erl = float(dealing_low)
        if erl is None:
            htf = sorted(
                (
                    px
                    for px, kind in cands
                    if kind in self._ERL_SHORT and px < irl - 1e-12
                ),
                reverse=True,
            )
            if htf:
                erl = htf[0]
        if erl is None:
            nxt = [px for px, _ in cands if px < irl - 1e-12]
            if nxt:
                erl = nxt[0]
        return irl, erl
