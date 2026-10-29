"""Time-based liquidity levels: previous day / week high-low, Asia range, opening range.

Stops and resting orders cluster at these levels (Osler, NY Fed SR 125/150); they are the
"external liquidity" of SMC/ICT. Next to the swing-based levels of
``trading_lab.market_structure`` they form the liquidity library. Definitions
(``docs/MARKET_STRUCTURE.md``, section "Time-based liquidity"):

* **trading day** - 17:00 New York to 17:00 New York (DST-aware, the FX/CME convention);
  the weekend has no day of its own, Monday's previous day is Friday.
* **pdh / pdl** - high / low of the previous trading day that has data; valid for the
  whole current day.
* **pwh / pwl** - high / low of the previous week (weeks start Sunday 17:00 NY).
* **asia** - high / low of 20:00-24:00 NY (ICT Asian range) of the current day; valid from
  00:00 NY to the day's end at 17:00 NY.
* **or** - opening range: the first ``or_minutes`` after the local open (US indices 09:30
  NY, GER40 09:00 Berlin, FX and metals 08:00 London); valid from the end of the range to
  the end of that local date.

Every value is built from completed 1m mid bars only and is known at the START of the
period it is valid in - no look-ahead. ``events`` finds the FIRST minute of a period that
trades beyond a level and classifies the signal-timeframe bar that contains it: close back
inside = ``reclaim`` (a sweep), close beyond = ``through`` (a break).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .market_structure import mid_ohlc

NY = "America/New_York"
OPEN = {"US100": (NY, "09:30"), "SP500": (NY, "09:30"), "US30": (NY, "09:30"),
        "GER40": ("Europe/Berlin", "09:00")}
OPEN_DEFAULT = ("Europe/London", "08:00")          # FX majors, metals: London open
LEVELS = ("pd", "pw", "asia", "or")


@dataclass(frozen=True)
class LiquidityConfig:
    or_minutes: int = 30
    asia_start_ny: int = 20                       # hour, NY wall clock
    asia_end_ny: int = 24


@dataclass
class LevelTable:
    """One row per validity period: [valid_from, valid_to) in UTC ns and the two levels."""
    name: str
    valid_from: np.ndarray
    valid_to: np.ndarray
    high: np.ndarray
    low: np.ndarray

    def at(self, t_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(high, low, period row) valid at each time; NaN / -1 outside every period."""
        t_ns = np.asarray(t_ns, dtype=np.int64)
        row = np.searchsorted(self.valid_from, t_ns, "right") - 1
        ok = (row >= 0) & (t_ns < self.valid_to[np.clip(row, 0, None)])
        row = np.where(ok, row, -1)
        hi = np.where(ok, self.high[np.clip(row, 0, None)], np.nan)
        lo = np.where(ok, self.low[np.clip(row, 0, None)], np.nan)
        return hi, lo, row


def _ns(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.as_unit("ns").asi8


def trading_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Label (naive NY date the day STARTS on, at 17:00) of the 17:00-NY trading day."""
    ny = index.tz_convert(NY)
    return (ny - pd.Timedelta(hours=17)).normalize().tz_localize(None)


def _day_start_ns(label: pd.DatetimeIndex) -> np.ndarray:
    return _ns((label + pd.Timedelta(hours=17)).tz_localize(NY).tz_convert("UTC"))


def _week_label(day: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Sunday label of the week a trading day belongs to (Sunday-17:00 day = first day)."""
    return day - pd.to_timedelta((day.dayofweek + 1) % 7, unit="D")


def _previous_period(name: str, key: pd.DatetimeIndex, high: np.ndarray, low: np.ndarray,
                     start_ns_of) -> LevelTable:
    g = pd.DataFrame({"k": key, "h": high, "l": low}).groupby("k").agg(h=("h", "max"), l=("l", "min"))
    keys = pd.DatetimeIndex(g.index)
    start = start_ns_of(keys)
    # a period is valid from its own start to the next period's start, with the PREVIOUS
    # period's values (the first period has nothing before it)
    vf, vt = start[1:], np.r_[start[2:], np.iinfo(np.int64).max]
    return LevelTable(name, vf, vt, g["h"].to_numpy()[:-1], g["l"].to_numpy()[:-1])


def previous_day(m1: pd.DataFrame) -> LevelTable:
    mid = mid_ohlc(m1)
    return _previous_period("pd", trading_day(mid.index), mid["high"].to_numpy(), mid["low"].to_numpy(),
                            _day_start_ns)


def previous_week(m1: pd.DataFrame) -> LevelTable:
    mid = mid_ohlc(m1)
    return _previous_period("pw", _week_label(trading_day(mid.index)), mid["high"].to_numpy(),
                            mid["low"].to_numpy(), _day_start_ns)


def asia_range(m1: pd.DataFrame, cfg: LiquidityConfig = LiquidityConfig()) -> LevelTable:
    mid = mid_ohlc(m1)
    ny = mid.index.tz_convert(NY)
    sel = (ny.hour >= cfg.asia_start_ny) & (ny.hour < cfg.asia_end_ny)
    day = trading_day(mid.index)[sel]
    g = pd.DataFrame({"k": day, "h": mid["high"].to_numpy()[sel], "l": mid["low"].to_numpy()[sel]}
                     ).groupby("k").agg(h=("h", "max"), l=("l", "min"))
    keys = pd.DatetimeIndex(g.index)
    end_ny = keys + pd.Timedelta(hours=cfg.asia_end_ny)               # 24 -> 00:00 next date
    vf = _ns(end_ny.tz_localize(NY).tz_convert("UTC"))
    vt = _day_start_ns(keys + pd.Timedelta(days=1))                   # 17:00 NY, day's end
    return LevelTable("asia", vf, vt, g["h"].to_numpy(), g["l"].to_numpy())


def opening_range(m1: pd.DataFrame, asset: str, cfg: LiquidityConfig = LiquidityConfig()) -> LevelTable:
    tz, hhmm = OPEN.get(asset, OPEN_DEFAULT)
    mid = mid_ohlc(m1)
    loc = mid.index.tz_convert(tz)
    h, m = (int(x) for x in hhmm.split(":"))
    mins = loc.hour * 60 + loc.minute
    sel = (mins >= h * 60 + m) & (mins < h * 60 + m + cfg.or_minutes)
    date = loc.normalize().tz_localize(None)[sel]
    g = pd.DataFrame({"k": date, "h": mid["high"].to_numpy()[sel], "l": mid["low"].to_numpy()[sel]}
                     ).groupby("k").agg(h=("h", "max"), l=("l", "min"))
    keys = pd.DatetimeIndex(g.index)
    vf = _ns((keys + pd.Timedelta(hours=h, minutes=m + cfg.or_minutes)).tz_localize(tz).tz_convert("UTC"))
    vt = _ns((keys + pd.Timedelta(days=1)).tz_localize(tz).tz_convert("UTC"))
    return LevelTable("or", vf, vt, g["h"].to_numpy(), g["l"].to_numpy())


def tables(m1: pd.DataFrame, asset: str, cfg: LiquidityConfig = LiquidityConfig()) -> dict[str, LevelTable]:
    return {"pd": previous_day(m1), "pw": previous_week(m1), "asia": asia_range(m1, cfg),
            "or": opening_range(m1, asset, cfg)}


@dataclass
class Event:
    level: str
    side: int               # +1 the high was exceeded, -1 the low
    kind: str               # reclaim | through
    bar: int                # signal-timeframe bar that contains the first exceeding minute
    minute: int             # index of that minute in the 1m frame
    price: float            # the level


def events(m1: pd.DataFrame, bars: pd.DataFrame, tf: str, table: LevelTable) -> list[Event]:
    """First exceed of each level per validity period, classified on the ``tf`` bar close.

    ``bars`` are the mid OHLC bars of ``tf`` (index = bar start). The first minute at/after
    ``valid_from`` whose mid high (low) is beyond the level counts; later exceeds in the same
    period are ignored."""
    mid = mid_ohlc(m1)
    t = _ns(mid.index)
    hi, lo = mid["high"].to_numpy(), mid["low"].to_numpy()
    b_start = _ns(bars.index)
    b_close = bars["close"].to_numpy()
    width = pd.Timedelta(tf).value
    out: list[Event] = []
    for p in range(len(table.valid_from)):
        a = int(np.searchsorted(t, table.valid_from[p], "left"))
        z = int(np.searchsorted(t, table.valid_to[p], "left"))
        if z <= a:
            continue
        for side, lvl, path in ((1, table.high[p], hi[a:z]), (-1, table.low[p], lo[a:z])):
            if not np.isfinite(lvl):
                continue
            beyond = path > lvl if side > 0 else path < lvl
            if not beyond.any():
                continue
            k = a + int(np.argmax(beyond))
            i = int(np.searchsorted(b_start, t[k], "right")) - 1
            if i < 0 or t[k] >= b_start[i] + width:
                continue
            kind = "through" if side * (b_close[i] - lvl) > 0 else "reclaim"
            out.append(Event(table.name, side, kind, i, k, float(lvl)))
    out.sort(key=lambda e: (e.bar, e.level, e.side))
    return out


__all__ = ["LiquidityConfig", "LevelTable", "Event", "LEVELS", "trading_day", "previous_day", "previous_week",
           "asia_range", "opening_range", "tables", "events"]
