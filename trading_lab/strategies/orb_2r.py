"""Opening Range Breakout, fixed 2R target (US100 / GER40 / XAUUSD / JP225).

Rules (times are exchange-local, so DST shifts are handled by the timezone):
  1. Range = bid high/low of the 1m bars in [session_open, +range_minutes)
  2. Signal = first ``signal_bar_minutes`` bar after the range that *closes*
     beyond the range, within ``signal_window_minutes``; one trade per day
  3. Entry = market at the open of the next 1m bar (long buys ask, short sells bid)
  4. Stop = opposite range side ± ``stop_buffer_spreads`` × spread,
     1R = |entry − stop|, target = entry ± ``target_r`` × R (limit)
  5. Flat at ``flat_time``; no trailing, no partials
  6. Cost filter: skip when (spread + commission + slippage) / R > ``max_cost_r``
  7. Optional day filters (all off by default, all known before the entry):
     range / daily ATR at or above its trailing median, breakout with the
     daily SMA trend, breakout in the direction of the opening gap, no FOMC days

Results are in R. Costs sit in the fills: bid/ask spread from the data,
``slippage`` (+ ``slippage_pct`` × price) on every market/stop fill, ``commission`` per round trip.
``cost_mult`` scales all three for the fills only (0 = gross, 2 = stress); the
cost filter and the stop buffer always see the nominal costs, so the trade
geometry stays comparable between runs.

This is a fixed-rule test, deliberately separate from the evolvable
``opening_range_breakout`` (entry on the signal close, range-multiple target).
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from dateutil.easter import easter

from ..mr_common import ASK_COLS, ensure_utc

_MIN_NS = 60 * 10**9
_MAX_ENTRY_GAP_MIN = 5       # signal → entry bar; longer means a data hole
_MAX_FLAT_GAP_MIN = 30       # flat_time → next bar; longer means the market shut
_MIN_RANGE_COVERAGE = 0.8    # share of 1m bars that must exist in the range window


@dataclass(frozen=True)
class OrbConfig:
    """Config for the 15-minute opening range breakout with a fixed R target."""

    asset: str = "US100"
    asset_label: str = "US100"
    timeframe: str = "1min"           # bar size of the input frame

    tz: str = "America/New_York"
    calendar: str = "nyse"            # "nyse" | "xetra" — full and half holidays are skipped
    session_open: str = "09:30"
    flat_time: str = "15:55"

    range_minutes: int = 15
    signal_bar_minutes: int = 5
    signal_window_minutes: int = 120

    target_r: float = 2.0
    stop_buffer_spreads: float = 1.0

    slippage: float = 1.0             # price points per market / stop fill
    slippage_pct: float = 0.0         # plus this fraction of the price (scales with the index level)
    commission: float = 0.0           # price points per round trip
    max_cost_r: float = 0.04          # 0 = filter off
    max_range_mult: float = 0.0       # 0 = off; skip when range > mult × mean of the last N days
    range_lookback_days: int = 20

    # Day filters, 0 / False = off. "Daily" means session bars: open → flat_time.
    atr_period_daily: int = 14
    range_atr_median_days: int = 0    # trade only if range/ATR ≥ its median of the last N days
    trend_sma_days: int = 0           # long only above, short only below the SMA of N daily closes
    gap_filter: bool = False          # breakout must point the same way as open vs. prior close
    skip_fomc: bool = False           # no trade on FOMC decision days

    @staticmethod
    def search_space() -> list[tuple[str, float, float, float]]:
        return [
            ("range_minutes", 5, 60, 5),
            ("signal_window_minutes", 60, 240, 30),
            ("target_r", 1.0, 4.0, 0.25),
            ("stop_buffer_spreads", 0.0, 3.0, 0.5),
            ("max_cost_r", 0.02, 0.08, 0.005),
        ]


# Exchange-local equivalents of the German times in the plan
MARKETS: dict[str, OrbConfig] = {
    "US100": OrbConfig(),
    "GER40": OrbConfig(
        asset="GER40", asset_label="GER40", tz="Europe/Berlin", calendar="xetra",
        session_open="09:00", flat_time="17:25",
    ),
    "XAUUSD": OrbConfig(
        asset="XAUUSD", asset_label="XAU/USD", session_open="08:20", flat_time="16:55",
        slippage=0.10, commission=0.06,
    ),
    # A1 — Nikkei opening range. Tokyo opens 09:00 JST (00:00 UTC, no DST); flat
    # before the 15:00/15:30 cash close so the 11:30–12:30 lunch gap stays inside
    # the holding window and never falls on an exit.
    "JP225": OrbConfig(
        asset="JP225", asset_label="JP225", tz="Asia/Tokyo", calendar="tse",
        session_open="09:00", flat_time="14:55",
    ),
}


def _slippage(cfg: OrbConfig, price: float) -> float:
    return cfg.slippage + cfg.slippage_pct * price


# ── Session calendar ────────────────────────────────────────────────


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th weekday of the month; n = -1 for the last one."""
    if n > 0:
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    d = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


# Vernal / autumnal equinox days (Japan), 2016–2026. The equinox is a national
# holiday whose civil date is fixed by astronomical decree, not by a formula.
_JP_EQUINOX: dict[int, tuple[int, int]] = {
    2016: (20, 22), 2017: (20, 23), 2018: (21, 23), 2019: (21, 23),
    2020: (20, 22), 2021: (20, 23), 2022: (21, 23), 2023: (21, 23),
    2024: (20, 22), 2025: (20, 23), 2026: (20, 23),
}


def _jpx_holidays(year: int) -> set[date]:
    """Tokyo Stock Exchange full-day closures for one year.

    Japan has no exchange half-days, so only full holidays are returned: the
    16 national holidays, the citizens' holidays that fall between two of them,
    their substitute holidays when a holiday lands on a Sunday, and the
    exchange's own New-Year (Jan 2/3) and year-end (Dec 31) closures. One-off
    closures (the 2019 imperial transition, the 2020-10-01 system outage) and
    the Tokyo-Olympics holiday moves of 2020/2021 are included.
    """
    spring, autumnal = _JP_EQUINOX[year]
    national: set[date] = {
        date(year, 1, 1),                      # New Year's Day
        date(year, 2, 11),                     # National Foundation Day
        date(year, 3, spring),                 # Vernal Equinox Day
        date(year, 4, 29),                     # Showa Day
        date(year, 5, 3),                      # Constitution Memorial Day
        date(year, 5, 4),                      # Greenery Day
        date(year, 5, 5),                      # Children's Day
        date(year, 9, autumnal),               # Autumnal Equinox Day
        date(year, 11, 3),                     # Culture Day
        date(year, 11, 23),                    # Labor Thanksgiving Day
        _nth_weekday(year, 1, 0, 2),           # Coming of Age Day (2nd Mon Jan)
        _nth_weekday(year, 9, 0, 3),           # Respect for the Aged Day (3rd Mon Sep)
    }
    if year >= 2016:
        national.add(date(year, 8, 11))        # Mountain Day (since 2016)
    if year >= 2003:
        national.add(_nth_weekday(year, 7, 0, 3))  # Marine Day (3rd Mon Jul)
    if year == 2020:                           # Sports Day moved for the Olympics
        national.add(date(2020, 7, 24))
    elif year == 2021:
        national.add(date(2021, 7, 22))        # Marine Day moved
        national.add(date(2021, 7, 23))        # Sports Day moved
    elif year >= 2000:
        national.add(_nth_weekday(year, 10, 0, 2))  # Sports Day (2nd Mon Oct)
    if year <= 2018:
        national.add(date(year, 12, 23))       # Emperor's Birthday (until 2018)
    elif year >= 2020:
        national.add(date(year, 2, 23))        # Emperor's Birthday (from 2020)
    if year == 2019:                           # imperial transition
        national |= {date(2019, 5, 1), date(2019, 10, 22)}

    holidays = set(national)
    # Citizens' holidays: the single weekday squeezed between two national holidays.
    for d in list(national):
        if d + timedelta(days=2) in national and d + timedelta(days=1) not in national:
            holidays.add(d + timedelta(days=1))
    # Substitute holidays: a Sunday national holiday moves to the next non-holiday day.
    for d in sorted(national):
        if d.weekday() == 6:
            sub = d + timedelta(days=1)
            while sub in holidays:
                sub += timedelta(days=1)
            holidays.add(sub)
    # Exchange closures that are not national holidays.
    holidays |= {date(year, 1, 2), date(year, 1, 3), date(year, 12, 31)}
    if year == 2020:
        holidays.add(date(2020, 10, 1))        # system outage, market shut
    return holidays


def skipped_days(calendar: str, year: int) -> set[date]:
    """Exchange holidays and shortened sessions of one year."""
    good_friday = easter(year) - timedelta(days=2)
    if calendar == "tse":
        return _jpx_holidays(year)
    if calendar == "xetra":
        days = {
            date(year, 1, 1), good_friday, easter(year) + timedelta(days=1),
            date(year, 5, 1), date(year, 12, 24), date(year, 12, 25),
            date(year, 12, 26), date(year, 12, 31),
            date(year, 12, 30),  # last trading day closes at 14:00
        }
        if year <= 2021:  # trading days on Xetra since 2022
            days |= {easter(year) + timedelta(days=50), date(year, 10, 3)}
        return days

    thanksgiving = _nth_weekday(year, 11, 3, 4)
    days = {
        _nth_weekday(year, 1, 0, 3), _nth_weekday(year, 2, 0, 3), good_friday,
        _nth_weekday(year, 5, 0, -1), _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1), thanksgiving, _observed(date(year, 12, 25)),
        # shortened sessions
        thanksgiving + timedelta(days=1), date(year, 12, 24), date(year, 7, 3),
    }
    new_year = date(year, 1, 1)
    days.add(new_year + timedelta(days=1) if new_year.weekday() == 6 else new_year)
    if year >= 2022:
        days.add(_observed(date(year, 6, 19)))
    days |= {d for d in (date(2018, 12, 5), date(2025, 1, 9)) if d.year == year}
    return days


# FOMC rate decisions (New York dates), incl. the unscheduled cut of 2020-03-03
FOMC_DAYS: frozenset[date] = frozenset(
    date(y, m, d)
    for y, days in {
        2016: [(1, 27), (3, 16), (4, 27), (6, 15), (7, 27), (9, 21), (11, 2), (12, 14)],
        2017: [(2, 1), (3, 15), (5, 3), (6, 14), (7, 26), (9, 20), (11, 1), (12, 13)],
        2018: [(1, 31), (3, 21), (5, 2), (6, 13), (8, 1), (9, 26), (11, 8), (12, 19)],
        2019: [(1, 30), (3, 20), (5, 1), (6, 19), (7, 31), (9, 18), (10, 30), (12, 11)],
        2020: [(1, 29), (3, 3), (4, 29), (6, 10), (7, 29), (9, 16), (11, 5), (12, 16)],
        2021: [(1, 27), (3, 17), (4, 28), (6, 16), (7, 28), (9, 22), (11, 3), (12, 15)],
        2022: [(1, 26), (3, 16), (5, 4), (6, 15), (7, 27), (9, 21), (11, 2), (12, 14)],
        2023: [(2, 1), (3, 22), (5, 3), (6, 14), (7, 26), (9, 20), (11, 1), (12, 13)],
        2024: [(1, 31), (3, 20), (5, 1), (6, 12), (7, 31), (9, 18), (11, 7), (12, 18)],
        2025: [(1, 29), (3, 19), (5, 7), (6, 18), (7, 30), (9, 17), (10, 29), (12, 10)],
        2026: [(1, 28), (3, 18), (4, 29), (6, 17), (7, 29), (9, 16), (10, 28), (12, 9)],
    }.items()
    for m, d in days
)


# ── Prepared data ───────────────────────────────────────────────────


@dataclass
class Sessions:
    """1m arrays plus one row per tradeable session day."""

    ts: np.ndarray            # bar open, UTC ns
    bid: dict[str, np.ndarray]
    ask: dict[str, np.ndarray]  # fill side, spread scaled by cost_mult
    spread_close: np.ndarray  # nominal ask close − bid close
    days: list[date]
    open_ns: np.ndarray
    flat_ns: np.ndarray
    cost_mult: float
    day_ohlc: pd.DataFrame | None = None  # session open/high/low/close per day, built on demand


def _local_ns(days: pd.DatetimeIndex, hhmm: str, tz: str) -> np.ndarray:
    h, m = (int(x) for x in hhmm.split(":"))
    local = (days + pd.Timedelta(hours=h, minutes=m)).tz_localize(tz)
    return local.tz_convert("UTC").as_unit("ns").asi8


def build_sessions(df: pd.DataFrame, cfg: OrbConfig, cost_mult: float = 1.0) -> Sessions:
    """Arrays and session clock for a 1m bid+ask frame."""
    if not all(c in df.columns for c in ASK_COLS):
        raise ValueError("ORB needs a bid+ask frame (ask_open/high/low/close)")
    data = ensure_utc(df)
    bid = {k: data[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close")}
    ask = {
        k: bid[k] + cost_mult * (data[f"ask_{k}"].to_numpy(dtype=float) - bid[k])
        for k in ("open", "high", "low", "close")
    }
    spread_close = data["ask_close"].to_numpy(dtype=float) - bid["close"]

    local_days = data.index.tz_convert(cfg.tz).tz_localize(None).normalize().unique()
    skip: set[date] = set()
    for year in range(local_days[0].year, local_days[-1].year + 1):
        skip |= skipped_days(cfg.calendar, year)
    keep = [d for d in local_days if d.weekday() < 5 and d.date() not in skip]
    days = pd.DatetimeIndex(keep)
    return Sessions(
        ts=data.index.as_unit("ns").asi8,
        bid=bid,
        ask=ask,
        spread_close=spread_close,
        days=[d.date() for d in days],
        open_ns=_local_ns(days, cfg.session_open, cfg.tz),
        flat_ns=_local_ns(days, cfg.flat_time, cfg.tz),
        cost_mult=cost_mult,
    )


def day_context(s: Sessions, cfg: OrbConfig) -> pd.DataFrame:
    """Per session day, from earlier days only: prev_close, atr, sma. Plus today's open."""
    if s.day_ohlc is None:
        rows = np.full((len(s.days), 4), np.nan)
        for d in range(len(s.days)):
            i0, i1 = np.searchsorted(s.ts, [s.open_ns[d], s.flat_ns[d]], side="left")
            if i1 > i0:
                rows[d] = (
                    s.bid["open"][i0], s.bid["high"][i0:i1].max(),
                    s.bid["low"][i0:i1].min(), s.bid["close"][i1 - 1],
                )
        s.day_ohlc = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    days = s.day_ohlc.dropna()
    prev_close = days["close"].shift(1)
    true_range = pd.concat(
        [days["high"] - days["low"], (days["high"] - prev_close).abs(), (days["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    ctx = pd.DataFrame(
        {
            "open": days["open"],
            "prev_close": prev_close,
            "atr": true_range.rolling(cfg.atr_period_daily).mean().shift(1),
            "sma": days["close"].rolling(max(cfg.trend_sma_days, 1)).mean().shift(1),
        }
    )
    return ctx.reindex(s.day_ohlc.index)


# ── Trade simulation ────────────────────────────────────────────────


def simulate_exit(
    s: Sessions,
    i_entry: int,
    flat_ns: int,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    slippage: float,
) -> tuple[float, str, int]:
    """(exit price, reason, bar index) for a position opened at bar ``i_entry``.

    Exit side prices: bid for longs, ask for shorts. When stop and target fall
    into the same bar the stop wins. A stop that is gapped through fills at
    the bar open; the target is a limit and fills at its level.
    """
    px = s.bid if direction == "long" else s.ask
    i_flat = int(np.searchsorted(s.ts, flat_ns, side="left"))
    hi, lo = px["high"][i_entry:i_flat], px["low"][i_entry:i_flat]
    if direction == "long":
        stop_hit, target_hit = lo <= stop, hi >= target
    else:
        stop_hit, target_hit = hi >= stop, lo <= target
    k_stop = int(stop_hit.argmax()) if stop_hit.any() else -1
    k_target = int(target_hit.argmax()) if target_hit.any() else -1

    if k_stop >= 0 and (k_target < 0 or k_stop <= k_target):
        i = i_entry + k_stop
        if direction == "long":
            return min(stop, px["open"][i]) - slippage, "stop", i
        return max(stop, px["open"][i]) + slippage, "stop", i
    if k_target >= 0:
        return target, "target", i_entry + k_target

    # Session end: market order at the first bar from flat_time on
    if i_flat < len(s.ts) and s.ts[i_flat] - flat_ns <= _MAX_FLAT_GAP_MIN * _MIN_NS:
        i, price = i_flat, px["open"][i_flat]
    else:
        i, price = i_flat - 1, px["close"][i_flat - 1]
    return (price - slippage if direction == "long" else price + slippage), "flat", i


def find_signal(s: Sessions, d: int, cfg: OrbConfig) -> dict[str, Any]:
    """Range and first breakout of day ``d``. ``skip`` is set when there is no trade."""
    open_ns = int(s.open_ns[d])
    range_end = open_ns + cfg.range_minutes * _MIN_NS
    i0, i1 = (int(x) for x in np.searchsorted(s.ts, [open_ns, range_end], side="left"))
    if i1 - i0 < _MIN_RANGE_COVERAGE * cfg.range_minutes:
        return {"skip": "no_data"}
    rh = float(s.bid["high"][i0:i1].max())
    rl = float(s.bid["low"][i0:i1].min())
    out: dict[str, Any] = {"range_high": rh, "range_low": rl, "range": rh - rl, "i_range_end": i1}
    if rh <= rl:
        return {**out, "skip": "no_data"}

    n_bars = cfg.signal_window_minutes // cfg.signal_bar_minutes
    edges = range_end + np.arange(n_bars + 1) * cfg.signal_bar_minutes * _MIN_NS
    idx = np.searchsorted(s.ts, edges, side="left")
    out["i_window_end"] = int(idx[-1])
    for k in range(n_bars):
        if idx[k + 1] == idx[k]:
            continue  # empty bucket
        j = int(idx[k + 1]) - 1  # last 1m bar of the signal bar
        close = s.bid["close"][j]
        if close > rh:
            direction = "long"
        elif close < rl:
            direction = "short"
        else:
            continue
        i_entry = int(idx[k + 1])
        flat_ns = int(s.flat_ns[d])
        if (
            i_entry >= len(s.ts)
            or s.ts[i_entry] - edges[k + 1] > _MAX_ENTRY_GAP_MIN * _MIN_NS
            or s.ts[i_entry] >= flat_ns
        ):
            return {**out, "skip": "no_entry_bar"}
        spread = float(s.spread_close[j])
        buffer = cfg.stop_buffer_spreads * spread
        if direction == "long":
            stop = rl - buffer
            risk_est = (close + spread) - stop
        else:
            stop = rh + buffer
            risk_est = stop - close
        cost_r = (spread + cfg.commission + _slippage(cfg, close)) / risk_est
        out.update(
            direction=direction, i_signal=j, i_entry=i_entry, stop=stop,
            spread=spread, cost_r_est=float(cost_r),
        )
        if cfg.max_cost_r > 0 and cost_r > cfg.max_cost_r:
            out["skip"] = "cost_filter"
        return out
    return {**out, "skip": "no_breakout"}


def run_backtest(
    data: pd.DataFrame | Sessions, cfg: OrbConfig, *, cost_mult: float = 1.0
) -> dict[str, Any]:
    """ORB on a 1m bid+ask frame (or prepared ``Sessions``). Trade results in R."""
    s = data if isinstance(data, Sessions) else build_sessions(data, cfg, cost_mult)
    commission = cfg.commission * s.cost_mult

    trades: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    recent_ranges: deque[float] = deque(maxlen=cfg.range_lookback_days)

    day_filters = cfg.range_atr_median_days > 0 or cfg.trend_sma_days > 0 or cfg.gap_filter
    ctx = day_context(s, cfg) if day_filters else None
    ctx_open, ctx_prev, ctx_atr, ctx_sma = (
        (ctx[c].to_numpy() for c in ("open", "prev_close", "atr", "sma")) if ctx is not None else (None,) * 4
    )
    recent_ratios: deque[float] = deque(maxlen=max(cfg.range_atr_median_days, 1))

    for d in range(len(s.days)):
        sig = find_signal(s, d, cfg)
        rng = sig.get("range")
        small_range = False
        if cfg.range_atr_median_days > 0 and rng and ctx_atr[d] > 0:
            ratio = rng / ctx_atr[d]
            # no verdict before the window is full → no trade
            small_range = len(recent_ratios) < cfg.range_atr_median_days or ratio < float(
                np.median(recent_ratios)
            )
            recent_ratios.append(ratio)
        elif cfg.range_atr_median_days > 0:
            small_range = True
        wide = (
            cfg.max_range_mult > 0
            and rng is not None
            and len(recent_ranges) == cfg.range_lookback_days
            and rng > cfg.max_range_mult * float(np.mean(recent_ranges))
        )
        if rng:
            recent_ranges.append(rng)
        if "skip" in sig:
            skipped[sig["skip"]] += 1
            continue
        if wide:
            skipped["range_filter"] += 1
            continue
        if small_range:
            skipped["range_atr_filter"] += 1
            continue
        if cfg.skip_fomc and s.days[d] in FOMC_DAYS:
            skipped["fomc"] += 1
            continue

        direction, i_entry, stop = sig["direction"], sig["i_entry"], sig["stop"]
        sign = 1.0 if direction == "long" else -1.0
        # NaN context (first days) compares False → no trade
        if cfg.trend_sma_days > 0 and not sign * (ctx_prev[d] - ctx_sma[d]) > 0:
            skipped["trend_filter"] += 1
            continue
        if cfg.gap_filter and not sign * (ctx_open[d] - ctx_prev[d]) > 0:
            skipped["gap_filter"] += 1
            continue
        slippage = _slippage(cfg, s.bid["open"][i_entry]) * s.cost_mult
        if direction == "long":
            entry = s.ask["open"][i_entry] + slippage
            risk = entry - stop
            target = entry + cfg.target_r * risk
        else:
            entry = s.bid["open"][i_entry] - slippage
            risk = stop - entry
            target = entry - cfg.target_r * risk
        if risk <= 0:
            skipped["no_entry_bar"] += 1
            continue

        exit_price, reason, i_exit = simulate_exit(
            s, i_entry, int(s.flat_ns[d]), direction, entry, stop, target, slippage
        )
        move = exit_price - entry if direction == "long" else entry - exit_price
        trades.append(
            {
                "day": s.days[d].isoformat(),
                "day_idx": d,
                "direction": direction,
                "range_high": sig["range_high"],
                "range_low": sig["range_low"],
                "signal_time": pd.Timestamp(s.ts[sig["i_signal"]] + _MIN_NS, tz="UTC").isoformat(),
                "entry_time": pd.Timestamp(s.ts[i_entry], tz="UTC").isoformat(),
                "entry": round(float(entry), 5),
                "stop": round(float(stop), 5),
                "target": round(float(target), 5),
                "exit_time": pd.Timestamp(s.ts[i_exit], tz="UTC").isoformat(),
                "exit": round(float(exit_price), 5),
                "exit_reason": reason,
                "risk_points": float(risk),
                "spread": sig["spread"],
                "cost_r_est": sig["cost_r_est"],
                "r": float((move - commission) / risk),
                "i_range_end": sig["i_range_end"],
                "i_window_end": sig["i_window_end"],
            }
        )
    return {"trades": trades, "skipped": dict(skipped), "days": len(s.days), "sessions": s}


def random_baseline(
    s: Sessions,
    cfg: OrbConfig,
    trades: list[dict[str, Any]],
    *,
    runs: int = 1000,
    seed: int = 7,
) -> np.ndarray:
    """R of random entries on the strategy's own trade days, shape (runs, trades).

    Per day: a random 1m bar of the signal window, a random direction, the
    strategy's stop distance of that day, the same R target, flat time and
    costs. Every (bar, direction) outcome is simulated once and then sampled.
    """
    rng = np.random.default_rng(seed)
    commission = cfg.commission * s.cost_mult
    out = np.zeros((runs, len(trades)))
    for t, trade in enumerate(trades):
        flat_ns = int(s.flat_ns[trade["day_idx"]])
        risk = trade["risk_points"]
        i_last = min(trade["i_window_end"], int(np.searchsorted(s.ts, flat_ns, side="left")))
        bars = range(trade["i_range_end"], i_last)
        if not bars:
            continue
        outcomes = np.empty((len(bars), 2))
        for b, i in enumerate(bars):
            slippage = _slippage(cfg, s.bid["open"][i]) * s.cost_mult
            for side, direction in enumerate(("long", "short")):
                if direction == "long":
                    entry = s.ask["open"][i] + slippage
                    stop, target, sign = entry - risk, entry + cfg.target_r * risk, 1.0
                else:
                    entry = s.bid["open"][i] - slippage
                    stop, target, sign = entry + risk, entry - cfg.target_r * risk, -1.0
                exit_price, _, _ = simulate_exit(
                    s, i, flat_ns, direction, entry, stop, target, slippage
                )
                outcomes[b, side] = (sign * (exit_price - entry) - commission) / risk
        out[:, t] = outcomes[rng.integers(0, len(bars), runs), rng.integers(0, 2, runs)]
    return out
