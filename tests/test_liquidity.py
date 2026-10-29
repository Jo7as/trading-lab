"""Time-based liquidity levels (trading_lab/liquidity.py) on full 24 h 1m frames."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_lab import liquidity as L
from trading_lab.mr_common import resample_ohlc


def _m1(start: str, end: str, base: float = 1.0) -> pd.DataFrame:
    t = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    c = np.full(len(t), base)
    return pd.DataFrame({"open": c, "high": c + 0.001, "low": c - 0.001, "close": c}, index=t)


def _spike(df: pd.DataFrame, when: str, high: float | None = None, low: float | None = None,
           close: float | None = None) -> None:
    k = df.index.get_loc(pd.Timestamp(when, tz="UTC"))
    if high is not None:
        df.iloc[k, df.columns.get_loc("high")] = high
    if low is not None:
        df.iloc[k, df.columns.get_loc("low")] = low
    if close is not None:
        df.iloc[k, df.columns.get_loc("close")] = close


def _at(tab: L.LevelTable, when: str) -> tuple[float, float]:
    hi, lo, _ = tab.at(np.array([pd.Timestamp(when, tz="UTC").value]))
    return float(hi[0]), float(lo[0])


# ── the 17:00 New York day ─────────────────────────────────────────────────

def test_day_boundary_is_17_ny_and_dst_aware() -> None:
    # summer: 17:00 NY = 21:00 UTC; winter: 22:00 UTC
    s = pd.DatetimeIndex(["2024-07-02 20:59", "2024-07-02 21:00", "2024-01-09 21:59", "2024-01-09 22:00"], tz="UTC")
    d = L.trading_day(s)
    assert d[0] != d[1] and d[2] != d[3]
    assert d[1] == pd.Timestamp("2024-07-02") and d[3] == pd.Timestamp("2024-01-09")


def test_previous_day_high_low_is_known_for_the_whole_next_day() -> None:
    df = _m1("2024-07-01 21:00", "2024-07-03 21:00")             # Tue day + Wed day (NY 17:00 days)
    _spike(df, "2024-07-02 14:00", high=1.010)                  # inside the day starting Mon 17:00
    _spike(df, "2024-07-02 20:59", low=0.990)                   # last minute of that day
    _spike(df, "2024-07-02 21:00", high=1.050)                  # first minute of the NEXT day
    tab = L.previous_day(df)
    assert _at(tab, "2024-07-02 21:00") == (1.010, 0.990)
    assert _at(tab, "2024-07-03 20:59") == (1.010, 0.990)
    assert np.isnan(_at(tab, "2024-07-02 20:00")[0])            # first day has no previous day


def test_monday_sees_friday_as_previous_day() -> None:
    fri = _m1("2024-07-04 21:00", "2024-07-05 21:00")            # Friday's trading day
    mon = _m1("2024-07-07 21:00", "2024-07-08 21:00")            # Sunday 17:00 NY open -> Monday
    _spike(fri, "2024-07-05 15:00", high=1.020)
    tab = L.previous_day(pd.concat([fri, mon]))
    assert _at(tab, "2024-07-08 10:00")[0] == 1.020


def test_previous_week() -> None:
    w1 = _m1("2024-06-30 21:00", "2024-07-05 21:00")             # Sun 17:00 NY .. Fri 17:00 NY
    w2 = _m1("2024-07-07 21:00", "2024-07-09 21:00")
    _spike(w1, "2024-07-01 10:00", high=1.030)
    _spike(w1, "2024-07-05 20:00", low=0.970)
    _spike(w2, "2024-07-08 10:00", high=1.090)                  # current week must not leak in
    tab = L.previous_week(pd.concat([w1, w2]))
    assert _at(tab, "2024-07-09 12:00") == (1.030, 0.970)
    assert np.isnan(_at(tab, "2024-07-03 12:00")[0])


# ── session ranges ─────────────────────────────────────────────────────────

def test_asia_range_is_valid_only_after_it_is_complete() -> None:
    df = _m1("2024-07-01 21:00", "2024-07-02 21:00")             # 17:00 NY Mon .. 17:00 NY Tue
    _spike(df, "2024-07-02 01:30", high=1.004)                  # 21:30 NY - inside 20-24 NY
    _spike(df, "2024-07-02 05:00", high=1.008)                  # 01:00 NY - after the range
    tab = L.asia_range(df)
    assert np.isnan(_at(tab, "2024-07-02 03:59")[0])            # 23:59 NY - still building
    assert _at(tab, "2024-07-02 04:00")[0] == 1.004             # 00:00 NY
    assert _at(tab, "2024-07-02 20:59")[0] == 1.004
    assert np.isnan(_at(tab, "2024-07-02 21:00")[0])            # day over


def test_opening_range_uses_the_local_exchange_open() -> None:
    df = _m1("2024-07-02 00:00", "2024-07-03 00:00")
    _spike(df, "2024-07-02 13:45", high=1.006)                  # 09:45 NY (EDT)
    _spike(df, "2024-07-02 07:10", high=1.009)                  # 09:10 Berlin (CEST)
    us = L.opening_range(df, "US100")
    assert np.isnan(_at(us, "2024-07-02 13:59")[0])
    assert _at(us, "2024-07-02 14:00")[0] == 1.006
    de = L.opening_range(df, "GER40")
    assert _at(de, "2024-07-02 07:30")[0] == 1.009
    fx = L.opening_range(df, "EURUSD")                          # 08:00-08:30 London = 07:00-07:30 UTC
    assert _at(fx, "2024-07-02 07:30")[0] == 1.009


# ── events ─────────────────────────────────────────────────────────────────

def test_first_exceed_only_and_classified_by_the_bar_close() -> None:
    df = _m1("2024-07-01 21:00", "2024-07-03 21:00")
    _spike(df, "2024-07-02 14:00", high=1.010)                  # PDH of the next day = 1.010
    _spike(df, "2024-07-03 08:20", high=1.012)                  # wick above, the 08:00 bar closes inside
    _spike(df, "2024-07-03 12:30", high=1.020, close=1.015)     # later: would be a close through
    tab = L.previous_day(df)
    bars = resample_ohlc(df, "1h")
    ev = [e for e in L.events(df, bars, "1h", tab) if e.side == 1]
    assert len(ev) == 1
    assert ev[0].kind == "reclaim" and bars.index[ev[0].bar] == pd.Timestamp("2024-07-03 08:00", tz="UTC")


def test_close_beyond_is_a_through() -> None:
    df = _m1("2024-07-01 21:00", "2024-07-03 21:00")
    _spike(df, "2024-07-02 14:00", high=1.010)
    for m in range(10, 60):                                     # 08:10-08:59 trades above 1.010
        k = df.index.get_loc(pd.Timestamp(f"2024-07-03 08:{m:02d}", tz="UTC"))
        df.iloc[k, :4] = [1.012, 1.013, 1.011, 1.012]
    ev = [e for e in L.events(df, resample_ohlc(df, "1h"), "1h", L.previous_day(df)) if e.side == 1]
    assert [e.kind for e in ev] == ["through"]


def test_no_look_ahead_in_the_levels() -> None:
    rng = np.random.default_rng(1)
    t = pd.date_range("2024-03-01", "2024-04-15", freq="1min", tz="UTC")
    t = t[t.dayofweek < 5]
    c = 1.1 * np.exp(np.cumsum(rng.normal(0, 1e-4, len(t))))
    df = pd.DataFrame({"open": c, "high": c + 5e-5, "low": c - 5e-5, "close": c}, index=t)
    cut = pd.Timestamp("2024-04-01 12:34", tz="UTC")
    probe = np.array([cut.value - 60_000_000_000])
    for name in ("pd", "pw", "asia", "or"):
        full = L.tables(df, "EURUSD")[name].at(probe)[:2]
        part = L.tables(df[df.index < cut], "EURUSD")[name].at(probe)[:2]
        assert np.allclose(np.r_[full], np.r_[part], equal_nan=True), name
