"""S-006 (docs/specs/noise_area_momentum.yaml): one test group per rule, on full 24 h 1m frames."""
from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from trading_lab.strategies import noise_area_momentum as N
from trading_lab.strategies.noise_area_momentum import NoiseAreaConfig

SPREAD = 0.02
CFG = NoiseAreaConfig(slippage_bp=0.0)
NY = "America/New_York"


def _days(start: str, n: int) -> list[pd.Timestamp]:
    return list(pd.bdate_range(start, periods=n))


def _frame(paths: dict[pd.Timestamp, np.ndarray], tz: str = NY, open_hm=(9, 30), L: int = 390,
           base: float = 100.0, vol: float = 1.0) -> pd.DataFrame:
    """Full 24 h minutes over the given weekdays; ``paths[day]`` = L session closes (open of
    each minute = previous close, first open = ``base``); flat ``base`` outside the session."""
    days = sorted(paths)
    t = pd.date_range(days[0].tz_localize(tz).tz_convert("UTC").normalize(),
                      (days[-1] + pd.Timedelta(days=1)).tz_localize(tz).tz_convert("UTC"),
                      freq="1min", tz="UTC", inclusive="left")
    t = t[t.tz_convert(tz).dayofweek < 5]                     # no weekend quotes, like the real feed
    c = np.full(len(t), base)
    o = np.full(len(t), base)
    for d, path in paths.items():
        s = (d + pd.Timedelta(hours=open_hm[0], minutes=open_hm[1])).tz_localize(tz).tz_convert("UTC")
        i0 = t.get_loc(s)
        c[i0:i0 + L] = path
        o[i0:i0 + L] = np.r_[base, path[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c,
                       "volume": vol}, index=t)
    for k in ("open", "high", "low", "close"):
        df[f"ask_{k}"] = df[k] + SPREAD
    return df


def _history(n: int = 15, start: str = "2024-03-04", move: float = 0.01) -> dict:
    """n sessions whose every minute closes at base * (1 + move) -> sigma(m) = move; only the
    LAST minute alternates by 0.1 so the daily returns (rule 8) are not all zero."""
    out = {}
    for i, d in enumerate(_days(start, n)):
        path = np.full(390, 100.0 * (1 + move))
        path[-1] += 0.1 * ((i + 1) % 2)
        out[d] = path
    return out


def _mid_shift(df: pd.DataFrame) -> pd.DataFrame:
    """Bid = mid - spread/2 so that the MID equals the path (bands are drawn on mid)."""
    for k in ("open", "high", "low", "close"):
        df[k] -= SPREAD / 2
        df[f"ask_{k}"] -= SPREAD / 2
    return df


# ── rules 1-3: session, sigma, bands ───────────────────────────────────────

def test_sigma_and_gap_adjusted_bands() -> None:
    paths = _history(15)
    p = N.prepare(_mid_shift(_frame(paths)), "SP500")
    assert len(p.open_ns) == 15
    d = 14                                                     # 15th session, 14 before it
    O, C1 = 100.0, 101.0                                       # first open = base, previous close = 101
    assert p.upper[d, 100] == pytest.approx(max(O, C1) * 1.01)
    assert p.lower[d, 100] == pytest.approx(min(O, C1) * 0.99)
    assert np.isnan(p.upper[13]).all()                         # only 13 sessions before it


def test_session_is_in_exchange_time_dst_safe() -> None:
    winter = N.prepare(_mid_shift(_frame(_history(15, "2024-01-08"))), "SP500")
    summer = N.prepare(_mid_shift(_frame(_history(15, "2024-07-08"))), "SP500")
    assert pd.Timestamp(winter.open_ns[0], tz="UTC").strftime("%H:%M") == "14:30"
    assert pd.Timestamp(summer.open_ns[0], tz="UTC").strftime("%H:%M") == "13:30"
    de = N.prepare(_mid_shift(_frame({d: np.full(510, 101.0) for d in _days("2024-07-08", 15)},
                                     tz="Europe/Berlin", open_hm=(9, 0), L=510)), "GER40")
    assert pd.Timestamp(de.open_ns[0], tz="UTC").strftime("%H:%M") == "07:00"


def test_half_days_and_thin_sessions_are_skipped() -> None:
    assert N.us_half_day(date(2024, 11, 29)) and N.us_half_day(date(2024, 12, 24)) and N.us_half_day(date(2024, 7, 3))
    assert not N.us_half_day(date(2024, 11, 28)) and not N.us_half_day(date(2024, 7, 5))
    paths = _history(15)
    df = _mid_shift(_frame(paths))
    thin = _days("2024-03-04", 15)[3]
    s = (thin + pd.Timedelta(hours=9, minutes=30)).tz_localize(NY).tz_convert("UTC")
    df = df[~((df.index >= s + pd.Timedelta(minutes=200)) & (df.index < s + pd.Timedelta(minutes=390)))]
    p = N.prepare(df, "SP500")
    assert len(p.open_ns) == 14


# ── rules 4-7: decisions, fills, exits ─────────────────────────────────────

def _run(path: np.ndarray, cfg: NoiseAreaConfig = CFG, vol_path: np.ndarray | None = None):
    paths = _history(15)
    test_day = _days("2024-03-04", 16)[-1]
    paths[test_day] = path
    df = _mid_shift(_frame(paths))
    if vol_path is not None:
        s = (test_day + pd.Timedelta(hours=9, minutes=30)).tz_localize(NY).tz_convert("UTC")
        i0 = df.index.get_loc(s)
        df.iloc[i0:i0 + 390, df.columns.get_loc("volume")] = vol_path
    p = N.prepare(df, "SP500")
    return p, N.run_day(p, cfg, len(p.open_ns) - 1)


def test_only_check_times_count_and_entry_is_the_next_minute_ask() -> None:
    # upper band = 101 * 1.01 = 102.01. Above it 09:40-09:58 only -> no trade at 10:00 check
    path = np.full(390, 101.0)
    path[10:29] = 103.0
    assert _run(path)[1] == []
    path = np.full(390, 101.0)
    path[29:] = 103.0                                          # minute 29 closes 10:00 above the band
    p, tr = _run(path)
    assert len(tr) == 1 and tr[0]["direction"] == 1
    assert tr[0]["entry_time"] == pd.Timestamp(p.open_ns[-1], tz="UTC") + pd.Timedelta(minutes=30)
    assert tr[0]["entry"] == pytest.approx(103.0 + SPREAD / 2)            # ask open of 10:00


def test_flat_at_the_last_session_minute() -> None:
    path = np.full(390, 101.0)
    path[29:] = 103.0
    p, tr = _run(path)
    assert tr[0]["exit_reason"] == "session_ende"
    assert tr[0]["exit_time"] == pd.Timestamp(p.open_ns[-1], tz="UTC") + pd.Timedelta(minutes=389)
    sd = p.sigma_daily[-1]
    entry = 103.0 + SPREAD / 2
    assert tr[0]["r"] == pytest.approx((103.0 - SPREAD / 2 - entry) / (sd * entry))       # rule 8: 1 R = 1 daily sd


def test_trailing_band_vs_vwap_and_flip() -> None:
    path = np.full(390, 101.0)
    path[29:59] = 106.0                                        # long at 10:00
    path[59:] = 104.0                                          # 10:30: above the band (102.01) ...
    vol = np.zeros(390)
    vol[29:59] = 100.0                                         # ... but below the VWAP (~106)
    _, tr = _run(path, vol_path=vol)
    assert tr[0]["exit_reason"] == "trailing"
    _, tr_band = _run(path, replace(CFG, trailing="band"), vol_path=vol)
    assert tr_band[0]["exit_reason"] == "session_ende"
    flip = np.full(390, 101.0)
    flip[29:59] = 103.0
    flip[59:] = 98.0                                           # 10:30 below the lower band (99): exit + short
    _, tr = _run(flip, replace(CFG, trailing="band"))
    assert [t["direction"] for t in tr] == [1, -1]
    assert tr[0]["exit_time"] == tr[1]["entry_time"]


def test_15min_checks() -> None:
    assert N.check_minutes(390, "30min")[:2] == [30, 60] and N.check_minutes(390, "30min")[-1] == 360
    assert N.check_minutes(390, "15min")[-1] == 375
    path = np.full(390, 101.0)
    path[44:] = 103.0                                          # first above the band at 10:15
    _, tr30 = _run(path)
    _, tr15 = _run(path, replace(CFG, pruefung="15min"))
    assert tr15[0]["entry_time"] + pd.Timedelta(minutes=15) == tr30[0]["entry_time"]


def test_no_look_ahead() -> None:
    rng = np.random.default_rng(3)
    days = _days("2024-01-08", 40)
    paths = {d: 100 * np.exp(np.cumsum(rng.normal(0, 0.0008, 390))) for d in days}
    df = _mid_shift(_frame(paths))
    full = N.run_backtest(N.prepare(df, "SP500"), CFG)["trades"]
    assert len(full) > 5
    cut = (days[30] + pd.Timedelta(hours=20)).tz_localize(NY).tz_convert("UTC")
    part = N.run_backtest(N.prepare(df[df.index < cut], "SP500"), CFG)["trades"]
    key = lambda t: (t["entry_time"], t["direction"], round(t["r"], 9))  # noqa: E731
    done = {key(t) for t in full if t["exit_time"] < cut}
    assert done and done == {key(t) for t in part}
