"""Tests for the 15-minute opening range breakout with a fixed 2R target."""
from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from trading_lab.strategies.orb_2r import (
    MARKETS,
    OrbConfig,
    build_sessions,
    random_baseline,
    run_backtest,
    skipped_days,
)

SPREAD = 1.0
CFG = OrbConfig(slippage=0.5, commission=0.0, max_cost_r=0.0)


def _day(
    path: dict[int, tuple[float, float, float, float]],
    *,
    day: str = "2024-03-05",
    open_utc: str = "14:30",
    minutes: int = 420,
    base: float = 18000.0,
) -> pd.DataFrame:
    """One session of flat 1m bars at ``base``; ``path`` overrides minute → (o, h, l, c)."""
    idx = pd.date_range(f"{day} {open_utc}", periods=minutes, freq="1min", tz="UTC")
    o = np.full(minutes, base)
    h, l, c = o.copy(), o.copy(), o.copy()
    for m, (po, ph, pl, pc) in path.items():
        o[m], h[m], l[m], c[m] = po, ph, pl, pc
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1.0}, index=idx)
    for col in ("open", "high", "low", "close"):
        df[f"ask_{col}"] = df[col] + SPREAD
    return df


# Range 17990–18010 in minutes 0–14, breakout close in the 5m bar 15–19
_RANGE = {0: (18000, 18010, 17990, 18000)}
_BREAK_UP = {19: (18000, 18022, 18000, 18020)}


def _after(minute: int, price: float, until: int = 420) -> dict[int, tuple[float, float, float, float]]:
    return {m: (price, price, price, price) for m in range(minute, until)}


def test_long_breakout_enters_next_open_and_hits_2r():
    path = {**_RANGE, **_BREAK_UP, **_after(20, 18020), 30: (18020, 18200, 18020, 18200)}
    res = run_backtest(_day(path), CFG)
    assert len(res["trades"]) == 1
    t = res["trades"][0]
    # buys the ask of the bar after the signal bar, plus slippage
    assert t["direction"] == "long"
    assert t["entry_time"].startswith("2024-03-05T14:50")
    assert t["entry"] == pytest.approx(18020 + SPREAD + 0.5)
    # stop one spread below the range low, target 2R above the entry
    assert t["stop"] == pytest.approx(17990 - SPREAD)
    assert t["target"] == pytest.approx(t["entry"] + 2 * (t["entry"] - t["stop"]))
    assert t["exit_reason"] == "target"
    assert t["r"] == pytest.approx(2.0)


def test_short_breakout_stops_out_at_minus_one_r_plus_slippage():
    path = {
        **_RANGE,
        19: (18000, 18000, 17978, 17980),
        **_after(20, 17980),
        40: (17980, 18030, 17980, 18030),
    }
    t = run_backtest(_day(path), CFG)["trades"][0]
    assert t["direction"] == "short"
    assert t["entry"] == pytest.approx(17980 - 0.5)
    assert t["stop"] == pytest.approx(18010 + SPREAD)
    assert t["exit_reason"] == "stop"
    # stop fill slips by 0.5 against the position
    assert t["r"] == pytest.approx(-(t["risk_points"] + 0.5) / t["risk_points"])


def test_wick_beyond_range_without_close_is_no_signal():
    path = {**_RANGE, 17: (18000, 18050, 18000, 18005)}
    res = run_backtest(_day(path), CFG)
    assert res["trades"] == []
    assert res["skipped"] == {"no_breakout": 1}


def test_breakout_after_signal_window_is_ignored():
    late = 15 + 120 + 4
    path = {**_RANGE, late: (18000, 18022, 18000, 18020), **_after(late + 1, 18020)}
    assert run_backtest(_day(path), CFG)["trades"] == []


def test_only_first_breakout_trades_even_after_a_stop():
    path = {
        **_RANGE, **_BREAK_UP, **_after(20, 18020),
        25: (18020, 18020, 17900, 17900),      # stops the long
        **_after(26, 17900),                   # later closes far below the range
    }
    res = run_backtest(_day(path), CFG)
    assert [t["exit_reason"] for t in res["trades"]] == ["stop"]


def test_stop_wins_when_stop_and_target_share_a_bar():
    path = {**_RANGE, **_BREAK_UP, **_after(20, 18020), 30: (18020, 18200, 17900, 18100)}
    assert run_backtest(_day(path), CFG)["trades"][0]["exit_reason"] == "stop"


def test_flat_at_session_end():
    path = {**_RANGE, **_BREAK_UP, **_after(20, 18030)}
    t = run_backtest(_day(path), CFG)["trades"][0]
    assert t["exit_reason"] == "flat"
    # 15:55 New York (EST) = 20:55 UTC
    assert t["exit_time"].startswith("2024-03-05T20:55")
    assert t["exit"] == pytest.approx(18030 - 0.5)


def test_cost_filter_skips_small_stops():
    path = {**_RANGE, **_BREAK_UP, **_after(20, 18020)}
    # costs 1.0 spread + 0.5 slippage over a ~32 point stop ≈ 0.047R
    res = run_backtest(_day(path), replace(CFG, max_cost_r=0.04))
    assert res["trades"] == [] and res["skipped"] == {"cost_filter": 1}
    assert len(run_backtest(_day(path), replace(CFG, max_cost_r=0.05))["trades"]) == 1


def test_session_open_follows_new_york_dst():
    # July: 09:30 New York is 13:30 UTC
    path = {**_RANGE, **_BREAK_UP, **_after(20, 18020)}
    summer = _day(path, day="2024-07-09", open_utc="13:30")
    assert run_backtest(summer, CFG)["trades"][0]["entry_time"].startswith("2024-07-09T13:50")
    # the same UTC clock in winter is one hour before the open → no range bars
    early = _day({**_RANGE, **_BREAK_UP}, open_utc="13:30", minutes=45)
    assert run_backtest(early, CFG)["skipped"] == {"no_data": 1}


def test_holidays_and_half_days_are_skipped():
    assert date(2024, 11, 28) in skipped_days("nyse", 2024)   # Thanksgiving
    assert date(2024, 11, 29) in skipped_days("nyse", 2024)   # half day
    assert date(2024, 3, 29) in skipped_days("xetra", 2024)   # Good Friday
    assert date(2021, 5, 24) in skipped_days("xetra", 2021)   # Whit Monday, closed until 2021
    assert date(2024, 5, 20) not in skipped_days("xetra", 2024)
    path = {**_RANGE, **_BREAK_UP, **_after(20, 18020)}
    assert run_backtest(_day(path, day="2024-11-28"), CFG)["days"] == 0


def test_tse_holidays_are_skipped():
    tse = skipped_days("tse", 2024)
    assert date(2024, 1, 1) in tse and date(2024, 1, 2) in tse and date(2024, 1, 3) in tse
    assert date(2024, 5, 3) in tse and date(2024, 5, 6) in tse   # Golden Week + substitute
    assert date(2024, 2, 12) in tse                              # Foundation Day (Sun) substitute
    assert date(2024, 7, 15) in tse                              # Marine Day
    assert date(2024, 12, 31) in tse                             # year-end closure
    assert date(2024, 3, 5) not in tse
    assert date(2020, 10, 1) in skipped_days("tse", 2020)        # system outage
    assert date(2021, 7, 22) in skipped_days("tse", 2021)        # Olympics holiday moves
    assert date(2021, 7, 23) in skipped_days("tse", 2021)
    assert date(2019, 4, 30) in skipped_days("tse", 2019)        # imperial transition
    assert date(2019, 5, 1) in skipped_days("tse", 2019)
    path = {**_RANGE, **_BREAK_UP, **_after(20, 18020)}
    assert run_backtest(_day(path, day="2024-05-03", open_utc="00:00"), MARKETS["JP225"])["days"] == 0


def test_jp225_preset_uses_tokyo_local_time():
    assert MARKETS["JP225"].tz == "Asia/Tokyo"
    assert MARKETS["JP225"].calendar == "tse"
    assert MARKETS["JP225"].session_open == "09:00"
    # Tokyo has no DST: 09:00 JST is 00:00 UTC the whole year
    s = build_sessions(_day(_RANGE, open_utc="00:00"), MARKETS["JP225"])
    assert pd.Timestamp(s.open_ns[0], tz="UTC") == pd.Timestamp("2024-03-05 00:00", tz="UTC")


def test_cost_mult_scales_fills_but_keeps_the_stop():
    path = {**_RANGE, **_BREAK_UP, **_after(20, 18020), 30: (18020, 18020, 17900, 17900)}
    df = _day(path)
    gross = run_backtest(df, CFG, cost_mult=0.0)["trades"][0]
    net = run_backtest(df, CFG, cost_mult=1.0)["trades"][0]
    double = run_backtest(df, CFG, cost_mult=2.0)["trades"][0]
    assert gross["stop"] == net["stop"] == double["stop"]
    assert gross["entry"] == pytest.approx(18020)
    assert double["entry"] == pytest.approx(18020 + 2 * SPREAD + 1.0)
    assert gross["r"] == pytest.approx(-1.0)
    assert double["r"] < net["r"] < gross["r"]


def test_random_baseline_shape_and_cost_drag():
    path = {**_RANGE, **_BREAK_UP, **_after(20, 18020)}
    df = _day(path)
    res = run_backtest(df, CFG)
    rand = random_baseline(res["sessions"], CFG, res["trades"], runs=200, seed=1)
    assert rand.shape == (200, 1)
    # entries after the breakout sit in a flat market: spread + 2× slippage, either direction
    risk = res["trades"][0]["risk_points"]
    assert np.median(rand) == pytest.approx(-(SPREAD + 2 * 0.5) / risk)
    assert rand.min() >= -1.1 and rand.max() <= 2.0


def test_market_presets_use_exchange_local_times():
    assert MARKETS["GER40"].tz == "Europe/Berlin" and MARKETS["GER40"].session_open == "09:00"
    assert MARKETS["XAUUSD"].session_open == "08:20" and MARKETS["XAUUSD"].commission > 0
    s = build_sessions(_day(_RANGE), MARKETS["US100"])
    assert pd.Timestamp(s.open_ns[0], tz="UTC") == pd.Timestamp("2024-03-05 14:30", tz="UTC")


# ── Day filters ─────────────────────────────────────────────────────

_WEEK = ("2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08")


def _days(levels: list[float], last_path: dict[int, tuple[float, float, float, float]]) -> pd.DataFrame:
    """Flat sessions at ``levels`` on consecutive days; the last day follows ``last_path``."""
    frames = [_day({}, day=day, base=lvl) for day, lvl in zip(_WEEK, levels[:-1])]
    frames.append(_day(last_path, day=_WEEK[len(levels) - 1], base=levels[-1]))
    return pd.concat(frames)


_LONG_DAY = {**_RANGE, **_BREAK_UP, **_after(20, 18020)}
_SHORT_DAY = {**_RANGE, 19: (18000, 18000, 17978, 17980), **_after(20, 17980)}


def test_trend_filter_trades_only_with_the_daily_sma():
    cfg = replace(CFG, trend_sma_days=2)
    # prior closes 17000, 17500, 17900 → last close above the SMA of the last two
    up = [17000.0, 17500.0, 17900.0, 18000.0]
    assert [t["direction"] for t in run_backtest(_days(up, _LONG_DAY), cfg)["trades"]] == ["long"]
    res = run_backtest(_days(up, _SHORT_DAY), cfg)
    assert res["trades"] == [] and res["skipped"]["trend_filter"] == 1
    down = [19000.0, 18500.0, 18100.0, 18000.0]
    assert [t["direction"] for t in run_backtest(_days(down, _SHORT_DAY), cfg)["trades"]] == ["short"]


def test_gap_filter_needs_breakout_in_gap_direction():
    cfg = replace(CFG, gap_filter=True)
    gap_up = [17900.0, 18000.0]
    assert len(run_backtest(_days(gap_up, _LONG_DAY), cfg)["trades"]) == 1
    res = run_backtest(_days(gap_up, _SHORT_DAY), cfg)
    assert res["trades"] == [] and res["skipped"]["gap_filter"] == 1
    # no prior close on the first day → no trade
    assert run_backtest(_day(_LONG_DAY), cfg)["trades"] == []


def test_range_atr_filter_needs_an_above_median_range():
    cfg = replace(CFG, atr_period_daily=1, range_atr_median_days=2)

    def week(last_range: float) -> pd.DataFrame:
        frames = []
        for day, width in zip(_WEEK, (20.0, 20.0, 20.0, 20.0, last_range)):
            path = {0: (18000, 18000 + width / 2, 18000 - width / 2, 18000)}
            if day == _WEEK[-1]:
                path |= {19: (18000, 18100, 18000, 18090), **_after(20, 18090)}
            frames.append(_day(path, day=day))
        return pd.concat(frames)

    assert len(run_backtest(week(40.0), cfg)["trades"]) == 1
    res = run_backtest(week(10.0), cfg)
    assert res["trades"] == [] and res["skipped"]["range_atr_filter"] >= 1


def test_fomc_days_are_skipped():
    fomc = _day(_LONG_DAY, day="2024-03-20", open_utc="13:30")  # EDT
    assert len(run_backtest(fomc, CFG)["trades"]) == 1
    res = run_backtest(fomc, replace(CFG, skip_fomc=True))
    assert res["trades"] == [] and res["skipped"] == {"fomc": 1}
