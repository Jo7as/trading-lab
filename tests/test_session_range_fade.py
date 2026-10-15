"""Tests for the Session-Range-Fade strategy (Asia range, London fade)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_lab.mr_common import drop_unclosed_last_bar
from trading_lab.strategies.session_range_fade import (
    SessionRangeFadeConfig,
    compute_asia_range,
    run_backtest,
)
from trading_lab.mr_common import prepare_frame

_DAY = "2024-03-08"  # Friday; Mon-Thu before it warm up the daily ATR
_NOW = pd.Timestamp("2024-03-10", tz="UTC")


def _cfg(**kw) -> SessionRangeFadeConfig:
    base = dict(
        atr_period_daily=3,
        range_min_atr_mult=0.5,
        range_max_atr_mult=4.0,
        sl_buffer_atr_mult=0.0,
        fallback_spread_pct=0.0,
        slippage_pct=0.0,
    )
    base.update(kw)
    return SessionRangeFadeConfig(**base)


def _flat_week() -> pd.DataFrame:
    """Mon-Fri 15m bars, 00:00-21:00 UTC, every bar 1999-2001 around 2000."""
    frames = []
    for day in pd.date_range("2024-03-04", _DAY, freq="D", tz="UTC"):
        idx = pd.date_range(day, day + pd.Timedelta(hours=20, minutes=45), freq="15min")
        frames.append(
            pd.DataFrame(
                {"open": 2000.0, "high": 2001.0, "low": 1999.0, "close": 2000.0, "volume": 1.0},
                index=idx,
            )
        )
    return pd.concat(frames)


def _set(df: pd.DataFrame, hhmm: str, o: float, h: float, l: float, c: float) -> None:
    df.loc[pd.Timestamp(f"{_DAY} {hhmm}", tz="UTC"), ["open", "high", "low", "close"]] = [o, h, l, c]


def _quiet_from(df: pd.DataFrame, hhmm: str, level: float) -> None:
    """Pin all later bars of the test day to a tight band around ``level``."""
    later = df.index >= pd.Timestamp(f"{_DAY} {hhmm}", tz="UTC")
    df.loc[later, ["open", "high", "low", "close"]] = [level, level + 0.1, level - 0.1, level]


def _short_fade_day() -> pd.DataFrame:
    df = _flat_week()
    _quiet_from(df, "07:30", 2000.5)
    _set(df, "07:00", 2000.0, 2003.0, 2000.0, 2002.0)   # breaks range high 2001
    _set(df, "07:15", 2002.0, 2002.2, 2000.4, 2000.5)   # closes back inside
    return df


def test_short_fade_runs_to_opposite_range_side() -> None:
    df = _short_fade_day()
    _set(df, "07:45", 2000.5, 2000.6, 1999.9, 2000.0)   # TP1 = range mid 2000
    _set(df, "08:00", 2000.0, 2000.1, 1998.8, 1999.0)   # TP2 = range low 1999
    r = run_backtest(df, _cfg(), now=_NOW)

    assert r["metrics"]["total_trades"] == 1
    t = r["trades"][0]
    assert t["direction"] == "short"
    assert t["exit_reason"] == "tp2"
    assert t["tp1_hit"] is True
    assert t["pnl_pct"] > 0
    # filled at the open of the bar after the confirmation candle
    assert t["entry_time"] == pd.Timestamp(f"{_DAY} 07:30", tz="UTC").timestamp()
    assert t["entry_price"] == 2000.5
    assert t["stop_loss"] == 2000.5 or t["sl_moved_to_be"]  # BE after TP1
    # 0.75% risk on a 2.5 stop, half off at +0.5, half at +1.5
    assert abs(t["pnl_pct"] - 0.75 * (0.5 * 0.5 + 0.5 * 1.5) / 2.5) < 1e-6


def test_london_bar_is_not_part_of_the_asia_range() -> None:
    df = _short_fade_day()
    cfg = _cfg()
    ranges = compute_asia_range(prepare_frame(df, cfg.timeframe, 0.0, _NOW), cfg)
    day = ranges.loc[pd.Timestamp(_DAY, tz="UTC")]
    assert day["range_high"] == 2001.0  # the 07:00 bar's 2003 high is excluded
    assert day["range_low"] == 1999.0
    assert bool(day["valid"])


def test_stop_loss_at_breakout_extreme() -> None:
    df = _short_fade_day()
    _set(df, "07:45", 2000.5, 2003.5, 2000.4, 2003.2)
    r = run_backtest(df, _cfg(), now=_NOW)
    t = r["trades"][0]
    assert t["exit_reason"] == "stop_loss"
    assert t["exit_price"] == 2003.0
    assert abs(t["pnl_pct"] + 0.75) < 1e-6


def test_time_exit_at_london_end() -> None:
    df = _short_fade_day()
    r = run_backtest(df, _cfg(), now=_NOW)
    t = r["trades"][0]
    assert t["exit_reason"] == "time_exit"
    assert t["exit_time"] == pd.Timestamp(f"{_DAY} 11:00", tz="UTC").timestamp()


def test_no_entry_when_entry_bar_opens_at_london_end() -> None:
    df = _flat_week()
    _quiet_from(df, "11:00", 2000.5)
    _set(df, "10:30", 2000.0, 2003.0, 2000.0, 2002.0)
    _set(df, "10:45", 2002.0, 2002.2, 2000.4, 2000.5)   # entry would be 11:00
    r = run_backtest(df, _cfg(), now=_NOW)
    assert r["metrics"]["total_trades"] == 0


def test_range_size_filter_skips_day() -> None:
    r = run_backtest(_short_fade_day(), _cfg(range_min_atr_mult=5.0), now=_NOW)
    assert r["metrics"]["total_trades"] == 0
    assert r["metrics"]["skip_reasons"].get("range_too_small", 0) >= 1


def test_data_gap_in_asia_window_skips_day() -> None:
    df = _short_fade_day()
    hole = (df.index >= pd.Timestamp(f"{_DAY} 02:00", tz="UTC")) & (
        df.index < pd.Timestamp(f"{_DAY} 04:00", tz="UTC")
    )
    r = run_backtest(df[~hole], _cfg(), now=_NOW)
    assert r["metrics"]["total_trades"] == 0
    assert r["ranges"].loc[pd.Timestamp(_DAY, tz="UTC"), "skip_reason"] == "data_gap"


def test_forming_confirmation_candle_gives_no_signal() -> None:
    df = _short_fade_day()
    # Data ends inside the 07:30 bar: it must be dropped, so there is no entry bar
    cut = df[df.index <= pd.Timestamp(f"{_DAY} 07:30", tz="UTC")]
    now = pd.Timestamp(f"{_DAY} 07:40", tz="UTC")
    assert len(drop_unclosed_last_bar(cut, 15, now)) == len(cut) - 1
    r = run_backtest(cut, _cfg(), now=now)
    assert r["metrics"]["total_trades"] == 0


def test_spread_is_charged_on_entry_and_exit() -> None:
    df = _short_fade_day()
    _set(df, "07:45", 2000.5, 2000.6, 1999.0, 1999.5)
    _set(df, "08:00", 1999.5, 1999.6, 1998.0, 1998.5)
    free = run_backtest(df, _cfg(), now=_NOW)["trades"][0]
    paid = run_backtest(df, _cfg(fallback_spread_pct=0.0002), now=_NOW)["trades"][0]
    assert np.isfinite(paid["pnl_pct"])
    assert paid["pnl_pct"] < free["pnl_pct"]
