"""Tests for the fair-pricing strategy and its economic-calendar loader.

All synthetic — no network, no data files. The engine's primitives (displacement,
swings/BOS, consolidation) are tested directly, plus one engineered end-to-end
Setup A entry and the shared return contract.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from trading_lab.calendar_events import (
    load_events,
    high_impact_times,
    events_in,
    parse_impact,
)
from trading_lab.strategies.fair_pricing import (
    FairPricingConfig,
    SwingTracker,
    consolidation_arrays,
    displacement_at,
    prepare,
    run_backtest,
)

NY = "America/New_York"


# ── calendar loader ─────────────────────────────────────────────────


def test_load_events_csv_and_impact(tmp_path) -> None:
    p = tmp_path / "cal.csv"
    p.write_text(
        "time,event,impact\n"
        "2024-06-12T08:30:00,CPI,high\n"
        "2024-06-12T12:30:00Z,Initial Claims,low\n"
        "2024-06-12T18:00:00,FOMC,red\n"
    )
    events = load_events(p)
    assert [e.name for e in events] == ["CPI", "Initial Claims", "FOMC"]
    assert events[0].impact == "high" and events[0].is_high
    assert events[1].impact == "low"
    assert events[2].impact == "high"          # "red" alias
    # naive timestamps are UTC by default
    assert events[0].ts == pd.Timestamp("2024-06-12T08:30:00", tz="UTC").timestamp()


def test_load_events_json_object(tmp_path) -> None:
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"events": [
        {"timestamp": "2024-06-12 08:30", "event": "NFP", "impact": 3},
    ]}))
    events = load_events(p, tz="America/New_York")
    assert len(events) == 1 and events[0].impact == "high"
    assert events[0].ts == pd.Timestamp("2024-06-12 08:30", tz=NY).timestamp()


def test_missing_calendar_fails_loud(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_events(tmp_path / "nope.csv")


def test_impact_aliases_and_helpers() -> None:
    assert parse_impact("RED") == "high"
    assert parse_impact("") == "unknown"
    assert parse_impact("orange") == "medium"
    from trading_lab.calendar_events import CalendarEvent

    evs = [CalendarEvent(100.0, "a", "high"), CalendarEvent(200.0, "b", "low")]
    assert high_impact_times(evs) == [100.0]
    assert [e.name for e in events_in(evs, 50.0, 150.0)] == ["a"]


# ── primitives ──────────────────────────────────────────────────────


def test_displacement_short() -> None:
    o = np.array([100.0, 101.0, 102.0])
    h = np.array([101.0, 102.0, 102.5])
    lo = np.array([99.5, 101.0, 100.0])
    c = np.array([100.8, 101.5, 100.5])   # bar 2: red, bigger body, close below low[1]
    assert displacement_at(o, h, lo, c, 2, lookback=1) == -1


def test_displacement_needs_color_flip() -> None:
    o = np.array([100.0, 101.5, 102.0])
    h = np.array([101.0, 102.0, 102.5])
    lo = np.array([99.5, 101.0, 100.0])
    c = np.array([101.5, 101.0, 100.5])   # bar 1 is red too → no color flip
    assert displacement_at(o, h, lo, c, 2, lookback=1) == 0


def test_swing_tracker_bos_and_break() -> None:
    high = np.array([12.0, 11.0, 13.0, 10.0])
    low = np.array([10.0, 5.0, 8.0, 4.0])
    close = np.array([11.0, 6.0, 12.0, 4.0])
    st = SwingTracker()
    st.update(0, high, low, close)
    st.update(1, high, low, close)
    st.update(2, high, low, close)          # confirms swing low at j=1 (price 5)
    assert st.youngest_unbroken_low() == 5.0
    assert st.bos_dir(close=6.0) == 0       # 6 > 5, no break
    assert st.bos_dir(close=4.5) == -1      # close below the outstanding low
    st.update(3, high, low, close)          # close 4 breaks it
    assert st.youngest_unbroken_low() is None


def test_consolidation_detection() -> None:
    cfg = FairPricingConfig(consol_min_bars=3, consol_max_range_pts=5.0)
    high = np.array([10.0, 10.5, 11.0, 11.2, 30.0])
    low = np.array([9.0, 9.5, 10.0, 10.2, 20.0])
    close = np.array([9.5, 10.0, 10.5, 10.7, 25.0])
    ok, level = consolidation_arrays(high, low, close, cfg)
    assert ok[2] and not ok[4]              # bars 0..2 are a tight range
    assert level[2] == pytest.approx((11.0 + 9.0) / 2)


# ── end to end ──────────────────────────────────────────────────────


def _setup_a_frame() -> pd.DataFrame:
    """Bearish-bias, dip-swing-low morning, then a big red first NY-AM bar."""
    idx = pd.date_range("2024-06-12 08:00", "2024-06-12 10:00", freq="1min", tz=NY).tz_convert("UTC")
    loc = idx.tz_convert(NY)
    n = len(idx)
    close = np.full(n, 21000.0)

    def at(hhmm: str) -> int:
        return int(np.searchsorted(loc, pd.Timestamp(f"2024-06-12 {hhmm}", tz=NY)))

    close[at("08:30"):at("09:15")] = 21040.0   # rise → +1h delta → bias -1
    close[at("09:15")] = 21040.0
    close[at("09:16")] = 21025.0
    close[at("09:17")] = 21010.0               # swing low
    close[at("09:18")] = 21025.0
    close[at("09:19"):at("09:30")] = 21060.0
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    k = at("09:30")
    open_[k], close[k], high[k], low[k] = 21060.0, 21000.0, 21060.5, 20999.5
    close[at("09:31"):] = 21000.0
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
    df["volume"] = 100.0
    for side in ("open", "high", "low", "close"):
        df[f"ask_{side}"] = df[side] + 1.0
    return df


def test_setup_a_short_continuation() -> None:
    df = _setup_a_frame()
    cfg = FairPricingConfig(sessions=("ny_am",), htf_lookback_h=1)
    prep = prepare(df, cfg)
    k = prep.sessions[0].open_idx
    assert prep.bias[k] == -1.0                 # bearish bias at the open

    res = run_backtest(df, cfg)
    assert len(res["trades"]) == 1
    trade = res["trades"][0]
    assert trade["direction"] == "short"
    assert trade["meta"]["setup"] == "A"
    assert trade["meta"]["sl_pts"] == 50.0      # big candle → the 50/76 bracket


def test_run_backtest_contract() -> None:
    df = _setup_a_frame()
    res = run_backtest(df, FairPricingConfig(sessions=("ny_am",), htf_lookback_h=1))
    assert set(res) == {"trades", "signals", "equity_curve", "metrics"}
    assert res["metrics"]["total_trades"] == len(res["trades"])


def test_news_day_session_starts_at_release() -> None:
    """§4.2: on a Red Folder day the NY-AM session opens at 08:30, cash open 09:30."""
    from trading_lab.calendar_events import CalendarEvent

    df = _setup_a_frame()
    cfg = FairPricingConfig(sessions=("ny_am",), htf_lookback_h=1)
    news_ts = pd.Timestamp("2024-06-12 08:30", tz=NY).timestamp()
    prep = prepare(df, cfg, events=[CalendarEvent(news_ts, "CPI", "high")])
    s = prep.sessions[0]
    assert s.is_news
    assert prep.local[s.open_idx].strftime("%H:%M") == "08:30"
    assert prep.local[s.cash_open_idx].strftime("%H:%M") == "09:30"
    assert s.event_cutoff_ns == 0          # the anchoring release must not block Setup C

    plain = prepare(df, cfg).sessions[0]
    assert not plain.is_news
    assert prep.local[plain.open_idx].strftime("%H:%M") == "09:30"
