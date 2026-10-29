"""Levels x flow (H8) on hand-made 10 s paths."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_lab import level_flow as lf


def bars_from(close, buy=None, sell=None, start="2024-01-01", high=None, low=None):
    close = np.asarray(close, dtype=float)
    n = len(close)
    idx = pd.date_range(start, periods=n, freq="10s", tz="UTC", name="ts")
    return pd.DataFrame({
        "open": close, "high": close if high is None else np.asarray(high, float),
        "low": close if low is None else np.asarray(low, float), "close": close, "vwap": close,
        "buy_qty": 1.0, "sell_qty": 1.0,
        "buy_notional": np.ones(n) if buy is None else np.asarray(buy, float),
        "sell_notional": np.ones(n) if sell is None else np.asarray(sell, float),
        "n_trades": 1}, index=idx)


def test_period_levels_previous_day_and_week():
    n = 3 * 8640                                       # Mon 2024-01-01 .. Wed
    close = np.concatenate([np.full(8640, 100.0), np.full(8640, 110.0), np.full(8640, 90.0)])
    close[100] = 105.0                                 # day-1 high
    bars = bars_from(close)
    days = lf.period_levels(bars, "day", "pd")
    assert [(lv.kind, lv.price, lv.known_at.day) for lv in days[:2]] == [("pdh", 105.0, 2), ("pdl", 100.0, 2)]
    assert len(days) == 4                              # day 3 not complete before the data ends
    assert lf.period_levels(bars, "week", "pw") == []  # the week is not complete


def test_touch_event_flow_position_and_outcome():
    # price 100 -> rises to the level 101 at bar 10, window 3 bars, then drifts
    close = np.full(60, 100.0)
    close[10:] = 101.0
    close[12] = 100.9                                  # end of window: back below the level
    close[13:] = 100.0
    buy = np.ones(60)
    buy[10:13] = 9.0                                   # strong buying into the level
    bars = bars_from(close, buy=buy)
    atr = pd.Series(0.5, index=bars.index)
    lv = [lf.Level("pdh", 101.0, bars.index[0])]
    ev = lf.touch_events(bars, lv, atr, window=3, horizons={"1m": "1min"})
    assert len(ev) == 1
    e = ev.iloc[0]
    assert e["ts"] == bars.index[10] and e["d"] == 1
    assert e["flow"] == pytest.approx((27 - 3) / 30)
    assert e["beyond_atr"] == pytest.approx((100.9 - 101.0) / 0.5)
    assert e["ret_1m"] == pytest.approx(np.log(100.0 / 100.9) * 1e4)   # from window end, 6 bars on
    lab = lf.classify(ev).iloc[0]
    assert lab["label"] == "absorption" and lab["trade"] == -1


def test_level_known_later_is_not_touched_earlier_and_skips_when_at_level():
    close = np.full(40, 100.0)
    close[5] = 102.0                                   # touch BEFORE the level is known
    bars = bars_from(close)
    atr = pd.Series(1.0, index=bars.index)
    ev = lf.touch_events(bars, [lf.Level("x", 102.0, bars.index[8])], atr, window=2, horizons={"10s": "10s"})
    assert ev.empty
    at = lf.touch_events(bars, [lf.Level("x", 100.0, bars.index[0])], atr, window=2, horizons={"10s": "10s"})
    assert at.empty                                    # price sits on the level when it appears


def test_classify_labels():
    ev = pd.DataFrame({"d": [1, 1, 1, 1, -1, 1],
                       "flow": [0.5, 0.5, 0.0, 0.0, -0.5, 0.5],
                       "beyond_atr": [-0.1, 0.4, -0.2, 0.3, 0.5, 0.1]})
    lab = lf.classify(ev)
    assert list(lab["label"]) == ["absorption", "cascade", "reclaim", "break", "cascade", "none"]
    assert list(lab["trade"]) == [-1, 1, -1, 1, -1, 0]


def test_control_levels_shift_within_band():
    lv = [lf.Level("pdh", 100.0, pd.Timestamp("2024-01-02", tz="UTC"))] * 200
    ctl = lf.control_levels(lv, seed=1)
    moves = np.array([abs(c.price / 100.0 - 1) for c in ctl])
    assert (moves >= 0.0015).all() and (moves <= 0.005).all()
    assert all(c.known_at == lv[0].known_at for c in ctl)
