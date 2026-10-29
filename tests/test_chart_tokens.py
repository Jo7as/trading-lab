"""Rules of the chart tokens (H11) on tiny hand-made inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_lab.chart_tokens import (bars_before, candle_shape, candle_str, candle_word, sax_str,
                                      sax_word, tercile_letter, to_ns)


def test_sax_rising_path_and_letters():
    x = np.arange(8.0)
    # segment means of z are -1.31 / -0.44 / +0.44 / +1.31; 0.44 > 0.4307 -> a, a, c, c
    assert sax_str(sax_word(x, 4, 3), 4, 3) == "aacc"
    assert sax_str(sax_word(x, 4, 4), 4, 4) == "abcd"


def test_sax_is_price_and_size_invariant():
    x = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 4.0])
    w = sax_word(x, 3, 3)
    assert sax_word(x + 10_000.0, 3, 3) == w
    assert sax_word(x * 50.0, 3, 3) == w
    assert sax_word(-x, 3, 3) != w


def test_sax_flat_path_is_middle_letters_and_uneven_split():
    assert sax_str(sax_word(np.full(5, 7.0), 3, 3), 3, 3) == "bbb"
    assert sax_str(sax_word(np.full(5, 7.0), 3, 4), 3, 4) == "ccc"   # 0 sits on the middle edge
    # 4 points into 3 segments: [0,1] [2] [3]
    assert sax_str(sax_word(np.array([0.0, 0.0, 5.0, 10.0]), 3, 3), 3, 3) == "abc"


def test_candle_shapes():
    assert candle_shape(10, 11, 9, 10.05) == 0            # doji
    assert candle_shape(10, 10, 10, 10) == 0              # zero range
    assert candle_shape(9.1, 11, 9, 10.9) == 1            # strong up
    assert candle_shape(10.9, 11, 9, 9.1) == 4            # strong down
    assert candle_shape(10.0, 11, 9, 10.8) == 2           # other up
    assert candle_shape(10.0, 11, 9, 9.5) == 3            # other down


def test_candle_word_price_invariant_and_readable():
    o, h, lo, c = (np.array(v, dtype=float) for v in ([9.1, 10.0], [11, 11], [9, 9], [10.9, 9.5]))
    w = candle_word(o, h, lo, c)
    assert candle_str(w, 2) == "Un"
    assert candle_word(o + 500, h + 500, lo + 500, c + 500) == w


def test_tercile_letter():
    assert tercile_letter(np.array([0.5, 1.0, 1.5, 3.0, np.nan]), (1.0, 2.0)).tolist() == [0, 1, 1, 2, -1]


def _minutes(start: str, n: int, drop: tuple[int, ...] = ()) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    p = np.arange(n, dtype=float)
    df = pd.DataFrame({"open": p, "high": p + 0.5, "low": p - 0.5, "close": p + 0.25,
                       "volume": np.ones(n)}, index=idx)
    return df.drop(df.index[list(drop)])


def _call(df, t0, tf, n, **kw):
    ts = to_ns(df.index)
    return bars_before(ts, *(df[c].to_numpy() for c in ("open", "high", "low", "close", "volume")),
                       pd.Timestamp(t0).value, tf, n, **kw)


def test_bars_before_aligned_to_t0_on_a_full_24h_frame():
    df = _minutes("2021-03-01 00:00", 24 * 60)
    out = _call(df, "2021-03-01 13:30", 15, 2)
    # last bar = 13:15..13:29 -> minutes 795..809
    assert out[1].tolist() == [795.0, 809.5, 794.5, 809.25, 15.0]
    assert out[0].tolist() == [780.0, 794.5, 779.5, 794.25, 15.0]


def test_bars_before_gap_does_not_shift_bars():
    # drop 13:15..13:24 (10 of 15 minutes): last bar is under-filled -> NaN, earlier bar intact
    df = _minutes("2021-03-01 00:00", 24 * 60, drop=tuple(range(795, 805)))
    out = _call(df, "2021-03-01 13:30", 15, 2)
    assert np.isnan(out[1]).all()
    assert out[0][0] == 780.0
    # with 5 of 15 present and min_fill 0.3 the bar exists and starts at 13:25
    out2 = _call(df, "2021-03-01 13:30", 15, 2, min_fill=0.3)
    assert out2[1][0] == 805.0 and out2[1][4] == 5.0
