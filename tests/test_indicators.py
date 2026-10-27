"""Indicators against references: a published example and independent textbook loops.

Every indicator is checked three ways: (1) against a published worked example where one
exists (StockCharts' RSI table), (2) against an independent, deliberately naive loop
written from the textbook definition, (3) for causality - appending future bars must not
change any past value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_lab import indicators as I

# StockCharts ChartSchool, "Relative Strength Index" worked example (14 periods)
SC_CLOSE = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03,
            45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64, 46.21, 46.25, 45.71, 46.45,
            45.78, 45.35, 44.03, 44.18, 44.22, 44.57, 43.42, 42.66, 43.13]
SC_RSI = [70.53, 66.32, 66.55, 69.41, 66.36, 57.97, 62.93, 63.26, 56.06, 62.38, 54.71, 50.42,
          39.99, 41.46, 41.87, 45.46, 37.30, 33.08, 37.77]


def _ohlc(n: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    o = np.r_[c[0], c[:-1]] + rng.normal(0, 0.2, n)
    h = np.maximum(o, c) + rng.uniform(0, 1, n)
    l = np.minimum(o, c) - rng.uniform(0, 1, n)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c},
                        index=pd.date_range("2024-01-01", periods=n, freq="1h"))


def _wilder_loop(x: list[float], n: int) -> list[float]:
    out, prev = [], None
    vals = [v for v in x]
    for i, v in enumerate(vals):
        if i < n - 1:
            out.append(np.nan)
        elif i == n - 1:
            prev = float(np.mean(vals[:n]))
            out.append(prev)
        else:
            prev = prev + (v - prev) / n
            out.append(prev)
    return out


def test_rsi_matches_the_published_stockcharts_example() -> None:
    r = I.rsi(pd.Series(SC_CLOSE), 14).dropna().round(2).tolist()
    assert len(r) == len(SC_RSI)
    # the published table rounds its intermediate averages; the gap shrinks to 0.02
    assert np.allclose(r, SC_RSI, atol=0.1)


def test_rsi_edge_cases() -> None:
    up = pd.Series(np.arange(30, dtype=float))
    assert I.rsi(up, 14).dropna().eq(100.0).all()           # only gains -> 100 (was 50)
    down = pd.Series(np.arange(30, 0, -1, dtype=float))
    assert I.rsi(down, 14).dropna().eq(0.0).all()
    flat = pd.Series(np.full(30, 5.0))
    assert I.rsi(flat, 14).dropna().eq(50.0).all()
    assert I.rsi(up, 14).iloc[:14].isna().all()              # warm-up is NaN, not a number


def test_atr_equals_a_textbook_wilder_loop() -> None:
    df = _ohlc()
    tr = [df["high"].iloc[0] - df["low"].iloc[0]]
    for i in range(1, len(df)):
        h, l, pc = df["high"].iloc[i], df["low"].iloc[i], df["close"].iloc[i - 1]
        tr.append(max(h - l, abs(h - pc), abs(l - pc)))
    ref = _wilder_loop(tr, 14)
    got = I.atr(df["high"], df["low"], df["close"], 14).tolist()
    assert np.allclose(got, ref, equal_nan=True)


def test_ema_is_sma_seeded() -> None:
    s = pd.Series(np.arange(1.0, 41.0))
    e = I.ema(s, 10)
    assert e.iloc[:9].isna().all()
    assert e.iloc[9] == pytest.approx(5.5)                    # SMA of 1..10
    a = 2 / 11
    assert e.iloc[10] == pytest.approx(5.5 + a * (11 - 5.5))


def test_adx_equals_a_textbook_loop() -> None:
    df = _ohlc(300, seed=4)
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    pdm, mdm, tr = [], [], []
    for i in range(1, len(df)):
        up, dn = h[i] - h[i - 1], l[i - 1] - l[i]
        pdm.append(up if up > dn and up > 0 else 0.0)
        mdm.append(dn if dn > up and dn > 0 else 0.0)
        tr.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    sp, sm, st = (np.array(_wilder_loop(v, 14)) for v in (pdm, mdm, tr))
    pdi, mdi = 100 * sp / st, 100 * sm / st
    dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    first = int(np.flatnonzero(np.isfinite(dx))[0])
    adx_ref = np.full(len(dx), np.nan)
    adx_ref[first:] = _wilder_loop(list(dx[first:]), 14)
    got = I.adx(df["high"], df["low"], df["close"], 14)["adx"].to_numpy()[1:]
    ok = np.isfinite(adx_ref)
    assert np.allclose(got[ok], adx_ref[ok])


def test_bollinger_uses_the_population_std() -> None:
    s = pd.Series(np.r_[np.ones(19), 21.0])
    bb = I.bollinger_bands(s, 20, 2.0)
    assert bb["bb_middle"].iloc[-1] == pytest.approx(2.0)
    assert bb["bb_upper"].iloc[-1] == pytest.approx(2.0 + 2 * np.std(s.to_numpy()))


def test_supertrend_equals_the_tradingview_algorithm() -> None:
    df = _ohlc(300, seed=7)
    st = I.supertrend(df["high"], df["low"], df["close"], 10, 3.0)
    a = I.atr(df["high"], df["low"], df["close"], 10).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    c = df["close"].to_numpy()
    up, dn, d = np.nan, np.nan, np.nan
    for i in range(len(df)):
        if not np.isfinite(a[i]):
            continue
        ub, lb = hl2[i] + 3 * a[i], hl2[i] - 3 * a[i]
        if np.isnan(d):                                     # first valid bar: down, like ta.supertrend
            up, dn, d = ub, lb, -1.0
        else:
            up = ub if (ub < up or c[i - 1] > up) else up
            dn = lb if (lb > dn or c[i - 1] < dn) else dn
            d = (-1.0 if c[i] < dn else 1.0) if d > 0 else (1.0 if c[i] > up else -1.0)
        assert st["st_dir"].iloc[i] == d
        assert st["supertrend"].iloc[i] == pytest.approx(dn if d > 0 else up)


@pytest.mark.parametrize("fn", [
    lambda d: I.rsi(d["close"], 14),
    lambda d: I.atr(d["high"], d["low"], d["close"], 14),
    lambda d: I.ema(d["close"], 20),
    lambda d: I.adx(d["high"], d["low"], d["close"], 14)["adx"],
    lambda d: I.supertrend(d["high"], d["low"], d["close"], 10, 3.0)["supertrend"],
    lambda d: I.bollinger_bands(d["close"], 20)["bb_upper"],
])
def test_no_indicator_looks_ahead(fn) -> None:
    full = _ohlc(400, seed=9)
    part = full.iloc[:300]
    a, b = fn(part).to_numpy(), fn(full).to_numpy()[:300]
    assert np.allclose(a, b, equal_nan=True)


def test_momentum_uses_the_shared_indicators() -> None:
    """No private ATR/ADX copies with different conventions any more."""
    from trading_lab.strategies import momentum as M
    df = _ohlc(200, seed=2)
    assert np.allclose(M.atr(df, 14), I.atr(df["high"], df["low"], df["close"], 14), equal_nan=True)
    assert np.allclose(M.adx(df, 14), I.adx(df["high"], df["low"], df["close"], 14)["adx"])
