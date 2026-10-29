"""Crypto trade dumps -> signed flow bars. Hand-made dumps only, never the network."""
from __future__ import annotations

import gzip
import io
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from trading_lab.adapters import crypto_trades as ct

DAY = date(2024, 1, 1)
T0_MS = 1704067200000          # 2024-01-01 00:00:00 UTC


def _zip(name: str, text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, text)
    return buf.getvalue()


BINANCE_HEADER = (
    "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
    f"1,100.0,2.0,1,1,{T0_MS + 1000},false\n"        # buyer aggressive, bar 00:00:00
    f"2,101.0,1.0,2,3,{T0_MS + 9999},true\n"         # seller aggressive, still bar 00:00:00
    f"3,102.0,3.0,4,4,{T0_MS + 10000},false\n"       # first trade of bar 00:00:10
)


def test_binance_header_and_aggressor_sign():
    tr = ct.parse_binance_aggtrades(_zip("BTCUSDT-aggTrades-2024-01-01.csv", BINANCE_HEADER))
    assert list(tr["side"]) == [1, -1, 1]            # is_buyer_maker=true -> seller took
    assert tr["ts"].iloc[0] == pd.Timestamp("2024-01-01 00:00:01", tz="UTC")
    assert list(tr["qty"]) == [2.0, 1.0, 3.0]


def test_binance_without_header_same_result():
    no_header = BINANCE_HEADER.split("\n", 1)[1]
    a = ct.parse_binance_aggtrades(_zip("x.csv", BINANCE_HEADER))
    b = ct.parse_binance_aggtrades(no_header.encode())
    pd.testing.assert_frame_equal(a, b)


def test_binance_microsecond_timestamps():
    """Binance spot dumps use microseconds since 2025-01-01."""
    us = T0_MS * 1000 + 1_500_000
    tr = ct.parse_binance_aggtrades(f"1,100,1,1,1,{us},False\n".encode())
    assert tr["ts"].iloc[0] == pd.Timestamp("2024-01-01 00:00:01.5", tz="UTC")


def test_binance_bad_maker_flag_raises():
    with pytest.raises(ValueError):
        ct.parse_binance_aggtrades(f"1,100,1,1,1,{T0_MS},maybe\n".encode())


def test_bybit_seconds_with_decimals_and_taker_side():
    text = ("timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional\n"
            "1704067200.25,SOLUSDT,Sell,5,100.0,MinusTick,a,0,5,500\n"
            "1704067215.5,SOLUSDT,Buy,1,101.0,PlusTick,b,0,1,101\n")
    tr = ct.parse_bybit_trades(gzip.compress(text.encode()))
    assert list(tr["side"]) == [-1, 1]
    assert tr["ts"].iloc[0] == pd.Timestamp("2024-01-01 00:00:00.25", tz="UTC")
    assert tr["ts"].iloc[1] == pd.Timestamp("2024-01-01 00:00:15.5", tz="UTC")


def test_bars_split_buy_sell_ohlc_vwap_and_boundary():
    tr = ct.parse_binance_aggtrades(BINANCE_HEADER.encode())
    bars = ct.trades_to_bars(tr, DAY, "10s")
    b0 = bars.loc[pd.Timestamp("2024-01-01 00:00:00", tz="UTC")]
    assert b0["buy_qty"] == 2.0 and b0["sell_qty"] == 1.0
    assert b0["open"] == 100.0 and b0["close"] == 101.0
    assert b0["high"] == 101.0 and b0["low"] == 100.0
    assert b0["vwap"] == pytest.approx((100 * 2 + 101 * 1) / 3)
    assert b0["buy_notional"] == 200.0 and b0["sell_notional"] == 101.0
    assert b0["n_trades"] == 2
    b1 = bars.loc[pd.Timestamp("2024-01-01 00:00:10", tz="UTC")]  # 10.000 s belongs to the next bar
    assert b1["buy_qty"] == 3.0 and b1["n_trades"] == 1


def test_bars_cover_full_day_grid():
    tr = ct.parse_binance_aggtrades(BINANCE_HEADER.encode())
    bars = ct.trades_to_bars(tr, DAY, "10s")
    assert len(bars) == 8640
    assert bars.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert bars.index[-1] == pd.Timestamp("2024-01-01 23:59:50", tz="UTC")
    empty = bars.iloc[100]
    assert empty["buy_qty"] == 0 and empty["n_trades"] == 0 and np.isnan(empty["close"])
    # quarter-hour opening bars sit exactly on the clock marks
    assert bars.index[90] == pd.Timestamp("2024-01-01 00:15:00", tz="UTC")


def test_trades_outside_day_raise():
    tr = ct.parse_binance_aggtrades(f"1,100,1,1,1,{T0_MS - 1},false\n".encode())
    with pytest.raises(ValueError):
        ct.trades_to_bars(tr, DAY, "10s")


def test_load_day_caches_and_marks_missing(tmp_path):
    calls: list[str] = []

    def fake_fetch(url: str):
        calls.append(url)
        return None if "2024-01-02" in url else _zip("x.csv", BINANCE_HEADER)

    a = ct.load_day("binance_um", "BTCUSDT", DAY, cache_dir=tmp_path, fetch=fake_fetch)
    b = ct.load_day("binance_um", "BTCUSDT", DAY, cache_dir=tmp_path, fetch=fake_fetch)
    pd.testing.assert_frame_equal(a, b)
    assert len(calls) == 1
    assert calls[0].endswith("/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-01.zip")

    assert ct.load_day("binance_um", "BTCUSDT", date(2024, 1, 2), cache_dir=tmp_path, fetch=fake_fetch) is None
    assert ct.load_day("binance_um", "BTCUSDT", date(2024, 1, 2), cache_dir=tmp_path, fetch=fake_fetch) is None
    assert len(calls) == 2                            # the 404 is remembered


def test_load_bars_range_keeps_gaps_visible(tmp_path):
    def fake_fetch(url: str):
        return None if "2024-01-02" in url else _zip("x.csv", BINANCE_HEADER)

    # the fixture holds 2024-01-01 trades, so only day 1 can carry data; day 2 is a gap
    bars = ct.load_bars("binance_um", "BTC", DAY, date(2024, 1, 2), cache_dir=tmp_path, fetch=fake_fetch)
    assert len(bars) == 2 * 8640
    assert bars.attrs["missing_days"] == [date(2024, 1, 2)]
    assert bars.loc["2024-01-02", "n_trades"].sum() == 0
    assert bars.index.is_monotonic_increasing


def test_load_day_offline_without_cache_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ct.load_day("bybit_linear", "SOLUSDT", DAY, cache_dir=tmp_path, allow_download=False)


def test_venue_urls():
    assert ct.VENUES["bybit_linear"].url("SOLUSDT", DAY) == \
        "https://public.bybit.com/trading/SOLUSDT/SOLUSDT2024-01-01.csv.gz"
    assert ct.VENUES["binance_spot"].url("SOLUSDT", DAY) == \
        "https://data.binance.vision/data/spot/daily/aggTrades/SOLUSDT/SOLUSDT-aggTrades-2024-01-01.zip"
