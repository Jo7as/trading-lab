"""S-004 (docs/specs/breakout_fade_fx.yaml): one test group per rule, plus no-look-ahead."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trading_lab.market_structure import BOS_DN, BOS_UP, CHOCH_UP
from trading_lab.strategies import breakout_fade_fx as B
from trading_lab.strategies.breakout_fade_fx import BreakoutFadeConfig, Prepared
from trading_lab.strategies.fib_retracement import rollover_weights

CFG = BreakoutFadeConfig()


# ── rules 1-3: which breaks are faded ────────────────────────────────────

def _close(n: int, start: str = "2024-07-01 08:00") -> np.ndarray:
    return pd.date_range(start, periods=n, freq="1h", tz="UTC").as_unit("ns").asi8


def test_a_break_up_in_an_uptrend_is_faded_with_a_short() -> None:
    ev = np.array([0, 0, BOS_UP, 0, BOS_DN])
    trend = np.array([1, 1, 1, -1, -1])
    assert B.signals(ev, trend, _close(5), CFG) == [(2, -1), (4, 1)]


def test_the_trend_filter_uses_the_previous_bar() -> None:
    ev = np.array([0, CHOCH_UP])                    # this break itself flips the trend up
    trend = np.array([-1, 1])
    assert B.signals(ev, trend, _close(2), CFG) == []                       # prior bar was down
    assert B.signals(ev, trend, _close(2), replace(CFG, trendfilter="egal")) == [(1, -1)]


def test_session_filter_on_the_signal_close() -> None:
    ev = np.array([BOS_UP, BOS_UP])
    trend = np.array([1, 1, 1])[:2]
    close = pd.DatetimeIndex(["2024-07-01 06:00", "2024-07-01 12:00"], tz="UTC").as_unit("ns").asi8
    cfg = replace(CFG, session="london_ny", trendfilter="egal")
    assert B.signals(ev, trend, close, cfg) == [(1, -1)]


# ── rules 4-5: execution ──────────────────────────────────────────────────

SPREAD = 0.0002


def _prep(rows, start="2024-07-02 10:00") -> Prepared:
    a = np.asarray(rows, float)
    t = pd.date_range(start, periods=len(a), freq="1min", tz="UTC")
    bid = {k: a[:, j] for j, k in enumerate(("open", "high", "low", "close"))}
    return Prepared(asset="EURUSD", m1_ns=t.as_unit("ns").asi8, bid=bid,
                    ask={k: v + SPREAD for k, v in bid.items()}, tfs={})


C0 = replace(CFG, slippage_bp=0.0)
F = (1.1000, 1.1002, 1.0998, 1.1000)


def _exec(rows, stop=1.1020, target=np.nan, deadline_min=None, d=-1):
    p = _prep(rows)
    dl = int(p.m1_ns[deadline_min]) if deadline_min is not None else int(p.m1_ns[-1]) + 10**12
    return B.execute(p, C0, d, int(p.m1_ns[0]), stop, target, dl)


def test_short_enters_at_the_bid_open_and_exits_on_time() -> None:
    tr = _exec([F, F, (1.0990, 1.0992, 1.0988, 1.0990), (1.0990, 1.0991, 1.0989, 1.0990)], deadline_min=3)
    assert tr["entry"] == pytest.approx(1.1000)
    assert tr["exit_reason"] == "zeit"
    # exit buys the ASK at the open of the deadline minute: 1.0990 + 0.0002
    assert tr["r"] == pytest.approx((1.1000 - 1.0992) / 0.0020)


def test_stop_first_and_not_target_in_the_entry_minute() -> None:
    tr = _exec([(1.1000, 1.1030, 1.0980, 1.1000), F], target=1.0990)       # entry minute spans both
    assert tr["exit_reason"] == "stop" and tr["r"] == pytest.approx(-1.0)
    tr = _exec([(1.1000, 1.1005, 1.0995, 1.1000), (1.0995, 1.0996, 1.0985, 1.0990)], target=1.0990)
    assert tr["exit_reason"] == "ziel"


def test_gap_through_the_stop_fills_at_the_open() -> None:
    tr = _exec([F, (1.1030, 1.1035, 1.1028, 1.1030)])
    assert tr["exit_reason"] == "stop"
    assert tr["r"] == pytest.approx(-(1.1032 - 1.1000) / 0.0020)          # ask open 1.1030 + spread


def test_no_trade_when_the_target_is_already_passed_at_the_open() -> None:
    assert _exec([F, F], target=1.1010) is None                           # short target above entry


def test_no_entry_after_the_friday_cut() -> None:
    assert not B.entry_allowed(pd.Timestamp("2024-07-05 21:00", tz="UTC").value)   # Friday 21:00
    assert not B.entry_allowed(pd.Timestamp("2024-07-06 10:00", tz="UTC").value)   # Saturday
    assert B.entry_allowed(pd.Timestamp("2024-07-05 20:59", tz="UTC").value)
    p = _prep([F, F], start="2024-07-05 22:00")
    assert B.execute(p, C0, -1, int(p.m1_ns[0]), 1.1020, np.nan, int(p.m1_ns[-1]) + 10**12) is None


def test_fx_swaps_triple_on_wednesday() -> None:
    wed = pd.Timestamp("2024-07-03 18:00", tz="UTC").value                 # Wed 14:00 NY
    thu = pd.Timestamp("2024-07-04 18:00", tz="UTC").value
    assert rollover_weights(wed, thu, triple_dow=2) == 3.0
    assert rollover_weights(wed, thu) == 1.0                               # index convention: Friday


# ── whole engine ───────────────────────────────────────────────────────────

def _random_m1(days: int = 120, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2023-01-02", periods=days * 1440, freq="1min", tz="UTC")
    t = t[t.dayofweek < 5]
    c = 1.10 * np.exp(np.cumsum(rng.normal(0, 0.00012, len(t))))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + rng.uniform(0, 0.00008, len(t))
    l = np.minimum(o, c) - rng.uniform(0, 0.00008, len(t))
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=t)
    for k in ("open", "high", "low", "close"):
        df[f"ask_{k}"] = df[k] + 0.00002
    return df


@pytest.fixture(scope="module")
def random_run():
    m1 = _random_m1()
    prep = B.prepare(m1, "EURUSD", tfs=("1h",))
    return m1, prep, B.run_backtest(prep, CFG)["trades"]


def test_trades_obey_the_spec(random_run) -> None:
    m1, prep, trades = random_run
    assert len(trades) > 20, "random data should produce trades - otherwise this proves nothing"
    for tr in trades:
        d = tr["direction"]
        assert d * (tr["entry"] - tr["stop"]) > 0
        assert (tr["exit_time"] - tr["entry_time"]) <= pd.Timedelta(hours=4, minutes=1)
        assert tr["r"] >= -1.5
    assert {t["direction"] for t in trades} == {-1, 1}


def test_no_look_ahead_in_the_engine(random_run) -> None:
    m1, prep, trades = random_run
    cut = m1.index[len(m1) * 2 // 3]
    part = B.run_backtest(B.prepare(m1[m1.index < cut], "EURUSD", tfs=("1h",)), CFG)["trades"]
    done = [t for t in trades if t["exit_time"] < cut - pd.Timedelta(days=1)]
    key = lambda t: (t["entry_time"], t["direction"], round(t["entry"], 8), round(t["r"], 9))  # noqa: E731
    assert done and {key(t) for t in done} <= {key(t) for t in part}
