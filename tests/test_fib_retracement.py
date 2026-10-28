"""S-002 (docs/specs/fib_retracement.yaml): one test group per rule, plus no-look-ahead."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trading_lab.market_structure import BOS_UP, CHOCH_DN
from trading_lab.strategies import fib_retracement as F
from trading_lab.strategies.fib_retracement import FibConfig, Prepared

CFG = FibConfig()


# ── rule 4: fib arithmetic, control levels, targets ──────────────────────

def test_fib_prices() -> None:
    assert F.fib_price(100, 200, 0.618) == pytest.approx(138.2)
    assert F.fib_price(100, 200, 0.786) == pytest.approx(121.4)
    assert F.fib_price(100, 200, -0.272) == pytest.approx(227.2)
    assert F.fib_price(200, 100, 0.5) == pytest.approx(150.0)           # short leg: A above B


def test_control_levels_avoid_the_fib_numbers_and_are_reproducible() -> None:
    xs = [F.control_level(7, k) for k in range(2000)]
    assert all(0.40 <= x <= 0.75 for x in xs)
    assert all(abs(x - f) > 0.03 for x in xs for f in F.FIB_LEVELS)
    assert F.control_level(7, 5) == F.control_level(7, 5)
    assert len({round(x, 6) for x in xs}) > 1000                       # really random across impulses


def test_targets() -> None:
    a, b, e, r = 100.0, 200.0, 138.2, 40.0
    t = lambda z: F.targets(replace(CFG, ziel=z), a, b, e, r, 1)        # noqa: E731
    assert t("altes_hoch")[1] == 200 and np.isnan(t("altes_hoch")[0])
    assert t("ext_0618")[1] == pytest.approx(261.8)
    assert t("fest_2r")[1] == pytest.approx(218.2)
    assert t("teil_hoch_ext0618") == pytest.approx((200, 261.8))


# ── data: daily bars end at 17:00 New York, DST-aware ─────────────────────

def test_daily_bars_close_at_17_new_york() -> None:
    idx = pd.date_range("2024-03-07 12:00", "2024-03-13 12:00", freq="1h", tz="UTC")
    m1 = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "ask_open": 1.0,
                       "ask_high": 1.0, "ask_low": 1.0, "ask_close": 1.0}, index=idx)
    d = F.daily_ny17(m1)
    ends = pd.to_datetime(F._close_ns(d, "1D"), utc=True)
    by_start = dict(zip(d.index.tz_convert("America/New_York").strftime("%m-%d"), ends.strftime("%m-%d %H:%M")))
    assert by_start["03-07"] == "03-08 22:00"                           # EST: 17:00 NY = 22:00 UTC
    assert by_start["03-11"] == "03-12 21:00"                           # EDT: 17:00 NY = 21:00 UTC
    assert not (d.index.tz_convert("America/New_York").dayofweek == 5).any()


# ── rules 1-3, 5: impulse tracking ────────────────────────────────────────

def _impulse_arrays(n=20):
    high = np.full(n, 110.0); low = np.full(n, 105.0); close = np.full(n, 108.0)
    low[2] = 100.0                                                      # A
    high[3:6] = [120, 130, 140]                                         # the impulse
    atr = np.full(n, 5.0)
    ev = np.zeros(n, int); ev[5] = BOS_UP
    pl = np.full(n, np.nan); pl[5:] = 100.0
    return high, low, close, atr, ev, pl, np.full(n, np.nan)


def test_impulse_anchors_and_moving_b() -> None:
    h, l, c, a, ev, pl, ph = _impulse_arrays()
    h[7] = 150.0                                                        # a new high extends B
    imps = F.track_impulses(h, l, c, a, ev, pl, ph, CFG)
    assert imps[4] is None
    assert (imps[5].a, imps[5].b, imps[5].direction) == (100.0, 140.0, 1)
    assert imps[8].b == 150.0


def test_impulse_dies_on_a_close_below_a_and_after_expiry() -> None:
    h, l, c, a, ev, pl, ph = _impulse_arrays()
    c[8] = 99.0
    imps = F.track_impulses(h, l, c, a, ev, pl, ph, CFG)
    assert imps[7] is not None and imps[8] is None and imps[9] is None
    h, l, c, a, ev, pl, ph = _impulse_arrays()
    imps = F.track_impulses(h, l, c, a, ev, pl, ph, CFG)
    assert imps[15] is not None and imps[16] is None                   # valid for 10 bars after bar 5


def test_impulse_must_be_big_enough() -> None:
    h, l, c, a, ev, pl, ph = _impulse_arrays()
    a[:] = 20.0                                                         # 40 points < 3 x 20
    assert all(x is None for x in F.track_impulses(h, l, c, a, ev, pl, ph, CFG))


# ── execution: manage() on 1-minute bid/ask ──────────────────────────────

SPREAD = 0.1


def _prep(rows, start="2024-07-01 13:00", asset="US100") -> Prepared:
    a = np.asarray(rows, float)
    t = pd.date_range(start, periods=len(a), freq="1min", tz="UTC")
    bid = {k: a[:, j] for j, k in enumerate(("open", "high", "low", "close"))}
    return Prepared(asset=asset, m1_ns=t.as_unit("ns").asi8, bid=bid,
                    ask={k: v + SPREAD for k, v in bid.items()}, combos={})


NO_CHOCH = np.array([], dtype=np.int64)
C0 = replace(CFG, slippage_bp=0.0)
FLAT = (101, 101.2, 100.8, 101)


def _manage(rows, t1=np.nan, t2=106.0, deadline=None, choch=NO_CHOCH, start="2024-07-01 13:00"):
    p = _prep(rows, start=start)
    dl = deadline if deadline is not None else int(p.m1_ns[-1]) + 10**12
    return F.manage(p, C0, 1, 0, 100.0, 98.0, t1, t2, dl, choch)


def test_stop_is_checked_first_and_target_not_in_the_fill_minute() -> None:
    tr = _manage([(100, 107, 99.5, 100.5), FLAT, (101, 106.5, 100.9, 106)])
    assert tr["exit_reason"] == "ziel" and tr["exit_time"] > pd.Timestamp("2024-07-01 13:00", tz="UTC")
    tr = _manage([(100, 100.5, 99.5, 100), (100, 106.5, 97.5, 101)])   # same minute: stop and target
    assert tr["exit_reason"] == "stop" and tr["r"] == pytest.approx(-1.0, abs=1e-6)


def test_partial_at_the_old_high_then_break_even() -> None:
    tr = _manage([(100, 100.5, 99.5, 100), (100, 102.2, 100.1, 102), (101, 101, 99.9, 100)], t1=102.0)
    assert tr["exit_reason"] == "breakeven" and tr["r"] == pytest.approx(0.5)


def test_time_stop_exits_on_the_last_price_before_the_cut() -> None:
    p = _prep([(100, 100.5, 99.5, 100), FLAT, FLAT, (90, 90, 89, 89)])
    dl = int(p.m1_ns[3])
    tr = F.manage(p, C0, 1, 0, 100.0, 98.0, np.nan, 106.0, dl, NO_CHOCH)
    assert tr["exit_reason"] == "zeit" and tr["r"] == pytest.approx((101 - 100) / 2)


def test_opposite_choch_closes_the_trade() -> None:
    p = _prep([(100, 100.5, 99.5, 100), FLAT, (101.5, 101.7, 101.3, 101.5), FLAT])
    tr = F.manage(p, C0, 1, 0, 100.0, 98.0, np.nan, 106.0, int(p.m1_ns[-1]) + 10**12, np.array([p.m1_ns[2]]))
    assert tr["exit_reason"] == "gegenbruch" and tr["r"] == pytest.approx((101.5 - 100) / 2)


def test_swaps_per_rollover_friday_counts_three() -> None:
    fri = pd.Timestamp("2024-07-05 18:00", tz="UTC").value                  # Fri 14:00 NY
    mon = pd.Timestamp("2024-07-08 18:00", tz="UTC").value                  # Mon 14:00 NY
    assert F.rollover_weights(fri, mon) == 3.0
    wed = pd.Timestamp("2024-07-03 18:00", tz="UTC").value
    assert F.rollover_weights(wed, fri) == 2.0                              # Wed + Thu rollovers
    tr_a = _manage([(100, 100.5, 99.5, 100), FLAT, (101, 106.5, 100.9, 106)])
    assert tr_a["swap_r"] == 0.0                                            # no rollover inside


def test_friday_cut_for_the_standard_account() -> None:
    t = pd.Timestamp("2024-07-03 10:00", tz="UTC").value
    assert pd.Timestamp(F._friday_cut_ns(t), tz="UTC") == pd.Timestamp("2024-07-05 21:00", tz="UTC")


# ── whole engine: no look-ahead, rules hold on every trade ───────────────

def _random_m1(days: int = 240, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2023-01-02", periods=days * 288, freq="5min", tz="UTC")     # 5-min steps keep it fast
    t = t[t.dayofweek < 5]
    c = 15000 * np.exp(np.cumsum(rng.normal(0, 0.0012, len(t))))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.0008, len(t)))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.0008, len(t)))
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=t)
    for k in ("open", "high", "low", "close"):
        df[f"ask_{k}"] = df[k] + 1.0
    return df


@pytest.fixture(scope="module")
def random_run():
    m1 = _random_m1()
    prep = F.prepare(m1, "US100", combos=("intraday",))
    cfg = replace(CFG, zeitrahmen="intraday", level="0.5")
    return m1, prep, cfg, F.run_backtest(prep, cfg)["trades"]


def test_trades_obey_the_spec(random_run) -> None:
    m1, prep, cfg, trades = random_run
    assert trades, "random data should produce some trades - otherwise this proves nothing"
    for tr in trades:
        d = tr["direction"]
        assert d * (tr["entry"] - tr["stop"]) > 0                           # stop on the right side
        assert tr["entry"] == pytest.approx(F.fib_price(tr["a"], tr["b"], 0.5))   # limit at the level
        assert d * (tr["target"] - tr["entry"]) > 0
        assert tr["r"] >= -1.3


def test_no_look_ahead_in_the_engine(random_run) -> None:
    m1, prep, cfg, trades = random_run
    cut = m1.index[len(m1) * 2 // 3]
    part = F.run_backtest(F.prepare(m1[m1.index < cut], "US100", combos=("intraday",)), cfg)["trades"]
    done = [t for t in trades if t["exit_time"] < cut - pd.Timedelta(days=2)]
    key = lambda t: (t["entry_time"], t["direction"], round(t["entry"], 6), round(t["r"], 9))  # noqa: E731
    assert done and {key(t) for t in done} <= {key(t) for t in part}


def test_no_intraday_entry_after_the_session_end() -> None:
    p = _prep([FLAT], start="2024-07-01 20:30")                         # 16:30 NY, after 15:55
    cfg = replace(CFG, zeitrahmen="intraday")
    assert not F._may_enter(p, cfg, int(p.m1_ns[0]))
    p2 = _prep([FLAT], start="2024-07-01 17:00")                        # 13:00 NY
    assert F._may_enter(p2, cfg, int(p2.m1_ns[0]))
    assert F._may_enter(p, replace(CFG, zeitrahmen="swing"), int(p.m1_ns[0]))


def test_engine_never_enters_after_the_session_end(random_run) -> None:
    m1, prep, cfg, trades = random_run
    for tr in trades:
        flat = F.flat_time_ns(tr["entry_time"], "US100")
        assert tr["entry_time"].value < flat
