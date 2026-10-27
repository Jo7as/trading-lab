"""S-001 (docs/specs/smc_ict2022.yaml): one test group per rule, plus no-look-ahead."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trading_lab.strategies import smc_ict2022 as S
from trading_lab.strategies.smc_ict2022 import Prepared, Setup, SmcIct2022Config

CFG = SmcIct2022Config()


# ── rule 2: killzones in New York time, DST-aware ────────────────────────

def test_killzones_follow_new_york_time_across_dst() -> None:
    idx = pd.DatetimeIndex(["2024-03-08 13:30", "2024-03-08 13:25",     # EST: 08:30 / 08:25 NY
                            "2024-03-12 12:30", "2024-03-12 15:00",     # EDT: 08:30 / 11:00 NY (end excl.)
                            "2024-03-12 07:00"], tz="UTC")              # EDT: 03:00 NY (London)
    assert S.killzone_ids(idx, "US100").tolist() == [2, 0, 2, 0, 0]
    assert S.killzone_ids(idx, "GER40").tolist() == [0, 0, 0, 0, 1]
    assert S.killzone_ids(idx, "XAUUSD").tolist() == [2, 0, 2, 0, 1]


def test_session_end_per_market() -> None:
    ts = pd.Timestamp("2024-07-01 14:00", tz="UTC")
    assert pd.Timestamp(S.flat_time_ns(ts, "US100"), tz="UTC") == pd.Timestamp("2024-07-01 19:55", tz="UTC")
    assert pd.Timestamp(S.flat_time_ns(ts, "GER40"), tz="UTC") == pd.Timestamp("2024-07-01 15:25", tz="UTC")


# ── rule 1: bias only from CLOSED higher-timeframe bars ──────────────────

def test_htf_bias_uses_only_closed_bars() -> None:
    htf_close = np.array([10, 20, 30])
    trend = np.array([1, -1, 1])
    ltf_close = np.array([9, 10, 19, 25, 30])
    assert S.htf_bias_at(ltf_close, htf_close, trend).tolist() == [0, 1, 1, -1, 1]


# ── rule 6: targets ───────────────────────────────────────────────────────

def test_nearest_unbroken_target() -> None:
    idx, conf, price = np.array([2, 5, 8]), np.array([4, 7, 12]), np.array([105.0, 103.0, 110.0])
    close = np.full(15, 100.0)
    assert S.nearest_unbroken(idx, conf, price, close, 10, 100.0, 1) == 103.0
    close[6] = 104.0                                   # a close above 103 breaks it
    assert S.nearest_unbroken(idx, conf, price, close, 10, 100.0, 1) == 105.0
    assert np.isnan(S.nearest_unbroken(idx, conf, price, close, 3, 100.0, 1))   # nothing confirmed yet


def test_targets_need_the_minimum_rr() -> None:
    assert S.choose_targets(100, 1, 1, 101, 101.5, CFG) is None          # T2 only 1.5 R
    assert S.choose_targets(100, 1, 1, 101, 103, CFG) == (101, 103)
    t1, t2 = S.choose_targets(100, 1, 1, 104, 103, CFG)                 # T1 beyond T2 -> one exit
    assert np.isnan(t1) and t2 == 103
    t1, t2 = S.choose_targets(100, 2, -1, np.nan, np.nan, replace(CFG, ziel="fest_3r"))
    assert np.isnan(t1) and t2 == 94


# ── rules 6/7: execution on 1-minute bid/ask ─────────────────────────────

SPREAD = 0.1


def _prep(rows: list[tuple[float, float, float, float]], start: str = "2024-07-01 13:00") -> Prepared:
    a = np.asarray(rows, float)
    t = pd.date_range(start, periods=len(a), freq="1min", tz="UTC")
    bid = {k: a[:, j] for j, k in enumerate(("open", "high", "low", "close"))}
    ask = {k: v + SPREAD for k, v in bid.items()}
    return Prepared(asset="US100", m1_ns=t.as_unit("ns").asi8, bid=bid, ask=ask)


def _setup(d=1, entry=100.0, stop=98.0, t1=102.0, t2=106.0, extreme=98.2) -> Setup:
    return Setup(d, 0, 0, extreme, entry, stop, t1, t2, (0, 2))


def _run(rows, setup, cfg=replace(CFG, slippage_bp=0.0)):
    p = _prep(rows)
    return S.simulate(setup, p, int(p.m1_ns[0]), int(p.m1_ns[-1]) + 1, cfg)


FLAT = (101, 101.2, 100.8, 101)


def test_a_touch_is_not_a_fill_only_trading_through() -> None:
    # ask low = bid low + 0.1: bid low 99.9 -> ask low 100.0 = touch; 99.85 -> 99.95 = through
    touch = [(101, 101, 99.9, 100.5)] + [FLAT] * 3
    assert _run(touch, _setup()) is None
    through = [(101, 101, 99.85, 100.5)] + [FLAT] * 3
    assert _run(through, _setup()) is not None


def test_no_fill_if_the_target_comes_first() -> None:
    assert _run([(101, 102.1, 100.5, 102), (101, 101, 99, 99.5)], _setup()) is None        # T1 first


def test_through_the_limit_and_the_stop_in_one_minute_is_a_full_loss() -> None:
    """For a long the sweep extreme lies BELOW the limit, so price always passes the limit
    first: a minute that trades through both is filled and stopped (conservative), never
    quietly cancelled."""
    tr = _run([(100.5, 100.6, 97.9, 98.1), FLAT], _setup())
    assert tr["exit_reason"] == "stop" and tr["r"] == pytest.approx(-1.0)


def test_t1_partial_then_break_even() -> None:
    rows = [(101, 101, 99.8, 100.2),          # fill at 100
            (100.2, 102.1, 100.1, 102),        # T1 102: half closed, stop -> 100
            (101, 101, 99.9, 100)]             # back to 100: break-even
    tr = _run(rows, _setup())
    assert tr["exit_reason"] == "breakeven"
    assert tr["r"] == pytest.approx(0.5 * 2 / 2)                      # +0.5 R


def test_t1_then_t2() -> None:
    rows = [(101, 101, 99.8, 100.2), (100.2, 102.1, 100.1, 102), (102, 106.2, 101.5, 106)]
    tr = _run(rows, _setup())
    assert tr["exit_reason"] == "t2"
    assert tr["r"] == pytest.approx(0.5 * 1.0 + 0.5 * 3.0)           # T1 = 1 R, T2 = 3 R


def test_full_stop_with_a_gap_fills_at_the_open() -> None:
    rows = [(101, 101, 99.8, 100.2), (97.0, 97.5, 96.5, 97)]           # opens below the stop
    tr = _run(rows, _setup())
    assert tr["exit_reason"] == "stop" and tr["r"] == pytest.approx((97.0 - 100) / 2)


def test_no_target_inside_the_fill_minute() -> None:
    rows = [(101, 106.5, 99.8, 100.2), FLAT, FLAT]                     # same minute reaches T2
    tr = _run(rows, _setup(t1=np.nan))
    assert tr["exit_reason"] != "t2" or tr["exit_time"] > pd.Timestamp("2024-07-01 13:00", tz="UTC")


def test_session_end_closes_the_trade() -> None:
    rows = [(101, 101, 99.8, 100.2)] + [FLAT] * 400                  # 13:00 + 400 min > 19:55 UTC? no -
    p = _prep(rows, start="2024-07-01 19:50")                          # flat 15:55 NY = 19:55 UTC
    tr = S.simulate(_setup(), p, int(p.m1_ns[0]), int(p.m1_ns[-1]) + 1, replace(CFG, slippage_bp=0.0))
    assert tr["exit_reason"] == "session"
    assert tr["exit_time"] == pd.Timestamp("2024-07-01 19:55", tz="UTC")


def test_short_is_the_mirror() -> None:
    # short limit 100: fills when the BID trades above 100; exits on the ask
    rows = [(99, 100.15, 98.9, 99.8), (99.8, 99.9, 97.8, 98)]          # ask low 97.9 <= T1 98
    tr = _run(rows, _setup(d=-1, entry=100, stop=102, t1=98, t2=94, extreme=101.8))
    assert tr is not None and tr["direction"] == -1
    assert tr["r"] > 0


# ── whole engine: no look-ahead and every trade obeys the spec ──────────

def _random_m1(days: int = 30, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2024-03-04", periods=days * 1440, freq="1min", tz="UTC")
    t = t[t.dayofweek < 5]
    c = 18000 + np.cumsum(rng.normal(0, 4, len(t)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + rng.uniform(0, 3, len(t))
    l = np.minimum(o, c) - rng.uniform(0, 3, len(t))
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=t)
    for k in ("open", "high", "low", "close"):
        df[f"ask_{k}"] = df[k] + 1.0
    return df


@pytest.fixture(scope="module")
def random_run():
    m1 = _random_m1()
    cfg = replace(CFG, bias="aus", min_rr=1.5)
    prep = S.prepare(m1, "US100", combos=("1h/5m",))
    return m1, cfg, prep, S.run_backtest(prep, cfg)["trades"]


def test_trades_obey_the_spec(random_run) -> None:
    m1, cfg, prep, trades = random_run
    assert trades, "random data should produce at least a few setups - otherwise this proves nothing"
    kz = S.killzone_ids(prep.combos["1h/5m"]["ltf"].index, "US100")
    for tr in trades:
        d = tr["direction"]
        risk = d * (tr["entry"] - tr["stop"])
        assert risk > 0
        assert d * (tr["target"] - tr["entry"]) / risk >= cfg.min_rr - 1e-9
        assert kz[tr["sweep_idx"]] > 0                                  # sweep inside a killzone
        assert tr["mss_idx"] - tr["sweep_idx"] <= cfg.mss_window
        assert tr["r"] >= -1.2                                          # never much worse than the stop


def test_no_look_ahead_in_the_engine(random_run) -> None:
    m1, cfg, prep, trades = random_run
    cut = m1.index[len(m1) * 2 // 3]
    part = S.run_backtest(S.prepare(m1[m1.index < cut], "US100", combos=("1h/5m",)), cfg)["trades"]
    done = [t for t in trades if t["exit_time"] < cut - pd.Timedelta(hours=6)]
    key = lambda t: (t["entry_time"], t["direction"], round(t["entry"], 6), round(t["r"], 9))  # noqa: E731
    assert {key(t) for t in done} <= {key(t) for t in part}


# ── rule 5 detail: the FVG of the displacement leg may be confirmed at MSS + 1 ──

class _Res:
    def __init__(self, bars, zones):
        self.bars, self.zones = bars, zones


def _fake_prep(fvg_created: int) -> Prepared:
    from trading_lab.market_structure import CHOCH_UP, Zone
    n = 20
    idx = pd.date_range("2024-07-01 12:30", periods=n, freq="5min", tz="UTC")     # 08:30 NY
    low = np.full(n, 100.0); low[3] = 97.0                                          # sweep extreme
    ltf = pd.DataFrame({"open": 100.0, "high": 101.0, "low": low, "close": 100.5}, index=idx)
    bars = pd.DataFrame({"sweep_low": 0, "sweep_high": 0, "int_event": 0,
                         "int_swing_sweep_low": 0, "int_swing_sweep_high": 0}, index=idx)
    bars.iloc[3, 0] = 1                                                             # sweep at bar 3
    bars.iloc[6, 2] = CHOCH_UP                                                      # MSS at bar 6
    z = Zone("fvg", 1, top=100.6, bottom=99.4, created_idx=fvg_created, origin_idx=fvg_created - 1)
    l_close = idx.as_unit("ns").asi8 + pd.Timedelta("5min").value
    d = {"ltf": ltf, "res": _Res(bars, [z]), "ldur": pd.Timedelta("5min").value, "l_close": l_close,
         "atr": np.full(n, 1.0), "kz": np.full(n, 2), "day": np.zeros(n, int), "bias": np.ones(n, int),
         "int_hi": (np.array([1]), np.array([2]), np.array([102.0])), "int_lo": (np.array([], int),) * 2 + (np.array([]),),
         "htf_hi": (np.array([0]), np.array([0]), np.array([110.0])), "htf_lo": (np.array([], int),) * 2 + (np.array([]),),
         "htf_close": np.array([100.0]), "h_close_ns": np.array([l_close[0] - 1])}
    p = Prepared(asset="US100", m1_ns=np.array([], dtype=np.int64), bid={}, ask={})
    p.combos["1h/5m"] = d
    return p


def test_fvg_confirmed_one_bar_after_the_mss_is_used() -> None:
    s = S.find_setups(_fake_prep(fvg_created=7), CFG)
    assert len(s) == 1 and s[0].mss_idx == 6 and s[0].ready_idx == 7
    assert s[0].entry == pytest.approx(100.0) and s[0].stop == pytest.approx(97.0 - 0.1)
    assert S.find_setups(_fake_prep(fvg_created=8), CFG) == []        # two bars later: not this leg


def test_one_trade_per_killzone_counts_filled_trades_only(monkeypatch) -> None:
    p = _fake_prep(fvg_created=7)
    a = _setup(); a.ready_idx = 7
    b = _setup(); b.ready_idx = 9
    monkeypatch.setattr(S, "find_setups", lambda prep, cfg: [a, b])
    fake = {"exit_time": pd.Timestamp("2024-07-01 13:40", tz="UTC")}
    monkeypatch.setattr(S, "simulate", lambda s, *x: None if s is a else dict(fake))
    assert len(S.run_backtest(p, CFG)["trades"]) == 1                 # a unfilled -> b may trade
    monkeypatch.setattr(S, "simulate", lambda s, *x: dict(fake, exit_time=pd.Timestamp("2024-07-01 13:10", tz="UTC")))
    assert len(S.run_backtest(p, CFG)["trades"]) == 1                 # a filled -> b blocked (same killzone)


def test_a_sweep_of_an_internal_swing_low_counts() -> None:
    """Spec rule 3: ANY unbroken 5m swing low - also an internal one - can be swept."""
    p = _fake_prep(fvg_created=7)
    bars = p.combos["1h/5m"]["res"].bars
    bars.iloc[3, 0] = 0                                                # no external / pool sweep ...
    bars.loc[bars.index[3], "int_swing_sweep_low"] = 1                 # ... but an internal one
    assert len(S.find_setups(p, CFG)) == 1


def test_mss_may_be_an_internal_bos_too() -> None:
    """Spec change 2026-10-27: the first internal break after the sweep counts, BOS or CHoCH."""
    from trading_lab.market_structure import BOS_UP
    p = _fake_prep(fvg_created=7)
    p.combos["1h/5m"]["res"].bars.iloc[6, 2] = BOS_UP
    assert len(S.find_setups(p, CFG)) == 1
