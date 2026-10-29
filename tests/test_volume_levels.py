"""H10 volume-profile levels on hand-made 1-min paths (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_lab import market_structure as ms
from trading_lab import volume_levels as vl
from trading_lab.level_flow import Level


def m1_from(close, start="2024-01-01", high=None, low=None, vol=None, buy=None, sell=None):
    close = np.asarray(close, dtype=float)
    n = len(close)
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC", name="ts")
    v = np.ones(n) if vol is None else np.asarray(vol, float)
    b = v / 2 if buy is None else np.asarray(buy, float)
    s = v / 2 if sell is None else np.asarray(sell, float)
    return pd.DataFrame({
        "open": close, "high": close if high is None else np.asarray(high, float),
        "low": close if low is None else np.asarray(low, float), "close": close, "vwap": close,
        "buy_qty": b / close, "sell_qty": s / close, "buy_notional": b, "sell_notional": s,
        "n_trades": 1}, index=idx)


# ── profile nodes ───────────────────────────────────────────────────────
def test_profile_nodes_poc_value_area_hvn_lvn():
    # 12 rows of 1.0 between 100 and 112: heavy at 102 (POC), second peak at 109,
    # a thin stretch around 106 between them
    prices = np.arange(100.5, 112.5, 1.0)
    vol = np.array([1, 5, 20, 6, 2, 0.1, 0.1, 0.1, 2, 14, 3, 1], float)
    bars = m1_from(prices, vol=vol, high=prices + 0.5, low=prices - 0.5)
    nodes = vl.profile_nodes(bars, n_rows=12)
    assert nodes["poc"] == [pytest.approx(102.5)]
    # CBOT growth: 70 % of 54.4 needs rows 100..109 (the far peak)
    assert nodes["val"] == [pytest.approx(100.5)] and nodes["vah"] == [pytest.approx(109.5)]
    assert nodes["hvn"] == [pytest.approx(109.5)]          # second peak, not the POC
    assert nodes["lvn"] == [pytest.approx(106.5)]          # thinnest smoothed row in value


def test_profile_nodes_flat_range_is_none():
    bars = m1_from(np.full(30, 100.0))
    assert vl.profile_nodes(bars) is None


def test_lvn_outside_value_area_is_ignored():
    prices = np.arange(100.5, 110.5, 1.0)
    vol = np.array([2, 0.01, 2, 2, 30, 30, 30, 2, 2, 2], float)   # thin row far below value
    bars = m1_from(prices, vol=vol, high=prices + 0.5, low=prices - 0.5)
    nodes = vl.profile_nodes(bars, n_rows=10)
    assert nodes["val"][0] > 101.5
    assert all(p > nodes["val"][0] for p in nodes["lvn"])


# ── windows ─────────────────────────────────────────────────────────────
def test_fixed_windows_length_and_quarter_step():
    bars = m1_from(np.full(3 * 1440, 100.0))
    w = vl.fixed_windows(bars, 96)                 # 1 day, every 6 h
    assert w[0].start == bars.index[0] and w[0].end == bars.index[0] + pd.Timedelta("1D")
    assert w[1].end - w[0].end == pd.Timedelta("6h")
    assert w[-1].end == bars.index[-1] + pd.Timedelta("1min")
    assert len(w) == 9


def test_period_windows_only_complete():
    bars = m1_from(np.full(2 * 1440 + 60, 100.0))  # Mon, Tue, 1 h of Wed
    days = vl.period_windows(bars, "day")
    assert [d.start.day for d in days] == [1, 2]
    assert vl.period_windows(bars, "week") == []


def test_session_windows_on_full_24h_frame_dst_aware():
    # 2024-03-29 (London still GMT) .. 2024-04-02 (BST from 31 March)
    bars = m1_from(np.full(5 * 1440, 100.0), start="2024-03-29")
    lon = vl.session_windows(bars, "london")
    starts = [w.start for w in lon]
    assert starts[0] == pd.Timestamp("2024-03-29 08:00", tz="UTC")
    assert pd.Timestamp("2024-04-01 07:00", tz="UTC") in starts       # 08:00 BST
    assert all(w.end - w.start == pd.Timedelta("8h30min") for w in lon)
    ny = vl.session_windows(bars, "ny")
    assert ny[0].start == pd.Timestamp("2024-03-29 13:30", tz="UTC")  # 09:30 EDT
    asia = vl.session_windows(bars, "asia")
    assert asia[0].start == pd.Timestamp("2024-03-29 00:00", tz="UTC")
    assert asia[0].end == pd.Timestamp("2024-03-29 06:00", tz="UTC")


def test_incomplete_session_is_dropped():
    bars = m1_from(np.full(600, 100.0), start="2024-01-01 08:30")    # London from 08:30 only
    assert vl.session_windows(bars, "london") == []


def test_leg_windows_known_at_confirmation_of_second_swing():
    m15 = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0},
                       index=pd.date_range("2024-01-01", periods=40, freq="15min", tz="UTC"))
    res = ms.StructureResult(bars=m15, zones=[], pools=[], swings={"ext": [
        ms.Swing(3, 8, 90.0, -1), ms.Swing(12, 17, 110.0, 1), ms.Swing(20, 25, 95.0, -1)]})
    legs = vl.leg_windows(m15, res)
    assert len(legs) == 2
    assert legs[0].start == m15.index[3] and legs[0].end == m15.index[12] + pd.Timedelta("15min")
    assert vl.known_at(legs[0]) == m15.index[17] + pd.Timedelta("15min")


def test_window_levels_skip_windows_with_missing_data():
    close = np.linspace(100, 110, 1440)
    close[:300] = np.nan                                 # 21 % missing
    bars = m1_from(close)
    w = [vl.Window("day", bars.index[0], bars.index[-1] + pd.Timedelta("1min"))]
    assert vl.window_levels(bars, w) == []


def test_naked_poc_only_if_next_day_never_trades_through():
    day1 = np.r_[np.full(1000, 100.0), np.full(440, 101.0)]           # POC ~100
    day2_away = np.full(1440, 105.0)                                  # never back
    day3 = np.full(1440, 104.0)
    bars = m1_from(np.r_[day1, day2_away, day3])
    naked = vl.naked_pocs(bars)
    assert len(naked) >= 1
    first = naked[0]
    assert first.kind == "naked:poc" and abs(first.price - 100.0) < 0.05
    assert first.known_at == pd.Timestamp("2024-01-03", tz="UTC")
    day2_back = np.r_[np.full(700, 105.0), np.full(40, 99.0), np.full(700, 105.0)]
    bars2 = m1_from(np.r_[day1, day2_back, day3])
    assert all(lv.known_at != pd.Timestamp("2024-01-03", tz="UTC") for lv in vl.naked_pocs(bars2))


# ── magnet ──────────────────────────────────────────────────────────────
def _path(close, atr=1.0, **kw):
    bars = m1_from(close, **kw)
    return vl.Path(bars, pd.Series(atr, index=bars.index)), bars


def test_magnet_hit_control_and_mirror():
    # flat at 100, the level 102 is reached 2 h after it is known; nothing else moves.
    n = 4 * 1440
    close = np.full(n, 100.0)
    t0 = 2 * 1440
    close[t0 + 120:t0 + 130] = 102.0
    p, bars = _path(close)
    lv = [Level("x:poc", 102.0, bars.index[t0])]
    ev = vl.magnet_events(p, lv, {"1h": "1h", "4h": "4h"}, n_ctl=5)
    e = ev.iloc[0]
    assert e["side"] == 1 and e["dist_atr"] == pytest.approx(2.0)
    assert e["hit_1h"] == 0 and e["hit_4h"] == 1
    assert e["mir_4h"] == 0                       # 98 never traded
    assert e["ctl_4h"] == 0                       # 2 ATR above any other time: never reached
    assert e["eff_4h"] == 1.0


def test_magnet_distance_band_and_inside_bar_skipped():
    close = np.full(4 * 1440, 100.0)
    p, bars = _path(close)
    t = bars.index[2 * 1440]
    far, near, inside = Level("a", 107.0, t), Level("b", 100.1, t), Level("c", 100.0, t)
    assert vl.magnet_events(p, [far, near, inside], {"1h": "1h"}).empty


def test_magnet_level_not_used_before_known():
    close = np.full(4 * 1440, 100.0)
    close[2 * 1440 - 30] = 102.0                    # touch BEFORE known_at
    p, bars = _path(close)
    ev = vl.magnet_events(p, [Level("x", 102.0, bars.index[2 * 1440])], {"4h": "4h"})
    assert ev.iloc[0]["hit_4h"] == 0


# ── hold ────────────────────────────────────────────────────────────────
def _touch_path(after):
    """Flat 100, rises to the level 101 at bar 100 (known at bar 10), then ``after``."""
    close = np.full(100, 100.0)
    close = np.r_[close, 101.0, after, np.full(400, after[-1])]
    return _path(close, atr=1.0)


def test_hold_bounce():
    p, bars = _touch_path(np.full(20, 100.4))        # back 0.6 ATR below the level
    ev = vl.hold_events(p, [Level("w:poc", 101.0, bars.index[10])])
    e = ev.iloc[0]
    assert e["ts"] == bars.index[100] and e["d"] == 1
    assert e["outcome"] == "bounce" and e["hold"] == 1.0
    assert e["fade_bp"] == pytest.approx(-np.log(100.4 / 101.0) * 1e4)


def test_hold_break():
    p, bars = _touch_path(np.full(20, 101.6))
    e = vl.hold_events(p, [Level("w:poc", 101.0, bars.index[10])]).iloc[0]
    assert e["outcome"] == "break" and e["hold"] == 0.0


def test_hold_unresolved_when_neither():
    p, bars = _touch_path(np.full(20, 101.1))
    e = vl.hold_events(p, [Level("w:poc", 101.0, bars.index[10])]).iloc[0]
    assert e["outcome"] == "unresolved" and np.isnan(e["hold"])


def test_hold_touch_bar_low_does_not_count_as_bounce():
    # the touch bar itself spans 100..101 (approach from below) - no bounce yet; then break
    close = np.r_[np.full(100, 100.0), 101.0, np.full(420, 101.6)]
    low = close.copy()
    low[100] = 100.0
    p, bars = _path(close, atr=1.0, low=low)
    e = vl.hold_events(p, [Level("w", 101.0, bars.index[10])]).iloc[0]
    assert e["outcome"] == "break"


def test_hold_flow_into_level_over_five_bars():
    close = np.r_[np.full(100, 100.0), 101.0, np.full(420, 100.4)]
    buy = np.ones(len(close))
    buy[96:101] = 4.0                                 # buying on the approach
    p, bars = _path(close, atr=1.0, buy=buy, sell=np.ones(len(close)))
    e = vl.hold_events(p, [Level("w", 101.0, bars.index[10])]).iloc[0]
    assert e["flow"] == pytest.approx((20 - 5) / 25)


def test_hold_from_above_mirrors():
    close = np.r_[np.full(100, 102.0), 101.0, np.full(420, 101.6)]
    p, bars = _path(close, atr=1.0)
    e = vl.hold_events(p, [Level("w", 101.0, bars.index[10])]).iloc[0]
    assert e["d"] == -1 and e["outcome"] == "bounce"


# ── confluence ──────────────────────────────────────────────────────────
def test_smc_confluence_only_active_and_near():
    m15 = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0},
                       index=pd.date_range("2024-01-01", periods=20, freq="15min", tz="UTC"))
    atr15 = pd.Series(1.0, index=m15.index)
    ref = [(100.05, 2, 20),          # active from bar 2 on, 0.05 away
           (200.0, 2, 5)]            # broken at bar 5
    lv = pd.DataFrame({"known_at": [m15.index[4], m15.index[1], m15.index[4], m15.index[10]],
                       "level": [100.0, 100.0, 100.5, 200.0]})
    got = vl.smc_confluence(lv, m15, ref, atr15)
    # bar 3 has closed at index[4]; bar 0 at index[1] -> ref not yet active; 0.5 too far; broken
    assert got.tolist() == [True, False, False, False]


# ── statistics ──────────────────────────────────────────────────────────
def test_nw_mean_t_lag0_is_plain_t_with_population_variance():
    x = np.array([1.0, 2.0, 0.5, 3.0, 1.5])
    t = vl.nw_mean_t(x, lags=0)
    assert t == pytest.approx(x.mean() / (x.std(ddof=0) / np.sqrt(len(x))))
    assert abs(vl.nw_mean_t(x, lags=2)) > 0


def test_nw_matches_hac_ols_on_a_constant():
    from trading_lab.hac import newey_west_ols
    rng = np.random.default_rng(1)
    y = rng.normal(0.3, 1.0, 200)
    x = rng.normal(0.0, 1.0, 200)
    x -= x.mean()
    # with a centred regressor orthogonal-ish to the constant, the intercept se ~ the mean se
    r = newey_west_ols(y, x, lags=5)
    assert vl.nw_mean_t(y, 5) == pytest.approx(r.t[0], rel=0.1)


def test_cell_verdict_rules():
    disc = {"mean": 0.08, "t": 3.5}
    assert vl.cell_verdict(disc, {"mean": 0.05, "t": 2.4}, {"BTC": 0.04, "SOL": 0.06})
    assert not vl.cell_verdict(disc, {"mean": 0.03, "t": 2.4}, {"BTC": 0.04, "SOL": 0.06})   # < half
    assert not vl.cell_verdict(disc, {"mean": 0.05, "t": 1.9}, {"BTC": 0.04, "SOL": 0.06})
    assert not vl.cell_verdict(disc, {"mean": 0.05, "t": 2.4}, {"BTC": -0.01, "SOL": 0.06})
    assert not vl.cell_verdict({"mean": 0.04, "t": 5.0}, {"mean": 0.04, "t": 3}, {"BTC": 0.04})
    assert vl.cell_verdict({"mean": -0.08, "t": -3.1}, {"mean": -0.06, "t": -2.1}, {"BTC": -0.1})
