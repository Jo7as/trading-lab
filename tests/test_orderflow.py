"""Order-flow measures on tiny hand-made flow bars (values checked by hand)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_lab import orderflow as of


def make_bars(n: int, start: str = "2024-01-01", freq: str = "10s", **cols) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="ts")
    base = {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "vwap": 100.0,
            "buy_qty": 0.0, "sell_qty": 0.0, "buy_notional": 0.0, "sell_notional": 0.0,
            "n_trades": 0}
    df = pd.DataFrame({k: np.full(n, v, dtype=float) for k, v in base.items()}, index=idx)
    for k, v in cols.items():
        df[k] = np.asarray(v, dtype=float)
    return df


def test_imbalance_and_delta():
    bars = make_bars(3, buy_notional=[30, 0, 0], sell_notional=[10, 0, 5])
    assert list(of.imbalance(bars).iloc[[0, 2]]) == [0.5, -1.0]
    assert np.isnan(of.imbalance(bars).iloc[1])             # no volume -> undefined, not 0
    assert list(of.delta(bars)) == [20, 0, -5]


def test_cvd_resets_at_anchor():
    # 4 hourly bars crossing midnight: reset at 00:00
    bars = make_bars(4, start="2024-01-01 22:00", freq="1h",
                     buy_notional=[5, 5, 1, 0], sell_notional=[0, 0, 0, 2])
    assert list(of.cvd(bars)) == [5, 10, 1, -1]
    assert list(of.cvd(bars, anchor=None)) == [5, 10, 11, 9]


def test_combine_venues_weights_and_reference_price():
    a = make_bars(2, buy_notional=[10, 0], sell_notional=[0, 4], close=[100, 101])
    b = make_bars(2, buy_notional=[2, 2], sell_notional=[6, 0], close=[100.5, 101.5])
    c = of.combine_venues({"binance": a, "bybit": b})
    assert list(c["buy_notional"]) == [12, 2] and list(c["sell_notional"]) == [6, 4]
    assert list(c["close"]) == [100, 101]                    # price from the first venue
    w = of.combine_venues({"binance": a, "bybit": b}, weights={"binance": 1.0, "bybit": 0.5})
    assert list(w["buy_notional"]) == [11, 1]
    with pytest.raises(ValueError):
        of.combine_venues({"a": a, "b": make_bars(2, start="2024-01-02")})


def test_resample_ohlc_sums_and_vwap():
    bars = make_bars(6, open=[1, 2, 3, 4, 5, 6], high=[2, 9, 4, 5, 6, 7], low=[1, 1, 0.5, 4, 5, 6],
                     close=[2, 3, 4, 5, 6, 7], buy_qty=[1, 0, 0, 1, 1, 0], sell_qty=[0, 1, 0, 0, 0, 2],
                     buy_notional=[2, 0, 0, 5, 6, 0], sell_notional=[0, 3, 0, 0, 0, 14])
    m = of.resample(bars, "30s")
    assert len(m) == 2
    first = m.iloc[0]
    assert (first["open"], first["high"], first["low"], first["close"]) == (1, 9, 0.5, 4)
    assert first["buy_qty"] == 1 and first["sell_qty"] == 1
    assert first["vwap"] == pytest.approx(5 / 2)
    assert m.iloc[1]["vwap"] == pytest.approx(25 / 4)


def test_clock_openings_quarter_hour_window_and_timing():
    n = 360  # one hour of 10 s bars
    buy = np.zeros(n)
    sell = np.zeros(n)
    buy[0], sell[0] = 30, 10          # 00:00:00-00:00:10 -> +0.5
    buy[1] = 1000                      # 00:00:10 is outside the 10 s opening window
    sell[90] = 7                       # 00:15:00 -> -1
    bars = make_bars(n, buy_notional=buy, sell_notional=sell)
    op = of.clock_openings(bars)
    assert list(op.index.strftime("%H:%M:%S")) == ["00:00:00", "00:15:00", "00:30:00", "00:45:00"]
    assert op["imbalance"].iloc[0] == 0.5
    assert op["imbalance"].iloc[1] == -1.0
    assert np.isnan(op["imbalance"].iloc[2])
    assert op["known_at"].iloc[0] == pd.Timestamp("2024-01-01 00:00:10", tz="UTC")
    wide = of.clock_openings(bars, window="20s")
    assert wide["delta"].iloc[0] == 1020
    with pytest.raises(ValueError):
        of.clock_openings(bars, window="5s")


def test_clock_openings_on_full_day_grid_stay_in_phase():
    """96 marks on a full 24 h frame, each exactly on a quarter hour (AGENTS.md index trap)."""
    bars = make_bars(8640, buy_notional=np.ones(8640))
    op = of.clock_openings(bars)
    assert len(op) == 96
    assert (op.index.minute % 15 == 0).all() and (op.index.second == 0).all()


def test_corwin_schultz_known_value_and_no_lookahead():
    # two bars with high/low 101/99 -> beta = 2*ln(101/99)^2, gamma = ln(101/99)^2
    bars = make_bars(3, high=[101, 101, 150], low=[99, 99, 50])
    cs = of.corwin_schultz(bars)          # close 100 sits inside every range -> no gap shift
    l = np.log(101 / 99)
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(4 * l * l) - np.sqrt(2 * l * l)) / k - np.sqrt(l * l / k)
    assert cs.iloc[1] == pytest.approx(max(0.0, 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))))
    assert np.isnan(cs.iloc[0])
    # changing a later bar must not change an earlier value
    later = bars.copy()
    later.loc[later.index[2], ["high", "low"]] = [1000, 1]
    assert of.corwin_schultz(later).iloc[1] == cs.iloc[1]


def test_corwin_schultz_positive_for_bid_ask_bounce():
    # price alternates bid 99.9 / ask 100.1 inside each bar -> spread ~ 0.2 %
    bars = make_bars(50, high=np.full(50, 100.1), low=np.full(50, 99.9))
    est = of.corwin_schultz(bars).iloc[1:]
    assert (est > 0).all()


def test_kyle_lambda_recovers_slope():
    rng = np.random.default_rng(0)
    flow = rng.normal(0, 1e6, 400)                      # signed notional per bar
    ret = 2e-4 * flow / 1e6                            # lambda = 2e-4 per 1M USD
    close = 100 * np.exp(np.cumsum(ret))
    bars = make_bars(400, close=close, buy_notional=np.clip(flow, 0, None),
                     sell_notional=np.clip(-flow, 0, None))
    lam = of.kyle_lambda(bars, window=100)
    assert lam.iloc[-1] == pytest.approx(2e-4, rel=1e-6)


def test_vpin_buckets_and_publication_time():
    # each bar 10 notional, bucket 20 -> buckets fill exactly at bars 1, 3, 5
    buy = [10, 10, 10, 0, 5, 5]
    sell = [0, 0, 0, 10, 5, 5]
    bars = make_bars(6, buy_notional=buy, sell_notional=sell)
    v = of.vpin(bars, bucket_notional=20, n_buckets=1)
    # bucket 1 = bars 0-1: all buy -> 1; bucket 2 = bars 2-3: 10 vs 10 -> 0; bucket 3 -> 0
    assert np.isnan(v.iloc[0])
    assert v.iloc[1] == 1.0 and v.iloc[2] == 1.0       # published on the filling bar
    assert v.iloc[3] == 0.0 and v.iloc[5] == 0.0
    v2 = of.vpin(bars, bucket_notional=20, n_buckets=2)
    assert np.isnan(v2.iloc[1]) and v2.iloc[3] == 0.5 and v2.iloc[5] == 0.0


def test_vpin_splits_an_overfilling_bar_like_the_paper():
    # bar 0: 30 buy -> fills bucket 1 (20 buy) and puts 10 buy into bucket 2;
    # bar 1: 10 sell completes bucket 2 -> |10 - 10| = 0
    bars = make_bars(2, buy_notional=[30, 0], sell_notional=[0, 10])
    v = of.vpin(bars, bucket_notional=20, n_buckets=1)
    assert v.iloc[0] == 1.0 and v.iloc[1] == 0.0
    # one bar filling two buckets publishes the later one
    big = make_bars(1, buy_notional=[40])
    assert of.vpin(big, bucket_notional=20, n_buckets=1).iloc[0] == 1.0


def test_clock_openings_placebo_offset():
    bars = make_bars(360, buy_notional=np.ones(360))
    op = of.clock_openings(bars, offset="2min")
    assert list(op.index.strftime("%H:%M:%S")) == ["00:02:00", "00:17:00", "00:32:00", "00:47:00"]


def test_forward_log_returns_uses_last_price_and_ffill():
    close = np.full(10, np.nan)
    close[0], close[2], close[5] = 100.0, 110.0, 121.0      # empty bars in between
    bars = make_bars(10, close=close)
    at = bars.index[[0, 2]]
    fr = of.forward_log_returns(bars, at, {"20s": "20s", "40s": "40s", "5min": "5min"})
    assert fr["20s"].iloc[0] == pytest.approx(np.log(1.1))
    assert fr["40s"].iloc[0] == pytest.approx(np.log(1.1))   # bar 4 empty -> last price 110
    assert fr["20s"].iloc[1] == pytest.approx(0.0)           # bar 4 carries 110
    assert np.isnan(fr["5min"].iloc[0])                       # beyond the data


def test_volume_profile_and_value_area():
    # vwaps 100, 101, 101, 102, 105 with notional 1, 5, 3, 2, 1 -> bins of 1.0
    bars = make_bars(5, vwap=[100.2, 101.5, 101.1, 102.9, 105.0],
                     buy_notional=[1, 4, 0, 2, 1], sell_notional=[0, 1, 3, 0, 0])
    prof = of.volume_profile(bars, 1.0)
    assert prof.loc[101.0, "total"] == 8 and prof.loc[101.0, "buy"] == 4 and prof.loc[101.0, "sell"] == 4
    lv = of.profile_levels(prof, value_area=0.7)          # total 12 -> need 8.4
    # POC 101 (8); neighbours up 102 (2) vs down 100 (1) -> take 102 -> 10 >= 8.4
    assert lv == {"poc": 101.0, "val": 101.0, "vah": 102.0}


def test_profile_levels_counts_empty_bins_as_zero():
    bars = make_bars(3, vwap=[100.0, 103.0, 104.0], buy_notional=[5, 1, 4])
    lv = of.profile_levels(of.volume_profile(bars, 1.0), value_area=0.9)  # need 9 of 10
    # POC 100 (5); up neighbour 101 is empty (0) - must still walk through it to 103/104
    assert lv["poc"] == 100.0 and lv["val"] == 100.0 and lv["vah"] == 104.0


def test_session_profile_levels_are_from_the_previous_day():
    n = 2 * 8640
    vwap = np.where(np.arange(n) < 8640, 100.0, 200.0)
    bars = make_bars(n, vwap=vwap, buy_notional=np.ones(n))
    lv = of.session_profile_levels(bars, 1.0)
    assert lv["poc"].iloc[:8640].isna().all()
    assert (lv["poc"].iloc[8640:] == 100.0).all()          # day 2 sees day 1, never itself


def test_flow_vs_move_absorption_shape():
    n = 200
    close = np.full(n, 100.0)
    close[150:] = 100.0 * np.exp(np.cumsum(np.full(50, 1e-4)))   # small drift late
    rng = np.random.default_rng(4)
    close[:150] = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-3, 150)))
    buy = np.ones(n)
    buy[190:] = 50.0                                          # heavy buying at the end
    bars = make_bars(n, close=close, buy_notional=buy, sell_notional=np.ones(n))
    fm = of.flow_vs_move(bars, window=10, lookback=10)
    last = fm.iloc[-1]
    assert last["flow"] == pytest.approx(49 * 10 / (51 * 10))  # strong buy flow
    assert abs(last["move_z"]) < 1.0                          # but the price hardly moved


def test_corwin_schultz_recovers_known_spread_in_simulation():
    """Corwin & Schultz's own design: a continuous true price, every bar's high is
    touched at the ask and its low at the bid. The averaged estimate must recover the
    true spread (S = 20 bp) without being swamped by volatility."""
    rng = np.random.default_rng(11)
    n_bars, steps, spread = 4000, 400, 0.0020
    sigma_step = 0.0010 / np.sqrt(steps)              # 10 bp volatility per bar
    path = np.cumsum(rng.normal(0, sigma_step, n_bars * steps)).reshape(n_bars, steps)
    true_hi, true_lo = np.exp(path.max(axis=1)), np.exp(path.min(axis=1))
    true_close = np.exp(path[:, -1])
    bars = make_bars(n_bars, high=true_hi * (1 + spread / 2), low=true_lo * (1 - spread / 2),
                     close=true_close)
    est = of.corwin_schultz(bars).iloc[1:]
    assert est.mean() == pytest.approx(spread, rel=0.15)


def test_corwin_schultz_gap_adjustment_section_2_1():
    """A jump between bars (bar 2 entirely above bar 1's close) inflates the two-bar range;
    the paper's adjustment shifts bar 2 back to the previous close."""
    rng = np.random.default_rng(12)
    n_bars, steps, spread = 3000, 300, 0.0020
    path = np.cumsum(rng.normal(0, 0.0010 / np.sqrt(steps), n_bars * steps)).reshape(n_bars, steps)
    jumps = np.cumsum(np.where(rng.random(n_bars) < 0.3, 0.004, 0.0))   # 40 bp gaps between bars
    path = path + jumps[:, None]
    hi = np.exp(path.max(axis=1)) * (1 + spread / 2)
    lo = np.exp(path.min(axis=1)) * (1 - spread / 2)
    bars = make_bars(n_bars, high=hi, low=lo, close=np.exp(path[:, -1]))
    raw = of.corwin_schultz(bars, adjust_gaps=False).iloc[1:].mean()
    adj = of.corwin_schultz(bars).iloc[1:].mean()
    assert abs(adj - spread) < abs(raw - spread)


def test_norm_cdf_matches_known_values():
    x = np.array([-1.96, -1.0, 0.0, 0.5, 1.0, 1.96, 3.0])
    want = np.array([0.0249979, 0.1586553, 0.5, 0.6914625, 0.8413447, 0.9750021, 0.9986501])
    assert of._norm_cdf(x) == pytest.approx(want, abs=1e-6)


def test_bvc_split_follows_price_change_and_is_causal():
    close = np.array([100, 100.1, 99.9, 100.1, 99.9, 100.1, 99.9, 100.1, 102.0, 100.0])
    bars = make_bars(10, close=close, buy_notional=np.full(10, 6.0), sell_notional=np.full(10, 4.0))
    b, s = of.bvc_split(bars, sigma_window=6)
    assert (b + s).iloc[4:].to_numpy() == pytest.approx(np.full(6, 10.0))   # sigma needs 3 past changes
    assert b.iloc[8] > 9.9                                   # big up move -> ~all buy
    assert s.iloc[9] > 9.9                                   # big down move -> ~all sell
    assert np.isnan(b.iloc[1])                               # no sigma yet
    moved = bars.copy()
    moved.loc[moved.index[9], "close"] = 200.0               # a later bar changes nothing earlier
    assert of.bvc_split(moved, sigma_window=6)[0].iloc[8] == b.iloc[8]


def test_vpin_bvc_option_and_bad_name():
    rng = np.random.default_rng(5)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, 3000)))
    bars = make_bars(3000, close=close, buy_notional=np.full(3000, 5.0), sell_notional=np.full(3000, 5.0))
    v_aggr = of.vpin(bars, bucket_notional=100, n_buckets=20)
    v_bvc = of.vpin(bars, bucket_notional=100, n_buckets=20, classification="bvc")
    assert v_aggr.dropna().eq(0).all()                       # taker flow perfectly balanced
    assert v_bvc.iloc[1000:].between(0.05, 0.6).all()        # after sigma warm-up: price-driven imbalance
    with pytest.raises(ValueError):
        of.vpin(bars, 100, classification="tick")
