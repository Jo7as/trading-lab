"""H9 (follow the flow after absorption) - every pre-registered rule on hand-made numbers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_lab import costs
from trading_lab import level_flow as lf


def ev_frame(rows):
    df = pd.DataFrame(rows, columns=["d", "flow", "beyond_atr", "ret_4h"])
    df["ts"] = pd.date_range("2023-01-01", periods=len(df), freq="1h", tz="UTC")
    df["market"] = "BTC"
    return df


def test_absorption_definition_and_follow_direction():
    ev = ev_frame([
        (+1, 0.20, -0.1, 10.0),   # flow up into a level above, close back below -> absorption, trade +1
        (-1, -0.30, 0.0, -5.0),   # flow down into a level below, close exactly at it -> absorption, trade -1
        (+1, 0.15, -0.5, 0.0),    # flow exactly at the threshold -> absorption
        (+1, 0.14, -0.5, 0.0),    # flow just below threshold -> not
        (+1, 0.40, 0.01, 0.0),    # close beyond the level -> not (cascade / none)
        (-1, 0.40, -0.2, 0.0),    # flow AGAINST the approach -> not
    ])
    out = lf.follow_absorption(ev)
    assert list(out.index) == [0, 1, 2]
    assert list(out["trade"]) == [1, -1, 1]
    assert list(out["trade"] * out["ret_4h"]) == [10.0, 5.0, 0.0]   # follow = +d, opposite of H8's fade


def test_fade_and_follow_are_mirror_images():
    ev = ev_frame([(+1, 0.3, -0.1, 7.0), (-1, -0.3, -0.1, 3.0)])
    follow = lf.follow_absorption(ev)
    fade = lf.classify(ev)
    assert (fade["label"] == "absorption").all()
    np.testing.assert_allclose(follow["trade"] * follow["ret_4h"], -(fade["trade"] * fade["ret_4h"]))


def test_net_uses_bybit_taker_round_trip():
    c = costs.crypto_round_trip_bp("BTC", "bybit_perp")
    assert 9.0 < c < 14.0                                   # taker both legs + half spreads
    assert c == pytest.approx(costs.crypto_round_trip_bp("BTCUSDT", "bybit_perp"))


def test_day_clustered_t_uses_per_day_means():
    ts = pd.to_datetime(["2023-01-01 01:00", "2023-01-01 02:00", "2023-01-02 01:00",
                         "2023-01-03 01:00"], utc=True)
    x = pd.Series([1.0, 3.0, 4.0, 6.0])                     # day means 2, 4, 6
    assert lf.day_clustered_t(x, ts) == pytest.approx(4.0 / (2.0 / np.sqrt(3)))
    assert np.isnan(lf.day_clustered_t(x[:2], ts[:2]))       # fewer than 3 days


def test_cluster_diff_t_difference_and_clustering():
    rng = np.random.default_rng(1)
    days = pd.date_range("2023-01-01", periods=200, freq="1D", tz="UTC")
    ts = days.repeat(5)
    real = 2.0 + rng.normal(0, 1, len(ts))
    ctl = rng.normal(0, 1, len(ts))
    diff, t = lf.cluster_diff_t(real, ts, ctl, ts)
    assert diff == pytest.approx(real.mean() - ctl.mean())
    assert t > 10
    # copying every observation 10x inside its day adds no information: clustered t barely moves
    a, b = real - 1.8, ctl
    _, t1 = lf.cluster_diff_t(a, ts, b, ts)
    _, t10 = lf.cluster_diff_t(np.repeat(a, 10), ts.repeat(10), np.repeat(b, 10), ts.repeat(10))
    assert t10 == pytest.approx(t1, rel=0.01)


def test_random_direction_p():
    r = pd.Series(np.r_[np.full(50, 10.0), np.full(50, -10.0)])
    perfect = np.sign(r)                                    # always right -> no random draw beats it
    assert lf.random_direction_p(r, perfect, n_draws=500) == 0.0
    assert lf.random_direction_p(r, -perfect, n_draws=500) == 1.0
    p = lf.random_direction_p(r, np.ones(100), n_draws=2000)   # mean 0 -> about half
    assert 0.4 < p < 0.6


def test_non_overlapping_per_market():
    ts = pd.to_datetime(["2023-01-01 00:00", "2023-01-01 02:00", "2023-01-01 04:00",
                         "2023-01-01 01:00", "2023-01-01 09:00"], utc=True)
    ev = pd.DataFrame({"ts": ts, "market": ["BTC", "BTC", "BTC", "SOL", "SOL"]})
    kept = lf.non_overlapping(ev, "4h")
    assert sorted(kept.index) == [0, 2, 3, 4]               # BTC 02:00 is inside the first hold


def test_verdict_rules():
    ok = dict(gross=15.0, t_gross=2.5, net=3.0, diff=6.0, t_diff=2.0, p_random=0.01,
              gross_by_market={"BTC": 10.0, "SOL": 20.0})
    assert lf.h9_verdict(**ok)[0].startswith("confirmed")
    assert lf.h9_verdict(**{**ok, "t_gross": 1.99})[0] == "rejected"           # C1
    assert lf.h9_verdict(**{**ok, "net": -0.1})[0] == "rejected"               # C2
    v, c = lf.h9_verdict(**{**ok, "t_diff": 1.0})                               # C3
    assert v == "rejected: flow continuation, not a level effect" and not c["C3"]
    assert lf.h9_verdict(**{**ok, "p_random": 0.05})[0] == "rejected"          # C4
    assert lf.h9_verdict(**{**ok, "gross_by_market": {"BTC": -1.0, "SOL": 30.0}})[0] == "rejected"  # C5
