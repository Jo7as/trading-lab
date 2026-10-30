"""Paket 1: the planted-edge rig (studies/planted_edge.py) on small synthetic frames, no network."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, ".")

from studies import planted_edge as P  # noqa: E402

SPREAD = 0.0001


def _m1(days: int = 40, seed: int = 0, sd: float = 0.00015) -> pd.DataFrame:
    t = pd.date_range("2023-01-02", periods=days * 1440, freq="1min", tz="UTC")
    t = t[t.dayofweek < 5]
    c = 1.1 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, sd, len(t))))
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.00001, "low": np.minimum(o, c) * 0.99999,
                       "close": c}, index=t)
    for k in ("open", "high", "low", "close"):
        df[f"ask_{k}"] = df[k] + SPREAD
    return df


def test_entries_are_one_per_weekday_at_14_utc() -> None:
    m1 = _m1()
    rows = P.entry_rows(m1.index)
    when = m1.index[rows]
    assert len(rows) > 20 and (when.hour == 14).all() and (when.minute == 0).all()
    assert when.normalize().is_unique


def test_zero_edge_changes_nothing() -> None:
    m1 = _m1()
    out = P.plant(m1, P.entry_rows(m1.index)[:5], np.ones(5, int), 0.0)
    pd.testing.assert_frame_equal(out, m1)


def test_drift_moves_mid_bid_ask_together_and_stays() -> None:
    m1 = _m1()
    k = int(P.entry_rows(m1.index)[3])
    out = P.plant(m1, np.array([k]), np.array([1]), 0.01, hold=240)
    for col in P.PRICE_COLS:
        assert out[col].iloc[:k].equals(m1[col].iloc[:k]), "nothing changes before the entry"
    assert out["open"].iloc[k] == m1["open"].iloc[k] and out["ask_open"].iloc[k] == m1["ask_open"].iloc[k]  # the fill
    factor = (out["close"] / m1["close"]).iloc[k + 240]
    assert factor == pytest.approx(np.exp(0.01))
    assert (out["ask_close"] / m1["ask_close"]).iloc[k + 500] == pytest.approx(factor)     # persists
    assert (out["close"] / m1["close"]).iloc[k + 500] == pytest.approx(factor)


def test_ohlc_stays_consistent() -> None:
    m1 = _m1()
    rows = P.entry_rows(m1.index)
    out = P.plant(m1, rows, np.random.default_rng(0).choice([-1, 1], len(rows)), 0.01)
    for side in ("", "ask_"):
        assert (out[f"{side}high"] >= out[[f"{side}open", f"{side}close"]].max(axis=1) - 1e-12).all()
        assert (out[f"{side}low"] <= out[[f"{side}open", f"{side}close"]].min(axis=1) + 1e-12).all()


def test_mu_is_the_wanted_annual_sharpe() -> None:
    assert P.mu_for(1.0, 0.01) * P.TRADES_PER_YEAR / (0.01 * np.sqrt(P.TRADES_PER_YEAR)) == pytest.approx(1.0)
    assert P.mu_for(0.0, 0.01) == 0.0


def test_rig_buys_the_ask_and_sells_the_bid() -> None:
    n = 400
    t = pd.date_range("2023-01-02 14:00", periods=n, freq="1min", tz="UTC").as_unit("ns").asi8
    d = {"t": t, "bid_open": np.full(n, 1.1), "ask_open": np.full(n, 1.1001), "rows": np.array([0]),
         "dirs": np.array([1]), "sigma": 0.01}
    tr = P.run_rig(d, P.RigConfig(hold_min=240, delay_min=5, slippage_bp=0.0))[0]
    assert tr["entry"] == pytest.approx(1.1001)
    assert tr["r"] == pytest.approx((1.1 - 1.1001) / (1.1001 * 0.01))              # pays the spread
    assert tr["exit_time"] - tr["entry_time"] == pd.Timedelta(minutes=240)


def test_planted_trade_earns_the_drift_on_mid() -> None:
    m1 = _m1(seed=2, sd=1e-7)                                                      # almost flat path
    rows = P.entry_rows(m1.index)
    dirs = np.random.default_rng(1).choice([-1, 1], len(rows))
    out = P.plant(m1, rows, dirs, 0.002)
    mid = ((out["close"] + out["ask_close"]) / 2).to_numpy()
    got = dirs * np.log(mid[rows + 240] / mid[rows])
    assert got.mean() == pytest.approx(0.002, rel=0.02)


@pytest.mark.parametrize("sharpe", [0.0, 8.0])
def test_the_whole_path_runs_and_tells_edge_from_no_edge(sharpe: float) -> None:
    m1 = _m1(days=400, seed=1)
    r = P.judge(m1, "EURUSD", sharpe, seed=1)
    assert r["n_trials"] >= 2
    if sharpe > 0:
        assert r["S1"] and r["placebo_edge"] > 0 and r["gross_mean"] > 0.1
    else:
        assert abs(r["gross_mean"]) < 0.3
