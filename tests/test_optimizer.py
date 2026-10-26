"""Optimiser: finds a known edge on train, counts trials, never touches the holdout unasked."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from trading_lab import screening
from trading_lab.optimizer import Spec, normalise, optimise, prop_profile, replay_placebo


def _m1(days: int = 520, drift: float = 0.02, seed: int = 3) -> pd.DataFrame:
    """One hour of 1-minute bars per business day; price drifts up inside the hour."""
    rng = np.random.default_rng(seed)
    idx, px, p = [], [], 100.0
    for d in pd.bdate_range("2021-01-04", periods=days, tz="UTC"):
        for m in range(60):
            idx.append(d + pd.Timedelta(hours=10, minutes=m))
            p += drift + rng.normal(0, 0.1)
            px.append(p)
    c = np.array(px)
    df = pd.DataFrame({"open": c, "high": c + 0.05, "low": c - 0.05, "close": c},
                      index=pd.DatetimeIndex(idx))
    for col in ("open", "high", "low", "close"):
        df[f"ask_{col}"] = df[col] + 0.02
    return df


@dataclass(frozen=True)
class ToyCfg:
    side: int = -1
    hold: int = 10


def _toy_run(cfg: ToyCfg, m1: pd.DataFrame) -> list[dict]:
    out = []
    day = m1.index.normalize()
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    mid = ((m1["close"] + m1["ask_close"]) / 2).to_numpy()
    for i in starts:
        j = min(i + cfg.hold, len(m1) - 1)
        if day[j] != day[i]:
            continue
        entry = m1["ask_close"].iat[i] if cfg.side > 0 else m1["close"].iat[i]
        exit_ = m1["close"].iat[j] if cfg.side > 0 else m1["ask_close"].iat[j]
        risk = 1.0
        out.append({"entry_time": m1.index[i], "exit_time": m1.index[j], "direction": cfg.side,
                    "entry": entry, "stop": entry - cfg.side * 50.0, "target": None,
                    "r": cfg.side * (exit_ - entry) / 50.0, "risk": risk})
    return out


def _spec() -> Spec:
    return Spec(name="toy", asset="SYN", base=ToyCfg(),
                grid={"side": (-1, 1), "hold": (10, 30, 50)},
                prepare=lambda m1: m1, run=_toy_run)


HOLD = pd.Timestamp("2022-11-01", tz="UTC")


def test_optimiser_finds_the_known_edge_and_counts_trials() -> None:
    res = optimise(_spec(), _m1(), holdout_start=HOLD, budget=20, verbose=False)
    best = max(res.rows, key=lambda r: r["objective_lcb"])
    assert best["params"] == {"hold": 50, "side": 1}
    assert res.n_trials == 6                      # the whole grid, each counted once
    assert all(r["n_trials"] == 6 for r in res.rows)


def test_holdout_is_not_touched_without_release() -> None:
    res = optimise(_spec(), _m1(), holdout_start=HOLD, budget=20, verbose=False)
    assert all(r["oos_pass"] is None for r in res.rows)
    assert all("oos_n" not in r for r in res.rows)


def test_holdout_runs_with_release_for_s4_survivors_only() -> None:
    res = optimise(_spec(), _m1(), holdout_start=HOLD, budget=20, verbose=False,
                   holdout_release="test")
    f = screening.run_funnel(pd.DataFrame(res.rows), up_to="S4")
    passed = set(f.stages[-1].kept["name"])
    for r in res.rows:
        assert ("oos_n" in r) == (r["name"] in passed)


def test_normalise_adds_the_spread_back_for_gross() -> None:
    m1 = _m1(days=5)
    tr = normalise(_toy_run(ToyCfg(side=1, hold=10), m1), m1, None)
    assert (tr["r_gross"] > tr["r"]).all()
    assert (tr["r_gross"] - tr["r"]).iloc[0] == pytest.approx(0.02 / 50.0)


def test_normalise_reads_position_logs_in_account_percent() -> None:
    m1 = _m1(days=3)
    t = {"status": "closed", "entry_time": m1.index[0].timestamp(), "exit_time": m1.index[5].timestamp(),
         "direction": "long", "entry_price": 100.0, "stop_loss": 99.0, "tp1": 102.0, "pnl_pct": 1.0}
    tr = normalise([t], m1, risk_pct=0.5)
    assert tr["r"].iloc[0] == pytest.approx(2.0)
    assert tr["direction"].iloc[0] == 1 and tr["target_dist"].iloc[0] == pytest.approx(2.0)


def test_replay_placebo_long_beats_random_on_an_up_drift() -> None:
    m1 = _m1(days=200)
    tr = normalise(_toy_run(ToyCfg(side=1, hold=50), m1), m1, None)
    own, rnd = replay_placebo(tr, m1)
    assert own > rnd
    # and the short side is the mirror image
    tr_s = normalise(_toy_run(ToyCfg(side=-1, hold=50), m1), m1, None)
    own_s, rnd_s = replay_placebo(tr_s, m1)
    assert own_s < rnd_s and rnd_s == pytest.approx(rnd, abs=1e-9)


def test_prop_profile_scales_risk_to_the_target() -> None:
    times = pd.date_range("2021-01-04", periods=24, freq="MS", tz="UTC")
    tr = pd.DataFrame({"entry_time": times, "exit_time": times, "r": [1.0, -0.5] * 12})
    p = prop_profile(tr, months=24, target_month_pct=1.5)
    assert p["r_per_month"] == pytest.approx(0.25)
    assert p["risk_for_target_pct"] == pytest.approx(6.0)
    assert p["dd_at_target_pct"] == pytest.approx(3.0)


def test_normalise_keeps_break_even_trades_and_recovers_the_initial_risk() -> None:
    """After a partial TP the logged stop equals the entry - the trade must stay."""
    m1 = _m1(days=3)
    t = {"status": "closed", "entry_time": m1.index[0].timestamp(), "exit_time": m1.index[9].timestamp(),
         "direction": "long", "entry_price": 100.0, "stop_loss": 100.0, "sl_moved_to_be": True,
         "tp1": 101.0, "tp2": 103.0, "size": 0.5, "pnl_pct": 0.6}
    tr = normalise([t], m1, risk_pct=1.0)
    assert len(tr) == 1
    assert tr["risk"].iloc[0] == pytest.approx(2.0)       # 100 * 1 % / 0.5
    assert tr["r"].iloc[0] == pytest.approx(0.6)


def test_effective_trials_collapse_for_identical_series() -> None:
    from trading_lab.optimizer import effective_trials
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    same = pd.DataFrame({i: x + rng.normal(0, 1e-6, 500) for i in range(50)})
    indep = pd.DataFrame(rng.normal(size=(500, 50)))
    assert effective_trials(same) < 2
    assert effective_trials(indep) > 45


def test_settings_with_identical_trades_count_once() -> None:
    """A parameter without effect must not inflate the trial count (S4 bar)."""
    spec = _spec()
    spec.grid = {"side": (-1, 1), "hold": (10, 30, 50)}
    spec.run = lambda cfg, m1: _toy_run(ToyCfg(side=cfg.side, hold=50), m1)   # hold has no effect
    res = optimise(spec, _m1(), holdout_start=HOLD, budget=20, verbose=False)
    assert res.n_trials == 2


def test_time_units_do_not_matter() -> None:
    """Cache frames have a ms index, trade logs us/ns - lookups must still line up."""
    m1 = _m1(days=200)
    m1.index = m1.index.as_unit("ms")
    raw = _toy_run(ToyCfg(side=1, hold=50), m1)
    for t in raw:                                   # ORB-style: ISO strings (-> us/ns)
        t["entry_time"] = pd.Timestamp(t["entry_time"]).as_unit("ns").isoformat()
        t["exit_time"] = pd.Timestamp(t["exit_time"]).as_unit("ns").isoformat()
    tr = normalise(raw, m1, None)
    assert (tr["r_gross"] - tr["r"]).iloc[0] == pytest.approx(0.02 / 50.0)
    own, rnd = replay_placebo(tr, m1)
    assert np.isfinite(own) and own > rnd


def test_entry_offset_moves_the_replay_start_to_the_fill() -> None:
    """A bar strategy filling at the bar CLOSE but stamping the bar OPEN must not let
    the placebo replay start before the signal existed."""
    m1 = _m1(days=3)
    raw = _toy_run(ToyCfg(side=1, hold=30), m1)
    a = normalise(raw, m1, None)
    b = normalise(raw, m1, None, entry_offset_min=15)
    assert (b["entry_time"] - a["entry_time"]).iloc[0] == pd.Timedelta(minutes=15)
    assert (b["exit_time"] >= b["entry_time"]).all()


def test_bid_only_engines_get_the_spread_charged() -> None:
    """An engine that trades the bid only logs gross R; normalise must charge the spread."""
    m1 = _m1(days=3)
    raw = _toy_run(ToyCfg(side=1, hold=10), m1)
    for t in raw:
        t["r"] = 0.5                                  # pretend: bid-only gross result
    tr = normalise(raw, m1, None, charges_spread=False)
    assert (tr["r_gross"] == 0.5).all()
    assert tr["r"].iloc[0] == pytest.approx(0.5 - 0.02 / 50.0)
