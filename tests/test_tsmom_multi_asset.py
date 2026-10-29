"""S-007 (docs/specs/tsmom_multi_asset.yaml): one test group per rule + engine properties."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trading_lab.long_daily import clean_bad_ticks, fix_fx_dates
from trading_lab.strategies import tsmom_multi_asset as M
from trading_lab.strategies.tsmom_multi_asset import TsmomConfig

CFG = TsmomConfig()


def _frame(close: np.ndarray, start: str = "2010-01-01", spread: float = 0.0) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(close))
    c = np.asarray(close, float)
    return pd.DataFrame({"open": c, "high": c * (1 + spread), "low": c * (1 - spread), "close": c}, index=idx)


def _rates(rate_usd: float = 0.0, **other: float) -> pd.DataFrame:
    idx = pd.date_range("2000-01-31", "2030-12-31", freq="ME")
    cols = {c: other.get(c, 0.0) for c in ("EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD")}
    return pd.DataFrame({"USD": rate_usd, **cols}, index=idx)


# ── rule 1: signal ─────────────────────────────────────────────────────────

def test_12m_signal_is_the_sign_of_the_past_year() -> None:
    c = np.r_[np.linspace(100, 120, 300), np.linspace(120, 90, 300)][:, None]
    s = M.signals(c, "12m")
    assert np.isnan(s[251, 0]) and s[299, 0] == 1.0
    assert s[599, 0] == -1.0                                      # 90 < price 252 days earlier


def test_blend_signal_averages_three_horizons() -> None:
    c = np.r_[np.linspace(100, 130, 270), np.linspace(130, 128, 21)][:, None]   # long up, last month down
    s = M.signals(c, "blend_1_3_12")
    assert s[-1, 0] == pytest.approx(1 / 3)                       # 12m +, 3m +, 1m -


# ── rule 2: volatility ─────────────────────────────────────────────────────

def test_ewma_vol_annualises_the_daily_move() -> None:
    r = np.tile([0.01, -0.01], 300)[:, None]
    v = M.ewma_vol(r, 60)
    assert np.isnan(v[10, 0])
    assert v[-1, 0] == pytest.approx(0.01 * np.sqrt(252), rel=1e-3)


# ── rule 3: budgets ────────────────────────────────────────────────────────

def test_budgets_equal_per_class_crypto_half() -> None:
    groups = ["fx_usd"] * 7 + ["aktienindizes", "krypto"]
    w = M.budgets(np.ones(9, bool), groups)
    assert w[:7].sum() == pytest.approx(w[7] ) == pytest.approx(0.4)   # 1 / 2.5 each
    assert w[8] == pytest.approx(0.2)
    w2 = M.budgets(np.r_[np.ones(7, bool), True, False], groups)     # crypto missing
    assert w2[:7].sum() == pytest.approx(0.5) and w2[8] == 0.0


# ── rule 4: target positions ───────────────────────────────────────────────

def test_targets_hit_the_portfolio_vol_and_respect_caps() -> None:
    groups = ["aktienindizes", "metalle"]
    vol = np.array([0.20, 0.10])
    cov = np.diag(vol ** 2)
    pos = M.target_positions(np.array([1.0, -1.0]), vol, cov, groups, CFG)
    assert np.sign(pos).tolist() == [1.0, -1.0]
    assert np.sqrt(pos @ cov @ pos) == pytest.approx(0.08)
    assert abs(pos[0]) * 0.20 == pytest.approx(abs(pos[1]) * 0.10)       # equal risk per class
    tiny = M.target_positions(np.array([1.0, 1.0]), np.array([1e-5, 0.1]), np.diag([1e-10, 0.01]), groups, CFG)
    assert tiny[0] == pytest.approx(20.0)                               # leverage cap 1:20


# ── rule 5: rebalance threshold ────────────────────────────────────────────

def test_needs_trade() -> None:
    assert M.needs_trade(0.0, 0.5, 0.25) and M.needs_trade(0.5, 0.0, 0.25)
    assert M.needs_trade(0.5, -0.5, 0.25)
    assert not M.needs_trade(1.0, 1.2, 0.25) and M.needs_trade(1.0, 1.3, 0.25)
    assert not M.needs_trade(0.0, 0.0, 0.25)


# ── financing ──────────────────────────────────────────────────────────────

def test_carry_formulas() -> None:
    r = _rates(5.0, EUR=3.0)
    lo, sh = M.carry_rates("EURUSD", r, CFG)
    assert lo.iloc[0] == pytest.approx(-0.03) and sh.iloc[0] == pytest.approx(0.01)
    lo, sh = M.carry_rates("SP500", r, CFG)
    assert lo.iloc[0] == pytest.approx(-0.075) and sh.iloc[0] == pytest.approx(0.025)
    lo, _ = M.carry_rates("GER40", r, replace(CFG, markup_factor=2.0))
    assert lo.iloc[0] == pytest.approx(-(0.03 + 0.05))


# ── engine ─────────────────────────────────────────────────────────────────

def _up(n: int = 520, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(0.001 + rng.normal(0, 0.005, n)))


def _prep(frames: dict, rates=None, cost=None, **kw) -> M.Prepared:
    return M.prepare(frames, rates if rates is not None else _rates(), cost or {}, **kw)


def test_rebalances_only_on_wednesday_with_tuesday_information() -> None:
    p = _prep({"SP500": _frame(_up())})
    res = M.run_backtest(p, CFG)
    tr = res["trades"]
    assert tr and all(t["entry_time"].dayofweek == 2 for t in tr)
    first = tr[0]["entry_time"]
    assert first >= p.dates[252]                                  # not before 12 months of data
    assert res["daily"].loc[:first, "gross"].abs().sum() == 0.0    # no P&L before the first fill


def test_costs_are_charged_per_change_of_position() -> None:
    frames = {"SP500": _frame(_up())}
    free = M.run_backtest(_prep(frames), CFG)["daily"]
    paid = M.run_backtest(_prep(frames, cost={"SP500": 2.0}), CFG)["daily"]
    assert (paid["cost"] > 0).sum() >= 1 and free["cost"].sum() == 0.0
    assert paid["net"].sum() == pytest.approx(free["net"].sum() - paid["cost"].sum())


def test_financing_counts_calendar_days() -> None:
    p = _prep({"SP500": _frame(_up())}, rates=_rates(4.0))
    res = M.run_backtest(p, CFG)
    d = res["daily"]
    held = d["fin"] != 0
    mon = d.index.dayofweek == 0
    assert (d["fin"][held & mon].mean() / d["fin"][held & ~mon].mean()) == pytest.approx(3.0, rel=0.05)
    assert (d["fin"][held] < 0).all()                              # long pays rate + markup


def test_emergency_stop_and_no_reentry_until_the_signal_changes() -> None:
    c = _up(600)
    c[400:] = c[399] * 0.85                                       # a crash and then flat
    p = _prep({"SP500": _frame(c, spread=0.0)})
    res = M.run_backtest(p, replace(CFG, stop="stop_3atr"))
    reasons = [t["exit_reason"] for t in res["trades"]]
    assert "stop" in reasons
    k = reasons.index("stop")
    stop_day = res["trades"][k]["exit_time"]
    later = [t for t in res["trades"][k + 1:] if t["direction"] == 1]
    sig = M.signals(p.close, "12m")[:, 0]
    if later:                                                     # a new long only after a down signal
        between = sig[(p.dates > stop_day) & (p.dates < later[0]["entry_time"])]
        assert (between < 0).any()
    no_stop = M.run_backtest(p, CFG)
    assert "stop" not in [t["exit_reason"] for t in no_stop["trades"]]


def test_placebo_keeps_the_timing_and_randomises_the_direction() -> None:
    sig = np.r_[np.ones(10), -np.ones(10), np.ones(5)][:, None]
    f = M.placebo_flips(sig, seed=3)
    seg = [f[0, 0], f[10, 0], f[20, 0]]
    assert (f[:10, 0] == seg[0]).all() and (f[10:20, 0] == seg[1]).all() and (f[20:, 0] == seg[2]).all()
    assert set(np.unique(M.placebo_flips(np.ones((50, 30)), seed=1))) == {-1.0, 1.0}


def test_no_look_ahead() -> None:
    rng = np.random.default_rng(8)
    walk = lambda: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 700)))  # noqa: E731 - signals flip
    frames = {"SP500": _frame(walk()), "XAUUSD": _frame(walk()), "EURUSD": _frame(walk() / 100)}
    full = M.run_backtest(_prep(frames, rates=_rates(2.0, EUR=1.0), cost={"SP500": 1.0}), CFG)
    cut = pd.bdate_range("2010-01-01", periods=700)[600]
    part = M.run_backtest(_prep(frames, rates=_rates(2.0, EUR=1.0), cost={"SP500": 1.0}, end=str(cut.date())), CFG)
    a, b = full["daily"]["net"], part["daily"]["net"]
    assert np.allclose(a.loc[b.index[:-1]], b.iloc[:-1])
    done = [t for t in full["trades"] if t["exit_time"] < b.index[-1] and t["exit_reason"] != "data_end"]
    key = lambda t: (t["asset"], t["entry_time"], round(t["r"], 9))  # noqa: E731
    assert done and {key(t) for t in done} <= {key(t) for t in part["trades"]}


# ── data: bad ticks ────────────────────────────────────────────────────────

def test_bad_ticks_are_replaced_real_shocks_stay() -> None:
    rng = np.random.default_rng(5)
    c = 1.2 * np.exp(np.cumsum(rng.normal(0, 0.004, 300)))
    spike = c.copy()
    spike[200] = c[199] * 1.17                                    # +17 % for one day, back next day
    shock = c.copy()
    shock[200:] = shock[200:] * 0.83                              # a real -17 % shock that stays
    fixed, bad = clean_bad_ticks(_frame(spike))
    assert len(bad) == 1 and fixed["close"].iloc[200] == pytest.approx(c[199])
    kept, bad2 = clean_bad_ticks(_frame(shock))
    assert bad2 == [] and kept["close"].iloc[200] == pytest.approx(shock[200])


def test_yfinance_fx_close_is_re_dated_to_the_end_of_the_day() -> None:
    # yfinance "=X": close(D) == open(D) == price at 00:00 UTC of D
    idx = pd.bdate_range("2024-03-04", periods=4)
    raw = pd.DataFrame({"open": [1.0, 1.1, 1.2, 1.3], "high": [1.15, 1.25, 1.35, 1.4],
                        "low": [0.95, 1.05, 1.15, 1.25], "close": [1.0, 1.1, 1.2, 1.3]}, index=idx)
    fixed = fix_fx_dates(raw)
    assert fixed["close"].tolist() == [1.1, 1.2, 1.3]                  # end of D = start of D+1
    assert fixed.index[-1] == idx[-2]
