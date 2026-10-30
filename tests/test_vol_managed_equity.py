"""S-010 (docs/specs/vol_managed_equity.yaml): one test group per rule."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trading_lab.strategies import tsmom_multi_asset as M
from trading_lab.strategies import vol_managed_equity as V
from trading_lab.strategies.vol_managed_equity import VolManagedConfig

CFG = VolManagedConfig()


def _rates() -> pd.DataFrame:
    return pd.DataFrame({"USD": 0.0}, index=pd.date_range("2000-01-31", "2030-12-31", freq="ME"))


def _frame(r: np.ndarray, idx: pd.DatetimeIndex) -> pd.DataFrame:
    c = 100 * np.cumprod(1 + r)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=idx)


def _prep(n_days: int = 900, sd: float = 0.01, seed: int = 0, cost=None) -> M.Prepared:
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    r = np.random.default_rng(seed).normal(0.0003, sd, n_days)
    return M.prepare({"SP500": _frame(r, idx), "GER40": _frame(r, idx)}, _rates(), cost or {})


def test_monthly_variance_is_the_annualised_sum_of_squares() -> None:
    idx = pd.bdate_range("2020-01-01", "2020-02-28")
    r = pd.Series(0.01, index=idx)
    v = V.monthly_variance(r)
    n = (idx.month == 1).sum()
    assert v.iloc[0] == pytest.approx(n * 1e-4 * 12)


def test_exposure_rules_and_cap_use_known_data_only() -> None:
    var = pd.Series(np.r_[np.full(30, 0.04), 0.01],
                    index=pd.period_range("2015-01", periods=31, freq="M"))
    w = V.exposure(var, CFG)
    assert w.iloc[:MIN].isna().all()
    assert w.iloc[25] == pytest.approx(0.08 * 0.2 / 0.04)              # c / var
    mean = var.iloc[:31].mean()
    assert w.iloc[30] == pytest.approx(min(3.0, 0.08 * np.sqrt(mean) / 0.01))
    assert V.exposure(var, replace(CFG, gewicht="vola")).iloc[30] == pytest.approx(0.08 / 0.1)
    assert V.exposure(var, replace(CFG, gewicht="konstant")).iloc[25] == pytest.approx(0.08 / 0.2)
    calm = var.copy()
    calm.iloc[30] = 1e-6
    assert V.exposure(calm, CFG).iloc[30] == 3.0                       # leverage cap
    # changing a LATER month does not change an earlier decision
    later = var.copy()
    later.iloc[30] = 9.0
    assert V.exposure(later, CFG).iloc[29] == pytest.approx(w.iloc[29])


MIN = V.MIN_MONTHS - 1


def test_trades_at_the_first_trading_day_of_a_month_after_warmup() -> None:
    p = _prep()
    res = V.run_backtest(p, CFG, {})
    tr = res["trades"]
    assert tr and all(t["direction"] == 1 for t in tr)
    starts = sorted({t["entry_time"] for t in tr})
    months = p.dates.to_period("M")
    first_of_month = set(p.dates[np.r_[False, months[1:] != months[:-1]]])
    assert set(starts) <= first_of_month
    assert starts[0] >= p.dates[0] + pd.DateOffset(months=V.MIN_MONTHS)
    assert res["daily"].loc[:starts[0], "gross"].abs().sum() == 0.0


def test_positions_split_over_indices_and_costs_charged() -> None:
    free = V.run_backtest(_prep(), CFG, {})
    paid = V.run_backtest(_prep(cost={"SP500": 2.0, "GER40": 2.0}), CFG, {})
    assert paid["daily"]["cost"].sum() > 0 and free["daily"]["cost"].sum() == 0
    assert paid["daily"]["net"].sum() == pytest.approx(free["daily"]["net"].sum() - paid["daily"]["cost"].sum())
    assert sum(t["r"] for t in paid["trades"]) * paid["unit"] == pytest.approx(paid["daily"]["net"].sum())


def test_financing_is_charged_per_calendar_day() -> None:
    d = V.run_backtest(_prep(), CFG, {"SP500": -0.02, "GER40": -0.02})["daily"]
    held = d["fin"] != 0
    assert (d["fin"][held] < 0).all()
    mon = d.index.dayofweek == 0
    assert (d["fin"][held & mon].mean() / d["fin"][held & ~mon].mean()) == pytest.approx(3.0, rel=0.3)


def test_buffer_skips_small_changes() -> None:
    p = _prep(sd=0.01)
    many = V.run_backtest(p, replace(CFG, gewicht="konstant", buffer=0.0), {})["trades"]
    few = V.run_backtest(p, replace(CFG, gewicht="konstant", buffer=0.10), {})["trades"]
    assert len(few) < len(many)


def test_placebo_flips_the_direction() -> None:
    p = _prep()
    months = pd.period_range("2015-01", "2019-12", freq="M")
    flips = pd.Series(-1.0, index=months)
    tr = V.run_backtest(p, CFG, {}, flips=flips)["trades"]
    assert tr and all(t["direction"] == -1 for t in tr)


def test_no_look_ahead() -> None:
    p = _prep(n_days=1000, seed=3)
    idx = pd.bdate_range("2015-01-01", periods=1000)
    r = np.random.default_rng(3).normal(0.0003, 0.01, 1000)
    frames = {"SP500": _frame(r, idx), "GER40": _frame(r, idx)}
    cut = idx[800]
    part = M.prepare(frames, _rates(), {}, end=str(cut.date()))
    a = V.run_backtest(p, CFG, {})["daily"]["net"]
    b = V.run_backtest(part, CFG, {})["daily"]["net"]
    assert np.allclose(a.loc[b.index], b)
