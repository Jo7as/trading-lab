"""S-008 (docs/specs/turn_of_month.yaml): one test group per rule."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_lab.strategies import tsmom_multi_asset as M
from trading_lab.strategies import turn_of_month as TOM


def _rates(usd: float = 0.0) -> pd.DataFrame:
    idx = pd.date_range("2000-01-31", "2030-12-31", freq="ME")
    return pd.DataFrame({c: (usd if c == "USD" else 0.0) for c in ("USD", "EUR", "JPY", "GBP")}, index=idx)


def _frame(close: np.ndarray, idx: pd.DatetimeIndex) -> pd.DataFrame:
    c = np.asarray(close, float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=idx)


# ── rule 1: the window from the index's own trading days ───────────────────

def test_window_is_day_minus_2_to_day_plus_3_with_holidays() -> None:
    idx = pd.bdate_range("2024-01-02", "2024-03-29")
    idx = idx[idx != pd.Timestamp("2024-01-30")]            # a holiday near the month end
    idx = idx[idx != pd.Timestamp("2024-02-02")]            # and one at the start
    p = M.prepare({"SP500": _frame(np.full(len(idx), 100.0), idx)}, _rates(), {})
    w = TOM.tom_windows(p.dates, p.has_bar[:, 0])
    e, x = w[0]
    assert p.dates[e] == pd.Timestamp("2024-01-29")         # 2nd-last trading day (30th is off)
    assert p.dates[x] == pd.Timestamp("2024-02-06")         # Feb 1, 5, 6 (2nd is off) -> 3rd = Feb 6
    assert p.dates[w[1][0]] == pd.Timestamp("2024-02-28")


def test_month_without_data_is_skipped() -> None:
    idx = pd.bdate_range("2024-01-02", "2024-05-31")
    idx = idx[(idx.month != 3)]
    p = M.prepare({"SP500": _frame(np.full(len(idx), 100.0), idx)}, _rates(), {})
    ends = [p.dates[e].month for e, _ in TOM.tom_windows(p.dates, p.has_bar[:, 0])]
    assert ends == [1, 4]                                   # Feb->Mar and Mar->Apr have no March data


# ── rules 2-3: P&L, sizing, costs ──────────────────────────────────────────

def _path(idx: pd.DatetimeIndex, jumps: dict[str, float]) -> np.ndarray:
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.002, len(idx))
    for d, j in jumps.items():
        r[idx.get_loc(pd.Timestamp(d))] = j
    return 100 * np.exp(np.cumsum(r))


def test_only_the_window_days_count() -> None:
    idx = pd.bdate_range("2023-10-02", "2024-02-29")
    base = _path(idx, {})
    jump = _path(idx, {"2024-01-31": 0.05})                  # day -1 of January: inside
    early = _path(idx, {"2024-01-30": 0.05})                 # day -2 itself: before the entry close
    late = _path(idx, {"2024-02-06": 0.05})                  # day +4: after the exit close
    run = lambda c: TOM.run_backtest(M.prepare({"SP500": _frame(c, idx)}, _rates(), {}))  # noqa: E731
    jan = lambda res: [t for t in res["trades"] if t["entry_time"].month == 1][0]["r_gross"]  # noqa: E731
    b = jan(run(base))
    assert jan(run(jump)) > b + 5                            # a +5 % day at ~0.2 % sigma
    # day -2 and day +4 add nothing (day -2 still raises the EWMA vol -> a smaller position)
    assert run(early)["daily"].loc["2024-01-30", "gross"] == 0.0
    assert run(late)["daily"].loc["2024-02-06", "gross"] == 0.0
    assert jan(run(late)) == pytest.approx(b)


def test_indices_share_one_budget() -> None:
    idx = pd.bdate_range("2023-10-02", "2024-02-29")
    c = _path(idx, {})
    one = TOM.run_backtest(M.prepare({"SP500": _frame(c, idx)}, _rates(), {}))
    two = TOM.run_backtest(M.prepare({"SP500": _frame(c, idx), "US100": _frame(c, idx)}, _rates(), {}))
    assert two["daily"]["gross"].sum() == pytest.approx(one["daily"]["gross"].sum())


def test_budget_does_not_depend_on_who_enters_the_same_day() -> None:
    # placebo windows scatter the entries - the size must still be 1/(active indices)
    idx = pd.bdate_range("2023-10-02", "2024-02-29")
    c = _path(idx, {})
    p = M.prepare({"SP500": _frame(c, idx), "US100": _frame(c, idx)}, _rates(), {})
    real = TOM.run_backtest(p)
    w = TOM.tom_windows(p.dates, p.has_bar[:, 0])
    alone = TOM.run_backtest(p, windows={"SP500": w, "US100": []})
    assert alone["daily"]["gross"].sum() == pytest.approx(real["daily"]["gross"].sum() / 2)


def test_costs_and_financing() -> None:
    idx = pd.bdate_range("2023-10-02", "2024-02-29")
    c = _path(idx, {})
    free = TOM.run_backtest(M.prepare({"SP500": _frame(c, idx)}, _rates(0.0), {}))
    paid = TOM.run_backtest(M.prepare({"SP500": _frame(c, idx)}, _rates(5.0), {"SP500": 1.5}))
    assert (paid["daily"]["fin"] <= 0).all() and paid["daily"]["fin"].sum() < 0
    assert paid["daily"]["cost"].sum() > 0 and free["daily"]["cost"].sum() == 0
    n = len(paid["trades"])
    assert (paid["daily"]["cost"] > 0).sum() == 2 * n        # entry and exit


def test_placebo_windows_are_mid_month_and_four_days() -> None:
    idx = pd.bdate_range("2023-01-02", "2024-12-31")
    p = M.prepare({"SP500": _frame(np.full(len(idx), 100.0), idx)}, _rates(), {})
    w = TOM.random_windows(p.dates, p.has_bar[:, 0], np.random.default_rng(1))
    assert len(w) == 24
    for e, x in w:
        assert x - e == 4 and 6 <= p.dates[e].day <= 24


def test_no_look_ahead() -> None:
    idx = pd.bdate_range("2022-01-03", "2024-06-28")
    rng = np.random.default_rng(4)
    frames = {a: _frame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), idx) for a in ("SP500", "GER40")}
    full = TOM.run_backtest(M.prepare(frames, _rates(2.0), {"SP500": 1.0}))
    part = TOM.run_backtest(M.prepare(frames, _rates(2.0), {"SP500": 1.0}, end="2024-01-15"))
    key = lambda t: (t["asset"], t["entry_time"], round(t["r"], 9))  # noqa: E731
    done = {key(t) for t in full["trades"] if t["exit_time"] < pd.Timestamp("2024-01-01")}
    assert done and done <= {key(t) for t in part["trades"]}
