"""S-005 (docs/specs/weekly_level_break.yaml): one test group per rule, on full 1m frames."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trading_lab.strategies import weekly_level_break as W
from trading_lab.strategies.weekly_level_break import WeeklyBreakConfig

CFG = WeeklyBreakConfig(slippage_bp=0.0)


def _frame(path: dict[str, float], start="2024-07-01", end="2024-07-13", base=1.10, seed=0) -> pd.DataFrame:
    """Weekday minutes; small noise around ``base``; ``path`` = {UTC time: level from then on}."""
    t = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    t = t[t.dayofweek < 5]
    rng = np.random.default_rng(seed)
    lvl = np.full(len(t), base)
    for when, v in sorted(path.items()):
        lvl[t >= pd.Timestamp(when, tz="UTC")] = v
    c = lvl + rng.normal(0, 0.00005, len(t))
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.00002, "low": np.minimum(o, c) - 0.00002,
                       "close": c}, index=t)
    for k in ("open", "high", "low", "close"):
        df[f"ask_{k}"] = df[k]
    return df


def _ramp(df: pd.DataFrame, a: str, b: str, v0: float, v1: float) -> pd.DataFrame:
    m = (df.index >= pd.Timestamp(a, tz="UTC")) & (df.index < pd.Timestamp(b, tz="UTC"))
    x = np.linspace(v0, v1, m.sum())
    for k in ("open", "high", "low", "close", "ask_open", "ask_high", "ask_low", "ask_close"):
        df.loc[m, k] = x
    after = df.index >= pd.Timestamp(b, tz="UTC")
    for k in ("open", "high", "low", "close", "ask_open", "ask_high", "ask_low", "ask_close"):
        df.loc[after, k] = df.loc[after, k] - 1.10 + v1
    return df


def _week2_break_up() -> pd.DataFrame:
    df = _frame({})
    return _ramp(df, "2024-07-09 10:00", "2024-07-09 16:00", 1.10, 1.115)   # Tue: break up and run


# ── rules 1-2: level and event ─────────────────────────────────────────────

def test_first_break_of_last_weeks_high_is_the_event() -> None:
    p = W.prepare(_week2_break_up(), "EURUSD")
    ev = [e for e in p.tfs["1h"]["events"] if e.side == 1 and e.level == "pw"]
    starts = pd.to_datetime(p.tfs["1h"]["start_ns"], utc=True)
    assert len(ev) == 1 and starts[ev[0].bar] == pd.Timestamp("2024-07-09 10:00", tz="UTC")
    assert ev[0].kind == "through"


def test_art_filters_the_events() -> None:
    from trading_lab.liquidity import Event
    evs = [Event("pw", 1, "through", 3, 0, 1.1), Event("pw", -1, "reclaim", 5, 0, 1.0)]
    assert len(W.signals(evs, "durch")) == 1 and len(W.signals(evs, "alle")) == 2


# ── rules 3-4: execution ───────────────────────────────────────────────────

def test_long_after_the_bar_close_hits_the_3r_target() -> None:
    p = W.prepare(_week2_break_up(), "EURUSD")
    tr = W.run_backtest(p, CFG)["trades"]
    up = [t for t in tr if t["direction"] == 1]
    assert len(up) == 1
    t = up[0]
    assert t["entry_time"] == pd.Timestamp("2024-07-09 11:00", tz="UTC")     # close of the 10:00 bar
    assert t["exit_reason"] == "ziel" and t["r"] == pytest.approx(3.0, abs=0.05)
    i = t["signal_bar"]
    assert t["entry"] - t["stop"] == pytest.approx(p.tfs["1h"]["atr"][i])        # 1 ATR stop


def test_time_exit_and_five_r_variant() -> None:
    p = W.prepare(_week2_break_up(), "EURUSD")
    t5 = [t for t in W.run_backtest(p, replace(CFG, ziel_r=50.0))["trades"] if t["direction"] == 1][0]
    assert t5["exit_reason"] == "zeit"
    assert t5["exit_time"] == pd.Timestamp("2024-07-10 11:00", tz="UTC")       # 24 bars after the signal bar


def test_no_entry_after_the_friday_cut() -> None:
    df = _ramp(_frame({}), "2024-07-12 21:00", "2024-07-12 23:00", 1.10, 1.115)   # break Fri 21:00
    p = W.prepare(df, "EURUSD")
    assert [t for t in W.run_backtest(p, CFG)["trades"] if t["direction"] == 1] == []


def test_measured_financing_is_charged() -> None:
    df = _ramp(_frame({}), "2024-07-09 10:00", "2024-07-09 12:00", 1.10, 1.1015)  # small move, held overnight
    p = W.prepare(df, "EURUSD")
    cfg = replace(CFG, ziel_r=50.0)
    free = W.run_backtest(p, cfg)["trades"][0]
    paid = W.run_backtest(p, replace(cfg, fin_pa=(-0.05, -0.05)))["trades"][0]
    assert paid["r"] < free["r"] and paid["swap_r"] > 0


def test_placebo_uses_the_same_hours() -> None:
    p = W.prepare(_week2_break_up(), "EURUSD")
    real = W.run_backtest(p, CFG)["trades"]
    ov = W.placebo_override(p, CFG, real, np.random.default_rng(0))
    hours = pd.to_datetime(p.tfs["1h"]["start_ns"], utc=True).hour
    assert [hours[i] for i, _ in ov] == [hours[t["signal_bar"]] for t in real]


def test_no_look_ahead() -> None:
    df = _week2_break_up()
    full = W.run_backtest(W.prepare(df, "EURUSD"), CFG)["trades"]
    part = W.run_backtest(W.prepare(df[df.index < "2024-07-11"], "EURUSD"), CFG)["trades"]
    key = lambda t: (t["entry_time"], t["direction"], round(t["r"], 9))  # noqa: E731
    done = {key(t) for t in full if t["exit_time"] < pd.Timestamp("2024-07-10 12:00", tz="UTC")}
    assert done and done <= {key(t) for t in part}
