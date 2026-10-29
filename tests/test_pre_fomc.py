"""S-009 (docs/specs/pre_fomc.yaml): one test group per rule, on full 24 h 1m frames."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_lab.strategies import pre_fomc as P

SPREAD = 0.2


def _m1(start: str, days: int, base: float = 5000.0, seed: int = 0, jump: dict | None = None) -> pd.DataFrame:
    t = pd.date_range(start, periods=days * 1440, freq="1min", tz="UTC")
    t = t[t.dayofweek < 5]
    rng = np.random.default_rng(seed)
    c = base * np.exp(np.cumsum(rng.normal(0, 0.0002, len(t))))
    for when, f in (jump or {}).items():
        c[t >= pd.Timestamp(when, tz="UTC")] *= f
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c}, index=t)
    for k in ("open", "high", "low", "close"):
        df[f"ask_{k}"] = df[k] + SPREAD
    return df


def _mk(df: pd.DataFrame, asset: str = "SP500", rate: float = 0.0) -> P.Market:
    r = pd.Series(rate, index=pd.date_range("2020-01-01", "2030-01-01", freq="D"))
    return P.market(df, asset, r)


# ── rule 1: the window ─────────────────────────────────────────────────────

def test_window_is_14_00_ny_the_day_before_to_13_50_dst_safe() -> None:
    e, x = P.window(pd.Timestamp("2024-03-20"))             # EDT
    assert e == pd.Timestamp("2024-03-19 18:00", tz="UTC") and x == pd.Timestamp("2024-03-20 17:50", tz="UTC")
    e, x = P.window(pd.Timestamp("2024-01-31"))             # EST
    assert e == pd.Timestamp("2024-01-30 19:00", tz="UTC") and x == pd.Timestamp("2024-01-31 18:50", tz="UTC")


def test_no_window_across_a_weekend() -> None:
    assert P.window(pd.Timestamp("2024-03-18")) is None     # a Monday statement -> Friday entry -> skip


def test_dates_file_has_the_2024_meetings() -> None:
    d = [x for x in P.statement_dates() if x.year == 2024]
    assert len(d) == 8 and pd.Timestamp("2024-09-18") in d


# ── rules 2-4: fills, sizing, financing ────────────────────────────────────

def test_buys_the_ask_sells_the_bid_and_counts_only_the_window() -> None:
    df = _m1("2023-12-01", 70, jump={"2024-01-30 20:00": 1.01, "2024-01-31 19:30": 1.05})
    tr = P.trade(_mk(df), pd.Timestamp("2024-01-31"), 1, P.PreFomcConfig(slippage_bp=0.0))
    assert tr["entry_time"] == pd.Timestamp("2024-01-30 19:00", tz="UTC")
    assert tr["exit_time"] == pd.Timestamp("2024-01-31 18:50", tz="UTC")
    assert tr["move_bp"] == pytest.approx(100, abs=15)      # the +1 % inside, not the +5 % after 13:50
    k_e = df.index.get_loc(tr["entry_time"])
    k_x = df.index.get_loc(tr["exit_time"])
    gross = tr["pos"] * (df["open"].iloc[k_x] / df["ask_open"].iloc[k_e] - 1) / P.UNIT
    assert tr["r"] == pytest.approx(gross)


def test_markets_share_one_budget() -> None:
    df = _m1("2023-12-01", 70)
    one = P.trade(_mk(df), pd.Timestamp("2024-01-31"), 1)
    four = P.trade(_mk(df), pd.Timestamp("2024-01-31"), 4)
    assert four["pos"] == pytest.approx(one["pos"] / 4)


def test_one_night_of_financing() -> None:
    df = _m1("2023-12-01", 70)
    free = P.trade(_mk(df, rate=0.0), pd.Timestamp("2024-01-31"), 1)
    paid = P.trade(_mk(df, rate=-0.075), pd.Timestamp("2024-01-31"), 1)
    assert (free["r"] - paid["r"]) * P.UNIT == pytest.approx(free["pos"] * 0.075 / 365)


def test_missing_minute_waits_then_gives_up() -> None:
    df = _m1("2023-12-01", 70)
    gap = df[~((df.index >= "2024-01-30 19:00") & (df.index < "2024-01-30 19:10"))]
    tr = P.trade(_mk(gap), pd.Timestamp("2024-01-31"), 1)
    assert tr["entry_time"] == pd.Timestamp("2024-01-30 19:10", tz="UTC")
    hole = df[~((df.index >= "2024-01-30 19:00") & (df.index < "2024-01-30 20:00"))]
    assert P.trade(_mk(hole), pd.Timestamp("2024-01-31"), 1) is None


def test_size_uses_volatility_known_before_the_entry() -> None:
    df = _m1("2023-12-01", 70)
    wild = df.copy()
    after = wild.index >= "2024-01-30 21:00"                    # chaos only after the entry
    wild.loc[after, ["open", "close", "ask_open", "ask_close"]] *= 1.2
    a = P.trade(_mk(df), pd.Timestamp("2024-01-31"), 1)
    b = P.trade(_mk(wild), pd.Timestamp("2024-01-31"), 1)
    assert a["pos"] == pytest.approx(b["pos"])


def test_placebo_days_avoid_fomc_and_keep_the_weekday() -> None:
    cand = pd.bdate_range("2024-01-01", "2024-12-31")
    real = [x for x in P.statement_dates() if x.year == 2024]
    pl = P.placebo_days(cand, real, np.random.default_rng(1))
    assert len(pl) == len(real)
    assert sorted(d.dayofweek for d in pl) == sorted(d.dayofweek for d in real)
    assert not set(pl) & (set(real) | {d - pd.offsets.BDay(1) for d in real})
