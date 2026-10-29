"""S-009: pre-FOMC drift - equity indices long from 14:00 New York the day before a scheduled
FOMC statement to 13:50 on statement day (Lucca/Moench 2015).

Spec (agreed with the user 2026-10-29): ``docs/specs/pre_fomc.yaml``; tests:
``tests/test_pre_fomc.py``. Dates: ``trading_lab/fomc_meetings.json`` (federalreserve.gov).

 1. ``window``: entry 14:00 NY on the last weekday before the statement day, exit 13:50 NY on
    the statement day (10 min before the release - outside the prop news window); a window
    across a weekend is skipped (no weekend holding on this spec).
 2. Fills on 1m bid/ask: buy the ask open of the first quoted minute at/after the entry time,
    sell the bid open at/after the exit time (each within ``max_wait_min``), +- slippage.
 3. Sizing: the indices share ONE budget - (0.5 % / number of indices) / daily sigma (EWMA 60 of
    17:00-NY daily closes up to the day before entry) -> 1 R = 0.5 % of the account.
 4. Financing: one night, long pays rate + markup (S-007 formula), per calendar day.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NY = "America/New_York"
INDICES = ("SP500", "US100", "US30", "GER40")
UNIT = 0.005
DATES_FILE = Path(__file__).resolve().parents[1] / "fomc_meetings.json"


@dataclass(frozen=True)
class PreFomcConfig:
    entry_hm: tuple[int, int] = (14, 0)
    exit_hm: tuple[int, int] = (13, 50)
    slippage_bp: float = 0.5
    markup_cfd: float = 0.025
    max_wait_min: int = 30
    vol_com: int = 60


@dataclass
class Market:
    asset: str
    t_ns: np.ndarray                 # 1m bar starts (UTC ns)
    bid_open: np.ndarray
    ask_open: np.ndarray
    daily_sigma: pd.Series           # index = NY trading day label, sigma known at its close
    rate: pd.Series                  # long financing (fraction p.a., negative = pays), daily


def statement_dates(path: Path = DATES_FILE) -> list[pd.Timestamp]:
    return [pd.Timestamp(m["date"]) for m in json.loads(path.read_text())["meetings"]]


def window(day: pd.Timestamp, cfg: PreFomcConfig = PreFomcConfig()) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """(entry, exit) in UTC for a statement on ``day`` (NY date); None across a weekend."""
    prev = day - pd.offsets.BDay(1)
    if (day - prev).days > 1:
        return None
    e = (prev + pd.Timedelta(hours=cfg.entry_hm[0], minutes=cfg.entry_hm[1])).tz_localize(NY).tz_convert("UTC")
    x = (day + pd.Timedelta(hours=cfg.exit_hm[0], minutes=cfg.exit_hm[1])).tz_localize(NY).tz_convert("UTC")
    return e, x


def daily_sigma(m1: pd.DataFrame, com: int = 60) -> pd.Series:
    mid = (m1["close"] + m1["ask_close"]) / 2
    ny = mid.index.tz_convert(NY)
    day = (ny - pd.Timedelta(hours=17)).normalize().tz_localize(None) + pd.Timedelta(days=1)
    c = mid.groupby(day).last()
    c = c[c.index.dayofweek < 5]
    return c.pct_change().ewm(com=com, min_periods=20).std()


def market(m1: pd.DataFrame, asset: str, long_rate: pd.Series) -> Market:
    return Market(asset, m1.index.as_unit("ns").asi8, m1["open"].to_numpy(float), m1["ask_open"].to_numpy(float),
                  daily_sigma(m1), long_rate)


def _fill(mk: Market, t: pd.Timestamp, side: np.ndarray, wait: int) -> tuple[int, float]:
    k = int(np.searchsorted(mk.t_ns, t.value, "left"))
    if k >= len(mk.t_ns) or mk.t_ns[k] - t.value > wait * 60_000_000_000 or not np.isfinite(side[k]):
        return -1, np.nan
    return k, float(side[k])


def trade(mk: Market, day: pd.Timestamp, n_markets: int, cfg: PreFomcConfig = PreFomcConfig()) -> dict | None:
    w = window(day, cfg)
    if w is None:
        return None
    e, x = w
    ke, ask = _fill(mk, e, mk.ask_open, cfg.max_wait_min)
    kx, bid = _fill(mk, x, mk.bid_open, cfg.max_wait_min)
    if ke < 0 or kx < 0 or kx <= ke:
        return None
    sig = mk.daily_sigma[mk.daily_sigma.index < e.tz_convert(NY).tz_localize(None).normalize()].dropna()
    if sig.empty or not sig.iloc[-1] > 0:
        return None
    pos = UNIT / n_markets / float(sig.iloc[-1])
    slip = cfg.slippage_bp * 1e-4
    entry, exit_ = ask * (1 + slip), bid * (1 - slip)
    nights = (x.tz_convert(NY).normalize() - e.tz_convert(NY).normalize()).days
    rate = mk.rate[mk.rate.index <= e.tz_localize(None)]
    fin = pos * (float(rate.iloc[-1]) if len(rate) else 0.0) * nights / 365.0
    mid_e = (mk.ask_open[ke] + mk.bid_open[ke]) / 2
    mid_x = (mk.ask_open[kx] + mk.bid_open[kx]) / 2
    return {"asset": mk.asset, "statement": day, "entry_time": pd.Timestamp(mk.t_ns[ke], tz="UTC"),
            "exit_time": pd.Timestamp(mk.t_ns[kx], tz="UTC"), "direction": 1, "pos": pos,
            "r_gross": pos * (mid_x / mid_e - 1) / UNIT, "r": (pos * (exit_ / entry - 1) + fin) / UNIT,
            "move_bp": (mid_x / mid_e - 1) * 1e4, "exit_reason": "zeit"}


def run_backtest(markets: list[Market], days: list[pd.Timestamp], cfg: PreFomcConfig = PreFomcConfig()) -> dict:
    trades = [t for d in days for mk in markets if (t := trade(mk, d, len(markets), cfg)) is not None]
    return {"trades": trades, "unit": UNIT}


def placebo_days(candidates: pd.DatetimeIndex, real: list[pd.Timestamp], rng: np.random.Generator) -> list[pd.Timestamp]:
    """Random non-FOMC days with the weekday distribution of the real statement days (one
    per real day; neither the statement day nor the day before)."""
    block = set(real) | {d - pd.offsets.BDay(1) for d in real}
    pool = [d for d in candidates if d not in block]
    by_dow = {k: [d for d in pool if d.dayofweek == k] for k in range(5)}
    return [by_dow[d.dayofweek][int(rng.integers(len(by_dow[d.dayofweek])))] for d in real if by_dow[d.dayofweek]]


__all__ = ["PreFomcConfig", "Market", "INDICES", "UNIT", "statement_dates", "window", "daily_sigma", "market",
           "trade", "run_backtest", "placebo_days"]
