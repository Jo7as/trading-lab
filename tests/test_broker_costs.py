"""MT5 cost logger import (trading_lab/broker_costs.py) and its use in S-007."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from trading_lab import broker_costs as B
from trading_lab.strategies import tsmom_multi_asset as M


def _write(tmp, kind: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(tmp / f"tl_{kind}_123.csv", sep=";", index=False)


def test_summary_from_logger_files(tmp_path) -> None:
    # 2024-07-02 19:05 UTC = 15:05 New York (EDT); 13:05 UTC = 09:05 NY must not count
    _write(tmp_path, "spreads", [
        {"utc": "2024.07.02 19:05:00", "lab": "SP500", "symbol": "US500", "bid": 5000, "ask": 5000.5, "spread_bp": 1.0},
        {"utc": "2024.07.02 19:06:00", "lab": "SP500", "symbol": "US500", "bid": 5000, "ask": 5000.5, "spread_bp": 1.2},
        {"utc": "2024.07.02 13:05:00", "lab": "SP500", "symbol": "US500", "bid": 5000, "ask": 5005, "spread_bp": 10.0},
    ])
    _write(tmp_path, "symbols", [
        {"utc": "2024.07.02 00:01:00", "lab": "SP500", "symbol": "US500", "tick_value": 1.0, "tick_size": 0.01,
         "swap_long_pa": -0.07, "swap_short_pa": 0.02},
        {"utc": "2024.07.03 00:01:00", "lab": "SP500", "symbol": "US500", "tick_value": 1.0, "tick_size": 0.01,
         "swap_long_pa": -0.07, "swap_short_pa": 0.02},
    ])
    _write(tmp_path, "probes", [
        {"utc": "2024.07.02 19:00:01", "lab": "SP500", "symbol": "US500", "side": "open_buy", "volume": 0.1,
         "quote": 5000.5, "fill": 5000.75, "slip_bp": 0.5, "exec_ms": 40, "commission": -0.25, "deal": 1},
        {"utc": "2024.07.02 19:00:07", "lab": "SP500", "symbol": "US500", "side": "close_sell", "volume": 0.1,
         "quote": 5000.0, "fill": 5000.0, "slip_bp": 0.0, "exec_ms": 40, "commission": -0.25, "deal": 2},
    ])
    s = B.summary(tmp_path, "123").loc["SP500"]
    assert s["half_spread_bp_15ny"] == pytest.approx(0.55)              # median 1.1 / 2, 09:05 NY ignored
    assert s["swap_long_pa"] == pytest.approx(-0.07) and s["slip_bp"] == pytest.approx(0.25)
    # commission 0.25 on 0.1 x 5000 x 100 = 50 000 notional -> 0.05 bp
    assert s["comm_bp_side"] == pytest.approx(0.05, rel=1e-3)
    assert s["cost_bp_side"] == pytest.approx(0.55 + 0.25 + 0.05, rel=1e-3)
    out = tmp_path / "o.json"
    B.write_override(B.summary(tmp_path, "123"), out, source="demo")
    j = json.loads(out.read_text())
    assert j["swaps_pa"]["SP500"] == [-0.07, 0.02]


def test_measured_swaps_replace_the_formula() -> None:
    rates = pd.DataFrame({"USD": 5.0, "EUR": 3.0}, index=pd.date_range("2020-01-31", periods=3, freq="ME"))
    cfg = M.TsmomConfig(swaps_pa=(("SP500", -0.01, -0.005),))
    lo, sh = M.carry_rates("SP500", rates, cfg)
    assert lo.iloc[0] == -0.01 and sh.iloc[0] == -0.005
    lo, _ = M.carry_rates("US100", rates, cfg)                            # not measured: formula
    assert lo.iloc[0] == pytest.approx(-0.075)


def test_ea_new_york_hour_formula_matches_the_time_zone_database() -> None:
    """Port of NyHour() in mt5/TradingLabCostLogger.mq5 - checked against zoneinfo."""
    def ny_hour(g: datetime) -> int:
        dow = (g.weekday() + 1) % 7                                         # MQL5: Sunday = 0
        dow1 = (dow - (g.day - 1) % 7 + 7) % 7
        first_sun = 1 + (7 - dow1) % 7
        dst = (3 < g.month < 11) or (g.month == 3 and g.day >= first_sun + 7) or (g.month == 11 and g.day < first_sun)
        return (g.hour - (4 if dst else 5) + 24) % 24
    t = datetime(2019, 1, 1, 12, tzinfo=timezone.utc)
    bad = 0
    while t.year < 2027:
        if ny_hour(t) != t.astimezone(ZoneInfo("America/New_York")).hour:
            bad += 1
        t += timedelta(hours=24)
    assert bad == 0
