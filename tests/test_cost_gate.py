"""The cost convention: one per-leg rule, in units of the full spread.

Before this existed in one place, the studies disagreed by a factor of two on the
same trade — `run_h001`, `mfe_mae` and `first_passage_grid` charged two full
spreads for a round trip where `cost_gate` assumed one, which is enough to flip a
candidate (`docs/AEP_STAGE0_K3.md`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_lab.cost_gate import (
    leg_cost,
    horizon_gate,
    round_trip_spread,
    spread_pct,
    trades_move_gate,
)


def test_the_convention_table() -> None:
    assert round_trip_spread("limit", "limit") == 0.0
    assert round_trip_spread("limit", "market") == 0.5
    assert round_trip_spread("limit", "stop") == 0.5
    assert round_trip_spread("market", "target") == 0.5
    assert round_trip_spread("market", "market") == 1.0
    assert round_trip_spread() == 1.0                      # default: both market
    assert leg_cost("TP2") == 0.0 and leg_cost(" Time ") == 0.5


def test_an_unknown_order_type_is_an_error_not_a_silent_guess() -> None:
    with pytest.raises(ValueError):
        leg_cost("voodoo")


def _frame(n: int = 400, spread: float = 0.02) -> pd.DataFrame:
    idx = pd.date_range("2024-03-05 14:30", periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
    df = pd.DataFrame({"open": close, "high": close + 0.01, "low": close - 0.01, "close": close},
                      index=idx)
    df["ask_close"] = df["close"] + spread
    return df


def test_horizon_gate_charges_one_spread_for_both_market_legs() -> None:
    df = _frame()
    g = horizon_gate(df, 60)
    full = float(spread_pct(df).median())
    assert g.cost_taker_pct == pytest.approx(full * 100)
    assert g.cost_limit_pct == pytest.approx(full * 50)     # half of the taker cost


def test_trades_move_gate_charges_the_limit_exit_less() -> None:
    df = _frame()
    t0 = float(df.index[10].timestamp())
    trades = [
        {"entry_price": 100.0, "exit_price": 100.1, "entry_time": t0,
         "exit_time": t0 + 60, "exit_reason": "target"},
        {"entry_price": 100.0, "exit_price": 100.1, "entry_time": t0,
         "exit_time": t0 + 60, "exit_reason": "stop"},
    ]
    out = trades_move_gate(trades, df)
    full = float(spread_pct(df).median()) * 100
    assert out["cost_pct_median"] == pytest.approx((0.5 * full + 1.0 * full) / 2, rel=0.05)
