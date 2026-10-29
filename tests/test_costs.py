"""Cost profiles: commission per venue, spread rescaling, and that every path charges them."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_lab import costs
from trading_lab.optimizer import normalise


def _frame(spread: float, n: int = 600) -> pd.DataFrame:
    idx = pd.date_range("2024-03-04 12:00", periods=n, freq="1min", tz="UTC")
    c = np.full(n, 0.66)
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=idx)
    for col in ("open", "high", "low", "close"):
        df[f"ask_{col}"] = df[col] + spread
    return df


def test_ftmo_fx_commission_matches_the_published_rate() -> None:
    # $2.50 per lot per side on 100k EUR at 1.08 USD -> 0.23 bp per side
    assert costs.commission_bp_side("EURUSD", 1.08, "ftmo") == pytest.approx(2.5 / 108_000 * 1e4)
    assert costs.commission_bp_side("USDJPY", 150.0, "ftmo") == pytest.approx(0.25)
    assert costs.commission_bp_side("US100", 20_000, "ftmo") == 0.0       # no index commission
    assert costs.commission_bp_side("XAUUSD", 2300, "ftmo") == pytest.approx(0.07)


def test_the_old_model_charged_no_commission_at_all() -> None:
    assert costs.commission_bp_side("EURUSD", 1.08, "data") == 0.0


def test_ecn_profile_shrinks_the_wide_aud_spread_of_the_feed() -> None:
    m1 = _frame(spread=0.00009)                       # 0.9 pip, what the Dukascopy feed shows
    out = costs.adjust_frame(m1, "AUDUSD", "ecn_raw")
    assert (out["ask_close"] - out["close"]).median() == pytest.approx(0.000005)
    assert (out["ask_close"] >= out["close"]).all()
    # idempotent: applying it twice changes nothing
    again = costs.adjust_frame(out, "AUDUSD", "ecn_raw")
    assert np.allclose(again["ask_close"], out["ask_close"])


def test_profiles_without_an_override_keep_the_data() -> None:
    m1 = _frame(spread=0.00009)
    assert costs.adjust_frame(m1, "AUDUSD", "ftmo") is m1


def test_normalise_charges_the_commission_in_R() -> None:
    m1 = _frame(spread=0.00002)
    trades = [{"entry_time": m1.index[10], "exit_time": m1.index[40], "direction": 1,
               "entry": 1.08, "stop": 1.0790, "target": None, "r": 0.5}]
    plain = normalise(trades, m1, None)
    ftmo = normalise(trades, m1, None, asset="EURUSD", cost_profile="ftmo")
    want = 2 * costs.commission_bp_side("EURUSD", 1.08, "ftmo") * 1e-4 * 1.08 / 0.001
    assert plain["r"].iloc[0] - ftmo["r"].iloc[0] == pytest.approx(want)
    assert ftmo["r_gross"].iloc[0] == pytest.approx(plain["r_gross"].iloc[0])   # gross is pre-cost


def test_unknown_profile_is_an_error() -> None:
    with pytest.raises(KeyError):
        costs.get("nope")


def test_crypto_round_trip_taker_maker_and_spread():
    from trading_lab import costs

    assert costs.asset_class("BTCUSDT") == "crypto"
    # Bybit: 5.5 bp taker per side + 0.05 bp half spread per taker leg on BTC
    assert costs.crypto_round_trip_bp("BTC", "bybit_perp") == pytest.approx(11.1)
    # limit entry, market exit: 2 bp + 5.5 bp + SOL half spread 0.4 bp
    assert costs.crypto_round_trip_bp("SOLUSDT", "bybit_perp", maker_entry=True) == pytest.approx(7.9)
    assert costs.crypto_round_trip_bp("BTC", "binance_um", maker_entry=True, maker_exit=True) == pytest.approx(4.0)
    assert costs.commission_bp_side("SOL", 150.0, "binance_um") == pytest.approx(5.0)
    with pytest.raises(ValueError):
        costs.crypto_round_trip_bp("BTC", "ftmo")
    with pytest.raises(ValueError):
        costs.crypto_round_trip_bp("US100", "bybit_perp")
