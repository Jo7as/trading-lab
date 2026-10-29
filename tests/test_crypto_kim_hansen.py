"""Kim & Hansen replication harness on a synthetic path with a known opening effect."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _study():
    path = ROOT / "studies" / "crypto_kim_hansen_replication.py"
    spec = importlib.util.spec_from_file_location("kh_study", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic(days: int, effect_bp: float, seed: int = 3) -> pd.DataFrame:
    """10 s bars; each quarter-hour opening has a random OI, and the price jumps by
    effect_bp x OI right after the opening window (a permanent impact)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=days * 8640, freq="10s", tz="UTC", name="ts")
    n = len(idx)
    other = rng.uniform(-1, 1, n)                     # flow on all other bars: no effect
    buy, sell = 1 + other, 1 - other
    logp = rng.normal(0, 2e-5, n)                    # small noise per bar
    marks = np.flatnonzero((idx.minute % 15 == 0) & (idx.second == 0))
    oi = rng.uniform(-1, 1, len(marks))
    buy[marks], sell[marks] = 1 + oi, 1 - oi          # imbalance of the opening bar = oi
    logp[marks + 1] += effect_bp * 1e-4 * oi          # first bar after the window
    close = 100 * np.exp(np.cumsum(logp))
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                         "vwap": close, "buy_qty": buy, "sell_qty": sell,
                         "buy_notional": buy, "sell_notional": sell,
                         "n_trades": np.ones(n)}, index=idx)


def test_measure_recovers_known_opening_effect_and_placebo_is_flat():
    st = _study()
    bars = _synthetic(days=20, effect_bp=5.0)
    real = st.measure(bars)
    for h in ("30min", "4h", "8h"):
        assert real["rows"][h]["beta_bp"] == pytest.approx(5.0, abs=1.0)
    assert real["rows"]["4h"]["t"] > 3
    placebo = st.measure(bars, offset="2min")
    assert abs(placebo["rows"]["4h"]["t"]) < 3


def test_holdout_is_refused(monkeypatch):
    st = _study()
    monkeypatch.setattr("sys.argv", ["x", "--end", "2025-01-02"])
    with pytest.raises(SystemExit):
        st.main()
