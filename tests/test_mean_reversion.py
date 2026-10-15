"""Tests for the two-tier z-score mean reversion."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_lab.mr_common import prepare_frame
from trading_lab.strategies.mean_reversion import (
    MeanReversionConfig,
    generate_signals,
    run_backtest,
)

_NOW = pd.Timestamp("2024-03-10", tz="UTC")
_DROP = 60  # bar index of the stretched close


def _cfg(**kw) -> MeanReversionConfig:
    base = dict(
        adx_block_threshold=101.0,  # the synthetic chop pins ADX at 100
        fallback_spread_pct=0.0,
        slippage_pct=0.0,
        session_filter_enabled=False,
    )
    base.update(kw)
    return MeanReversionConfig(**base)


def _series(start: str = "2024-03-05 08:00", drop_to: float = 99.6) -> pd.DataFrame:
    """Chop of ±0.1 around 100, one stretched close, then a snap back to 100."""
    idx = pd.date_range(start, periods=80, freq="15min", tz="UTC")
    close = np.where(np.arange(80) % 2 == 0, 100.1, 99.9)
    close[_DROP] = drop_to
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx
    )
    # entry bar: opens at the stretched close, recovers through TP1
    df.iloc[_DROP + 1] = [drop_to, 99.95, drop_to - 0.05, 99.93, 1.0]
    df.iloc[_DROP + 2] = [99.93, 100.15, 99.9, 100.1, 1.0]
    return df


def test_tier_a_long_reverts_to_mean() -> None:
    r = run_backtest(_series(), _cfg(), now=_NOW)
    assert r["metrics"]["total_trades"] >= 1
    t = r["trades"][0]
    assert t["direction"] == "long"
    assert t["meta"]["tier"] == "A"
    assert t["meta"]["z"] <= -2.0
    assert t["exit_reason"] == "tp2"
    assert t["pnl_pct"] > 0
    assert abs(t["meta"]["risk_pct"] - 1.0) < 1e-6


def test_entry_fills_at_next_bar_open() -> None:
    df = _series()
    t = run_backtest(df, _cfg(), now=_NOW)["trades"][0]
    assert t["entry_time"] == df.index[_DROP + 1].timestamp()
    assert t["entry_price"] == df["open"].iloc[_DROP + 1]


def test_tier_b_trades_half_size_and_never_double_counts() -> None:
    cfg = _cfg(z_entry_tier_a=10.0)  # tier A out of reach → same setup is tier B
    r = run_backtest(_series(), cfg, now=_NOW)
    t = r["trades"][0]
    assert t["meta"]["tier"] == "B"
    assert abs(t["meta"]["risk_pct"] - 0.5) < 1e-6

    sig = generate_signals(prepare_frame(_series(), "15min", 0.0, _NOW), _cfg())
    fired = sig[(sig["long_signal"] > 0) | (sig["short_signal"] > 0)]
    assert len(fired) >= 1
    assert ((fired["tier"] == "A") == (fired["z"].abs() >= 2.0)).all()
    assert (sig.loc[~sig.index.isin(fired.index), "tier"] == "").all()


def test_adx_block_stops_all_entries() -> None:
    r = run_backtest(_series(), _cfg(adx_block_threshold=50.0), now=_NOW)
    assert r["metrics"]["total_trades"] == 0


def test_dead_market_has_no_signal_and_no_division_by_zero() -> None:
    idx = pd.date_range("2024-03-05 08:00", periods=80, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 0.0}, index=idx
    )
    r = run_backtest(df, _cfg(), now=_NOW)
    assert r["metrics"]["total_trades"] == 0
    assert not np.isinf(r["signals_df"]["z"]).any()


def test_gap_inside_z_window_blocks_signal() -> None:
    df = _series()
    # push everything from 5 bars before the stretch three hours later
    shifted = df.index.to_series()
    shifted.iloc[_DROP - 5 :] += pd.Timedelta(hours=3)
    df.index = pd.DatetimeIndex(shifted.to_numpy(), tz="UTC")
    r = run_backtest(df, _cfg(), now=_NOW)
    assert all(t["entry_idx"] != _DROP + 1 for t in r["trades"])


def test_session_filter_blocks_asia_hours() -> None:
    df = _series(start="2024-03-05 00:00")  # stretch closes at 15:15, inside session
    night = _series(start="2024-03-04 12:00")  # stretch closes at 03:15, outside
    on = _cfg(session_filter_enabled=True)
    assert run_backtest(df, on, now=_NOW)["metrics"]["total_trades"] >= 1
    assert all(
        t["entry_idx"] != _DROP + 1 for t in run_backtest(night, on, now=_NOW)["trades"]
    )
    assert run_backtest(night, _cfg(), now=_NOW)["trades"][0]["entry_idx"] == _DROP + 1


def test_long_buys_at_ask_when_ask_side_present() -> None:
    df = _series()
    for side in ("open", "high", "low", "close"):
        df[f"ask_{side}"] = df[side] + 0.04
    t = run_backtest(df, _cfg(), now=_NOW)["trades"][0]
    assert t["entry_price"] == df["ask_open"].iloc[_DROP + 1]
    free = run_backtest(_series(), _cfg(), now=_NOW)["trades"][0]
    assert t["pnl_pct"] < free["pnl_pct"]


def test_time_stop_closes_stale_position() -> None:
    df = _series()
    # no recovery: pin everything after the stretch just above the entry
    df.iloc[_DROP + 1 :] = [99.6, 99.65, 99.55, 99.6, 1.0]
    t = run_backtest(df, _cfg(max_hold_bars=4), now=_NOW)["trades"][0]
    assert t["exit_reason"] == "time_stop"
    assert t["exit_time"] == (df.index[_DROP + 4] + pd.Timedelta(minutes=15)).timestamp()
