"""Unit tests for SMC/ICT Phase-1 engines and strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_lab.risk import RiskManager
from trading_lab.strategies.smc.liquidity import LiquidityEngine
from trading_lab.strategies.smc.state_machine import StateMachine
from trading_lab.strategies.smc.structure import (
    MarketStructureEngine,
    detect_swings,
    structure_break_at_bar,
)
from trading_lab.strategies.smc.supply_demand import (
    detect_fvg_at,
    detect_all_fvgs,
    is_mitigated,
)
from trading_lab.strategies.smc.types import TradeState, TrendBias, Zone
from trading_lab.strategies.smc_ict import (
    SmcIctConfig,
    _exit_targets,
    _limit_and_sl,
    _rr_targets,
    last_closed_bar_idx,
    manage_position,
    run_backtest,
    EntryZone1m,
)
from trading_lab.strategies.smc.types import Zone


def _ohlcv(
    rows: list[tuple[float, float, float, float]],
    *,
    freq: str = "1min",
    start: str = "2024-01-02 12:00:00",
) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    data = {
        "open": [r[0] for r in rows],
        "high": [r[1] for r in rows],
        "low": [r[2] for r in rows],
        "close": [r[3] for r in rows],
        "volume": [100.0] * len(rows),
    }
    return pd.DataFrame(data, index=idx)


def test_fvg_bullish_causal() -> None:
    # bar0 high=10, bar2 low=11 → bullish FVG 10–11 at bar2 close
    rows = [
        (9.0, 10.0, 8.5, 9.5),
        (9.5, 10.5, 9.0, 10.0),
        (11.0, 12.0, 11.0, 11.5),
    ]
    df = _ohlcv(rows)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    assert detect_fvg_at(high, low, df.index, 1) is None
    z = detect_fvg_at(high, low, df.index, 2)
    assert z is not None
    assert z.direction == 1
    assert z.bottom == 10.0
    assert z.top == 11.0


def test_fvg_bearish_causal() -> None:
    rows = [
        (11.0, 12.0, 10.0, 11.5),
        (11.0, 11.5, 10.5, 10.8),
        (9.0, 9.5, 8.5, 9.0),
    ]
    df = _ohlcv(rows)
    z = detect_fvg_at(df["high"].to_numpy(), df["low"].to_numpy(), df.index, 2)
    assert z is not None
    assert z.direction == -1
    assert z.top == 10.0
    assert z.bottom == 9.5


def test_fvg_no_lookahead_in_list() -> None:
    rows = [
        (9.0, 10.0, 8.5, 9.5),
        (9.5, 10.5, 9.0, 10.0),
        (11.0, 12.0, 11.0, 11.5),
        (11.5, 12.5, 11.2, 12.0),
    ]
    df = _ohlcv(rows)
    fvgs = detect_all_fvgs(df)
    assert all(z.created_idx >= 2 for z in fvgs)
    assert fvgs[0].created_idx == 2


def test_swing_confirmed_with_lag() -> None:
    # Clear pivot high in the middle with L=2
    highs = [1, 2, 5, 3, 2, 2, 2]
    lows = [0.5, 1, 2, 1, 0.5, 0.5, 0.5]
    idx = pd.RangeIndex(len(highs))
    swings = detect_swings(np.array(highs, float), np.array(lows, float), idx, 2)
    highs_s = [s for s in swings if s.kind == 1]
    assert any(s.idx == 2 and s.confirmed_idx == 4 for s in highs_s)


def test_body_close_bos_not_wick() -> None:
    swings_levels = []
    # Manufacture prior swings via structure_break: need swing list
    from trading_lab.strategies.smc.types import SwingPoint

    swings = [
        SwingPoint(0, 0, 100.0, -1, 2),
        SwingPoint(3, 3, 110.0, 1, 5),
        SwingPoint(6, 6, 102.0, -1, 8),
        SwingPoint(9, 9, 115.0, 1, 11),
    ]
    # Wick above last high 115 but close below → no bullish break
    bos, choch, _ = structure_break_at_bar(114.0, swings, TrendBias.BULLISH)
    assert bos == 0
    # Body close above → BoS
    bos, choch, lvl = structure_break_at_bar(116.0, swings, TrendBias.BULLISH)
    assert bos == 1
    assert lvl == 115.0


def test_choch_against_trend() -> None:
    from trading_lab.strategies.smc.types import SwingPoint

    swings = [
        SwingPoint(0, 0, 100.0, -1, 2),
        SwingPoint(3, 3, 110.0, 1, 5),
        SwingPoint(6, 6, 105.0, -1, 8),
    ]
    bos, choch, lvl = structure_break_at_bar(104.0, swings, TrendBias.BULLISH)
    assert bos == 0
    assert choch == -1
    assert lvl == 105.0


def test_nested_choch_vs_pullback_not_bos() -> None:
    """HTF bearish + rolling 1m already bearish: break of low is nested ChoCh.

    Classic snapshot labels this BOS; nested_choch_dir uses pullback bias
    (bullish) so the same break is ChoCh — matching discretionary LTF confirm.
    """
    n = 40
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    # Uptrend into swing high, pullback HL, then break HL → with local bearish
    # bias after an earlier LL this is BOS; nested vs pullback → ChoCh.
    close = np.linspace(1.10, 1.12, n)
    high = close + 0.0003
    low = close - 0.0003
    # Build a clear swing low mid-series then break it
    pivot = 20
    low[pivot] = 1.1050
    high[pivot] = 1.1055
    close[pivot] = 1.1052
    # After pivot: bounce then close below pivot low
    for j in range(pivot + 1, pivot + 4):
        low[j] = 1.1051 + (j - pivot) * 0.0001
        high[j] = low[j] + 0.0004
        close[j] = (high[j] + low[j]) / 2
    break_i = pivot + 6
    low[break_i] = 1.1040
    high[break_i] = 1.1048
    close[break_i] = 1.1042  # body below 1.1050
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 1.0},
        index=idx,
    )
    eng = MarketStructureEngine(swing_length=2)
    eng.precompute(df)
    snap = eng.snapshot_at(df, break_i)
    nested = eng.nested_choch_dir(df, break_i, TrendBias.BEARISH)
    # Nested path must report bearish ChoCh for HTF shorts
    assert nested == -1
    # Rolling label may be BOS or ChoCh depending on bias_break; nested is ChoCh
    assert snap.last_bos_dir == -1 or snap.last_choch_dir == -1


def test_fvg_mitigation_wick() -> None:
    z = Zone(1, top=11.0, bottom=10.0, created_idx=2, created_ts=0, kind="fvg")
    assert not is_mitigated(z, high=10.8, low=10.2, close=10.5, mode="wick")
    assert is_mitigated(z, high=10.5, low=9.9, close=10.0, mode="wick")


def test_liquidity_asia_sweep() -> None:
    # Bars spanning Asia session 19:00–00:00 ET = 00:00–05:00 UTC (winter approx)
    # Use a full day of 15m bars
    idx = pd.date_range("2024-01-02 00:00:00", periods=96, freq="15min", tz="UTC")
    close = np.linspace(100, 110, len(idx))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1.0,
        },
        index=idx,
    )
    # Spike asia high inside early UTC hours
    df.loc[idx[2], "high"] = 120.0
    eng = LiquidityEngine(asia_start="19:00", asia_end="00:00")
    eng.build(df)
    asia_highs = [lv for lv in eng.levels if lv.kind == "asia_high"]
    assert asia_highs
    # Next day wick sweep
    asof = idx[-1]
    swept = eng.detect_sweeps(121.0, 110.0, asof)
    assert any(s.kind == "asia_high" for s in swept)


def test_state_machine_nested_flow() -> None:
    sm = StateMachine()
    sm.on_4h_bias(TrendBias.BULLISH)
    assert sm.ctx.state == TradeState.HTF_ALIGNED
    assert sm.on_sweep("asia_low")
    assert sm.ctx.state == TradeState.SWEPT
    z = Zone(1, 105, 100, 1, 0, "ob_fvg")
    # 15m align → immediately WAIT_FOR_CHOCH (no retest gate)
    assert sm.on_15m_align(TrendBias.BULLISH, choch_dir=0, poi=z)
    assert sm.ctx.state == TradeState.WAIT_FOR_CHOCH
    assert sm.on_choch(1)
    assert sm.ctx.state == TradeState.ENTRY_PENDING


def test_1m_choch_without_poi_retest() -> None:
    sm = StateMachine()
    sm.on_4h_bias(TrendBias.BULLISH)
    sm.on_sweep("asia_low")
    # Align without POI zone still arms 1m watch
    assert sm.on_15m_align(TrendBias.BULLISH, choch_dir=1, poi=None)
    assert sm.ctx.state == TradeState.WAIT_FOR_CHOCH
    assert sm.on_choch(1)


def test_fill_consumes_sweep_needs_fresh_liquidity() -> None:
    """After a fill, the next trade requires a new session/PD/PW sweep."""
    sm = StateMachine()
    sm.on_4h_bias(TrendBias.BEARISH)
    assert sm.on_sweep("asia_high")
    assert sm.on_15m_align(TrendBias.BEARISH, choch_dir=-1, poi=None)
    assert sm.on_choch(-1)
    sm.on_fill()
    assert sm.ctx.state == TradeState.IN_TRADE
    assert sm.ctx.recent_sweep is False
    sm.on_trade_closed()
    assert sm.ctx.state == TradeState.HTF_ALIGNED
    assert sm.ctx.recent_sweep is False
    # Same kind cannot re-fire without going through HTF_ALIGNED sweep again
    assert sm.on_sweep("asia_high")
    assert sm.ctx.state == TradeState.SWEPT


def test_new_trading_day_expires_sweep() -> None:
    sm = StateMachine()
    sm.on_4h_bias(TrendBias.BEARISH)
    sm.on_sweep("pdh")
    sm.on_15m_align(TrendBias.BEARISH, choch_dir=-1, poi=None)
    assert sm.ctx.state == TradeState.WAIT_FOR_CHOCH
    sm.on_new_trading_day()
    assert sm.ctx.recent_sweep is False
    assert sm.ctx.state == TradeState.HTF_ALIGNED


def test_15m_may_align_after_sweep() -> None:
    sm = StateMachine()
    sm.on_4h_bias(TrendBias.BULLISH)
    sm.on_sweep("pdl")
    z = Zone(1, 1.10, 1.09, 5, 0, "ob_fvg")
    assert sm.on_15m_align(TrendBias.NEUTRAL, choch_dir=1, poi=z)
    assert sm.ctx.state == TradeState.WAIT_FOR_CHOCH


def test_last_closed_bar_idx() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}, index=idx
    )
    # At exactly bar0 + 15m, bar0 is closed
    now = idx[0] + pd.Timedelta(minutes=15)
    assert last_closed_bar_idx(df, now, pd.Timedelta(minutes=15)) == 0
    # One minute before close → nothing closed from bar0
    now2 = idx[0] + pd.Timedelta(minutes=14)
    assert last_closed_bar_idx(df, now2, pd.Timedelta(minutes=15)) == -1


def test_rr_targets() -> None:
    cfg = SmcIctConfig(tp1_r=1.5, tp2_r=3.0, tp_mode="rr")
    tp1, tp2 = _rr_targets(100.0, 90.0, "long", cfg)
    assert tp1 == 115.0
    assert tp2 == 130.0
    tp1s, tp2s = _rr_targets(100.0, 110.0, "short", cfg)
    assert tp1s == 85.0
    assert tp2s == 70.0


def test_sl_on_sweep_extreme() -> None:
    """P0: SL beyond swept liquidity extreme + buffer."""
    fvg = Zone(direction=1, top=1.14620, bottom=1.14590, created_idx=10, created_ts=0, kind="fvg")
    ob = Zone(direction=1, top=1.14600, bottom=1.14580, created_idx=8, created_ts=0, kind="ob")
    entry = EntryZone1m(direction=1, ob=ob, fvg=fvg, merged=fvg)
    cfg = SmcIctConfig(
        entry_limit_mode="fvg_edge",
        entry_edge_offset=0.00005,
        sl_mode="sweep",
        sl_buffer=0.00005,
        max_sl_dist=0.0,  # no cap
    )
    limit, sl = _limit_and_sl(
        entry, "long", swing_price=1.14570, cfg=cfg, sweep_extreme=1.14550
    )
    assert abs(limit - (1.14590 + 0.00005)) < 1e-9
    assert abs(sl - (1.14550 - 0.00005)) < 1e-9


def test_sl_sweep_falls_back_when_too_wide() -> None:
    """Deep sweep wick → fall back to OB so risk stays tradeable."""
    fvg = Zone(direction=1, top=1.14620, bottom=1.14590, created_idx=10, created_ts=0, kind="fvg")
    ob = Zone(direction=1, top=1.14600, bottom=1.14580, created_idx=8, created_ts=0, kind="ob")
    entry = EntryZone1m(direction=1, ob=ob, fvg=fvg, merged=fvg)
    cfg = SmcIctConfig(
        entry_limit_mode="fvg_edge",
        entry_edge_offset=0.0,
        sl_mode="sweep",
        sl_buffer=0.0,
        max_sl_dist=0.0005,
    )
    # Sweep 20 pips below entry → exceeds cap → OB bottom
    _, sl = _limit_and_sl(
        entry, "long", swing_price=1.14570, cfg=cfg, sweep_extreme=1.14390
    )
    assert abs(sl - 1.14580) < 1e-9


def test_liquidity_trade_targets_irl_erl() -> None:
    """Nearest buy-side pool = IRL; farther PDH = ERL for a long."""
    idx = pd.date_range("2024-01-02 12:00:00", periods=4, freq="15min", tz="UTC")
    # Minimal OHLC so build() succeeds; we inject levels after
    df = pd.DataFrame(
        {
            "open": [1.1, 1.1, 1.1, 1.1],
            "high": [1.12, 1.12, 1.12, 1.12],
            "low": [1.08, 1.08, 1.08, 1.08],
            "close": [1.1, 1.1, 1.1, 1.1],
            "volume": [1.0] * 4,
        },
        index=idx,
    )
    eng = LiquidityEngine(use_asia=False, use_london=False, use_pd=False, use_pw=False)
    eng.build(df)
    from trading_lab.strategies.smc.types import LiquidityLevel

    eng.levels = [
        LiquidityLevel(1.1020, "asia_high", idx[0], swept=False),
        LiquidityLevel(1.1080, "pdh", idx[0], swept=False),
        LiquidityLevel(1.0950, "asia_low", idx[0], swept=False),
    ]
    for lv in eng.levels:
        lv.created_unix = float(pd.Timestamp(lv.created_ts).timestamp())
    eng._all_levels = list(eng.levels)
    eng._next_level_i = len(eng.levels)
    eng._unswept = list(eng.levels)
    eng._built = True

    irl, erl = eng.trade_targets(1.1000, "long", idx[-1], min_dist=0.0)
    assert irl == 1.1020
    assert erl == 1.1080

    cfg = SmcIctConfig(
        tp_mode="liquidity",
        min_irl_r=0.5,
        min_erl_r=1.5,
        max_irl_r=3.0,
        max_erl_r=6.0,
        tp1_r=1.5,
        tp2_r=3.0,
        tp_fallback_rr=True,
    )
    # risk = 0.002 → min_irl=0.001, max_erl=0.012; PDH is 0.008 away → ok
    tp1, tp2 = _exit_targets(
        1.1000, 1.0980, "long", cfg, liq=eng, asof_ts=idx[-1]
    )
    assert abs(tp1 - 1.1020) < 1e-9
    assert abs(tp2 - 1.1080) < 1e-9

    # Stale far PDH must not become ERL when beyond max_dist
    eng.levels.append(
        LiquidityLevel(1.2000, "pdh", idx[0], swept=False),
    )
    eng.levels[-1].created_unix = float(pd.Timestamp(idx[0]).timestamp())
    eng._all_levels = list(eng.levels)
    irl2, erl2 = eng.trade_targets(
        1.1000, "long", idx[-1], min_dist=0.0, max_dist=0.012
    )
    assert irl2 == 1.1020
    assert erl2 == 1.1080  # nearest HTF inside band, not 1.20


def test_sl_on_ob_and_fvg_entry_offset() -> None:
    """Playbook: short SL at OB top + buffer; limit slightly below FVG top (proximal)."""
    fvg = Zone(direction=-1, top=1.14620, bottom=1.14598, created_idx=10, created_ts=0, kind="fvg")
    ob = Zone(direction=-1, top=1.14653, bottom=1.14610, created_idx=8, created_ts=0, kind="ob")
    entry = EntryZone1m(direction=-1, ob=ob, fvg=fvg, merged=fvg)
    cfg = SmcIctConfig(
        entry_limit_mode="fvg_edge",
        entry_edge_offset=0.00005,
        sl_mode="ob",
        sl_buffer=0.0,
    )
    limit, sl = _limit_and_sl(entry, "short", swing_price=1.14700, cfg=cfg)
    assert abs(limit - (1.14620 - 0.00005)) < 1e-9
    assert abs(sl - 1.14653) < 1e-9


def test_manage_position_sl_and_tp() -> None:
    cfg = SmcIctConfig(tp1_close_pct=0.5, max_hold_bars=100, move_sl_to_be_on_tp1=True)
    rm = RiskManager(cfg, account_size=10_000, backtest_mode=True)  # type: ignore[arg-type]
    pos = rm.open_position(
        100.0,
        atr=1.0,
        direction="long",
        entry_time=1.0,
        entry_idx=10,
        stop_loss=90.0,
        tp1=115.0,
        tp2=130.0,
    )
    assert pos is not None
    trades: list[dict] = []
    manage_position(pos, 11, 2.0, 100.0, 101.0, 89.0, cfg, rm, trades)
    assert pos.status == "closed"
    assert pos.exit_reason == "stop_loss"


def test_manage_position_tp1_keeps_sl_when_be_off() -> None:
    cfg = SmcIctConfig(tp1_close_pct=0.5, max_hold_bars=100, move_sl_to_be_on_tp1=False)
    rm = RiskManager(cfg, account_size=10_000, backtest_mode=True)  # type: ignore[arg-type]
    pos = rm.open_position(
        100.0,
        atr=1.0,
        direction="long",
        entry_time=1.0,
        entry_idx=10,
        stop_loss=90.0,
        tp1=115.0,
        tp2=130.0,
    )
    assert pos is not None
    trades: list[dict] = []
    manage_position(pos, 11, 2.0, 116.0, 116.0, 100.0, cfg, rm, trades)
    assert pos.tp1_hit
    assert pos.status == "open"
    assert pos.stop_loss == 90.0
    assert not pos.sl_moved_to_be


def test_manage_position_tp1_then_tp2() -> None:
    cfg = SmcIctConfig(tp1_close_pct=0.5, max_hold_bars=100, move_sl_to_be_on_tp1=True)
    rm = RiskManager(cfg, account_size=10_000, backtest_mode=True)  # type: ignore[arg-type]
    pos = rm.open_position(
        100.0,
        atr=1.0,
        direction="long",
        entry_time=1.0,
        entry_idx=10,
        stop_loss=90.0,
        tp1=115.0,
        tp2=130.0,
    )
    assert pos is not None
    trades: list[dict] = []
    manage_position(pos, 11, 2.0, 116.0, 116.0, 100.0, cfg, rm, trades)
    assert pos.tp1_hit
    assert pos.status == "open"
    assert pos.stop_loss == 100.0
    manage_position(pos, 12, 3.0, 130.0, 131.0, 120.0, cfg, rm, trades)
    assert pos.status == "closed"
    assert pos.exit_reason == "tp2"


def test_run_backtest_smoke() -> None:
    """Synthetic uptrend multi-TF series — must run without error."""
    n_1m = 600
    idx_1m = pd.date_range("2024-03-04 08:00:00", periods=n_1m, freq="1min", tz="UTC")
    # Gentle drift with noise
    rng = np.random.default_rng(0)
    close_1m = 2000 + np.cumsum(rng.normal(0.02, 0.3, n_1m))
    df_1m = pd.DataFrame(
        {
            "open": close_1m,
            "high": close_1m + 0.8,
            "low": close_1m - 0.8,
            "close": close_1m,
            "volume": 100.0,
        },
        index=idx_1m,
    )
    df_15m = (
        df_1m.resample("15min")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )
    df_4h = (
        df_1m.resample("4h")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )
    cfg = SmcIctConfig(
        swing_length_4h=2,
        swing_length_15m=2,
        swing_length_1m=2,
        entry_timeout_bars=20,
        cooldown_hours=0,
    )
    result = run_backtest(df_4h, df_15m, df_1m, cfg)
    assert "trades" in result
    assert "metrics" in result
    assert result["metrics"]["total_trades"] >= 0


def test_structure_engine_snapshot_causal() -> None:
    highs = [1, 2, 5, 3, 2, 2, 1, 0.5, 1, 2]
    lows = [0.5, 1, 2, 1, 0.5, 0.4, 0.3, 0.2, 0.5, 1]
    closes = [0.8, 1.5, 4, 2, 1.5, 1.2, 0.7, 0.4, 0.8, 1.5]
    opens = closes
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1},
        index=pd.date_range("2024-01-01", periods=len(highs), freq="h", tz="UTC"),
    )
    eng = MarketStructureEngine(2)
    eng.precompute(df)
    early = eng.snapshot_at(df, 3)
    late = eng.snapshot_at(df, 9)
    # More swings can appear later; never fewer confirmations with time advance
    assert len(late.swings) >= len(early.swings)
