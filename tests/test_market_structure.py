"""Market structure framework: one hand-made price path per rule + no-look-ahead property.

Definitions: docs/MARKET_STRUCTURE.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_lab.market_structure import (BOS_DN, BOS_UP, CHOCH_DN, CHOCH_UP, StructureConfig,
                                          analyze, raw_pivots)

W = 0.2
# deterministic settings for hand-made paths: tiny pivots, no ATR filter
CFG = StructureConfig(atr_period=3, ext_left=2, ext_right=2, ext_min_swing_atr=0.0,
                      int_left=1, int_right=1, int_min_swing_atr=0.0, fvg_min_atr=0.0)


def _bars(closes: list[float]) -> pd.DataFrame:
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + W, "low": np.minimum(o, c) - W, "close": c},
                        index=pd.date_range("2024-01-01", periods=len(c), freq="1h"))


def _ohlc(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    a = np.asarray(rows, float)
    return pd.DataFrame(a, columns=["open", "high", "low", "close"],
                        index=pd.date_range("2024-01-01", periods=len(a), freq="1h"))


#        0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
TREND = [10, 11, 12, 13, 12, 11, 12, 13, 14, 15, 16, 15, 14, 15, 16, 17, 17, 17, 17,
         18, 17, 16.5, 17, 17.5,          # 19-23: a higher internal low at 16.5 (idx 21)
         16, 15, 14.5, 13.5, 13]          # 24-28: breaks 16.3 (not protected), then 13.8 (protected)


def test_pivots_are_known_only_after_the_right_side() -> None:
    df = _bars(TREND)
    piv = raw_pivots(df["high"].to_numpy(), df["low"].to_numpy(), 2, 2)
    # bars 3 and 4 share the high 13.2 (open = previous close): with "left <=, right <"
    # the LATER of two equal highs is the pivot, so the same high is never counted twice
    highs = [p for p in piv if p.kind == 1 and abs(p.price - (13 + W)) < 1e-9]
    assert len(highs) == 1 and highs[0].idx == 4
    assert highs[0].confirmed_idx == 6            # known only after 2 bars on the right


def test_small_wiggles_are_not_swings_and_highs_alternate() -> None:
    # 10 -> 12 (high) -> 11.9 (tiny dip) -> 12.5 (higher high) -> 10 : with a 1-ATR filter
    df = _bars([10, 10.5, 11, 12, 11.9, 12.5, 11.5, 10.5, 10, 10.2, 10.4, 10.6, 10.8, 11])
    res = analyze(df, StructureConfig(atr_period=3, ext_left=1, ext_right=1, ext_min_swing_atr=1.0))
    kinds = [s.kind for s in res.swings["ext"]]
    assert all(a != b for a, b in zip(kinds, kinds[1:])), "swings must alternate"
    assert not any(abs(s.price - (11.9 - W)) < 1e-9 for s in res.swings["ext"]), "the tiny dip is noise"


def test_bos_is_an_event_on_one_bar_only() -> None:
    res = analyze(_bars(TREND), CFG)
    ev = res.bars["ext_event"]
    assert ev.iloc[8] == CHOCH_UP                 # first direction out of neutral (close 14 > 13.2)
    assert ev.iloc[15] == BOS_UP                  # close 17 > swing high 16.2
    assert (ev.iloc[16:19] == 0).all()            # price stays above - no repeat
    assert ev.tolist().count(BOS_UP) == 1


def test_choch_needs_the_protected_low_not_any_low() -> None:
    res = analyze(_bars(TREND), CFG)
    ev = res.bars["ext_event"]
    assert res.bars["ext_protected_low"].iloc[15] == 14 - W    # low of the leg that broke out
    assert ev.iloc[24] == 0                       # 16 < internal low 16.3 - not protected: no CHoCH
    assert ev.iloc[27] == CHOCH_DN                # 13.5 < protected 13.8
    assert res.bars["ext_trend"].iloc[27] == -1


def test_order_block_is_the_last_opposite_candle_of_the_breaking_leg() -> None:
    res = analyze(_bars(TREND), CFG)
    obs = [z for z in res.zones if z.kind == "ob" and z.created_idx == 15]
    assert len(obs) == 1 and obs[0].direction == 1 and obs[0].origin_idx == 12


def test_fvg_fill_by_wick_then_inversion_by_close_then_ifvg_retest() -> None:
    rows = [(10, 10.2, 9.8, 10), (10, 10.2, 9.8, 10), (10, 10.2, 9.8, 10), (10, 10.2, 9.8, 10),
            (10, 10.2, 9.9, 10.1),            # 4: candle 1  (high 10.2)
            (10.1, 12.2, 10.0, 12.0),         # 5: displacement
            (12.0, 12.6, 11.0, 12.5),         # 6: candle 3 (low 11.0 > 10.2) -> bullish FVG [10.2, 11.0]
            (12.5, 12.6, 10.1, 10.5),         # 7: wick to 10.1 <= 10.2: FILLED, close 10.5 inside - not inverted
            (10.5, 10.6, 9.6, 9.8),           # 8: CLOSE 9.8 < 10.2 -> inverted -> bearish IFVG
            (9.8, 10.5, 9.7, 10.0),           # 9: back up into [10.2, 11.0] -> IFVG retest
            (10.0, 11.8, 9.9, 11.5)]          # 10: close 11.5 > 11.0 -> IFVG invalidated
    res = analyze(_ohlc(rows), CFG)
    fvg = next(z for z in res.zones if z.kind == "fvg" and z.direction == 1)
    assert (fvg.bottom, fvg.top, fvg.created_idx) == (10.2, 11.0, 6)
    assert fvg.filled_idx == 7 and fvg.inverted_idx == 8
    assert not fvg.active_at(7)                   # filled -> no longer an entry zone
    ifvg = next(z for z in res.zones if z.kind == "ifvg")
    assert ifvg.direction == -1 and ifvg.created_idx == 8 and ifvg.origin_idx == 6
    assert ifvg.retest_idx == 9 and ifvg.invalidated_idx == 10
    assert ifvg.active_at(9) and not ifvg.active_at(10)


def test_fvg_minimum_size() -> None:
    rows = [(10, 10.2, 9.8, 10)] * 5 + [(10, 10.25, 9.9, 10.2), (10.2, 10.5, 10.2, 10.4), (10.4, 10.6, 10.26, 10.5)]
    big = analyze(_ohlc(rows), StructureConfig(atr_period=3, fvg_min_atr=0.0))
    small = analyze(_ohlc(rows), StructureConfig(atr_period=3, fvg_min_atr=1.0))
    assert any(z.kind == "fvg" for z in big.zones) and not any(z.kind == "fvg" for z in small.zones)


def test_equal_highs_pool_is_swept_then_taken() -> None:
    #        0   1   2   3   4   5   6   7   8   9  10  11  12  13
    path = [10, 11, 12, 13, 12, 11, 12, 13, 12, 11, 11, 11, 11, 11]
    df = _bars(path)
    df.loc[df.index[11], "high"] = 13.5           # 11: wick above both highs, close back at 11
    df = pd.concat([df, _bars([11, 12, 13, 14]).iloc[1:].set_axis(
        pd.date_range(df.index[-1] + pd.Timedelta(hours=1), periods=3, freq="1h"))])
    res = analyze(df, StructureConfig(atr_period=3, ext_left=2, ext_right=2, ext_min_swing_atr=0.0,
                                      eq_tol_atr=0.5))
    pool = next(p for p in res.pools if p.kind == 1)
    assert abs(pool.level - (13 + W)) < 1e-9
    assert 11 in pool.swept_idx and res.bars["sweep_high"].iloc[11] == 1
    assert pool.taken_idx == len(df) - 1          # the close at 14 takes it


def _random(n: int = 700, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + rng.uniform(0, 1, n),
                         "low": np.minimum(o, c) - rng.uniform(0, 1, n), "close": c},
                        index=pd.date_range("2024-01-01", periods=n, freq="1h"))


def test_events_never_repeat_on_consecutive_bars() -> None:
    res = analyze(_random(), StructureConfig())
    for col in ("ext_event", "int_event"):
        e = res.bars[col].to_numpy()
        assert not np.any((e[1:] != 0) & (e[1:] == e[:-1])), col
    # and they are rare: an event, not a state
    assert (res.bars["int_event"] != 0).mean() < 0.15


def test_internal_structure_moves_more_often_than_external() -> None:
    res = analyze(_random(1500, seed=5), StructureConfig())
    assert (res.bars["int_event"] != 0).sum() > (res.bars["ext_event"] != 0).sum()
    assert len(res.swings["int"]) > len(res.swings["ext"])


def test_no_look_ahead_anywhere() -> None:
    """Cutting the series must not change anything already known at the cut."""
    full = _random(700, seed=11)
    cfg = StructureConfig()
    a = analyze(full, cfg)
    for k in (250, 400, 600):
        b = analyze(full.iloc[:k], cfg)
        cols = [c for c in b.bars.columns]
        assert np.allclose(b.bars[cols].to_numpy(), a.bars[cols].iloc[:k].to_numpy(), equal_nan=True), k
        known = lambda zs: sorted((z.kind, z.direction, z.created_idx, round(z.top, 9), round(z.bottom, 9))
                                  for z in zs if z.created_idx < k)
        assert known(b.zones) == known(a.zones), k
        for zb in b.zones:
            za = next(z for z in a.zones if (z.kind, z.created_idx, z.direction, z.top) ==
                      (zb.kind, zb.created_idx, zb.direction, zb.top))
            for f in ("touched_idx", "filled_idx", "inverted_idx", "retest_idx", "invalidated_idx"):
                vb, va = getattr(zb, f), getattr(za, f)
                assert vb == (va if va is not None and va < k else None), (k, f)
