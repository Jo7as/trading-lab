"""Which strategies the optimiser and the guided check know, and how to feed them.

One adapter per strategy: a discrete search grid (entry, filters, stop, target,
holding time, session), a ``prepare`` (1-minute bid/ask frame -> what the strategy's
``run_backtest`` consumes) and a ``run`` returning the raw trade log. Used by
``studies/optimize_all.py`` (batch) and ``studies/pruefen.py`` (guided, step by step),
so both run exactly the same thing.

Adding a strategy: write ``_my_idea(asset) -> Spec`` and register it in ``REGISTRY``
with the instruments it makes sense on.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .minute_data import DEFAULT_CACHE
from .mr_common import resample_ohlc
from .optimizer import Spec

YEAR_START = 2019
HOLDOUT_START = pd.Timestamp("2025-01-01", tz="UTC")
ASSETS = ("US100", "SP500", "GER40", "XAUUSD", "EURUSD", "USDJPY")


def load_cached(asset: str, year_start: int = YEAR_START, cache: Path = DEFAULT_CACHE) -> pd.DataFrame:
    """1-minute bid/ask from the cache only. The running year: newest dated file."""
    safe = re.sub(r"[^A-Za-z0-9]+", "", asset)
    years = []
    for y in range(year_start, pd.Timestamp.now(tz="UTC").year + 1):
        sides = {}
        for side in ("bid", "ask"):
            p = cache / f"{safe}_1m_{side}_{y}.pkl"
            if not p.exists():
                dated = sorted(cache.glob(f"{safe}_1m_{side}_{y}_*.pkl"))
                p = dated[-1] if dated else None
            if p is None:
                break
            sides[side] = pd.read_pickle(p)
        if len(sides) < 2:
            continue
        bid, ask = sides["bid"], sides["ask"]
        years.append(bid.join(ask[["open", "high", "low", "close"]].add_prefix("ask_"), how="inner"))
    if not years:
        return pd.DataFrame()
    out = pd.concat(years)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    for c in ("open", "high", "low", "close"):
        out[f"ask_{c}"] = np.maximum(out[f"ask_{c}"], out[c])
    return out[["open", "high", "low", "close", "ask_open", "ask_high", "ask_low", "ask_close"]
               + (["volume"] if "volume" in out else [])]


# ── Adapters: one function per strategy, returning the Spec for an asset ──

def _momentum(asset: str) -> Spec:
    from .strategies.momentum import MomentumConfig, run_backtest
    sess = ("ny",) if asset in ("US100", "SP500") else ("london",) if asset == "GER40" else ("london", "ny")
    grid = {"entry": ("breakout", "pullback", "momentum", "market"), "stop": ("atr", "swing"),
            "exit": ("target", "trail_atr", "be_trail", "time"), "regime": ("none", "er", "adx"),
            "regime_min": (0.2, 0.3, 0.45), "session": sess, "entry_lookback": (10, 20, 40),
            "target_r": (1.0, 1.5, 2.0, 3.0), "stop_atr": (1.0, 1.5, 2.5),
            "max_hold_minutes": (120, 240, 480)}
    def run(c, d):
        tr = run_backtest(d, c)["trades"]
        if c.exit != "target":              # trailing/time exits have no fixed target
            for t in tr:
                t["target"] = None
        return tr
    return Spec("momentum", asset, MomentumConfig(asset=asset, session=sess[0]), grid,
                prepare=lambda m1: m1, run=run)


def _pullback(asset: str) -> Spec:
    from .strategies.pullback import PullbackConfig, run_backtest
    sess = ("ny",) if asset in ("US100", "SP500") else ("london",) if asset == "GER40" else ("london", "ny")
    grid = {"session": sess, "trend_min": (0.3, 0.5, 1.0), "pullback_min": (0.2, 0.3, 0.5),
            "pullback_max": (1.5, 3.0), "stop_buffer": (0.1, 0.2, 0.4),
            "target_r": (1.0, 1.5, 2.0, 3.0), "max_hold_minutes": (60, 120, 240),
            "entry": ("limit", "market")}
    return Spec("pullback", asset, PullbackConfig(asset=asset, session=sess[0]), grid,
                prepare=lambda m1: m1, run=lambda c, d: run_backtest(d, c)["trades"])


def _orb2r(asset: str) -> Spec:
    from .strategies.orb_2r import MARKETS, OrbConfig, build_sessions, run_backtest
    base = MARKETS.get(asset, OrbConfig(asset=asset, asset_label=asset))
    grid = {"range_minutes": (5, 15, 30, 60), "signal_window_minutes": (60, 120, 240),
            "target_r": (1.0, 1.5, 2.0, 3.0, 4.0), "stop_buffer_spreads": (0.0, 1.0, 2.0),
            "max_cost_r": (0.03, 0.05, 0.08), "trend_sma_days": (0, 20),
            "range_atr_median_days": (0, 20)}
    return Spec("orb_2r", asset, base, grid,
                prepare=lambda m1: build_sessions(m1, base, 1.0),
                run=lambda c, d: run_backtest(d, c)["trades"])


def _pos_spec(name: str, asset: str, base, grid: dict, run, tf: str, *,
              fills_at_close: bool = False, charges_spread: bool = True) -> Spec:
    """``fills_at_close``: the strategy enters at the close of the bar it stamps
    with the bar's open time -> the fill is one bar later than ``entry_time``.
    ``charges_spread=False``: the engine trades the bid only (no ask side), so
    the optimiser charges the measured spread itself."""
    from .mr_common import bar_minutes
    return Spec(name, asset, base, grid, prepare=lambda m1: resample_ohlc(m1, tf),
                run=lambda c, d: run(d, c)["trades"], risk_pct=base.risk_per_trade * 100.0,
                entry_offset_min=bar_minutes(tf) if fills_at_close else 0,
                charges_spread=charges_spread)


def _mean_reversion(asset: str) -> Spec:
    from .strategies.mean_reversion import MeanReversionConfig, run_backtest
    base = MeanReversionConfig(asset=asset, asset_label=asset)
    grid = {"z_entry_tier_a": (1.8, 2.2, 2.6, 3.0), "z_entry_tier_b": (1.2, 1.6, 2.0),
            "adx_block_threshold": (20.0, 30.0, 40.0, 50.0), "sl_atr_mult": (0.8, 1.5, 2.2, 3.0),
            "max_hold_bars": (4, 12, 24, 48), "z_period": (20, 40, 80),
            "session_filter_enabled": (True, False)}
    return _pos_spec("mean_reversion", asset, base, grid, run_backtest, base.timeframe)


def _session_range_fade(asset: str) -> Spec:
    from .strategies.session_range_fade import SessionRangeFadeConfig, run_backtest
    base = SessionRangeFadeConfig(asset=asset, asset_label=asset)
    grid = {"range_min_atr_mult": (0.5, 1.0, 1.5, 2.5), "range_max_atr_mult": (2.0, 4.0, 6.0, 8.0),
            "sl_buffer_atr_mult": (0.0, 0.3, 0.7, 1.2), "trend_day_atr_mult": (0.5, 1.0, 2.0, 3.0),
            "tp1_r": (1.0, 1.5, 2.0)}
    return _pos_spec("session_range_fade", asset, base, grid, run_backtest, base.timeframe)


def _bb_squeeze(asset: str) -> Spec:
    from .strategies.bb_squeeze_breakout import BBSqueezeBreakoutConfig, run_backtest
    base = BBSqueezeBreakoutConfig(asset=asset, asset_label=asset)
    grid = {"squeeze_threshold": (0.1, 0.2, 0.35), "bb_std": (1.5, 2.0, 2.5),
            "ema_trend_period": (20, 50, 100), "use_trend_filter": (True, False),
            "activity_mult": (0.6, 1.0, 1.5), "sl_atr_mult": (1.0, 1.6, 2.4),
            "tp1_r": (1.5, 2.2, 3.0), "max_hold_hours": (12, 24, 48)}
    return _pos_spec("bb_squeeze", asset, base, grid, run_backtest, "1h", fills_at_close=True,
                     charges_spread=False)


def _supertrend(asset: str) -> Spec:
    from .strategies.supertrend_volume import SupertrendVolumeConfig, run_backtest
    base = SupertrendVolumeConfig(asset=asset, asset_label=asset)
    grid = {"supertrend_period": (7, 10, 14), "supertrend_mult": (1.4, 2.0, 3.0),
            "volume_mult": (0.8, 1.0, 1.3), "ema_fast": (9, 21, 50), "ema_slow": (50, 100, 150),
            "sl_atr_mult": (1.0, 1.6, 2.4), "tp1_r": (1.5, 2.5, 3.5),
            "max_hold_hours": (8, 24, 48), "pullback_entry": (True, False)}
    return _pos_spec("supertrend", asset, base, grid, run_backtest, base.timeframe,
                     charges_spread=False)   # bid only; its 2 bp slippage stays on top


def _orb_classic(asset: str) -> Spec:
    from .strategies.opening_range_breakout import OpeningRangeBreakoutConfig, run_backtest
    base = OpeningRangeBreakoutConfig(asset=asset, asset_label=asset)
    grid = {"orb_minutes": (5, 15, 30), "max_orb_pct": (0.004, 0.008, 0.015),
            "tp_range_mult": (0.5, 1.0, 1.5, 2.0), "sl_buffer": (0.0, 2.0, 5.0),
            "breakout_buffer": (0.0, 2.0, 5.0), "long_only": (True, False)}
    return _pos_spec("orb_classic", asset, base, grid, run_backtest, "5min",
                     charges_spread=False)   # stamps the bar CLOSE already -> no offset


def _smc_ict2022(asset: str) -> Spec:
    """S-001 (docs/specs/smc_ict2022.yaml): the grid is exactly the agreed variant grid."""
    from .strategies import smc_ict2022 as M
    grid = {"zeitrahmen": ("1h/5m", "4h/15m"), "einstieg": ("ce", "fvg_kante"),
            "min_rr": (1.5, 2.0, 3.0), "ziel": ("liquiditaet", "fest_3r"),
            "setup": ("fvg", "ifvg"), "bias": ("an", "aus")}
    return Spec("smc_ict2022", asset, M.SmcIct2022Config(asset=asset), grid,
                prepare=lambda m1: M.prepare(m1, asset),
                run=lambda c, d: M.run_backtest(d, c)["trades"])


REGISTRY = {
    "momentum": (_momentum, ASSETS),
    "pullback": (_pullback, ASSETS),
    "orb_2r": (_orb2r, ("US100", "SP500", "GER40", "XAUUSD")),
    "mean_reversion": (_mean_reversion, ASSETS),
    "session_range_fade": (_session_range_fade, ("XAUUSD", "EURUSD", "USDJPY")),
    "bb_squeeze": (_bb_squeeze, ASSETS),
    "supertrend": (_supertrend, ASSETS),
    "orb_classic": (_orb_classic, ("US100", "SP500")),
    "smc_ict2022": (_smc_ict2022, ("US100", "SP500", "GER40", "XAUUSD", "EURUSD")),
}

# Slow engines get a smaller search (seconds per run measured on US100 train):
# pullback ~19 s, orb_classic ~4 s, momentum ~1.5 s, the rest < 1 s.
BUDGET_SCALE = {"pullback": 0.4, "orb_classic": 0.7}
SHORTLIST = {"pullback": (3, 2)}            # (top by net LCB, top by gross LCB)

# Not adapted, with the reason (reported, never silently dropped):
NOT_ADAPTED = {
    "smc_ict": "legacy engine on the old structure code - superseded by smc_ict2022 (S-001)",
    "silver_bullet": "15m+1m engine without a search space; rejected family (SMC_ICT_PNL_RESEARCH)",
    "trend_range_rejection": "1m engine without a search space, no prior positive result",
    "regime_hybrid": "a combination of supertrend + mean_reversion, both run here on their own",
    "orb_rejection_ifvg": "REJECTED after the day-slicing fix on 11 markets (docs/strategies)",
    "pair_spread": "needs two instruments; rejected (gross PF ~1.0)",
}


#: Plain-language description shown by the guided check.
DESCRIPTIONS = {
    "momentum": "Trendfolge intraday: steigt (faellt) der Kurs ueber ein Hoch (Tief) bzw. mit Schwung, "
                "wird in Trendrichtung eingestiegen. Filter: Trendstaerke (ER/ADX), Session.",
    "pullback": "Ruecksetzer im Trend: im Aufwaertstrend wird ein kurzer Ruecklauf gekauft (Abwaertstrend "
                "spiegelbildlich), Stop hinter dem Ruecklauf-Extrem.",
    "orb_2r": "Opening-Range-Breakout: die ersten Minuten der Boersensitzung definieren eine Spanne, "
              "der Ausbruch daraus wird gehandelt, Ziel ein festes Vielfaches des Risikos.",
    "mean_reversion": "Rueckkehr zum Mittelwert: weicht der Kurs stark (z-Score) vom Mittel ab und ist "
                      "kein starker Trend (ADX), wird gegen die Bewegung gehandelt.",
    "session_range_fade": "Asien-Spanne fade: Ausbrueche aus der Asien-Spanne in London werden "
                          "zurueckverkauft, solange der Tag kein Trendtag ist.",
    "bb_squeeze": "Bollinger-Squeeze-Ausbruch: nach einer Phase enger Baender wird der Ausbruch "
                  "mit Trend- und Aktivitaetsfilter gehandelt.",
    "supertrend": "Supertrend-Wechsel mit Volumen- und EMA-Filter, optional Einstieg im Ruecksetzer.",
    "orb_classic": "Klassischer Opening-Range-Breakout auf 5-Minuten-Kerzen, Ziel als Vielfaches der Spanne.",
    "smc_ict2022": "ICT 2022 (Spec S-001): Sweep in der Killzone, MSS mit Displacement, Einstieg im FVG, "
                   "Stop hinter dem Sweep, Ziele an interner/externer Liquidität.",
}
