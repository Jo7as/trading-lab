#!/usr/bin/env python3
"""SMC nested-entry funnel: count how many events die at each gate (EUR/USD)."""
from __future__ import annotations

import sys as _sys, pathlib as _path
_sys.path.insert(0, str(_path.Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from trading_lab.adapters.dukascopy_adapter import DukascopyAdapter
from trading_lab.indicators import atr as atr_indicator
from trading_lab.risk import RiskManager
from trading_lab.strategies.smc.liquidity import LiquidityEngine
from trading_lab.strategies.smc.state_machine import StateMachine
from trading_lab.strategies.smc.structure import MarketStructureEngine
from trading_lab.strategies.smc.supply_demand import SupplyDemandEngine
from trading_lab.strategies.smc.types import PendingEntry, TradeState
from trading_lab.strategies.smc_ict import (
    SmcIctConfig,
    _advance_closed_ts,
    _bar_close_ts,
    _entry_zone_1m,
    _limit_and_sl,
    _rr_targets,
    _TF_DELTA,
    _unix_seconds,
    manage_position,
)
from trading_lab.backtest import _pos_to_dict


@dataclass
class FunnelCounts:
    bars_1m: int = 0
    bars_with_4h_bias: int = 0
    # unconstrained reference (like OB research)
    raw_1m_choch_any: int = 0
    # pipeline
    to_htf_aligned: int = 0
    sweep_ticks_15m: int = 0
    sweep_ticks_1m: int = 0
    sweep_accepted: int = 0
    to_wait_for_choch: int = 0  # 15m align/ChoCh success
    bars_watching_1m: int = 0
    choch_while_watching_any: int = 0
    choch_matched_bias: int = 0
    entry_zone_missing: int = 0
    pending_placed: int = 0
    pending_timeout: int = 0
    pending_invalidated: int = 0
    pending_risk_blocked: int = 0
    pending_long_only_blocked: int = 0
    fills: int = 0
    trades_closed: int = 0
    by_side_fills: dict = field(default_factory=lambda: {"long": 0, "short": 0})
    exit_reasons: dict = field(default_factory=dict)


def run_funnel(
    df_4h, df_15m, df_1m, cfg: SmcIctConfig
) -> tuple[FunnelCounts, dict]:
    c = FunnelCounts()

    for name, df in (("4h", df_4h), ("15m", df_15m), ("1m", df_1m)):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

    df_1m = df_1m.copy()
    if "volume" not in df_1m.columns:
        df_1m["volume"] = 0.0
    df_1m["atr"] = atr_indicator(
        df_1m["high"], df_1m["low"], df_1m["close"], cfg.atr_period
    )

    eng_4h = MarketStructureEngine(cfg.swing_length_4h)
    eng_15m = MarketStructureEngine(cfg.swing_length_15m)
    eng_1m = MarketStructureEngine(cfg.swing_length_1m)
    eng_4h.precompute(df_4h)
    eng_15m.precompute(df_15m)
    eng_1m.precompute(df_1m)

    # Unconstrained 1m ChoCh count (research reference)
    for i in range(max(cfg.swing_length_1m * 2 + 5, 20), len(df_1m)):
        d = eng_1m.snapshot_at(df_1m, i).last_choch_dir
        if d != 0:
            c.raw_1m_choch_any += 1

    liq = LiquidityEngine(
        asia_start=cfg.asia_start,
        asia_end=cfg.asia_end,
        london_start=cfg.london_start,
        london_end=cfg.london_end,
        use_asia=cfg.use_asia,
        use_london=cfg.use_london,
        use_pd=cfg.use_pd,
        use_pw=cfg.use_pw,
    )
    liq.build(df_15m)

    sd_15m = SupplyDemandEngine(
        ob_require_fvg=cfg.ob_require_fvg,
        fvg_mitigation=cfg.fvg_mitigation,
        max_zones=cfg.max_active_zones,
        ob_lookback=cfg.ob_lookback,
    )

    sm = StateMachine()
    rm = RiskManager(cfg, account_size=cfg.account_size, backtest_mode=True)  # type: ignore[arg-type]
    open_pos = None
    trades: list[dict] = []

    delta_4h = _TF_DELTA[cfg.tf_bias]
    delta_15m = _TF_DELTA[cfg.tf_poi]
    last_4h_i = -1
    last_15m_i = -1
    snap_4h = None
    snap_15m = None

    closes_1m = df_1m["close"].to_numpy(dtype=float)
    highs_1m = df_1m["high"].to_numpy(dtype=float)
    lows_1m = df_1m["low"].to_numpy(dtype=float)
    atrs_1m = df_1m["atr"].to_numpy(dtype=float)
    times_1m = df_1m.index
    ts_1m = _unix_seconds(times_1m)
    close_ts_4h = _bar_close_ts(df_4h.index, delta_4h)
    close_ts_15m = _bar_close_ts(df_15m.index, delta_15m)
    highs_15m = df_15m["high"].to_numpy(dtype=float)
    lows_15m = df_15m["low"].to_numpy(dtype=float)

    n = len(df_1m)
    warmup = max(cfg.atr_period + 5, cfg.swing_length_1m * 2 + 5)
    c.bars_1m = n - warmup
    prev_state = sm.ctx.state

    for i in range(warmup, n):
        now = times_1m[i]
        current_ts = float(ts_1m[i])
        price = float(closes_1m[i])
        high = float(highs_1m[i])
        low = float(lows_1m[i])
        atr_val = float(atrs_1m[i]) if not np.isnan(atrs_1m[i]) else 0.0

        if open_pos is not None:
            manage_position(
                open_pos, i, current_ts, price, high, low, cfg, rm, trades
            )
            if open_pos.status == "closed":
                reason = open_pos.exit_reason or "closed"
                c.exit_reasons[reason] = c.exit_reasons.get(reason, 0) + 1
                c.trades_closed += 1
                open_pos = None
                sm.on_trade_closed()

        i4 = _advance_closed_ts(close_ts_4h, current_ts, last_4h_i)
        i15 = _advance_closed_ts(close_ts_15m, current_ts, last_15m_i)

        if i4 >= 0 and i4 != last_4h_i:
            last_4h_i = i4
            snap_4h = eng_4h.snapshot_at(df_4h, i4)
        if i15 >= 0 and i15 != last_15m_i:
            last_15m_i = i15
            snap_15m = eng_15m.snapshot_at(df_15m, i15)
            sd_15m.update_through(df_15m, i15)
            for sw in liq.detect_sweeps(
                float(highs_15m[i15]), float(lows_15m[i15]), df_15m.index[i15]
            ):
                c.sweep_ticks_15m += 1
                if sm.on_sweep(sw.kind, 0):
                    c.sweep_accepted += 1

        if i4 < 0 or snap_4h is None:
            prev_state = sm.ctx.state
            continue

        sm.on_4h_bias(snap_4h.bias)
        if sm.ctx.bias.value != 0:
            c.bars_with_4h_bias += 1
        if prev_state != TradeState.HTF_ALIGNED and sm.ctx.state == TradeState.HTF_ALIGNED:
            c.to_htf_aligned += 1

        for sw in liq.detect_sweeps(high, low, now):
            c.sweep_ticks_1m += 1
            if sm.on_sweep(sw.kind, 0):
                c.sweep_accepted += 1

        if (
            sm.ctx.recent_sweep
            and sm.ctx.state
            in (TradeState.SWEPT, TradeState.IN_POI, TradeState.WAIT_FOR_CHOCH)
            and snap_15m is not None
            and i15 >= 0
        ):
            want = int(sm.ctx.bias)
            poi = sd_15m.zone_touched(
                float(highs_15m[i15]),
                float(lows_15m[i15]),
                direction=want,
            )
            if poi is None:
                zones = sd_15m.active_zones(direction=want)
                poi = zones[-1] if zones else None
            before = sm.ctx.state
            if sm.on_15m_align(snap_15m.bias, snap_15m.last_choch_dir, poi):
                if before != TradeState.WAIT_FOR_CHOCH:
                    c.to_wait_for_choch += 1

        if open_pos is None and sm.ctx.state == TradeState.ENTRY_PENDING and sm.ctx.pending:
            pend = sm.ctx.pending
            if i - pend.placed_idx >= pend.timeout_bars:
                c.pending_timeout += 1
                sm.cancel_pending()
            else:
                invalidated = False
                if pend.direction == "long" and price < pend.invalidation_level:
                    invalidated = True
                elif pend.direction == "short" and price > pend.invalidation_level:
                    invalidated = True
                if invalidated:
                    c.pending_invalidated += 1
                    sm.cancel_pending()
                else:
                    filled = low <= pend.limit_price <= high
                    if filled and atr_val > 0:
                        allowed, _ = rm.can_trade(current_ts)
                        if not allowed:
                            c.pending_risk_blocked += 1
                        elif pend.direction == "short" and cfg.long_only:
                            c.pending_long_only_blocked += 1
                            sm.cancel_pending()
                        else:
                            tp1, tp2 = _rr_targets(
                                pend.limit_price, pend.stop_loss, pend.direction, cfg
                            )
                            pos = rm.open_position(
                                entry_price=pend.limit_price,
                                atr=atr_val,
                                direction=pend.direction,
                                entry_time=current_ts,
                                entry_idx=i,
                                stop_loss=pend.stop_loss,
                                tp1=tp1,
                                tp2=tp2,
                            )
                            if pos is not None:
                                open_pos = pos
                                sm.on_fill()
                                c.fills += 1
                                c.by_side_fills[pend.direction] = (
                                    c.by_side_fills.get(pend.direction, 0) + 1
                                )

        if open_pos is None and sm.ctx.state == TradeState.WAIT_FOR_CHOCH:
            c.bars_watching_1m += 1
            snap_1m = eng_1m.snapshot_at(df_1m, i)
            if snap_1m.last_choch_dir != 0:
                c.choch_while_watching_any += 1
            if sm.on_choch(snap_1m.last_choch_dir):
                c.choch_matched_bias += 1
                direction_i = int(sm.ctx.bias)
                entry_z = _entry_zone_1m(df_1m, i, direction_i, cfg)
                if entry_z is None:
                    c.entry_zone_missing += 1
                    sm.cancel_pending()
                else:
                    direction = "long" if direction_i > 0 else "short"
                    swing_sl = (
                        snap_1m.last_swing_low.price
                        if direction == "long" and snap_1m.last_swing_low
                        else (
                            snap_1m.last_swing_high.price
                            if direction == "short" and snap_1m.last_swing_high
                            else (
                                entry_z.merged.bottom
                                if direction == "long"
                                else entry_z.merged.top
                            )
                        )
                    )
                    limit, sl = _limit_and_sl(entry_z, direction, swing_sl, cfg)
                    inv = (
                        snap_1m.last_swing_high.price
                        if direction == "long" and snap_1m.last_swing_high
                        else (
                            snap_1m.last_swing_low.price
                            if direction == "short" and snap_1m.last_swing_low
                            else limit
                        )
                    )
                    if direction == "long" and sl >= limit:
                        sl = limit - max(cfg.sl_buffer, 1e-6)
                    if direction == "short" and sl <= limit:
                        sl = limit + max(cfg.sl_buffer, 1e-6)
                    anchor = entry_z.anchor
                    sm.set_pending(
                        PendingEntry(
                            direction=direction,
                            limit_price=limit,
                            stop_loss=sl,
                            swing_sl_price=swing_sl,
                            zone_top=anchor.top,
                            zone_bottom=anchor.bottom,
                            placed_idx=i,
                            placed_ts=now,
                            timeout_bars=cfg.entry_timeout_bars,
                            invalidation_level=inv,
                        )
                    )
                    c.pending_placed += 1

        prev_state = sm.ctx.state

    if open_pos is not None and open_pos.status != "closed":
        last = df_1m.iloc[-1]
        last_ts = float(ts_1m[-1])
        rm.close_position(open_pos, float(last["close"]), "backtest_end", last_ts)
        trades.append(_pos_to_dict(open_pos))
        c.trades_closed += 1
        c.exit_reasons["backtest_end"] = c.exit_reasons.get("backtest_end", 0) + 1

    summary = {
        "funnel": asdict(c),
        "survival_pct": {
            "raw_1m_choch_to_fill": round(100 * c.fills / max(c.raw_1m_choch_any, 1), 3),
            "wait_armed_to_fill": round(100 * c.fills / max(c.to_wait_for_choch, 1), 2),
            "pending_to_fill": round(100 * c.fills / max(c.pending_placed, 1), 2),
            "choch_match_to_pending": round(
                100 * c.pending_placed / max(c.choch_matched_bias, 1), 2
            ),
        },
    }
    return c, summary


async def main() -> None:
    ap = argparse.ArgumentParser(description="SMC setup funnel counts")
    ap.add_argument("--year-start", type=int, default=2020)
    ap.add_argument("--year-end", type=int, default=2021)
    ap.add_argument("--asset", default="EUR/USD")
    args = ap.parse_args()

    cfg = SmcIctConfig(
        asset=args.asset,
        asset_label=args.asset,
        sl_buffer=0.0008,
        ob_lookback=15,
        entry_limit_mode="fvg_edge",
        ob_range="wick",
    )

    start = datetime(args.year_start, 1, 1, tzinfo=timezone.utc)
    end = datetime(args.year_end, 1, 1, tzinfo=timezone.utc)
    ad = DukascopyAdapter()
    print(f"Fetching {args.asset} {args.year_start}→{args.year_end}...", flush=True)
    df_4h = await ad.fetch_history(args.asset, "4h", start=start, end=end)
    df_15m = await ad.fetch_history(args.asset, "15m", start=start, end=end)
    df_1m = await ad.fetch_history(args.asset, "1m", start=start, end=end)
    print(f"  4h={len(df_4h)} 15m={len(df_15m)} 1m={len(df_1m)}", flush=True)

    print("Running funnel...", flush=True)
    counts, summary = run_funnel(df_4h, df_15m, df_1m, cfg)
    f = summary["funnel"]

    print("\n=== SMC FUNNEL", args.asset, f"{args.year_start}–{args.year_end - 1}", "===")
    print(f"  1m bars (post-warmup):     {f['bars_1m']:,}")
    print(f"  bars with 4h bias:         {f['bars_with_4h_bias']:,}")
    print(f"  → HTF_ALIGNED entries:     {f['to_htf_aligned']:,}")
    print(f"  sweep ticks 15m / 1m:      {f['sweep_ticks_15m']:,} / {f['sweep_ticks_1m']:,}")
    print(f"  sweep ACCEPTED (dir OK):   {f['sweep_accepted']:,}")
    print(f"  → WAIT_FOR_CHOCH (15m):    {f['to_wait_for_choch']:,}")
    print(f"  bars watching 1m:          {f['bars_watching_1m']:,}")
    print(f"  ChoCh while watching:      {f['choch_while_watching_any']:,}")
    print(f"  ChoCh matched 4h bias:     {f['choch_matched_bias']:,}")
    print(f"  entry zone missing:        {f['entry_zone_missing']:,}")
    print(f"  pending limits placed:     {f['pending_placed']:,}")
    print(f"    timeout / invalidated:   {f['pending_timeout']:,} / {f['pending_invalidated']:,}")
    print(f"  FILLS (trades opened):     {f['fills']:,}  {f['by_side_fills']}")
    print(f"  trades closed:             {f['trades_closed']:,}  exits={f['exit_reasons']}")
    print()
    print("  Reference (no HTF filter):")
    print(f"  raw 1m ChoCh any dir:      {f['raw_1m_choch_any']:,}")
    print()
    print("  Survival:")
    for k, v in summary["survival_pct"].items():
        print(f"    {k}: {v}%")

    out = Path("artifacts") / f"smc_funnel_{args.asset.replace('/','')}_{args.year_start}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
