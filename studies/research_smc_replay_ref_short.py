#!/usr/bin/env python3
"""Replay the Jul-2026 EURUSD reference short against current SMC config."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from trading_lab.adapters.dukascopy_adapter import DukascopyAdapter
from trading_lab.strategies.smc_ict import SmcIctConfig, run_backtest

# User-marked reference levels (Berlin time Jul 16 2026)
# 13:50 low 1.14566 → 14:37 nested 1m ChoCh → FVG proximal fill ~15:30
REF = {
    "sweep_1m_low": 1.14566,
    "choch_1m_bar": "14:37",
    "entry": 1.14593,
    "sl": 1.14653,
}


async def main() -> None:
    ad = DukascopyAdapter()
    start = datetime(2026, 7, 14, tzinfo=timezone.utc)
    end = datetime(2026, 7, 18, tzinfo=timezone.utc)
    print("Fetching EUR/USD Jul 14–18 2026...", flush=True)
    df_4h = await ad.fetch_history("EUR/USD", "4h", start=start, end=end)
    # Need more 4h history for bias — fetch from June
    df_4h_long = await ad.fetch_history(
        "EUR/USD",
        "4h",
        start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        end=end,
    )
    df_15m = await ad.fetch_history("EUR/USD", "15m", start=start, end=end)
    df_1m = await ad.fetch_history("EUR/USD", "1m", start=start, end=end)
    print(
        f"  4h_long={len(df_4h_long)} 15m={len(df_15m)} 1m={len(df_1m)}",
        flush=True,
    )

    cfg = SmcIctConfig(
        asset="EUR/USD",
        asset_label="EUR/USD",
        sl_buffer=0.00005,
        sl_mode="ob",
        entry_limit_mode="fvg_edge",
        entry_edge_offset=0.00005,
        require_1m_fvg=True,
        use_swing_liq=False,
        nested_choch_vs_pullback=True,
        choch_refine_bars=5,
        choch_lookback_bars=90,
        entry_timeout_bars=90,
        ob_lookback=15,
        ob_range="wick",
        cooldown_hours=0,
        max_hold_bars=480,
    )
    r = run_backtest(df_4h_long, df_15m, df_1m, cfg, verbose=False)
    print(f"\nTrades in window: {len(r['trades'])}")
    print(f"Signals: {len(r['signals'])}")
    for s in r["signals"]:
        print("  SIG", s)
    for t in r["trades"]:
        et = datetime.fromtimestamp(float(t["entry_time"]), tz=timezone.utc)
        xt = datetime.fromtimestamp(float(t["exit_time"]), tz=timezone.utc)
        print(
            f"  TR {t.get('direction')} entry={t.get('entry_price')} "
            f"sl={t.get('stop_loss')} "
            f"in={et.astimezone().strftime('%H:%M %Z')} "
            f"out={xt.astimezone().strftime('%H:%M')} "
            f"reason={t.get('exit_reason')} pnl={t.get('pnl_pct')}"
        )

    # Nearest short to reference entry
    shorts = [s for s in r["signals"] if s.get("direction") == "short"]
    if not shorts:
        print("\nNO short signals — reference trade NOT found.")
        print("Reference:", REF)
        return
    best = min(shorts, key=lambda s: abs(float(s["entry"]) - REF["entry"]))
    best_sl = min(shorts, key=lambda s: abs(float(s["sl"]) - REF["sl"]))
    print("\nClosest short to reference entry:")
    print(f"  bot entry={best['entry']}  ref={REF['entry']}  Δ={float(best['entry'])-REF['entry']:.5f}")
    print(f"  bot sl={best['sl']}  ref={REF['sl']}  Δ={float(best['sl'])-REF['sl']:.5f}")
    print(f"  ts={best['ts']}")
    if best is not best_sl:
        print("Closest short to reference SL:")
        print(f"  bot entry={best_sl['entry']}  sl={best_sl['sl']}  ts={best_sl['ts']}")


if __name__ == "__main__":
    asyncio.run(main())
