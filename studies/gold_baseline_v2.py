"""Gold Supertrend baseline, FIXED engine (next-open entry, slippage, gap stops,
time stop in bars) on the audit-correct frame — before/after the code review."""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "studies")
from pass_speed import load_cached
from trading_lab.data_audit import build_study_frame
from trading_lab.strategies.supertrend_volume import SupertrendVolumeConfig, run_backtest

BASE = dict(supertrend_period=7, supertrend_mult=1.5, volume_ma_period=20, volume_mult=1.0,
            ema_fast=3, ema_slow=3, atr_period=14, min_atr_pct=0.0, pullback_entry=False,
            account_size=10000.0, risk_per_trade=0.01, sl_atr_mult=1.5, tp1_r=2.0,
            tp1_close_pct=0.60, max_hold_hours=48, max_daily_loss=0.03, cooldown_hours=0,
            max_concurrent_positions=1, long_only=False, timeframe="1h")

df, info = build_study_frame(load_cached("XAUUSD"), "XAUUSD")
h = df.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                           "volume": "sum", "ask_close": "last"}).dropna()
t = pd.DataFrame(run_backtest(h, SupertrendVolumeConfig(**BASE))["trades"])
half = ((df["ask_close"] - df["close"]) / 2.0).to_numpy(); idx = df.index.as_unit("ns").asi8
def h_at(sec):
    p = np.searchsorted(idx, (sec * 1e9).astype("int64"), "left")
    return half[np.clip(p, 0, len(idx) - 1)]
he, hx = h_at(t["entry_time"].astype(float).to_numpy()), h_at(t["exit_time"].astype(float).to_numpy())
taker = ~t["exit_reason"].astype(str).str.lower().str.contains("tp")
t["cost"] = (he + np.where(taker, hx, 0.0)) / t["entry_price"].astype(float) * t["size"] * 100
gross = float(t["pnl_pct"].astype(float).sum())
t["year"] = pd.to_datetime(t["entry_time"], unit="s", utc=True).dt.year
print(f"XAUUSD 1h, FIXED engine, audit-correct frame  (dropped rows {info['dropped']}, thin {info['thin_days']})")
print(f"  trades {len(t)}  ({len(t)/((h.index[-1]-h.index[0]).days/7):.1f}/week)  gross {gross:+.1f}%  "
      f"cost {t['cost'].sum():.1f}%  net(0x) {gross-t['cost'].sum():+.1f}%  "
      f"net(0.5x) {gross-1.5*t['cost'].sum():+.1f}%")
print("  exit reasons:", t["exit_reason"].value_counts().to_dict())
print("  per year (net at 0.5x):")
for y, gg in t.groupby("year"):
    net = gg["pnl_pct"].astype(float).sum() - gg["cost"].sum() * 1.5
    print(f"    {y}: n={len(gg):4d}  net {net:+7.1f}%")
