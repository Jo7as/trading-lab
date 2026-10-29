"""VPIN as a RISK filter (not a direction signal): does high VPIN warn of big moves / jumps in
the next hours, beyond what recent volatility already says? Measurement only, 2024 (train).

Claim to test: Easley, López de Prado & O'Hara (2012) and Kitvanitphasu et al. (2026, BTC):
VPIN predicts toxicity-induced volatility and price jumps. For a prop account that would be a
way to cut size before the days that break the daily-loss limit.

Design (fixed before the run): VPIN with V = average daily notional of January 2024 / 50 and
n = 50 (the paper's default, one day of volume), both classifications (taker side, BVC).
Sampled at every full hour. Targets: |log return| and realised volatility of the next 1 h / 4 h,
and a "jump" = next-1h |return| > 3 x the std of hourly returns over the previous 7 days.
Control: realised volatility of the past hour and past 24 h - VPIN only counts if it adds to
them. Regression with Newey-West errors (lags = horizon in hours).

    uv run python studies/crypto_vpin_risk_filter.py
"""
from __future__ import annotations

import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from trading_lab import orderflow as of  # noqa: E402
from trading_lab.adapters import crypto_trades as ct  # noqa: E402
from trading_lab.hac import newey_west_ols  # noqa: E402

START, END = date(2024, 1, 1), date(2024, 12, 31)


def hourly_panel(bars: pd.DataFrame, vpin_aggr: pd.Series, vpin_bvc: pd.Series) -> pd.DataFrame:
    """One row per full hour t: features known at t, targets over (t, t + h].

    Features use bars with labels < t only (the bar labelled t starts at t), so nothing from
    the target window leaks in.
    """
    px = bars["close"].ffill()
    r10 = np.log(px).diff()
    step = bars.index[1] - bars.index[0]
    per_h = int(pd.Timedelta("1h") / step)
    rv_1h = r10.rolling(per_h).std() * np.sqrt(per_h) * 1e4            # bp per hour
    rv_24h = r10.rolling(24 * per_h).std() * np.sqrt(per_h) * 1e4
    hours = bars.index[(bars.index.minute == 0) & (bars.index.second == 0)][25:]
    known = hours - step                                                 # last bar before t
    p0 = px.reindex(known).to_numpy()
    out = pd.DataFrame({
        "vpin": vpin_aggr.reindex(known).to_numpy(),
        "vpin_bvc": vpin_bvc.reindex(known).to_numpy(),
        "rv_1h": rv_1h.reindex(known).to_numpy(),
        "rv_24h": rv_24h.reindex(known).to_numpy(),
    }, index=hours)
    for h in (1, 4):
        p1 = px.reindex(known + pd.Timedelta(hours=h)).to_numpy()
        out[f"absret_{h}h"] = np.abs(np.log(p1 / p0)) * 1e4
        fut = r10.rolling(h * per_h).std().shift(-h * per_h) * np.sqrt(per_h) * 1e4
        out[f"rv_next_{h}h"] = fut.reindex(known + step).to_numpy()
    hourly_ret = np.log(px.reindex(known).to_numpy()[1:] / px.reindex(known).to_numpy()[:-1]) * 1e4
    sd7 = pd.Series(np.r_[np.nan, hourly_ret], index=hours).rolling(24 * 7, min_periods=48).std().shift(1)
    out["jump_1h"] = (out["absret_1h"] > 3 * sd7).astype(float).where(sd7.notna())
    return out


def analyse(panel: pd.DataFrame, col: str) -> list[str]:
    p = panel.dropna()
    lines = []
    for target, lags in (("absret_1h", 1), ("absret_4h", 4), ("rv_next_1h", 1), ("rv_next_4h", 4)):
        z = lambda x: (x - x.mean()) / x.std()                          # noqa: E731
        res = newey_west_ols(p[target], np.column_stack([z(p["rv_1h"]), z(p["rv_24h"]), z(p[col])]), lags)
        base = newey_west_ols(p[target], np.column_stack([z(p["rv_1h"]), z(p["rv_24h"])]), lags)
        lines.append(f"| {target} | {res.coef[3]:+.2f} | {res.t[3]:+.2f} | {base.r2 * 100:.1f} % | "
                     f"{res.r2 * 100:.1f} % |")
    return lines


def quintiles(panel: pd.DataFrame, col: str) -> list[str]:
    p = panel.dropna()
    q = pd.qcut(p[col], 5, labels=False)
    g = p.groupby(q)
    lines = []
    for k, grp in g:
        lines.append(f"| Q{k + 1} | {grp[col].mean():.3f} | {grp['absret_1h'].mean():.1f} | "
                     f"{grp['absret_4h'].mean():.1f} | {grp['jump_1h'].mean() * 100:.2f} % | {len(grp)} |")
    # within volatility terciles: does VPIN still sort the jumps?
    rv_t = pd.qcut(p["rv_24h"], 3, labels=["low", "mid", "high"])
    hi_v = p[col] >= p[col].quantile(0.8)
    lines.append("")
    lines.append("| past 24h vol | jump rate, VPIN top 20 % | jump rate, rest | abs ret 4h top 20 % | rest |")
    lines.append("|---|---|---|---|---|")
    for t in ["low", "mid", "high"]:
        m = rv_t == t
        lines.append(f"| {t} | {p['jump_1h'][m & hi_v].mean() * 100:.2f} % | {p['jump_1h'][m & ~hi_v].mean() * 100:.2f} % | "
                     f"{p['absret_4h'][m & hi_v].mean():.1f} | {p['absret_4h'][m & ~hi_v].mean():.1f} |")
    return lines


def main() -> None:
    lines = ["# VPIN as a risk filter (2024, Binance USD-M)", "",
             "Coefficients per 1 std of VPIN, controlling for past 1h and 24h realised volatility "
             "(all standardised); bp. Newey-West t.", ""]
    for market in ("BTC", "SOL"):
        bars = ct.load_bars("binance_um", market, START, END, allow_download=False)
        jan = bars.loc[:"2024-01-31"]
        bucket = float((jan["buy_notional"] + jan["sell_notional"]).sum() / 31 / 50)
        va = of.vpin(bars, bucket, 50)
        vb = of.vpin(bars, bucket, 50, classification="bvc")
        panel = hourly_panel(bars, va, vb)
        for col, name in (("vpin", "taker side"), ("vpin_bvc", "BVC (paper)")):
            lines += [f"## {market} - VPIN {name}", "",
                      "| target | VPIN coef bp | t | R² vol only | R² + VPIN |", "|---|---|---|---|---|",
                      *analyse(panel, col), "",
                      "| VPIN quintile | mean VPIN | abs ret 1h | abs ret 4h | jump rate 1h | n |",
                      "|---|---|---|---|---|---|", *quintiles(panel, col), ""]
    text = "\n".join(lines)
    out = ct.CACHE.parent.parent / "crypto" / "vpin_risk_filter.md"
    out.write_text(text)
    print(text)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
