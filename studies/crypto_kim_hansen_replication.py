"""Replication of Kim & Hansen (2026), "The Quarter-Hour Effect" (arXiv 2607.09426), step 1
of phase 4 in docs/research/CRYPTO_ORDERFLOW_PLAN.md. Measurement only, no trading.

Order imbalance OI = (B - S) / (B + S) in the first 10 s after :00/:15/:30/:45 (taker-signed
notional), regressed against the log return from the end of that window over 30 min /
4 h / 8 h / 12 h. Newey-West errors with lags = overlapping quarter hours. Reported in bp
per unit of OI, like the paper's Table A.4 (BTC 3.84 / 5.55 / 6.39 bp at 4 / 8 / 12 h,
SOL 3.29 / 2.41 / 6.73 bp; 2021-01..2024-10, Binance USDT perps).

Checked against the paper PDF (2026-10-29): OI from base-asset trade sizes (Table A.13),
r = log P[t+l] - log P[t] from the last trade of the opening interval (eq. 2), Newey-West.
Remaining differences: (a) openings-only regression instead of the full 10-s panel with a
fifteen-minute interaction (Table A.4 spec) - same slope up to (b) the paper's reversal
controls W_t (aggressor switches of the last trade per interval), which our bars do not
store; (c) our sample is 2024 only (paper: 2021-01..2024-10). Placebo: marks shifted by
2 min (:02/:17/:32/:47).

    uv run python studies/crypto_kim_hansen_replication.py --start 2024-01-01 --end 2024-10-31
    uv run python studies/crypto_kim_hansen_replication.py --venues binance_um bybit_linear
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from trading_lab import orderflow as of  # noqa: E402
from trading_lab.adapters import crypto_trades as ct  # noqa: E402
from trading_lab.hac import newey_west_ols  # noqa: E402

OUT = Path("artifacts/crypto/kim_hansen_replication.md")
HOLDOUT_START = date(2025, 1, 1)
HORIZONS = {"30min": "30min", "4h": "4h", "8h": "8h", "12h": "12h"}
PAPER_A4 = {  # bp per unit OI, Table A.4 (fifteen-minute-only specification)
    "BTC": {"4h": 3.84, "8h": 5.55, "12h": 6.39},
    "SOL": {"4h": 3.29, "8h": 2.41, "12h": 6.73},
}
ROUND_TRIP_BP = 10.0  # Binance/Bybit taker both legs, for the economic sanity line


def cached_until(venue: str, market: str, start: date, end: date, bar: str) -> date | None:
    """Last day of the contiguous cached run from ``start`` (download may still be running)."""
    symbol = ct.SYMBOLS[market][venue]
    day, last = start, None
    while day <= end:
        p = ct._cache_path(ct.CACHE, venue, symbol, day, bar)
        if not (p.exists() or p.with_suffix(".missing").exists()):
            break
        last, day = day, day + timedelta(days=1)
    return last


def load(market: str, venues: list[str], start: date, end: date) -> pd.DataFrame:
    per = {v: ct.load_bars(v, market, start, end, allow_download=False) for v in venues}
    for v, b in per.items():
        if b.attrs["missing_days"]:
            print(f"  {v} {market}: {len(b.attrs['missing_days'])} days without a dump")
    return per[venues[0]] if len(venues) == 1 else of.combine_venues(per)


def measure(bars: pd.DataFrame, *, offset: str = "0s") -> dict:
    # paper: V_k = trade size in the base asset (Table A.13), not notional
    op = of.clock_openings(bars, every="15min", window="10s", offset=offset, by="qty")
    step = bars.index[1] - bars.index[0]
    at = pd.DatetimeIndex(op["known_at"] - step)          # last bar of the opening window
    fr = of.forward_log_returns(bars, at, HORIZONS)
    oi = op["imbalance"].to_numpy()
    rows = {}
    for name, h in HORIZONS.items():
        lags = max(int(pd.Timedelta(h) / pd.Timedelta("15min")) - 1, 0)
        res = newey_west_ols(fr[name].to_numpy() * 1e4, oi, lags=lags)
        rows[name] = {"beta_bp": res.coef[1], "t": res.t[1], "r2": res.r2, "n": res.n}
    return {"rows": rows, "oi": pd.Series(oi, index=op.index), "fwd": fr.set_axis(op.index)}


def economics(m: dict, horizon: str, q: float = 0.9) -> tuple[float, float, int]:
    """Mean sign(OI) x return (bp) for all openings and for the top (1 - q) of |OI|."""
    oi, r = m["oi"], m["fwd"][horizon] * 1e4
    ok = oi.notna() & r.notna()
    oi, r = oi[ok], r[ok]
    allm = float((np.sign(oi) * r).mean())
    top = oi.abs() >= oi.abs().quantile(q)
    return allm, float((np.sign(oi[top]) * r[top]).mean()), int(top.sum())


def monthly_signs(m: dict, horizon: str) -> str:
    oi, r = m["oi"], m["fwd"][horizon] * 1e4
    df = pd.DataFrame({"oi": oi, "r": r}).dropna()
    betas = df.groupby(df.index.tz_localize(None).to_period("M")).apply(
        lambda g: np.polyfit(g["oi"], g["r"], 1)[0] if len(g) > 50 else np.nan, include_groups=False)
    pos = int((betas > 0).sum())
    return f"{pos}/{int(betas.notna().sum())} months beta > 0"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2024, 10, 31))
    ap.add_argument("--markets", nargs="+", default=["BTC", "SOL"])
    ap.add_argument("--venues", nargs="+", default=["binance_um"])
    a = ap.parse_args()
    if a.end >= HOLDOUT_START:
        sys.exit("holdout (2025+) is not opened by studies - see AGENTS.md")

    lines = ["# Kim & Hansen replication (quarter-hour opening imbalance)", "",
             f"Venues: {' + '.join(a.venues)}; requested {a.start} .. {a.end}; "
             "simplified spec (openings only, NW errors). bp per unit OI.", ""]
    for market in a.markets:
        ends = [cached_until(v, market, a.start, a.end, "10s") for v in a.venues]
        if any(e is None for e in ends):
            lines.append(f"## {market}: no cached data yet\n")
            continue
        end = min(ends)
        bars = load(market, a.venues, a.start, end)
        real, placebo = measure(bars), measure(bars, offset="2min")
        lines += [f"## {market}  ({a.start} .. {end}, {real['rows']['4h']['n']} openings)", "",
                  "| horizon | beta bp/OI | t (NW) | R² | paper A.4 | placebo :02 beta | placebo t |",
                  "|---|---|---|---|---|---|---|"]
        for h in HORIZONS:
            r, p = real["rows"][h], placebo["rows"][h]
            ref = PAPER_A4.get(market, {}).get(h)
            lines.append(f"| {h} | {r['beta_bp']:+.2f} | {r['t']:+.2f} | {r['r2'] * 100:.3f} % | "
                         f"{'' if ref is None else f'{ref:+.2f}'} | {p['beta_bp']:+.2f} | {p['t']:+.2f} |")
        oi = real["oi"].dropna()
        lines += ["", f"OI: mean {oi.mean():+.3f}, std {oi.std():.3f}, |OI| 90 % quantile "
                      f"{oi.abs().quantile(0.9):.3f}", ""]
        for h in ("4h", "8h", "12h"):
            allm, topm, ntop = economics(real, h)
            lines.append(f"- {h}: sign(OI) x return = {allm:+.2f} bp (all), {topm:+.2f} bp "
                         f"(top 10 % |OI|, n={ntop}) vs ~{ROUND_TRIP_BP:.0f} bp round trip; "
                         f"{monthly_signs(real, h)}")
        lines.append("")
    text = "\n".join(lines)
    out = OUT if a.venues == ["binance_um"] else OUT.with_name(f"{OUT.stem}_{'+'.join(a.venues)}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
