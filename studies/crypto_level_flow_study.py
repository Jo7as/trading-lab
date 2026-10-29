"""H8: SMC levels x order flow on crypto (docs/research/CRYPTO_ORDERFLOW_PLAN.md, phase 4 step 9).
Measurement only, no strategy, train data (2024) only.

Levels (``trading_lab/level_flow.py``): previous day / week high-low (UTC), external swings,
EQH/EQL pools, order blocks (15-min structure from ``market_structure``), previous-day POC /
VAH / VAL. Event = first touch after the level is known; reaction window 5 min; labels
(pre-registered thresholds, ``level_flow.FLOW_THR`` / ``BREAK_ATR``):

    absorption  strong flow into the level, close back      -> fade      (flow + price)
    reclaim     no strong flow, close back                  -> fade      (price only = SMC)
    cascade     strong flow into the level, close through   -> follow    (flow + price)
    break       no strong flow, close through               -> follow    (price only)

Questions: (1) is absorption better than reclaim, cascade better than break - does flow add
anything? (2) is any of it better than the same rule at a random level (placebo, 3 seeds)?
(3) killzones vs rest. Outcomes signed by the label's trade, from the end of the window, in bp;
t-statistics clustered by day; net = gross - Bybit taker round trip.

    uv run python studies/crypto_level_flow_study.py [--venues binance_um bybit_linear]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from trading_lab import costs  # noqa: E402
from trading_lab import level_flow as lf  # noqa: E402
from trading_lab import orderflow as of  # noqa: E402
from trading_lab.adapters import crypto_trades as ct  # noqa: E402
from trading_lab.indicators import atr as atr_fn  # noqa: E402

START, END = date(2024, 1, 1), date(2024, 12, 31)
BIN = {"BTC": 25.0, "SOL": 0.10}
HORIZONS = {"1h": "1h", "4h": "4h"}
GROUPS = {"pdh": "day", "pdl": "day", "pwh": "week", "pwl": "week", "swing_high": "swing",
          "swing_low": "swing", "eqh": "eq", "eql": "eq", "ob_bear": "ob", "ob_bull": "ob",
          "poc": "profile", "vah": "profile", "val": "profile"}
NY = ZoneInfo("America/New_York")
LABELS = ["absorption", "reclaim", "cascade", "break"]


def killzone(ts: pd.Series) -> pd.Series:
    """Asia open 00-03 UTC, London 07-10 UTC, New York 08:30-11:30 local (DST-aware), else rest."""
    utc_h = ts.dt.hour + ts.dt.minute / 60
    ny = ts.dt.tz_convert(NY)
    ny_h = ny.dt.hour + ny.dt.minute / 60
    return pd.Series(np.select([(ny_h >= 8.5) & (ny_h < 11.5), (utc_h >= 7) & (utc_h < 10), utc_h < 3],
                               ["ny", "london", "asia"], default="rest"), index=ts.index)


def day_t(x: pd.Series, ts: pd.Series) -> float:
    per_day = x.groupby(ts.dt.floor("1D")).mean()
    if len(per_day) < 3 or per_day.std(ddof=1) == 0:
        return float("nan")
    return float(per_day.mean() / (per_day.std(ddof=1) / np.sqrt(len(per_day))))


def load_flow(market: str, venues: list[str]) -> pd.DataFrame:
    per = {v: ct.load_bars(v, market, START, END, allow_download=False) for v in venues}
    return per[venues[0]] if len(venues) == 1 else of.combine_venues(per)


def events_for(market: str, venues: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = load_flow(market, venues)
    m15 = of.resample(bars, "15min").dropna(subset=["close"])
    a15 = atr_fn(m15["high"], m15["low"], m15["close"], 14)
    a15.index = a15.index + pd.Timedelta("15min")            # known at the end of the bar
    atr10 = a15.reindex(bars.index, method="ffill")
    levels = (lf.period_levels(bars, "day", "pd") + lf.period_levels(bars, "week", "pw")
              + lf.structure_levels(m15) + lf.profile_levels_daily(bars, BIN[market]))
    real = lf.classify(lf.touch_events(bars, levels, atr10, horizons=HORIZONS))
    ctl = pd.concat([lf.classify(lf.touch_events(bars, lf.control_levels(levels, seed=s), atr10,
                                                 horizons=HORIZONS)) for s in (1, 2, 3)],
                    ignore_index=True)
    for df in (real, ctl):
        df["market"] = market
        df["group"] = df["kind"].map(GROUPS)
        df["zone"] = killzone(df["ts"])
        for h in HORIZONS:
            df[f"sig_{h}"] = df["trade"] * df[f"ret_{h}"]
    return real, ctl


def row(df: pd.DataFrame, h: str, cost: float) -> str:
    if len(df) < 10:
        return f"{len(df)} | - | - | -"
    x = df[f"sig_{h}"]
    return f"{len(df)} | {x.mean():+.1f} | {day_t(x, df['ts']):+.2f} | {x.mean() - cost:+.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--venues", nargs="+", default=["binance_um"])
    a = ap.parse_args()
    reals, ctls = [], []
    for market in ("BTC", "SOL"):
        r, c = events_for(market, a.venues)
        reals.append(r)
        ctls.append(c)
    real, ctl = pd.concat(reals, ignore_index=True), pd.concat(ctls, ignore_index=True)
    cost = {m: costs.crypto_round_trip_bp(m, "bybit_perp") for m in ("BTC", "SOL")}
    avg_cost = float(np.mean(list(cost.values())))
    lines = [f"# H8 levels x order flow (2024, flow: {' + '.join(a.venues)}, prices: {a.venues[0]})", "",
             f"Signed bp per event from the end of the 5-min window; t clustered by day; net = gross - "
             f"~{avg_cost:.1f} bp taker round trip. Control = same levels moved 0.15-0.5 %, 3 seeds.", ""]
    for h in HORIZONS:
        lines += [f"## All levels, horizon {h}", "",
                  "| market | label | n | gross bp | t | net bp | control n | control gross | control t |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for market in ("BTC", "SOL", "both"):
            for lab in LABELS:
                rr = real[(real["label"] == lab) & ((real["market"] == market) | (market == "both"))]
                cc = ctl[(ctl["label"] == lab) & ((ctl["market"] == market) | (market == "both"))]
                c_cost = avg_cost if market == "both" else cost[market]
                cr = row(cc, h, c_cost).split(" | ")
                lines.append(f"| {market} | {lab} | {row(rr, h, c_cost)} | {cr[0]} | {cr[1]} | {cr[2]} |")
        lines.append("")
    h = "4h"
    lines += [f"## By level group, both markets, {h}", "",
              "| group | label | n | gross bp | t | net bp | control gross |", "|---|---|---|---|---|---|---|"]
    for grp in ["day", "week", "swing", "eq", "ob", "profile"]:
        for lab in LABELS:
            rr = real[(real["group"] == grp) & (real["label"] == lab)]
            cc = ctl[(ctl["group"] == grp) & (ctl["label"] == lab)]
            cg = f"{cc[f'sig_{h}'].mean():+.1f}" if len(cc) >= 10 else "-"
            lines.append(f"| {grp} | {lab} | {row(rr, h, avg_cost)} | {cg} |")
    lines += ["", f"## By killzone, both markets, {h}", "",
              "| zone | label | n | gross bp | t | net bp | control gross |", "|---|---|---|---|---|---|---|"]
    for zone in ["asia", "london", "ny", "rest"]:
        for lab in LABELS:
            rr = real[(real["zone"] == zone) & (real["label"] == lab)]
            cc = ctl[(ctl["zone"] == zone) & (ctl["label"] == lab)]
            cg = f"{cc[f'sig_{h}'].mean():+.1f}" if len(cc) >= 10 else "-"
            lines.append(f"| {zone} | {lab} | {row(rr, h, avg_cost)} | {cg} |")
    lines += ["", f"Events: real {len(real)}, control {len(ctl)} (3 seeds). Label shares real: "
              + ", ".join(f"{k} {v:.0%}" for k, v in real['label'].value_counts(normalize=True).items())]
    text = "\n".join(lines)
    out = ct.CACHE.parent.parent / "crypto" / "level_flow_study.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    real.to_pickle(out.with_suffix(".events.pkl"))
    print(text)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
