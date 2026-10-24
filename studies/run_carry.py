#!/usr/bin/env python3
"""H-008 probe: cross-sectional FX carry (price-only, plus estimated swap).

Fetches DAILY bars for a G10+ basket from Dukascopy (cheap), forms a monthly
long-top / short-bottom carry portfolio, and reports price P&L, the estimated
swap accrual, a random-ranking placebo, skew (carry-crash), per-year stability
and the goal gate (2-4 %/month needs +27..+60 %/yr).

  .venv/bin/python studies/run_carry.py
"""
from __future__ import annotations

import datetime as dt
import glob
import pathlib
import sys

import numpy as np
import pandas as pd
import dukascopy_python as duka

sys.path.insert(0, ".")
from trading_lab.carry import CarryConfig, metrics, monthly_returns  # noqa: E402

CACHE = pathlib.Path("artifacts/cache/daily")
PAIRS = {
    "EUR/USD": "INSTRUMENT_FX_MAJORS_EUR_USD", "GBP/USD": "INSTRUMENT_FX_MAJORS_GBP_USD",
    "USD/JPY": "INSTRUMENT_FX_MAJORS_USD_JPY", "AUD/USD": "INSTRUMENT_FX_MAJORS_AUD_USD",
    "NZD/USD": "INSTRUMENT_FX_MAJORS_NZD_USD", "USD/CAD": "INSTRUMENT_FX_MAJORS_USD_CAD",
    "USD/CHF": "INSTRUMENT_FX_MAJORS_USD_CHF",
    "EUR/CHF": "INSTRUMENT_FX_CROSSES_EUR_CHF", "EUR/JPY": "INSTRUMENT_FX_CROSSES_EUR_JPY",
    "EUR/GBP": "INSTRUMENT_FX_CROSSES_EUR_GBP", "EUR/PLN": "INSTRUMENT_FX_CROSSES_EUR_PLN",
    "EUR/HUF": "INSTRUMENT_FX_CROSSES_EUR_HUF", "EUR/CZK": "INSTRUMENT_FX_CROSSES_EUR_CZK",
    "EUR/NOK": "INSTRUMENT_FX_CROSSES_EUR_NOK", "EUR/SEK": "INSTRUMENT_FX_CROSSES_EUR_SEK",
    "GBP/JPY": "INSTRUMENT_FX_CROSSES_GBP_JPY", "GBP/CHF": "INSTRUMENT_FX_CROSSES_GBP_CHF",
    "AUD/JPY": "INSTRUMENT_FX_CROSSES_AUD_JPY", "NZD/JPY": "INSTRUMENT_FX_CROSSES_NZD_JPY",
    "CAD/JPY": "INSTRUMENT_FX_CROSSES_CAD_JPY", "AUD/CHF": "INSTRUMENT_FX_CROSSES_AUD_CHF",
    "NZD/CHF": "INSTRUMENT_FX_CROSSES_NZD_CHF", "AUD/CAD": "INSTRUMENT_FX_CROSSES_AUD_CAD",
    "NZD/CAD": "INSTRUMENT_FX_CROSSES_NZD_CAD", "AUD/NZD": "INSTRUMENT_FX_CROSSES_AUD_NZD",
    "EUR/AUD": "INSTRUMENT_FX_CROSSES_EUR_AUD",
}


def daily(pair: str, const: str) -> pd.Series:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{pair.replace('/', '')}.pkl"
    if f.exists():
        return pd.read_pickle(f)
    from dukascopy_python import instruments as ins
    inst = getattr(ins, const)
    df = duka.fetch(inst, duka.INTERVAL_DAY_1, duka.OFFER_SIDE_BID,
                    dt.datetime(2019, 1, 1), dt.datetime(2026, 10, 1), max_retries=2)
    s = df["close"] if "close" in df else df.iloc[:, -1]
    s.index = pd.to_datetime(s.index, utc=True)
    s.to_pickle(f)
    return s


def main() -> None:
    pairs = {}
    for p, const in PAIRS.items():
        try:
            pairs[p] = daily(p, const)
        except Exception as e:  # noqa: BLE001
            print(f"  {p}: fetch failed ({type(e).__name__})")
    print(f"basket: {len(pairs)} pairs, {list(pairs)[:5]}...")

    base = CarryConfig()
    for label, cfg in (("price only (what our data sees)", CarryConfig(include_swap=False)),
                       ("price + estimated swap", CarryConfig(include_swap=True)),
                       ("price, k=2 (concentrated)", CarryConfig(top_k=2, bottom_k=2)),
                       ("price, k=6 (broad)", CarryConfig(top_k=6, bottom_k=6))):
        m = monthly_returns(pairs, cfg)
        for col in ("price", "swap", "total"):
            print(f"\n{label} — {col}:")
            print("   ", metrics(m[col]))
            if col == "price":
                per_year = m["price"].groupby(m.index.year).apply(lambda s: (1 + s).prod() - 1)
                print("    per year %:", {int(y): round(float(v) * 100, 1) for y, v in per_year.items()})
    return


if __name__ == "__main__":
    main()
