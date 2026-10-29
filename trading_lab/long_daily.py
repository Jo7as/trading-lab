"""Long daily history (yfinance) for slow strategies - S-007 trend following.

Our 1m CFD cache starts in 2019; a 12-month trend signal needs decades. This module
downloads daily OHLC once into ``artifacts/cache/daily/<asset>.csv`` and afterwards reads
only the cache (``load_daily``), so tests and re-runs stay offline. Costs and execution
still come from the CFD data (``docs/specs/tsmom_multi_asset.yaml``).

Bad ticks: yfinance FX has isolated errors (EURUSD / USDJPY 2008-12-08: +17 % in one day,
back the next). Rule of the spec: a day whose |return| exceeds ``n_sigma`` EWMA sigmas and
which the next close takes back by >= ``revert`` is a data error and is replaced by the
previous close. Real shocks (EURCHF 2015-01-15) do not revert and stay. This repair looks
one day ahead by construction - it fixes the data, it is not a signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("artifacts/cache/daily")
TICKERS = {
    "SP500": "^SP500TR", "US100": "^NDX", "US30": "^DJI", "GER40": "^GDAXI", "UK100": "^FTSE", "JP225": "^N225",
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X", "AUDUSD": "AUDUSD=X", "NZDUSD": "NZDUSD=X",
    "USDCAD": "CAD=X", "USDCHF": "CHF=X", "EURJPY": "EURJPY=X", "EURGBP": "EURGBP=X", "EURCHF": "EURCHF=X",
    "GBPJPY": "GBPJPY=X", "AUDJPY": "AUDJPY=X", "AUDNZD": "AUDNZD=X", "AUDCAD": "AUDCAD=X",
    "XAUUSD": "GC=F", "XAGUSD": "SI=F", "BRENT": "BZ=F", "COPPER": "HG=F", "UST10Y": "ZN=F", "BTCUSD": "BTC-USD",
    # WTI: no long series (negative price 2020) - only in the CFD data
}
# Further markets FundedNext offers (measurement for a wider trend universe, 2026-10-29) -
# NOT part of the agreed S-007 universe; studies pass them explicitly.
EXTRA_TICKERS = {
    "AUS200": "^AXJO", "FRA40": "^FCHI", "EU50": "^STOXX50E", "ESP35": "^IBEX", "HK50": "^HSI",
    "XPTUSD": "PL=F", "XPDUSD": "PA=F",
    "EURAUD": "EURAUD=X", "EURCAD": "EURCAD=X", "EURNZD": "EURNZD=X", "GBPAUD": "GBPAUD=X", "GBPCAD": "GBPCAD=X",
    "GBPCHF": "GBPCHF=X", "GBPNZD": "GBPNZD=X", "NZDJPY": "NZDJPY=X", "CADJPY": "CADJPY=X", "CHFJPY": "CHFJPY=X",
    "AUDCHF": "AUDCHF=X", "NZDCAD": "NZDCAD=X", "NZDCHF": "NZDCHF=X", "CADCHF": "CADCHF=X",
    "USDSEK": "SEK=X", "USDNOK": "NOK=X",
}
ALL_TICKERS = {**TICKERS, **EXTRA_TICKERS}
COLS = ("open", "high", "low", "close")


def download(asset: str, start: str = "1995-01-01") -> pd.DataFrame:
    import yfinance as yf
    d = yf.download(ALL_TICKERS[asset], start=start, progress=False, auto_adjust=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.rename(columns=str.lower)[list(COLS)].dropna(subset=["close"])
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    d.index.name = "date"
    CACHE.mkdir(parents=True, exist_ok=True)
    d.to_csv(CACHE / f"{asset}.csv")
    return d


def fix_fx_dates(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance "=X" bars: the "close" of day D is the price at 00:00 UTC of D (it equals
    the open), i.e. the PREVIOUS day's close. The price at the end of D is the open of the
    next bar - re-dated so that every market's close of D is the close of D (checked
    against the Dukascopy CFD closes, ``studies/tsmom_check.py --referenz``)."""
    out = df.copy()
    out["close"] = df["open"].shift(-1)
    return out.dropna(subset=["close"])


def load_daily(asset: str, *, fetch: bool = False) -> pd.DataFrame:
    """Daily OHLC from the cache (``fetch=True`` downloads it if missing); FX re-dated."""
    f = CACHE / f"{asset}.csv"
    if not f.exists():
        if not fetch:
            raise FileNotFoundError(f"{f} missing - run studies/tsmom_check.py --fetch once")
        download(asset)
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    return fix_fx_dates(df) if ALL_TICKERS[asset].endswith("=X") else df


def clean_bad_ticks(df: pd.DataFrame, n_sigma: float = 8.0, revert: float = 0.8, com: int = 60
                    ) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Replace isolated one-day spikes that revert the next day (see module doc)."""
    out = df.copy()
    c = out["close"].to_numpy(float).copy()
    r = np.r_[np.nan, c[1:] / c[:-1] - 1]
    sig = pd.Series(r).ewm(com=com, min_periods=20).std().shift(1).to_numpy()
    bad = []
    for t in range(1, len(c) - 1):
        if not np.isfinite(sig[t]) or not np.isfinite(r[t]) or abs(r[t]) <= n_sigma * sig[t]:
            continue
        move = c[t] - c[t - 1]
        back = c[t] - c[t + 1]
        if move != 0 and back / move >= revert:
            bad.append(out.index[t])
            c[t] = c[t - 1]
    out["close"] = c
    for k in ("open", "high", "low"):
        out.loc[bad, k] = out.loc[bad, "close"]               # a flat bar at the previous close
    return out, bad


__all__ = ["TICKERS", "EXTRA_TICKERS", "ALL_TICKERS", "CACHE", "download", "load_daily", "fix_fx_dates", "clean_bad_ticks"]
