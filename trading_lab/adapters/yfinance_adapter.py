"""Yahoo Finance data adapter — fetches OHLCV for Gold/Forex/Crypto.

Handles NaN, gaps, forward-fill. Returns clean DataFrame with
columns: open, high, low, close, volume (all lowercase).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Yahoo Finance ticker map
TICKER_MAP = {
    "XAU/USD": "GC=F",     # Gold futures
    "GC=F": "GC=F",
    "EUR/USD": "EURUSD=X",
    "EURUSD=X": "EURUSD=X",
    "USD/JPY": "USDJPY=X",
    "USDJPY=X": "USDJPY=X",
    "GBP/USD": "GBPUSD=X",
    "GBPUSD=X": "GBPUSD=X",
    "SOL/USDT": "SOL-USD",  # crypto via yfinance
}


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw yfinance DataFrame: lowercase cols, handle NaN/gaps."""
    if df.empty:
        return df

    # yfinance returns MultiIndex columns sometimes — flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Rename to lowercase
    rename_map = {}
    for col in df.columns:
        cl = str(col).lower().strip()
        if cl in ("open", "high", "low", "close", "adj close", "volume"):
            rename_map[col] = cl.replace("adj close", "close")
    df = df.rename(columns=rename_map)

    # Ensure required columns exist
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            if col == "volume":
                df[col] = 0.0
            else:
                df[col] = np.nan

    # Forward-fill any NaN in OHLC (gap handling)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].ffill()
    # Volume: fill NaN with 0
    df["volume"] = df["volume"].fillna(0.0)

    # Drop any remaining rows where close is still NaN (start of data)
    df = df.dropna(subset=["close"]).copy()

    # Remove duplicate indices
    df = df[~df.index.duplicated(keep="last")].copy()

    # Ensure sorted by timestamp
    df = df.sort_index().copy()

    # Clip obvious data errors (negative prices)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].clip(lower=0.0)

    # Ensure high >= max(open, close) and low <= min(open, close)
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)

    return df[required]


class YFinanceAdapter:
    """Async-compatible Yahoo Finance data adapter."""

    async def fetch(
        self, asset: str, interval: str = "1h", period: str = "730d"
    ) -> dict[str, Any]:
        """Fetch latest OHLCV data.

        Returns dict with schema_version=1 for compatibility with the loop.
        """
        ticker = TICKER_MAP.get(asset, asset)

        try:
            df = await asyncio.to_thread(
                yf.download,
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            logger.error("yfinance fetch failed for %s: %s", ticker, exc)
            return {"schema_version": 1, "price": None, "source": "yfinance_error"}

        df = _clean_ohlcv(df)
        if df.empty:
            return {"schema_version": 1, "price": None, "source": "yfinance_empty"}

        last = df.iloc[-1]
        return {
            "schema_version": 1,
            "price": float(last["close"]),
            "volume": float(last["volume"]),
            "source": "yfinance",
            "asset": asset,
            "df": df,  # full DataFrame for indicator computation
        }

    async def fetch_history(
        self,
        asset: str,
        interval: str = "1h",
        period: str = "730d",
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch full OHLCV history as a clean DataFrame."""
        del start, end  # yfinance uses period only
        ticker = TICKER_MAP.get(asset, asset)
        try:
            df = await asyncio.to_thread(
                yf.download,
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            logger.error("yfinance fetch_history failed for %s: %s", ticker, exc)
            return pd.DataFrame()

        return _clean_ohlcv(df)
