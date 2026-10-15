"""Historical Yahoo Finance adapter for backtest/replay mode.

Pre-loads N days of 1h candles from Yahoo Finance, then feeds them
candle-by-candle to the backtest engine. Indicators are computed
on the full series (vectorized) — no look-ahead since all signals
are evaluated at candle close and trades execute at that close.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import pandas as pd

from .yfinance_adapter import YFinanceAdapter, _clean_ohlcv

logger = logging.getLogger(__name__)


class HistoricalYFAdapter:
    """Load historical OHLCV from Yahoo Finance for backtesting."""

    def __init__(self, period: str = "730d", interval: str = "1h"):
        self.period = period
        self.interval = interval
        self._df: pd.DataFrame | None = None
        self._adapter = YFinanceAdapter()

    async def load(self, asset: str) -> pd.DataFrame:
        """Fetch and cache full OHLCV history."""
        if self._df is not None:
            return self._df

        df = await self._adapter.fetch_history(
            asset, interval=self.interval, period=self.period
        )
        if df.empty:
            logger.error("No historical data for %s", asset)
            return df

        self._df = df
        print(f"[historical-yf] loaded {len(df)} candles of {asset} "
              f"({self.period}, {self.interval})")
        return df

    def get_df(self) -> pd.DataFrame:
        """Return cached DataFrame."""
        if self._df is None:
            raise RuntimeError("Call load() first")
        return self._df
