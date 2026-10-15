"""Market data adapter factory (yfinance | dukascopy)."""
from __future__ import annotations
from ..env import env

import os
from typing import Protocol

import pandas as pd


class MarketDataAdapter(Protocol):
    async def fetch_history(
        self,
        asset: str,
        interval: str = "1h",
        period: str = "730d",
        *,
        start: object | None = None,
        end: object | None = None,
    ) -> pd.DataFrame: ...


def get_market_adapter(source: str | None = None) -> MarketDataAdapter:
    """Return data adapter from source or LAB_DATA_SOURCE env."""
    src = (source or env("LAB_DATA_SOURCE", "yfinance")).strip().lower()
    if src in ("dukascopy", "duka"):
        from .dukascopy_adapter import DukascopyAdapter

        return DukascopyAdapter()
    from .yfinance_adapter import YFinanceAdapter

    return YFinanceAdapter()
