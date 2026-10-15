"""Dukascopy Bank historical OHLCV adapter (XAU/USD spot, forex, etc.).

Free public API — supports long history (years) for 1h/15m/5m unlike yfinance.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

import dukascopy_python as duka
from dukascopy_python.instruments import (
    INSTRUMENT_FX_MAJORS_AUD_USD,
    INSTRUMENT_FX_MAJORS_USD_JPY,
    INSTRUMENT_FX_METALS_XAG_USD,
    INSTRUMENT_FX_METALS_XAU_USD,
    INSTRUMENT_IDX_AMERICA_E_NQ_100,
)

from .yfinance_adapter import _clean_ohlcv

logger = logging.getLogger(__name__)

# Map asset codes → Dukascopy instrument strings
INSTRUMENT_MAP: dict[str, str] = {
    "GC=F": INSTRUMENT_FX_METALS_XAU_USD,
    "XAU/USD": INSTRUMENT_FX_METALS_XAU_USD,
    "XAUUSD": INSTRUMENT_FX_METALS_XAU_USD,
    "XAG/USD": INSTRUMENT_FX_METALS_XAG_USD,
    "XAGUSD": INSTRUMENT_FX_METALS_XAG_USD,
    "SI=F": INSTRUMENT_FX_METALS_XAG_USD,
    "EUR/USD": "EUR/USD",
    "EURUSD=X": "EUR/USD",
    "GBP/USD": "GBP/USD",
    "GBPUSD=X": "GBP/USD",
    "USD/JPY": INSTRUMENT_FX_MAJORS_USD_JPY,
    "USDJPY=X": INSTRUMENT_FX_MAJORS_USD_JPY,
    "AUD/USD": INSTRUMENT_FX_MAJORS_AUD_USD,
    "AUDUSD=X": INSTRUMENT_FX_MAJORS_AUD_USD,
    # Nasdaq-100 CFD (Dukascopy)
    "NQ": INSTRUMENT_IDX_AMERICA_E_NQ_100,
    "NAS100": INSTRUMENT_IDX_AMERICA_E_NQ_100,
    "US100": INSTRUMENT_IDX_AMERICA_E_NQ_100,
    "NQ100": INSTRUMENT_IDX_AMERICA_E_NQ_100,
    "E_NQ-100": INSTRUMENT_IDX_AMERICA_E_NQ_100,
}

INTERVAL_MAP: dict[str, str] = {
    "1h": duka.INTERVAL_HOUR_1,
    "60m": duka.INTERVAL_HOUR_1,
    "30m": duka.INTERVAL_MIN_30,
    "15m": duka.INTERVAL_MIN_15,
    "10m": duka.INTERVAL_MIN_10,
    "5m": duka.INTERVAL_MIN_5,
    "1m": duka.INTERVAL_MIN_1,
    "1d": duka.INTERVAL_DAY_1,
    "4h": duka.INTERVAL_HOUR_4,
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def resolve_date_range(
    period: str = "730d",
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> tuple[datetime, datetime]:
    """Resolve start/end (naive UTC) from period string or explicit bounds."""
    end_dt = _as_naive_utc(end) if end is not None else _utc_now_naive()
    if start is not None:
        return _as_naive_utc(start), end_dt

    period = (period or "730d").strip().lower()
    if period.endswith("d") and period[:-1].isdigit():
        days = int(period[:-1])
        return end_dt - timedelta(days=days), end_dt
    if _DATE_RE.match(period):
        return datetime.fromisoformat(period), end_dt
    if period in ("max", "all"):
        return datetime(2000, 1, 1), end_dt
    if period.startswith("20") and ":" in period:
        a, b = period.split(":", 1)
        return datetime.fromisoformat(a), datetime.fromisoformat(b)
    raise ValueError(f"Unsupported period for dukascopy: {period!r}")


def _as_naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class DukascopyAdapter:
    """Async-compatible Dukascopy historical data adapter."""

    def __init__(self, *, offer_side: str = duka.OFFER_SIDE_BID):
        self.offer_side = offer_side

    def _resolve_instrument(self, asset: str) -> str:
        key = asset.strip()
        if key in INSTRUMENT_MAP:
            return INSTRUMENT_MAP[key]
        if "/" in key:
            return key
        raise ValueError(f"No Dukascopy instrument mapping for asset {asset!r}")

    def _resolve_interval(self, interval: str) -> str:
        key = interval.strip().lower()
        if key not in INTERVAL_MAP:
            raise ValueError(
                f"Unsupported Dukascopy interval {interval!r}. "
                f"Use one of: {', '.join(sorted(INTERVAL_MAP))}"
            )
        return INTERVAL_MAP[key]

    def _fetch_sync(
        self,
        instrument: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        return duka.fetch(
            instrument,
            interval,
            self.offer_side,
            start,
            end,
            max_retries=7,
        )

    async def fetch(
        self,
        asset: str,
        interval: str = "1h",
        period: str = "730d",
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, Any]:
        df = await self.fetch_history(
            asset, interval=interval, period=period, start=start, end=end
        )
        if df.empty:
            return {"schema_version": 1, "price": None, "source": "dukascopy_empty"}
        last = df.iloc[-1]
        return {
            "schema_version": 1,
            "price": float(last["close"]),
            "volume": float(last["volume"]),
            "source": "dukascopy",
            "asset": asset,
            "df": df,
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
        """Fetch OHLCV history. period examples: 730d, 2020-01-01, max."""
        try:
            instrument = self._resolve_instrument(asset)
            duka_interval = self._resolve_interval(interval)
            start_dt, end_dt = resolve_date_range(period, start=start, end=end)
        except Exception as exc:
            logger.error("dukascopy resolve failed for %s: %s", asset, exc)
            return pd.DataFrame()

        try:
            df = await asyncio.to_thread(
                self._fetch_sync,
                instrument,
                duka_interval,
                start_dt,
                end_dt,
            )
        except Exception as exc:
            logger.error(
                "dukascopy fetch failed for %s (%s→%s): %s",
                asset,
                start_dt,
                end_dt,
                exc,
            )
            return pd.DataFrame()

        if df.empty:
            return df

        df = df.rename(columns=str.lower)
        return _clean_ohlcv(df)
