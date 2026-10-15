"""Historical price adapter — replays past candles one by one.

Pre-fetches N days of 1h candles AND N days of 4h candles from ccxt (Binance),
then each fetch() call returns the next 1h candle in sequence. RSI, ATR, and
EMA200-trend are computed from the rolling window up to the current position.

Strategy-driven: reads rsi_period, atr_period, trend_timeframe from strategy
at fetch time so the reflection loop can adjust them.
"""
from __future__ import annotations

import asyncio
from typing import Any

import ccxt.async_support as ccxt
import numpy as np

from .price import _rsi, _atr, _ema


class HistoricalPriceAdapter:
    def __init__(self, days: int = 365, timeframe: str = "1h"):
        self.days = days
        self.timeframe = timeframe
        self._candles: list[dict] = []
        self._trend_candles: list[dict] = []
        self._pointer = 0
        self._loaded = False

    async def _load(self, asset: str, trend_timeframe: str = "4h") -> None:
        """Fetch all historical candles upfront — 1h for trading, 4h for trend."""
        exchange = ccxt.binance({"enableRateLimit": True})
        try:
            # --- 1h candles ---
            all_1h = []
            since = exchange.milliseconds() - self.days * 24 * 60 * 60 * 1000
            while True:
                batch = await exchange.fetch_ohlcv(
                    asset, timeframe=self.timeframe, since=since, limit=1000
                )
                if not batch:
                    break
                all_1h.extend(batch)
                last_ts = batch[-1][0]
                if len(batch) < 1000:
                    break
                since = last_ts + 1
                if since > exchange.milliseconds():
                    break

            # --- trend candles (4h) ---
            all_trend = []
            since_t = exchange.milliseconds() - self.days * 24 * 60 * 60 * 1000
            while True:
                batch_t = await exchange.fetch_ohlcv(
                    asset, timeframe=trend_timeframe, since=since_t, limit=1000
                )
                if not batch_t:
                    break
                all_trend.extend(batch_t)
                last_ts_t = batch_t[-1][0]
                if len(batch_t) < 1000:
                    break
                since_t = last_ts_t + 1
                if since_t > exchange.milliseconds():
                    break
        finally:
            await exchange.close()

        # deduplicate
        seen_1h = set()
        unique_1h = [c for c in all_1h if c[0] not in seen_1h and not seen_1h.add(c[0])]
        seen_t = set()
        unique_trend = [c for c in all_trend if c[0] not in seen_t and not seen_t.add(c[0])]

        # store trend candles for EMA200 lookup
        trend_closes = np.array([c[4] for c in unique_trend], dtype=float)
        self._trend_candles = [
            {
                "ts_ms": int(unique_trend[i][0]),
                "close": float(trend_closes[i]),
            }
            for i in range(len(unique_trend))
        ]

        # store 1h candles with precomputed arrays
        self._candles = [
            {
                "ts": int(unique_1h[i][0]) / 1000,
                "ts_ms": int(unique_1h[i][0]),
                "open": float(unique_1h[i][1]),
                "high": float(unique_1h[i][2]),
                "low": float(unique_1h[i][3]),
                "close": float(unique_1h[i][4]),
                "volume": float(unique_1h[i][5]),
                "index": i,
            }
            for i in range(len(unique_1h))
        ]

        self._loaded = True
        self._pointer = 0
        print(f"[replay] loaded {len(self._candles)} 1h candles + "
              f"{len(self._trend_candles)} {trend_timeframe} trend candles "
              f"({self.days}d of {asset})")

    def _compute_indicators(self, idx: int, strategy: dict) -> dict:
        """Compute RSI, ATR, EMA200-trend at position idx using rolling window."""
        entry_cfg = strategy.get("entry", {})
        risk_cfg = strategy.get("risk", {})
        rsi_period = int(entry_cfg.get("rsi_period", 14))
        atr_period = int(risk_cfg.get("atr_period", 14))

        # rolling window up to current candle
        start = max(0, idx - max(rsi_period, atr_period) - 120)
        window = self._candles[start:idx + 1]
        if len(window) < max(rsi_period, atr_period) + 2:
            return {"rsi": 50.0, "atr": 0.0, "trend_up": True}

        closes = np.array([c["close"] for c in window], dtype=float)
        highs = np.array([c["high"] for c in window], dtype=float)
        lows = np.array([c["low"] for c in window], dtype=float)

        rsi = _rsi(closes, rsi_period)
        atr = _atr(highs, lows, closes, atr_period)

        # EMA200 trend: find trend candles up to current 1h timestamp
        current_ts_ms = self._candles[idx]["ts_ms"]
        trend_closes = np.array(
            [c["close"] for c in self._trend_candles if c["ts_ms"] <= current_ts_ms],
            dtype=float,
        )
        trend_up = True
        if len(trend_closes) >= 200:
            ema200 = _ema(trend_closes, 200)
            trend_up = bool(self._candles[idx]["close"] >= ema200)

        return {"rsi": round(rsi, 2), "atr": round(atr, 4), "trend_up": trend_up}

    async def fetch(self, asset: str, strategy: dict | None = None) -> dict:
        if not self._loaded:
            entry_cfg = (strategy or {}).get("entry", {})
            trend_tf = entry_cfg.get("trend_timeframe", "4h")
            await self._load(asset, trend_tf)
        if self._pointer >= len(self._candles):
            return {
                "schema_version": 1,
                "price": None,
                "rsi": None,
                "atr": None,
                "trend_up": None,
                "source": "historical_exhausted",
                "asset": asset,
                "replay_done": True,
            }

        candle = self._candles[self._pointer]
        self._pointer += 1

        indicators = self._compute_indicators(candle["index"], strategy or {})

        return {
            "schema_version": 1,
            "price": candle["close"],
            "rsi": indicators["rsi"],
            "atr": indicators["atr"],
            "trend_up": indicators["trend_up"],
            "volume": candle["volume"],
            "source": "historical",
            "asset": asset,
            "ts": candle["ts"],
            "candle_index": candle["index"],
        }

    @property
    def progress(self) -> tuple[int, int]:
        return (self._pointer, len(self._candles))

    @property
    def exhausted(self) -> bool:
        return self._loaded and self._pointer >= len(self._candles)
