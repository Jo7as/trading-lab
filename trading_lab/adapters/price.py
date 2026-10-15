"""Price adapter — fetches OHLCV from ccxt (free public endpoints).

Falls back to yfinance (Yahoo Finance) if ccxt fails.
Exposes: price, rsi(N), volume, ema200_trend (4h), atr(N).

Strategy-driven: reads rsi_period, atr_period, atr_multiplier, trend_timeframe
from strategy.yaml at fetch time so the reflection loop can adjust them.
"""
from __future__ import annotations
from ..env import env

import os
from typing import Any

import ccxt.async_support as ccxt
import numpy as np
import yaml


class PriceAdapter:
    async def fetch(self, asset: str, strategy: dict | None = None) -> dict:
        return await _fetch(asset, strategy)


async def _fetch(asset: str, strategy: dict | None = None) -> dict:
    exchange = ccxt.binance({
        "apiKey": env("EXCHANGE_API_KEY", ""),
        "secret": env("EXCHANGE_API_SECRET", ""),
        "enableRateLimit": True,
    })
    try:
        # --- read strategy params ---
        entry_cfg = (strategy or {}).get("entry", {})
        risk_cfg = (strategy or {}).get("risk", {})
        rsi_period = int(entry_cfg.get("rsi_period", 14))
        atr_period = int(risk_cfg.get("atr_period", 14))
        trend_tf = entry_cfg.get("trend_timeframe", "4h")

        # --- 1h candles for price + RSI + ATR (need >= period+1) ---
        limit_1h = max(rsi_period, atr_period) + 120  # extra buffer
        ohlcv_1h = await exchange.fetch_ohlcv(asset, timeframe="1h", limit=limit_1h)
        if not ohlcv_1h or len(ohlcv_1h) < max(rsi_period, atr_period) + 2:
            return _fallback_yfinance(asset, strategy)

        closes_1h = np.array([c[4] for c in ohlcv_1h], dtype=float)
        highs_1h = np.array([c[2] for c in ohlcv_1h], dtype=float)
        lows_1h = np.array([c[3] for c in ohlcv_1h], dtype=float)
        volumes_1h = np.array([c[5] for c in ohlcv_1h], dtype=float)

        price = float(closes_1h[-1])
        rsi = _rsi(closes_1h, rsi_period)
        atr = _atr(highs_1h, lows_1h, closes_1h, atr_period)
        volume = float(volumes_1h[-1])

        # --- trend filter: EMA200 on higher timeframe ---
        trend_up = True  # default permissive
        tf_map = {"15m": "15m", "1h": "1h", "2h": "2h", "4h": "4h", "1d": "1d"}
        trend_tf_key = tf_map.get(trend_tf, "4h")
        try:
            ohlcv_trend = await exchange.fetch_ohlcv(
                asset, timeframe=trend_tf_key, limit=250
            )
            if ohlcv_trend and len(ohlcv_trend) >= 200:
                closes_trend = np.array([c[4] for c in ohlcv_trend], dtype=float)
                ema200 = _ema(closes_trend, 200)
                trend_up = bool(price >= ema200)
        except Exception:
            pass  # keep permissive if trend data unavailable

        return {
            "schema_version": 1,
            "price": price,
            "rsi": round(rsi, 2),
            "atr": round(atr, 4),
            "volume": volume,
            "trend_up": trend_up,
            "source": "ccxt",
            "asset": asset,
        }
    finally:
        await exchange.close()


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 0.0
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )
    return float(np.mean(tr[-period:]))


def _ema(values: np.ndarray, period: int) -> float:
    if len(values) < period:
        return float(values[-1])
    alpha = 2.0 / (period + 1)
    ema = float(values[0])
    for v in values[1:]:
        ema = alpha * float(v) + (1 - alpha) * ema
    return ema


def _fallback_yfinance(asset: str, strategy: dict | None = None) -> dict:
    """Synchronous yfinance fallback (runs in thread implicitly)."""
    import yfinance as yf

    entry_cfg = (strategy or {}).get("entry", {})
    risk_cfg = (strategy or {}).get("risk", {})
    rsi_period = int(entry_cfg.get("rsi_period", 14))
    atr_period = int(risk_cfg.get("atr_period", 14))

    ticker = asset.replace("/USDT", "-USD").replace("/USDC", "-USD")
    df = yf.download(ticker, period="7d", interval="1h", progress=False)
    if df.empty:
        return {"schema_version": 1, "price": None, "rsi": None, "source": "yfinance"}
    closes = df["Close"].values.flatten()
    highs = df["High"].values.flatten()
    lows = df["Low"].values.flatten()
    volumes = df["Volume"].values.flatten()

    price = float(closes[-1])
    rsi = 50.0
    if len(closes) > rsi_period:
        rsi = _rsi(closes, rsi_period)
    atr = _atr(highs, lows, closes, atr_period) if len(closes) > atr_period else 0.0

    # yfinance doesn't give us 4h EMA200 easily; default permissive
    return {
        "schema_version": 1,
        "price": price,
        "rsi": round(rsi, 2),
        "atr": round(atr, 4),
        "volume": float(volumes[-1]),
        "trend_up": True,
        "source": "yfinance",
        "asset": asset,
    }
