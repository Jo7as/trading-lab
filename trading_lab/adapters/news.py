"""News adapter — fetches recent news headlines for sentiment.

Free endpoint: CryptoCompare News API (no key required for basic news).
"""
from __future__ import annotations

import os
import httpx


class NewsAdapter:
    async def fetch(self, asset: str) -> dict:
        return await _fetch(asset)


async def _fetch(asset: str) -> dict:
    base = asset.split("/")[0]
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://min-api.cryptocompare.com/data/v2/news/",
                params={"lang": "EN", "categories": base},
            )
            data = resp.json()
            articles = data.get("Data", [])
            headlines = [a.get("title", "") for a in articles[:5]]
            sentiment = _simple_sentiment(headlines)
    except Exception:
        headlines = []
        sentiment = 0.0

    return {
        "schema_version": 1,
        "news_headlines": headlines,
        "news_sentiment": sentiment,
        "source": "cryptocompare",
        "asset": asset,
    }


def _simple_sentiment(headlines: list[str]) -> float:
    """Crude keyword-based sentiment in [-1, +1]."""
    positive = ["surge", "rally", "bull", "breakout", "adoption", "upgrade", "partnership", "gain"]
    negative = ["crash", "bear", "hack", "dump", "ban", "lawsuit", "decline", "fear", "sell"]
    if not headlines:
        return 0.0
    text = " ".join(headlines).lower()
    pos = sum(1 for w in positive if w in text)
    neg = sum(1 for w in negative if w in text)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 2)
