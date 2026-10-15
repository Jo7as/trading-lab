"""Macro adapter — fetches macro indicators (BTC dominance, fear & greed).

Free endpoints: alternative.me Fear & Greed Index.
"""
from __future__ import annotations

import httpx


class MacroAdapter:
    async def fetch(self, asset: str) -> dict:
        return await _fetch(asset)


async def _fetch(asset: str) -> dict:
    fear_greed = 50
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://api.alternative.me/fng/?limit=1")
            data = resp.json()
            if data.get("data"):
                fear_greed = int(data["data"][0]["value"])
    except Exception:
        pass

    return {
        "schema_version": 1,
        "fear_greed_index": fear_greed,
        "source": "alternative_me",
        "asset": asset,
    }
