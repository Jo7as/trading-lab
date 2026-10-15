"""On-chain adapter — fetches basic on-chain metrics.

Free public endpoint: blockchain.info / etherscan / solscan.
Returns minimal data for now; expand with Glassnode key if present.
"""
from __future__ import annotations

import os
import httpx


class OnChainAdapter:
    async def fetch(self, asset: str) -> dict:
        return await _fetch(asset)


async def _fetch(asset: str) -> dict:
    # For SOL/USDT we use a simple heuristic: transaction count from Solana
    # public RPC. This is a best-effort free endpoint.
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Solana public RPC — get recent performance samples
            resp = await client.post(
                "https://api.mainnet-beta.solana.com",
                json={"jsonrpc": "2.0", "id": 1, "method": "getRecentPerformanceSamples", "params": [2]},
                headers={"Content-Type": "application/json"},
            )
            data = resp.json()
            samples = data.get("result", [])
            tx_count = samples[0]["numTransactions"] if samples else 0
    except Exception:
        tx_count = 0

    return {
        "schema_version": 1,
        "onchain_tx_count": tx_count,
        "source": "solana_rpc",
        "asset": asset,
    }
