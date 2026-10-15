"""Environment access with a legacy fallback.

Canonical names use the ``LAB_`` prefix. Older deployments still set the
adapter's historical ``HERMES_*`` names, so ``env()`` reads the new name first
and falls back to the legacy one. Existing Railway services therefore keep
working without any env change, while new setups use the ``LAB_*`` names
documented in ``.env.example``.
"""
from __future__ import annotations

import os

# canonical name → historical alias
_LEGACY: dict[str, str] = {
    "LAB_STATE": "HERMES_TRADING_STATE",
    "LAB_MODE": "HERMES_TRADING_MODE",
    "LAB_WORKER": "HERMES_TRADING_WORKER",
    "LAB_DATA_SOURCE": "HERMES_DATA_SOURCE",
    "LAB_DATA_CACHE": "HERMES_DATA_CACHE",
    "LAB_STRATEGY_PATH": "HERMES_STRATEGY_PATH",
    "LAB_FORCE_HF": "HERMES_FORCE_HF",
    "LAB_NO_REFLECT_ROUNDS": "HERMES_NO_REFLECT_ROUNDS",
    "LAB_BASELINE_PATH": "HERMES_BASELINE_PATH",
    "LAB_STATE_TEMPLATE": "HERMES_STATE_TEMPLATE",
    "LAB_TRADES_MAX_LINES": "HERMES_TRADES_MAX_LINES",
    "LAB_HYPOTHESES_MAX_LINES": "HERMES_HYPOTHESES_MAX_LINES",
    "LAB_HISTORY_MAX_KEEP": "HERMES_HISTORY_MAX_KEEP",
    "LAB_LEADERBOARD_REFRESH": "HERMES_LEADERBOARD_REFRESH",
    "LAB_JPY_STATE": "HERMES_JPY_STATE",
}


def env(name: str, default: str | None = None) -> str | None:
    """Return ``name``, else its legacy ``HERMES_*`` alias, else ``default``."""
    value = os.environ.get(name)
    if value is None and (legacy := _LEGACY.get(name)):
        value = os.environ.get(legacy)
    return default if value is None else value
