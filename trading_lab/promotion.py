"""The bar a promotion to ``best_oos`` has to clear.

The promotion path used to ask one question — "is the OOS PnL higher than the last
record?" — with no trade count, no Sharpe floor and no drawdown limit. That is how
a five-trade window with a lucky shape became a record. The significance gate
(:func:`trading_lab.significance.promotion_gate`) still runs first; this is the
second, cheaper filter, and both are recorded in ``best_oos_meta.json``.

Thresholds come from the environment so a deployment can tighten them without a
code change.
"""
from __future__ import annotations

from dataclasses import dataclass

from .env import env
from .significance import MIN_TRADES


@dataclass(frozen=True)
class PromotionBar:
    """Minimum OOS quality for a promotion."""

    min_trades: int = MIN_TRADES
    min_sharpe: float = 0.5
    max_dd_pct: float = 20.0

    @classmethod
    def from_env(cls) -> "PromotionBar":
        return cls(
            min_trades=int(env("LAB_BEST_OOS_MIN_TRADES", str(MIN_TRADES))),
            min_sharpe=float(env("LAB_BEST_OOS_MIN_SHARPE", "0.5")),
            max_dd_pct=float(env("LAB_BEST_OOS_MAX_DD_PCT", "20")),
        )

    def describe(self) -> str:
        return (f"n>={self.min_trades}, sharpe>={self.min_sharpe:g}, "
                f"dd<={self.max_dd_pct:g}%")


def check(metrics: dict, trades: list[dict] | None, bar: PromotionBar) -> tuple[bool, str]:
    """``(ok, reason)`` for the OOS window; ``reason`` is empty when ok."""
    closed = [t for t in (trades or []) if t.get("status") in (None, "closed")]
    if len(closed) < bar.min_trades:
        return False, f"oos trades {len(closed)} < {bar.min_trades}"
    sharpe = float(metrics.get("sharpe", 0) or 0)
    if sharpe < bar.min_sharpe:
        return False, f"oos sharpe {sharpe:.2f} < {bar.min_sharpe:g}"
    dd = abs(float(metrics.get("max_drawdown_pct", 0) or 0))
    if dd > bar.max_dd_pct:
        return False, f"oos drawdown {dd:.1f}% > {bar.max_dd_pct:g}%"
    return True, ""
