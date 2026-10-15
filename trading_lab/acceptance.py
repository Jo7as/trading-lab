"""The standard acceptance path every strategy must run through.

One function, one report. It wraps the shared significance test
(``trading_lab.significance``) and lets a caller attach strategy-specific
pass/fail criteria (period sums, cost stress, neighbour robustness) so the
report — and the accept/reject decision — is produced the same way everywhere.

    report = run_acceptance(
        trades,                                   # a trade log
        extra={"test_period_positive": ..., "double_cost_positive": ...},
    )
    if not report.passed:
        ...
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from . import significance
from .significance import MAX_PVALUE, MIN_PERCENTILE, MIN_TRADES, Significance


@dataclass(frozen=True)
class AcceptanceReport:
    """Result of the standard path: significance + any extra criteria."""

    passed: bool
    significance: Significance
    criteria: dict[str, bool] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"ACCEPTANCE: {'PASS' if self.passed else 'FAIL'}", f"  significance: {self.significance.summary()}"]
        for name, ok in self.criteria.items():
            lines.append(f"  {name}: {'ok' if ok else 'FAIL'}")
        return "\n".join(lines)


def run_acceptance(
    trades: Iterable[dict] | None = None,
    *,
    returns: Sequence[float] | None = None,
    observed: float | None = None,
    null_samples: Sequence[float] | None = None,
    min_percentile: float = MIN_PERCENTILE,
    max_pvalue: float = MAX_PVALUE,
    min_trades: int = MIN_TRADES,
    extra: dict[str, bool] | None = None,
    seed: int = 7,
) -> AcceptanceReport:
    """Run the significance test plus optional extra criteria.

    Provide either a ``trades`` log or an explicit ``returns`` series. When
    ``null_samples`` is given the random-entry (percentile) test is used,
    otherwise the generic sign-flip test. ``passed`` is True only when the
    significance test **and** every extra criterion pass.
    """
    series = list(returns) if returns is not None else significance.returns_from_trades(trades or [])
    sig = significance.assess(
        series,
        observed=observed,
        null_samples=null_samples,
        min_percentile=min_percentile,
        max_pvalue=max_pvalue,
        min_trades=min_trades,
        seed=seed,
    )
    criteria = dict(extra or {})
    return AcceptanceReport(
        passed=bool(sig.passed and all(criteria.values())),
        significance=sig,
        criteria=criteria,
    )
