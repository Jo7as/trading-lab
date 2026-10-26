"""Stage **S4** of the screening funnel: significance, on the daily series.

This is not "the pipeline". It is one rung of it - see ``trading_lab/screening.py``
for the order (S1 gross -> S2 cost -> S3 robustness -> S4 significance -> S5 OOS).

What changed and why: the significance test used to be one of several gates applied
*at the same time*, and its t was computed on the **per-trade** series. With an exit
model of ``sd(R) ~ 4`` a 0.1 R edge needs ~20,000 trades to reach t = 3, so nothing
could ever pass - candidates were rejected for lack of power, not lack of edge.
Note: the daily series does **not** add power (with independent trades its t equals
the per-trade t, see ``screening``); it is used because it is the honest series for
correlated same-day trades. A trade log is judged on its **daily** aggregate
(:func:`significance.assess_trades`), the same criterion as funnel stage S4; the
per-trade t is only the fallback for a log with no clock at all.

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

from . import significance, screening
from .significance import MAX_PVALUE, MIN_PERCENTILE, MIN_TRADES, Significance

STAGE = "S4"
STAGE_NAME = "Signifikanz"


@dataclass(frozen=True)
class AcceptanceReport:
    """Result of stage S4: the significance test plus any extra criteria.

    ``t_daily`` and ``n_days`` are reported because they are what a significance
    call on the *daily* series would use - the per-trade t next to it is the one
    that made every candidate look insignificant.
    """

    passed: bool
    significance: Significance
    criteria: dict[str, bool] = field(default_factory=dict)
    t_daily: float = float("nan")
    n_days: int = 0

    def summary(self) -> str:
        sig = self.significance
        lines = [f"ACCEPTANCE [{STAGE} {STAGE_NAME}]: {'PASS' if self.passed else 'FAIL'}",
                 f"  significance ({sig.method}): {sig.summary()}",
                 f"  t der geprueften Reihe: {sig.t:.2f}   "
                 f"t je Tag: {self.t_daily:.2f} ({self.n_days} Tage)"]
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
    """Run stage S4 (significance) plus optional extra criteria.

    Provide either a ``trades`` log or an explicit ``returns`` series. A trade
    log is judged on its **daily** series (``significance.assess_trades``, the
    same criterion as funnel stage S4); a bare ``returns`` series cannot be
    aggregated and keeps the per-trade test. When ``null_samples`` is given the
    random-entry (percentile) test is used instead. ``passed`` is True only when
    the significance test **and** every extra criterion pass.
    """
    if returns is None and trades is not None:
        log = list(trades)
        sig = significance.assess_trades(
            log,
            observed=observed,
            null_samples=null_samples,
            min_percentile=min_percentile,
            max_pvalue=max_pvalue,
            min_trades=min_trades,
            seed=seed,
        )
        daily = screening.daily_series(log)
    else:
        series = list(returns or [])
        sig = significance.assess(
            series,
            observed=observed,
            null_samples=null_samples,
            min_percentile=min_percentile,
            max_pvalue=max_pvalue,
            min_trades=min_trades,
            seed=seed,
        )
        daily = series
    criteria = dict(extra or {})
    return AcceptanceReport(
        passed=bool(sig.passed and all(criteria.values())),
        significance=sig,
        criteria=criteria,
        t_daily=screening.t_stat(daily),
        n_days=int(len(daily)),
    )
