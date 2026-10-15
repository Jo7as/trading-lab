"""Significance tests shared by every strategy's acceptance path.

A result is only accepted — a backtest PASS, an evolution promotion to
``best_oos`` — if its per-trade return series is distinguishable from a
no-edge null. Two models, one threshold set:

* **generic (default).** A sign-flip permutation test on the trade returns:
  under "no edge" the returns are symmetric around zero, so flipping their
  signs yields the null distribution of the mean. Answers "is the edge real
  given this sample?" without any strategy cooperation.
* **strategy-provided.** A random-entry baseline (a percentile): the caller
  supplies a null sample of the same statistic (sum R). Used by the ORB
  machine (``strategies/orb_2r.py``) and the C1 harness, which can build a true
  random-entry null from their own bar/session data.

Callers can also override ``run_acceptance`` with extra pass/fail criteria
(period sums, cost stress, neighbours) which are reported alongside.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .env import env

# ── Thresholds: the single place every caller reads them from ──────────
MIN_PERCENTILE = 95.0   # must beat this % of a provided random-entry null
MAX_PVALUE = 0.05       # one-sided p-value for the generic sign-flip test
MIN_TRADES = 10         # below this the test is not trusted (avoids 1-lucky-trade)

_BOOT = 2000
_FLIPS = 5000
_RETURN_KEYS = ("r", "pnl_pct", "equity_pnl_pct")


@dataclass(frozen=True)
class Significance:
    """Outcome of one significance test."""

    method: str
    n: int
    observed: float
    ci_low: float
    ci_high: float
    passed: bool
    p_value: float | None = None
    percentile: float | None = None
    reason: str = ""

    def summary(self) -> str:
        bits = [
            self.method,
            f"n={self.n}",
            f"obs={self.observed:+.3f}",
            f"95%-CI=[{self.ci_low:+.3f},{self.ci_high:+.3f}]",
        ]
        if self.p_value is not None:
            bits.append(f"p={self.p_value:.4f}")
        if self.percentile is not None:
            bits.append(f"pct={self.percentile:.1f}")
        bits.append("PASS" if self.passed else "FAIL")
        if self.reason:
            bits.append(f"({self.reason})")
        return " ".join(bits)

    def as_dict(self) -> dict:
        """JSON-serializable form, for the ``best_oos`` meta record."""
        return {
            "method": self.method,
            "n": self.n,
            "observed": round(self.observed, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "p_value": None if self.p_value is None else round(self.p_value, 4),
            "percentile": None if self.percentile is None else round(self.percentile, 1),
            "passed": self.passed,
            "reason": self.reason,
        }


def bootstrap_ci(
    values: Sequence[float], *, n_boot: int = _BOOT, alpha: float = 0.05, seed: int = 7
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean of ``values``."""
    a = _clean(values)
    if a.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, (n_boot, a.size))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def sign_flip_pvalue(values: Sequence[float], *, n: int = _FLIPS, seed: int = 7) -> float:
    """One-sided p-value for H1: mean(values) > 0, under random ±1 signs."""
    a = _clean(values)
    if a.size == 0:
        return 1.0
    observed = float(a.mean())
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, (n, a.size)) * 2 - 1
    null = (signs * a).mean(axis=1)
    return float((1 + int((null >= observed).sum())) / (n + 1))


def percentile_of(observed: float, null_samples: Sequence[float]) -> float:
    """Share of the null distribution that ``observed`` beats, in percent."""
    n = _clean(null_samples)
    if n.size == 0:
        return 0.0
    return float((n < observed).mean() * 100.0)


def returns_from_trades(
    trades: Iterable[dict], keys: Sequence[str] = _RETURN_KEYS
) -> list[float]:
    """Per-trade returns from a trade log (closed trades only).

    Uses the first available key (R for the ORB machine, account % for the
    Position-based strategies), so the same gate works for every strategy.
    """
    out: list[float] = []
    for t in trades or []:
        if t.get("status") not in (None, "closed"):
            continue
        for k in keys:
            v = t.get(k)
            if v is not None:
                out.append(float(v))
                break
    return out


def assess(
    returns: Sequence[float],
    *,
    observed: float | None = None,
    null_samples: Sequence[float] | None = None,
    min_percentile: float = MIN_PERCENTILE,
    max_pvalue: float = MAX_PVALUE,
    min_trades: int = MIN_TRADES,
    seed: int = 7,
) -> Significance:
    """Run the applicable test and decide pass/fail.

    ``null_samples`` selects the random-entry (percentile) test; otherwise the
    generic sign-flip test is used.
    """
    a = _clean(returns)
    lo, hi = bootstrap_ci(a, seed=seed)

    if null_samples is not None:
        obs = float(a.sum()) if observed is None else float(observed)
        pct = percentile_of(obs, null_samples)
        ok = a.size >= min_trades and pct >= min_percentile
        reason = "" if a.size >= min_trades else f"n<{min_trades}"
        return Significance(
            method="random-entry", n=a.size, observed=obs,
            ci_low=lo, ci_high=hi, percentile=pct, passed=ok, reason=reason,
        )

    obs = float(observed) if observed is not None else (float(a.mean()) if a.size else 0.0)
    if a.size < min_trades:
        return Significance(
            method="sign-flip", n=a.size, observed=obs, ci_low=lo, ci_high=hi,
            passed=False, reason=f"n<{min_trades}",
        )
    p = sign_flip_pvalue(a, seed=seed)
    return Significance(
        method="sign-flip", n=a.size, observed=obs, ci_low=lo, ci_high=hi,
        p_value=p, passed=p <= max_pvalue,
        reason="" if p <= max_pvalue else f"p>{max_pvalue}",
    )


def promotion_gate(
    trades: Iterable[dict], *, null_samples: Sequence[float] | None = None, seed: int = 7
) -> tuple[bool, Significance, bool]:
    """Gate for evolution promotion. Returns ``(allowed, significance, evaluated)``.

    Defaults are read from the environment:

    * ``LAB_SIGNIFICANCE`` — ``0`` disables the gate entirely (debug only).
    * ``LAB_SIGNIFICANCE_MODE`` — ``enforce`` (default) blocks a non-significant
      promotion; ``flag`` records the result in ``best_oos_meta.json`` but still
      promotes.
    * ``LAB_SIGNIFICANCE_MIN_PERCENTILE`` / ``_MAX_PVALUE`` / ``_MIN_TRADES``.

    **Fail-open:** when there is nothing to evaluate (no closed trades) the gate
    never blocks — ``evaluated`` is ``False`` and ``allowed`` is ``True``. This
    avoids a silent, permanent block from a missing/empty trade log; the missing
    evidence is surfaced in the meta as ``significant: null``.
    """
    returns = returns_from_trades(trades)
    enabled = env("LAB_SIGNIFICANCE", "1").strip().lower() not in ("0", "false", "no", "off")
    if not enabled:
        return True, assess(returns, min_trades=0, seed=seed), False
    if not returns:
        return True, assess(returns, min_trades=0, seed=seed), False

    sig = assess(
        returns,
        null_samples=null_samples,
        min_percentile=float(env("LAB_SIGNIFICANCE_MIN_PERCENTILE", str(MIN_PERCENTILE))),
        max_pvalue=float(env("LAB_SIGNIFICANCE_MAX_PVALUE", str(MAX_PVALUE))),
        min_trades=int(env("LAB_SIGNIFICANCE_MIN_TRADES", str(MIN_TRADES))),
        seed=seed,
    )
    if sig.passed:
        return True, sig, True
    mode = (env("LAB_SIGNIFICANCE_MODE", "enforce") or "enforce").strip().lower()
    return (mode == "flag"), sig, True


def _clean(values: Sequence[float]) -> np.ndarray:
    a = np.asarray(list(values), dtype=float)
    return a[np.isfinite(a)] if a.size else a
