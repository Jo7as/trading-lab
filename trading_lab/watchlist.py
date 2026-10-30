"""Power hint and watchlist: do not lose small edges - without loosening any gate.

* ``power``: chance that an edge of a given Sharpe reaches the S4 bar on the available years
  (t = Sharpe x sqrt(years), normal approximation, the principle of ``docs/PIPELINE_AUDIT.md``).
  A warning only, it blocks nothing.
* ``benjamini_hochberg`` + ``family_p``: a watchlist at 20 % false discovery rate. Each family (strategy x
  market) counts once; its best variant is corrected for the search inside the family (Bonferroni on the
  effective trials). The list NEVER releases trading or the basket - it only names what deserves a test
  on fresh data or inside the basket. S1-S5 stay as they are.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from .screening import required_t

_N = NormalDist()
WATCH_FDR = 0.20


def power(sharpe: float, years: float, n_trials: float | None = 1) -> float:
    """P(daily t >= the S4 bar) for a true annual Sharpe on ``years`` of data."""
    if years <= 0:
        return 0.0
    return float(_N.cdf(sharpe * np.sqrt(years) - required_t(n_trials)))


def power_note(years: float, n_trials: float | None = 1) -> str:
    cells = ", ".join(f"Sharpe {s:.1f}: {power(s, years, n_trials):.0%}".replace(".", ",", 1)
                      for s in (0.5, 0.75, 1.0, 1.5))
    weak = power(0.75, years, n_trials) < 0.5
    tail = " - kleine Kanten sind hier kaum nachweisbar, ein Fehlschlag ist dann kein Gegenbeweis." if weak else ""
    return f"Trefferchance bei {years:.1f} Jahren und {n_trials} Versuchen: {cells}{tail}"


def family_p(t: float, n_trials: float = 1) -> float:
    """One-sided p of the best variant, corrected for the search inside its family."""
    return float(min(1.0, (1.0 - _N.cdf(t)) * max(float(n_trials), 1.0)))


def benjamini_hochberg(p: np.ndarray | list[float], fdr: float = WATCH_FDR) -> np.ndarray:
    """True for the hypotheses kept at false discovery rate ``fdr``."""
    a = np.asarray(p, dtype=float)
    keep = np.zeros(len(a), dtype=bool)
    if len(a) == 0:
        return keep
    order = np.argsort(a)
    ok = a[order] <= fdr * np.arange(1, len(a) + 1) / len(a)
    if ok.any():
        keep[order[: int(np.flatnonzero(ok).max()) + 1]] = True
    return keep


def build(rows: pd.DataFrame, fdr: float = WATCH_FDR) -> pd.DataFrame:
    """One row per family (needs ``family, t, trials``), sorted, with ``p`` and ``beobachten``."""
    best = rows.sort_values("t", ascending=False).groupby("family", sort=False).head(1).copy()
    best["p"] = [family_p(t, n) for t, n in zip(best["t"], best["trials"])]
    best["beobachten"] = benjamini_hochberg(best["p"].to_numpy(), fdr)
    return best.sort_values("p").reset_index(drop=True)


__all__ = ["power", "power_note", "family_p", "benjamini_hochberg", "build", "WATCH_FDR"]
