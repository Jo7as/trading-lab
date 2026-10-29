"""OLS with Newey-West (Bartlett kernel) standard errors, for overlapping-horizon regressions.

Forward returns over h periods overlap, so plain OLS t-statistics are far too large;
Newey-West with ``lags >= h - 1`` is the standard fix (Kim & Hansen 2026 use it).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OLSResult:
    coef: np.ndarray        # [intercept, slopes...]
    se: np.ndarray
    n: int
    r2: float

    @property
    def t(self) -> np.ndarray:
        return self.coef / self.se


def newey_west_ols(y, x, lags: int) -> OLSResult:
    """Regress ``y`` on a constant and ``x`` (1-D or 2-D), HAC errors with ``lags`` lags.

    Rows with NaN in y or x are dropped first; the remaining rows are treated as an
    evenly spaced series (lags count rows, not clock time).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y, x = y[ok], x[ok]
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    if n <= X.shape[1]:
        raise ValueError("not enough observations")
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError as exc:
        raise ValueError("regressors are collinear or constant") from exc
    coef = xtx_inv @ X.T @ y
    u = y - X @ coef
    xu = X * u[:, None]
    s = xu.T @ xu
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        g = xu[k:].T @ xu[:-k]
        s += w * (g + g.T)
    cov = xtx_inv @ s @ xtx_inv
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((u ** 2).sum()) / tss if tss > 0 else float("nan")
    return OLSResult(coef=coef, se=np.sqrt(np.diag(cov)), n=n, r2=r2)
