"""Newey-West OLS: coefficients, White case, and overlap inflation of the error."""
from __future__ import annotations

import numpy as np
import pytest

from trading_lab.hac import newey_west_ols


def test_exact_fit_coefficients():
    x = np.arange(10.0)
    res = newey_west_ols(2.0 + 3.0 * x, x, lags=2)
    assert res.coef == pytest.approx([2.0, 3.0])
    assert res.r2 == pytest.approx(1.0)


def test_lag_zero_is_white_hc0():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    y = 0.5 * x + rng.normal(size=200) * (1 + np.abs(x))
    res = newey_west_ols(y, x, lags=0)
    X = np.column_stack([np.ones(200), x])
    u = y - X @ res.coef
    inv = np.linalg.inv(X.T @ X)
    white = inv @ (X.T * u**2) @ X @ inv
    assert res.se == pytest.approx(np.sqrt(np.diag(white)))


def test_overlapping_sums_inflate_error_and_nan_rows_dropped():
    rng = np.random.default_rng(2)
    e = rng.normal(size=3000)
    x = rng.normal(size=3000)
    h = 20                                     # overlapping 20-step sums of noise
    y = np.convolve(e, np.ones(h), mode="full")[: len(e)]
    plain = newey_west_ols(y, x, lags=0)
    hac = newey_west_ols(y, x, lags=h - 1)
    assert hac.se[1] > plain.se[1]             # x is iid, but the intercept must widen
    assert hac.se[0] > 2 * plain.se[0]
    y2 = y.copy()
    y2[:5] = np.nan
    assert newey_west_ols(y2, x, lags=3).n == len(y) - 5


def test_constant_regressor_raises_clear_error():
    with pytest.raises(ValueError, match="collinear or constant"):
        newey_west_ols(np.arange(5.0), np.ones(5), lags=1)
