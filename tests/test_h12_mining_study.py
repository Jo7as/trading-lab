"""H12 mining machinery on small synthetic tables (no network)."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from studies.h12_mining_study import (combo_codes, control_p, family_neighbors, mine_groups,  # noqa: E402
                                      period_of, unit_rows)


def test_periods_follow_the_amendment():
    assert period_of(pd.Timestamp("2017-06-01"), "US100") is None
    assert period_of(pd.Timestamp("2017-06-01"), "XAUUSD") == "mining"
    assert period_of(pd.Timestamp("2018-05-02"), "US100") == "mining"
    assert period_of(pd.Timestamp("2021-03-01"), "GER40") == "validation"
    assert period_of(pd.Timestamp("2023-03-01"), "GER40") == "confirmation"
    assert period_of(pd.Timestamp("2025-01-02"), "GER40") is None


def test_combo_codes_and_missing():
    df = pd.DataFrame({"a": [0, 2, -1], "b": [1, 1, 1]})
    assert combo_codes(df, ("a", "b"), {"a": 3, "b": 2}).tolist() == [1, 5, -1]


def test_family_stats_equal_brute_force():
    rng = np.random.default_rng(3)
    n = 3000
    # one token group (radices 3,3: two mutable letters -> 9 values) and one context group (2 values)
    df = pd.DataFrame({"tok": rng.integers(0, 9, n), "ctx": rng.integers(0, 2, n)})
    groups = ("tok", "ctx")
    card = {"tok": 9, "ctx": 2}
    radices = {"tok": (3, 3)}
    mut = {"tok": (True, True)}
    y = (rng.random(n) < 0.2).astype(float)
    p = np.full(n, 0.2)
    code = combo_codes(df, groups, card)
    res = mine_groups(code, [y], [p], lambda u: family_neighbors(u, groups, card, mut, radices))
    for j, c in enumerate(res["uniq"]):
        tok, ctx = divmod(int(c), 2)
        d0, d1 = divmod(tok, 3)
        fam_tok = {tok} | {a * 3 + d1 for a in (d0 - 1, d0 + 1) if 0 <= a < 3} | \
                  {d0 * 3 + b for b in (d1 - 1, d1 + 1) if 0 <= b < 3}
        rows = df["tok"].isin(fam_tok).to_numpy() & (df["ctx"] == ctx).to_numpy()
        assert res["family"]["n"][j] == rows.sum()
        assert abs(res["family"]["o0"][j] - y[rows].sum()) < 1e-9
        assert (unit_rows(code, res, j, True) == rows).all()
        assert res["exact"]["n"][j] == (code == c).sum()


def test_control_p_is_stratum_mean():
    df = pd.DataFrame({"m": ["a", "a", "b", "b"], "q": [0, 0, 1, 1]})
    assert control_p(df, np.array([1.0, 0.0, 1.0, 1.0]), ["m", "q"]).tolist() == [0.5, 0.5, 1.0, 1.0]


def test_prior_median_skips_gaps():
    from studies.h12_mining_study import NORM, prior_median
    x = pd.Series(np.arange(1.0, 31.0))
    x[5] = np.nan
    m = prior_median(x)
    # 21st valid value (index 21) uses the 20 valid values before it: 1..5, 7..21 -> median 11.5
    assert m[21] == np.median(np.r_[np.arange(1.0, 6.0), np.arange(7.0, 22.0)])
    assert np.isnan(m[5]) and m.notna().sum() == 30 - 1 - NORM


def test_directional_stats_ignores_pure_volatility():
    from studies.h12_mining_study import directional_stats
    # both directions rise equally above the control: no directional excess
    y_d = np.array([1.0, 0.0, 0.0, 0.0])
    y_o = np.array([0.0, 1.0, 0.0, 0.0])
    p = np.full(4, 0.1)
    r = directional_stats(y_d, y_o, p, p, np.arange(4), np.array(["a", "a", "b", "b"]))
    assert r["dir_excess"] == 0.0
    # only D rises
    r = directional_stats(np.array([1.0, 1.0, 0.0, 1.0]), np.zeros(4), p, p, np.arange(4),
                          np.array(["a", "a", "b", "b"]))
    assert r["dir_excess"] == 0.75 and r["dir_markets_up"] == 2
