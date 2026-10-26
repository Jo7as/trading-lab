"""Tests for the Dukascopy swap table (the measured broker markup)."""
from __future__ import annotations

import pytest

from trading_lab.duka_swaps import (
    POLICIES,
    carry_pct,
    load_table,
    markup_bp_yr,
    markup_by_pair,
    pip_size,
)

# spot prices as of the last cached daily bar (2026-09-30) and the broker's pips
PRICE = {"AUD/CAD": 0.98853, "AUD/CHF": 0.58031, "AUD/JPY": 109.432,
         "EUR/HUF": 366.02, "EUR/PLN": 4.36092, "EUR/USD": 1.13268}
PIP = {"AUD/CAD": 1e-4, "AUD/CHF": 1e-4, "AUD/JPY": 1e-2,
       "EUR/HUF": 1e-2, "EUR/PLN": 1e-4, "EUR/USD": 1e-4}


def test_table_loads_with_all_policies() -> None:
    t = load_table()
    assert len(t) == 349 * len(POLICIES)
    assert set(t["policy"]) == set(POLICIES)
    row = t[(t["pair"] == "AUD/CAD") & (t["policy"] == "regular")].iloc[0]
    assert row["long"] == pytest.approx(-0.34530033)
    assert row["short"] == pytest.approx(-0.75030166)


def test_sign_convention_reproduces_the_fred_differential() -> None:
    """The mid of the two quotes is the fair carry (FRED says EUR/USD -1.9 %)."""
    assert carry_pct("EUR/USD", PRICE["EUR/USD"], PIP["EUR/USD"]) == pytest.approx(-1.42, abs=0.02)
    assert carry_pct("AUD/CHF", PRICE["AUD/CHF"], PIP["AUD/CHF"]) == pytest.approx(4.50, abs=0.02)
    assert carry_pct("USD/CAD", 1.4236, 1e-4) == pytest.approx(1.57, abs=0.02)


def test_markup_is_positive_on_both_sides() -> None:
    # AUD/CAD: 0.345 pts long, -0.750 pts short -> 0.2025 pts/day kept per side.
    assert markup_bp_yr("AUD/CAD", PRICE["AUD/CAD"], PIP["AUD/CAD"]) == pytest.approx(74.8, abs=0.2)
    assert markup_bp_yr("EUR/USD", PRICE["EUR/USD"], PIP["EUR/USD"]) == pytest.approx(54.4, abs=0.2)


def test_markup_is_flat_across_differentials_not_proportional() -> None:
    """AUD majors keep the same markup although their carry differs 2x."""
    cad = markup_bp_yr("AUD/CAD", PRICE["AUD/CAD"], PIP["AUD/CAD"])
    chf = markup_bp_yr("AUD/CHF", PRICE["AUD/CHF"], PIP["AUD/CHF"])
    jpy = markup_bp_yr("AUD/JPY", PRICE["AUD/JPY"], PIP["AUD/JPY"])
    assert max(cad, chf, jpy) / min(cad, chf, jpy) < 1.05   # ~75 bp for all three
    assert carry_pct("AUD/CHF", PRICE["AUD/CHF"], PIP["AUD/CHF"]) / \
        carry_pct("AUD/CAD", PRICE["AUD/CAD"], PIP["AUD/CAD"]) > 2.0
    # a percentage-of-differential model would have to differ by the same 2x


def test_exotic_quotes_cost_more_than_majors() -> None:
    assert markup_bp_yr("EUR/HUF", PRICE["EUR/HUF"], PIP["EUR/HUF"]) == pytest.approx(200.5, abs=0.5)
    assert markup_bp_yr("EUR/PLN", PRICE["EUR/PLN"], PIP["EUR/PLN"]) == pytest.approx(101.0, abs=0.5)
    assert markup_bp_yr("EUR/HUF", PRICE["EUR/HUF"], PIP["EUR/HUF"]) > \
        2 * markup_bp_yr("EUR/USD", PRICE["EUR/USD"], PIP["EUR/USD"])


def test_premium_terms_are_cheaper_than_regular() -> None:
    reg = markup_bp_yr("EUR/USD", PRICE["EUR/USD"], PIP["EUR/USD"], policy="regular")
    pre = markup_bp_yr("EUR/USD", PRICE["EUR/USD"], PIP["EUR/USD"], policy="premium")
    assert pre < reg
    # the mid does not depend on the policy, only the markup does
    assert carry_pct("EUR/USD", PRICE["EUR/USD"], PIP["EUR/USD"], policy="premium") == \
        pytest.approx(carry_pct("EUR/USD", PRICE["EUR/USD"], PIP["EUR/USD"], policy="regular"))


def test_markup_by_pair_returns_a_fraction() -> None:
    m = markup_by_pair(["AUD/CAD"], PRICE, PIP)
    assert m["AUD/CAD"] == pytest.approx(0.00748, abs=1e-5)


def test_pip_size_matches_the_broker_instrument_list() -> None:
    """The convention used for the 26 carry pairs, pinned against the broker's own list."""
    expected = {
        "EUR/USD": 1e-4, "GBP/USD": 1e-4, "USD/JPY": 1e-2, "AUD/USD": 1e-4,
        "NZD/USD": 1e-4, "USD/CAD": 1e-4, "USD/CHF": 1e-4, "EUR/CHF": 1e-4,
        "EUR/JPY": 1e-2, "EUR/GBP": 1e-4, "EUR/PLN": 1e-4, "EUR/HUF": 1e-2,
        "EUR/CZK": 1e-3, "EUR/NOK": 1e-4, "EUR/SEK": 1e-4, "GBP/JPY": 1e-2,
        "GBP/CHF": 1e-4, "AUD/JPY": 1e-2, "NZD/JPY": 1e-2, "CAD/JPY": 1e-2,
        "AUD/CHF": 1e-4, "NZD/CHF": 1e-4, "AUD/CAD": 1e-4, "NZD/CAD": 1e-4,
        "AUD/NZD": 1e-4, "EUR/AUD": 1e-4,
    }
    for pair, pip in expected.items():
        assert pip_size(pair.split("/")[1]) == pytest.approx(pip), pair
