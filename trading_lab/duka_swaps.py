"""Dukascopy overnight-swap table: the broker's *measured* financing, not a model.

H-009 modelled the carry accrual as ``rate differential x (1 - 30%)``. That 30 %
was a guess. Dukascopy publishes the real thing on its "Overnight Swaps" page
(widget ``widgets.dukascopy.com/en/overnight-calculator``), fed by
``jetta.dukascopy.com/v1/overnight-policies/``. That endpoint is *flaky* (it
answers 503 for long stretches), so the table is kept next to this module as a CSV
export: :data:`SWAP_TABLE`, 349 instruments x 3 rollover policies.

Sign convention. ``long``/``short`` are stored as **points of the quote
currency per day**, with the client receiving ``-long`` on a long position and
``+short`` on a short one. So for a pair

    fair carry (received on a long) = -(long + short) / 2
    broker markup per side          = (long - short) / 2      (always > 0)

The convention was pinned against the FRED rate differentials: over the 26 carry
pairs ``corr(mid, differential) = +0.986`` with slope ``0.914`` and an RMSE of
0.50 % p.a. (the opposite reading gives ``corr = -0.56``). It also reproduces the
page the widget renders: AUD/CAD is quoted ``0.345 / -0.750`` there, and this
table stores ``-0.34530033 / -0.75030166``.

The markup is a **flat fee, not a share of the differential**: over the 26 carry
pairs it is a median of 74 bp/yr per side for the 21 major-quoted legs (range
54-88) and 168 bp/yr for the five exotic-quoted ones (PLN/HUF/CZK/NOK/SEK), at
differentials from -3.4 % to +4.6 % p.a. So the honest replacement for H-009's
percentage haircut is an absolute drag:

    swap = (differential - markup_long - markup_short) * days / 365

Rollover policy matters and is counter-intuitive: Dukascopy grants *Premium* to
accounts whose trading activity is > 90 % of their volume (day traders) and
*Regular* to accounts that mostly hold overnight - i.e. to a carry strategy.
Regular is the worst of the three, so it is the default here.
"""
from __future__ import annotations

import csv
import pathlib

import pandas as pd

SWAP_TABLE = pathlib.Path(__file__).with_name("duka_swaps_2026-10-24.csv")
POLICIES = ("regular", "advanced", "premium")
LOT = 100_000  # the page's "Betrag 0.1 Mio"


def _number(raw: str) -> float:
    """Parse the export's numbers, which use a decimal comma and a dot group."""
    raw = raw.strip()
    if "," in raw:
        return float(raw.replace(".", "").replace(",", "."))
    return float(raw)


def load_table(path: pathlib.Path | str | None = None) -> pd.DataFrame:
    """The swap table as ``[pair, policy, long, short]``, in points/day."""
    p = pathlib.Path(path) if path is not None else SWAP_TABLE
    with open(p, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    out = [
        {"pair": r["Instrument"], "category": r["Unterkategorie"],
         "base": r["Basis"], "quote": r["Quote"], "policy": pol,
         "long": _number(r[f"{pol}_long"]), "short": _number(r[f"{pol}_short"])}
        for r in rows for pol in POLICIES
    ]
    return pd.DataFrame(out)


def carry_pct(pair: str, price: float, pip_value: float, *,
              policy: str = "regular", table: pd.DataFrame | None = None) -> float:
    """The table's own implied fair carry, in % p.a. of the position notional."""
    long_, short = _legs(pair, policy, table)
    return -(long_ + short) / 2.0 * pip_value / price * 365.0 * 100.0


def markup_bp_yr(pair: str, price: float, pip_value: float, *,
                 policy: str = "regular", table: pd.DataFrame | None = None) -> float:
    """The broker's markup per side, in bp per year of the position notional."""
    long_, short = _legs(pair, policy, table)
    return (long_ - short) / 2.0 * pip_value / price * 365.0 * 1e4


def markup_by_pair(pairs, prices: dict[str, float], pip_values: dict[str, float], *,
                   policy: str = "regular", table: pd.DataFrame | None = None
                   ) -> dict[str, float]:
    """``{pair: markup per side in %/yr}`` for a basket, ready for the carry model."""
    return {p: markup_bp_yr(p, prices[p], pip_values[p], policy=policy, table=table) / 1e4
            for p in pairs}


def pip_size(quote: str) -> float:
    """The quote currency's pip, by the FX convention the export's points use.

    Verified against Dukascopy's own instrument list for all 26 carry pairs
    (``tests/test_duka_swaps.py::test_pip_size_matches_the_broker_instrument_list``).
    """
    if quote in {"JPY", "HUF"}:
        return 0.01
    if quote == "CZK":
        return 0.001
    return 0.0001


def _legs(pair: str, policy: str, table: pd.DataFrame | None) -> tuple[float, float]:
    df = load_table() if table is None else table
    row = df[(df["pair"] == pair) & (df["policy"] == policy)]
    if row.empty:
        raise KeyError(f"{pair} not in the swap table ({policy})")
    return float(row["long"].iloc[0]), float(row["short"].iloc[0])
