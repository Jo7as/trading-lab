"""Real interest rates from FRED (free, no key) for the FX carry table.

The earlier carry test used hand-written annual policy-rate averages. This module
replaces them with **measured monthly 3-month interbank rates** (OECD series via
FRED), cached under ``artifacts/cache/rates/`` so tests and re-runs stay offline.

Series that the OECD discontinued end early; the loader reports the coverage and
the carry study falls back to the last available value for the missing months
(flagged in the output).
"""
from __future__ import annotations

import pathlib
import urllib.request

import pandas as pd

FRED_SERIES: dict[str, str] = {
    "USD": "IR3TIB01USM156N", "EUR": "IR3TIB01EZM156N", "JPY": "IR3TIB01JPM156N",
    "GBP": "IR3TIB01GBM156N", "CHF": "IR3TIB01CHM156N", "CAD": "IR3TIB01CAM156N",
    "AUD": "IR3TIB01AUM156N", "NZD": "IR3TIB01NZM156N", "SEK": "IR3TIB01SEM156N",
    "NOK": "IR3TIB01NOM156N", "PLN": "IR3TIB01PLM156N", "HUF": "IR3TIB01HUM156N",
    "CZK": "IR3TIB01CZM156N", "MXN": "IR3TIB01MXM156N",
}
CACHE = pathlib.Path("artifacts/cache/rates")


def _series(ccy: str, series_id: str, *, refetch: bool = False) -> pd.Series | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{ccy}.csv"
    if refetch or not f.exists():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                f.write_bytes(r.read())
        except Exception:  # noqa: BLE001
            return None
    df = pd.read_csv(f)
    if df.shape[1] < 2:
        return None
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    s = pd.to_numeric(df["value"], errors="coerce").dropna()
    s.index = df["date"].iloc[s.index.values if hasattr(s.index, "values") else s.index]
    s.index = pd.to_datetime(s.index)
    return s.resample("ME").last()


def load_rates(currencies: tuple[str, ...] | None = None, *, refetch: bool = False) -> pd.DataFrame:
    """Monthly 3-month interbank rates in % p.a., one column per currency."""
    ccys = currencies or tuple(FRED_SERIES)
    out = {}
    for c in ccys:
        s = _series(c, FRED_SERIES[c], refetch=refetch)
        if s is not None and len(s):
            out[c] = s
    df = pd.DataFrame(out).sort_index()
    return df


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"first": df.apply(lambda s: s.dropna().index.min()),
                         "last": df.apply(lambda s: s.dropna().index.max()),
                         "n": df.notna().sum()})
