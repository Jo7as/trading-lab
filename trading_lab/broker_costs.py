"""Real broker / prop-account costs from the MT5 logger (``mt5/TradingLabCostLogger.mq5``).

The logger writes three ';'-separated files per account login into MQL5/Files:

* ``tl_spreads_<login>.csv`` - bid/ask snapshots per symbol (UTC);
* ``tl_symbols_<login>.csv`` - once a day: swap mode/values, swap as fraction p.a. of the
  notional (``swap_long_pa`` / ``swap_short_pa``, computed in the EA), contract data;
* ``tl_probes_<login>.csv`` - optional demo slippage probes (open + close fills, commission).

``summary`` turns them into the numbers the backtests use - one-way cost in bp at 15:00 New
York (S-007 rebalance time) and swaps p.a. per side - and ``write_override`` stores them for
``studies/tsmom_check.py --broker <file>``. Measured costs replace assumed ones; this makes
the cost model more realistic, it does not change any screening threshold.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

NY = "America/New_York"


def _read(folder: Path, kind: str, login: str) -> pd.DataFrame:
    f = Path(folder) / f"tl_{kind}_{login}.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f, sep=";")
    if "utc" in df:
        df["utc"] = pd.to_datetime(df["utc"], format="%Y.%m.%d %H:%M:%S", utc=True)
    return df


def load(folder: str | Path, login: str) -> dict[str, pd.DataFrame]:
    return {k: _read(Path(folder), k, login) for k in ("spreads", "symbols", "probes")}


def spread_bp_at(spreads: pd.DataFrame, ny_hour: int = 15, window_min: int = 30) -> pd.Series:
    """Median HALF spread in bp per lab symbol within ``window_min`` after ``ny_hour`` New York."""
    if spreads.empty:
        return pd.Series(dtype=float)
    ny = spreads["utc"].dt.tz_convert(NY)
    m = (ny.dt.hour == ny_hour) & (ny.dt.minute < window_min) & (ny.dt.dayofweek < 5)
    return spreads[m].groupby("lab")["spread_bp"].median() / 2


def swaps_pa(symbols: pd.DataFrame) -> pd.DataFrame:
    """Median swap per side as a fraction p.a. of the notional (negative = the position pays)."""
    if symbols.empty:
        return pd.DataFrame(columns=["swap_long_pa", "swap_short_pa"])
    s = symbols.replace(1.7976931348623157e308, np.nan)          # EMPTY_VALUE of MQL5
    return s.groupby("lab")[["swap_long_pa", "swap_short_pa"]].median()


def probe_costs(probes: pd.DataFrame, symbols: pd.DataFrame) -> pd.DataFrame:
    """Mean slippage (bp, > 0 worse) and commission per side (bp of the notional) per symbol."""
    if probes.empty:
        return pd.DataFrame(columns=["slip_bp", "comm_bp_side", "n_probes"])
    last = symbols.sort_values("utc").groupby("lab").last() if not symbols.empty else pd.DataFrame()
    rows = {}
    for lab, g in probes.groupby("lab"):
        comm_bp = np.nan
        if lab in last.index and last.loc[lab, "tick_size"] > 0:
            tv, ts = last.loc[lab, "tick_value"], last.loc[lab, "tick_size"]
            notional = g["volume"] * g["fill"] * tv / ts
            comm_bp = float((g["commission"].abs() / notional).mean() * 1e4)
        rows[lab] = {"slip_bp": float(g["slip_bp"].mean()), "comm_bp_side": comm_bp, "n_probes": int(len(g))}
    return pd.DataFrame(rows).T


def summary(folder: str | Path, login: str) -> pd.DataFrame:
    d = load(folder, login)
    out = pd.DataFrame({"half_spread_bp_15ny": spread_bp_at(d["spreads"])})
    out = out.join(swaps_pa(d["symbols"]), how="outer").join(probe_costs(d["probes"], d["symbols"]), how="left")
    slip = out["slip_bp"].fillna(0.5) if "slip_bp" in out else 0.5           # unmeasured: the spec's 0.5 bp
    comm = out["comm_bp_side"].fillna(0.0) if "comm_bp_side" in out else 0.0
    out["cost_bp_side"] = out["half_spread_bp_15ny"] + slip.clip(lower=0) + comm
    return out


def write_override(df: pd.DataFrame, path: str | Path, *, source: str) -> None:
    payload = {"source": source,
               "cost_bp": {k: round(float(v), 4) for k, v in df["cost_bp_side"].dropna().items()},
               "swaps_pa": {k: [float(r["swap_long_pa"]), float(r["swap_short_pa"])]
                            for k, r in df[["swap_long_pa", "swap_short_pa"]].dropna().iterrows()}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1))


__all__ = ["load", "spread_bp_at", "swaps_pa", "probe_costs", "summary", "write_override"]
