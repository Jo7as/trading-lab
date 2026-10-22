"""1-minute bid+ask history from Dukascopy, cached per calendar year.

Dukascopy throttles bulk requests (HTTP 429), so every finished year is
written to disk once and never fetched again; only the running year is
refreshed (at most once per day). Frames follow the ``mr_common`` layout:
open/high/low/close are bid, ask_* the ask side, UTC index of bar open times.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .mr_common import ensure_utc

DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "artifacts" / "cache" / "m1"

# Asset codes → Dukascopy instruments (index CFDs are not in the 1h map)
INSTRUMENTS: dict[str, str] = {
    "US100": "E_NQ-100",
    "SP500": "E_SandP-500",   # S&P 500 CFD (ES/MES proxy)
    "GER40": "E_DAAX",
    "XAUUSD": "XAU/USD",
    "JP225": "E_N225Jap",   # Nikkei 225 CFD (Tokyo cash open 09:00 JST)
}


def _fetch_year_side(instrument: str, year: int, side: str, pause: float) -> pd.DataFrame:
    import dukascopy_python as duka

    offer = duka.OFFER_SIDE_ASK if side == "ask" else duka.OFFER_SIDE_BID
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    parts: list[pd.DataFrame] = []
    # Quarter-sized requests keep single calls small and the server calm
    for month in (1, 4, 7, 10):
        start = datetime(year, month, 1)
        end = datetime(year + 1, 1, 1) if month == 10 else datetime(year, month + 3, 1)
        if start >= now:
            break
        df = duka.fetch(
            instrument, duka.INTERVAL_MIN_1, offer, start, min(end, now), max_retries=10
        )
        if not df.empty:
            parts.append(df[df.index < pd.Timestamp(end, tz="UTC")])
        time.sleep(pause)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).rename(columns=str.lower)
    return ensure_utc(out)


def _year_path(cache_dir: Path, asset: str, year: int, side: str, current: bool) -> Path:
    safe = re.sub(r"[^A-Za-z0-9]+", "", asset)
    tag = f"_{datetime.now(timezone.utc).date().isoformat()}" if current else ""
    return cache_dir / f"{safe}_1m_{side}_{year}{tag}.pkl"


def load_minute_bid_ask(
    asset: str,
    year_start: int,
    *,
    cache_dir: Path | None = None,
    pause: float = 1.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Bid OHLCV + ask_* 1-minute bars from ``year_start`` until now."""
    instrument = INSTRUMENTS.get(asset, asset)
    cache = cache_dir or DEFAULT_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    this_year = datetime.now(timezone.utc).year

    years: list[pd.DataFrame] = []
    for year in range(year_start, this_year + 1):
        sides: dict[str, pd.DataFrame] = {}
        for side in ("bid", "ask"):
            path = _year_path(cache, asset, year, side, current=year == this_year)
            if path.exists():
                sides[side] = pd.read_pickle(path)
                continue
            t0 = time.time()
            df = _fetch_year_side(instrument, year, side, pause)
            if not df.empty:
                df.to_pickle(path)
            if verbose:
                print(f"  [m1] {asset} {year} {side}: {len(df)} bars in {time.time() - t0:.0f}s", flush=True)
            sides[side] = df
        bid, ask = sides["bid"], sides["ask"]
        if bid.empty or ask.empty:
            continue
        joined = bid.join(ask[["open", "high", "low", "close"]].add_prefix("ask_"), how="inner")
        years.append(joined)

    if not years:
        return pd.DataFrame()
    out = pd.concat(years)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    # A crossed quote would hand out negative spreads — never let ask sit below bid
    for col in ("open", "high", "low", "close"):
        out[f"ask_{col}"] = np.maximum(out[f"ask_{col}"], out[col])
    return out
