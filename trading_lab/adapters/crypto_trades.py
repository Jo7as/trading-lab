"""Historical crypto trades from the exchanges' free public dumps, as clock-aligned flow bars.

Venues (no API key, no money):

* ``binance_um``   - USD-M perps, ``data.binance.vision/data/futures/um/daily/aggTrades``
* ``binance_spot`` - spot, ``data.binance.vision/data/spot/daily/aggTrades``
* ``bybit_linear`` - USDT perps, ``public.bybit.com/trading``

Raw trades are far too large to keep (BTC perps: millions per day), so each day is
downloaded once, signed by aggressor and aggregated into fixed clock bars
(default 10 s, aligned to UTC midnight - Kim & Hansen's quarter-hour opening window is
exactly one such bar). Bars are cached per venue/symbol/day under
``artifacts/cache/crypto/``; the raw file is discarded.

Schema of normalised trades (``parse_*``): ``ts`` (UTC, tz-aware), ``price``,
``qty`` (base currency), ``side`` (+1 buyer was the taker, -1 seller was the taker).

Bars (``trades_to_bars``) cover the **full** day grid - empty bars keep zero volume and NaN
prices - so slices by clock time never drift out of phase (AGENTS.md, index spaces).
"""
from __future__ import annotations

import gzip
import io
import logging
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE = Path("artifacts/cache/crypto")
BAR_COLUMNS = ["open", "high", "low", "close", "vwap", "buy_qty", "sell_qty",
               "buy_notional", "sell_notional", "n_trades"]


@dataclass(frozen=True)
class Venue:
    name: str
    url: Callable[[str, date], str]
    parse: Callable[[bytes], pd.DataFrame]
    first_day: date           # earliest dump we rely on (older files: other formats)


# ── timestamps ──────────────────────────────────────────────────────────
def _to_utc(raw: pd.Series) -> pd.Series:
    """Epoch numbers in s / ms / us -> tz-aware UTC timestamps (unit picked by magnitude).

    Binance spot switched from ms to us on 2025-01-01, Bybit writes seconds with decimals.
    """
    x = pd.to_numeric(raw, errors="raise").astype("float64")
    if x.empty:
        return pd.to_datetime(x, unit="ms", utc=True)
    mag = float(np.nanmedian(np.abs(x.to_numpy())))
    if mag < 1e11:
        unit, scale = "us", 1e6        # seconds -> microseconds (keeps sub-second decimals)
    elif mag < 1e14:
        unit, scale = "us", 1e3        # milliseconds
    else:
        unit, scale = "us", 1.0        # microseconds
    return pd.to_datetime(np.round(x.to_numpy() * scale).astype("int64"), unit=unit, utc=True)


# ── parsers ─────────────────────────────────────────────────────────────
_BINANCE_AGG = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
                "transact_time", "is_buyer_maker"]


def _unzip_single(blob: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one csv in zip, got {zf.namelist()}")
        return zf.read(names[0])


def parse_binance_aggtrades(blob: bytes) -> pd.DataFrame:
    """Binance aggTrades (zip or plain csv, with or without header) -> normalised trades.

    ``is_buyer_maker = true`` means the buyer rested in the book, i.e. the **seller** was
    the aggressor -> side -1.
    """
    raw = _unzip_single(blob) if blob[:2] == b"PK" else blob
    first = raw.split(b"\n", 1)[0].lower()
    has_header = b"price" in first or b"agg_trade_id" in first
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None,
                     names=None if has_header else _BINANCE_AGG)
    df.columns = [str(c).strip().lower() for c in df.columns]
    maker = df["is_buyer_maker"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if maker.isna().any():
        raise ValueError("unexpected is_buyer_maker values")
    return _normalised(ts=_to_utc(df["transact_time"]), price=df["price"], qty=df["quantity"],
                       side=np.where(maker.to_numpy(dtype=bool), -1, 1))


def parse_bybit_trades(blob: bytes) -> pd.DataFrame:
    """Bybit public trading dump (csv.gz) -> normalised trades. ``side`` is the taker side."""
    raw = gzip.decompress(blob) if blob[:2] == b"\x1f\x8b" else blob
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [str(c).strip() for c in df.columns]
    side = df["side"].astype(str).str.strip().str.lower().map({"buy": 1, "sell": -1})
    if side.isna().any():
        raise ValueError("unexpected Bybit side values")
    return _normalised(ts=_to_utc(df["timestamp"]), price=df["price"], qty=df["size"],
                       side=side.to_numpy(dtype="int8"))


def _normalised(*, ts: pd.Series, price, qty, side) -> pd.DataFrame:
    out = pd.DataFrame({
        "ts": pd.DatetimeIndex(ts),
        "price": pd.to_numeric(price).astype("float64").to_numpy(),
        "qty": pd.to_numeric(qty).astype("float64").to_numpy(),
        "side": np.asarray(side, dtype="int8"),
    })
    if (out["price"] <= 0).any() or (out["qty"] <= 0).any():
        raise ValueError("non-positive price or quantity in trade dump")
    # stable sort keeps the exchange's sequence for trades with equal timestamps
    return out.sort_values("ts", kind="stable").reset_index(drop=True)


VENUES: dict[str, Venue] = {
    "binance_um": Venue(
        "binance_um",
        lambda s, d: (f"https://data.binance.vision/data/futures/um/daily/aggTrades/"
                      f"{s}/{s}-aggTrades-{d:%Y-%m-%d}.zip"),
        parse_binance_aggtrades, date(2020, 1, 1)),
    "binance_spot": Venue(
        "binance_spot",
        lambda s, d: (f"https://data.binance.vision/data/spot/daily/aggTrades/"
                      f"{s}/{s}-aggTrades-{d:%Y-%m-%d}.zip"),
        parse_binance_aggtrades, date(2020, 1, 1)),
    "bybit_linear": Venue(
        "bybit_linear",
        lambda s, d: f"https://public.bybit.com/trading/{s}/{s}{d:%Y-%m-%d}.csv.gz",
        parse_bybit_trades, date(2020, 3, 25)),
}

# Our markets -> symbol per venue (all USDT-quoted, base-currency quantities)
SYMBOLS: dict[str, dict[str, str]] = {
    "BTC": {"binance_um": "BTCUSDT", "binance_spot": "BTCUSDT", "bybit_linear": "BTCUSDT"},
    "SOL": {"binance_um": "SOLUSDT", "binance_spot": "SOLUSDT", "bybit_linear": "SOLUSDT"},
}


# ── aggregation ─────────────────────────────────────────────────────────
def day_grid(day: date, bar: str = "10s") -> pd.DatetimeIndex:
    """All bar starts of one UTC day, e.g. 8640 for 10 s."""
    start = pd.Timestamp(day, tz="UTC")
    return pd.date_range(start, start + pd.Timedelta(days=1), freq=bar, inclusive="left")


def trades_to_bars(trades: pd.DataFrame, day: date, bar: str = "10s") -> pd.DataFrame:
    """Aggregate one day of normalised trades into flow bars on the full day grid.

    A bar labelled t covers [t, t + bar). Trades outside the day are an error, not
    silently dropped: they mean the dump and the requested day disagree.
    """
    grid = day_grid(day, bar)
    end = grid[0] + pd.Timedelta(days=1)
    if len(trades) and ((trades["ts"] < grid[0]).any() or (trades["ts"] >= end).any()):
        raise ValueError(f"trades outside {day} in dump")
    t = trades.assign(key=trades["ts"].dt.floor(bar),
                      notional=trades["price"] * trades["qty"])
    buy = t["side"] > 0
    t["buy_qty"] = np.where(buy, t["qty"], 0.0)
    t["sell_qty"] = np.where(buy, 0.0, t["qty"])
    t["buy_notional"] = np.where(buy, t["notional"], 0.0)
    t["sell_notional"] = np.where(buy, 0.0, t["notional"])
    g = t.groupby("key", sort=True)
    bars = pd.DataFrame({
        "open": g["price"].first(), "high": g["price"].max(), "low": g["price"].min(),
        "close": g["price"].last(),
        "vwap": g["notional"].sum() / g["qty"].sum(),
        "buy_qty": g["buy_qty"].sum(), "sell_qty": g["sell_qty"].sum(),
        "buy_notional": g["buy_notional"].sum(), "sell_notional": g["sell_notional"].sum(),
        "n_trades": g["price"].size(),
    }).reindex(grid)
    vol_cols = ["buy_qty", "sell_qty", "buy_notional", "sell_notional", "n_trades"]
    bars[vol_cols] = bars[vol_cols].fillna(0.0)
    bars["n_trades"] = bars["n_trades"].astype("int64")
    bars.index.name = "ts"
    return bars[BAR_COLUMNS]


# ── download + cache ────────────────────────────────────────────────────
def _http_get(url: str) -> bytes | None:
    """GET with a few retries; None when the exchange has no file for that day (404)."""
    import httpx

    for attempt in range(3):
        try:
            r = httpx.get(url, timeout=120.0, follow_redirects=True)
        except httpx.HTTPError as exc:
            logger.warning("download failed (%s), attempt %d: %s", url, attempt + 1, exc)
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.content
    raise RuntimeError(f"download failed after retries: {url}")


def _cache_path(cache_dir: Path, venue: str, symbol: str, day: date, bar: str) -> Path:
    return cache_dir / venue / symbol / bar / f"{day:%Y-%m-%d}.pkl"


def load_day(venue: str, symbol: str, day: date, *, bar: str = "10s",
             cache_dir: Path = CACHE, fetch: Callable[[str], bytes | None] = _http_get,
             allow_download: bool = True) -> pd.DataFrame | None:
    """Flow bars for one venue/symbol/day; downloads once, then reads the cache.

    Returns None when the venue has no dump for that day (cached as a marker, so the
    404 is not retried on every run).
    """
    v = VENUES[venue]
    path = _cache_path(cache_dir, venue, symbol, day, bar)
    missing = path.with_suffix(".missing")
    if path.exists():
        return pd.read_pickle(path)
    if missing.exists():
        return None
    if not allow_download:
        raise FileNotFoundError(f"not cached and downloads disabled: {path}")
    blob = fetch(v.url(symbol, day))
    path.parent.mkdir(parents=True, exist_ok=True)
    if blob is None:
        missing.touch()
        return None
    bars = trades_to_bars(v.parse(blob), day, bar)
    tmp = path.with_suffix(".tmp")
    bars.to_pickle(tmp)
    tmp.replace(path)
    return bars


def load_bars(venue: str, market: str, start: date, end: date, *, bar: str = "10s",
              cache_dir: Path = CACHE, fetch: Callable[[str], bytes | None] = _http_get,
              allow_download: bool = True) -> pd.DataFrame:
    """Flow bars for ``market`` (e.g. "BTC") on one venue, days ``start`` .. ``end`` inclusive.

    Days without a dump stay on the grid as empty bars (zero volume, NaN prices) and are
    listed in ``df.attrs["missing_days"]`` - a data gap must be visible, not interpolated.
    """
    if end < start:
        raise ValueError("end before start")
    symbol = SYMBOLS[market][venue]
    frames, missing = [], []
    day = start
    while day <= end:
        bars = load_day(venue, symbol, day, bar=bar, cache_dir=cache_dir, fetch=fetch,
                        allow_download=allow_download)
        if bars is None:
            missing.append(day)
            bars = trades_to_bars(_EMPTY_TRADES, day, bar)
        frames.append(bars)
        day += timedelta(days=1)
    out = pd.concat(frames)
    out.attrs["missing_days"] = missing
    out.attrs["venue"], out.attrs["market"], out.attrs["bar"] = venue, market, bar
    return out


_EMPTY_TRADES = pd.DataFrame({"ts": pd.DatetimeIndex([], tz="UTC"), "price": [], "qty": [],
                              "side": np.array([], dtype="int8")})
