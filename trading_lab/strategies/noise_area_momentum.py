"""S-006: Noise-Area intraday momentum on index CFDs (Zarattini, Aziz, Barbon 2024).

Spec (agreed with the user 2026-10-29): ``docs/specs/noise_area_momentum.yaml``. Each rule
is a small function with a test in ``tests/test_noise_area_momentum.py``.

 1. Session in the exchange time zone (US 09:30-16:00 NY, GER40 09:00-17:30 Berlin);
    O = open of the first session minute, C1 = close of the previous session.
 2. sigma(m) = mean over the previous 14 valid sessions of |close(m) / O - 1|, m = minute
    of the session.
 3. Upper band = max(O, C1) * (1 + sigma(m)), lower band = min(O, C1) * (1 - sigma(m)).
 4. Decisions only at check times (every 30 / 15 min from 30 min after the open), on the
    mid close of the minute that ends at the check time.
 5. Long above the upper band, short below the lower band; fills at the open of the next
    minute (ask / bid) + slippage.
 6. Trailing exit at a check: long when price < max(upper band, session VWAP) (variant
    ``band``: < upper band); short mirrored. Re-entry / flip at the same check allowed.
 7. Flat at the open of the last session minute.
 8. 1 R = one daily standard deviation (14 sessions, close to close) of the position.
Sessions with < 80 % of the minutes, without an open in the first 5 minutes, or US half
days are skipped and do not enter sigma.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ..market_structure import mid_ohlc

SESSION = {"SP500": ("America/New_York", "09:30", 390), "US100": ("America/New_York", "09:30", 390),
           "US30": ("America/New_York", "09:30", 390), "GER40": ("Europe/Berlin", "09:00", 510)}
LOOKBACK = 14
MIN_COVERAGE = 0.8
OPEN_GRACE = 5


@dataclass(frozen=True)
class NoiseAreaConfig:
    asset: str = "SP500"
    pruefung: str = "30min"         # 30min | 15min
    trailing: str = "band_vwap"     # band_vwap | band
    slippage_bp: float = 0.5


@dataclass
class Prepared:
    asset: str
    open_ns: np.ndarray              # session start (UTC ns) per valid session
    close: np.ndarray                # D x L mid close, forward-filled inside the session
    vwap: np.ndarray                 # D x L session VWAP (tick volume; TWAP where no volume)
    bid_open: np.ndarray             # D x L (NaN where the minute is missing)
    ask_open: np.ndarray
    upper: np.ndarray                # D x L, NaN without 14 previous sessions
    lower: np.ndarray
    sigma_daily: np.ndarray          # D, NaN without 14 previous returns


def us_half_day(d: date) -> bool:
    """NYSE early closes (13:00): 3 July, the day after Thanksgiving, 24 December (weekdays)."""
    if d.weekday() >= 5:
        return False
    if (d.month, d.day) in ((7, 3), (12, 24)):
        return True
    if d.month == 11 and d.weekday() == 4:
        thanksgiving = 22 + (3 - date(d.year, 11, 1).weekday()) % 7
        return d.day == thanksgiving + 1
    return False


def session_frames(m1: pd.DataFrame, asset: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Valid sessions as D x L matrices (rule 1 + the skip rules); returns (open_ns, mats)."""
    tz, hhmm, L = SESSION[asset]
    h, m = (int(x) for x in hhmm.split(":"))
    loc = m1.index.tz_convert(tz)
    day = loc.normalize()
    k = ((loc - day) / pd.Timedelta(minutes=1)).astype(int) - (h * 60 + m)
    sel = (k >= 0) & (k < L)
    mid = mid_ohlc(m1)
    cols = {"o": mid["open"], "h": mid["high"], "l": mid["low"], "c": mid["close"],
            "bo": m1["open"], "ao": m1["ask_open"],
            "v": m1["volume"] if "volume" in m1 else pd.Series(0.0, index=m1.index)}
    dkey = day[sel].tz_localize(None)
    days, d_idx = np.unique(dkey.to_numpy(), return_inverse=True)
    kk = np.asarray(k[sel])
    mats = {}
    for name, s in cols.items():
        a = np.full((len(days), L), np.nan)
        a[d_idx, kk] = s.to_numpy(float)[sel]
        mats[name] = a
    present = np.isfinite(mats["c"])
    ok = present.mean(axis=1) >= MIN_COVERAGE
    ok &= present[:, :OPEN_GRACE].any(axis=1)
    if tz == "America/New_York":
        ok &= np.array([not us_half_day(pd.Timestamp(d).date()) for d in days])
    days = days[ok]
    mats = {n: a[ok] for n, a in mats.items()}
    open_ns = (pd.DatetimeIndex(days) + pd.Timedelta(hours=h, minutes=m)).tz_localize(tz) \
        .tz_convert("UTC").as_unit("ns").asi8
    return open_ns, mats


def _ffill_rows(a: np.ndarray) -> np.ndarray:
    idx = np.where(np.isfinite(a), np.arange(a.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = a[np.arange(a.shape[0])[:, None], idx]
    return out


def prepare(m1: pd.DataFrame, asset: str) -> Prepared:
    open_ns, M = session_frames(m1, asset)
    D = len(open_ns)
    close = _ffill_rows(M["c"])
    first = np.argmax(np.isfinite(M["o"]), axis=1)
    O = M["o"][np.arange(D), first]
    C = close[:, -1]
    C1 = np.r_[np.nan, C[:-1]]
    move = np.abs(close / O[:, None] - 1.0)                        # rule 2
    sigma = np.full_like(move, np.nan)
    cs = np.nancumsum(np.r_[np.zeros((1, move.shape[1])), np.nan_to_num(move)], axis=0)
    cnt = np.cumsum(np.r_[np.zeros((1, move.shape[1])), np.isfinite(move)], axis=0)
    for d in range(LOOKBACK, D):
        n = cnt[d] - cnt[d - LOOKBACK]
        sigma[d] = np.where(n > 0, (cs[d] - cs[d - LOOKBACK]) / np.maximum(n, 1), np.nan)
    upper = np.maximum(O, C1)[:, None] * (1 + sigma)               # rule 3
    lower = np.minimum(O, C1)[:, None] * (1 - sigma)
    typ = (M["h"] + M["l"] + M["c"]) / 3
    vol = np.where(np.isfinite(typ), np.nan_to_num(M["v"]), 0.0)
    tv = np.cumsum(np.nan_to_num(typ) * vol, axis=1)
    cv = np.cumsum(vol, axis=1)
    twap = np.cumsum(np.nan_to_num(typ), axis=1) / np.maximum(np.cumsum(np.isfinite(typ), axis=1), 1)
    vwap = np.where(cv > 0, tv / np.where(cv > 0, cv, 1), twap)
    ret = C / C1 - 1.0
    sd = np.full(D, np.nan)
    for d in range(LOOKBACK + 1, D + 1):
        if d < D:
            sd[d] = np.std(ret[d - LOOKBACK:d], ddof=1)                # rule 8: previous 14 returns
    return Prepared(asset, open_ns, close, vwap, M["bo"], M["ao"], upper, lower, sd)


def check_minutes(L: int, pruefung: str) -> list[int]:
    """Session minutes at which a decision is made (rule 4); the last minute is the flat exit."""
    step = int(pd.Timedelta(pruefung) / pd.Timedelta(minutes=1))
    return list(range(30, L - 1, step))


def _fill(row_o: np.ndarray, k: int) -> tuple[int, float]:
    """First quoted minute >= k in the session (the order waits for a quote)."""
    for j in range(k, len(row_o)):
        if np.isfinite(row_o[j]):
            return j, float(row_o[j])
    return -1, np.nan


def run_day(p: Prepared, cfg: NoiseAreaConfig, d: int) -> list[dict]:
    if not np.isfinite(p.sigma_daily[d]) or not np.isfinite(p.upper[d]).any():
        return []
    L = p.close.shape[1]
    slip = cfg.slippage_bp * 1e-4
    out, pos, entry, k_in = [], 0, np.nan, -1

    def close_pos(k: int, reason: str) -> None:
        nonlocal pos
        j, px = _fill(p.bid_open[d] if pos > 0 else p.ask_open[d], k)
        if j < 0:                                    # no quote left: last mid close
            j, px = L - 1, float(p.close[d, -1])
        px *= (1 - pos * slip)
        risk = p.sigma_daily[d] * entry
        out.append({"entry_time": pd.Timestamp(p.open_ns[d] + k_in * 60_000_000_000, tz="UTC"),
                    "exit_time": pd.Timestamp(p.open_ns[d] + j * 60_000_000_000, tz="UTC"),
                    "direction": pos, "entry": entry, "stop": entry - pos * risk,
                    "r": pos * (px - entry) / risk, "exit_reason": reason})
        pos = 0

    for k in check_minutes(L, cfg.pruefung):
        px, ub, lb, vw = p.close[d, k - 1], p.upper[d, k - 1], p.lower[d, k - 1], p.vwap[d, k - 1]
        if not (np.isfinite(px) and np.isfinite(ub)):
            continue
        if pos > 0 and px < (max(ub, vw) if cfg.trailing == "band_vwap" else ub):      # rule 6
            close_pos(k, "trailing")
        elif pos < 0 and px > (min(lb, vw) if cfg.trailing == "band_vwap" else lb):
            close_pos(k, "trailing")
        if pos == 0 and (px > ub or px < lb):                                           # rule 5
            d_ = 1 if px > ub else -1
            j, fill = _fill(p.ask_open[d] if d_ > 0 else p.bid_open[d], k)
            if j < 0 or j >= L - 1:
                continue
            pos, entry, k_in = d_, fill * (1 + d_ * slip), j
    if pos:
        close_pos(L - 1, "session_ende")                                                # rule 7
    return out


def run_backtest(p: Prepared, cfg: NoiseAreaConfig) -> dict:
    trades = [t for d in range(len(p.open_ns)) for t in run_day(p, cfg, d)]
    return {"trades": trades}


__all__ = ["NoiseAreaConfig", "Prepared", "SESSION", "prepare", "session_frames", "check_minutes", "run_day",
           "run_backtest", "us_half_day"]
