"""Data audit per ``pipeline_drop_in_research/docs/DATA_AUDIT_SPEC.md`` (A1–A6).

Design decision that matters: corrupt bars are **classified, not deleted**. A
blanket mask (drop everything beyond N× the median, plus the roll window) also
deletes real crisis bars — 2020-03, 2025-04 — which is exactly where trends pay
and stops trigger, so it flatters every backtest. Here each candidate is put in a
class and only ``spike`` / ``session_edge`` / ``unclear`` are excluded from
signals; ``spread_event`` and ``news_like`` are **kept and marked**.

Classes (A1): spike (tick error), spread_explosion, session_edge, news_like,
unclear. Trigger: ``range / same-hour trailing median range >= 20`` **or**
``|Δmid| >= 15 x same-hour trailing sigma_1m``.

A2 session/day bounds, A3 DST, A4 spread by 15-min bucket, A5 consistency (incl.
a yfinance daily cross-check), A6 the per-instrument ``audit_passed`` flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time as dtime
from typing import Any

import numpy as np
import pandas as pd

# core session per instrument (local time of the home market) + optional yfinance proxy
SESSIONS: dict[str, dict[str, Any]] = {
    "US100": {"tz": "America/New_York", "open": "09:30", "close": "16:00", "proxy": "^NDX", "cal": "nyse"},
    "US30": {"tz": "America/New_York", "open": "09:30", "close": "16:00", "proxy": "^DJI", "cal": "nyse"},
    "SP500": {"tz": "America/New_York", "open": "09:30", "close": "16:00", "proxy": "^GSPC", "cal": "nyse"},
    "GER40": {"tz": "Europe/Berlin", "open": "09:00", "close": "17:30", "proxy": "^GDAXI", "cal": "xetra"},
    "UK100": {"tz": "Europe/London", "open": "08:00", "close": "16:30", "proxy": "^FTSE", "cal": "nyse"},
    "JP225": {"tz": "Asia/Tokyo", "open": "09:00", "close": "15:00", "proxy": "^N225", "cal": "tse"},
    "XAUUSD": {"tz": "America/New_York", "open": "08:20", "close": "17:00", "proxy": "GC=F", "cal": "nyse"},
    "XAGUSD": {"tz": "America/New_York", "open": "08:20", "close": "17:00", "proxy": "SI=F", "cal": "nyse"},
    "COPPER": {"tz": "America/New_York", "open": "08:00", "close": "13:00", "proxy": "HG=F", "cal": "nyse"},
    "BRENT": {"tz": "America/New_York", "open": "09:00", "close": "14:30", "proxy": "BZ=F", "cal": "nyse"},
    "UST10Y": {"tz": "America/New_York", "open": "08:20", "close": "15:00", "proxy": "ZN=F", "cal": "nyse"},
}

# the "volume/volatility peak at the local open" check is only meaningful where a
# cash session actually opens with a spike (equity indices), not for 24h metals/FX
OPEN_CHECK_ASSETS = ("US100", "US30", "SP500", "GER40", "UK100", "JP225")
THIN_SHARE_MAX = 0.10       # thin days are excluded and documented; more than this is a real defect
CROSS_CHECK_MAX = 1.0       # median |daily close CFD vs yfinance proxy| in %

RANGE_RATIO = 20.0          # A1 trigger: range / same-hour median range
SIGMA_MULT = 15.0           # A1 trigger: |Δmid| / same-hour sigma
SPREAD_MULT = 20.0          # spread explosion
TRAIL_DAYS = 60             # trailing window for the same-hour medians
ROLL_UTC = 21.0             # the CFD daily roll sits at 21:00-22:00 UTC (17:00 NY)


def _minutes(hhmm: str) -> int:
    h, m = (int(x) for x in hhmm.split(":"))
    return h * 60 + m


@dataclass
class AuditReport:
    asset: str
    bars: int
    days: int
    passed: bool
    counts: dict[str, int] = field(default_factory=dict)
    thin_days: int = 0
    dst_ok: bool = True
    dst_notes: list[str] = field(default_factory=list)
    consistency: dict[str, int] = field(default_factory=dict)
    daily_cross_check_pct: float | None = None
    extremes: list[dict] = field(default_factory=list)
    exclude_fraction: float = 0.0
    spread_table: pd.DataFrame | None = None

    def summary(self) -> str:
        c = self.counts
        return (f"{self.asset:7s} bars={self.bars:>9,} days={self.days:>4} "
                f"passed={str(self.passed):5s} spike={c.get('spike', 0):>4} "
                f"spread={c.get('spread_event', 0):>5} edge={c.get('session_edge', 0):>4} "
                f"news={c.get('news_like', 0):>4} unclear={c.get('unclear', 0):>4} "
                f"excl={self.exclude_fraction:.3%} thin={self.thin_days} "
                f"dst={'ok' if self.dst_ok else 'FAIL'} cross={self.daily_cross_check_pct}")


def _same_hour_stats(df: pd.DataFrame, local: pd.DatetimeIndex):
    """Trailing same-hour medians (range, spread) and sigma of |Δmid| (A1)."""
    mid = (df["close"] + df["ask_close"]) / 2.0
    tmp = pd.DataFrame({
        "rng": (df["high"] - df["low"]).to_numpy(),
        "spr": (df["ask_close"] - df["close"]).to_numpy(),
        "sig": mid.pct_change().abs().to_numpy(),
        "h": local.hour.to_numpy(),
        "d": local.date,
    }, index=df.index)
    per_day = tmp.groupby(["d", "h"])[["rng", "spr", "sig"]].median()
    piv_r = per_day["rng"].unstack("h")
    piv_s = per_day["spr"].unstack("h")
    piv_i = per_day["sig"].unstack("h")
    med_r = piv_r.rolling(TRAIL_DAYS, min_periods=10).median()
    med_s = piv_s.rolling(TRAIL_DAYS, min_periods=10).median()
    sig_i = piv_i.rolling(TRAIL_DAYS, min_periods=10).median()
    return tmp, med_r, med_s, sig_i


def classify(df: pd.DataFrame, asset: str) -> tuple[pd.Series, AuditReport]:
    """Return the per-bar class label and the audit report (A1, A2, A5)."""
    sess = SESSIONS[asset]
    tz = sess["tz"]
    local = df.index.tz_convert(tz)
    tmp, med_r, med_s, sig_i = _same_hour_stats(df, local)

    med_r_map = med_r.stack().rename("mr")
    med_s_map = med_s.stack().rename("ms")
    sig_i_map = sig_i.stack().rename("si")
    med_r_map.index.names = med_s_map.index.names = sig_i_map.index.names = ["d", "h"]
    joined = tmp.join(med_r_map, on=["d", "h"]).join(med_s_map, on=["d", "h"]).join(sig_i_map, on=["d", "h"])
    ratio = joined["rng"] / joined["mr"]
    sig_ratio = joined["sig"] / joined["si"]
    spread_ratio = joined["spr"] / joined["ms"]

    mid = (df["close"] + df["ask_close"]) / 2.0
    dmove = mid.pct_change()
    # continuation over the next 10 bars, used to tell news from a tick error
    fwd = mid.shift(-10) / mid - 1.0
    confirmed = (fwd.abs() >= 0.5 * dmove.abs())

    minutes = (local.hour * 60 + local.minute).to_numpy()
    open_m, close_m = _minutes(sess["open"]), _minutes(sess["close"])
    utc_hour = df.index.hour + df.index.minute / 60.0
    edge = ((minutes >= open_m) & (minutes < open_m + 5)) | \
           ((minutes > close_m - 5) & (minutes <= close_m)) | \
           (np.abs(utc_hour.to_numpy() - ROLL_UTC) <= 5 / 60)

    candidate = (ratio >= RANGE_RATIO) | (sig_ratio >= SIGMA_MULT)
    spread_event = candidate & (spread_ratio >= SPREAD_MULT) & ~edge
    spike = candidate & ~spread_event & (~confirmed | (df["ask_close"] <= df["close"]))
    news_like = candidate & ~spread_event & ~spike & confirmed & ~edge
    session_edge = candidate & edge & ~spread_event
    unclear = candidate & ~(spread_event | spike | news_like | session_edge)

    label = pd.Series("normal", index=df.index, dtype=object)
    label[unclear] = "unclear"
    label[news_like] = "news_like"
    label[session_edge] = "session_edge"
    label[spike] = "spike"
    label[spread_event] = "spread_event"

    # A2: thin day = well below the instrument's OWN median session length
    # (the nominal window ignores session breaks, e.g. the JP225 lunch hour)
    in_core = (minutes >= open_m) & (minutes < close_m)
    per_day = pd.Series(1, index=local[in_core].normalize()).groupby(level=0).sum()
    typical = float(per_day.median()) if len(per_day) else 0.0
    thin = int((per_day < 0.90 * typical).sum()) if typical else 0

    # A5 consistency
    consistency = {
        "ask_lt_bid": int((df["ask_close"] < df["close"]).sum()),
        "ohlc_violations": int(((df["low"] > df["open"]) | (df["open"] > df["high"]) |
                                (df["low"] > df["close"]) | (df["close"] > df["high"])).sum()),
        "unsorted": int(not df.index.is_monotonic_increasing),
        "duplicates": int(df.index.duplicated().sum()),
    }

    thin_dates = set(per_day.index[per_day < 0.90 * typical]) if typical else set()
    rep_thin = thin
    exclude = label.isin(("spike", "session_edge", "unclear")) | \
        pd.Series(local.normalize().to_numpy(), index=df.index).isin(
            pd.DatetimeIndex(sorted(thin_dates)))
    rep = AuditReport(
        asset=asset, bars=len(df), days=int(local.normalize().nunique()), passed=False,
        counts={k: int((label == k).sum()) for k in
                ("spike", "spread_event", "session_edge", "news_like", "unclear")},
        thin_days=thin, consistency=consistency,
        exclude_fraction=float(exclude.mean()),
        extremes=[{"ts": str(ts), "range_ratio": round(float(ratio.loc[ts]), 1),
                   "label": label.loc[ts]}
                  for ts in joined["rng"].nlargest(5).index],
    )
    return label, rep


def spread_by_bucket(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """A4: median/p75/p95 spread per 15-minute bucket of the local day."""
    tz = SESSIONS[asset]["tz"]
    local = df.index.tz_convert(tz)
    bucket = (local.hour * 4 + local.minute // 15)
    mid = (df["close"] + df["ask_close"]) / 2.0
    sp = (df["ask_close"] - df["close"]) / mid
    g = sp.groupby(bucket)
    return pd.DataFrame({"median": g.median(), "p75": g.quantile(0.75), "p95": g.quantile(0.95)})


def dst_check(df: pd.DataFrame, asset: str) -> tuple[bool, list[str]]:
    """A3: the local session open must be constant in local time (UTC shifts with DST).

    The opening-spike proxy is applied only to instruments with a cash open
    (``OPEN_CHECK_ASSETS``); elsewhere it is not applicable.
    """
    if asset not in OPEN_CHECK_ASSETS:
        return True, ["opening-spike check not applicable (no cash open)"]
    sess = SESSIONS[asset]
    tz = sess["tz"]
    local = df.index.tz_convert(tz)
    minutes = local.hour * 60 + local.minute
    open_m = _minutes(sess["open"])
    close_m_utc = _minutes(sess["close"])
    notes: list[str] = []
    ok = True
    for year in sorted(set(local.year)):
        m = (local.year == year)
        if not m.any():
            continue
        # proxy for the opening minute: the highest mean |1m return| (robust where
        # volume is degenerate), requiring a real spike above the rest of the day
        r = (df.loc[m, "close"].pct_change().abs())
        prof = r.groupby(minutes[m]).mean()
        first = prof[(prof.index >= open_m) & (prof.index < open_m + 120)]
        rest = prof[(prof.index >= open_m + 120) & (prof.index < close_m_utc)]
        if first.empty or rest.empty or float(rest.mean()) <= 0:
            continue
        peak = int(first.idxmax())
        if float(first.max()) < 2.0 * float(rest.mean()):
            notes.append(f"{year}: opening spike unconfirmed (flat volatility profile)")
            continue
        delta = abs(peak - open_m)
        if delta > 5:
            if delta > 30:                     # a gross mismatch means a session/tz error
                ok = False
                notes.append(f"{year}: volatility peak at {peak // 60:02d}:{peak % 60:02d} local, "
                             f"expected {sess['open']} — GROSS mismatch")
            else:
                notes.append(f"{year}: opening minute near but not at {sess['open']} "
                             f"(±{delta} min) — unconfirmed")
    return ok, notes


def build_study_frame(df: pd.DataFrame, asset: str) -> tuple[pd.DataFrame, dict]:
    """Rows a study may use: normal + news_like; spike/session_edge/unclear removed.

    Returns the cleaned frame (rows dropped so indicators stay sane) and a small
    summary. This is the audit-correct alternative to masking everything extreme.
    """
    label, rep = classify(df, asset)
    keep = label.isin(("normal", "news_like", "spread_event"))
    # drop thin days entirely (documented in the report)
    local = df.index.tz_convert(SESSIONS[asset]["tz"])
    in_core = pd.Series(local.hour * 60 + local.minute).between(
        _minutes(SESSIONS[asset]["open"]), _minutes(SESSIONS[asset]["close"]) - 1).to_numpy()
    per_day = pd.Series(1, index=local[in_core].normalize()).groupby(level=0).sum()
    typical = float(per_day.median()) if len(per_day) else 0.0
    thin = per_day[per_day < 0.90 * typical] if typical else per_day.iloc[:0]
    day_ok = ~pd.Series(local.normalize().to_numpy(), index=df.index).isin(
        pd.DatetimeIndex(thin.index))
    out = df[keep.to_numpy() & day_ok.to_numpy()]
    return out, {"kept": int(len(out)), "dropped": int(len(df) - len(out)),
                 "thin_days": int(len(thin))}
