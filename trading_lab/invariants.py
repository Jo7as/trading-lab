"""Calendar and session invariants — the bug class that made a strategy look good.

Mandatory checks for any session-based strategy (must be asserted in tests):

1. **clock mapping** — a per-day slice built from a mask must start at the
   session open and end inside the session *on full 24 h data* (this is the bug
   that silently walked windows out of phase with the clock);
2. **entries/exits inside the session window**;
3. **daytrading** — no position may span two calendar days (or, optionally, be
   held past the flat time).

``assert_invariants`` raises ``AssertionError`` with the offending trades, so a
test failure is self-explaining.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import time as dtime
from typing import Any

import numpy as np
import pandas as pd

TZ_ANY = "UTC"


def _hhmm_to_min(hhmm: str) -> int:
    h, m = (int(x) for x in hhmm.split(":"))
    return h * 60 + m


def _local_index(trades: Sequence[dict], tz: str) -> pd.DatetimeIndex:
    """Entry/exit timestamps of a trade log in ``tz`` (accepts epoch or ISO)."""
    if "entry_iso" in trades[0]:
        return pd.DatetimeIndex([pd.Timestamp(t["entry_iso"]) for t in trades]).tz_convert(tz)
    return pd.DatetimeIndex([pd.Timestamp(float(t["entry_time"]), unit="s", tz="UTC")
                             for t in trades]).tz_convert(tz)


def slice_problems(local: pd.DatetimeIndex, slices: Sequence[slice],
                   *, session_open: str, session_close: str) -> list[str]:
    """Verify each slice is exactly a session on the clock of the full frame."""
    o, c = _hhmm_to_min(session_open), _hhmm_to_min(session_close)
    problems: list[str] = []
    for k, sl in enumerate(slices):
        first, last = local[sl.start], local[sl.stop - 1]
        fm, lm = first.hour * 60 + first.minute, last.hour * 60 + last.minute
        if not (o <= fm < c):
            problems.append(f"slice {k} starts {first:%H:%M}, outside [{session_open},{session_close})")
        if not (o <= lm < c):
            problems.append(f"slice {k} ends {last:%H:%M}, outside [{session_open},{session_close})")
    return problems


def outside_session(trades: Sequence[dict], tz: str, *, session_open: str,
                    session_close: str) -> list[dict]:
    """Entries *and* exits must fall inside the session window."""
    if not trades:
        return []
    loc = _local_index(trades, tz)
    ex = (pd.DatetimeIndex([pd.Timestamp(t["exit_iso"]) for t in trades])
          if "exit_iso" in trades[0]
          else pd.DatetimeIndex([pd.Timestamp(float(t["exit_time"]), unit="s", tz="UTC")
                                 for t in trades])).tz_convert(tz)
    o, c = _hhmm_to_min(session_open), _hhmm_to_min(session_close)
    bad: list[dict] = []
    for t, a, b in zip(trades, loc, ex):
        for leg, ts in (("entry", a), ("exit", b)):
            m = ts.hour * 60 + ts.minute
            if not (o <= m < c):
                bad.append({"leg": leg, "iso": ts.isoformat(), "reason": "outside session",
                            "direction": t.get("direction")})
    return bad


def spanning_days(trades: Sequence[dict], tz: str) -> list[dict]:
    """Daytrading violation: entry date != exit date."""
    if not trades:
        return []
    loc = _local_index(trades, tz)
    if "exit_iso" in trades[0]:
        ex = pd.DatetimeIndex([pd.Timestamp(t["exit_iso"]) for t in trades]).tz_convert(tz)
    else:
        ex = pd.DatetimeIndex([pd.Timestamp(float(t["exit_time"]), unit="s", tz="UTC")
                               for t in trades]).tz_convert(tz)
    return [{"entry_iso": a.isoformat(), "exit_iso": b.isoformat(), "reason": "spans days"}
            for a, b in zip(loc, ex) if a.date() != b.date()]


def held_past(trades: Sequence[dict], tz: str, *, flat_hhmm: str) -> list[dict]:
    """Positions still open after the flat time on their exit day."""
    if not trades:
        return []
    loc = _local_index(trades, tz)
    ex = (pd.DatetimeIndex([pd.Timestamp(t["exit_iso"]) for t in trades])
          if "exit_iso" in trades[0]
          else pd.DatetimeIndex([pd.Timestamp(float(t["exit_time"]), unit="s", tz="UTC")
                                 for t in trades])).tz_convert(tz)
    lim = _hhmm_to_min(flat_hhmm)
    return [{"exit_iso": b.isoformat(), "reason": f"past flat {flat_hhmm}"}
            for b in ex if b.hour * 60 + b.minute > lim]


def assert_invariants(trades: Sequence[dict], tz: str, *, session_open: str,
                      session_close: str, flat_hhmm: str | None = None,
                      daytrading: bool = True) -> None:
    """Raise with details if any calendar/session invariant is violated."""
    problems: dict[str, Any] = {}
    if bad := outside_session(trades, tz, session_open=session_open, session_close=session_close):
        problems["outside_session"] = bad[:5]
    if daytrading and (bad := spanning_days(trades, tz)):
        problems["spans_days"] = bad[:5]
    if flat_hhmm and (bad := held_past(trades, tz, flat_hhmm=flat_hhmm)):
        problems["held_past_flat"] = bad[:5]
    if problems:
        raise AssertionError(f"calendar/session invariant violated: {problems}")
