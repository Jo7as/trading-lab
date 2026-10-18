"""Economic calendar — offline loader for scheduled high-impact events.

The fair-pricing strategy needs the time and impact of scheduled releases
(US "Red Folder" data like CPI, NFP, FOMC) to place the pre-news fair price
and to block new entries around a scheduled event.

There is deliberately **no network access here**: the loader reads a local
CSV or JSON file, so tests and backtests stay offline. A file can be produced
by any tool the user likes; the schema is intentionally tiny.

CSV
    ``time,event,impact`` with one row per event; ``time`` is ISO-8601
    (``2024-06-12T08:30:00`` or ``2024-06-12T12:30:00Z``). A naive timestamp is
    interpreted as UTC unless ``tz`` is given.

JSON
    Either a list of objects ``[{"time": ..., "event": ..., "impact": ...}, ...]``
    or an object ``{"events": [...]}``.

Impact is normalized to ``high`` / ``medium`` / ``low`` / ``unknown``:
``high`` (or ``red``/``3``), ``medium`` (``orange``/``2``), ``low`` (``yellow``/``1``).
The strategy only treats ``high`` as Red Folder.

    events = load_events("configs/calendar/us_red_folder.csv")
    times  = high_impact_times(events)
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

IMPACTS = ("high", "medium", "low", "unknown")

_IMPACT_ALIASES: dict[str, str] = {
    "high": "high", "red": "high", "3": "high", "h": "high",
    "medium": "medium", "med": "medium", "orange": "medium", "2": "medium", "m": "medium",
    "low": "low", "yellow": "low", "1": "low", "l": "low",
    "holiday": "low", "": "unknown", "-": "unknown", "none": "unknown",
}


@dataclass(frozen=True)
class CalendarEvent:
    """One scheduled release. ``ts`` is unix seconds UTC."""

    ts: float
    name: str
    impact: str = "unknown"

    def __post_init__(self) -> None:
        if self.impact not in IMPACTS:
            raise ValueError(f"impact must be one of {IMPACTS}, got {self.impact!r}")

    @property
    def is_high(self) -> bool:
        return self.impact == "high"


def parse_impact(value: Any) -> str:
    """Normalize an impact label from a calendar file."""
    return _IMPACT_ALIASES.get(str(value).strip().lower(), "unknown")


def _to_ts(value: Any, tz: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("empty event time")
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:  # try a couple of common spellings
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"cannot parse event time {text!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(tz))
    return dt.astimezone(timezone.utc).timestamp()


def _event_from_row(row: dict[str, Any], tz: str) -> CalendarEvent:
    raw_time = row.get("time", row.get("timestamp", row.get("date")))
    if raw_time in (None, ""):
        raise ValueError(f"event row has no time: {row!r}")
    name = str(row.get("event", row.get("name", row.get("title", "")))).strip()
    return CalendarEvent(ts=_to_ts(raw_time, tz), name=name, impact=parse_impact(row.get("impact", "")))


def load_events(path: str | Path, *, tz: str = "UTC") -> list[CalendarEvent]:
    """Read a CSV or JSON calendar file into sorted events (by time).

    ``tz`` is the timezone used for naive timestamps (default UTC). Raises
    ``FileNotFoundError`` when the file is missing and ``ValueError`` on a bad
    schema, so callers fail loudly instead of silently trading without news.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"calendar file not found: {p}")
    text = p.read_text(encoding="utf-8")

    rows: Iterable[dict[str, Any]]
    if p.suffix.lower() == ".json":
        data = json.loads(text)
        rows = data.get("events", []) if isinstance(data, dict) else data
    else:
        lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        rows = list(csv.DictReader(lines))

    events = [_event_from_row(dict(r), tz) for r in rows]
    return sorted(events, key=lambda e: e.ts)


def high_impact_times(events: Iterable[CalendarEvent]) -> list[float]:
    """Sorted timestamps of Red Folder events."""
    return sorted(e.ts for e in events if e.is_high)


def events_in(events: Iterable[CalendarEvent], start: float, end: float) -> list[CalendarEvent]:
    """Events with ``start <= ts < end`` (unix seconds)."""
    return [e for e in events if start <= e.ts < end]


def next_high_impact(
    events: Iterable[CalendarEvent], after: float, *, within: float | None = None
) -> CalendarEvent | None:
    """First Red Folder event at/after ``after`` (optionally within ``within`` s)."""
    hi = [e for e in events if e.is_high and e.ts >= after]
    if not hi:
        return None
    ev = min(hi, key=lambda e: e.ts)
    if within is not None and ev.ts - after > within:
        return None
    return ev
