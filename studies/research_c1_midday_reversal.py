#!/usr/bin/env python3
"""C1 Stage 1 — US100 New York midday reversal, measurement only (no trades).

Pre-registered hypothesis, written before any result was looked at:
    New York lunch is a low-participation lull. A directional displacement of
    the Nasdaq-100 from the morning session unwinds during it. Concretely, with
    the cash open (09:30 ET) as reference:
        m = P(12:00 ET) − P(09:30 ET)      morning displacement
        r = P(13:30 ET) − P(12:00 ET)      midday return (18:00→19:30 German)
    r points against m more often than a same-length window started at a random
    time of day, and the effect grows with |m|.

Measurement only — no entry, stop, target or cost model. Everything here is
causal: m uses bars up to 12:00 ET, r is the realised future return after it.

Counter-check (from the nischen plan): the identical measurement with a random
start minute each day gives the null reversal rate; C1 must beat 99% of runs.
A second reference, the morning VWAP proxy (mean of the 09:30–12:00 closes, no
volume feed), counts a hit when the midday return points back toward it.

Pre-registered go/no-go: reversal rate ≥ 58%, better than 99% of random runs,
in at least two of the three sub-periods (2016–21 / 22–24 / 25–now).

  python studies/research_c1_midday_reversal.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _path
_sys.path.insert(0, str(_path.Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_lab.minute_data import load_minute_bid_ask
from trading_lab.significance import percentile_of

NY = "America/New_York"
MIN_NS = 60 * 10**9
TOL_NS = 4 * MIN_NS  # a target time must have a bar within this window to count

PERIODS: dict[str, tuple[int, int]] = {
    "dev_2016_2021": (2016, 2021),
    "test_2022_2024": (2022, 2024),
    "confirm_2025_now": (2025, 9999),
}
RANDOM_RUNS = 2000
RANDOM_LO_MIN, RANDOM_HI_MIN = 30, 300  # random start: 10:00–14:30 ET
REVERSAL_MIN = 0.58
RANDOM_PCT_MIN = 99.0


@dataclass(frozen=True)
class Clock:
    """Exchange-local (New York) intraday timetable in HH:MM."""

    open: str = "09:30"
    midday: str = "12:00"
    midday_end: str = "13:30"
    close: str = "16:00"


@dataclass
class Session:
    day: str
    year: int
    p_open: float
    p_mid: float
    p_end: float
    vwap_proxy: float

    @property
    def morning(self) -> float:
        return self.p_mid - self.p_open

    @property
    def midday(self) -> float:
        return self.p_end - self.p_mid

    @property
    def reversal(self) -> bool:
        return self.morning * self.midday < 0

    @property
    def to_vwap_hit(self) -> bool:
        # midday move points back toward the morning mean
        return (self.vwap_proxy - self.p_mid) * self.midday > 0


# ── Data / clock helpers ────────────────────────────────────────────


def _local_ns(days: pd.DatetimeIndex, hhmm: str) -> np.ndarray:
    h, m = (int(x) for x in hhmm.split(":"))
    local = (days + pd.Timedelta(hours=h, minutes=m)).tz_localize(NY)
    return local.tz_convert("UTC").as_unit("ns").asi8


def _at(ts: np.ndarray, px: np.ndarray, target_ns: int, tol_ns: int = TOL_NS) -> float | None:
    """Close of the last 1m bar at or before ``target_ns`` (within ``tol_ns``)."""
    i = int(np.searchsorted(ts, target_ns, side="right")) - 1
    if i < 0 or target_ns - int(ts[i]) > tol_ns:
        return None
    return float(px[i])


def build_sessions(data: pd.DataFrame, clock: Clock) -> list[Session]:
    """One row per trading day, only when every clock point has a nearby bar."""
    ts = data.index.as_unit("ns").asi8
    px = data["close"].to_numpy(dtype=float)
    local_days = data.index.tz_convert(NY).tz_localize(None).normalize().unique()
    days = pd.DatetimeIndex([d for d in local_days if d.weekday() < 5])
    open_ns = _local_ns(days, clock.open)
    mid_ns = _local_ns(days, clock.midday)
    end_ns = _local_ns(days, clock.midday_end)

    out: list[Session] = []
    for d in range(len(days)):
        p_open = _at(ts, px, int(open_ns[d]))
        p_mid = _at(ts, px, int(mid_ns[d]))
        p_end = _at(ts, px, int(end_ns[d]))
        if p_open is None or p_mid is None or p_end is None:
            continue
        i0, i1 = (int(x) for x in np.searchsorted(ts, [open_ns[d], mid_ns[d]], side="left"))
        if i1 - i0 < 30:  # a genuine morning needs real bars, not a stub
            continue
        out.append(
            Session(
                day=days[d].date().isoformat(),
                year=days[d].year,
                p_open=p_open,
                p_mid=p_mid,
                p_end=p_end,
                vwap_proxy=float(px[i0:i1].mean()),
            )
        )
    return out


# ── Statistics ──────────────────────────────────────────────────────


def reversal_rate(sessions: list[Session]) -> float:
    if not sessions:
        return float("nan")
    return float(np.mean([s.reversal for s in sessions]))


def period(year: int) -> str:
    for name, (lo, hi) in PERIODS.items():
        if lo <= year <= hi:
            return name
    return "other"


def summarize(sessions: list[Session]) -> dict[str, Any]:
    m = np.array([s.morning for s in sessions])
    r = np.array([s.midday for s in sessions])
    rev = np.array([s.reversal for s in sessions])
    vwap = np.array([s.to_vwap_hit for s in sessions])
    corr = float(np.corrcoef(m, r)[0, 1]) if len(m) > 2 and m.std() and r.std() else float("nan")
    sizes = {
        "reversal": float(np.mean([abs(s.midday) for s in sessions if s.reversal])) if rev.any() else 0.0,
        "continuation": float(np.mean([abs(s.midday) for s in sessions if not s.reversal])) if (~rev).any() else 0.0,
    }
    return {
        "days": len(sessions),
        "reversal_rate": round(float(rev.mean()), 4),
        "to_vwap_rate": round(float(vwap.mean()), 4),
        "corr_morning_midday": round(corr, 4),
        "mean_abs_morning_pts": round(float(np.mean(np.abs(m))), 2),
        "mean_abs_midday_pts": round(float(np.mean(np.abs(r))), 2),
        "mean_abs_midday_reversal_pts": round(sizes["reversal"], 2),
        "mean_abs_midday_continuation_pts": round(sizes["continuation"], 2),
    }


def by_period(sessions: list[Session]) -> dict[str, dict[str, Any]]:
    return {
        name: summarize([s for s in sessions if period(s.year) == name])
        for name in PERIODS
    }


def by_morning_size(sessions: list[Session], quantiles: int = 5) -> list[dict[str, Any]]:
    """Reversal rate per |morning| quintile — the effect should grow with size."""
    if len(sessions) < quantiles * 20:
        return []
    sizes = np.array([abs(s.morning) for s in sessions])
    edges = np.quantile(sizes, np.linspace(0, 1, quantiles + 1))
    rows = []
    for q in range(quantiles):
        lo, hi = edges[q], edges[q + 1]
        bucket = [
            s for s in sessions
            if (lo <= abs(s.morning) < hi) or (q == quantiles - 1 and abs(s.morning) >= hi)
        ]
        rows.append({"quintile": q + 1, "lo_pts": round(float(lo), 1), "hi_pts": round(float(hi), 1), **summarize(bucket)})
    return rows


# ── Random counter-check ────────────────────────────────────────────


def random_null(
    data: pd.DataFrame, clock: Clock, days: list[str], *, runs: int, seed: int
) -> np.ndarray:
    """Null reversal rate: same m/r measurement started at a random minute.

    Every run picks one random start minute per day in [10:00, 14:30) ET, then
    computes the reversal rate of (open→start) against (start→start+90 min).
    """
    ts = data.index.as_unit("ns").asi8
    px = data["close"].to_numpy(dtype=float)
    want = pd.DatetimeIndex(pd.to_datetime(days))
    open_ns = _local_ns(want, clock.open)
    rng = np.random.default_rng(seed)
    offsets = rng.integers(RANDOM_LO_MIN, RANDOM_HI_MIN, size=(runs, len(days)))
    rate = np.empty(runs)
    for run in range(runs):
        hits = total = 0
        for d in range(len(days)):
            t0 = int(open_ns[d]) + int(offsets[run, d]) * MIN_NS
            pa = _at(ts, px, int(open_ns[d]))
            pb = _at(ts, px, t0)
            pc = _at(ts, px, t0 + 90 * MIN_NS)
            if pa is None or pb is None or pc is None or pb == pa:
                continue
            total += 1
            hits += (pb - pa) * (pc - pb) < 0
        rate[run] = hits / total if total else np.nan
    return rate[~np.isnan(rate)]


# ── Report ──────────────────────────────────────────────────────────


def render(result: dict[str, Any], args: argparse.Namespace) -> str:
    o = result["overall"]
    p = result["periods"]
    r = result["random"]
    out = [
        "# C1 — US100 Mittags-Umkehr (New York), Stufe 1: Messung",
        "",
        f"Erzeugt: {result['generated_at']} UTC · Dukascopy 1-Minuten Bid ab {args.start_year} · "
        f"Zufallsgegenprobe: {RANDOM_RUNS} Läufe",
        "",
        "Reine Messung, keine Trades, keine Kosten. "
        "m = P(12:00 ET) − P(09:30 ET), r = P(13:30 ET) − P(12:00 ET). "
        "„Umkehr“ = r und m haben verschiedene Vorzeichen. "
        "12:00–13:30 ET entspricht 18:00–19:30 deutscher Zeit (ganzjährig konstant).",
        "",
        "## Ergebnis",
        "",
        f"- Handelstage mit vollständiger Messung: **{o['days']}**",
        f"- Umkehrrate gesamt: **{o['reversal_rate'] * 100:.1f} %**",
        f"- Rückkehr-zum-VWAP-Trefferquote: {o['to_vwap_rate'] * 100:.1f} %",
        f"- Korrelation Morgen↔Mittag: {o['corr_morning_midday']:+.3f}",
        f"- Ø |Morgenschub| {o['mean_abs_morning_pts']} Pkt, Ø |Mittagsbewegung| {o['mean_abs_midday_pts']} Pkt "
        f"(Umkehr {o['mean_abs_midday_reversal_pts']} / Fortsetzung {o['mean_abs_midday_continuation_pts']})",
        "",
        "## Nach Perioden",
        "",
        "| Zeitraum | Tage | Umkehrrate | VWAP-Treffer | Korr. | Ø |Morgenschub| | Ø |Mittag| |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in PERIODS:
        s = p[name]
        out.append(
            f"| {name} | {s['days']} | {s['reversal_rate'] * 100:.1f} % | {s['to_vwap_rate'] * 100:.1f} % | "
            f"{s['corr_morning_midday']:+.3f} | {s['mean_abs_morning_pts']} | {s['mean_abs_midday_pts']} |"
        )
    out += [
        "",
        "## Umkehrrate nach Größe des Morgenschubs (Quintile)",
        "",
        "| Quintil | |m| von–bis (Pkt) | Tage | Umkehrrate | Korr. m↔r |",
        "|---|---|---|---|---|",
    ]
    for q in result["by_size"]:
        out.append(
            f"| {q['quintile']} | {q['lo_pts']}–{q['hi_pts']} | {q['days']} | "
            f"{q['reversal_rate'] * 100:.1f} % | {q['corr_morning_midday']:+.3f} |"
        )
    out += [
        "",
        "## Zufallsgegenprobe (gleiche Messung, zufällige Startzeit je Tag)",
        "",
        f"- Zufallsmittel: {r['mean'] * 100:.2f} % · 95 %-Grenze: {r['p95'] * 100:.2f} % · "
        f"99 %-Grenze: {r['p99'] * 100:.2f} %",
        f"- C1-Umkehrrate besser als **{r['percentile']:.1f} %** der Zufallsläufe",
        "",
        "## Vorab festgelegte Go/No-Go-Kriterien",
        "",
        "| Kriterium | Erfüllt |",
        "|---|---|",
        f"| Umkehrrate ≥ {REVERSAL_MIN * 100:.0f} % | {'ja' if o['reversal_rate'] >= REVERSAL_MIN else 'nein'} |",
        f"| besser als {RANDOM_PCT_MIN:.0f} % der Zufallsläufe | {'ja' if r['percentile'] >= RANDOM_PCT_MIN else 'nein'} |",
        f"| ≥ 58 % in mind. zwei Teilperioden | {'ja' if result['periods_ok'] >= 2 else 'nein'} |",
        "",
        f"**Verdikt: {'WEITER (Stufe 2)' if result['passed'] else 'STOP — Muster nicht tragfähig'}**",
    ]
    return "\n".join(out) + "\n"


def main() -> None:
    logging.getLogger("DUKASCRIPT").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="C1 US100 midday reversal — stage-1 measurement")
    parser.add_argument("--asset", default="US100")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--runs", type=int, default=RANDOM_RUNS)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path, default=Path("artifacts/c1_midday_reversal"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== C1 {args.asset} midday reversal (stage 1) ===", flush=True)
    data = load_minute_bid_ask(args.asset, args.start_year)
    if data.empty:
        print("no data"); return
    print(f"  {len(data)} 1m bars {data.index[0]} → {data.index[-1]}", flush=True)

    clock = Clock()
    sessions = build_sessions(data, clock)
    overall = summarize(sessions)
    periods = by_period(sessions)
    size_rows = by_morning_size(sessions)

    print(f"  {overall['days']} complete days · reversal {overall['reversal_rate'] * 100:.1f} % "
          f"· corr {overall['corr_morning_midday']:+.3f}", flush=True)

    null = random_null(data, clock, [s.day for s in sessions], runs=args.runs, seed=args.seed)
    random_info = {
        "runs": int(len(null)),
        "mean": round(float(null.mean()), 4),
        "p95": round(float(np.quantile(null, 0.95)), 4),
        "p99": round(float(np.quantile(null, 0.99)), 4),
        "percentile": round(percentile_of(overall["reversal_rate"], null), 1),
    }
    periods_ok = sum(1 for name in PERIODS if periods[name]["reversal_rate"] >= REVERSAL_MIN)
    passed = bool(
        overall["reversal_rate"] >= REVERSAL_MIN
        and random_info["percentile"] >= RANDOM_PCT_MIN
        and periods_ok >= 2
    )
    print(f"  random null: mean {random_info['mean'] * 100:.2f} % · p99 {random_info['p99'] * 100:.2f} % "
          f"· C1 percentile {random_info['percentile']:.1f}", flush=True)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset": args.asset,
        "start_year": args.start_year,
        "clock": asdict(clock),
        "data": {"start": data.index[0].isoformat(), "end": data.index[-1].isoformat(), "bars": len(data)},
        "overall": overall,
        "periods": periods,
        "by_size": size_rows,
        "random": random_info,
        "periods_ok": periods_ok,
        "passed": passed,
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2))
    args.output.with_suffix(".md").write_text(render(result, args))
    print(f"\nSaved: {args.output.with_suffix('.md')} / .json")
    print(f"  VERDICT: {'PASS → Stufe 2' if passed else 'FAIL → kein Handel'}")


if __name__ == "__main__":
    main()
