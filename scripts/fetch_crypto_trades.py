"""Download crypto trade dumps into the 10-s flow-bar cache (``artifacts/cache/crypto``).

    uv run python scripts/fetch_crypto_trades.py --start 2024-01-01 --end 2024-12-31 \
        --markets BTC SOL --venues binance_um bybit_linear

One worker per venue/market (polite to the exchanges); already cached days are skipped,
so the script can be stopped and restarted at any time. Progress -> stdout.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_lab.adapters import crypto_trades as ct  # noqa: E402


def _worker(venue: str, market: str, start: date, end: date, bar: str) -> str:
    symbol = ct.SYMBOLS[market][venue]
    day, done, missing, t0 = start, 0, 0, time.monotonic()
    n_days = (end - start).days + 1
    while day <= end:
        try:
            if ct.load_day(venue, symbol, day, bar=bar) is None:
                missing += 1
        except Exception as exc:  # keep going; the day is retried on the next run
            print(f"[{venue} {market}] {day} FAILED: {exc}", flush=True)
        done += 1
        if done % 10 == 0 or day == end:
            rate = done / max(time.monotonic() - t0, 1e-9) * 3600
            print(f"[{venue} {market}] {done}/{n_days} days, {missing} missing, "
                  f"{rate:.0f} days/h", flush=True)
        day += timedelta(days=1)
    return f"{venue} {market}: {done} days, {missing} missing"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", type=date.fromisoformat, required=True)
    ap.add_argument("--end", type=date.fromisoformat, required=True)
    ap.add_argument("--markets", nargs="+", default=["BTC", "SOL"])
    ap.add_argument("--venues", nargs="+", default=["binance_um", "bybit_linear"])
    ap.add_argument("--bar", default="10s")
    a = ap.parse_args()
    jobs = [(v, m) for v in a.venues for m in a.markets]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        for res in pool.map(lambda j: _worker(j[0], j[1], a.start, a.end, a.bar), jobs):
            print("DONE", res, flush=True)


if __name__ == "__main__":
    main()
