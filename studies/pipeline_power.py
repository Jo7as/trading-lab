#!/usr/bin/env python3
"""Is the pipeline too strict? Measure its power on edges we *know* are real.

Every gate in this repo rejects things, and after nine dead cards it is fair to
ask whether the pipeline would have recognised a real edge if one had been there.
A rejection is only informative if the test has power, so this measures the power
directly: synthesise trade series with a *known* edge (a Sharpe we chose), run the
real acceptance path over them, and count how often it passes. The path judges the
**daily** series (stage S4, ``t >= 3``), so the synth log carries ``exit_time``.

The result is a detection threshold: "with N trades and this pipeline you can
detect a Sharpe of at least X". Compare that with the Sharpe a candidate would
plausibly have - if the threshold sits far above the plausible range, the
pipeline is too strict; if it sits below, the nine rejections are real.

  .venv/bin/python studies/pipeline_power.py
  .venv/bin/python studies/pipeline_power.py --trades 600 --runs 400
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

sys.path.insert(0, ".")

import numpy as np  # noqa: E402

from trading_lab.acceptance import run_acceptance  # noqa: E402


@dataclass
class Power:
    sharpe: float
    n: int
    pass_rate: float


def synthesise(sharpe: float, n: int, *, trades_per_year: int, rng: np.random.Generator) -> list[dict]:
    """A trade log whose *true* annualised Sharpe is ``sharpe``.

    Per-trade edge ``mu = sharpe / sqrt(trades_per_year)`` with unit volatility,
    so the annualised Sharpe of the series is exactly the requested number. The
    trades are i.i.d. normal - the most favourable case for any test (real trades
    are autocorrelated, fat-tailed and often regime-clustered, all of which cost
    power).

    The log carries an ``exit_time`` so the acceptance path judges it on its
    **daily** series (stage S4), exactly as a real log is judged; one trade per
    day at the default ``trades_per_year``.
    """
    mu = sharpe / np.sqrt(trades_per_year)
    r = rng.normal(mu, 1.0, n)
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    per_day = max(1, int(round(trades_per_year / 250)))
    return [{"r": float(x), "exit_time": (base + timedelta(days=i // per_day)).isoformat()}
            for i, x in enumerate(r)]


def measure(sharpe: float, n: int, *, trades_per_year: int, runs: int,
            rng: np.random.Generator) -> Power:
    passed = 0
    for _ in range(runs):
        trades = synthesise(sharpe, n, trades_per_year=trades_per_year, rng=rng)
        if run_acceptance(trades).passed:
            passed += 1
    return Power(sharpe, n, passed / runs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trades", type=int, nargs="+", default=[150, 400, 1000, 2500])
    ap.add_argument("--trades-per-year", type=int, default=250)
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    sharpes = [0.0, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    print("### Power of the acceptance path (S4: t of the DAILY series >= 3)")
    print("    Anteil der Laeufe, in denen ein ECHTER Edge dieser Sharpe erkannt wird.\n")
    header = f"    {'true Sharpe':>12s}" + "".join(f"{n:>9d}" for n in args.trades)
    print(header + "   (Trades)")
    results: dict[int, list[Power]] = {}
    for s in sharpes:
        row = []
        for n in args.trades:
            p = measure(s, n, trades_per_year=args.trades_per_year, runs=args.runs, rng=rng)
            row.append(p)
            results.setdefault(n, []).append(p)
        print(f"    {s:12.2f}" + "".join(f"{p.pass_rate*100:8.0f}%" for p in row))

    print("\n    Nachweisgrenze (80 % Power) je Stichprobe:")
    for n, ps in results.items():
        hit = next((p for p in ps if p.pass_rate >= 0.80), None)
        print(f"      {n:5d} Trades: "
              + (f"Sharpe >= {hit.sharpe:.2f} wird mit 80 % erkannt"
                 if hit else "selbst Sharpe 3.0 erreicht keine 80 % Power"))

    fp = [next(x for x in results[n] if x.sharpe == 0.0) for n in args.trades]
    print("\n    Falsch-Positiv-Rate bei Sharpe 0 (soll < 1 % sein, t >= 3 einseitig): "
          + ", ".join(f"n={p.n}: {p.pass_rate*100:.1f}%" for p in fp))
    print("\n    Einordnung: ein realistischer Kandidat liegt bei Sharpe 0.5-1.5.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
