#!/usr/bin/env python3
"""S-009 on the years it has never seen: 2016-2018, same rule, nothing changed. No holdout (2025+) is read.

The spec's train period starts in 2019; the 1m cache reaches back to 2016. Results per period and split by
meetings with / without a press conference (Lucca/Moench 2018: the drift lives at press-conference meetings).
Placebo: same window on random non-FOMC days, same weekdays, 100 runs. Combined 2016-2024 for the power.

  .venv/bin/python studies/pre_fomc_extended.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_lab import costs, rates, screening, watchlist  # noqa: E402
from trading_lab.registry import HOLDOUT_START, load_cached  # noqa: E402
from trading_lab.strategies import pre_fomc as P  # noqa: E402

QUOTE = {"GER40": "EUR"}
START = pd.Timestamp("2016-01-01")
SPLIT = pd.Timestamp("2019-01-01")
END = HOLDOUT_START.tz_localize(None)
PLACEBO_RUNS = 100


def markets() -> list[P.Market]:
    r = rates.load_rates(("USD", "EUR")).ffill()
    out = []
    for a in P.INDICES:
        m1 = costs.adjust_frame(load_cached(a, year_start=2015), a, "ftmo")
        m1 = m1[m1.index < HOLDOUT_START]
        q = r[QUOTE.get(a, "USD")] / 100
        daily = q.reindex(pd.date_range(q.index.min(), "2030-12-31", freq="D")).ffill()
        out.append(P.market(m1, a, -(daily + 0.025)))
    return out


def report(name: str, mk: list[P.Market], days: list[pd.Timestamp], cand: pd.DatetimeIndex) -> dict:
    df = pd.DataFrame(P.run_backtest(mk, days)["trades"])
    per = df.groupby("statement")["r"].sum() / 1.0
    move = df.groupby("asset")["move_bp"].mean().round(1).to_dict()
    pl = [np.mean([t["r_gross"] for t in P.run_backtest(mk, P.placebo_days(cand, days, np.random.default_rng(k)))["trades"]])
          for k in range(PLACEBO_RUNS)]
    yrs = df.assign(y=df["exit_time"].dt.year).groupby("y")["r"].sum().round(2).to_dict()
    t = float(screening.t_stat(per))
    out = {"name": name, "sitzungen": len(days), "n": len(df), "brutto": float(df["r_gross"].mean()),
           "netto": float(df["r"].mean()), "placebo": float(np.mean(pl)), "t_sitzung": t, "jahre": yrs, "bewegung_bp": move,
           "positiv": float((per > 0).mean())}
    print(f"{name}: {len(days)} Sitzungen, {len(df)} Positionen, brutto {out['brutto']:+.3f} R, netto {out['netto']:+.3f} R, "
          f"Placebo {out['placebo']:+.3f} R, t {t:+.2f}, {out['positiv']:.0%} der Sitzungen positiv\n   Bewegung bp {move}\n   Jahre {yrs}")
    return out


def main() -> int:
    warnings.filterwarnings("ignore")
    mk = markets()
    all_m = json.loads(P.DATES_FILE.read_text())["meetings"]
    press = {pd.Timestamp(m["date"]) for m in all_m if m["press"]}
    days = [pd.Timestamp(m["date"]) for m in all_m if START <= pd.Timestamp(m["date"]) < END]
    cand = pd.bdate_range(START, END - pd.Timedelta(days=2))
    res = [report("2016-2018 (neu)", mk, [d for d in days if d < SPLIT], cand[cand < SPLIT]),
           report("  davon mit Pressekonferenz", mk, [d for d in days if d < SPLIT and d in press], cand[cand < SPLIT]),
           report("  davon ohne Pressekonferenz", mk, [d for d in days if d < SPLIT and d not in press], cand[cand < SPLIT]),
           report("2019-2024 (bekannt)", mk, [d for d in days if d >= SPLIT], cand[cand >= SPLIT]),
           report("2016-2024 gesamt", mk, days, cand)]
    print(f"\nTrefferchance-Hinweis: {watchlist.power_note(9.0, 1)}")
    Path("artifacts/pre_fomc/extended.json").write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
