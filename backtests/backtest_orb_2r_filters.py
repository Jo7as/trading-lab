#!/usr/bin/env python3
"""Pre-registered day-filter test for the 15-minute ORB (2R target).

Protocol — fixed before looking at any result, no thresholds are tuned:
  1. Four filters, each with one textbook setting, tested one at a time
     against the plan rules on the development period 2016–2021 only.
  2. A filter "helps" if, on 2016–2021, Ø R is above the base, the sum of R
     is positive and at least MIN_DEV_TRADES trades remain.
  3. All helping filters are combined and that one combination is checked
     once on 2022–now: sum R > 0, Ø R ≥ +0.05, sum R at doubled costs > 0.
The 2022–now numbers of the single filters are printed for transparency;
they play no part in the selection.

  python backtests/backtest_orb_2r_filters.py
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backtest_orb_2r import PERIODS, in_period, stats
from trading_lab.minute_data import load_minute_bid_ask
from trading_lab.strategies.orb_2r import MARKETS, OrbConfig, build_sessions, run_backtest

MIN_DEV_TRADES = 100
MIN_OOS_AVG_R = 0.05
DEV, OOS = PERIODS["dev_2016_2021"], PERIODS["oos_2022_now"]

# key → (label, config changes, markets). F1 comes in two forms; at most one is combined.
FILTERS: dict[str, tuple[str, dict[str, Any], tuple[str, ...]]] = {
    "range_atr": (
        "Range/ATR ≥ 60-Tage-Median, zusätzlich zum Kostenfilter",
        {"range_atr_median_days": 60},
        ("US100", "GER40", "XAUUSD", "JP225"),
    ),
    "range_atr_only": (
        "Range/ATR ≥ 60-Tage-Median, statt Kostenfilter",
        {"range_atr_median_days": 60, "max_cost_r": 0.0},
        ("US100", "GER40", "XAUUSD", "JP225"),
    ),
    "trend": (
        "Nur in Richtung des 50-Tage-Durchschnitts",
        {"trend_sma_days": 50},
        ("US100", "GER40", "XAUUSD", "JP225"),
    ),
    "gap": (
        "Nur in Richtung der Eröffnungslücke",
        {"gap_filter": True},
        ("US100", "GER40", "XAUUSD", "JP225"),
    ),
    "fomc": (
        "FOMC-Tage auslassen",
        {"skip_fomc": True},
        ("US100", "XAUUSD"),
    ),
}
_RANGE_ATR_KEYS = ("range_atr", "range_atr_only")


def summary(trades: list[dict]) -> dict[str, Any]:
    return {"dev": stats(in_period(trades, DEV)), "oos": stats(in_period(trades, OOS)), "full": stats(trades)}


def study(market: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg = MARKETS[market]
    print(f"\n=== {market} ===", flush=True)
    df = load_minute_bid_ask(market, args.start_year)
    if df.empty:
        return {"market": market, "error": "no data"}
    sessions = {mult: build_sessions(df, cfg, mult) for mult in (1.0, 2.0)}

    def run(c: OrbConfig, mult: float = 1.0) -> list[dict]:
        return run_backtest(sessions[mult], c)["trades"]

    base_trades = run(cfg)
    base = summary(base_trades)
    base_by_day = {t["day"]: t for t in base_trades}

    rows: list[dict[str, Any]] = []
    for key, (label, changes, markets) in FILTERS.items():
        if market not in markets:
            continue
        trades = run(replace(cfg, **changes))
        s = summary(trades)
        kept_days = {t["day"] for t in trades}
        removed = [t for day, t in base_by_day.items() if day not in kept_days]
        helps = bool(
            s["dev"]["avg_r"] > base["dev"]["avg_r"]
            and s["dev"]["sum_r"] > 0
            and s["dev"]["trades"] >= MIN_DEV_TRADES
        )
        rows.append(
            {
                "key": key, "label": label, "changes": changes, **s,
                "removed_dev": stats(in_period(removed, DEV)),
                "helps_on_dev": helps,
            }
        )
        print(
            f"  {key:15s} dev {s['dev']['trades']:4d} trades Ø {s['dev']['avg_r']:+.3f}R "
            f"Σ {s['dev']['sum_r']:+6.1f}R → {'helps' if helps else 'no'}",
            flush=True,
        )

    helping = [r for r in rows if r["helps_on_dev"]]
    # the two range/ATR forms exclude each other — keep the better one on dev
    range_forms = [r for r in helping if r["key"] in _RANGE_ATR_KEYS]
    if len(range_forms) == 2:
        drop = min(range_forms, key=lambda r: r["dev"]["avg_r"])
        helping = [r for r in helping if r is not drop]

    combined: dict[str, Any] | None = None
    if helping:
        changes: dict[str, Any] = {}
        for r in helping:
            changes.update(r["changes"])
        c = replace(cfg, **changes)
        trades = run(c)
        stress = run(c, 2.0)
        s = summary(trades)
        oos_stress = stats(in_period(stress, OOS))
        checks = {
            "oos_sum_r_positive": bool(s["oos"]["sum_r"] > 0),
            "oos_avg_r_ge_0.05": bool(s["oos"]["avg_r"] >= MIN_OOS_AVG_R),
            "oos_double_cost_positive": bool(oos_stress["sum_r"] > 0),
        }
        combined = {
            "filters": [r["key"] for r in helping], "changes": changes, **s,
            "oos_double_cost": oos_stress,
            "yearly_oos": {
                y: stats([t for t in trades if t["day"][:4] == y])
                for y in sorted({t["day"][:4] for t in trades if int(t["day"][:4]) >= OOS[0]})
            },
            "checks": checks,
            "confirmed": all(checks.values()),
        }
        print(
            f"  combined {combined['filters']}: OOS {s['oos']['trades']} trades "
            f"Ø {s['oos']['avg_r']:+.3f}R Σ {s['oos']['sum_r']:+.1f}R "
            f"(2× costs {oos_stress['sum_r']:+.1f}R) → "
            f"{'CONFIRMED' if combined['confirmed'] else 'NOT confirmed'}",
            flush=True,
        )
    else:
        print("  no filter helps on 2016–2021 → nothing to confirm", flush=True)

    return {"market": market, "base": base, "filters": rows, "combined": combined}


def _cells(s: dict[str, Any]) -> str:
    return f"{s['trades']} | {s['avg_r']:+.3f} | {s['sum_r']:+.1f} | {s['max_dd_r']:.1f}"


def render(rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    out = [
        "# ORB 15 min / 2R — Filtertest",
        "",
        f"Erzeugt: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · Dukascopy 1-Minuten-Kerzen Bid+Ask ab "
        f"{args.start_year} · alle Werte in R nach Kosten",
        "",
        "Ablauf, vorab festgelegt: Jeder Filter hat genau eine Einstellung und wird einzeln nur auf "
        "2016–2021 geprüft. Ein Filter „hilft“, wenn dort Ø R über der Basis liegt, die Summe R positiv "
        f"ist und mindestens {MIN_DEV_TRADES} Trades bleiben. Alle helfenden Filter werden kombiniert und "
        "diese eine Kombination einmal auf 2022–heute bestätigt (Summe R > 0, Ø R ≥ +0,05, Summe R bei "
        "doppelten Kosten > 0). Die 2022–heute-Werte der Einzelfilter stehen nur zur Information da.",
    ]
    for r in rows:
        out += [
            "",
            f"## {r['market']}",
            "",
            "| | 2016–2021: Trades | Ø R | Summe R | Max. DD | Entfernte Trades: Ø R | Hilft | "
            "2022–heute: Trades | Ø R | Summe R | Max. DD |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
            f"| Basis (Plan-Regeln) | {_cells(r['base']['dev'])} | — | — | {_cells(r['base']['oos'])} |",
        ]
        for f in r["filters"]:
            rem = f["removed_dev"]
            rem_txt = f"{rem['avg_r']:+.3f} ({rem['trades']})" if rem["trades"] else "—"
            out.append(
                f"| {f['label']} | {_cells(f['dev'])} | {rem_txt} | "
                f"{'**ja**' if f['helps_on_dev'] else 'nein'} | {_cells(f['oos'])} |"
            )
        c = r["combined"]
        if c is None:
            out += ["", "Kein Filter hilft auf 2016–2021. Es gibt nichts zu bestätigen."]
            continue
        out += [
            "",
            f"**Kombination:** {', '.join(c['filters'])} → "
            f"{'**bestätigt**' if c['confirmed'] else '**nicht bestätigt**'}",
            "",
            "| Zeitraum | Trades | Ø R | Summe R | Max. DD |",
            "|---|---|---|---|---|",
            f"| 2016–2021 | {_cells(c['dev'])} |",
            f"| 2022–heute | {_cells(c['oos'])} |",
            f"| 2022–heute, doppelte Kosten | {_cells(c['oos_double_cost'])} |",
        ]
        out += [f"| {y} | {_cells(s)} |" for y, s in c["yearly_oos"].items()]
    return "\n".join(out) + "\n"


def main() -> None:
    logging.getLogger("DUKASCRIPT").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="ORB 15 min / 2R — pre-registered filter test")
    parser.add_argument("--markets", default="US100,GER40,XAUUSD")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--output", type=Path, default=Path("artifacts/orb_2r_filters"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = [study(m.strip(), args) for m in args.markets.split(",") if m.strip()]
    rows = [r for r in rows if "error" not in r]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {"min_dev_trades": MIN_DEV_TRADES, "min_oos_avg_r": MIN_OOS_AVG_R},
        "markets": rows,
    }
    args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=str))
    args.output.with_suffix(".md").write_text(render(rows, args))
    print(f"\nSaved: {args.output.with_suffix('.md')} / .json")


if __name__ == "__main__":
    main()
