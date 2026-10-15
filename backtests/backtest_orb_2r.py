#!/usr/bin/env python3
"""Phase-1 backtest for the 15-minute opening range breakout (2R target).

Per market (US100, GER40, XAUUSD) on Dukascopy 1m bid+ask:
  - fixed rules, results in R: gross, net and with doubled costs
  - development 2016–2021 / test 2022–2024 / confirmation 2025–now, per year
  - random test: random minute + direction on the same days, same stop distance
  - robustness grid: range 5/15/30 min × signal bar 1/5/15 min × target 1.5/2/3 R
  - pass/fail against the plan's criteria, 20 sample trades for a manual chart check

  python backtests/backtest_orb_2r.py
  python backtests/backtest_orb_2r.py --markets US100 --runs 200
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from trading_lab.minute_data import load_minute_bid_ask
from trading_lab.significance import percentile_of
from trading_lab.strategies.orb_2r import (
    MARKETS,
    OrbConfig,
    build_sessions,
    random_baseline,
    run_backtest,
)

PERIODS: dict[str, tuple[int, int]] = {
    "dev_2016_2021": (2016, 2021),
    "test_2022_2024": (2022, 2024),
    "confirm_2025_now": (2025, 9999),
    "oos_2022_now": (2022, 9999),
    "full": (0, 9999),
}
RANGE_MINUTES = (5, 15, 30)
SIGNAL_MINUTES = (1, 5, 15)
TARGETS = (1.5, 2.0, 3.0)

CRITERIA = {
    "min_avg_r": 0.05,
    "min_random_percentile": 95.0,
    "max_losing_years": 1,
}

# Extra looks per market: (label, config changes)
VARIANTS: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "US100": [
        ("ohne Kostenfilter", {"max_cost_r": 0.0}),
        ("Max-Range-Filter 2,5×", {"max_range_mult": 2.5}),
        ("Slippage proportional (0,4 bp)", {"slippage": 0.0, "slippage_pct": 0.00004}),
        ("Slippage proportional, ohne Kostenfilter", {"slippage": 0.0, "slippage_pct": 0.00004, "max_cost_r": 0.0}),
    ],
    "GER40": [
        ("ohne Kostenfilter", {"max_cost_r": 0.0}),
        ("Max-Range-Filter 2,5×", {"max_range_mult": 2.5}),
        ("Slippage proportional (0,4 bp)", {"slippage": 0.0, "slippage_pct": 0.00004}),
        ("Slippage proportional, ohne Kostenfilter", {"slippage": 0.0, "slippage_pct": 0.00004, "max_cost_r": 0.0}),
    ],
    "XAUUSD": [
        ("ohne Kostenfilter", {"max_cost_r": 0.0}),
        ("Max-Range-Filter 2,5×", {"max_range_mult": 2.5}),
        ("Eröffnung 09:30 NY (15:30 dt.)", {"session_open": "09:30"}),
        ("Range 30 min", {"range_minutes": 30}),
        ("Range 60 min", {"range_minutes": 60}),
    ],
    "JP225": [
        ("ohne Kostenfilter", {"max_cost_r": 0.0}),
        ("Max-Range-Filter 2,5×", {"max_range_mult": 2.5}),
        ("Slippage proportional (0,4 bp)", {"slippage": 0.0, "slippage_pct": 0.00004}),
        ("nur Vormittag (11:20 JST flat)", {"flat_time": "11:20"}),
        ("Range 30 min", {"range_minutes": 30}),
    ],
}


# ── Metrics ─────────────────────────────────────────────────────────


def in_period(trades: list[dict], years: tuple[int, int]) -> list[dict]:
    return [t for t in trades if years[0] <= int(t["day"][:4]) <= years[1]]


def stats(trades: list[dict]) -> dict[str, Any]:
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "avg_r": 0.0, "sum_r": 0.0, "max_dd_r": 0.0,
                "max_losing_streak": 0, "neg_months_pct": 0.0}
    r = np.array([t["r"] for t in trades])
    cum = r.cumsum()
    dd = float((cum - np.maximum.accumulate(np.maximum(cum, 0.0))).min())
    streak = longest = 0
    for x in r:
        streak = streak + 1 if x < 0 else 0
        longest = max(longest, streak)
    months = pd.Series(r, index=[t["day"][:7] for t in trades]).groupby(level=0).sum()
    return {
        "trades": int(len(r)),
        "win_rate": round(float((r > 0).mean() * 100), 1),
        "avg_r": round(float(r.mean()), 4),
        "sum_r": round(float(r.sum()), 1),
        "max_dd_r": round(dd, 1),
        "max_losing_streak": int(longest),
        "neg_months_pct": round(float((months < 0).mean() * 100), 1),
    }


def by_year(trades: list[dict]) -> dict[str, dict[str, Any]]:
    years = sorted({t["day"][:4] for t in trades})
    return {y: stats([t for t in trades if t["day"][:4] == y]) for y in years}


def percentile(strategy_sum: float, random_sums: np.ndarray) -> float:
    return round(percentile_of(strategy_sum, random_sums), 1)


# ── Study ───────────────────────────────────────────────────────────


def study(market: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg = replace(MARKETS[market], **args.overrides)
    print(f"\n=== {market} ===", flush=True)
    df = load_minute_bid_ask(market, args.start_year)
    if df.empty:
        return {"market": market, "error": "no data"}
    print(f"  {len(df)} 1m bars {df.index[0]} → {df.index[-1]}", flush=True)

    # Sessions depend on tz / open / flat only → shared by all rule variants of a market
    sessions = {mult: build_sessions(df, cfg, mult) for mult in (0.0, 1.0, 2.0)}

    def run(c: OrbConfig, mult: float = 1.0) -> dict[str, Any]:
        s = sessions[mult]
        if (c.session_open, c.flat_time) != (cfg.session_open, cfg.flat_time):
            s = build_sessions(df, c, mult)
        return run_backtest(s, c)

    net = run(cfg)
    trades = net["trades"]
    gross = run(cfg, 0.0)["trades"]
    stress = run(cfg, 2.0)["trades"]

    periods = {
        name: {
            "net": stats(in_period(trades, years)),
            "gross": stats(in_period(gross, years)),
            "double_cost": stats(in_period(stress, years)),
        }
        for name, years in PERIODS.items()
    }
    full = periods["full"]["net"]
    print(
        f"  net: {full['trades']} trades, {full['sum_r']:+.1f}R, avg {full['avg_r']:+.3f}R | "
        f"gross {periods['full']['gross']['sum_r']:+.1f}R | 2× costs "
        f"{periods['full']['double_cost']['sum_r']:+.1f}R",
        flush=True,
    )

    rand = random_baseline(sessions[1.0], cfg, trades, runs=args.runs, seed=args.seed)
    year_of = np.array([int(t["day"][:4]) for t in trades])
    r_strat = np.array([t["r"] for t in trades])
    random_test = {}
    for name in ("full", "oos_2022_now"):
        lo, hi = PERIODS[name]
        mask = (year_of >= lo) & (year_of <= hi)
        sums = rand[:, mask].sum(axis=1)
        random_test[name] = {
            "strategy_sum_r": round(float(r_strat[mask].sum()), 1),
            "random_mean_sum_r": round(float(sums.mean()), 1),
            "random_p95_sum_r": round(float(np.quantile(sums, 0.95)), 1),
            "percentile": percentile(float(r_strat[mask].sum()), sums),
        }
    print(f"  random test: better than {random_test['full']['percentile']}% of {args.runs} runs", flush=True)

    grid = []
    for rm in RANGE_MINUTES:
        for sm in SIGNAL_MINUTES:
            for tr in TARGETS:
                c = replace(cfg, range_minutes=rm, signal_bar_minutes=sm, target_r=tr)
                t = run(c)["trades"]
                grid.append(
                    {
                        "range_minutes": rm, "signal_bar_minutes": sm, "target_r": tr,
                        "full": stats(t),
                        "dev": stats(in_period(t, PERIODS["dev_2016_2021"])),
                        "oos": stats(in_period(t, PERIODS["oos_2022_now"])),
                    }
                )
    base_key = (cfg.range_minutes, cfg.signal_bar_minutes, cfg.target_r)
    neighbours = [
        g for g in grid
        if sum(a != b for a, b in zip((g["range_minutes"], g["signal_bar_minutes"], g["target_r"]), base_key)) == 1
    ]

    variants = []
    for label, changes in VARIANTS.get(market, []):
        t = run(replace(cfg, **changes))["trades"]
        variants.append(
            {
                "label": label, "full": stats(t),
                "dev": stats(in_period(t, PERIODS["dev_2016_2021"])),
                "oos": stats(in_period(t, PERIODS["oos_2022_now"])),
            }
        )

    yearly = by_year(trades)
    oos_years = {y: s for y, s in yearly.items() if int(y) >= 2022}
    losing_years = [y for y, s in oos_years.items() if s["sum_r"] < 0]
    oos = periods["oos_2022_now"]
    checks = {
        "sum_r_test_and_confirm_positive": bool(
            periods["test_2022_2024"]["net"]["sum_r"] > 0
            and periods["confirm_2025_now"]["net"]["sum_r"] > 0
        ),
        "avg_r_oos_ge_0.05": bool(oos["net"]["avg_r"] >= CRITERIA["min_avg_r"]),
        "random_percentile_ge_95": bool(
            random_test["full"]["percentile"] >= CRITERIA["min_random_percentile"]
        ),
        "double_cost_oos_positive": bool(oos["double_cost"]["sum_r"] > 0),
        "losing_years_oos_le_1": bool(len(losing_years) <= CRITERIA["max_losing_years"]),
        "neighbours_positive": bool(all(g["full"]["sum_r"] > 0 for g in neighbours)),
    }

    costs = {
        "median_spread": round(float(np.median([t["spread"] for t in trades])), 3) if trades else None,
        "median_risk_points": round(float(np.median([t["risk_points"] for t in trades])), 2) if trades else None,
        "median_cost_r_est": round(float(np.median([t["cost_r_est"] for t in trades])), 4) if trades else None,
        "realised_cost_r_per_trade": round(
            (periods["full"]["gross"]["sum_r"] - full["sum_r"]) / max(full["trades"], 1), 4
        ),
    }

    rng = np.random.default_rng(args.seed)
    picks = sorted(rng.choice(len(trades), size=min(20, len(trades)), replace=False)) if trades else []
    cols = ["day", "direction", "range_high", "range_low", "signal_time", "entry_time",
            "entry", "stop", "target", "exit_time", "exit", "exit_reason", "r"]
    samples = [{k: trades[i][k] for k in cols} for i in picks]
    sample_path = args.output.with_name(f"{args.output.name}_samples_{market}.csv")
    pd.DataFrame(samples, columns=cols).to_csv(sample_path, index=False)

    return {
        "market": market,
        "config": asdict(cfg),
        "data": {"start": df.index[0].isoformat(), "end": df.index[-1].isoformat(), "bars": len(df)},
        "session_days": net["days"],
        "skipped": net["skipped"],
        "exit_reasons": pd.Series([t["exit_reason"] for t in trades]).value_counts().to_dict(),
        "by_direction": {
            d: stats([t for t in trades if t["direction"] == d]) for d in ("long", "short")
        },
        "costs": costs,
        "periods": periods,
        "yearly_net": yearly,
        "random_test": random_test,
        "grid": grid,
        "neighbours_positive": f"{sum(g['full']['sum_r'] > 0 for g in neighbours)}/{len(neighbours)}",
        "variants": variants,
        "losing_years_oos": losing_years,
        "checks": checks,
        "passed": all(checks.values()),
        "samples_csv": str(sample_path),
    }


# ── Report ──────────────────────────────────────────────────────────

_CHECK_LABELS = {
    "sum_r_test_and_confirm_positive": "Summe R nach Kosten > 0 in Prüfung und Bestätigung",
    "avg_r_oos_ge_0.05": "Ø R pro Trade nach Kosten ≥ +0,05 (2022–heute)",
    "random_percentile_ge_95": "Zufallstest: besser als ≥ 95 % der Läufe (gesamt)",
    "double_cost_oos_positive": "Doppelte Kosten: Summe R > 0 (2022–heute)",
    "losing_years_oos_le_1": "Höchstens 1 Verlustjahr ab 2022",
    "neighbours_positive": "Nachbar-Einstellungen positiv (gesamt)",
}
_PERIOD_LABELS = {
    "dev_2016_2021": "Entwicklung 2016–2021",
    "test_2022_2024": "Prüfung 2022–2024",
    "confirm_2025_now": "Bestätigung 2025–heute",
    "oos_2022_now": "2022–heute",
    "full": "Gesamt",
}


def _row(label: str, s: dict[str, Any]) -> str:
    return (
        f"| {label} | {s['trades']} | {s['win_rate']:.1f} % | {s['avg_r']:+.3f} | "
        f"{s['sum_r']:+.1f} | {s['max_dd_r']:.1f} | {s['max_losing_streak']} | {s['neg_months_pct']:.0f} % |"
    )


def render(rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    out = [
        "# ORB 15 min / 2R — Backtest Phase 1",
        "",
        f"Erzeugt: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · Daten: Dukascopy 1-Minuten-Kerzen "
        f"Bid+Ask ab {args.start_year} · Zufallstest: {args.runs} Läufe"
        + (f" · Abweichung von den Plan-Regeln: {args.overrides}" if args.overrides else ""),
        "",
        "Alle Ergebnisse in R. „Netto“ enthält den historischen Spread, Slippage auf jede "
        "Market-/Stop-Ausführung und die Kommission. „Brutto“ = dieselben Regeln ohne Kosten.",
        "",
        "## Ergebnis",
        "",
        "| Markt | Trades | Ø R netto | Summe R netto | Summe R brutto | 2× Kosten | Zufallstest | Bestanden |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        p = r["periods"]["full"]
        out.append(
            f"| {r['market']} | {p['net']['trades']} | {p['net']['avg_r']:+.3f} | {p['net']['sum_r']:+.1f} | "
            f"{p['gross']['sum_r']:+.1f} | {p['double_cost']['sum_r']:+.1f} | "
            f"{r['random_test']['full']['percentile']:.1f} % | {'**JA**' if r['passed'] else '**NEIN**'} |"
        )
    for r in rows:
        c = r["costs"]
        out += [
            "",
            f"## {r['market']}",
            "",
            f"Handelstage: {r['session_days']} · ohne Trade: {r['skipped']} · Ausstiege: {r['exit_reasons']}",
            "",
            f"Kosten: Spread zum Signal (Median) {c['median_spread']} Pkt · Stop-Distanz (Median) "
            f"{c['median_risk_points']} Pkt · geschätzt {c['median_cost_r_est']} R · "
            f"tatsächlich {c['realised_cost_r_per_trade']} R pro Trade (brutto − netto)",
            "",
            "### Kriterien",
            "",
            "| Kriterium | Erfüllt |",
            "|---|---|",
        ]
        out += [f"| {_CHECK_LABELS[k]} | {'ja' if v else 'nein'} |" for k, v in r["checks"].items()]
        out += [
            "",
            "### Zeiträume (netto)",
            "",
            "| Zeitraum | Trades | Treffer | Ø R | Summe R | Max. DD (R) | Längste Verlustserie | Negative Monate |",
            "|---|---|---|---|---|---|---|---|",
        ]
        out += [_row(_PERIOD_LABELS[k], v["net"]) for k, v in r["periods"].items()]
        out += [
            "",
            "| Zeitraum | Summe R brutto | Summe R netto | Summe R bei 2× Kosten |",
            "|---|---|---|---|",
        ]
        out += [
            f"| {_PERIOD_LABELS[k]} | {v['gross']['sum_r']:+.1f} | {v['net']['sum_r']:+.1f} | {v['double_cost']['sum_r']:+.1f} |"
            for k, v in r["periods"].items()
        ]
        out += [
            "",
            "### Je Jahr (netto)",
            "",
            "| Jahr | Trades | Treffer | Ø R | Summe R | Max. DD (R) | Längste Verlustserie | Negative Monate |",
            "|---|---|---|---|---|---|---|---|",
        ]
        out += [_row(y, s) for y, s in r["yearly_net"].items()]
        out += ["", "| Richtung | Trades | Treffer | Ø R | Summe R | Max. DD (R) | Längste Verlustserie | Negative Monate |",
                "|---|---|---|---|---|---|---|---|"]
        out += [_row(d, s) for d, s in r["by_direction"].items()]
        out += [
            "",
            "### Zufallstest",
            "",
            "| Zeitraum | Strategie Summe R | Zufall Mittel | Zufall 95 %-Grenze | Strategie besser als |",
            "|---|---|---|---|---|",
        ]
        out += [
            f"| {_PERIOD_LABELS[k]} | {v['strategy_sum_r']:+.1f} | {v['random_mean_sum_r']:+.1f} | "
            f"{v['random_p95_sum_r']:+.1f} | {v['percentile']:.1f} % |"
            for k, v in r["random_test"].items()
        ]
        out += [
            "",
            f"### Robustheit (Summe R netto, gesamt / 2016–2021 / 2022–heute) — Nachbarn positiv: {r['neighbours_positive']}",
            "",
            "| Range | Signalkerze | Ziel | Trades | Ø R | Gesamt | 2016–2021 | 2022–heute |",
            "|---|---|---|---|---|---|---|---|",
        ]
        out += [
            f"| {g['range_minutes']} min | {g['signal_bar_minutes']} min | {g['target_r']} R | {g['full']['trades']} | "
            f"{g['full']['avg_r']:+.3f} | {g['full']['sum_r']:+.1f} | {g['dev']['sum_r']:+.1f} | {g['oos']['sum_r']:+.1f} |"
            for g in r["grid"]
        ]
        if r["variants"]:
            out += [
                "",
                "### Varianten (netto)",
                "",
                "| Variante | Trades | Ø R | Summe R gesamt | 2016–2021 | 2022–heute |",
                "|---|---|---|---|---|---|",
            ]
            out += [
                f"| {v['label']} | {v['full']['trades']} | {v['full']['avg_r']:+.3f} | "
                f"{v['full']['sum_r']:+.1f} | {v['dev']['sum_r']:+.1f} | {v['oos']['sum_r']:+.1f} |"
                for v in r["variants"]
            ]
        out += ["", f"20 Stichproben-Trades für die Chart-Kontrolle: `{r['samples_csv']}`"]
    return "\n".join(out) + "\n"


def main() -> None:
    logging.getLogger("DUKASCRIPT").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="ORB 15 min / 2R — phase-1 backtest")
    parser.add_argument("--markets", default="US100,GER40,XAUUSD")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--runs", type=int, default=1000, help="random test runs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                        help="override a config field for all markets, e.g. --set gap_filter=true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/orb_2r"),
                        help="path prefix for the .json / .md / sample files")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.overrides = {k: yaml.safe_load(v) for k, v in (item.split("=", 1) for item in args.set)}

    rows = [study(m.strip(), args) for m in args.markets.split(",") if m.strip()]
    rows = [r for r in rows if "error" not in r]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "dukascopy 1m bid+ask",
        "start_year": args.start_year,
        "overrides": args.overrides,
        "random_runs": args.runs,
        "criteria": CRITERIA,
        "markets": rows,
        "passed": [r["market"] for r in rows if r["passed"]],
    }
    args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=str))
    args.output.with_suffix(".md").write_text(render(rows, args))
    print(f"\nSaved: {args.output.with_suffix('.md')} / .json")
    for r in rows:
        print(f"  {r['market']}: {'PASS' if r['passed'] else 'FAIL'} {r['checks']}")


if __name__ == "__main__":
    main()
