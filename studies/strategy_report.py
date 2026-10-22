#!/usr/bin/env python3
"""Standard strategy report: every number a sim should answer.

For a strategy over a period and an instrument it prints: test window, trades per
week/month/year, win rate, expectancy (R and %), profit factor, monthly return
distribution, drawdown, streaks, Sharpe/Sortino, long/short split, exit reasons,
and the prop-firm outcome (pass rate, days to pass, cost per funded).

The trade log is strategy-agnostic — to add a strategy, register a loader that
returns ``(trades, trading_days)`` (same shape as ``studies/run_prop_sim.py``).

  .venv/bin/python studies/strategy_report.py --asset US100 --start-year 2024 --firm lucid
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from trading_lab.minute_data import load_minute_bid_ask
from trading_lab.prop_sim import FIRMS, daily_pnl, evaluate, get_firm, risk_usd_for, simulate_funded
from trading_lab.strategies.orb_2r import skipped_days
from trading_lab.strategies.orb_rejection_ifvg import OrbRejConfig, run_backtest


def _trading_days(start_year: int, end_year: int) -> list[date]:
    skip: set[date] = set()
    for y in range(start_year, end_year + 1):
        skip |= skipped_days("nyse", y)
    return [d.date() for d in pd.bdate_range(f"{start_year}-01-01", f"{end_year}-12-31")
            if d.weekday() < 5 and d.date() not in skip]


def load_orb_fade(asset: str, start_year: int, end_year: int):
    df = load_minute_bid_ask(asset, start_year, verbose=False)
    years = df.index.tz_convert("UTC").year
    sub = df[(years >= start_year) & (years <= end_year)]
    res = run_backtest(sub, OrbRejConfig(asset=asset, asset_label=asset, mode="naive"))
    return res["trades"], _trading_days(start_year, end_year)


STRATEGIES = {"orb_fade": load_orb_fade}


def _max_dd_pct(equity_pct: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_pct)
    return float((equity_pct - peak).min())


def _streaks(rs: list[float]) -> tuple[int, int]:
    worst = cur = 0
    for r in rs:
        cur = cur + 1 if r < 0 else 0
        worst = max(worst, cur)
    return worst


def report(trades: list[dict], days: list[date], *, risk_pct: float, firm_name: str,
           runs: int = 5000) -> dict:
    rs = np.array([float(t["r"]) for t in trades])
    n = len(rs)
    wins, losses = rs[rs > 0], rs[rs < 0]
    first_day, last_day = trades[0]["day"], trades[-1]["day"]
    span_days = (pd.Timestamp(last_day) - pd.Timestamp(first_day)).days + 1
    weeks = span_days / 7.0
    months = span_days / 30.44

    risk = risk_pct / 100.0
    pnl_pct = rs * risk_pct                       # non-compounded, % of account
    equity = 100.0 + np.cumsum(pnl_pct)
    daily = pd.Series(pnl_pct, index=pd.to_datetime([t["day"] for t in trades])).groupby(level=0).sum()
    full = pd.Series(0.0, index=pd.to_datetime([str(d) for d in days]))
    full.update(daily)
    monthly = full.resample("ME").sum().dropna()
    monthly = monthly[monthly.index <= pd.Timestamp(last_day)]

    firm = get_firm(firm_name)
    risk_usd = risk_usd_for(firm.account_size, risk_pct)
    stream = daily_pnl(trades, risk_usd=risk_usd, days=days)
    if firm.is_instant:
        res = simulate_funded(stream, firm, runs=runs)
        prop = {"firm": firm.name, "risk_usd": round(risk_usd, 0), "kind": "instant",
                "survival_pct": round(res.survival_rate * 100, 1),
                "payout_rate_pct": round(res.payout_rate * 100, 1),
                "mean_payout_usd": round(res.mean_payout_usd, 0),
                "median_days_to_breach": round(res.median_days_to_breach, 0)
                if res.breaches else None,
                "worst_day_usd": round(float(stream.min()), 0),
                "worst_trade_usd": round(float(rs.min()) * risk_usd, 0)}
    else:
        prop = evaluate(stream, firm, runs=runs)
        prop = {"firm": prop.firm, "risk_usd": round(risk_usd, 0), "kind": "eval",
                "pass_rate_pct": round(prop.pass_rate * 100, 1),
                "phase_pass_pct": [round(r * 100, 1) for r in prop.phase_pass_rate],
                "median_days": round(prop.median_days, 0), "p95_days": round(prop.p95_days, 0),
                "cost_per_funded_usd": round(prop.cost_per_funded, 0),
                "worst_day_usd": round(float(stream.min()), 0),
                "worst_trade_usd": round(float(rs.min()) * risk_usd, 0)}

    longs = [t for t in trades if t["direction"] == "long"]
    shorts = [t for t in trades if t["direction"] == "short"]
    reasons = pd.Series([t["exit_reason"] for t in trades]).value_counts().to_dict()

    def _side(ts):
        rr = [float(t["r"]) for t in ts]
        return {"n": len(ts), "wr": round(100 * sum(1 for r in rr if r > 0) / len(rr), 1) if rr else 0.0,
                "sum_r": round(sum(rr), 1)}

    return {
        "period": f"{first_day} .. {last_day} ({span_days} days, ~{len(days)} trading days)",
        "trades": n,
        "trades_per_week": round(n / weeks, 2),
        "trades_per_month": round(n / months, 1),
        "trades_per_day": round(n / len(days), 2),
        "win_rate_pct": round(100 * len(wins) / n, 1),
        "avg_win_r": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss_r": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "payoff": round(float(wins.mean() / abs(losses.mean())), 2) if len(losses) and len(wins) else 0.0,
        "expectancy_r": round(float(rs.mean()), 3),
        "expectancy_pct": round(float(rs.mean()) * risk_pct, 3),
        "profit_factor": round(float(wins.sum() / abs(losses.sum())), 2) if len(losses) else float("inf"),
        "total_r": round(float(rs.sum()), 1),
        "total_pct": round(float(pnl_pct.sum()), 1),
        "per_year_pct": round(float(pnl_pct.sum()) * 365.0 / span_days, 1),
        "monthly_mean_pct": round(float(monthly.mean()), 2) if len(monthly) else 0.0,
        "monthly_median_pct": round(float(monthly.median()), 2) if len(monthly) else 0.0,
        "positive_months_pct": round(100 * float((monthly > 0).mean()), 1) if len(monthly) else 0.0,
        "best_month_pct": round(float(monthly.max()), 1) if len(monthly) else 0.0,
        "worst_month_pct": round(float(monthly.min()), 1) if len(monthly) else 0.0,
        "max_dd_pct": round(_max_dd_pct(equity), 1),
        "longest_losing_streak": _streaks(list(rs)),
        "sharpe_daily_ann": round(float(full.mean() / full.std() * np.sqrt(252)), 2) if full.std() else 0.0,
        "long": _side(longs), "short": _side(shorts), "exit_reasons": reasons,
        "prop": prop,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default="orb_fade", choices=sorted(STRATEGIES))
    ap.add_argument("--asset", default="US100")
    ap.add_argument("--firm", default="lucid", choices=sorted(FIRMS))
    ap.add_argument("--risk-pct", type=float, default=0.5)
    ap.add_argument("--start-year", type=int, default=2019)
    ap.add_argument("--end-year", type=int, default=datetime.now(timezone.utc).year)
    ap.add_argument("--runs", type=int, default=5000)
    args = ap.parse_args()

    trades, days = STRATEGIES[args.strategy](args.asset, args.start_year, args.end_year)
    if not trades:
        print("no trades")
        return 1
    rep = report(trades, days, risk_pct=args.risk_pct, firm_name=args.firm, runs=args.runs)
    print(f"\n=== {args.strategy} / {args.asset} / risk {args.risk_pct}% per trade ===")
    for k, v in rep.items():
        if k in ("long", "short", "exit_reasons", "prop"):
            continue
        print(f"  {k:24s}: {v}")
    print(f"  long                     : {rep['long']}")
    print(f"  short                    : {rep['short']}")
    print(f"  exit_reasons             : {rep['exit_reasons']}")
    print(f"  prop                     : {rep['prop']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
