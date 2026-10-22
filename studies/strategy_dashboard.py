#!/usr/bin/env python3
"""Monthly dashboard for a strategy: every number, month by month, to a file.

For each registered strategy × instrument it writes, under
``artifacts/dashboards/``:

* ``<strategy>_<asset>_monthly.csv`` — one row per month (Excel-friendly),
* ``<strategy>_<asset>.md``         — the same as readable tables plus key
  metrics, constancy diagnostics and prop-firm outcomes,
* ``<strategy>_<asset>.html``       — a self-contained dashboard (no JS deps).

Monthly "payout" is the instant-funding view: 80 % of that month's profit above
the starting balance. Months are calendar months of the trade log.

  .venv/bin/python studies/strategy_dashboard.py --risk-pct 0.5
"""
from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trading_lab.minute_data import load_minute_bid_ask
from trading_lab.prop_sim import FIRMS, daily_pnl, evaluate, risk_usd_for, simulate_funded
from trading_lab.strategies.orb_2r import skipped_days
from trading_lab.strategies.orb_rejection_ifvg import OrbRejConfig, run_backtest

ASSETS = ("US100", "SP500")
PROP_ORDER = ("ftmo", "lucid", "one_step", "instant_25k", "instant_100k")


def _trading_days(a: int, b: int) -> list[date]:
    skip: set[date] = set()
    for y in range(a, b + 1):
        skip |= skipped_days("nyse", y)
    return [d.date() for d in pd.bdate_range(f"{a}-01-01", f"{b}-12-31")
            if d.weekday() < 5 and d.date() not in skip]


def load_orb_fade(asset: str, start_year: int, end_year: int):
    df = load_minute_bid_ask(asset, start_year, verbose=False)
    years = df.index.tz_convert("UTC").year
    sub = df[(years >= start_year) & (years <= end_year)]
    res = run_backtest(sub, OrbRejConfig(asset=asset, asset_label=asset, mode="naive"))
    return res["trades"], _trading_days(start_year, end_year)


STRATEGIES = {"orb_fade": load_orb_fade}


@dataclass
class Cell:
    strategy: str
    asset: str
    start: int
    end: int
    monthly: pd.DataFrame
    key: dict
    prop: dict[str, dict]


def build_cell(strategy: str, asset: str, start: int, end: int, risk_pct: float,
               runs: int = 500) -> Cell:
    trades, days = STRATEGIES[strategy](asset, start, end)
    risk = risk_pct / 100.0
    rs = np.array([float(t["r"]) for t in trades])
    n = len(rs)
    d = pd.Series(rs * risk_pct, index=pd.to_datetime([t["day"] for t in trades]))
    daily = d.groupby(level=0).sum()
    full = pd.Series(0.0, index=pd.to_datetime([str(x) for x in days]))
    full.update(daily)
    monthly = full.resample("ME").sum()
    monthly = monthly[monthly.index <= pd.Timestamp(trades[-1]["day"])]

    mdf = pd.DataFrame({"ret_pct": monthly})
    mdf["month"] = mdf.index.strftime("%Y-%m")
    mdf["cum_pct"] = mdf["ret_pct"].cumsum()
    mdf["trades"] = d.resample("ME").count().reindex(monthly.index).fillna(0).astype(int)
    mdf["sum_r"] = d.resample("ME").sum().reindex(monthly.index).div(risk_pct).fillna(0)
    # instant-funding payout view (80 % split of a profitable month)
    mdf["payout_25k_usd"] = (mdf["ret_pct"] / 100 * 25_000 * 0.8).clip(lower=0).round(0)
    mdf["payout_100k_usd"] = (mdf["ret_pct"] / 100 * 100_000 * 0.8).clip(lower=0).round(0)
    mdf = mdf.reset_index(drop=True)[["month", "trades", "sum_r", "ret_pct", "cum_pct",
                                     "payout_25k_usd", "payout_100k_usd"]]
    mdf["sum_r"] = mdf["sum_r"].round(1)
    mdf["ret_pct"] = mdf["ret_pct"].round(2)
    mdf["cum_pct"] = mdf["cum_pct"].round(1)

    rets = mdf["ret_pct"].to_numpy()
    neg_streak = cur = worst = 0
    for x in rets:
        cur = cur + 1 if x < 0 else 0
        worst = max(worst, cur)
    years = max((pd.Timestamp(trades[-1]["day"]) - pd.Timestamp(trades[0]["day"])).days / 365.25, 1e-9)
    equity = 100 + np.cumsum(rs * risk_pct)
    peak = np.maximum.accumulate(equity)
    key = {
        "trades": n,
        "trades_per_week": round(n / (len(days) / 5), 2),
        "win_rate_pct": round(100 * (rs > 0).mean(), 1),
        "expectancy_r": round(float(rs.mean()), 3),
        "profit_factor": round(float(rs[rs > 0].sum() / abs(rs[rs < 0].sum())), 2),
        "total_r": round(float(rs.sum()), 1),
        "total_pct": round(float((rs * risk_pct).sum()), 1),
        "per_year_pct": round(float((rs * risk_pct).sum()) / years, 1),
        "max_dd_pct": round(float((equity - peak).min()), 1),
        "months": int(len(rets)),
        "monthly_mean_pct": round(float(np.mean(rets)), 2),
        "monthly_median_pct": round(float(np.median(rets)), 2),
        "monthly_std_pct": round(float(np.std(rets, ddof=1)), 2),
        "monthly_min_pct": round(float(rets.min()), 2),
        "monthly_max_pct": round(float(rets.max()), 2),
        "positive_months_pct": round(100 * float((rets > 0).mean()), 1),
        "months_ge_1pct": int((rets >= 1).sum()),
        "months_ge_3pct": int((rets >= 3).sum()),
        "months_negative": int((rets < 0).sum()),
        "longest_negative_month_streak": worst,
        "month_autocorr_lag1": round(float(pd.Series(rets).autocorr(1)), 2) if len(rets) > 2 else 0.0,
        "cv_monthly": round(float(np.std(rets, ddof=1) / abs(np.mean(rets))), 2) if np.mean(rets) else 0.0,
    }

    risk_usd_25 = risk_usd_for(25_000, risk_pct)
    stream = daily_pnl(trades, risk_usd=risk_usd_25, days=days)
    prop: dict[str, dict] = {}
    for name in PROP_ORDER:
        firm = FIRMS[name]
        ru = risk_usd_for(firm.account_size, risk_pct)
        s = daily_pnl(trades, risk_usd=ru, days=days)
        if firm.is_instant:
            r = simulate_funded(s, firm, runs=runs)
            prop[name] = {"kind": "instant", "survival_pct": round(r.survival_rate * 100, 1),
                          "payout_rate_pct": round(r.payout_rate * 100, 1),
                          "mean_payout_usd": round(r.mean_payout_usd, 0)}
        else:
            r = evaluate(s, firm, runs=runs)
            prop[name] = {"kind": "eval", "pass_pct": round(r.pass_rate * 100, 1),
                          "phase_pct": [round(x * 100, 1) for x in r.phase_pass_rate],
                          "median_days": round(r.median_days, 0),
                          "cost_per_funded_usd": round(r.cost_per_funded, 0)}
    return Cell(strategy, asset, start, end, mdf, key, prop)


# ── Rendering ───────────────────────────────────────────────────────


def _md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def to_markdown(cells: list[Cell], risk_pct: float) -> str:
    out = [f"# Strategy dashboard — risk {risk_pct}% per trade",
           f"_generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}_", ""]
    for c in cells:
        out += [f"## {c.strategy} / {c.asset} ({c.start}–{c.end})", "", "### Key numbers", "",
                "\n".join(f"- **{k}**: {v}" for k, v in c.key.items()), "",
                "### Prop-firm outcomes", "",
                _md_table(pd.DataFrame(
                    [{"firm": FIRMS[n].name, **v} for n, v in c.prop.items()])), "",
                "### Monthly", "", _md_table(c.monthly), ""]
    return "\n".join(out)


def to_html(cells: list[Cell], risk_pct: float) -> str:
    parts = [f"""<!doctype html><meta charset="utf-8">
<title>Strategy dashboard</title>
<style>
 body {{ font: 14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif; margin: 24px; color:#111; }}
 h1 {{ font-size: 20px; }} h2 {{ margin-top: 32px; border-bottom: 2px solid #eee; padding-bottom:4px; }}
 table {{ border-collapse: collapse; margin: 8px 0 20px; }}
 th, td {{ border: 1px solid #ddd; padding: 3px 8px; text-align: right; }}
 th {{ background: #f6f6f6; }} td:first-child, th:first-child {{ text-align: left; }}
 .pos {{ color:#0a0; }} .neg {{ color:#c00; }} .kv {{ columns: 3; }}
 .grid {{ display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); column-gap:24px; }}
</style>
<h1>Strategy dashboard — risk {risk_pct}% per trade</h1>
<p>generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}</p>"""]

    def tbl(df: pd.DataFrame, color_col: str | None = None) -> str:
        head = "<tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns) + "</tr>"
        body = []
        for _, row in df.iterrows():
            tds = []
            for col in df.columns:
                v = row[col]
                cls = ""
                if color_col and col == color_col:
                    cls = " class='pos'" if float(v) >= 0 else " class='neg'"
                tds.append(f"<td{cls}>{html.escape(str(v))}</td>")
            body.append("<tr>" + "".join(tds) + "</tr>")
        return f"<table>{head}{''.join(body)}</table>"

    for c in cells:
        kv = "".join(f"<div><b>{html.escape(str(k))}</b>: {v}</div>" for k, v in c.key.items())
        prop = pd.DataFrame([{"firm": FIRMS[n].name, **v} for n, v in c.prop.items()])
        parts.append(f"<h2>{c.strategy} / {c.asset} ({c.start}–{c.end})</h2>"
                     f"<div class='kv'>{kv}</div><h3>Prop-firm outcomes</h3>{tbl(prop)}"
                     f"<h3>Monthly</h3>{tbl(c.monthly, 'ret_pct')}")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default="orb_fade", choices=sorted(STRATEGIES))
    ap.add_argument("--assets", default=",".join(ASSETS))
    ap.add_argument("--start-year", type=int, default=2019)
    ap.add_argument("--end-year", type=int, default=datetime.now(timezone.utc).year)
    ap.add_argument("--risk-pct", type=float, default=0.5)
    ap.add_argument("--runs", type=int, default=500, help="Monte-Carlo runs per firm")
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/dashboards"))
    args = ap.parse_args()

    cells = [build_cell(args.strategy, a, args.start_year, args.end_year, args.risk_pct, args.runs)
             for a in args.assets.split(",") if a.strip()]
    args.outdir.mkdir(parents=True, exist_ok=True)
    for c in cells:
        stem = args.outdir / f"{c.strategy}_{c.asset}"
        c.monthly.to_csv(stem.with_name(stem.name + "_monthly.csv"), index=False)
        (stem.with_suffix(".md")).write_text(to_markdown([c], args.risk_pct))
        (stem.with_suffix(".html")).write_text(to_html([c], args.risk_pct))
        print(f"{c.strategy}/{c.asset}: {c.key['months']} months, "
              f"mean {c.key['monthly_mean_pct']}%/mo, pos {c.key['positive_months_pct']}% "
              f"→ {stem}.html")
    (args.outdir / f"{args.strategy}_all.md").write_text(to_markdown(cells, args.risk_pct))
    (args.outdir / f"{args.strategy}_all.html").write_text(to_html(cells, args.risk_pct))
    print(f"combined: {args.outdir / (args.strategy + '_all.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
