"""Does H-009 fit a prop firm? A daily P&L path through the prop simulator.

Everything about H-009 so far was *monthly*. A prop evaluation is judged on
**daily** loss limits and a total drawdown, so this builds the daily path the
monthly study never needed: the basket is rebalanced at month end, but every day
inside the month carries its own price move plus one day of swap accrual (tripled
on Wednesdays, which is the 3-day rollover). It then runs the reusable
``trading_lab.prop_sim`` evaluation - FTMO 2-step, Lucid 1-step, a stylized
1-step, and instant funding - at 1x and at the 2x that a 2 %/month goal needs.

It also prints the monthly distribution and the losing-streak profile, because
those are what decide a prop outcome, not the CAGR.

  .venv/bin/python studies/carry_prop_sim.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "."); sys.path.insert(0, "studies")
from trading_lab.duka_swaps import markup_bp_yr, pip_size  # noqa: E402
from trading_lab.prop_sim import FIRMS, evaluate, simulate_funded  # noqa: E402
from trading_lab.rates import load_rates  # noqa: E402
import run_carry  # noqa: E402

K, VIX_THR = 4, 0.70
ACCOUNT = 100_000.0


def daily_path(rates: pd.DataFrame, prices: pd.DataFrame, markup: dict[str, float],
               vix_pct: pd.Series, *, markup_scale: float = 1.0,
               triple: bool = True) -> pd.DataFrame:
    """Daily H-009 return on the position notional, plus the leg bookkeeping."""
    me = prices.resample("ME").last().index
    rows = []
    for a, b in zip(me[:-1], me[1:]):
        # legs are fixed for the month (set at the rebalance, using rates at `a`)
        aa = a.tz_localize(None) if a.tzinfo else a
        c = {p: float((rates[p.split("/")[0]] - rates[p.split("/")[1]])
                      .reindex([aa], method="ffill").iloc[0]) for p in prices.columns}
        ranked = sorted(c, key=c.get)
        lon, sho = ranked[-K:], ranked[:K]
        spread = float(np.mean([c[p] for p in lon]) - np.mean([c[p] for p in sho]))
        fee = markup_scale * float(np.mean([markup[p] for p in lon])
                                   + np.mean([markup[p] for p in sho]))
        active = bool(vix_pct.reindex([aa], method="ffill").iloc[0] < VIX_THR)
        seg = prices.loc[a:b].iloc[1:]
        prev = prices.loc[a:b].shift(1).loc[seg.index]
        ret = seg.div(prev).sub(1.0)
        for t, r in ret.iterrows():
            tt = t.tz_localize(None) if t.tzinfo else t
            if not active:
                rows.append({"date": tt, "ret": 0.0, "price": 0.0, "swap": 0.0,
                             "longs": ",".join(lon), "shorts": ",".join(sho)})
                continue
            price_r = float(np.mean([r[p] for p in lon]) - np.mean([r[p] for p in sho]))
            day = (spread / 100.0 - fee) / 365.0
            if triple and tt.weekday() == 2:      # Wednesday books the weekend
                day *= 3.0
            rows.append({"date": tt, "ret": price_r + day, "price": price_r,
                         "swap": day, "longs": ",".join(lon), "shorts": ",".join(sho)})
    return pd.DataFrame(rows).set_index("date")


def streaks(x: np.ndarray) -> tuple[int, int]:
    best = cur = 0
    for v in x:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    worst = cur = 0
    for v in x:
        cur = cur + 1 if v <= 0 else 0
        worst = max(worst, cur)
    return best, worst


def main() -> None:
    rates = load_rates()
    daily = {p: run_carry.daily(p, c) for p, c in run_carry.PAIRS.items()}
    prices = pd.DataFrame(daily).sort_index().resample("1D").last().ffill()
    last = {p: float(s.dropna().iloc[-1]) for p, s in daily.items()}
    markup = {p: markup_bp_yr(p, last[p], pip_size(p.split("/")[1])) / 1e4
              for p in run_carry.PAIRS}

    vix = yf.download("^VIX", start="2018-12-01", end="2026-10-01", interval="1d",
                      progress=False, auto_adjust=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix_pct = vix.rolling(250, min_periods=60).rank(pct=True)

    path = daily_path(rates, prices, markup, vix_pct)
    m = path.resample("ME")["ret"].apply(lambda s: (1 + s).prod() - 1)
    m = m[m != 0]
    print(f"### H-009 daily path: {len(path)} days, {len(m)} active months, "
          f"{path.index.min().date()}..{path.index.max().date()}")
    print(f"    daily: mean {path['ret'].mean()*100:+.4f}%  sd {path['ret'].std()*100:.3f}%  "
          f"worst {path['ret'].min()*100:+.2f}%  best {path['ret'].max()*100:+.2f}%")
    print(f"    monthly: mean {m.mean()*100:+.2f}%  sd {m.std()*100:.2f}%  "
          f"worst {m.min()*100:+.2f}%  best {m.max()*100:+.2f}%")
    print(f"    positive months {100*(m > 0).mean():.0f}%   "
          f"longest losing run {streaks(m.to_numpy())[0]} months  "
          f"(incl. flat {streaks(m.to_numpy())[1]})")
    print("    monthly returns %:", {str(i.date())[:7]: round(float(v)*100, 1)
                                     for i, v in m.items()})

    # ── prop evaluation at several leverages ──
    print(f"\n### prop-firm evaluation, account {ACCOUNT:,.0f} USD, block-bootstrap MC")
    for lev in (1.0, 2.0):
        pnl = path["ret"].to_numpy() * ACCOUNT * lev
        print(f"\n  --- leverage {lev:.1f}x  (annualised vol "
              f"{path['ret'].std()*np.sqrt(252)*lev*100:.1f}%) ---")
        for name in ("ftmo", "lucid", "one_step", "instant_25k", "instant_100k"):
            firm = FIRMS[name]
            if firm.phase_targets:
                res = evaluate(pnl, firm, runs=4000)
                print(f"    {name:12s} {res.summary()}")
            else:
                res = simulate_funded(pnl, firm, runs=4000, horizon_days=len(pnl))
                print(f"    {name:12s} {res.summary()}")
    print("\n  (ftmo = 2-step 10%+5%, static 10% total / 5% daily; "
          "lucid = 1-step 5%, EOD-trailing 4% / 2.4% daily;")
    print("   instant_25k = 2.4% daily / 4% static; one_step = 8% target / 5% daily / 10% static)")


if __name__ == "__main__":
    main()
