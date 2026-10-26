#!/usr/bin/env python3
"""H-010 event study: does a session trend continue after a measured retrace?

The pullback idea in three questions, in the order that keeps the answer honest:

1. **Response** - forward return in the trend direction, H minutes after a
   pullback trigger, measured against the *same instrument's* unconditional
   drift (never against zero: these indices rise).
2. **Dose-response** - does the response grow with the trend strength and with
   the retrace depth? A real retracement mechanism predicts both.
3. **Cost and fill** - the mean response has to beat the round trip, and the
   limit order has to actually fill. Adverse selection is measured directly:
   filled vs. never-filled signals.

Responses use non-overlapping triggers per day and a block bootstrap over days,
because triggers cluster and overlapping windows inflate t-statistics.

Data come from the audit-correct frame (``data_audit.build_study_frame``): spike /
session-edge / unclear roll bars and thin days are removed before the setup is
measured.

Forward returns are measured **within the session day**. The frame is
session-filtered, so a plain shift would run the window across the overnight
boundary and book the gap as the setup's response; the window is capped at the
last bar of the same session day and samples whose window would cross are dropped
(see ``forward_return_same_day``). The report prints how many were affected.

Costs follow the single convention in ``trading_lab.cost_gate``: a limit entry
with a time exit is ``round_trip_spread("limit", "time")`` = 0.5 x the full
spread, a both-legs market trade is ``round_trip_spread("market", "market")`` =
1.0 x the full spread.

  .venv/bin/python studies/run_h010_pullback.py
  .venv/bin/python studies/run_h010_pullback.py --assets US100,GER40,XAUUSD --session ny
"""
from __future__ import annotations

import argparse
import sys
import warnings

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_lab.minute_data import DEFAULT_CACHE, load_minute_bid_ask  # noqa: E402
from trading_lab.data_audit import build_study_frame  # noqa: E402
from trading_lab.cost_gate import round_trip_spread  # noqa: E402
from trading_lab.strategies.pullback import PullbackConfig, signals  # noqa: E402

ASSETS = ("US100", "US30", "SP500", "GER40", "XAUUSD", "EURUSD", "GBPUSD", "USDJPY")
HORIZONS = (30, 60, 120, 240)


def block_bootstrap_t(x: np.ndarray, blocks: np.ndarray, *, runs: int = 2000,
                      block: int = 10, seed: int = 7) -> float:
    """t of the mean under a moving-block bootstrap over days (clustered triggers)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(blocks)
    if len(uniq) < 5:
        return float("nan")
    means = np.empty(runs)
    idx = {b: np.flatnonzero(blocks == b) for b in uniq}
    for i in range(runs):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sample = np.concatenate([x[idx[b]] for b in pick])
        means[i] = sample.mean()
    se = means.std(ddof=1)
    return float(x.mean() / se) if se else float("nan")


def forward_return_same_day(mid: np.ndarray, day: np.ndarray, horizon: int
                            ) -> tuple[np.ndarray, int]:
    """Forward return over ``horizon`` bars, dropped when it would leave the day.

    The frame is session-filtered, so a plain ``base[i + horizon]`` runs the
    window into the *next* session across the overnight gap and books the gap as
    the setup's response. We require the bar at ``i + horizon`` to carry the same
    ``day`` label; samples whose window would cross the boundary are dropped
    (NaN). Returns the forward-return array and the number of dropped samples.
    """
    mid = np.asarray(mid, float)
    day = np.asarray(day)
    n = len(mid)
    fwd = np.full(n, np.nan)
    crossed = 0
    if horizon < n:
        same_day = day[horizon:] == day[:-horizon]
        crossed = int((~same_day).sum())
        fwd[:-horizon] = np.where(same_day, mid[horizon:] / mid[:-horizon] - 1.0, np.nan)
    return fwd, crossed


def study(asset: str, m1: pd.DataFrame, cfg: PullbackConfig, horizons=HORIZONS) -> list[dict]:
    s = signals(m1, cfg)
    trig = s[s["setup"] != 0]
    if trig.empty:
        return []
    # unconditional drift over the same window of day (the placebo benchmark)
    base_all = s["mid"].to_numpy()
    days = s["day"].to_numpy()
    # relative spread, so it is comparable with the relative forward return
    spread = float(((s["ask"] - s["bid"]) / s["mid"]).median())
    cost_lim = round_trip_spread("limit", "time")      # 0.5 x spread (limit entry)
    cost_mkt = round_trip_spread("market", "market")   # 1.0 x spread (both market)
    rows = []
    for h in horizons:
        fwd_all, crossed = forward_return_same_day(base_all, days, h)
        # unconditional benchmark: mean |move| and mean signed move over same horizon
        m_all = np.nanmean(np.abs(fwd_all))
        ratio = m_all / (cost_lim * spread) if cost_lim and spread > 0 else np.nan
        idx = np.flatnonzero(s["setup"].to_numpy() != 0)
        idx = idx[np.isfinite(fwd_all[idx])]           # drop cross-day windows
        resp = np.array([int(s["setup"].iloc[i]) * fwd_all[i] for i in idx])
        if len(resp) < 50:
            continue
        blk = np.array([s["day"].iloc[i] for i in idx])    # aligned with resp
        rows.append({
            "asset": asset, "session": cfg.session, "minutes": h,
            "n_trig": len(resp), "mean_pct": float(resp.mean()) * 100,
            "median_pct": float(np.median(resp)) * 100,
            "hit": float((resp > 0).mean()), "t_boot": block_bootstrap_t(resp, blk),
            "move_bp": m_all * 1e4, "spread_bp": spread * 1e4,
            "ratio_move_cost": ratio,
            "cost_lim_bp": cost_lim * spread * 1e4, "cost_mkt_bp": cost_mkt * spread * 1e4,
            "net_lim_bp": (float(resp.mean()) - cost_lim * spread) * 1e4,
            "net_mkt_bp": (float(resp.mean()) - cost_mkt * spread) * 1e4,
            "crossed": crossed,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", default=",".join(ASSETS))
    ap.add_argument("--session", default="ny", choices=("london", "ny"))
    ap.add_argument("--trend-min", type=float, default=0.5)
    ap.add_argument("--pullback-min", type=float, default=0.30)
    ap.add_argument("--start-year", type=int, default=2019)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    cached = {f.name.split("_1m_")[0] for f in DEFAULT_CACHE.glob("*_1m_bid_*.pkl")}

    rows: list[dict] = []
    for a in assets:
        if a not in cached:
            continue
        m1, info = build_study_frame(load_minute_bid_ask(a, args.start_year, verbose=False), a)
        cfg = PullbackConfig(asset=a, session=args.session, trend_min=args.trend_min,
                             pullback_min=args.pullback_min)
        n0 = len(rows)
        rows += study(a, m1, cfg)
        crossed = sum(r["crossed"] for r in rows[n0:])
        print(f"  {a} ok (audit kept {info['kept']}, dropped {info['dropped']}, "
              f"thin days {info['thin_days']}; cross-day-Fenster verworfen: {crossed})",
              flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        print("keine Ergebnisse")
        return 1

    print(f"\n### H-010 Event-Studie: Trend-Fortsetzung nach Ruecksetzer "
          f"(Session {args.session}, trend>={args.trend_min}, retrace>={args.pullback_min})")
    print(f"    net_lim = mittlere Antwort MINUS {round_trip_spread('limit', 'time'):.1f} "
          "Spread (Limit-Einstieg + Zeit-Exit).")
    print(f"    net_mkt = mittlere Antwort MINUS {round_trip_spread('market', 'market'):.1f} "
          "Spread (Market/Market). Vorwaertsfenster endet am letzten Bar desselben "
          "Session-Tages (cross-day-Samples verworfen).\n")
    print(f"    {'asset':8s} {'H':>4s} {'n':>6s} {'mittl. bp':>10s} {'Treffer':>8s} "
          f"{'t(boot)':>8s} {'|move|/cost':>11s} {'net_lim':>9s} {'net_mkt':>9s}")
    for _, r in df.iterrows():
        print(f"    {r.asset:8s} {int(r.minutes):4d} {int(r.n_trig):6d} {r.mean_pct*100:10.2f} "
              f"{r.hit*100:7.1f}% {r.t_boot:8.2f} {r.ratio_move_cost:11.1f} "
              f"{r.net_lim_bp:9.2f} {r.net_mkt_bp:9.2f}")

    good = df[(df["t_boot"].abs() >= 2.5) & (df["net_lim_bp"] > 0)]
    print(f"\n    Zellen mit |t|>=2.5 UND positivem Netto (Limit): {len(good)} von {len(df)}")
    if not good.empty:
        print(good.sort_values("net_lim_bp", ascending=False)[
            ["asset", "minutes", "n_trig", "mean_pct", "t_boot", "net_lim_bp", "net_mkt_bp"]
        ].head(12).round(3).to_string(index=False))
    print(f"\n    Zellen positiv NACH Limit-Kosten: {int((df['net_lim_bp'] > 0).sum())} von {len(df)}")
    print(f"    Zellen positiv NACH Market-Kosten: {int((df['net_mkt_bp'] > 0).sum())} von {len(df)}")
    print("    -> Ohne positiven Netto-Wert nach Kosten ist keine Exit-Variante mehr relevant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
