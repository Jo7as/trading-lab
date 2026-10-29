#!/usr/bin/env python3
"""S-007 trend following: data, reference check and the screening funnel (portfolio level).

Spec: ``docs/specs/tsmom_multi_asset.yaml``. The PORTFOLIO is the unit (agreed
2026-10-29): S1/S2 on the position episodes (all markets pooled), S3 on the years of
the daily portfolio returns, S4 on the daily returns (t, Deflated Sharpe with the 4
pre-registered variants) and the placebo (same timing, random direction). Thresholds
are the unchanged ones from ``trading_lab/screening.py``.

  .venv/bin/python studies/tsmom_check.py --fetch          # once: long daily data + costs
  .venv/bin/python studies/tsmom_check.py --referenz       # step 5: reference + data check
  .venv/bin/python studies/tsmom_check.py --trichter       # S1-S4 on train (2004-2024)
  .venv/bin/python studies/tsmom_check.py --trichter --holdout 2026-10-29   # S5, once
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_lab import costs, long_daily, rates, screening  # noqa: E402
from trading_lab.registry import HOLDOUT_START, load_cached  # noqa: E402
from trading_lab.strategies import tsmom_multi_asset as M  # noqa: E402

TRAIN_START = "2004-01-01"
OUT = Path("artifacts/tsmom")
COSTS = OUT / "costs.json"
AQR = Path("artifacts/cache/aqr/tsmom.xlsx")
AQR_URL = "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Time-Series-Momentum-Factors-Monthly.xlsx"
GRID = {"signal": ("12m", "blend_1_3_12"), "stop": ("keiner", "stop_3atr")}
PLACEBO_RUNS = 100
CCYS = ("USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD")


# ── data ───────────────────────────────────────────────────────────────────

def measure_cost(asset: str) -> float:
    """One-way cost in bp: median half spread at 15:00 New York (2019-2024, ftmo profile)
    + 0.5 bp slippage + ftmo commission."""
    m1 = load_cached(asset)
    m1 = costs.adjust_frame(m1[m1.index < HOLDOUT_START], asset, "ftmo")
    ny = m1.index.tz_convert("America/New_York")
    at = m1[(ny.hour == 15) & (ny.minute == 0)]
    mid = (at["close"] + at["ask_close"]) / 2
    half = float(np.nanmedian((at["ask_close"] - at["close"]) / 2 / mid * 1e4))
    px = float(np.nanmedian(mid))
    return half + 0.5 + costs.commission_bp_side(asset, px, "ftmo")


def fetch() -> None:
    for a in long_daily.TICKERS:
        d = long_daily.download(a)
        print(f"  {a:7s} {d.index[0].date()} .. {d.index[-1].date()}  {len(d)} Tage", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cost = {}
    for a in M.GROUP_OF:
        try:
            cost[a] = round(measure_cost(a), 3)
        except Exception as e:  # noqa: BLE001
            print(f"  Kosten {a}: {e}")
    COSTS.write_text(json.dumps(cost, indent=1))
    print("Kosten je Seite (bp):", cost)
    if not AQR.exists():
        import urllib.request
        AQR.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(AQR_URL, headers={"User-Agent": "Mozilla/5.0"})
        AQR.write_bytes(urllib.request.urlopen(req, timeout=60).read())


def load_all(end: pd.Timestamp | None) -> tuple[M.Prepared, dict]:
    frames, cleaned = {}, {}
    for a in long_daily.TICKERS:
        d, bad = long_daily.clean_bad_ticks(long_daily.load_daily(a))
        frames[a] = d
        if bad:
            cleaned[a] = [str(b.date()) for b in bad]
    cost = json.loads(COSTS.read_text())
    r = rates.load_rates(CCYS).ffill()
    p = M.prepare(frames, r, cost, start="2003-01-01", end=str(end.date()) if end is not None else None)
    return p, cleaned


def aqr_monthly() -> pd.Series:
    z = zipfile.ZipFile(AQR)
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    out = {}
    for row in root.iter(f"{{{ns['a']}}}row"):
        cells = {c.get("r").rstrip("0123456789"): c.find("a:v", ns) for c in row.findall("a:c", ns)}
        a, b = cells.get("A"), cells.get("B")
        if a is None or b is None or a.text is None or b.text is None:
            continue
        try:
            day = pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(a.text))
            out[day.to_period("M")] = float(b.text)
        except ValueError:
            continue
    return pd.Series(out).sort_index()


# ── evaluation ─────────────────────────────────────────────────────────────

def _cfg(params: dict) -> M.TsmomConfig:
    return replace(M.TsmomConfig(), **params)


def evaluate(p: M.Prepared, params: dict, start: str) -> dict:
    res = M.run_backtest(p, _cfg(params))
    unit = res["unit"]
    d = res["daily"].loc[start:]
    tr = [t for t in res["trades"] if t["entry_time"] >= pd.Timestamp(start)]
    net_r = d["net"] / unit
    yrs = d["net"].groupby(d.index.year).sum()
    return {"params": params, "res": res, "daily_r": net_r, "trades": tr, "unit": unit,
            "n": len(tr), "gross_mean": float(np.mean([t["r_gross"] for t in tr])) if tr else np.nan,
            "net_mean": float(np.mean([t["r"] for t in tr])) if tr else np.nan,
            "pos_years": int((yrs > 0).sum()), "n_years": int(len(yrs)),
            "per_year_pct": {int(k): round(100 * v, 2) for k, v in yrs.items()},
            "sharpe_net": float(d["net"].mean() / d["net"].std() * np.sqrt(252)),
            "sharpe_gross": float(d["gross"].mean() / d["gross"].std() * np.sqrt(252)),
            "t_daily": float(screening.t_stat(net_r))}


def _placebo_one(args) -> float:
    p, params, start, seed = args
    sig = M.signals(p.close, params["signal"])
    res = M.run_backtest(p, _cfg(params), sign_flip=M.placebo_flips(sig, seed))
    tr = [t for t in res["trades"] if t["entry_time"] >= pd.Timestamp(start)]
    return float(np.mean([t["r_gross"] for t in tr])) if tr else np.nan


def placebo_gross(p: M.Prepared, params: dict, start: str, runs: int = PLACEBO_RUNS) -> float:
    with ProcessPoolExecutor(4) as ex:
        vals = list(ex.map(_placebo_one, [(p, params, start, s) for s in range(runs)]))
    return float(np.nanmean(vals))


def referenz() -> int:
    p, cleaned = load_all(HOLDOUT_START)
    print("Ersetzte Fehlkurse:", cleaned or "keine")
    ev = evaluate(p, {"signal": "12m", "stop": "keiner"}, TRAIN_START)
    d = ev["res"]["daily"].loc[TRAIN_START:]
    print(f"\nBasis 2004-2024: Sharpe vor Kosten {ev['sharpe_gross']:.2f}, netto {ev['sharpe_net']:.2f}, "
          f"Vola {d['net'].std() * np.sqrt(252):.1%}, Kosten {d['cost'].sum():.1%}, Finanzierung {d['fin'].sum():+.1%} "
          f"(über {ev['n_years']} Jahre), {ev['n']} Positionen")
    print("Jahre (netto % des Kontos):", ev["per_year_pct"])
    mon = d["net"].groupby(d.index.to_period("M")).sum()
    aqr = aqr_monthly()
    both = pd.concat([mon, aqr], axis=1, join="inner").dropna()
    print(f"Korrelation mit AQR TSMOM (monatlich, {len(both)} Monate): {both.corr().iloc[0, 1]:.2f}")
    by_grp = pd.DataFrame(ev["trades"]).groupby("group")["r"].agg(["count", "mean", "sum"]).round(3)
    print("\nBeitrag je Anlageklasse (Positionen, Ø R, Summe R):\n" + by_grp.to_string())
    dbl = evaluate(p, {"signal": "12m", "stop": "keiner", "markup_factor": 2.0}, TRAIN_START)
    print(f"\nRobustheit doppelter Swap-Aufschlag: Sharpe netto {dbl['sharpe_net']:.2f}")
    # data source check 2019-2024: yfinance vs Dukascopy CFD closes at 17:00 NY
    print("\nAbgleich Datenquellen 2019-2024 (Korrelation der Tagesrenditen):")
    rows = []
    for i, a in enumerate(p.assets):
        try:
            m1 = load_cached(a if a != "UST10Y" else "UST10Y")
        except Exception:  # noqa: BLE001
            continue
        m1 = m1[m1.index < HOLDOUT_START]
        ny = m1.index.tz_convert("America/New_York")
        c = ((m1["close"] + m1["ask_close"]) / 2).groupby((ny - pd.Timedelta(hours=17)).normalize().tz_localize(None)).last()
        c.index = c.index + pd.Timedelta(days=1)
        yf_c = pd.Series(p.close[:, i], index=p.dates)
        j = pd.concat([c.pct_change(), yf_c.pct_change()], axis=1, join="inner").loc["2019-02-01":].dropna()
        j = j[(j.iloc[:, 0] != 0) & (j.iloc[:, 1] != 0)]
        rows.append((a, len(j), round(float(j.corr().iloc[0, 1]), 3)))
    print("  " + ", ".join(f"{a} {c}" for a, _, c in rows))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "referenz.json").write_text(json.dumps({k: v for k, v in ev.items() if k in (
        "n", "gross_mean", "net_mean", "pos_years", "n_years", "per_year_pct", "sharpe_net", "sharpe_gross", "t_daily")},
        indent=1))
    return 0


def trichter(holdout: str | None) -> int:
    p, _ = load_all(HOLDOUT_START)
    keys = [dict(zip(GRID, v)) for v in __import__("itertools").product(*GRID.values())]
    evs = {tuple(k.values()): evaluate(p, k, TRAIN_START) for k in keys}
    rows = []
    for key, ev in evs.items():
        nbs = [evs[k] for k in evs if sum(a != b for a, b in zip(k, key)) == 1]
        pg = placebo_gross(p, ev["params"], TRAIN_START)
        rows.append({"name": f"tsmom_multi_asset:PORTFOLIO#{len(rows)}", "strategy": "tsmom_multi_asset",
                     "asset": "PORTFOLIO", "params": ev["params"], "n": ev["n"],
                     "gross_mean": ev["gross_mean"], "net_mean": ev["net_mean"],
                     "placebo_gross": pg, "placebo_edge": ev["gross_mean"] - pg,
                     "pos_years": ev["pos_years"], "n_years": ev["n_years"],
                     "neighbour_positive": float(np.mean([x["net_mean"] > 0 for x in nbs])),
                     "t_daily": ev["t_daily"], "n_trials": len(keys),
                     "dsr": screening.deflated_sharpe(ev["daily_r"].to_numpy(), n_trials=len(keys)),
                     "pbo": np.nan, "sharpe_net": ev["sharpe_net"], "per_year_pct": ev["per_year_pct"]})
        print(f"  {ev['params']}: n {ev['n']}, Ø brutto {ev['gross_mean']:+.3f} R, netto {ev['net_mean']:+.3f} R, "
              f"Placebo {pg:+.3f} R, Jahre {ev['pos_years']}/{ev['n_years']}, t {ev['t_daily']:.2f}, "
              f"Sharpe netto {ev['sharpe_net']:.2f}", flush=True)
    if holdout:
        full, _ = load_all(None)
        for row in rows:
            res = M.run_backtest(full, _cfg(row["params"]))
            d = res["daily"].loc[HOLDOUT_START.tz_localize(None):]
            tr = [t for t in res["trades"] if t["exit_time"] >= HOLDOUT_START.tz_localize(None)]
            r = d["net"] / res["unit"]
            row.update({"oos_days": len(d), "oos_net_pct": float(100 * d["net"].sum()),
                        "oos_sharpe": float(d["net"].mean() / d["net"].std() * np.sqrt(252)),
                        "oos_t_daily": float(screening.t_stat(r)), "oos_n": len(tr),
                        "oos_pass": bool(d["net"].sum() > 0)})
    f = screening.run_funnel(pd.DataFrame(rows), up_to="S5" if holdout else "S4", holdout_freigabe=holdout)
    print("\n" + f.summary())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("funnel_holdout.json" if holdout else "funnel.json")).write_text(json.dumps(rows, indent=1, default=str))
    return 0


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--referenz", action="store_true")
    ap.add_argument("--trichter", action="store_true")
    ap.add_argument("--holdout", metavar="FREIGABE")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    if a.referenz:
        referenz()
    if a.trichter:
        trichter(a.holdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
