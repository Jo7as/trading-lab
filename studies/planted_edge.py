#!/usr/bin/env python3
"""Paket 1: does the whole pipeline find an edge we planted in real prices - and only then?

``studies/pipeline_power.py`` feeds synthetic trade lists into the last gate. This one starts at the
1-minute bid/ask prices: a known drift is written into real paths, a neutral test rig trades it, and the
normal path judges the result (``optimizer.optimise`` with costs, grid neighbours, replay placebo, Deflated
Sharpe, PBO, then ``screening.run_funnel``). No strategy, no registry entry, TRAIN only (2019-2024).

Rig: one trade per weekday at 14:00 UTC (+ delay), random direction from the seed, time exit, market fills on
bid/ask + 0.5 bp. Planted: during the first ``HOLD_MIN`` minutes of each trade the whole price (mid, bid, ask)
drifts by ``d x mu`` in the trade's direction and STAYS shifted. ``mu`` is set so that the trade has the wanted
annualised Sharpe BEFORE costs; Sharpe 0 is the negative control.

Reported per Sharpe level: share of seeds in which at least one of the 9 variants survives each stage
(hit rate; for Sharpe 0 the false-alarm rate), plus the realised net Sharpe of the best row.

  .venv/bin/python studies/planted_edge.py [--seeds 40] [--asset SP500]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trading_lab import screening  # noqa: E402
from trading_lab.optimizer import Spec, optimise  # noqa: E402
from trading_lab.registry import HOLDOUT_START, load_cached  # noqa: E402

HOLD_MIN = 240
ENTRY_UTC = 14
TRADES_PER_YEAR = 252
LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
GRID = {"hold_min": (120, 240, 360), "delay_min": (0, 5, 10)}
OUT = Path("artifacts/planted_edge.json")
PRICE_COLS = ("open", "high", "low", "close", "ask_open", "ask_high", "ask_low", "ask_close")


@dataclass(frozen=True)
class RigConfig:
    hold_min: int = 240
    delay_min: int = 0
    slippage_bp: float = 0.5


def entry_rows(index: pd.DatetimeIndex, hold: int = 360 + 10) -> np.ndarray:
    """First minute at/after 14:00 UTC per weekday, with a full holding window of real minutes after it."""
    t = index.as_unit("ns").asi8
    days = pd.DatetimeIndex(pd.unique(index.normalize()))
    days = days[days.dayofweek < 5]
    want = (days + pd.Timedelta(hours=ENTRY_UTC)).as_unit("ns").asi8
    k = np.searchsorted(t, want)
    ok = (k + hold < len(t))
    k, want = k[ok], want[ok]
    near = (t[k] - want) < pd.Timedelta(minutes=5).value
    k = k[near]
    span = t[k + hold] - t[k]
    return k[span <= pd.Timedelta(minutes=hold + 30).value]


def plant(m1: pd.DataFrame, rows: np.ndarray, dirs: np.ndarray, mu: float, hold: int = HOLD_MIN) -> pd.DataFrame:
    """Write the edge into the prices: a linear log drift ``d x mu`` over ``hold`` minutes after each
    entry row; the shift persists, so nothing reverts and bid, ask and mid move together."""
    out = m1.copy()
    step = np.zeros(len(m1) + 1)
    for k, d in zip(rows, dirs):
        step[k + 1:k + 1 + hold] += d * mu / hold
    f = np.exp(np.cumsum(step[1:]))                      # factor at the END of each minute
    prev = np.r_[1.0, f[:-1]]                            # factor at the START of each minute
    for side in ("", "ask_"):
        out[f"{side}open"] = m1[f"{side}open"].to_numpy() * prev
        out[f"{side}close"] = m1[f"{side}close"].to_numpy() * f
        out[f"{side}high"] = m1[f"{side}high"].to_numpy() * np.maximum(prev, f)
        out[f"{side}low"] = m1[f"{side}low"].to_numpy() * np.minimum(prev, f)
    return out


def trade_sigma(m1: pd.DataFrame, rows: np.ndarray, hold: int = HOLD_MIN) -> float:
    mid = ((m1["close"] + m1["ask_close"]) / 2).to_numpy()
    return float(np.std(np.log(mid[rows + hold] / mid[rows])))


def mu_for(sharpe: float, sigma: float) -> float:
    return sharpe * sigma / np.sqrt(TRADES_PER_YEAR)


def run_rig(m1: dict, cfg: RigConfig) -> list[dict]:
    """Time-exit engine on bid/ask; 1 R = one trade sigma of the price."""
    t, bo, ao = m1["t"], m1["bid_open"], m1["ask_open"]
    slip = cfg.slippage_bp * 1e-4
    out = []
    for k0, d in zip(m1["rows"], m1["dirs"]):
        k, x = int(k0) + cfg.delay_min, int(k0) + cfg.delay_min + cfg.hold_min
        if x >= len(t):
            continue
        entry = (ao[k] if d > 0 else bo[k]) * (1 + d * slip)
        exit_ = (bo[x] if d > 0 else ao[x]) * (1 - d * slip)
        risk = entry * m1["sigma"]
        out.append({"entry_time": pd.Timestamp(t[k], tz="UTC"), "exit_time": pd.Timestamp(t[x], tz="UTC"),
                    "direction": int(d), "entry": entry, "stop": entry - d * risk, "r": d * (exit_ - entry) / risk})
    return out


def rig_spec(asset: str, rows: np.ndarray, dirs: np.ndarray, sigma: float) -> Spec:
    def prepare(m1: pd.DataFrame) -> dict:
        ns = m1.index.as_unit("ns").asi8
        use = rows < len(ns)
        return {"t": ns, "bid_open": m1["open"].to_numpy(), "ask_open": m1["ask_open"].to_numpy(),
                "rows": rows[use], "dirs": dirs[use], "sigma": sigma}
    return Spec("planted_rig", asset, RigConfig(), GRID, prepare=prepare, run=lambda c, d: run_rig(d, c))


def judge(m1: pd.DataFrame, asset: str, sharpe: float, seed: int, *, budget: int = 9) -> dict:
    """One seed end to end: plant, optimise, funnel. Returns the stages survived by any variant."""
    rows = entry_rows(m1.index)
    dirs = np.random.default_rng(seed).choice([-1, 1], len(rows))
    sigma = trade_sigma(m1, rows)
    planted = plant(m1, rows, dirs, mu_for(sharpe, sigma))
    end = planted.index[-1] + pd.Timedelta(minutes=1)
    res = optimise(rig_spec(asset, rows, dirs, sigma), planted, holdout_start=end, budget=budget, verbose=False)
    df = pd.DataFrame(res.rows)
    f = screening.run_funnel(df, up_to="S4")
    stages = {s.name.split()[0]: int(len(s.kept)) for s in f.stages}
    best = df.sort_values("net_mean", ascending=False).iloc[0]
    s3 = next((s.kept for s in f.stages if s.name.startswith("S3")), df.iloc[0:0])
    parts = {"s3_dsr": bool((s3["dsr"] >= screening.S4_MIN_DSR).any()) if len(s3) else False,
             "s3_placebo": bool((s3["placebo_edge"] > 0).any()) if len(s3) else False,
             "s3_pbo": bool(res.pbo <= screening.S4_MAX_PBO) if len(s3) else False}
    years = (m1.index[-1] - m1.index[0]).days / 365.25
    return {"sharpe": sharpe, "seed": seed, **{k: v > 0 for k, v in stages.items()},
            "net_mean": float(best["net_mean"]), "gross_mean": float(best["gross_mean"]),
            "net_sharpe": float(best["t_daily"]) / np.sqrt(years), "dsr": float(best["dsr"]),
            "placebo_edge": float(best["placebo_edge"]), "pbo": float(res.pbo), "n_trials": int(res.n_trials), **parts}


_M1 = None


def _init(asset: str) -> None:
    global _M1
    warnings.filterwarnings("ignore")
    m1 = load_cached(asset)
    _M1 = m1[m1.index < HOLDOUT_START]


def _one(args) -> dict:
    asset, s, seed = args
    return judge(_M1, asset, s, seed)


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--asset", default="SP500")
    a = ap.parse_args()
    jobs = [(a.asset, s, seed) for s in LEVELS for seed in range(a.seeds)]
    with ProcessPoolExecutor(4, initializer=_init, initargs=(a.asset,)) as ex:
        rows = list(ex.map(_one, jobs))
    OUT.write_text(json.dumps(rows, indent=1))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    g = df.groupby("sharpe").agg(S1=("S1", "mean"), S2=("S2", "mean"), S3=("S3", "mean"), S4=("S4", "mean"),
                                 netto_sharpe=("net_sharpe", "mean"), dsr=("dsr", "mean"), edge=("placebo_edge", "mean"),
                                 pbo=("pbo", "mean"), versuche=("n_trials", "mean"), s3_dsr=("s3_dsr", "mean"),
                                 s3_placebo=("s3_placebo", "mean"), s3_pbo=("s3_pbo", "mean"))
    print(f"Eingebaute Kante, {a.asset}, {a.seeds} Seeds je Stärke, Training 2019-2024")
    print(g.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
