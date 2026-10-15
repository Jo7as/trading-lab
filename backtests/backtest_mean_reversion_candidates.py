#!/usr/bin/env python3
"""Comparison report for the mean-reversion candidates (Dukascopy bid+ask).

Per candidate (A Session-Range-Fade Gold, B z-score MR Gold, B z-score MR USD/JPY):
  - seed config on the full period and on the last 30% (fixed params)
  - walk-forward: evolve on a rolling train window, trade the next window
  - ±20% parameter sensitivity of the evolved config
  - Monte-Carlo bootstrap of the walk-forward trade sequence
  - correlation and combined equity with the Supertrend baseline bot
All numbers are net of historical spread and slippage unless marked gross.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from trading_lab.backtest import _compute_metrics
from trading_lab.mean_reversion_evolution_worker import MeanReversionEvolutionWorker
from trading_lab.mr_common import ASK_COLS, resample_ohlc
from trading_lab.mr_evolution_base import Frames, MrEvolutionBase, frames_span, slice_frames
from trading_lab.session_range_evolution_worker import SessionRangeEvolutionWorker
from trading_lab.strategies.supertrend_volume import (
    SupertrendVolumeConfig,
    run_backtest as st_run_backtest,
)

ROOT = Path(__file__).resolve().parent

GO_CRITERIA = {
    "oos_pnl_pct": 0.0,
    "oos_sharpe": 0.5,
    "oos_dd_pct": -10.0,
    "oos_tpw": 2.0,
}

CANDIDATES: list[dict[str, Any]] = [
    {
        "key": "A_gold",
        "label": "A Session-Range-Fade (Gold)",
        "worker": SessionRangeEvolutionWorker,
        "template": "session-range-state-template",
        "asset": None,
        "st_baseline": "hf_baseline.yaml",
        "old_asset": "GC=F",
        "core_params": ["range_min_atr_mult", "range_max_atr_mult", "sl_buffer_atr_mult", "trend_day_atr_mult"],
    },
    {
        "key": "B_gold",
        "label": "B Z-Score MR (Gold)",
        "worker": MeanReversionEvolutionWorker,
        "template": "mean-reversion-state-template",
        "asset": None,
        "st_baseline": "hf_baseline.yaml",
        "old_asset": "GC=F",
        "core_params": ["z_entry_tier_a", "z_entry_tier_b", "adx_block_threshold", "sl_atr_mult", "max_hold_bars"],
    },
    {
        "key": "B_jpy",
        "label": "B Z-Score MR (USD/JPY)",
        "worker": MeanReversionEvolutionWorker,
        "template": "mean-reversion-state-template",
        "asset": "USD/JPY",
        "st_baseline": "jpy_baseline.yaml",
        "old_asset": "USD/JPY",
        "core_params": ["z_entry_tier_a", "z_entry_tier_b", "adx_block_threshold", "sl_atr_mult", "max_hold_bars"],
    },
]


# ── Metrics helpers ─────────────────────────────────────────────────


def _closed(trades: list[dict]) -> list[dict]:
    return [t for t in trades if t.get("status") == "closed"]


def summarize(trades: list[dict], days: float) -> dict[str, Any]:
    closed = _closed(trades)
    m = _compute_metrics(closed, [], None)  # type: ignore[arg-type]
    r_mult = [
        t["pnl_pct"] / t["meta"]["risk_pct"]
        for t in closed
        if (t.get("meta") or {}).get("risk_pct")
    ]
    return {
        "trades": len(closed),
        "tpw": round(len(closed) / max(days / 7.0, 1e-9), 2),
        "pnl_pct": round(float(m["total_pnl_pct"]), 2),
        "pnl_pct_per_30d": round(float(m["total_pnl_pct"]) / max(days / 30.0, 1e-9), 2),
        "sharpe": round(float(m["sharpe"]), 2),
        "dd_pct": round(float(m["max_drawdown_pct"]), 2),
        "win_rate": m["win_rate"],
        "pf": round(float(m["profit_factor"]), 2) if np.isfinite(m["profit_factor"]) else None,
        "avg_r": round(float(np.mean(r_mult)), 3) if r_mult else None,
    }


def go_check(s: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "oos_pnl_positive": bool(s["pnl_pct"] > GO_CRITERIA["oos_pnl_pct"]),
        "oos_sharpe_ge_0.5": bool(s["sharpe"] >= GO_CRITERIA["oos_sharpe"]),
        "oos_dd_le_10pct": bool(s["dd_pct"] >= GO_CRITERIA["oos_dd_pct"]),
        "oos_tpw_ge_2": bool(s["tpw"] >= GO_CRITERIA["oos_tpw"]),
    }
    return {"go": all(checks.values()), "go_checks": checks}


def daily_pnl(trades: list[dict], start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Account % PnL booked per business day (by exit time), zero-filled."""
    days = pd.bdate_range(start.normalize(), end.normalize(), tz="UTC")
    out = pd.Series(0.0, index=days)
    for t in _closed(trades):
        # Sunday-evening exits count towards Monday
        day = pd.Timestamp(t["exit_time"], unit="s", tz="UTC").normalize() + pd.offsets.BDay(0)
        if day in out.index:
            out[day] += float(t["pnl_pct"] or 0.0)
    return out


def daily_stats(series: pd.Series) -> dict[str, float]:
    cum = series.cumsum()
    dd = float((cum - cum.cummax().clip(lower=0.0)).min()) if len(cum) else 0.0
    std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    sharpe = float(series.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    return {"pnl_pct": round(float(series.sum()), 2), "dd_pct": round(dd, 2), "daily_sharpe": round(sharpe, 2)}


# ── Study steps ─────────────────────────────────────────────────────


def run_window(
    worker: MrEvolutionBase,
    frames: Frames,
    cfg: Any,
    start: pd.Timestamp,
    end: pd.Timestamp,
    warmup_days: int = 30,
) -> list[dict]:
    """Closed trades entered in [start, end); earlier bars only warm up indicators."""
    window = slice_frames(frames, start - pd.Timedelta(days=warmup_days), end)
    trades = _closed(worker.run(window, cfg)["trades"])
    return [t for t in trades if t["entry_time"] >= start.timestamp()]


def walk_forward(
    worker: MrEvolutionBase,
    frames: Frames,
    seed_cfg: Any,
    *,
    train_months: int,
    test_months: int,
    rounds: int,
) -> dict[str, Any]:
    """Rolling evolve → trade-next-window. The config carries over between folds."""
    first, last = frames_span(frames)
    cfg = seed_cfg
    folds: list[dict[str, Any]] = []
    trades: list[dict] = []
    train_start = first.normalize()
    while True:
        test_start = train_start + pd.DateOffset(months=train_months)
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_start >= last:
            break
        test_end = min(test_end, last + pd.Timedelta(seconds=1))
        cfg = worker.evolve(cfg, slice_frames(frames, train_start, test_start), rounds)
        fold_trades = run_window(worker, frames, cfg, test_start, test_end)
        days = (test_end - test_start).total_seconds() / 86400.0
        s = summarize(fold_trades, days)
        folds.append(
            {
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "trades": s["trades"],
                "pnl_pct": s["pnl_pct"],
                "avg_r": s["avg_r"],
                "params": {k: getattr(cfg, k) for k in worker_params(worker, cfg)},
            }
        )
        trades.extend(fold_trades)
        train_start = train_start + pd.DateOffset(months=test_months)

    span_start = pd.Timestamp(folds[0]["test_start"], tz="UTC")
    span_days = (last - span_start).total_seconds() / 86400.0
    summary = summarize(trades, span_days)
    return {
        "train_months": train_months,
        "test_months": test_months,
        "rounds_per_fold": rounds,
        "span_start": span_start,
        "span_end": last,
        "folds": folds,
        "folds_positive": sum(1 for f in folds if f["pnl_pct"] > 0),
        "summary": summary,
        **go_check(summary),
        "trades": trades,
    }


def worker_params(worker: MrEvolutionBase, cfg: Any) -> list[str]:
    names = [name for name, *_ in worker.config_cls.search_space()]
    for extra in ("timeframe", "require_reversal_candle", "session_filter_enabled"):
        if hasattr(cfg, extra) and extra not in names:
            names.append(extra)
    return names


def sensitivity(
    worker: MrEvolutionBase, frames: Frames, cfg: Any, params: list[str], days: float
) -> dict[str, Any]:
    """PnL of the config with each core parameter moved ±20% (one at a time)."""
    base = summarize(worker.run(frames, cfg)["trades"], days)
    rows: list[dict[str, Any]] = []
    for name in params:
        old = getattr(cfg, name)
        for factor in (0.8, 1.2):
            new: Any = old * factor
            new = max(1, int(round(new))) if isinstance(old, int) else round(float(new), 4)
            cand = dataclasses.replace(cfg, **{name: new})
            if new == old or not worker._valid(cand):
                continue
            s = summarize(worker.run(frames, cand)["trades"], days)
            rows.append(
                {"param": name, "from": old, "to": new, "pnl_pct": s["pnl_pct"], "avg_r": s["avg_r"], "trades": s["trades"]}
            )
    pnls = [r["pnl_pct"] for r in rows]
    return {
        "base_pnl_pct": base["pnl_pct"],
        "base_avg_r": base["avg_r"],
        "worst_pnl_pct": min(pnls) if pnls else None,
        "best_pnl_pct": max(pnls) if pnls else None,
        "variants_positive": sum(1 for p in pnls if p > 0),
        "variants": len(pnls),
        "rows": rows,
    }


def monte_carlo(trades: list[dict], n: int = 2000, seed: int = 7) -> dict[str, Any]:
    """Bootstrap the trade sequence: distribution of total PnL and max drawdown."""
    pnls = np.array([float(t["pnl_pct"] or 0.0) for t in _closed(trades)])
    if len(pnls) < 10:
        return {"samples": 0}
    rng = np.random.default_rng(seed)
    draws = rng.choice(pnls, size=(n, len(pnls)), replace=True)
    cum = draws.cumsum(axis=1)
    peak = np.maximum.accumulate(np.maximum(cum, 0.0), axis=1)
    max_dd = (cum - peak).min(axis=1)
    total = cum[:, -1]
    return {
        "samples": n,
        "pnl_p05": round(float(np.quantile(total, 0.05)), 2),
        "pnl_median": round(float(np.median(total)), 2),
        "pnl_p95": round(float(np.quantile(total, 0.95)), 2),
        "prob_profit": round(float((total > 0).mean()), 3),
        "dd_median": round(float(np.median(max_dd)), 2),
        "dd_p95": round(float(np.quantile(max_dd, 0.05)), 2),
        "prob_dd_worse_10pct": round(float((max_dd <= -10.0).mean()), 3),
    }


def supertrend_trades(frames: Frames, baseline_yaml: str, asset: str) -> list[dict]:
    """The existing Supertrend bot on 1h bid bars of the same data."""
    base = next(iter(frames.values()))
    h1 = resample_ohlc(base.drop(columns=[c for c in ASK_COLS if c in base.columns]), "1h")
    raw = yaml.safe_load((ROOT / baseline_yaml).read_text()) or {}
    valid = {f.name for f in dataclasses.fields(SupertrendVolumeConfig)}
    cfg = SupertrendVolumeConfig(**{k: v for k, v in raw.items() if k in valid})
    cfg = dataclasses.replace(cfg, asset=asset)
    return _closed(st_run_backtest(h1, cfg, verbose=False)["trades"])


def correlation(
    cand_trades: list[dict], st_trades: list[dict], start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, Any]:
    """Daily-PnL correlation and the combined equity curve (candidate + Supertrend)."""
    st_in = [t for t in st_trades if start.timestamp() <= t["entry_time"] < end.timestamp()]
    d_cand = daily_pnl(cand_trades, start, end)
    d_st = daily_pnl(st_in, start, end)
    corr = float(d_cand.corr(d_st)) if d_cand.std() > 0 and d_st.std() > 0 else 0.0
    return {
        "daily_pnl_corr": round(corr, 3),
        "supertrend_alone": daily_stats(d_st),
        "candidate_alone": daily_stats(d_cand),
        "combined": daily_stats(d_cand + d_st),
        "supertrend_trades": len(st_in),
    }


def gross_frames(frames: Frames) -> Frames:
    return {tf: df.drop(columns=[c for c in ASK_COLS if c in df.columns]) for tf, df in frames.items()}


async def study(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    worker: MrEvolutionBase = spec["worker"](
        state_dir=ROOT / spec["template"],
        year_start=int(args.start[:4]),
        oos_split=args.oos_split,
        num_candidates=args.candidates,
        forward_eval_days=args.forward_days,
        explore_every=8,
        asset=spec["asset"],
        verbose=False,
    )
    print(f"\n=== {spec['label']} ===")
    frames = await worker.load_frames()
    if not frames:
        return {"key": spec["key"], "label": spec["label"], "error": "no data"}
    first, last = frames_span(frames)
    asset, _ = worker._goal_asset()
    seed_cfg = worker._pin(worker._load_config(ROOT / spec["template"] / worker.BASELINE_FILE))
    train, oos, split_ts = worker.split(frames)
    full_days = (last - first).total_seconds() / 86400.0
    oos_days = (last - split_ts).total_seconds() / 86400.0

    seed_full = summarize(worker.run(frames, seed_cfg)["trades"], full_days)
    seed_oos = summarize(run_window(worker, frames, seed_cfg, split_ts, last), oos_days)
    zero_cost = dataclasses.replace(seed_cfg, fallback_spread_pct=0.0, slippage_pct=0.0)
    seed_gross = summarize(worker.run(gross_frames(frames), zero_cost)["trades"], full_days)
    print(
        f"  seed full: {seed_full['trades']} trades, {seed_full['pnl_pct']:+.1f}% "
        f"(avg {seed_full['avg_r']}R net, {seed_gross['avg_r']}R gross)"
    )

    random.seed(args.seed)
    wf = walk_forward(
        worker, frames, seed_cfg,
        train_months=args.train_months, test_months=args.test_months, rounds=args.rounds,
    )
    print(
        f"  walk-forward OOS: {wf['summary']['trades']} trades, {wf['summary']['pnl_pct']:+.1f}%, "
        f"{wf['folds_positive']}/{len(wf['folds'])} folds positive, GO={'YES' if wf['go'] else 'NO'}"
    )

    random.seed(args.seed)
    evolved = worker.evolve(seed_cfg, train, args.rounds * 2)
    evolved_oos = summarize(run_window(worker, frames, evolved, split_ts, last), oos_days)
    sens = sensitivity(worker, slice_frames(frames, split_ts, None), evolved, spec["core_params"], oos_days)

    st_trades = supertrend_trades(frames, spec["st_baseline"], asset)
    corr = correlation(wf["trades"], st_trades, wf["span_start"], wf["span_end"])
    print(
        f"  vs Supertrend: corr {corr['daily_pnl_corr']:+.2f} | ST alone "
        f"{corr['supertrend_alone']['pnl_pct']:+.1f}% DD {corr['supertrend_alone']['dd_pct']:.1f}% | "
        f"combined {corr['combined']['pnl_pct']:+.1f}% DD {corr['combined']['dd_pct']:.1f}%"
    )

    mc = monte_carlo(wf["trades"])
    wf_out = {k: v for k, v in wf.items() if k != "trades"}
    wf_out["span_start"] = wf["span_start"].isoformat()
    wf_out["span_end"] = wf["span_end"].isoformat()
    return {
        "key": spec["key"],
        "label": spec["label"],
        "asset": asset,
        "old_asset": spec["old_asset"],
        "data": {"start": first.isoformat(), "end": last.isoformat(), "oos_split_ts": split_ts.isoformat()},
        "seed_config": dataclasses.asdict(seed_cfg),
        "seed_full": seed_full,
        "seed_full_gross": seed_gross,
        "seed_oos_30pct": {**seed_oos, **go_check(seed_oos)},
        "walk_forward": wf_out,
        "evolved_on_train_70pct": {
            "config": {k: getattr(evolved, k) for k in worker_params(worker, evolved)},
            "oos_30pct": {**evolved_oos, **go_check(evolved_oos)},
        },
        "sensitivity_pm20_on_oos": sens,
        "monte_carlo_walk_forward": mc,
        "vs_supertrend": corr,
    }


# ── Output ──────────────────────────────────────────────────────────


def _old_baselines() -> dict[str, dict[str, Any]]:
    path = ROOT / "artifacts" / "regime_forex_backtest.json"
    if not path.exists():
        return {}
    return {a["asset"]: a for a in json.loads(path.read_text()).get("assets", [])}


def print_table(rows: list[dict[str, Any]]) -> None:
    old = _old_baselines()
    print("\n" + "=" * 100)
    print("COMPARISON — walk-forward OOS, net of spread + slippage")
    print("=" * 100)
    head = f"{'':28s}" + "".join(f"{r['label'][:23]:>24s}" for r in rows)
    print(head)

    def line(name: str, fn: Any) -> None:
        print(f"{name:28s}" + "".join(f"{fn(r):>24s}" for r in rows))

    wf = lambda r: r["walk_forward"]["summary"]  # noqa: E731
    line("OOS trades/week", lambda r: f"{wf(r)['tpw']:.2f}")
    line("OOS PnL %", lambda r: f"{wf(r)['pnl_pct']:+.2f}")
    line("OOS PnL % / 30d", lambda r: f"{wf(r)['pnl_pct_per_30d']:+.2f}")
    line("OOS Sharpe", lambda r: f"{wf(r)['sharpe']:.2f}")
    line("OOS Max DD %", lambda r: f"{wf(r)['dd_pct']:.2f}")
    line("OOS avg R/trade", lambda r: f"{wf(r)['avg_r']}")
    line("Folds positive", lambda r: f"{r['walk_forward']['folds_positive']}/{len(r['walk_forward']['folds'])}")
    line("Seed avg R net / gross", lambda r: f"{r['seed_full']['avg_r']} / {r['seed_full_gross']['avg_r']}")
    line("±20% variants positive", lambda r: f"{r['sensitivity_pm20_on_oos']['variants_positive']}/{r['sensitivity_pm20_on_oos']['variants']}")
    line("MC P(profit)", lambda r: f"{r['monte_carlo_walk_forward'].get('prob_profit')}")
    line("MC P(DD worse than 10%)", lambda r: f"{r['monte_carlo_walk_forward'].get('prob_dd_worse_10pct')}")
    line("Corr vs Supertrend (daily)", lambda r: f"{r['vs_supertrend']['daily_pnl_corr']:+.2f}")
    line("Supertrend alone PnL/DD %", lambda r: f"{r['vs_supertrend']['supertrend_alone']['pnl_pct']:+.1f} / {r['vs_supertrend']['supertrend_alone']['dd_pct']:.1f}")
    line("Combined PnL/DD %", lambda r: f"{r['vs_supertrend']['combined']['pnl_pct']:+.1f} / {r['vs_supertrend']['combined']['dd_pct']:.1f}")
    line("Go?", lambda r: "YES" if r["walk_forward"]["go"] else "NO")
    if old:
        print("-" * 100)
        print("Baselines (artifacts/regime_forex_backtest.json, 1h, simple 30% OOS):")

        def old_val(r: dict[str, Any], key: str, fmt: str) -> str:
            a = old.get(r["old_asset"])
            return format(a[key], fmt) if a and a.get(key) is not None else "—"

        line("regime_hybrid OOS tpw", lambda r: old_val(r, "oos_tpw", ".2f"))
        line("regime_hybrid OOS PnL %", lambda r: old_val(r, "oos_pnl_pct", "+.2f"))
        line("regime_hybrid OOS Sharpe", lambda r: old_val(r, "oos_sharpe", ".2f"))
        line("regime_hybrid OOS DD %", lambda r: old_val(r, "oos_dd_pct", ".2f"))
        line("pure Supertrend OOS PnL %", lambda r: old_val(r, "st_baseline_oos_pnl_pct", "+.2f"))
    print("=" * 100)


async def main_async(args: argparse.Namespace) -> None:
    selected = [c for c in CANDIDATES if not args.only or c["key"] in args.only.split(",")]
    rows = [await study(spec, args) for spec in selected]
    rows = [r for r in rows if "error" not in r]
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "dukascopy bid+ask",
        "start": args.start,
        "go_criteria": GO_CRITERIA,
        "settings": {
            "train_months": args.train_months, "test_months": args.test_months,
            "rounds_per_fold": args.rounds, "candidates_per_round": args.candidates,
            "forward_eval_days": args.forward_days, "seed": args.seed,
        },
        "candidates": rows,
        "summary": {"all_go": [r["key"] for r in rows if r["walk_forward"]["go"]]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, default=str))
    print_table(rows)
    print(f"\nSaved: {args.output}")


def main() -> None:
    logging.getLogger("DUKASCRIPT").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Mean-reversion candidates comparison report")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--only", default="", help="comma list of A_gold,B_gold,B_jpy")
    parser.add_argument("--oos-split", type=float, default=0.30)
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=30, help="evolution rounds per fold")
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--forward-days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/mean_reversion_candidates.json")
    )
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
