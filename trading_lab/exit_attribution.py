"""Exit attribution - which exit actually produces the P&L?

A fixed R target is often decorative. ORB-2R on US100 closes 686 of 1505 trades
at the session end ("flat"), 624 on the stop and only 195 (13 %) on the 2R
target - so the "2R strategy" is really a time-exit model whose target rarely
fires. Optimising the reward/risk of such a strategy is tuning a parameter that
barely participates.

This module answers three questions from a trade log alone:

* **Who earns the R?** - group the trades by ``exit_reason`` and show each
  group's count, mean R and share of the total.
* **Does the target ever fire?** - the target share, and how far the trades that
  did *not* reach it actually got (``progress``, the fraction of the entry→target
  distance that was covered). A median far below 1 means the target is out of
  reach in the window the strategy allows.
* **Is the horizon consistent with the target?** - the holding time per exit
  group, so a "2R" target that needs six hours inside a four-hour window is
  visible as such.

Everything here is a pure function over a list of trade dicts, so it is testable
without market data. The trade dicts only need a P&L key (``r`` or ``pnl_pct``)
and an ``exit_reason``; ``entry``/``stop``/``target``/``exit``/``direction`` add
the progress analysis.

    from trading_lab.exit_attribution import report
    print(report(res["trades"], "ORB-2R US100"))
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Exits that mean "the target was reached" / "time ran out", across the repo's
# strategies (orb_2r: target|stop|flat, fade/mr: time_exit|time_stop|backtest_end).
TARGET_REASONS = {"target", "tp", "tp1", "tp2", "take_profit"}
TIME_REASONS = {"flat", "time_exit", "time_stop", "backtest_end", "max_hold", "timeout"}


@dataclass(frozen=True)
class ExitRow:
    reason: str
    n: int
    share: float            # of all trades
    mean_r: float
    sum_r: float
    win_rate: float         # fraction with r > 0
    share_of_r: float       # this group's share of the total R


def _frame(trades: list[dict], r_key: str, reason_key: str) -> pd.DataFrame:
    rows = [
        {"reason": str(t.get(reason_key) or "unknown"), "r": float(t[r_key])}
        for t in trades if r_key in t
    ]
    if not rows:
        raise ValueError(f"no trade carries {r_key!r} (keys seen: "
                         f"{sorted({k for t in trades for k in t})[:12]})")
    return pd.DataFrame(rows)


def attribution(trades: list[dict], *, r_key: str = "r",
                reason_key: str = "exit_reason") -> pd.DataFrame:
    """Per-exit-reason count, mean R and contribution to the total."""
    df = _frame(trades, r_key, reason_key)
    total = float(df["r"].sum())
    g = df.groupby("reason")["r"]
    out = pd.DataFrame({
        "n": g.size(),
        "mean_r": g.mean(),
        "sum_r": g.sum(),
        "win_rate": g.apply(lambda s: float((s > 0).mean())),
    })
    out["share"] = out["n"] / len(df)
    out["share_of_r"] = out["sum_r"] / total if total else np.nan
    out["is_target"] = [r in TARGET_REASONS for r in out.index]
    out["is_time"] = [r in TIME_REASONS for r in out.index]
    return out.sort_values("n", ascending=False)


def progress(trades: list[dict], *, reason_key: str = "exit_reason",
             r_key: str = "r") -> pd.DataFrame:
    """How far toward the target each exit got (1.0 = target reached).

    ``progress = (exit - entry) / (target - entry)``. The sign works out for both
    directions without mirroring: a short has ``target < entry``, so a favourable
    move makes both numerator and denominator negative.
    """
    rows = []
    for t in trades:
        need = ("entry", "target", "exit")
        if not all(k in t for k in need):
            continue
        span = float(t["target"]) - float(t["entry"])
        if span == 0:
            continue
        rows.append({"reason": str(t.get(reason_key) or "unknown"),
                     "progress": (float(t["exit"]) - float(t["entry"])) / span,
                     "r": float(t[r_key]) if r_key in t else np.nan})
    return pd.DataFrame(rows)


def holding_minutes(trades: list[dict], *, reason_key: str = "exit_reason") -> pd.DataFrame:
    """Holding time per exit group, in minutes (needs entry_time/exit_time)."""
    rows = []
    for t in trades:
        if "entry_time" not in t or "exit_time" not in t:
            continue
        a, b = pd.Timestamp(t["entry_time"]), pd.Timestamp(t["exit_time"])
        rows.append({"reason": str(t.get(reason_key) or "unknown"),
                     "minutes": (b - a).total_seconds() / 60.0})
    if not rows:
        return pd.DataFrame(columns=["reason", "median_minutes", "mean_minutes"])
    df = pd.DataFrame(rows)
    g = df.groupby("reason")["minutes"]
    return pd.DataFrame({"median_minutes": g.median(), "mean_minutes": g.mean()})


def verdict(attr: pd.DataFrame) -> str:
    """One line: is the target doing any work?"""
    tgt = attr.loc[attr["is_target"], "share"].sum()
    tim = attr.loc[attr["is_time"], "share"].sum()
    if not attr["is_target"].any():
        return ("kein Ziel-Exit vorhanden - die Strategie ist ein reiner "
                "Stop/Zeit-Exit; ein RR-Parameter existiert nicht.")
    if tgt < 0.20:
        return (f"Ziel greift nur in {tgt*100:.0f} % der Trades, Zeit-Exits in "
                f"{tim*100:.0f} % -> das Ziel ist dekorativ, nicht wirksam.")
    if tgt < 0.40:
        return (f"Ziel greift in {tgt*100:.0f} % der Trades (Zeit-Exits {tim*100:.0f} %) "
                f"-> Ziel und Horizont nur teilweise abgestimmt.")
    return f"Ziel greift in {tgt*100:.0f} % der Trades - das RR ist ein echter Parameter."


def report(trades: list[dict], label: str = "", *, r_key: str = "r",
           reason_key: str = "exit_reason") -> str:
    attr = attribution(trades, r_key=r_key, reason_key=reason_key)
    lines = [f"### Exit-Attribution{': ' + label if label else ''}",
             f"    Trades {len(trades)}   Summe R {attr['sum_r'].sum():+.1f}   "
             f"Ø R {attr['sum_r'].sum()/max(len(trades), 1):+.4f}", ""]
    lines.append(f"    {'Exit':16s} {'n':>5s} {'Anteil':>7s} {'Ø R':>8s} {'Summe R':>9s} "
                 f"{'Treffer':>8s} {'Anteil R':>9s}")
    for reason, r in attr.iterrows():
        lines.append(f"    {reason:16s} {int(r.n):5d} {r.share*100:6.1f}% {r.mean_r:+8.4f} "
                     f"{r.sum_r:+9.1f} {r.win_rate*100:7.1f}% {r.share_of_r*100:+8.1f}%")
    pr = progress(trades, reason_key=reason_key, r_key=r_key)
    if not pr.empty:
        lines.append("")
        lines.append("    Ziel-Fortschritt (1.0 = Ziel erreicht):")
        for reason, g in pr.groupby("reason"):
            lines.append(f"      {reason:14s} n={len(g):4d}  Median {g['progress'].median():+5.2f}  "
                         f"p90 {g['progress'].quantile(0.9):+5.2f}")
    hm = holding_minutes(trades, reason_key=reason_key)
    if not hm.empty:
        lines.append("")
        lines.append("    Haltedauer (Minuten):")
        for reason, r in hm.iterrows():
            lines.append(f"      {reason:14s} Median {r.median_minutes:7.1f}  "
                         f"Mittel {r.mean_minutes:7.1f}")
    lines += ["", "    -> " + verdict(attr)]
    return "\n".join(lines)
