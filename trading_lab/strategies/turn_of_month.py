"""S-008: Turn of the Month - equity indices long over the month change (McConnell/Xu 2008).

Spec (agreed with the user 2026-10-29): ``docs/specs/turn_of_month.yaml``. Tests:
``tests/test_turn_of_month.py``.

 1. ``tom_windows``: per index, from ITS trading days: enter at the close of the second-last
    trading day of a month (day -2), exit at the close of the 3rd trading day of the next
    month (day +3) - the -1..+3 window.
 2. Sizing: the indices share ONE risk budget: each position = (0.5 % / number of indices
    active on that day) / daily EWMA sigma of that index -> 1 R = 0.5 % of the account.
 3. Costs on entry and exit (one-way bp), financing per calendar day held (``carry_rates``
    of S-007: long pays the quote-currency rate + markup).
Data alignment, rates and costs come from ``trading_lab.strategies.tsmom_multi_asset``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .tsmom_multi_asset import Prepared, TsmomConfig, carry_rates, daily_returns, ewma_vol

INDICES = ("SP500", "US100", "US30", "GER40", "UK100", "JP225")
UNIT = 0.005


@dataclass(frozen=True)
class TomConfig:
    pre_days: int = 2               # enter at the close of day -pre_days (day -2 -> window -1..)
    post_days: int = 3              # exit at the close of day +post_days
    vol_com: int = 60
    markup_cfd: float = 0.025


def tom_windows(dates: pd.DatetimeIndex, has_bar: np.ndarray, cfg: TomConfig = TomConfig()
                ) -> list[tuple[int, int]]:
    """(entry row, exit row) on the calendar ``dates`` for one index (its own trading days)."""
    rows = np.flatnonzero(has_bar)
    if len(rows) == 0:
        return []
    month = dates[rows].to_period("M")
    out = []
    groups = pd.Series(rows).groupby(np.asarray(month)).apply(list)
    keys = list(groups.index)
    for a, b in zip(keys[:-1], keys[1:]):
        cur, nxt = groups[a], groups[b]
        if b != a + 1 or len(cur) < cfg.pre_days or len(nxt) < cfg.post_days:
            continue
        out.append((cur[-cfg.pre_days], nxt[cfg.post_days - 1]))
    return out


def random_windows(dates: pd.DatetimeIndex, has_bar: np.ndarray, rng: np.random.Generator,
                   hold: int = 4) -> list[tuple[int, int]]:
    """Placebo: per month one entry on a random mid-month trading day (6th .. 6th-last),
    held ``hold`` trading days."""
    rows = np.flatnonzero(has_bar)
    month = dates[rows].to_period("M")
    out = []
    for _, g in pd.Series(rows).groupby(np.asarray(month)):
        g = list(g)
        lo, hi = 5, len(g) - 6 - hold
        if hi < lo:
            continue
        k = int(rng.integers(lo, hi + 1))
        out.append((g[k], g[k + hold]))
    return out


def run_backtest(p: Prepared, cfg: TomConfig = TomConfig(), *,
                 windows: dict[str, list[tuple[int, int]]] | None = None) -> dict:
    T = len(p.dates)
    idx = [i for i, a in enumerate(p.assets) if a in INDICES]
    ret = daily_returns(p.close)
    sd = ewma_vol(ret, cfg.vol_com) / np.sqrt(252)
    days = np.r_[0, np.diff(p.dates.values).astype("timedelta64[D]").astype(int)]
    tcfg = TsmomConfig(markup_cfd=cfg.markup_cfd)
    if windows is None:
        windows = {p.assets[i]: tom_windows(p.dates, p.has_bar[:, i], cfg) for i in idx}
    # shared budget: divided by the number of indices ACTIVE (data + vol) on the entry day -
    # not by how many enter that day (a placebo with scattered entries would get 6x the size)
    active = np.isfinite(sd[:, idx]) & (sd[:, idx] > 0)
    n_active = np.maximum(active.sum(axis=1), 1)
    out = np.zeros((T, 4))                         # gross, net, cost, fin
    trades = []
    for i in idx:
        a = p.assets[i]
        lo_, _ = carry_rates(a, p.rates, tcfg)
        car = lo_.reindex(p.dates).ffill().fillna(0).to_numpy()
        for e, x in windows.get(a, []):
            if not np.isfinite(sd[e, i]) or sd[e, i] <= 0:
                continue
            pos = UNIT / int(n_active[e]) / sd[e, i]
            c = pos * p.cost_bp[i] / 1e4
            g = pos * ret[e + 1:x + 1, i]
            f = pos * car[e + 1:x + 1] * days[e + 1:x + 1] / 365.0
            out[e + 1:x + 1, 0] += g
            out[e + 1:x + 1, 3] += f
            out[e, 2] += c
            out[x, 2] += c
            trades.append({"asset": a, "entry_time": p.dates[e], "exit_time": p.dates[x], "direction": 1,
                           "r_gross": g.sum() / UNIT, "r": (g.sum() + f.sum() - 2 * c) / UNIT,
                           "exit_reason": "zeit"})
    out[:, 1] = out[:, 0] - out[:, 2] + out[:, 3]
    return {"daily": pd.DataFrame(out, index=p.dates, columns=["gross", "net", "cost", "fin"]),
            "trades": trades, "unit": UNIT}


__all__ = ["TomConfig", "INDICES", "UNIT", "tom_windows", "random_windows", "run_backtest"]
