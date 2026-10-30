"""S-010: volatility-managed equity premium (Moreira/Muir 2017) - second basket block next to S-007.

Spec (agreed with the user 2026-10-30): ``docs/specs/vol_managed_equity.yaml``; tests:
``tests/test_vol_managed_equity.py``.

 1. Basket return: equal-weight mean of the daily returns of the active indices.
 2. Monthly variance: sum of squared basket returns of a month x 12.
 3. Exposure for the next month: ``varianz`` c / var (paper), c = target x sqrt(expanding mean of
    the monthly variances, known data only); ``vola`` target / sqrt(var); ``konstant`` target /
    sqrt(expanding mean) - the plain premium at the same average risk (comparison, not a variant).
    Cap 3 x the account, split equally over the active indices.
 4. Signal from the month's last close, executed at the close of the first trading day of the next
    month; changes below 10 % are not traded. Long only (``flips`` = random sign per month: placebo).
Data alignment comes from ``tsmom_multi_asset.prepare``. P&L is not compounded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .tsmom_multi_asset import Prepared, daily_returns

INDICES = ("SP500", "US100", "US30", "GER40", "UK100", "JP225")
MIN_MONTHS = 24


@dataclass(frozen=True)
class VolManagedConfig:
    gewicht: str = "varianz"        # varianz | vola | konstant
    target_vol: float = 0.08
    max_leverage: float = 3.0
    buffer: float = 0.10


def basket_returns(p: Prepared) -> tuple[pd.Series, list[int]]:
    idx = [i for i, a in enumerate(p.assets) if a in INDICES]
    ret = daily_returns(p.close)[:, idx]
    live = np.cumsum(p.has_bar[:, idx], axis=0) >= 2
    n = np.maximum(live.sum(axis=1), 1)
    return pd.Series((ret * live).sum(axis=1) / n, index=p.dates), idx


def monthly_variance(r: pd.Series) -> pd.Series:
    """Rule 2, indexed by month (period)."""
    return (r ** 2).groupby(r.index.to_period("M")).sum() * 12


def exposure(var: pd.Series, cfg: VolManagedConfig) -> pd.Series:
    """Rule 3: exposure decided at the end of each month, known data only."""
    mean = var.expanding(MIN_MONTHS).mean()
    if cfg.gewicht == "varianz":
        w = cfg.target_vol * np.sqrt(mean) / var
    elif cfg.gewicht == "vola":
        w = (cfg.target_vol / np.sqrt(var)).where(mean.notna())
    elif cfg.gewicht == "konstant":
        w = cfg.target_vol / np.sqrt(mean)
    else:
        raise ValueError(cfg.gewicht)
    return w.clip(upper=cfg.max_leverage)


def run_backtest(p: Prepared, cfg: VolManagedConfig, long_pa: dict[str, float],
                 flips: pd.Series | None = None) -> dict:
    """``long_pa`` = financing of a long p.a. per index (negative = pays); short: same cost."""
    r, idx = basket_returns(p)
    w_month = exposure(monthly_variance(r), cfg)
    ret = daily_returns(p.close)[:, idx]
    names = [p.assets[i] for i in idx]
    fin = np.array([long_pa.get(a, 0.0) for a in names])
    cost = p.cost_bp[idx] / 1e4
    days = np.r_[0, np.diff(p.dates.values).astype("timedelta64[D]").astype(int)]
    months = p.dates.to_period("M")
    unit = cfg.target_vol / np.sqrt(252)
    T, N = ret.shape
    pos, cur = np.zeros(N), 0.0
    out = np.zeros((T, 4))                         # gross, net, cost, fin
    trades, ep = [], [None] * N

    def close_ep(i: int, t: int) -> None:
        e = ep[i]
        trades.append({"asset": names[i], "entry_time": e["start"], "exit_time": p.dates[t], "direction": e["dir"],
                       "r": e["net"] / unit, "r_gross": e["gross"] / unit, "exit_reason": "monat"})
        ep[i] = None

    for t in range(1, T):
        g = pos * ret[t]
        f = np.abs(pos) * fin * days[t] / 365.0    # both sides pay on the measured account
        out[t, 0] += g.sum()
        out[t, 3] += f.sum()
        for i in np.flatnonzero(pos):
            ep[i]["gross"] += g[i]
            ep[i]["net"] += g[i] + f[i]
        if months[t] != months[t - 1]:             # first trading day of a month: rule 4
            w = w_month.get(months[t - 1], np.nan)
            if not np.isfinite(w):
                continue
            s = float(flips.get(months[t], 1.0)) if flips is not None else 1.0
            tgt_w = w * s
            if cur != 0.0 and np.sign(tgt_w) == np.sign(cur) and abs(tgt_w - cur) <= cfg.buffer * abs(cur):
                continue
            live = p.has_bar[:t + 1, idx].sum(axis=0) >= 2
            tgt = np.where(live, tgt_w / max(live.sum(), 1), 0.0)
            c = np.abs(tgt - pos) * cost
            out[t, 2] += c.sum()
            for i in range(N):
                if ep[i] is not None:
                    close_ep(i, t)
                if tgt[i] != 0:                    # only the CHANGE is traded and paid for
                    ep[i] = {"start": p.dates[t], "dir": int(np.sign(tgt[i])), "gross": 0.0, "net": -c[i]}
            pos, cur = tgt, tgt_w
    for i in np.flatnonzero(pos):
        close_ep(i, T - 1)
    out[:, 1] = out[:, 0] - out[:, 2] + out[:, 3]
    return {"daily": pd.DataFrame(out, index=p.dates, columns=["gross", "net", "cost", "fin"]),
            "trades": trades, "unit": unit, "exposure": w_month}


__all__ = ["VolManagedConfig", "INDICES", "basket_returns", "monthly_variance", "exposure", "run_backtest"]
