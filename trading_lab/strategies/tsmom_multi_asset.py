"""S-007: time-series momentum across asset classes (Moskowitz/Ooi/Pedersen 2012).

Spec (agreed with the user 2026-10-29): ``docs/specs/tsmom_multi_asset.yaml``. Each rule
is a small function with a test in ``tests/test_tsmom_multi_asset.py``.

 1. ``signals``: sign of the 12-month return (variant ``blend``: mean sign over 1/3/12 months).
 2. ``ewma_vol``: ex-ante volatility, EWMA of squared daily returns, centre of mass 60 days.
 3. ``budgets``: equal risk per asset class (crypto half), equal inside a class.
 4. ``target_positions``: signal x budget / vol, scaled to 8 % p.a. ex-ante portfolio
    volatility (EWMA covariance), capped by the account's leverage per class.
 5. weekly rebalance (Wednesday close; signal from Tuesday's close), only when the target
    changes sign or by more than 25 % (``needs_trade``).
 6. variant ``stop_3atr``: emergency stop entry -/+ 3 x ATR(20); after a stop the market stays
    flat until its signal changes direction.
Costs: one-way cost in bp per market (measured half spread + slippage + commission) on
every change of position. Financing per calendar day held (``carry_rates``): FX = rate
differential - markup, CFDs = long pays quote-currency rate + markup, short receives rate -
markup. Summing calendar days puts the weekend carry on Monday instead of Wednesday /
Friday - the same total.

Positions are fractions of the (fixed) account; P&L is not compounded. 1 R = target
volatility / sqrt(252) of the account (one daily sigma of the portfolio).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

GROUPS = {
    "aktienindizes": ("SP500", "US100", "US30", "GER40", "UK100", "JP225"),
    "fx_usd": ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"),
    "fx_kreuz": ("EURJPY", "EURGBP", "EURCHF", "GBPJPY", "AUDJPY", "AUDNZD", "AUDCAD"),
    "metalle": ("XAUUSD", "XAGUSD"),
    "rohstoffe": ("BRENT", "WTI", "COPPER"),
    "anleihen": ("UST10Y",),
    "krypto": ("BTCUSD",),
}
GROUP_OF = {a: g for g, xs in GROUPS.items() for a in xs}
GROUP_WEIGHT = {"krypto": 0.5}
LEVERAGE = {"aktienindizes": 20.0, "fx_usd": 100.0, "fx_kreuz": 100.0, "metalle": 10.0, "rohstoffe": 10.0,
            "anleihen": 10.0, "krypto": 2.0}
QUOTE_CCY = {"GER40": "EUR", "UK100": "GBP", "JP225": "JPY"}       # everything else not FX: USD
YEAR = 252
LOOKBACKS = {"12m": (252,), "blend_1_3_12": (21, 63, 252)}


@dataclass(frozen=True)
class TsmomConfig:
    signal: str = "12m"             # 12m | blend_1_3_12
    stop: str = "keiner"            # keiner | stop_3atr
    target_vol: float = 0.08
    vol_com: int = 60
    buffer: float = 0.25
    rebalance_dow: int = 2          # Wednesday
    stop_atr: float = 3.0
    atr_n: int = 20
    markup_fx: float = 0.010
    markup_cfd: float = 0.025
    markup_factor: float = 1.0      # robustness run: 2.0
    # measured broker swaps (trading_lab/broker_costs.py): ((asset, long p.a., short p.a.), ...)
    swaps_pa: tuple = ()


@dataclass
class Prepared:
    dates: pd.DatetimeIndex
    assets: list[str]
    open: np.ndarray                # T x N, NaN where the market has no bar
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray               # forward-filled
    has_bar: np.ndarray             # T x N bool
    rates: pd.DataFrame             # daily, % p.a. per currency (forward-filled)
    cost_bp: np.ndarray             # N, one-way
    groups: list[str] = field(default_factory=list)


def prepare(frames: dict[str, pd.DataFrame], rates: pd.DataFrame, cost_bp: dict[str, float],
            start: str | None = None, end: str | None = None) -> Prepared:
    """Align daily OHLC frames on one business-day calendar (no weekends)."""
    assets = [a for a in frames if a in GROUP_OF]
    idx = pd.DatetimeIndex(sorted(set().union(*[frames[a].index for a in assets])))
    idx = idx[idx.dayofweek < 5]
    if start:
        idx = idx[idx >= pd.Timestamp(start)]
    if end:
        idx = idx[idx < pd.Timestamp(end)]
    mats = {k: np.column_stack([frames[a][k].reindex(idx).to_numpy(float) for a in assets])
            for k in ("open", "high", "low", "close")}
    has = np.isfinite(mats["close"])
    close = pd.DataFrame(mats["close"]).ffill().to_numpy()
    r = rates.copy()
    r.index = pd.DatetimeIndex(r.index)
    daily_rates = r.reindex(r.index.union(idx)).ffill().reindex(idx)
    return Prepared(idx, assets, mats["open"], mats["high"], mats["low"], close, has, daily_rates,
                    np.array([cost_bp.get(a, 0.0) for a in assets]), [GROUP_OF[a] for a in assets])


# ── rules 1-4 ────────────────────────────────────────────────────────────────

def signals(close: np.ndarray, kind: str) -> np.ndarray:
    """T x N signal in [-1, 1]; NaN until the longest lookback has data."""
    out = []
    for lb in LOOKBACKS[kind]:
        past = np.full_like(close, np.nan)
        past[lb:] = close[:-lb]
        out.append(np.sign(close / past - 1.0))
    s = np.mean(out, axis=0)
    return s


def daily_returns(close: np.ndarray) -> np.ndarray:
    r = np.zeros_like(close)
    r[1:] = close[1:] / close[:-1] - 1.0
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def ewma_vol(ret: np.ndarray, com: int, min_obs: int = 20) -> np.ndarray:
    """Annualised EWMA volatility known at each close (uses returns up to that day)."""
    lam = com / (1.0 + com)
    v = np.full_like(ret, np.nan)
    acc = np.zeros(ret.shape[1])
    for t in range(ret.shape[0]):
        acc = lam * acc + (1 - lam) * ret[t] ** 2
        if t + 1 >= min_obs:
            v[t] = np.sqrt(acc * YEAR)
    return v


def budgets(active: np.ndarray, groups: list[str]) -> np.ndarray:
    """Risk budget per market for one day: equal per active class (crypto half), equal inside."""
    w = np.zeros(len(groups))
    act = [g for g in dict.fromkeys(groups) if any(a and gg == g for a, gg in zip(active, groups))]
    if not act:
        return w
    tot = sum(GROUP_WEIGHT.get(g, 1.0) for g in act)
    for g in act:
        idx = [i for i, (a, gg) in enumerate(zip(active, groups)) if a and gg == g]
        for i in idx:
            w[i] = GROUP_WEIGHT.get(g, 1.0) / tot / len(idx)
    return w


def target_positions(sig: np.ndarray, vol: np.ndarray, cov: np.ndarray, groups: list[str],
                     cfg: TsmomConfig) -> np.ndarray:
    """One day: position per market as a fraction of the account (rule 4)."""
    active = np.isfinite(sig) & np.isfinite(vol) & (vol > 0)
    w = budgets(active, groups)
    raw = np.where(active, np.nan_to_num(sig) * w / np.where(active, vol, 1.0), 0.0)
    pv = float(np.sqrt(max(raw @ cov @ raw, 0.0)))
    if pv <= 0:
        return np.zeros_like(raw)
    pos = raw * cfg.target_vol / pv
    cap = np.array([LEVERAGE[g] for g in groups])
    return np.clip(pos, -cap, cap)


def needs_trade(cur: float, tgt: float, buffer: float) -> bool:
    """Rule 5: trade when opening, closing, flipping or changing by more than ``buffer``."""
    if cur == 0.0 or tgt == 0.0:
        return cur != tgt
    if np.sign(cur) != np.sign(tgt):
        return True
    return abs(tgt - cur) > buffer * abs(cur)


def carry_rates(asset: str, rates: pd.DataFrame, cfg: TsmomConfig) -> tuple[pd.Series, pd.Series]:
    """(long, short) financing in fraction p.a. per day; negative = the position pays.
    A measured broker swap (``cfg.swaps_pa``) replaces the formula for that market."""
    measured = {a: (lo, sh) for a, lo, sh in cfg.swaps_pa}
    if asset in measured:
        lo, sh = measured[asset]
        return pd.Series(lo, index=rates.index), pd.Series(sh, index=rates.index)
    if GROUP_OF[asset] in ("fx_usd", "fx_kreuz"):
        base, quote = asset[:3], asset[3:]
        diff = (rates[base] - rates[quote]) / 100.0
        m = cfg.markup_fx * cfg.markup_factor
        return diff - m, -diff - m
    q = rates[QUOTE_CCY.get(asset, "USD")] / 100.0
    m = cfg.markup_cfd * cfg.markup_factor
    return -(q + m), q - m


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    prev = np.vstack([np.full(close.shape[1], np.nan), close[:-1]])
    tr = np.nanmax(np.stack([high - low, np.abs(high - prev), np.abs(low - prev)]), axis=0)
    return pd.DataFrame(tr).rolling(n, min_periods=n).mean().to_numpy()


# ── engine ───────────────────────────────────────────────────────────────────

def run_backtest(p: Prepared, cfg: TsmomConfig, *, sign_flip: np.ndarray | None = None) -> dict:
    """Daily simulation. ``sign_flip`` (T x N of +-1) turns it into the placebo."""
    T, N = p.close.shape
    groups = p.groups or [GROUP_OF[a] for a in p.assets]
    ret = daily_returns(p.close)
    sig = signals(p.close, cfg.signal)
    if sign_flip is not None:
        sig = sig * sign_flip
    vol = ewma_vol(ret, cfg.vol_com)
    a20 = atr(p.high, p.low, p.close, cfg.atr_n)
    car = [carry_rates(a, p.rates, cfg) for a in p.assets]
    car_l = np.column_stack([c[0].reindex(p.dates).ffill().fillna(0).to_numpy() for c in car])
    car_s = np.column_stack([c[1].reindex(p.dates).ffill().fillna(0).to_numpy() for c in car])
    lam = cfg.vol_com / (1.0 + cfg.vol_com)
    cov = np.zeros((N, N))
    days = np.r_[0, np.diff(p.dates.values).astype("timedelta64[D]").astype(int)]
    unit = cfg.target_vol / np.sqrt(YEAR)

    pos = np.zeros(N)
    stop_px = np.full(N, np.nan)
    stopped = np.zeros(N)                         # sign the market was stopped in (0 = none)
    ep = [None] * N
    trades: list[dict] = []
    out = np.zeros((T, 4))                        # gross, net, cost, fin

    def close_ep(i: int, t: int, reason: str, cost_share: float) -> None:
        e = ep[i]
        e["net"] -= cost_share
        trades.append({"asset": p.assets[i], "entry_time": e["start"], "exit_time": p.dates[t],
                       "direction": e["dir"], "r": e["net"] / unit, "r_gross": e["gross"] / unit,
                       "exit_reason": reason, "group": groups[i]})
        ep[i] = None

    for t in range(1, T):
        # ---- P&L of the positions held from the close of t-1 ----
        for i in np.flatnonzero(pos):
            d = np.sign(pos[i])
            f = pos[i] * (car_l[t, i] if d > 0 else car_s[t, i]) * days[t] / 365.0
            pnl = pos[i] * ret[t, i]
            hit = False
            if cfg.stop == "stop_3atr" and np.isfinite(stop_px[i]) and p.has_bar[t, i]:
                lo, hi, op = p.low[t, i], p.high[t, i], p.open[t, i]
                if (d > 0 and lo <= stop_px[i]) or (d < 0 and hi >= stop_px[i]):
                    fill = stop_px[i] if not np.isfinite(op) else (min(stop_px[i], op) if d > 0
                                                                     else max(stop_px[i], op))
                    pnl = pos[i] * (fill / p.close[t - 1, i] - 1.0)
                    hit = True
            out[t, 0] += pnl
            out[t, 3] += f
            ep[i]["gross"] += pnl
            ep[i]["net"] += pnl + f
            if hit:
                c = abs(pos[i]) * p.cost_bp[i] / 1e4
                out[t, 2] += c
                close_ep(i, t, "stop", c)
                stopped[i] = d
                pos[i] = 0.0
                stop_px[i] = np.nan
        # ---- clear stops whose signal changed direction ----
        s_prev = np.sign(np.nan_to_num(sig[t - 1]))
        for i in np.flatnonzero(stopped):
            if s_prev[i] != stopped[i]:
                stopped[i] = 0.0
        # ---- covariance known at t-1 (updated with the return of t-1) ----
        r_prev = ret[t - 1]
        cov = lam * cov + (1 - lam) * np.outer(r_prev, r_prev)
        # ---- weekly rebalance at the close of t with information of t-1 ----
        if p.dates[t].dayofweek == cfg.rebalance_dow:
            tgt = target_positions(sig[t - 1], vol[t - 1], cov * YEAR, groups, cfg)
            tgt = np.where(stopped != 0, 0.0, tgt)
            for i in range(N):
                cur = pos[i]
                if not needs_trade(cur, tgt[i], cfg.buffer) or not p.has_bar[t, i]:
                    continue
                flip = cur != 0 and (tgt[i] == 0 or np.sign(tgt[i]) != np.sign(cur))
                c_old = abs(cur) * p.cost_bp[i] / 1e4 if flip else 0.0
                c_new = abs(tgt[i] - (0.0 if flip else cur)) * p.cost_bp[i] / 1e4
                out[t, 2] += c_old + c_new
                if flip:
                    close_ep(i, t, "signal", c_old)
                if tgt[i] != 0 and ep[i] is None:
                    ep[i] = {"start": p.dates[t], "dir": int(np.sign(tgt[i])), "gross": 0.0, "net": 0.0}
                    stop_px[i] = (p.close[t, i] - np.sign(tgt[i]) * cfg.stop_atr * a20[t - 1, i]
                                  if np.isfinite(a20[t - 1, i]) else np.nan)
                if ep[i] is not None:
                    ep[i]["net"] -= c_new
                pos[i] = tgt[i]
    for i in np.flatnonzero(pos):
        close_ep(i, T - 1, "data_end", 0.0)
    out[:, 1] = out[:, 0] - out[:, 2] + out[:, 3]
    daily = pd.DataFrame(out, index=p.dates, columns=["gross", "net", "cost", "fin"])
    return {"daily": daily, "trades": trades, "unit": unit}


def placebo_flips(sig: np.ndarray, seed: int) -> np.ndarray:
    """+-1 per market and per run of the REAL signal sign: same timing of changes, random
    direction (destroys the information, keeps the turnover)."""
    rng = np.random.default_rng(seed)
    s = np.sign(np.nan_to_num(sig))
    out = np.ones_like(s)
    for i in range(s.shape[1]):
        seg = np.r_[0, np.cumsum(s[1:, i] != s[:-1, i])]
        flips = rng.choice([-1.0, 1.0], size=seg.max() + 1)
        out[:, i] = flips[seg]
    return out


__all__ = ["TsmomConfig", "Prepared", "prepare", "GROUPS", "GROUP_OF", "signals", "ewma_vol", "budgets", "target_positions",
           "needs_trade", "carry_rates", "run_backtest", "placebo_flips", "daily_returns", "atr"]
