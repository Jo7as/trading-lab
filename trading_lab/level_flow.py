"""Levels x order flow (H8, docs/research/CRYPTO_ORDERFLOW_PLAN.md): what happens when price
reaches a level, and does the flow at that moment tell reversal from continuation?

Mechanism (docs/research/CRYPTO_MICROSTRUCTURE_THEORY.md): stops and liquidations cluster
beyond levels. When price gets there, aggressive flow hits the book. Either passive size
absorbs it (strong flow, price does not get through -> reversal) or the flow feeds on itself
(flow and price go through together -> continuation). SMC's "liquidity grab" story is the
first case; it was never shown to work on price alone (FINDINGS S-001, sweep study).

Building blocks, all causal (a level exists only from ``known_at``):

* ``period_levels``      previous day / week high and low (UTC periods - crypto trades 24/7)
* ``structure_levels``   external swings, EQH/EQL pools, order blocks (``market_structure``)
* ``profile_levels_daily`` previous day POC / VAH / VAL (``orderflow.volume_profile``)
* ``touch_events``       first touch of each level, flow and position in the reaction window,
                         forward returns after it
* ``classify``           absorption / cascade / reclaim / break by fixed thresholds
* ``control_levels``     the same levels shifted by a random amount (placebo)
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import market_structure as ms
from . import orderflow as of


@dataclass(frozen=True)
class Level:
    kind: str                    # e.g. "pdh", "pwl", "swing_high", "eqh", "ob_bear", "poc"
    price: float
    known_at: pd.Timestamp       # first bar label at which the level may be used


# ── level sources ───────────────────────────────────────────────────────
def period_levels(bars: pd.DataFrame, period: str, name: str) -> list[Level]:
    """High and low of every completed UTC day (``period="day"``) or week starting Monday
    00:00 UTC (``"week"``), known from the moment the period ends."""
    day = bars.index.floor("1D")
    if period == "day":
        key, step = day, pd.Timedelta(days=1)
    elif period == "week":
        key, step = day - pd.to_timedelta(bars.index.weekday, unit="D"), pd.Timedelta(days=7)
    else:
        raise ValueError("period must be 'day' or 'week'")
    g = bars.groupby(key)
    hi, lo = g["high"].max(), g["low"].min()
    out = []
    for start in hi.index:
        known = start + step
        if known > bars.index[-1] or not np.isfinite(hi[start]):
            continue
        out.append(Level(f"{name}h", float(hi[start]), known))
        out.append(Level(f"{name}l", float(lo[start]), known))
    return out


def structure_levels(m15: pd.DataFrame, cfg: ms.StructureConfig = ms.StructureConfig()) -> list[Level]:
    """External swing highs/lows, EQH/EQL pools and order-block edges from ``market_structure``.

    A swing is known when it is confirmed, a pool / zone when it is created - each at the
    END of that 15-min bar. Order blocks give their edge facing the approach: the bottom of a
    bearish block (resistance), the top of a bullish one (support).
    """
    res = ms.analyze(m15, cfg)
    step = m15.index[1] - m15.index[0]
    end = lambda i: m15.index[i] + step                      # noqa: E731
    out = []
    for s in res.swings.get("ext", []):
        out.append(Level("swing_high" if s.kind > 0 else "swing_low", float(s.price), end(s.confirmed_idx)))
    for p in res.pools:
        out.append(Level("eqh" if p.kind > 0 else "eql", float(p.level), end(p.created_idx)))
    for z in res.zones:
        if z.kind == "ob":
            out.append(Level("ob_bear" if z.direction < 0 else "ob_bull",
                             float(z.bottom if z.direction < 0 else z.top), end(z.created_idx)))
    return out


def profile_levels_daily(bars: pd.DataFrame, bin_size: float) -> list[Level]:
    """Previous UTC day's POC / VAH / VAL, known from the next day's start."""
    out = []
    for day, g in bars.groupby(bars.index.floor("1D")):
        lv = of.profile_levels(of.volume_profile(g, bin_size))
        known = day + pd.Timedelta(days=1)
        if known > bars.index[-1] or not np.isfinite(lv["poc"]):
            continue
        for k in ("poc", "vah", "val"):
            out.append(Level(k, lv[k] + bin_size / 2, known))   # bin centre
    return out


def control_levels(levels: list[Level], *, low_frac: float = 0.0015, high_frac: float = 0.005,
                   seed: int = 0) -> list[Level]:
    """Placebo: each level moved by a random +-0.15 %..0.5 % of its price, same timing."""
    rng = np.random.default_rng(seed)
    out = []
    for lv in levels:
        u = rng.uniform(low_frac, high_frac) * rng.choice([-1.0, 1.0])
        out.append(replace(lv, kind=lv.kind, price=lv.price * (1 + u)))
    return out


# ── events ──────────────────────────────────────────────────────────────
def touch_events(bars: pd.DataFrame, levels: list[Level], atr: pd.Series, *, window: int = 30,
                 horizons: dict[str, str] | None = None, max_age: str = "3D") -> pd.DataFrame:
    """First touch of each level after it is known, with reaction-window flow and outcomes.

    * approach ``d``: +1 if the level was ABOVE the price when it became known (price rises
      into it), -1 if below. A level within one bar's range at that moment is skipped.
    * touch bar ``i``: first bar with high >= level (d=+1) / low <= level (d=-1), at most
      ``max_age`` after ``known_at``.
    * reaction window = bars i .. i+window-1; ``flow`` = taker imbalance over it,
      ``beyond_atr`` = d x (close at its end - level) / ATR  (> 0: through the level).
    * outcomes ``ret_<h>`` = log return from the end of the window, in bp, NOT yet signed by
      any trade direction. ATR must be known at the touch (pass a causal series).
    """
    horizons = horizons or {"1h": "1h", "4h": "4h"}
    idx = bars.index
    step = idx[1] - idx[0]
    hi, lo = bars["high"].to_numpy(float), bars["low"].to_numpy(float)
    px = bars["close"].ffill().to_numpy(float)
    b, s = bars["buy_notional"].to_numpy(float), bars["sell_notional"].to_numpy(float)
    a = atr.reindex(idx).to_numpy(float)
    n_age = int(pd.Timedelta(max_age) / step)
    h_steps = {k: int(pd.Timedelta(v) / step) for k, v in horizons.items()}
    rows = []
    for lv in levels:
        k0 = idx.searchsorted(lv.known_at)
        if k0 >= len(idx) or not np.isfinite(px[k0]):
            continue
        if lo[k0] <= lv.price <= hi[k0]:
            continue                                          # already at the level
        d = 1 if lv.price > px[k0] else -1
        seg = slice(k0, min(k0 + n_age, len(idx)))
        hit = np.flatnonzero(hi[seg] >= lv.price) if d > 0 else np.flatnonzero(lo[seg] <= lv.price)
        if len(hit) == 0:
            continue
        i = k0 + int(hit[0])
        e = i + window - 1
        if e + max(h_steps.values()) >= len(idx) or not np.isfinite(a[i]) or a[i] <= 0:
            continue
        bb, ss = b[i:e + 1].sum(), s[i:e + 1].sum()
        row = {"ts": idx[i], "kind": lv.kind, "level": lv.price, "d": d,
               "flow": (bb - ss) / (bb + ss) if bb + ss > 0 else np.nan,
               "beyond_atr": d * (px[e] - lv.price) / a[i], "atr_bp": a[i] / px[i] * 1e4}
        for name, k in h_steps.items():
            row[f"ret_{name}"] = np.log(px[e + k] / px[e]) * 1e4
        rows.append(row)
    return pd.DataFrame(rows)


#: pre-registered thresholds (2026-10-29, before the first run on data)
FLOW_THR = 0.15          # |taker imbalance| in the approach direction over the 5-min window
BREAK_ATR = 0.25         # close this far beyond the level = "through"


def classify(ev: pd.DataFrame, *, flow_thr: float = FLOW_THR, break_atr: float = BREAK_ATR) -> pd.DataFrame:
    """Label each touch and give the trade direction the label implies.

    * absorption: flow pushes INTO the level (d x flow >= thr) but the close is back on the
      approach side (beyond_atr <= 0)                                    -> trade -d
    * cascade:    flow pushes into the level and the close is through by >= break_atr -> +d
    * reclaim:    close back on the approach side WITHOUT strong flow    -> -d (price only)
    * break:      close through by >= break_atr WITHOUT strong flow       -> +d (price only)
    * none:       in between
    The price-only labels are the comparison: what does the flow add to SMC's own reading?
    """
    out = ev.copy()
    push = out["d"] * out["flow"] >= flow_thr
    back = out["beyond_atr"] <= 0
    through = out["beyond_atr"] >= break_atr
    out["label"] = np.select([push & back, push & through, ~push & back, ~push & through],
                             ["absorption", "cascade", "reclaim", "break"], default="none")
    out["trade"] = np.select([out["label"].isin(["absorption", "reclaim"]),
                              out["label"].isin(["cascade", "break"])], [-out["d"], out["d"]], default=0)
    return out


# ── H9: follow the flow after absorption (pre-registered, FINDINGS 2026-10-29) ──
def follow_absorption(ev: pd.DataFrame, *, flow_thr: float = FLOW_THR) -> pd.DataFrame:
    """Absorption events (H8 definition: flow into the level, close not beyond it) traded in the
    FLOW direction ``+d`` - the reverse of H8's fade. Returns only those events, ``trade = d``."""
    push = ev["d"] * ev["flow"] >= flow_thr
    out = ev[push & (ev["beyond_atr"] <= 0)].copy()
    out["label"] = "absorption"
    out["trade"] = out["d"]
    return out


def day_clustered_t(x: pd.Series, ts: pd.Series) -> float:
    """Mean / standard error of the per-UTC-day means (the H8 statistic)."""
    per_day = pd.Series(np.asarray(x, float)).groupby(pd.DatetimeIndex(ts).floor("1D")).mean()
    if len(per_day) < 3 or per_day.std(ddof=1) == 0:
        return float("nan")
    return float(per_day.mean() / (per_day.std(ddof=1) / np.sqrt(len(per_day))))


def cluster_diff_t(x_real, ts_real, x_ctl, ts_ctl) -> tuple[float, float]:
    """Difference of means real - control and its t, OLS on a real dummy with standard errors
    clustered by UTC day (CR1 small-sample factor). Returns (difference, t)."""
    y = np.concatenate([np.asarray(x_real, float), np.asarray(x_ctl, float)])
    r = np.concatenate([np.ones(len(x_real)), np.zeros(len(x_ctl))])
    g = np.concatenate([pd.DatetimeIndex(ts_real).floor("1D").asi8, pd.DatetimeIndex(ts_ctl).floor("1D").asi8])
    X = np.column_stack([np.ones_like(r), r])
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    u = y - X @ beta
    _, gi = np.unique(g, return_inverse=True)
    n_g = gi.max() + 1
    scores = np.zeros((n_g, 2))
    np.add.at(scores, gi, X * u[:, None])
    n, k = X.shape
    c = n_g / (n_g - 1) * (n - 1) / (n - k)
    V = c * bread @ (scores.T @ scores) @ bread
    return float(beta[1]), float(beta[1] / np.sqrt(V[1, 1]))


def random_direction_p(ret: pd.Series, trade: pd.Series, *, n_draws: int = 2000, seed: int = 0) -> float:
    """One-sided p: share of random +-1 directions on the same events whose mean signed return
    is >= the observed mean of ``trade x ret``."""
    r = np.asarray(ret, float)
    obs = float(np.mean(np.asarray(trade, float) * r))
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n_draws, len(r)))
    return float(np.mean(signs @ r / len(r) >= obs))


def non_overlapping(ev: pd.DataFrame, hold: str) -> pd.DataFrame:
    """Per market, keep an event only if it starts at least ``hold`` after the last kept one."""
    keep = []
    gap = pd.Timedelta(hold)
    for _, g in ev.sort_values("ts").groupby("market", sort=False):
        last = None
        for i, t in zip(g.index, g["ts"]):
            if last is None or t - last >= gap:
                keep.append(i)
                last = t
    return ev.loc[sorted(keep, key=lambda i: ev.at[i, "ts"])]


#: H9 decision thresholds (pre-registered 2026-10-29)
H9_T_GROSS, H9_T_DIFF, H9_P_RANDOM = 2.0, 1.65, 0.05


def h9_verdict(*, gross: float, t_gross: float, net: float, diff: float, t_diff: float,
               p_random: float, gross_by_market: dict[str, float]) -> tuple[str, dict[str, bool]]:
    """C1..C5 of the pre-registration; "confirmed" only if all hold."""
    c = {"C1": gross > 0 and t_gross >= H9_T_GROSS, "C2": net > 0,
         "C3": diff > 0 and t_diff >= H9_T_DIFF, "C4": p_random < H9_P_RANDOM,
         "C5": all(v > 0 for v in gross_by_market.values())}
    if all(c.values()):
        return "confirmed: candidate for a spec (user must agree)", c
    if c["C1"] and c["C2"] and c["C4"] and not c["C3"]:
        return "rejected: flow continuation, not a level effect", c
    return "rejected", c
