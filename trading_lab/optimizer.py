"""Optimiser that feeds the screening funnel - search broadly, judge honestly.

The question this module answers: *which settings of a strategy (entry, filters,
stop, target) give the best expectancy - and is that best result more than the
luck of having tried many settings?*

How it differs from the evolution workers (``score.py`` hill-climb):

* **Objective = expectancy, not a composite.** ``score.py`` gives trade
  frequency 25 % weight and uses a per-trade Sharpe scaled by a constant; a
  climb on it finds "3-5 trades a week", not an edge. Here the objective is the
  **lower confidence bound of the expectancy per trade** (``mean R - 1 se``), net
  of the measured bid/ask costs. The LCB prefers a solid 0.15 R over a lucky
  0.40 R from 30 trades - "best R, but not by chance" at the level of one variant.
  The R:R itself is a *parameter* (target, stop), never the objective: a
  maximised R:R just finds rare big winners.
* **Train only.** The search never sees the holdout window.
* **Every trial is counted.** Distinct results (settings with identical trades
  count once) give ``n_trials``; their mean correlation gives the *effective*
  number of independent trials ``n_eff``, which sets the S4 bar (Deflated Sharpe,
  ``screening.deflated_sharpe``). The trial matrix also feeds PBO
  (``screening.pbo``). That is "not by chance" at the level of the search.
* **A plateau, not a spike.** Every shortlisted variant gets its grid neighbours
  (one step up/down per parameter) evaluated; S3 wants most of them positive.
* **Placebo by replay.** Every trade is replayed on the 1-minute mid in its own
  direction and in the opposite one, with the same entry time, stop distance,
  target distance and one fixed time cap (the median duration). The random-direction placebo is the
  average of the two - exact, no seed noise. ``placebo_edge`` = how much the
  signal's choice of side adds over a coin flip through the very same exits.

Output: one funnel row per shortlisted variant (``screening.run_funnel`` input),
plus the trial log.

    spec = Spec(name="momentum", asset="US100", base=MomentumConfig(), grid=GRID,
                prepare=lambda m1: m1, run=lambda cfg, d: run_backtest(d, cfg)["trades"])
    result = optimise(spec, m1, holdout_start=pd.Timestamp("2025-01-01", tz="UTC"))
    rows = result.rows                       # -> screening.run_funnel(pd.DataFrame(rows))
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import screening

MIN_TRADES = 100          # train trades below this: the variant is not ranked
LCB_SE = 1.0              # objective = mean - LCB_SE * se
OOS_MIN_TRADES = 20


# ── Strategy description ──────────────────────────────────────────────────

@dataclass
class Spec:
    """What the optimiser needs to know about one strategy on one instrument.

    ``prepare`` turns a 1-minute bid/ask slice into whatever ``run`` consumes
    (the 1-minute frame itself, a resampled frame, prepared sessions...).
    ``run`` returns the raw trade log. ``risk_pct`` is required for strategies
    whose log only carries account % (``pnl_pct``): R = pnl_pct / risk_pct.
    """

    name: str
    asset: str
    base: Any
    grid: dict[str, tuple]
    prepare: Callable[[pd.DataFrame], Any]
    run: Callable[[Any, Any], list[dict]]
    risk_pct: float | None = None
    valid: Callable[[Any], bool] = lambda cfg: True
    #: minutes from the logged ``entry_time`` to the actual fill. Bar strategies
    #: that fill at the CLOSE of a bar but stamp the bar's OPEN time need the bar
    #: length here; without it the placebo replay would start before the signal
    #: existed and "foresee" the breakout (found on bb_squeeze: placebo edge +1.2 R).
    entry_offset_min: int = 0
    #: False for engines that fill on the bid side only (no ask): their logged R is
    #: then a mid-like GROSS number, and ``normalise`` charges the measured spread
    #: itself. Found 2026-10-26: supertrend, bb_squeeze and orb_classic never paid
    #: the spread, so their "net" was gross.
    charges_spread: bool = True
    #: venue cost profile (``trading_lab.costs``): spread scaling + commission per side
    cost_profile: str = "ftmo"

    @property
    def label(self) -> str:
        return f"{self.name}:{self.asset}"


# ── Trade normalisation ───────────────────────────────────────────────────

def _ts(v: Any) -> pd.Timestamp:
    if isinstance(v, (int, float, np.integer, np.floating)):
        return pd.Timestamp(float(v), unit="s", tz="UTC")
    t = pd.Timestamp(v)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _direction(v: Any) -> int:
    if isinstance(v, str):
        return 1 if v.lower().startswith("l") else -1
    return 1 if float(v) > 0 else -1


def _ns(x: Any) -> np.ndarray:
    """Epoch nanoseconds of a DatetimeIndex / datetime Series, whatever its unit.

    The cached minute frames carry a millisecond index, trade logs come in micro- or
    nanoseconds; ``asi8`` / ``astype("int64")`` return the raw integers in the
    column's OWN unit, so comparing them across sources silently mis-indexes.
    """
    if isinstance(x, pd.DatetimeIndex):
        return x.as_unit("ns").asi8
    return pd.to_datetime(x, utc=True).dt.as_unit("ns").astype("int64").to_numpy()


def normalise(trades: list[dict], m1: pd.DataFrame, risk_pct: float | None,
              entry_offset_min: int = 0, charges_spread: bool = True, *,
              asset: str | None = None, cost_profile: str | None = None) -> pd.DataFrame:
    """One row per closed trade with ``entry_time, exit_time, direction, risk,
    target_dist, r, r_gross`` - the same columns for every strategy.

    ``r_gross``: taken from the log when it has one, otherwise ``r`` plus the
    measured half spread at entry and at exit, in R. Slippage/commission stay in;
    "gross" here means "before the bid/ask spread", which is the cost S2 judges.
    """
    rows = []
    for t in trades or []:
        if t.get("status") not in (None, "closed"):
            continue
        entry = t.get("entry", t.get("entry_price"))
        stop = t.get("stop", t.get("stop_loss"))
        if entry is None or stop is None or t.get("entry_time") is None or t.get("exit_time") is None:
            continue
        risk = abs(float(entry) - float(stop))
        # A Position log stores the stop as it was at the END: after a partial
        # take-profit it sits at entry (break-even), so it no longer tells the
        # initial risk. RiskManager sizes size = risk_per_trade / (stop_dist / entry),
        # which gives the initial distance back. Dropping these rows instead would
        # remove exactly the winning trades (it did, in the first draft).
        if risk_pct and t.get("size"):
            initial = float(entry) * (risk_pct / 100.0) / float(t["size"])
            if t.get("sl_moved_to_be") or not risk > 0:
                risk = initial
        if not risk > 0:
            continue
        if t.get("r") is not None and t.get("pnl_pct") is None:
            r = float(t["r"])
        elif t.get("pnl_pct") is not None and risk_pct:
            r = float(t["pnl_pct"]) / risk_pct
        elif t.get("r") is not None:
            r = float(t["r"])
        else:
            continue
        target = t.get("target")
        if target is None:
            target = t.get("tp2") if t.get("tp2") is not None else t.get("tp1")
        rows.append({
            "entry_time": _ts(t["entry_time"]), "exit_time": _ts(t["exit_time"]),
            "direction": _direction(t.get("direction", 1)),
            "entry": float(entry), "risk": risk,
            "target_dist": abs(float(target) - float(entry)) if target is not None else np.nan,
            "r": r, "r_gross": float(t["r_gross"]) if t.get("r_gross") is not None else np.nan,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if entry_offset_min:
        df["entry_time"] = df["entry_time"] + pd.Timedelta(minutes=entry_offset_min)
        df["exit_time"] = df[["exit_time", "entry_time"]].max(axis=1)
    spread = (m1["ask_close"] - m1["close"]).to_numpy()
    idx = _ns(m1.index)
    ie = np.clip(np.searchsorted(idx, _ns(df["entry_time"]), "right") - 1, 0, len(idx) - 1)
    ix = np.clip(np.searchsorted(idx, _ns(df["exit_time"]), "right") - 1, 0, len(idx) - 1)
    half = 0.5 * (spread[ie] + spread[ix]) / df["risk"].to_numpy()   # one spread per round trip, in R
    if not charges_spread:
        # bid-only engine: what it logged is gross; net = gross - one round-trip spread
        df["r_gross"] = df["r"]
        df["r"] = df["r"] - half
    else:
        need = df["r_gross"].isna()
        if need.any():
            df.loc[need, "r_gross"] = df.loc[need, "r"] + half[need.to_numpy()]
    if asset is not None and cost_profile is not None:
        # commission, two sides, in R - the part the old model never charged
        from . import costs
        bp = np.array([costs.commission_bp_side(asset, e, cost_profile) for e in df["entry"]])
        df["commission_r"] = 2 * bp * 1e-4 * df["entry"].to_numpy() / df["risk"].to_numpy()
        df["r"] = df["r"] - df["commission_r"]
    return df


def run_variant(spec: "Spec", cfg: Any, m1: pd.DataFrame, *, adjusted: bool = False) -> pd.DataFrame:
    """One variant, start to finish, with the spec's full cost model: venue spread
    (``costs.adjust_frame``), the engine, then ``normalise`` with offset, bid-only
    spread and commission. Every caller (guided check, portfolios, reports) uses this,
    so a cost rule can never be applied in one place and forgotten in another."""
    from . import costs
    if not adjusted:
        m1 = costs.adjust_frame(m1, spec.asset, spec.cost_profile)
    return normalise(spec.run(cfg, spec.prepare(m1)), m1, spec.risk_pct, spec.entry_offset_min,
                     spec.charges_spread, asset=spec.asset, cost_profile=spec.cost_profile)


# ── Placebo by replay ─────────────────────────────────────────────────────

def replay_placebo(trades: pd.DataFrame, m1: pd.DataFrame) -> tuple[float, float]:
    """``(own_direction, random_direction)`` mean gross R, replayed on the mid.

    Same entry time, stop distance and target distance as the real trade, and one
    **fixed** time cap for all trades: the median duration of the log. The trade's
    own exit time is deliberately *not* used - for a trailing exit it encodes where
    the move turned, which would leak into the comparison. The stop wins a bar
    where both are touched (repo convention). The random direction is the exact
    average of long and short, so there is no seed noise.
    """
    if trades.empty:
        return float("nan"), float("nan")
    cap = (trades["exit_time"] - trades["entry_time"]).median()
    cap_ns = max(int(cap.value), 60_000_000_000)
    mid_h = ((m1["high"] + m1["ask_high"]) / 2).to_numpy()
    mid_l = ((m1["low"] + m1["ask_low"]) / 2).to_numpy()
    mid_c = ((m1["close"] + m1["ask_close"]) / 2).to_numpy()
    idx = _ns(m1.index)
    t0 = _ns(trades["entry_time"])
    i0 = np.searchsorted(idx, t0, "left")
    i1 = np.searchsorted(idx, t0 + cap_ns, "right") - 1
    own, rnd = [], []
    for k in range(len(trades)):
        a, b = int(i0[k]), int(max(i1[k], i0[k]))
        if a >= len(idx):
            continue
        b = min(b, len(idx) - 1)
        e = mid_c[a]
        risk = float(trades["risk"].iat[k])
        tgt = float(trades["target_dist"].iat[k])
        res = {}
        for d in (1, -1):
            hi, lo = mid_h[a + 1:b + 1], mid_l[a + 1:b + 1]
            adverse = (e - lo) if d > 0 else (hi - e)
            favour = (hi - e) if d > 0 else (e - lo)
            s_hit = np.flatnonzero(adverse >= risk)
            t_hit = np.flatnonzero(favour >= tgt) if np.isfinite(tgt) and tgt > 0 else np.array([], int)
            s0 = s_hit[0] if len(s_hit) else np.inf
            t0 = t_hit[0] if len(t_hit) else np.inf
            if s0 <= t0 and np.isfinite(s0):
                res[d] = -1.0
            elif np.isfinite(t0):
                res[d] = tgt / risk
            else:
                res[d] = d * (mid_c[b] - e) / risk
        own.append(res[int(trades["direction"].iat[k])])
        rnd.append(0.5 * (res[1] + res[-1]))
    return float(np.mean(own)), float(np.mean(rnd))


# ── Statistics of one trade log ───────────────────────────────────────────

def lcb(r: np.ndarray, k: float = LCB_SE) -> float:
    r = r[np.isfinite(r)]
    if len(r) < MIN_TRADES:
        return -math.inf
    return float(r.mean() - k * r.std(ddof=1) / math.sqrt(len(r)))


def daily_on_calendar(trades: pd.DataFrame, days: pd.DatetimeIndex, col: str = "r") -> pd.Series:
    """Daily R sum on a fixed calendar, zero on days without a trade.

    A common calendar makes the Sharpes of different trials comparable (DSR) and
    is what an account experiences - flat days count.
    """
    if trades.empty:
        return pd.Series(0.0, index=days)
    d = trades["exit_time"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    return trades.groupby(d.to_numpy())[col].sum().reindex(days, fill_value=0.0)


def per_year(trades: pd.DataFrame, col: str = "r") -> pd.Series:
    return trades.groupby(trades["entry_time"].dt.year)[col].sum() if len(trades) else pd.Series(dtype=float)


def monthly(trades: pd.DataFrame, col: str = "r") -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    m = trades["exit_time"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M")
    return trades.groupby(m)[col].sum()


def max_drawdown_r(trades: pd.DataFrame, col: str = "r") -> float:
    if trades.empty:
        return 0.0
    eq = trades.sort_values("exit_time")[col].cumsum().to_numpy()
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    return float((peak - eq).max())


def prop_profile(trades: pd.DataFrame, *, months: float, target_month_pct: float = 1.5,
                 max_dd_pct: float = 10.0, max_day_pct: float = 5.0) -> dict:
    """What it takes to reach ``target_month_pct`` per month with this trade log.

    ``risk_for_target`` = risk per trade (in % of the account) at which the mean
    month returns the target; the drawdown and worst day are then scaled by it and
    checked against typical prop limits (10 % total, 5 % daily).
    """
    if trades.empty or months <= 0:
        return {}
    r_month = float(trades["r"].sum()) / months
    dd = max_drawdown_r(trades)
    d = trades["exit_time"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    worst_day = float(-trades.groupby(d.to_numpy())["r"].sum().min())
    mon = monthly(trades)
    out = {"r_per_month": round(r_month, 3), "max_dd_r": round(dd, 2),
           "worst_day_r": round(worst_day, 2), "pos_months": round(float((mon > 0).mean()), 3)
           if len(mon) else None, "trades_per_month": round(len(trades) / months, 1)}
    if r_month > 0:
        risk = target_month_pct / r_month
        out.update({"risk_for_target_pct": round(risk, 3),
                    "dd_at_target_pct": round(dd * risk, 2),
                    "worst_day_at_target_pct": round(worst_day * risk, 2),
                    "prop_ok": bool(dd * risk <= max_dd_pct and worst_day * risk <= max_day_pct)})
    else:
        out["prop_ok"] = False
    return out


# ── The search ────────────────────────────────────────────────────────────

@dataclass
class Trial:
    key: tuple
    cfg: Any
    trades: pd.DataFrame
    n: int
    net_mean: float
    gross_mean: float
    objective: float
    seconds: float


@dataclass
class Result:
    spec: Spec
    trials: dict[tuple, Trial]
    rows: list[dict]
    pbo: float
    n_trials: int
    n_eff: float = float("nan")
    note: str = ""

    def trial_log(self) -> list[dict]:
        return [{"params": dict(k), "n": t.n, "net_mean": round(t.net_mean, 4),
                 "gross_mean": round(t.gross_mean, 4),
                 "objective": None if not np.isfinite(t.objective) else round(t.objective, 4)}
                for k, t in self.trials.items()]


def _outcome(t: "Trial") -> tuple:
    """Fingerprint of a trial's result - equal trades, equal fingerprint."""
    if t.trades.empty:
        return (0,)
    return (t.n, round(float(t.trades["r"].sum()), 6), round(float(t.trades["r_gross"].sum()), 6))


def effective_trials(daily: pd.DataFrame) -> float:
    """Number of *independent* trials: ``rho + (1 - rho) * N`` (Bailey & Lopez de Prado).

    ``rho`` is the mean pairwise correlation of the trials' daily series. A grid
    of 200 near-identical settings is not 200 chances to get lucky; this measures
    how many it really was.
    """
    n = daily.shape[1]
    if n <= 1:
        return float(max(n, 1))
    c = daily.corr().to_numpy()
    off = c[~np.eye(n, dtype=bool)]
    off = off[np.isfinite(off)]
    rho = float(np.clip(off.mean(), 0.0, 1.0)) if off.size else 0.0
    return rho + (1.0 - rho) * n


def _key(cfg: Any, grid: dict[str, tuple]) -> tuple:
    d = asdict(cfg)
    return tuple((k, d[k]) for k in sorted(grid))


def _neighbours(cfg: Any, grid: dict[str, tuple]) -> list[Any]:
    out = []
    d = asdict(cfg)
    for p, values in grid.items():
        vals = list(values)
        if d[p] not in vals:
            continue
        i = vals.index(d[p])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                out.append(replace(cfg, **{p: vals[j]}))
    return out


class _Search:
    def __init__(self, spec: Spec, m1_train: pd.DataFrame, *, budget: int, verbose: bool):
        from . import costs
        m1_train = costs.adjust_frame(m1_train, spec.asset, spec.cost_profile)
        self.spec, self.m1, self.budget, self.verbose = spec, m1_train, budget, verbose
        self.data = spec.prepare(m1_train)
        self.trials: dict[tuple, Trial] = {}
        self.errors = 0
        self.progress: Callable[[int, int], None] | None = None

    def eval(self, cfg: Any) -> Trial | None:
        k = _key(cfg, self.spec.grid)
        if k in self.trials:
            return self.trials[k]
        if not self.spec.valid(cfg):
            return None
        t0 = time.time()
        try:
            raw = self.spec.run(cfg, self.data)
        except Exception as exc:          # a bad corner of the grid must not kill the search
            self.errors += 1
            if self.verbose and self.errors <= 3:
                print(f"    [{self.spec.label}] Fehler bei {dict(k)}: {exc!r}")
            return None
        tr = normalise(raw, self.m1, self.spec.risk_pct, self.spec.entry_offset_min,
                       self.spec.charges_spread, asset=self.spec.asset,
                       cost_profile=self.spec.cost_profile)
        r = tr["r"].to_numpy() if len(tr) else np.array([])
        g = tr["r_gross"].to_numpy() if len(tr) else np.array([])
        trial = Trial(key=k, cfg=cfg, trades=tr, n=len(tr),
                      net_mean=float(r.mean()) if len(r) else float("nan"),
                      gross_mean=float(g.mean()) if len(g) else float("nan"),
                      objective=lcb(r), seconds=time.time() - t0)
        self.trials[k] = trial
        if self.progress is not None:
            self.progress(len(self.trials), self.budget)
        return trial

    def random_cfg(self, rng: random.Random) -> Any:
        return replace(self.spec.base, **{p: rng.choice(list(v)) for p, v in self.spec.grid.items()})

    def run(self, seed: int = 7) -> None:
        rng = random.Random(seed)
        n_random = max(8, self.budget // 3)
        self.eval(self.spec.base)
        tries = 0
        while len(self.trials) < n_random and tries < n_random * 5:
            self.eval(self.random_cfg(rng))
            tries += 1
        # coordinate descent from the best few starting points
        starts = sorted((t for t in self.trials.values() if np.isfinite(t.objective)),
                        key=lambda t: -t.objective)[:3]
        for start in starts:
            cur = start
            improved = True
            while improved and len(self.trials) < self.budget:
                improved = False
                params = list(self.spec.grid)
                rng.shuffle(params)
                for p in params:
                    for v in self.spec.grid[p]:
                        if len(self.trials) >= self.budget:
                            break
                        t = self.eval(replace(cur.cfg, **{p: v}))
                        if t is not None and t.objective > cur.objective + 1e-9:
                            cur, improved = t, True
            if self.verbose:
                print(f"    [{self.spec.label}] Abstieg fertig: LCB {cur.objective:+.3f} "
                      f"netto {cur.net_mean:+.3f} R n={cur.n}  ({len(self.trials)} Trials)", flush=True)


def search(spec: Spec, m1_train: pd.DataFrame, *, budget: int = 150,
           verbose: bool = True, progress: Callable[[int, int], None] | None = None) -> "_Search":
    """Phase 1: the search on train (random starts + coordinate descent)."""
    s = _Search(spec, m1_train, budget=budget, verbose=verbose)
    s.progress = progress
    s.run()
    return s


def shortlist(s: "_Search", *, top_net: int = 8, top_gross: int = 4) -> list[Trial]:
    """Phase 2: best DISTINCT results by net LCB (+ a few by gross), plus their
    grid neighbours evaluated (needed for S3; they are trials too)."""
    uniq = list({_outcome(t): t for t in s.trials.values()}.values())
    ranked = [t for t in uniq if np.isfinite(t.objective)]
    by_net = sorted(ranked, key=lambda t: -t.objective)[:top_net]
    by_gross = sorted((t for t in uniq if t.n >= MIN_TRADES and np.isfinite(t.gross_mean)),
                      key=lambda t: -(t.gross_mean - t.trades["r_gross"].std(ddof=1) / math.sqrt(t.n)))[:top_gross]
    short = list({_outcome(t): t for t in by_net + by_gross}.values())
    for t in short:
        for nb in _neighbours(t.cfg, s.spec.grid):
            s.eval(nb)
    return short


def build_rows(s: "_Search", short: list[Trial]) -> tuple[list[dict], dict]:
    """Phase 3: one funnel row per shortlisted variant (train only).

    Returns ``(rows, search_stats)`` with ``n_trials``, ``n_eff`` and ``pbo``.
    """
    spec, m1_train = s.spec, s.m1
    days = pd.bdate_range(m1_train.index[0].tz_convert("UTC").tz_localize(None).normalize(),
                          m1_train.index[-1].tz_convert("UTC").tz_localize(None).normalize())
    # Settings that produce the same trades (a parameter without effect, e.g. a
    # regime threshold with regime="none") are ONE trial, not several.
    distinct: dict = {}
    for t in s.trials.values():
        distinct.setdefault(_outcome(t), t)
    usable = [t for t in distinct.values() if t.n >= 10]
    daily = {t.key: daily_on_calendar(t.trades, days) for t in usable}
    n_trials = len(distinct)
    n_eff = effective_trials(pd.DataFrame(daily))
    mon = pd.DataFrame({t.key: monthly(t.trades) for t in usable}).T.fillna(0.0)
    pbo = screening.pbo(mon) if mon.shape[0] >= 5 else float("nan")
    train_months = max((m1_train.index[-1] - m1_train.index[0]).days / 30.44, 1.0)

    rows = []
    for t in short:
        nbs = [s.trials.get(_key(nb, spec.grid)) for nb in _neighbours(t.cfg, spec.grid)]
        nbs = [x for x in nbs if x is not None and x.n >= 10]
        own, rnd = replay_placebo(t.trades, m1_train)
        placebo_edge = own - rnd if np.isfinite(own) else None
        yrs = per_year(t.trades)
        d = daily.get(t.key, daily_on_calendar(t.trades, days))
        rows.append({
            "name": f"{spec.label}#{len(rows)}", "strategy": spec.name, "asset": spec.asset,
            "cost_profile": spec.cost_profile,
            "params": {k: v for k, v in t.key},
            "n": t.n, "gross_mean": t.gross_mean, "net_mean": t.net_mean,
            "placebo_gross": (t.gross_mean - placebo_edge) if placebo_edge is not None else None,
            "placebo_edge": placebo_edge, "replay_own": own, "replay_random": rnd,
            "gross_bp": None, "cost_bp": None,
            "pos_years": int((yrs > 0).sum()), "n_years": int(len(yrs)),
            "per_year_r": {int(k): round(float(v), 2) for k, v in yrs.items()},
            "neighbour_positive": (float(np.mean([x.net_mean > 0 for x in nbs])) if nbs else None),
            "t_daily": screening.t_stat(d),
            "n_trials": n_trials, "n_eff": round(n_eff, 1),
            "dsr": screening.deflated_sharpe(d, n_trials=max(1, int(round(n_eff)))),
            "pbo": pbo,
            "oos_pass": None,
            "objective_lcb": t.objective,
            "prop_train": prop_profile(t.trades, months=train_months),
        })
    return rows, {"n_trials": n_trials, "n_eff": n_eff, "pbo": pbo}


def evaluate_holdout(spec: Spec, m1: pd.DataFrame, rows: list[dict], names: set[str],
                     *, holdout_start: pd.Timestamp) -> None:
    """Phase 5: the one-shot holdout for the named rows (in place).

    Runs on the full history (indicators keep their warm-up) and keeps only the
    trades that ENTER on/after ``holdout_start``.
    """
    from . import costs
    m1 = costs.adjust_frame(m1, spec.asset, spec.cost_profile)
    full = spec.prepare(m1)
    hold_months = max((m1.index[-1] - holdout_start).days / 30.44, 1.0)
    for row in rows:
        if row["name"] not in names:
            continue
        cfg = replace(spec.base, **row["params"])
        tr = normalise(spec.run(cfg, full), m1, spec.risk_pct, spec.entry_offset_min,
                       spec.charges_spread, asset=spec.asset, cost_profile=spec.cost_profile)
        tr = tr[tr["entry_time"] >= holdout_start] if len(tr) else tr
        r = tr["r"].to_numpy() if len(tr) else np.array([])
        row["oos_n"] = int(len(r))
        row["oos_net_mean"] = float(r.mean()) if len(r) else None
        row["oos_t_daily"] = screening.t_stat(daily_on_calendar(
            tr, pd.bdate_range(holdout_start.tz_localize(None), m1.index[-1].tz_localize(None).normalize())))
        row["oos_pass"] = bool(len(r) >= OOS_MIN_TRADES and r.mean() > 0)
        row["prop_holdout"] = prop_profile(tr, months=hold_months)


def optimise(spec: Spec, m1: pd.DataFrame, *, holdout_start: pd.Timestamp,
             budget: int = 150, top_net: int = 8, top_gross: int = 4,
             holdout_release: str | None = None, verbose: bool = True) -> Result:
    """All phases in one go (batch). The guided check calls them one by one.

    The holdout (``m1`` from ``holdout_start`` on) is only evaluated when
    ``holdout_release`` is given, and only for variants that pass S1-S4 on train.
    """
    m1_train = m1[m1.index < holdout_start]
    s = search(spec, m1_train, budget=budget, verbose=verbose)
    short = shortlist(s, top_net=top_net, top_gross=top_gross)
    rows, st = build_rows(s, short)
    note = ""
    if holdout_release:
        funnel = screening.run_funnel(pd.DataFrame(rows), up_to="S4")
        passed = set(funnel.stages[-1].kept["name"]) if funnel.stages else set()
        if passed:
            evaluate_holdout(spec, m1, rows, passed, holdout_start=holdout_start)
            note = f"Holdout freigegeben ({holdout_release}) fuer {len(passed)} S4-Ueberlebende"
        else:
            note = "Holdout NICHT verbraucht: kein Kandidat hat S4 bestanden"
    return Result(spec=spec, trials=s.trials, rows=rows, pbo=st["pbo"], n_trials=st["n_trials"],
                  n_eff=st["n_eff"], note=note)


def grid_size(grid: dict[str, tuple]) -> int:
    return int(np.prod([len(v) for v in grid.values()]))


__all__ = ["run_variant", "Spec", "Result", "optimise", "search", "shortlist", "build_rows", "evaluate_holdout", "normalise", "replay_placebo", "prop_profile",
           "lcb", "daily_on_calendar", "grid_size"]
