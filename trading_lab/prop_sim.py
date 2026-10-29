"""Prop-firm evaluation simulator — reusable across strategies, with firm presets.

Turns a strategy's per-trade log into a daily P&L stream and asks whether it
passes the firm's evaluation rules, and how the pass rate looks under a
**block-bootstrap Monte Carlo** (blocks of consecutive days, so losing streaks
are preserved). This is a risk-management test, not a new alpha.

Firms differ in three ways that matter:

* **profit target** (and whether there are several phases, e.g. FTMO's
  challenge + verification),
* **daily loss limit** (hard breach),
* **total loss limit**: ``static`` (10 % of the *initial* balance, e.g. FTMO) or
  ``eod_trailing`` (trails the highest end-of-day balance, e.g. Lucid/Apex).

Percentages are always of the account size. The caller supplies ``risk_usd``
(dollar risk per trade); ``risk_usd_for(account_size, risk_pct)`` helps size it.

    from trading_lab.prop_sim import FIRMS, evaluate, daily_pnl, risk_usd_for

    daily = daily_pnl(trades, risk_usd=risk_usd_for(25_000, 0.5), days=trading_days)
    res = evaluate(daily, FIRMS["lucid"], runs=5000)
    print(res.summary())

``intraday_trailing`` is accepted but falls back to ``eod_trailing`` (a
1-minute intrabar equity path is not available here); it is reported as such.

Rule variants of instant / one-step accounts (2026 research, ``docs/PROP_FIRMS.md``):

* ``daily_mode``: ``hard`` (the account is lost), ``pause`` (trading stops for the rest
  of the day, the loss is capped at the limit - The5ers) or ``none`` (FundedNext Instant);
* ``trail_lock_at_initial``: a trailing floor that stops rising once it reaches the
  initial balance (FundedNext / FundingPips wording);
* ``consistency_max_day_frac``: no single day may exceed this share of the profit -
  a payout (instant) or the pass (challenge) waits until it holds;
* ``min_days_before_payout`` and ``instant_price`` (fee, subtracted from the EV).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

DRAWDOWN_TYPES = ("static", "eod_trailing", "intraday_trailing")
DAILY_MODES = ("hard", "pause", "none")
_OUTCOMES = ("pass", "daily", "total", "min_days", "timeout")


@dataclass(frozen=True)
class PropFirm:
    """One firm's rules (percentages of ``account_size``).

    An empty ``phase_targets`` means **instant funding**: there is no evaluation
    to pass — the account is funded from day one and is judged by survival and
    payouts (see ``simulate_funded``).
    """

    name: str
    account_size: float = 25_000.0
    phase_targets: tuple[float, ...] = (0.05,)   # profit target per phase, fraction
    max_daily_loss_frac: float = 0.024
    max_total_loss_frac: float = 0.04
    drawdown_type: str = "eod_trailing"          # static | eod_trailing | intraday_trailing
    min_trading_days: int = 0                    # per phase
    time_limit_days: int | None = 365            # per phase; None = unlimited
    eval_price_per_phase: float = 100.0
    # instant-funding payout model
    payout_split: float = 0.8                    # trader's share of profit at payout
    payout_cycle_days: int = 14                  # how often payouts are taken
    min_profit_before_payout: float = 0.0
    daily_mode: str = "hard"                     # hard | pause | none
    trail_lock_at_initial: bool = False          # trailing floor stops at the initial balance
    consistency_max_day_frac: float | None = None  # best day <= frac x profit (payout / pass gate)
    min_days_before_payout: int = 0              # trading days since the last payout
    instant_price: float = 0.0                   # fee of an instant account (USD)
    verified: bool = False                       # rules checked on the firm's own page?
    note: str = ""

    def __post_init__(self) -> None:
        if self.drawdown_type not in DRAWDOWN_TYPES:
            raise ValueError(f"drawdown_type must be one of {DRAWDOWN_TYPES}")
        if self.daily_mode not in DAILY_MODES:
            raise ValueError(f"daily_mode must be one of {DAILY_MODES}")

    @property
    def is_instant(self) -> bool:
        return not self.phase_targets

    @property
    def daily_limit_usd(self) -> float:
        return self.account_size * self.max_daily_loss_frac

    @property
    def total_limit_usd(self) -> float:
        return self.account_size * self.max_total_loss_frac


# ── Presets ─────────────────────────────────────────────────────────
# FTMO: 2-step (Challenge 10 %, Verification 5 %), daily 5 %, **static** 10 %
# max loss, minimum 4 trading days, no time limit. (2023+ rules; verify with the
# firm before live use.)
FTMO = PropFirm(
    name="FTMO 2-step (challenge 10% + verification 5%)",
    account_size=100_000.0, phase_targets=(0.10, 0.05),
    max_daily_loss_frac=0.05, max_total_loss_frac=0.10, drawdown_type="static",
    min_trading_days=4, time_limit_days=None, eval_price_per_phase=89.0,
    note="static max loss; daily loss resets at midnight CE(S)T; verify current rules",
)

# Lucid-style (from the ORB spec): 1-step, 5 % target, 2.4 % daily, **EOD trailing**
# 4 % total loss.
LUCID = PropFirm(
    name="LucidPro 25K (5%/2.4%/4% EOD trailing)",
    account_size=25_000.0, phase_targets=(0.05,),
    max_daily_loss_frac=0.024, max_total_loss_frac=0.04, drawdown_type="eod_trailing",
    min_trading_days=0, time_limit_days=365, eval_price_per_phase=100.0,
)

# Stylized 1-step: one phase, higher target, same risk limits as FTMO.
ONE_STEP = PropFirm(
    name="1-step (8% target / 5% daily / 10% static)",
    account_size=100_000.0, phase_targets=(0.08,),
    max_daily_loss_frac=0.05, max_total_loss_frac=0.10, drawdown_type="static",
    min_trading_days=0, time_limit_days=None, eval_price_per_phase=99.0,
    note="stylized one-phase variant; confirm exact target/limits with the firm",
)

# Instant funding: no evaluation, funded from day one (static drawdown from the
# initial balance), payouts every cycle on the profit above the starting balance.
INSTANT_25K = PropFirm(
    name="Instant 25K (2.4% daily / 4% static total / 80% split)",
    account_size=25_000.0, phase_targets=(),
    max_daily_loss_frac=0.024, max_total_loss_frac=0.04, drawdown_type="static",
    time_limit_days=None, payout_split=0.8, payout_cycle_days=14,
)

INSTANT_100K = PropFirm(
    name="Instant 100K (5% daily / 8% static total / 80% split)",
    account_size=100_000.0, phase_targets=(),
    max_daily_loss_frac=0.05, max_total_loss_frac=0.08, drawdown_type="static",
    time_limit_days=None, payout_split=0.8, payout_cycle_days=14,
)

# ── 2026 presets from the research in docs/PROP_FIRMS.md ───────────────
# Only FundedNext's numbers come from the firm's own page; the rest from reviews that
# partly disagree (``verified=False``). Fees and splits vary by size - fill in the real
# fee (``instant_price`` / ``eval_price_per_phase``) before reading the EV.
FUNDEDNEXT_INSTANT = PropFirm(
    name="FundedNext Stellar Instant (no daily / 6% trailing, locks at start)",
    account_size=25_000.0, phase_targets=(), daily_mode="none", max_daily_loss_frac=1.0,
    max_total_loss_frac=0.06, drawdown_type="eod_trailing", trail_lock_at_initial=True,
    time_limit_days=None, payout_split=0.8, payout_cycle_days=14, verified=True,
    note="official page 2026-10: no daily limit, 6% trailing max loss, no consistency; split/fee: check",
)
THE5ERS_HYPER_GROWTH = PropFirm(
    name="The5ers Hyper Growth (10% target / 3% daily pause / 6% static)",
    account_size=20_000.0, phase_targets=(0.10,), daily_mode="pause", max_daily_loss_frac=0.03,
    max_total_loss_frac=0.06, drawdown_type="static", time_limit_days=None, eval_price_per_phase=0.0,
    payout_split=0.5,
    note="reviews 2026: daily 3% is a pause per most sources (one says termination); fee: check",
)
FUNDINGPIPS_ZERO = PropFirm(
    name="FundingPips Zero (3% daily / 5% trailing / 15% consistency)",
    account_size=25_000.0, phase_targets=(), daily_mode="hard", max_daily_loss_frac=0.03,
    max_total_loss_frac=0.05, drawdown_type="eod_trailing", trail_lock_at_initial=True,
    consistency_max_day_frac=0.15, min_days_before_payout=7, time_limit_days=None,
    payout_split=0.95, payout_cycle_days=14,
    note="reviews 2026, partly contradictory (trailing mechanics, weekend ban unconfirmed)",
)
FTMO_1STEP = PropFirm(
    name="FTMO 1-Step (10% target / 3% daily / 10% EOD trailing / best day 50%)",
    account_size=100_000.0, phase_targets=(0.10,), daily_mode="hard", max_daily_loss_frac=0.03,
    max_total_loss_frac=0.10, drawdown_type="eod_trailing", consistency_max_day_frac=0.5,
    time_limit_days=None, eval_price_per_phase=0.0,
    note="reviews 2026 (one says 5% daily); fee: check",
)

FIRMS: dict[str, PropFirm] = {
    "ftmo": FTMO, "lucid": LUCID, "one_step": ONE_STEP,
    "instant_25k": INSTANT_25K, "instant_100k": INSTANT_100K,
    "fundednext_instant": FUNDEDNEXT_INSTANT, "the5ers_hyper_growth": THE5ERS_HYPER_GROWTH,
    "fundingpips_zero": FUNDINGPIPS_ZERO, "ftmo_1step": FTMO_1STEP,
}


def get_firm(name: str) -> PropFirm:
    key = name.strip().lower()
    if key not in FIRMS:
        raise KeyError(f"unknown firm {name!r}; known: {sorted(FIRMS)}")
    return FIRMS[key]


def risk_usd_for(account_size: float, risk_pct: float) -> float:
    """Dollar risk per trade from a percent-of-account figure (``0.5`` = 0.5 %)."""
    return account_size * risk_pct / 100.0


# ── Daily stream ────────────────────────────────────────────────────


def daily_pnl(trades: Iterable[dict], *, risk_usd: float, days: Sequence | None = None) -> np.ndarray:
    """Per-trading-day P&L in USD from a trade log (``r`` × ``risk_usd``).

    ``days`` (an ordered sequence of trading days) fixes the calendar; days with
    no trade contribute 0 so daily limits, the minimum-day rule and the time
    limit are honest. Without ``days`` only days that traded appear.
    """
    by_day: dict[str, float] = {}
    for t in trades:
        by_day[t["day"]] = by_day.get(t["day"], 0.0) + float(t["r"]) * risk_usd
    if days is None:
        return np.array([by_day[k] for k in sorted(by_day)], dtype=float)
    return np.array([by_day.get(str(getattr(d, "date", lambda: d)()), 0.0) for d in days], dtype=float)


# ── Shared rule helpers ─────────────────────────────────────────────


def apply_daily(pnl: float, firm: PropFirm) -> tuple[float, bool]:
    """(effective day P&L, account lost?) under the firm's daily rule."""
    if firm.daily_mode == "none":
        return pnl, False
    if pnl <= -firm.daily_limit_usd:
        if firm.daily_mode == "pause":
            return -firm.daily_limit_usd, False      # trading stops at the limit
        return pnl, True
    return pnl, False


def floor_of(firm: PropFirm, peak: float) -> float:
    """Balance at which the total-loss rule breaches."""
    if firm.drawdown_type == "static":
        return firm.account_size - firm.total_limit_usd
    floor = peak - firm.total_limit_usd
    if firm.trail_lock_at_initial:
        floor = min(floor, firm.account_size)
    return floor


def consistent(best_day: float, profit: float, firm: PropFirm) -> bool:
    frac = firm.consistency_max_day_frac
    return frac is None or (profit > 0 and best_day <= frac * profit)


# ── One evaluation phase ────────────────────────────────────────────


def run_phase(daily: np.ndarray, firm: PropFirm, target_frac: float) -> tuple[str, int, int]:
    """Replay one phase over ``daily``.

    Returns ``(outcome, days_used, trading_days)``; ``outcome`` is one of
    ``pass`` / ``daily`` / ``total`` / ``min_days`` / ``timeout``.
    """
    balance = firm.account_size
    peak = firm.account_size
    target = firm.account_size * target_frac
    limit_days = firm.time_limit_days if firm.time_limit_days is not None else len(daily)
    trading_days = 0
    best_day = 0.0
    for i, raw in enumerate(daily[:limit_days], start=1):
        pnl, lost = apply_daily(float(raw), firm)
        balance += pnl
        if pnl != 0:
            trading_days += 1
        best_day = max(best_day, pnl)
        if lost:
            return "daily", i, trading_days
        peak = max(peak, balance)                   # eod_trailing (intraday falls back to it)
        if balance <= floor_of(firm, peak):
            return "total", i, trading_days
        # min_trading_days / consistency are completion gates: the target only counts once
        # they hold (we keep going, and can still fail).
        profit = balance - firm.account_size
        if profit >= target and trading_days >= firm.min_trading_days and consistent(best_day, profit, firm):
            return "pass", i, trading_days
    return "timeout", min(len(daily), limit_days), trading_days


# ── Full evaluation (all phases) ────────────────────────────────────


@dataclass(frozen=True)
class PropResult:
    firm: str
    attempts: int
    passes: int
    pass_rate: float
    phase_pass_rate: tuple[float, ...]
    median_days: float
    p95_days: float
    fails: dict[str, int] = field(default_factory=dict)
    cost_per_funded: float = float("inf")
    worst_day: float = 0.0

    def summary(self) -> str:
        phases = " ".join(f"p{i + 1}={r * 100:.0f}%" for i, r in enumerate(self.phase_pass_rate))
        fails = " ".join(f"{k}:{v}" for k, v in sorted(self.fails.items()) if v)
        return (
            f"{self.firm} | overall pass {self.pass_rate * 100:.1f}% ({self.passes}/{self.attempts}) "
            f"[{phases}] | median {self.median_days:.0f}d p95 {self.p95_days:.0f}d | "
            f"fails {fails or '—'} | cost/funded ${self.cost_per_funded:,.0f} | "
            f"worst day ${self.worst_day:,.0f}"
        )


def _attempt(path: np.ndarray, firm: PropFirm) -> dict:
    """Run all phases sequentially on one path; each phase starts fresh.

    Returns ``{passed, days, reason, reached, passed_phases}``.
    """
    total_days = 0
    reached: list[bool] = []
    passed_phases: list[bool] = []
    for idx, target in enumerate(firm.phase_targets):
        outcome, days, _ = run_phase(path[total_days:], firm, target)
        total_days += days
        reached.append(True)
        passed_phases.append(outcome == "pass")
        if outcome != "pass":
            reached.extend([False] * (len(firm.phase_targets) - idx - 1))
            passed_phases.extend([False] * (len(firm.phase_targets) - idx - 1))
            return {"passed": False, "days": total_days, "reason": outcome,
                    "reached": reached, "passed_phases": passed_phases}
    return {"passed": True, "days": total_days, "reason": "pass",
            "reached": reached, "passed_phases": passed_phases}


@dataclass(frozen=True)
class FundedResult:
    firm: str
    attempts: int
    survival_rate: float            # fraction that never breached within the horizon
    breaches: int
    median_days_to_breach: float
    payout_rate: float              # fraction of attempts that took ≥1 payout
    mean_payout_usd: float          # average total payout per account
    median_payout_usd: float
    ev_per_account_usd: float       # mean payout (instant accounts cost nothing to open)
    worst_day: float

    def summary(self) -> str:
        return (
            f"{self.firm} | survive {self.survival_rate * 100:.1f}% ({self.attempts - self.breaches}/"
            f"{self.attempts}) | breach median {self.median_days_to_breach:.0f}d | "
            f"payout in {self.payout_rate * 100:.1f}% of runs | "
            f"mean payout ${self.mean_payout_usd:,.0f} (median ${self.median_payout_usd:,.0f}) | "
            f"EV/account ${self.ev_per_account_usd:,.0f}"
        )


def _one_funded(daily: np.ndarray, firm: PropFirm, horizon: int) -> tuple[bool, int, float]:
    """Run one instant account over ``daily`` (up to ``horizon`` days).

    Returns ``(survived, days_until_breach_or_horizon, total_payout)``. Payouts are
    taken every ``payout_cycle_days`` on the profit above the initial balance and
    reset the balance to the initial amount (a simple, documented model).
    """
    balance = firm.account_size
    peak = firm.account_size
    payout = 0.0
    best_day, days_since = 0.0, 0
    for i, raw in enumerate(daily[:horizon], start=1):
        pnl, lost = apply_daily(float(raw), firm)
        balance += pnl
        if pnl != 0:
            days_since += 1
        best_day = max(best_day, pnl)
        if lost:
            return False, i, payout
        peak = max(peak, balance)
        if balance <= floor_of(firm, peak):
            return False, i, payout
        if firm.payout_cycle_days and i % firm.payout_cycle_days == 0:
            profit = balance - firm.account_size
            if (profit > 0 and profit >= firm.min_profit_before_payout
                    and days_since >= firm.min_days_before_payout and consistent(best_day, profit, firm)):
                payout += firm.payout_split * profit
                balance = firm.account_size          # reset after a payout
                peak = balance
                best_day, days_since = 0.0, 0
    return True, min(len(daily), horizon), payout


def simulate_funded(daily: np.ndarray, firm: PropFirm, *, runs: int = 5000,
                    block: int = 20, horizon_days: int = 252, seed: int = 7) -> FundedResult:
    """Instant-funding account: survival rate and expected payout over a horizon."""
    rng = np.random.default_rng(seed)
    n = len(daily)
    worst_day = float(daily.min()) if n else 0.0
    out: list[tuple[bool, int, float]] = [(_one_funded(daily, firm, horizon_days))]

    def _padded(a: np.ndarray) -> np.ndarray:
        return a[:horizon_days] if len(a) >= horizon_days else np.concatenate(
            [a, np.zeros(horizon_days - len(a))])

    out[0] = (_one_funded(_padded(daily), firm, horizon_days))
    if n:
        n_blocks = int(np.ceil(horizon_days / block)) + 1
        for _ in range(runs):
            starts = rng.integers(0, n, size=n_blocks)
            path = np.concatenate([daily[s:s + block] for s in starts])[:horizon_days]
            out.append(_one_funded(path, firm, horizon_days))

    attempts = len(out)
    breached = [d for ok, d, _ in out if not ok]
    payouts = np.array([p for _, _, p in out])
    return FundedResult(
        firm=firm.name, attempts=attempts,
        survival_rate=1 - len(breached) / attempts,
        breaches=len(breached),
        median_days_to_breach=float(np.median(breached)) if breached else float("nan"),
        payout_rate=float((payouts > 0).mean()),
        mean_payout_usd=float(payouts.mean()),
        median_payout_usd=float(np.median(payouts)),
        ev_per_account_usd=float(payouts.mean()) - firm.instant_price,
        worst_day=worst_day,
    )


def evaluate(daily: np.ndarray, firm: PropFirm, *, runs: int = 5000,
             block: int = 20, horizon_days: int | None = None, seed: int = 7) -> PropResult:
    """Historical replay + block-bootstrap Monte Carlo over all phases.

    ``horizon_days`` caps an unlimited-time-limit firm's path length for the
    Monte Carlo (default: two years of trading days).
    """
    rng = np.random.default_rng(seed)
    n = len(daily)
    if horizon_days is None:
        horizon_days = firm.time_limit_days if firm.time_limit_days is not None else 504
    worst_day = float(daily.min()) if n else 0.0

    def padded(a: np.ndarray, length: int) -> np.ndarray:
        return a[:length] if len(a) >= length else np.concatenate([a, np.zeros(length - len(a))])

    results: list[dict] = []
    # historical replay (phases over the observed days, padded to the horizon)
    results.append(_attempt(padded(daily, horizon_days), firm))
    # Monte Carlo: consecutive-day blocks across the whole horizon
    if n:
        n_blocks = int(np.ceil(horizon_days / block)) + 1
        for _ in range(runs):
            starts = rng.integers(0, n, size=n_blocks)
            path = np.concatenate([daily[s:s + block] for s in starts])[:horizon_days]
            results.append(_attempt(path, firm))

    attempts = len(results)
    passing = [r["days"] for r in results if r["passed"]]
    rate = len(passing) / attempts if attempts else 0.0
    fails: dict[str, int] = {}
    for r in results:
        if not r["passed"]:
            fails[r["reason"]] = fails.get(r["reason"], 0) + 1
    # per-phase rate: P(pass phase i | reached phase i)
    phase_rate: list[float] = []
    for idx in range(len(firm.phase_targets)):
        reached = sum(1 for r in results if r["reached"][idx])
        passed = sum(1 for r in results if r["passed_phases"][idx])
        phase_rate.append(passed / reached if reached else 0.0)
    price = firm.eval_price_per_phase * len(firm.phase_targets)
    return PropResult(
        firm=firm.name, attempts=attempts, passes=len(passing), pass_rate=rate,
        phase_pass_rate=tuple(phase_rate),
        median_days=float(np.median(passing)) if passing else float("nan"),
        p95_days=float(np.percentile(passing, 95)) if passing else float("nan"),
        fails=fails, cost_per_funded=(price / rate) if rate > 0 else float("inf"),
        worst_day=worst_day,
    )


# ── Risk sweep over firms (used by studies/prop_firms.py and the reports) ─────

def trade_log_days(trades) -> tuple[list[dict], list]:
    """(day log, business-day calendar) from a normalised trade frame (``exit_time``, ``r``)."""
    import pandas as pd
    if len(trades) == 0:
        return [], []
    ex = pd.to_datetime(trades["exit_time"], utc=True).dt.tz_localize(None)
    en = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_localize(None)
    log = [{"day": str(t.date()), "r": float(r)} for t, r in zip(ex, trades["r"])]
    days = [d.date() for d in pd.bdate_range(en.min().normalize(), ex.max().normalize())]
    return log, days


def risk_sweep(trades, firms: Sequence[str], risks: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
               *, runs: int = 2000, seed: int = 7) -> list[dict]:
    """One row per (firm, risk % per trade): pass rate / survival and payouts."""
    log, days = trade_log_days(trades)
    rows = []
    for name in firms:
        firm = get_firm(name)
        for risk in risks:
            daily = daily_pnl(log, risk_usd=risk_usd_for(firm.account_size, risk), days=days)
            row = {"firm": name, "risk_pct": risk, "verified": firm.verified}
            if firm.is_instant:
                fr = simulate_funded(daily, firm, runs=runs, seed=seed)
                row.update({"type": "instant", "survive_1y": fr.survival_rate, "payout_rate": fr.payout_rate,
                            "mean_payout_1y_usd": fr.mean_payout_usd, "ev_1y_usd": fr.ev_per_account_usd,
                            "payout_pct_of_account": 100 * fr.mean_payout_usd / firm.account_size,
                            "fee_known": firm.instant_price > 0})
            else:
                pr = evaluate(daily, firm, runs=runs, seed=seed)
                row.update({"type": "challenge", "pass_rate": pr.pass_rate, "median_days": pr.median_days,
                            "fails": pr.fails})
            rows.append(row)
    return rows
