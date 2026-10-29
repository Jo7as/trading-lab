"""Staged screening: loose first, strict later. Never the other way round.

What was wrong before
---------------------
The repo ran its gates as a **wall**: the S0 cost quote was computed *before* any
test (`hypothesis_card.py` refuses to validate a card whose ``s0_quote`` exceeds
the limit), and every other criterion (t >= 3, placebo distance, dose-response,
sign across years, positive net effect) had to hold **at the same time**. A single
prediction therefore killed a candidate before a single trade was simulated, and
the prediction was built on the wrong quantities to begin with:

* ``typische_bewegung_bp`` is the *mean absolute move*, but a trade only ever
  needs the move **in its own direction, as far as its target** - that is the
  first-passage/MFE quantity, not the mean move;
* the quote charged **market** cost on both legs, so every limit-entry candidate
  was taxed as if it crossed the spread twice.

Consequences that followed from that design, all of them measured in this repo:

* ``MAX_VARIANTEN = 10`` capped exploration at ten variants per idea, which is the
  opposite of how a strategy is actually found;
* the kill criteria may only be **tightened**, never relaxed, so a mis-set
  threshold can never be corrected;
* ``t_min = 3.0`` was applied to the *per-trade* series, where an exit model with
  ``sd(R) = 4.5`` needs ~20,000 trades to resolve a 0.1 R edge - a bar nothing in
  the repo could clear, so every candidate was rejected for lack of power rather
  than for lack of edge.

The funnel
----------
Each stage is a *filter on results that exist*, never a prediction. Nothing is
dropped silently: every stage reports what it removed and why, and the counts are
what tell you whether the screen is doing work or just cutting.

===  ==============  ==================================================
Stufe  Name            Kriterium
===  ==============  ==================================================
S1   Exploration     Brutto-Ertrag > 0 **und** besser als der Placebo
S2   Kosten          Netto > 0 bei realen Kosten der jeweiligen Uhrzeit
S3   Robustheit      Vorzeichen in >= 70 % der Jahre, Nachbar-Varianten positiv
S4   Signifikanz     Tages-t >= Schwelle **je Anzahl getesteter Varianten**
                     (oder Deflated Sharpe >= 0.95), Placebo-Abstand, PBO
S5   Out-of-Sample   Holdout, einmalig
===  ==============  ==================================================

The order is the whole point. A candidate may fail a *later* stage and be honestly
rejected; a candidate may never be rejected by a *prediction* before stage 1.

What the daily series does and does not do
------------------------------------------
An earlier version of this module claimed that aggregating to days "raises the
effective Sharpe" and so makes a small edge testable. That is false: with ``k``
independent trades per day the daily mean grows by ``k``, its sd by ``sqrt(k)`` and
the sample shrinks by ``k`` - the t is the same (simulated over 2,000 seeds:
0.55 per trade vs 0.56 per day). The daily series is the *honest* series (trades on
the same day are not independent, and that is what an account experiences), not a
more powerful one. Power comes from more trades, a larger edge per trade, or a lower
``sd(R)`` - never from re-binning.

The S4 bar therefore depends on how much was searched: one pre-registered variant
needs a one-sided t of 1.645; the best of many searched variants is judged by its
Deflated Sharpe (:func:`deflated_sharpe`), which uses the measured spread of the
trial Sharpes and so the effective number of trials (:func:`required_t` is the
conservative Bonferroni fallback). Without a recorded ``n_trials`` S4 assumes a broad search and
uses ``S4_MIN_T_DAILY = 3.0``.

    from trading_lab.screening import run_funnel
    funnel = run_funnel(variants)          # DataFrame with one row per variant
    funnel.summary()
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd

# Thresholds per stage - deliberately loose at the front, strict at the back.
S1_MIN_TRADES = 100          # below this the gross mean is meaningless
S3_MIN_POS_YEARS = 0.70
S3_NEIGHBOUR_MIN = 0.5       # share of neighbouring variants that must be positive
S4_MIN_T_DAILY = 3.0         # fallback when the number of searched variants is unknown
S4_MIN_PLACEBO_EDGE = 0.0    # > 0: must beat the placebo, sign only
S4_MAX_PBO = 0.25
S4_ALPHA = 0.05              # family-wise, one-sided
S4_MIN_DSR = 0.95
# Literature track (agreed with the user 2026-10-29, docs/PRUEFABLAUF.md): a row that
# implements a PUBLISHED rule 1:1 (``track == "literatur"``), whose reference check passed
# (``reference_ok``), is judged against the paper's out-of-sample evidence rather than as a
# fresh idea: S3 needs 60 % positive years when it holds positions > 1 week; S4 needs
# t >= 2.0 for the paper's base variant (``paper_base``) instead of the trial-deflated bar.
# Placebo and the holdout (S5) stay unchanged. Every other row keeps the normal bars.
LIT_S3_MIN_POS_YEARS = 0.60
LIT_S3_MIN_HOLD_DAYS = 5.0
LIT_S4_MIN_T = 2.0
_N = NormalDist()


def required_t(n_trials: int | float | None, *, alpha: float = S4_ALPHA) -> float:
    """One-sided t a variant needs when it is the best of ``n_trials`` searched ones.

    Bonferroni on the one-sided normal quantile: 1 trial -> 1.645, 10 -> 2.58,
    100 -> 3.29, 1000 -> 4.0. For a correlated grid this is conservative - that is
    why the optimiser reports a Deflated Sharpe (``dsr``) instead, which reads the
    real spread of the trial Sharpes and so the *effective* number of trials; this
    function is only the fallback for rows without one. ``None`` (unknown search
    breadth) -> ``S4_MIN_T_DAILY`` (= the Bonferroni bar of ~25 trials).
    """
    if n_trials is None or not np.isfinite(n_trials) or n_trials < 1:
        return S4_MIN_T_DAILY
    return float(_N.inv_cdf(1.0 - alpha / float(n_trials)))


def deflated_sharpe(daily: pd.Series | np.ndarray, *, n_trials: int = 1,
                    trial_sharpes: np.ndarray | None = None) -> float:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado 2014) of a daily series.

    Probability that the true (per-period) Sharpe exceeds the Sharpe the *best of
    ``n_trials`` pure-noise variants* would show. ``trial_sharpes`` - the per-period
    Sharpes of every searched variant - give the cross-trial variance; without them
    the null variance ``1/T`` is used. Skew and kurtosis of the series are taken
    into account, which matters for fat-tailed R distributions.
    """
    a = np.asarray(daily, dtype=float)
    a = a[np.isfinite(a)]
    T = len(a)
    if T < 10 or a.std(ddof=1) == 0:
        return float("nan")
    sr = a.mean() / a.std(ddof=1)
    if n_trials > 1:
        ts = None if trial_sharpes is None else np.asarray(trial_sharpes, dtype=float)
        ts = None if ts is None else ts[np.isfinite(ts)]
        var = float(np.var(ts, ddof=1)) if ts is not None and len(ts) >= 5 else 1.0 / T
        g = 0.5772156649
        sr0 = np.sqrt(var) * ((1 - g) * _N.inv_cdf(1 - 1.0 / n_trials)
                              + g * _N.inv_cdf(1 - 1.0 / (n_trials * np.e)))
    else:
        sr0 = 0.0
    z = (a - a.mean()) / a.std(ddof=0)
    sk, ku = float((z ** 3).mean()), float((z ** 4).mean())
    denom = np.sqrt(max(1e-12, 1 - sk * sr + (ku - 1) / 4.0 * sr * sr))
    return float(_N.cdf((sr - sr0) * np.sqrt(T - 1) / denom))


_RETURN_KEYS = ("r", "pnl_pct", "equity_pnl_pct")
_CLOCK_KEYS = ("day", "exit_time", "entry_time")


def _to_day(values: pd.Series) -> pd.Series:
    """Normalise a time column to calendar days.

    Accepts date/ISO strings and **unix seconds**. The numeric branch matters: the
    ``Position`` trade log stores ``entry_time``/``exit_time`` as unix seconds, and
    ``pd.to_datetime`` reads a bare float as nanoseconds - which silently collapsed
    every trade onto 1970-01-01 and made the daily t useless.
    """
    if pd.api.types.is_numeric_dtype(values):
        dt = pd.to_datetime(values, unit="s", utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(values, utc=True, errors="coerce")
    return dt.dt.tz_localize(None).dt.normalize()


def _day_axis(df: pd.DataFrame, time_key: str = "exit_time") -> pd.Series | None:
    """One day per row, or ``None`` when the frame carries no usable clock."""
    key = time_key if time_key in df else next((k for k in _CLOCK_KEYS if k in df), None)
    if key is None:
        return None
    day = _to_day(df[key])
    return None if day.isna().all() else day


def daily_series(trades: list[dict] | pd.DataFrame, *, r_key: str = "r",
                 time_key: str = "exit_time") -> pd.Series:
    """Collapse a trade log to one return per day.

    Why this exists: an account experiences days, not independent trades, and
    trades on the same day are correlated. The daily series is the honest one for
    significance. It is **not** more powerful - with independent trades its t equals
    the per-trade t (see the module docstring).

    Without a clock there is nothing to aggregate, so each row becomes its own block
    (this keeps a log that has no timestamps testable per row rather than empty).
    """
    df = trades if isinstance(trades, pd.DataFrame) else pd.DataFrame(trades)
    if df.empty or r_key not in df:
        return pd.Series(dtype=float)
    day = _day_axis(df, time_key)
    if day is None:
        day = pd.Series(range(len(df)), index=df.index)      # no clock: one block per row
    return df.groupby(day)[r_key].sum()


def _return_key(rows: list[dict], default: str = "r") -> str:
    """The first recognised return key present in the log (matches significance)."""
    return next((k for k in _RETURN_KEYS if any(t.get(k) is not None for t in rows)), default)


def funnel_row_from_trades(
    trades: list[dict],
    *,
    name: str = "",
    gross_bp: float | None = None,
    cost_bp: float | None = None,
    placebo_gross: float | None = None,
    pbo: float | None = None,
    oos_pass: bool | None = None,
    r_key: str | None = None,
) -> dict:
    """Build one funnel input row from a **gross** trade/event log (audit step C2).

    Fills what the log can prove - ``n``, ``gross_mean``, ``t_daily`` and
    ``pos_years``/``n_years`` - and passes the rest through untouched. A value left
    as ``None`` makes the funnel **skip** the stage that needs it, never drop the
    row (``screening.Stage``), which is the whole point: a missing number must not
    silently remove a candidate.

    ``gross_bp``/``cost_bp`` stay the caller's job. A study whose trade log is
    already **net of costs** must not feed it here as gross - S2 would charge the
    spread a second time. Feed the pre-cost (gross) move, or leave ``gross_bp``
    ``None`` and let S2 skip.
    """
    rows = [t for t in (trades or []) if t.get("status") in (None, "closed")]
    key = r_key or _return_key(rows)
    raw = [float(t[key]) for t in rows if t.get(key) is not None]
    arr = np.asarray(raw, dtype=float)
    arr = arr[np.isfinite(arr)]
    gross_mean = float(arr.mean()) if arr.size else None

    # t_daily / per-year only exist when the log has a *real* clock. Without one
    # "daily" is meaningless, so the field stays None and S3/S4 skip - better than
    # a bogus t_daily of 0.0 that would fail a candidate for the wrong reason.
    t_daily = pos_years = n_years = None
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    days = _day_axis(df) if not df.empty else None
    if days is not None and key in df:
        daily = df.groupby(days)[key].sum()
        t_daily = float(t_stat(daily)) if len(daily) else None
        per_year = pd.DataFrame({"y": days.dt.year, "r": df[key]}).dropna().groupby("y")["r"].sum()
        if not per_year.empty:
            n_years = int(per_year.size)
            pos_years = int((per_year > 0).sum())

    placebo_edge = (gross_mean - placebo_gross
                    if gross_mean is not None and placebo_gross is not None else None)
    return {
        "name": name,
        "n": int(arr.size),
        "gross_mean": gross_mean,
        "placebo_gross": placebo_gross,
        "gross_bp": gross_bp,
        "cost_bp": cost_bp,
        "pos_years": pos_years,
        "n_years": n_years,
        "neighbour_positive": None,
        "t_daily": t_daily,
        "placebo_edge": placebo_edge,
        "pbo": pbo,
        "oos_pass": oos_pass,
    }


def t_stat(x: pd.Series | np.ndarray) -> float:
    """t of the mean (0 when there is nothing to test)."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return 0.0
    sd = a.std(ddof=1)
    return float(a.mean() / (sd / np.sqrt(len(a)))) if sd else 0.0


def pbo(returns: pd.DataFrame, *, n_blocks: int = 16, seed: int = 7,
        max_splits: int = 2000) -> float:
    """Probability of backtest overfitting via CSCV (Bailey et al.).

    ``returns`` is variants x time. For every split of the blocks into two halves the
    best variant on one half is checked against the median on the other; PBO is the
    share of splits where the training winner lands below the test median. A random
    variant set gives ~0.5, a robust one clearly below.

    Returns ``nan`` when there is not enough data - S4 then skips this sub-check with
    a note instead of dropping the row (nothing falls out silently).
    """
    # Sample requirements are deliberately explicit. Measured behaviour of this
    # estimator on pure noise: with 20 variants the value swings between 0.25 and
    # 0.89 depending on the seed, so it is not usable there. With >= 5 variants and
    # >= 2 blocks per half it is stable enough to matter, and a genuine edge drives
    # it to ~0 (verified in tests/test_screening.py::test_pbo_detects_a_real_edge).
    # Returning nan rather than a noisy number is the honest option: S4 then skips
    # this sub-check with a note.
    r = returns.dropna(how="any")
    if r.shape[0] < 5 or r.shape[1] < 2 * n_blocks:
        return float("nan")
    blocks = np.array_split(np.arange(r.shape[1]), n_blocks)
    rng = np.random.default_rng(seed)
    combos = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    if len(combos) > max_splits:
        combos = [combos[i] for i in rng.choice(len(combos), max_splits, replace=False)]
    below = 0
    for combo in combos:
        tr = np.concatenate([blocks[i] for i in combo])
        te = np.concatenate([blocks[i] for i in range(n_blocks) if i not in combo])
        tr_mean = r.iloc[:, tr].mean(axis=1).to_numpy()
        te_mean = r.iloc[:, te].mean(axis=1).to_numpy()
        best = int(np.argmax(tr_mean))
        below += te_mean[best] < np.median(te_mean)
    return below / len(combos)


@dataclass
class Stage:
    """One rung of the funnel. Filters rows, but always reports what it cut.

    Three outcomes, never two:

    * ``kept`` - passed this stage
    * ``dropped`` - failed it, and that is a finding (too small a t, sign flips
      across years, ...). Read this frame to see *why* something died.
    * ``skipped`` - the stage could not judge it, because a needed value is
      missing. In S1-S3 these rows **carry on** to the next stage (a missing or
      predicted number must not silently remove a candidate). In S4/S5 they stop:
      a significance or holdout stage that could not judge a row has not proven
      anything, so the row cannot survive or reach the holdout.
    """

    name: str
    criterion: str
    kept: pd.DataFrame = field(default_factory=pd.DataFrame)
    dropped: pd.DataFrame = field(default_factory=pd.DataFrame)
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)
    note: str = ""

    @property
    def n_in(self) -> int:
        return len(self.kept) + len(self.dropped) + len(self.skipped)

    def summary_line(self) -> str:
        pct = 100 * len(self.kept) / max(self.n_in, 1)
        extra = f"   ({len(self.skipped)} nicht beurteilbar)" if len(self.skipped) else ""
        return (f"  {self.name:16s} {len(self.kept):5d} von {self.n_in:5d} "
                f"({pct:5.1f}%)  {self.criterion}{extra}")


def _split(df: pd.DataFrame, mask: pd.Series, stage: str, criterion: str,
           *, judgeable: pd.Series | None = None, note: str = "") -> Stage:
    """Split into kept / dropped / skipped. ``judgeable`` defaults to all rows."""
    j = pd.Series(True, index=df.index) if judgeable is None else judgeable.fillna(False)
    return Stage(name=stage, criterion=criterion,
                 kept=df[j & mask].copy(), dropped=df[j & ~mask].copy(),
                 skipped=df[~j].copy(), note=note)


def s1_explore(df: pd.DataFrame, *, gross_col: str = "gross_mean",
               placebo_col: str = "placebo_gross",
               n_col: str = "n") -> Stage:
    """Gross positive AND better than a random direction through the same exits.

    No cost, no significance. This is the stage where hundreds or thousands of
    variants are allowed to live. The placebo comparison is the only control that
    belongs here: without it, a positive gross number cannot be told apart from a
    favourable skew in the exit machinery.

    Rows without ``gross_mean`` / ``n`` are **skipped**, not dropped - a card whose
    study has not emitted its numbers yet must not disappear from the funnel.
    """
    judgeable = df[gross_col].notna() & df[n_col].notna()
    # gross > 0 is its own condition: comparing only against the placebo let a
    # negative variant through whenever the placebo was more negative still.
    placebo = pd.to_numeric(df[placebo_col], errors="coerce") if placebo_col in df \
        else pd.Series(np.nan, index=df.index)
    mask = ((df[n_col] >= S1_MIN_TRADES) & (df[gross_col] > 0)
            & (placebo.isna() | (df[gross_col] > placebo)))
    return _split(df, mask, "S1 Exploration",
                  f"brutto > 0, > Placebo, n >= {S1_MIN_TRADES}", judgeable=judgeable)


def s2_cost(df: pd.DataFrame, *, cost_col: str = "cost_bp",
            gross_col: str = "gross_bp", net_col: str = "net_mean") -> Stage:
    """Now, and only now, charge real costs.

    A *measured* net result (``net_mean``: the same backtest with real bid/ask
    fills) wins over ``gross_bp - cost_bp``. The bp pair is the fallback for
    studies that only know a gross move and a median spread.
    """
    net = pd.to_numeric(df[net_col], errors="coerce") if net_col in df \
        else pd.Series(np.nan, index=df.index)
    have_bp = (df[gross_col].notna() & df[cost_col].notna()) if gross_col in df and cost_col in df \
        else pd.Series(False, index=df.index)
    judgeable = net.notna() | have_bp
    bp_ok = (df[gross_col] - df[cost_col]) > 0 if gross_col in df and cost_col in df \
        else pd.Series(False, index=df.index)
    mask = net.gt(0).where(net.notna(), bp_ok)
    return _split(df, mask.astype(bool), "S2 Kosten",
                  f"{net_col} > 0 (gemessen), sonst ({gross_col} - {cost_col}) > 0",
                  judgeable=judgeable)


def s3_robust(df: pd.DataFrame, *, pos_col: str = "pos_years",
              n_years_col: str = "n_years",
              neighbour_col: str = "neighbour_positive") -> Stage:
    """Sign has to hold across years, and the neighbourhood has to agree.

    A variant that works in one year and nowhere else is a spike.
    """
    judgeable = df[pos_col].notna() & df[n_years_col].notna()
    yrs = df[pos_col] / df[n_years_col].replace(0, np.nan)
    hold = pd.to_numeric(df["holding_days"], errors="coerce") if "holding_days" in df \
        else pd.Series(np.nan, index=df.index)
    slow_lit = literature_rows(df) & (hold > LIT_S3_MIN_HOLD_DAYS)
    need = pd.Series(S3_MIN_POS_YEARS, index=df.index).where(~slow_lit, LIT_S3_MIN_POS_YEARS)
    mask = (yrs >= need)
    if neighbour_col in df:
        mask &= df[neighbour_col].isna() | (df[neighbour_col] >= S3_NEIGHBOUR_MIN)
    return _split(df, mask, "S3 Robustheit",
                  f"Vorzeichen in >= {S3_MIN_POS_YEARS:.0%} der Jahre (Literatur-Spur, Haltedauer "
                  f"> {LIT_S3_MIN_HOLD_DAYS:.0f} Tage: {LIT_S3_MIN_POS_YEARS:.0%}), "
                  f"Nachbarn >= {S3_NEIGHBOUR_MIN:.0%}", judgeable=judgeable)


def literature_rows(df: pd.DataFrame) -> pd.Series:
    """Rows on the literature track: published rule, reference check passed."""
    track = df["track"] if "track" in df else pd.Series(None, index=df.index)
    ref = df["reference_ok"] if "reference_ok" in df else pd.Series(False, index=df.index)
    return (track == "literatur") & ref.fillna(False).astype(bool)


def s4_significant(df: pd.DataFrame, *, t_col: str = "t_daily",
                   placebo_edge_col: str = "placebo_edge",
                   pbo_col: str = "pbo", trials_col: str = "n_trials",
                   dsr_col: str = "dsr") -> Stage:
    """Significance on the DAILY series, with a bar that knows how much was searched.

    * ``dsr`` present -> Deflated Sharpe >= ``S4_MIN_DSR`` (preferred: accounts for
      the number of trials, their spread, skew and kurtosis).
    * else ``t_daily >= required_t(n_trials)`` - 1.645 for one pre-registered
      variant, ~3.3 for the best of 500. No ``n_trials`` -> ``S4_MIN_T_DAILY``.

    The bar is strict because S4 is the last gate before the holdout, and because
    the optimiser upstream picks the *best* of many variants: the best of 500 noise
    variants shows a t of ~3 by construction.
    """
    judgeable = df[t_col].notna() & df[placebo_edge_col].notna()
    trials = df[trials_col] if trials_col in df else pd.Series(np.nan, index=df.index)
    t_req = trials.map(lambda n: required_t(None if pd.isna(n) else n))
    t_ok = df[t_col] >= t_req
    if dsr_col in df:
        dsr_ok = df[dsr_col] >= S4_MIN_DSR
        mask = dsr_ok.where(df[dsr_col].notna(), t_ok)
    else:
        mask = t_ok
    base = df["paper_base"] if "paper_base" in df else pd.Series(False, index=df.index)
    lit = literature_rows(df) & base.fillna(False).astype(bool)
    mask = mask.astype(bool).where(~lit, df[t_col] >= LIT_S4_MIN_T)
    mask = mask.astype(bool) & (df[placebo_edge_col] > S4_MIN_PLACEBO_EDGE)
    if pbo_col in df:
        mask &= df[pbo_col].isna() | (df[pbo_col] <= S4_MAX_PBO)
    return _split(df, mask, "S4 Signifikanz",
                  f"DSR >= {S4_MIN_DSR} bzw. t(taeglich) >= t(n_trials) "
                  f"[ohne n_trials {S4_MIN_T_DAILY}; Literatur-Spur Paper-Basis: t >= {LIT_S4_MIN_T}], "
                  f"Placebo-Vorsprung > 0, PBO <= {S4_MAX_PBO}", judgeable=judgeable)


def s5_oos(df: pd.DataFrame, *, passes_col: str = "oos_pass") -> Stage:
    """Holdout, once. Only variants that reached this point may look at it."""
    judgeable = df[passes_col].notna() if passes_col in df else pd.Series(False, index=df.index)
    mask = df[passes_col].fillna(False).astype(bool) if passes_col in df else pd.Series(False, index=df.index)
    return _split(df, mask, "S5 Out-of-Sample", "Holdout bestanden", judgeable=judgeable)


@dataclass
class Funnel:
    stages: list[Stage]

    def summary(self) -> str:
        lines = ["SCREENING-FUNNEL (locker -> streng; jede Stufe berichtet, was sie cuttet)"]
        lines += [s.summary_line() for s in self.stages]
        if self.stages:
            first, last = self.stages[0], self.stages[-1]
            lines.append(f"  {'':16s} {len(last.kept):5d} von {first.n_in:5d} ueberleben "
                         f"({100*len(last.kept)/max(first.n_in,1):.2f}%)")
            # a stage that cuts nothing is either useless or a sign the input is
            # already filtered - worth seeing either way
            for s in self.stages:
                if s.n_in and len(s.dropped) == 0:
                    lines.append(f"  HINWEIS: {s.name} cuttet nichts - "
                                 f"entweder wirkungslos oder Input schon gefiltert")
                if len(s.skipped):
                    lines.append(f"  HINWEIS: {s.name} konnte {len(s.skipped)} Zeilen "
                                 f"nicht beurteilen (Werte fehlen) - sie laufen weiter")
        return "\n".join(lines)

    @property
    def survivors(self) -> pd.DataFrame:
        """Was alle Stufen passiert hat. Endet der Trichter vor S4, zaehlen auch die
        nicht beurteilbaren Zeilen dazu (nie still verworfen); ab S4 nur ``kept`` -
        ohne Nachweis kein Ueberleben."""
        if not self.stages:
            return pd.DataFrame()
        last = self.stages[-1]
        tag = last.name.split()[0]
        if len(last.skipped) and tag in CARRY_SKIPPED:
            return pd.concat([last.kept, last.skipped])
        return last.kept


DEFAULT_UP_TO = "S4"
#: Stages whose "nicht beurteilbar" rows carry on. S4/S5 prove something; a row
#: they cannot judge has proven nothing and stops there (visible in ``skipped``).
CARRY_SKIPPED = ("S1", "S2", "S3")
HOLDOUT_SPERRE = ("S5 gesperrt: der Holdout wird nur einmal und nur mit Freigabe "
                  "verbraucht (Feld holdout_freigabe). Vorher ist S4 das Ende.")


def run_funnel(df: pd.DataFrame, *, up_to: str | None = DEFAULT_UP_TO,
               holdout_freigabe: str | None = None) -> Funnel:
    """Run the stages in order. Stops at ``up_to`` (default ``S4``).

    ``S5`` (the holdout) only runs when a release is passed explicitly. Without it
    the funnel stops at S4 and says so, rather than quietly consuming the one shot
    at an untouched sample. This mirrors the plan's rule: never look at the holdout
    unless a human released it.
    """
    stages: list[Stage] = []
    cur = df
    for fn, tag in ((s1_explore, "S1"), (s2_cost, "S2"), (s3_robust, "S3"),
                    (s4_significant, "S4"), (s5_oos, "S5")):
        if tag == "S5" and not holdout_freigabe:
            stages.append(Stage(name="S5 Out-of-Sample", criterion="Holdout (gesperrt)",
                                kept=cur, dropped=pd.DataFrame(), note=HOLDOUT_SPERRE))
            break
        s = fn(cur)
        stages.append(s)
        # weitergereicht wird kept + skipped: eine Zeile, die eine Stufe nicht
        # beurteilen konnte, darf nicht verschwinden - ABER nur bis S3. S4/S5 sind
        # Nachweis-Stufen: wer dort nicht beurteilbar ist (z.B. ohne Placebo), hat
        # nichts nachgewiesen und darf weder in den Holdout noch als Ueberlebender
        # gelten. Er bleibt in ``skipped`` sichtbar (Review 2026-10-26).
        cur = pd.concat([s.kept, s.skipped]) if len(s.skipped) and tag in CARRY_SKIPPED else s.kept
        if up_to and tag == up_to:
            break
    return Funnel(stages)
