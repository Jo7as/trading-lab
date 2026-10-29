"""Order-flow measures on signed flow bars (``trading_lab.adapters.crypto_trades``).

Input: bars indexed by UTC bar start with ``open, high, low, close, vwap, buy_qty, sell_qty,
buy_notional, sell_notional, n_trades`` on a full clock grid (empty bars: zero volume, NaN
prices). All measures are causal - the value at bar t uses bars <= t only.

Definitions and sources: ``docs/research/CRYPTO_MICROSTRUCTURE_THEORY.md``.

* ``imbalance``          OI = (B - S) / (B + S) in [-1, 1]    (Kim & Hansen 2026)
* ``delta`` / ``cvd``    signed notional, cumulative with an anchor (session reset)
* ``combine_venues``     several venues -> one flow picture (weighted sum of volumes)
* ``resample``           10 s -> 1 min / 15 min ... (OHLC + sums, VWAP recomputed)
* ``clock_openings``     flow in the first window after each clock mark (:00/:15/:30/:45)
* ``corwin_schultz``     high-low spread proxy, two-bar version, right-aligned (Corwin & Schultz
                         2012, used by Pindza 2026; no overnight adjustment - 24/7 market)
* ``kyle_lambda``        rolling slope of log return on signed notional
* ``vpin``               volume-clock order-flow toxicity (Easley, López de Prado, O'Hara 2012),
                         taker side or the paper's bulk volume classification (``bvc_split``)
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

VOLUME_COLS = ["buy_qty", "sell_qty", "buy_notional", "sell_notional", "n_trades"]


def _sides(bars: pd.DataFrame, by: str) -> tuple[pd.Series, pd.Series]:
    if by not in ("notional", "qty"):
        raise ValueError("by must be 'notional' or 'qty'")
    return bars[f"buy_{by}"], bars[f"sell_{by}"]


def imbalance(bars: pd.DataFrame, *, by: str = "notional") -> pd.Series:
    """Order imbalance per bar, NaN on bars without volume."""
    b, s = _sides(bars, by)
    tot = b + s
    return ((b - s) / tot.where(tot > 0)).rename("imbalance")


def delta(bars: pd.DataFrame, *, by: str = "notional") -> pd.Series:
    """Signed volume per bar (buyer-initiated minus seller-initiated)."""
    b, s = _sides(bars, by)
    return (b - s).rename("delta")


def cvd(bars: pd.DataFrame, *, anchor: str | None = "1D", by: str = "notional") -> pd.Series:
    """Cumulative delta, reset at every ``anchor`` boundary (UTC), or never with None.

    flowsurface's CVD never resets, so its level depends on how much history is loaded;
    an anchored CVD is comparable from day to day.
    """
    d = delta(bars, by=by)
    if anchor is None:
        return d.cumsum().rename("cvd")
    return d.groupby(d.index.floor(anchor)).cumsum().rename("cvd")


def combine_venues(venues: Mapping[str, pd.DataFrame], *, weights: Mapping[str, float] | None = None,
                   price_from: str | None = None) -> pd.DataFrame:
    """One flow picture from several venues on the same clock grid.

    Volumes are summed (optionally weighted, e.g. by price leadership); prices come from
    ``price_from`` (default: the first venue) because quotes differ between venues and
    averaging them would invent a price nobody could trade. Grids must match exactly.
    """
    if not venues:
        raise ValueError("no venues")
    names = list(venues)
    ref = price_from or names[0]
    idx = venues[ref].index
    for n in names:
        if not venues[n].index.equals(idx):
            raise ValueError(f"venue {n!r} is not on the same clock grid as {ref!r}")
    w = {n: 1.0 for n in names} if weights is None else {n: float(weights.get(n, 0.0)) for n in names}
    out = venues[ref][["open", "high", "low", "close", "vwap"]].copy()
    for c in VOLUME_COLS:
        out[c] = sum(venues[n][c] * w[n] for n in names)
    out.attrs["venues"], out.attrs["weights"], out.attrs["price_from"] = names, w, ref
    return out


def resample(bars: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate flow bars to a coarser clock (left-labelled, left-closed like the input)."""
    r = bars.resample(rule, label="left", closed="left")
    out = pd.DataFrame({
        "open": r["open"].first(), "high": r["high"].max(), "low": r["low"].min(),
        "close": r["close"].last(),
    })
    for c in VOLUME_COLS:
        out[c] = r[c].sum()
    qty = out["buy_qty"] + out["sell_qty"]
    out["vwap"] = (out["buy_notional"] + out["sell_notional"]) / qty.where(qty > 0)
    return out[["open", "high", "low", "close", "vwap", *VOLUME_COLS]]


def clock_openings(bars: pd.DataFrame, *, every: str = "15min", window: str = "10s",
                   offset: str = "0s", by: str = "notional") -> pd.DataFrame:
    """Flow in ``[mark, mark + window)`` after each clock mark (default :00/:15/:30/:45).

    ``offset`` shifts the marks (e.g. "2min" -> :02/:17/:32/:47), the placebo grid of
    Kim & Hansen: a real clock effect must vanish off the true boundary.

    Returns one row per mark with ``imbalance``, ``delta`` and total volume. The window
    must be a multiple of the bar size; the result is known at ``mark + window`` (column
    ``known_at``), which is when a strategy may act on it.
    """
    step = pd.Timedelta(bars.index[1] - bars.index[0])
    win = pd.Timedelta(window)
    if win < step or win % step:
        raise ValueError(f"window {window} must be a multiple of the bar size {step}")
    off = pd.Timedelta(offset)
    shifted = bars.index - off
    marks = bars.index[shifted == shifted.floor(every)]
    b, s = _sides(bars, by)
    k = int(win / step)
    bw = b.rolling(k).sum().shift(-(k - 1))      # sum over [t, t + window)
    sw = s.rolling(k).sum().shift(-(k - 1))
    bw, sw = bw.reindex(marks), sw.reindex(marks)
    tot = bw + sw
    return pd.DataFrame({
        "imbalance": (bw - sw) / tot.where(tot > 0),
        "delta": bw - sw,
        "volume": tot,
        "known_at": marks + win,
    }, index=marks)


def forward_log_returns(bars: pd.DataFrame, at: pd.DatetimeIndex,
                        horizons: dict[str, str]) -> pd.DataFrame:
    """log(P[t + h] / P[t]) for each time in ``at``, P = last traded price up to that bar.

    ``at`` are bar labels; the price of bar t is its close (the last trade in the bar),
    carried forward over empty bars. Horizons past the end of the data give NaN.
    """
    price = bars["close"].ffill()
    p0 = price.reindex(at).to_numpy()
    out = {}
    for name, h in horizons.items():
        p1 = price.reindex(at + pd.Timedelta(h)).to_numpy()
        out[name] = np.log(p1 / p0)
    return pd.DataFrame(out, index=at)


def corwin_schultz(bars: pd.DataFrame, *, adjust_gaps: bool = True) -> pd.Series:
    """Corwin & Schultz (2012) spread estimate from the high/low of bars t-1 and t.

    Equations (6), (9), (15), (16) of the paper; negative two-bar estimates are set to 0
    (their section 2.3, before any averaging). ``adjust_gaps`` applies their section 2.1:
    if bar t opens beyond the previous close (low_t > close_{t-1} or high_t < close_{t-1}),
    bar t's high and low are shifted by that gap so a jump between bars is not read as
    spread. Right-aligned (uses no future bar).
    """
    hi, lo = bars["high"].astype(float), bars["low"].astype(float)
    if adjust_gaps:
        prev_close = bars["close"].shift(1)
        gap = (lo - prev_close).clip(lower=0).fillna(0) - (prev_close - hi).clip(lower=0).fillna(0)
        hi, lo = hi - gap, lo - gap
    hl_raw = np.log(bars["high"] / bars["low"]) ** 2     # bar t-1 stays as observed
    beta = np.log(hi / lo) ** 2 + hl_raw.shift(1)
    h2 = np.maximum(hi, bars["high"].shift(1))
    l2 = np.minimum(lo, bars["low"].shift(1))
    gamma = np.log(h2 / l2) ** 2
    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return spread.clip(lower=0.0).rename("cs_spread")


def kyle_lambda(bars: pd.DataFrame, window: int, *, per: float = 1e6) -> pd.Series:
    """Rolling OLS slope of the bar's log return on its signed notional (per ``per`` USD).

    High lambda = the same flow moves the price more (thin book or informed flow).
    """
    ret = np.log(bars["close"]).diff()
    x = delta(bars, by="notional") / per
    ok = ret.notna() & x.notna()
    ret, x = ret.where(ok), x.where(ok)
    mx = x.rolling(window, min_periods=window // 2).mean()
    my = ret.rolling(window, min_periods=window // 2).mean()
    cov = (x * ret).rolling(window, min_periods=window // 2).mean() - mx * my
    var = (x * x).rolling(window, min_periods=window // 2).mean() - mx * mx
    return (cov / var.where(var > 0)).rename("kyle_lambda")


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF via Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7), vectorised."""
    z = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-z * z)
    return 0.5 * (1.0 + np.sign(x) * erf)


def bvc_split(bars: pd.DataFrame, sigma_window: int) -> tuple[pd.Series, pd.Series]:
    """Bulk volume classification (Easley, López de Prado & O'Hara 2012): the buy share of a
    bar's volume is Phi(dP / sigma_dP), from the price change alone - how VPIN is defined
    for data without an aggressor flag. sigma is the rolling std of past bar changes
    (causal, excluding the current bar). Bars without a defined sigma get NaN.
    """
    price = bars["close"].ffill()
    dp = np.log(price).diff()
    sigma = dp.rolling(sigma_window, min_periods=max(2, sigma_window // 2)).std().shift(1)
    share = pd.Series(_norm_cdf((dp / sigma.where(sigma > 0)).to_numpy()), index=bars.index)
    vol = bars["buy_notional"] + bars["sell_notional"]
    return vol * share, vol * (1.0 - share)


def vpin(bars: pd.DataFrame, bucket_notional: float, n_buckets: int = 50, *,
         classification: str = "aggressor", sigma_window: int = 360) -> pd.Series:
    """VPIN (Easley, López de Prado & O'Hara 2012, eq. 9): sum |V_S - V_B| / (n V) over the
    last ``n_buckets`` volume buckets of exactly ``bucket_notional`` each.

    As in the paper, volume that overfills a bucket goes to the next one: within a bar,
    buy volume accrues in proportion to its volume, so a bar can be split over several
    buckets. The paper's default is V = average daily volume / 50 and n = 50 (one day).

    ``classification``:
      * "aggressor" (default) - the exchanges' taker side; exact, but not the paper's measure
      * "bvc" - the paper's bulk volume classification (eq. 7, ``bvc_split``); bars before
        sigma is defined are split 50/50

    VPIN is updated when a bucket fills (paper: "after each volume bucket"), i.e. on the bar
    whose volume completes it, and carried forward; NaN until ``n_buckets`` are complete.
    """
    if bucket_notional <= 0:
        raise ValueError("bucket_notional must be positive")
    vol = (bars["buy_notional"] + bars["sell_notional"]).to_numpy(dtype=float)
    if classification == "aggressor":
        buy = bars["buy_notional"].to_numpy(dtype=float)
    elif classification == "bvc":
        bs, _ = bvc_split(bars, sigma_window)
        buy = bs.to_numpy(dtype=float)
        buy = np.where(np.isnan(buy), vol / 2.0, buy)
    else:
        raise ValueError("classification must be 'aggressor' or 'bvc'")
    cum_v = np.concatenate([[0.0], np.cumsum(vol)])
    cum_b = np.concatenate([[0.0], np.cumsum(buy)])
    n_full = int(cum_v[-1] // bucket_notional)
    out = pd.Series(np.nan, index=bars.index, name="vpin")
    if n_full == 0:
        return out
    edges = bucket_notional * np.arange(1, n_full + 1)
    b_at = np.interp(edges, cum_v, cum_b)              # cumulative buy at each bucket edge
    b_bucket = np.diff(np.concatenate([[0.0], b_at]))
    tox = np.abs(2.0 * b_bucket - bucket_notional) / bucket_notional
    roll = pd.Series(tox).rolling(n_buckets).mean().to_numpy()
    done_at = np.searchsorted(cum_v[1:], edges - 1e-9 * bucket_notional, side="left")
    last = pd.Series(roll).groupby(done_at).last()    # several buckets may fill in one bar
    out.iloc[last.index.to_numpy()] = last.to_numpy()
    return out.ffill()


# ── level building blocks for H8 (SMC x order flow) ───────────────────
def volume_profile(bars: pd.DataFrame, bin_size: float) -> pd.DataFrame:
    """Buy / sell / total notional per price bin over all given bars (a footprint summary).

    Practitioner convention (Market Profile, CBOT), not a published estimator.

    Approximation: a bar's whole volume sits at its VWAP (10 s bars move little, so the
    error is below one bin for sensible bin sizes). Index = lower edge of the bin.
    """
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")
    live = bars[(bars["buy_notional"] + bars["sell_notional"]) > 0]
    key = np.floor(live["vwap"] / bin_size) * bin_size
    prof = live.groupby(key)[["buy_notional", "sell_notional"]].sum()
    prof.columns = ["buy", "sell"]
    prof["total"] = prof["buy"] + prof["sell"]
    prof.index.name = "price"
    return prof


def profile_levels(profile: pd.DataFrame, value_area: float = 0.70) -> dict[str, float]:
    """Point of control and value area (CBOT method: grow from the POC towards the side
    with more volume until ``value_area`` of the total is inside).

    Returns ``poc``, ``val`` (lower edge of the lowest bin) and ``vah`` (lower edge of
    the highest bin in the value area); bins without trades count as zero volume.
    """
    if profile.empty:
        return {"poc": np.nan, "val": np.nan, "vah": np.nan}
    step = float(np.min(np.diff(profile.index))) if len(profile) > 1 else 1.0
    grid = np.round(np.arange(profile.index.min(), profile.index.max() + step / 2, step), 10)
    vol = profile["total"].reindex(np.round(profile.index, 10)).groupby(level=0).sum()
    vol = vol.reindex(grid, fill_value=0.0).to_numpy()
    i = j = int(np.argmax(vol))
    inside, target = vol[i], value_area * vol.sum()
    while inside < target and (i > 0 or j < len(vol) - 1):
        down = vol[i - 1] if i > 0 else -1.0
        up = vol[j + 1] if j < len(vol) - 1 else -1.0
        if up >= down:
            j += 1
            inside += vol[j]
        else:
            i -= 1
            inside += vol[i]
    return {"poc": float(grid[int(np.argmax(vol))]), "val": float(grid[i]), "vah": float(grid[j])}


def session_profile_levels(bars: pd.DataFrame, bin_size: float, *, anchor: str = "1D",
                           value_area: float = 0.70) -> pd.DataFrame:
    """POC / VAL / VAH of the **previous** session, aligned to every bar (causal).

    A level becomes known at the start of the next session, so bars of session k see the
    profile of session k-1; the first session gets NaN.
    """
    sess = bars.index.floor(anchor)
    levels = {s: profile_levels(volume_profile(g, bin_size), value_area)
              for s, g in bars.groupby(sess)}
    keys = sorted(levels)
    prev = {keys[n]: levels[keys[n - 1]] for n in range(1, len(keys))}
    rows = [prev.get(s, {"poc": np.nan, "val": np.nan, "vah": np.nan}) for s in sess]
    return pd.DataFrame(rows, index=bars.index)[["poc", "val", "vah"]]


def flow_vs_move(bars: pd.DataFrame, window: int, lookback: int) -> pd.DataFrame:
    """Order flow against price move over the last ``window`` bars - the raw material
    for absorption (strong flow, small move) and cascades (flow and move together).

    **Own construction, not a published indicator** (hypothesis H8): motivated by Kyle's
    lambda and by state-dependent order flow (Jeon 2026), it has to earn its place in the
    level study against random-level controls before anything is built on it.

    * ``flow``   = imbalance of the summed notional over the window, in [-1, 1]
    * ``move_z`` = log return over the window / rolling std of window returns over
      ``lookback`` windows (z-score, so markets and regimes are comparable)
    """
    b = bars["buy_notional"].rolling(window).sum()
    s = bars["sell_notional"].rolling(window).sum()
    tot = b + s
    flow = (b - s) / tot.where(tot > 0)
    price = bars["close"].ffill()
    move = np.log(price / price.shift(window))
    sd = move.rolling(lookback * window, min_periods=lookback).std()
    return pd.DataFrame({"flow": flow, "move_z": move / sd.where(sd > 0)}, index=bars.index)
