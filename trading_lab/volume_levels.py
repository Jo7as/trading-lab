"""Volume-profile levels and how price treats them (H10, pre-registered in
``docs/specs/h10_volume_profile_PREREG.md``). Measurement only, no strategy.

Input: 1-min flow bars (``orderflow.resample(bars_10s, "1min")``) on a full UTC clock grid.
Everything is causal: a level exists only from ``known_at`` (the end of its window).

* windows        ``fixed_windows`` (last N 15-min bars, rolling), ``period_windows`` (UTC day /
                 week), ``session_windows`` (Tokyo / London / New York, DST-aware, via
                 ``mr_common.day_slices``), ``leg_windows`` (between consecutive external swings
                 of ``market_structure``)
* levels         ``profile_nodes`` (POC / VAH / VAL from ``orderflow.profile_levels`` + HVN /
                 LVN of a lightly smoothed profile), ``window_levels``, ``naked_pocs``
* tests          ``magnet_events`` (reached within h vs a random level at the same distance and
                 side at another time, and vs the mirror level), ``hold_events`` (bounce or break
                 first after the first touch), ``smc_confluence``
* statistics     ``nw_mean_t`` (Newey-West t of a mean), ``cell_stats``, ``diff_stats``

Volume-profile settings have no academic specification; the choices here (50 rows, 70 % value
area, [1,2,3,2,1] smoothing) are practitioner conventions fixed before the first run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import level_flow as lf
from . import market_structure as ms
from . import orderflow as of
from .level_flow import Level
from .mr_common import day_slices

#: pre-registered constants (2026-10-29)
N_ROWS = 50
VALUE_AREA = 0.70
SMOOTH = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0
NODE_RADIUS = 2                    # strict local extremum over +-2 rows
HVN_MIN, LVN_MAX = 1.0, 0.5        # x mean row volume
MIN_DIST_ATR, MAX_DIST_ATR = 0.25, 6.0
HOLD_ATR = 0.5                     # bounce / break distance, ATR(1h)
HOLD_WINDOW = "4h"
MAX_AGE = "24h"                    # first touch within this time
FLOW_BARS = 5                      # 1-min bars ending with the touch bar
FLOW_THR = lf.FLOW_THR             # 0.15, as H8
CONFLUENCE_ATR = 0.10              # x ATR(14) of 15-min bars
MIN_COVERAGE = 0.90                # share of non-empty 1-min bars a window needs
SESSIONS = {"asia": ("Asia/Tokyo", "09:00", "15:00"),
            "london": ("Europe/London", "08:00", "16:30"),
            "ny": ("America/New_York", "09:30", "16:00")}


@dataclass(frozen=True)
class Window:
    name: str
    start: pd.Timestamp            # first bar label (inclusive)
    end: pd.Timestamp              # end of the last bar (exclusive)
    known: pd.Timestamp | None = None   # when the levels may be used (default: end)


# ── windows ─────────────────────────────────────────────────────────────
def fixed_windows(m1: pd.DataFrame, n15: int, *, name: str | None = None) -> list[Window]:
    """Last ``n15`` bars of 15 min, a new instance every quarter of the window length, on a
    grid anchored at the first UTC midnight of the data."""
    length = pd.Timedelta(minutes=15 * n15)
    step = length / 4
    anchor = m1.index[0].floor("1D")
    end_of_data = m1.index[-1] + (m1.index[1] - m1.index[0])
    out = []
    t = anchor + length
    while t <= end_of_data:
        if t - length >= m1.index[0]:
            out.append(Window(name or f"n{n15}", t - length, t))
        t += step
    return out


def period_windows(m1: pd.DataFrame, period: str) -> list[Window]:
    """Complete UTC days (``"day"``) or Monday-to-Monday weeks (``"week"``)."""
    day0 = m1.index.floor("1D")
    if period == "day":
        starts, step = day0.unique(), pd.Timedelta(days=1)
    elif period == "week":
        starts = (day0 - pd.to_timedelta(m1.index.weekday, unit="D")).unique()
        step = pd.Timedelta(days=7)
    else:
        raise ValueError("period must be 'day' or 'week'")
    end_of_data = m1.index[-1] + (m1.index[1] - m1.index[0])
    return [Window(period, s, s + step) for s in starts if s >= m1.index[0] and s + step <= end_of_data]


def session_windows(m1: pd.DataFrame, session: str) -> list[Window]:
    """One window per local trading day of the session (exchange-local clock, DST-aware).
    Slices come from ``day_slices`` on the FULL frame (never from a masked subset)."""
    tz, a, b = SESSIONS[session]
    local = m1.index.tz_convert(tz)
    minutes = local.hour * 60 + local.minute
    lo = int(a[:2]) * 60 + int(a[3:])
    hi = int(b[:2]) * 60 + int(b[3:])
    mask = np.asarray((minutes >= lo) & (minutes < hi))
    step = m1.index[1] - m1.index[0]
    full = (hi - lo) // int(step / pd.Timedelta("1min"))
    _, slices = day_slices(local, mask)
    return [Window(session, m1.index[sl.start], m1.index[sl.stop - 1] + step)
            for sl in slices if sl.stop - sl.start == full]      # complete sessions only


def leg_windows(m15: pd.DataFrame, res: ms.StructureResult | None = None) -> list[Window]:
    """Profile between two consecutive confirmed external swings (low -> high or high -> low),
    from the first swing's bar to the end of the second's; known at the end of the 15-min bar
    that confirms the second swing (the first is final by then)."""
    res = res if res is not None else ms.analyze(m15)
    sw = res.swings.get("ext", [])
    step = m15.index[1] - m15.index[0]
    return [Window("leg", m15.index[a.idx], m15.index[b.idx] + step, m15.index[b.confirmed_idx] + step)
            for a, b in zip(sw, sw[1:]) if a.kind != b.kind]


def known_at(w: Window) -> pd.Timestamp:
    return w.known if w.known is not None else w.end


# ── profile -> levels ───────────────────────────────────────────────────
def profile_nodes(bars: pd.DataFrame, n_rows: int = N_ROWS) -> dict[str, list[float]] | None:
    """POC / VAH / VAL (row centres, ``orderflow.profile_levels`` on a dense row grid) and
    HVN / LVN of the profile smoothed with ``SMOOTH``:

    * HVN: strict local max over +-``NODE_RADIUS`` rows, smoothed >= ``HVN_MIN`` x mean row
      volume, not the POC row;
    * LVN: strict local min over +-``NODE_RADIUS`` rows inside [VAL, VAH], smoothed <=
      ``LVN_MAX`` x mean row volume.
    Rows: ``n_rows`` equal rows over the bars' [low, high] (grid aligned to multiples of the
    row size, as ``orderflow.volume_profile`` bins). None if there is no range or no volume.
    """
    lo, hi = float(np.nanmin(bars["low"])), float(np.nanmax(bars["high"]))
    if not np.isfinite(lo) or not hi > lo:
        return None
    size = (hi - lo) / n_rows
    prof = of.volume_profile(bars, size)
    if prof.empty or prof["total"].sum() <= 0:
        return None
    k = np.rint(prof.index.to_numpy() / size).astype(np.int64)
    k0 = int(k.min())
    vol = np.zeros(int(k.max()) - k0 + 1)
    np.add.at(vol, k - k0, prof["total"].to_numpy())
    edges = (np.arange(len(vol)) + k0) * size
    dense = pd.DataFrame({"total": vol}, index=pd.Index(edges, name="price"))
    pl = of.profile_levels(dense, VALUE_AREA)
    row = lambda p: int(np.rint(p / size)) - k0           # noqa: E731
    i_poc, i_val, i_vah = row(pl["poc"]), row(pl["val"]), row(pl["vah"])
    sm = np.convolve(vol, SMOOTH, mode="same")
    mean = vol.mean()
    r = NODE_RADIUS
    hvn, lvn = [], []
    for j in range(r, len(vol) - r):
        nb = np.concatenate([sm[j - r:j], sm[j + 1:j + r + 1]])
        if sm[j] > nb.max() and sm[j] >= HVN_MIN * mean and j != i_poc:
            hvn.append(edges[j] + size / 2)
        if sm[j] < nb.min() and i_val <= j <= i_vah and sm[j] <= LVN_MAX * mean:
            lvn.append(edges[j] + size / 2)
    c = lambda i: float(edges[i] + size / 2)              # noqa: E731
    return {"poc": [c(i_poc)], "vah": [c(i_vah)], "val": [c(i_val)], "hvn": hvn, "lvn": lvn}


def _slice(m1: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    a, b = m1.index.searchsorted(start), m1.index.searchsorted(end)
    return m1.iloc[a:b]


def window_levels(m1: pd.DataFrame, windows: list[Window], n_rows: int = N_ROWS) -> list[Level]:
    """Levels ``"<window>:<type>"`` of every window with enough data, known at its end."""
    out = []
    for w in windows:
        g = _slice(m1, w.start, w.end)
        if len(g) == 0 or g["close"].notna().mean() < MIN_COVERAGE:
            continue
        nodes = profile_nodes(g, n_rows)
        if nodes is None:
            continue
        t = known_at(w)
        for typ, prices in nodes.items():
            out.extend(Level(f"{w.name}:{typ}", float(p), t) for p in prices)
    return out


def naked_pocs(m1: pd.DataFrame, n_rows: int = N_ROWS) -> list[Level]:
    """POC of UTC day D that day D+1 never traded through; known at 00:00 of D+2."""
    days = period_windows(m1, "day")
    pocs = {}
    for w in days:
        g = _slice(m1, w.start, w.end)
        if len(g) and g["close"].notna().mean() >= MIN_COVERAGE:
            nodes = profile_nodes(g, n_rows)
            if nodes is not None:
                pocs[w.start] = nodes["poc"][0]
    out = []
    one = pd.Timedelta(days=1)
    for d, poc in pocs.items():
        nxt = _slice(m1, d + one, d + 2 * one)
        if len(nxt) == 0 or nxt["close"].notna().mean() < MIN_COVERAGE:
            continue
        if np.nanmin(nxt["low"]) > poc or np.nanmax(nxt["high"]) < poc:
            out.append(Level("naked:poc", float(poc), d + 2 * one))
    return out


# ── shared arrays ───────────────────────────────────────────────────────
class Path:
    """1-min arrays the event functions share; ``atr`` is ATR(1h) known at each bar's START
    (i.e. from completed hours only)."""

    def __init__(self, m1: pd.DataFrame, atr_1h: pd.Series):
        self.idx = m1.index
        self.step = m1.index[1] - m1.index[0]
        self.hi = m1["high"].to_numpy(float)
        self.lo = m1["low"].to_numpy(float)
        self.close = m1["close"].to_numpy(float)
        self.buy = m1["buy_notional"].to_numpy(float)
        self.sell = m1["sell_notional"].to_numpy(float)
        self.atr = atr_1h.reindex(m1.index).to_numpy(float)
        self.nan_cum = np.concatenate([[0], np.cumsum(~np.isfinite(self.close))])

    def bars(self, td: str | pd.Timedelta) -> int:
        return int(pd.Timedelta(td) / self.step)

    def complete(self, a: int, b: int) -> bool:
        """Bars a .. b-1 all exist and have prices."""
        return 0 <= a and b <= len(self.close) and self.nan_cum[b] - self.nan_cum[a] == 0


def atr_1h(m1: pd.DataFrame) -> pd.Series:
    """ATR(14) of 1-h bars, shifted so a value is used only after its hour has ended."""
    from .indicators import atr
    h1 = m1[["open", "high", "low", "close"]].resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    a = atr(h1["high"], h1["low"], h1["close"], 14)
    a.index = a.index + pd.Timedelta("1h")
    return a.reindex(m1.index, method="ffill")


def _first_hit(p: Path, level: float, side: int, a: int, b: int) -> int:
    """First bar in a .. b-1 whose high (side +1) / low (side -1) reaches ``level``; -1 if none."""
    seg = p.hi[a:b] >= level if side > 0 else p.lo[a:b] <= level
    hit = np.flatnonzero(seg)
    return int(hit[0]) if len(hit) else -1


def _start(p: Path, lv: Level) -> tuple[int, float, float, int] | None:
    """Bar k0 at ``known_at``, last known close p0, ATR, side; None if the level is unusable
    (no data, inside the last bar's range, outside the distance band)."""
    k0 = int(p.idx.searchsorted(lv.known_at))
    if k0 < 1 or k0 >= len(p.close):
        return None
    p0, a = p.close[k0 - 1], p.atr[k0 - 1]
    if not (np.isfinite(p0) and np.isfinite(a) and a > 0):
        return None
    if p.lo[k0 - 1] <= lv.price <= p.hi[k0 - 1]:
        return None
    dist = abs(lv.price - p0) / a
    if not MIN_DIST_ATR <= dist <= MAX_DIST_ATR:
        return None
    return k0, p0, a, 1 if lv.price > p0 else -1


# ── (a) magnet ──────────────────────────────────────────────────────────
def magnet_events(p: Path, levels: list[Level], horizons: dict[str, str], *, n_ctl: int = 5,
                  min_gap: str = "24h", seed: int = 0) -> pd.DataFrame:
    """Did price reach the level within each horizon after it became known?

    Controls per event: ``n_ctl`` random start bars t' in the same calendar month, at least
    ``min_gap`` from t0, with a level at the same distance (in ATR) and side from p(t');
    ``ctl_<h>`` = their mean hit rate. ``mir_<h>`` = the mirror level (same time, same
    distance, other side). ``eff_<h>`` = hit - ctl.
    """
    rng = np.random.default_rng(seed)
    hb = {k: p.bars(v) for k, v in horizons.items()}
    hmax = max(hb.values())
    gap = p.bars(min_gap)
    month = p.idx.year * 12 + p.idx.month
    month_pos = {m: np.flatnonzero(month == m) for m in np.unique(month)}
    rows = []
    for lv in levels:
        st = _start(p, lv)
        if st is None:
            continue
        k0, p0, a, s = st
        if not p.complete(k0 - 1, k0 + hmax):
            continue
        dist = abs(lv.price - p0) / a
        f = _first_hit(p, lv.price, s, k0, k0 + hmax)
        fm = _first_hit(p, p0 - s * dist * a, -s, k0, k0 + hmax)
        cand = month_pos[month[k0]]
        cand = cand[(np.abs(cand - k0) >= gap) & (cand >= 1)]
        ctl_f = []
        for _ in range(50):                                     # draws until n_ctl usable ones
            if len(ctl_f) == n_ctl or len(cand) == 0:
                break
            k = int(cand[rng.integers(len(cand))])
            pc, ac = p.close[k - 1], p.atr[k - 1]
            if not (np.isfinite(pc) and np.isfinite(ac) and ac > 0) or not p.complete(k - 1, k + hmax):
                continue
            ctl_f.append(_first_hit(p, pc + s * dist * ac, s, k, k + hmax))
        if len(ctl_f) < n_ctl:
            continue
        ctl_f = np.array(ctl_f)
        row = {"ts": lv.known_at, "kind": lv.kind, "level": lv.price, "side": s, "dist_atr": dist}
        for name, n in hb.items():
            hit = float(0 <= f < n)
            row[f"hit_{name}"] = hit
            row[f"ctl_{name}"] = float(np.mean((ctl_f >= 0) & (ctl_f < n)))
            row[f"mir_{name}"] = float(0 <= fm < n)
            row[f"eff_{name}"] = hit - row[f"ctl_{name}"]
        rows.append(row)
    return pd.DataFrame(rows)


# ── (b) hold ────────────────────────────────────────────────────────────
def hold_events(p: Path, levels: list[Level], *, hold_atr: float = HOLD_ATR,
                window: str = HOLD_WINDOW, max_age: str = MAX_AGE) -> pd.DataFrame:
    """First touch within ``max_age`` of ``known_at``, then which comes first within ``window``:

    * break  = price ``hold_atr`` x ATR beyond the level (checked from the touch bar on),
    * bounce = price ``hold_atr`` x ATR back on the approach side (from the bar AFTER the touch).
    ``outcome`` is "bounce", "break" or "unresolved" (same bar, or neither). ``hold`` = 1 / 0 /
    NaN. ``flow`` = taker imbalance over the ``FLOW_BARS`` bars ending with the touch bar;
    ``fade_bp`` = -d x log return from the touch bar's close to ``window`` later, in bp.
    """
    n_age, n_win = p.bars(max_age), p.bars(window)
    rows = []
    for lv in levels:
        st = _start(p, lv)
        if st is None:
            continue
        k0, _, _, d = st
        f = _first_hit(p, lv.price, d, k0, min(k0 + n_age, len(p.close)))
        if f < 0:
            continue
        i = k0 + f
        # no gap between known_at and the touch (a gap could hide an earlier touch), and
        # complete data over the flow window and the outcome window
        if not p.complete(min(k0 - 1, i - FLOW_BARS + 1), i + n_win + 1):
            continue
        a = p.atr[i - 1] if i >= 1 else np.nan
        if not (np.isfinite(a) and a > 0):
            continue
        brk = _first_hit(p, lv.price + d * hold_atr * a, d, i, i + n_win)
        bnc = _first_hit(p, lv.price - d * hold_atr * a, -d, i + 1, i + n_win)
        bnc = bnc + 1 if bnc >= 0 else -1                        # relative to the touch bar
        if brk < 0 and bnc < 0 or brk == bnc:
            outcome = "unresolved"
        elif bnc < 0 or 0 <= brk < bnc:
            outcome = "break"
        else:
            outcome = "bounce"
        bb = p.buy[i - FLOW_BARS + 1:i + 1].sum()
        ss = p.sell[i - FLOW_BARS + 1:i + 1].sum()
        rows.append({"ts": p.idx[i], "known_at": lv.known_at, "kind": lv.kind, "level": lv.price,
                     "d": d, "outcome": outcome,
                     "hold": {"bounce": 1.0, "break": 0.0}.get(outcome, np.nan),
                     "flow": (bb - ss) / (bb + ss) if bb + ss > 0 else np.nan,
                     "fade_bp": -d * np.log(p.close[i + n_win] / p.close[i]) * 1e4})
    return pd.DataFrame(rows)


# ── (c) SMC confluence ──────────────────────────────────────────────────
def smc_reference(m15: pd.DataFrame, res: ms.StructureResult) -> list[tuple[float, int, int]]:
    """(price, first 15-min bar index it is active, last index exclusive) for unbroken external
    swings, untaken EQH/EQL pools and order-block facing edges (as ``level_flow``)."""
    n = len(m15)
    out = []
    for s in res.swings.get("ext", []):
        out.append((float(s.price), s.confirmed_idx, s.broken_idx if s.broken_idx is not None else n))
    for pl in res.pools:
        out.append((float(pl.level), pl.created_idx, pl.taken_idx if pl.taken_idx is not None else n))
    for z in res.zones:
        if z.kind == "ob":
            end = z.invalidated_idx if z.invalidated_idx is not None else n
            out.append((float(z.bottom if z.direction < 0 else z.top), z.created_idx, end))
    return out


def smc_confluence(levels: pd.DataFrame, m15: pd.DataFrame, ref: list[tuple[float, int, int]],
                   atr_15: pd.Series, *, tol: float = CONFLUENCE_ATR) -> np.ndarray:
    """True where an SMC level active at ``known_at`` lies within ``tol`` x ATR(15 min) of
    the profile level. A 15-min bar j is usable once it has closed (its end <= known_at)."""
    step = m15.index[1] - m15.index[0]
    ends = m15.index + step
    a = atr_15.to_numpy(float)
    px = np.array([r[0] for r in ref])
    first = np.array([r[1] for r in ref])
    last = np.array([r[2] for r in ref])
    out = np.zeros(len(levels), dtype=bool)
    for n, (t, lvl) in enumerate(zip(levels["known_at"], levels["level"])):
        j = int(ends.searchsorted(t, side="right")) - 1
        if j < 0 or not np.isfinite(a[j]) or len(px) == 0:
            continue
        live = (first <= j) & (last > j)
        out[n] = bool(np.any(live & (np.abs(px - lvl) <= tol * a[j])))
    return out


# ── statistics ──────────────────────────────────────────────────────────
def nw_mean_t(x, lags: int = 5) -> float:
    """t of the mean of an evenly spaced series with Newey-West (Bartlett) variance - the
    estimator of ``hac.newey_west_ols`` for a regression on a constant only."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    u = x - x.mean()
    s = float(u @ u)
    for k in range(1, min(lags, n - 1) + 1):
        s += 2.0 * (1.0 - k / (lags + 1.0)) * float(u[k:] @ u[:-k])
    if s <= 0:
        return float("nan")
    return float(x.mean() / np.sqrt(s / n ** 2))


def _per_day(x, ts) -> pd.Series:
    return pd.Series(np.asarray(x, float)).groupby(pd.DatetimeIndex(ts).floor("1D")).mean().sort_index()


def cell_stats(x, ts, lags: int = 5) -> dict[str, float]:
    """n, mean, day-clustered t, NW t on per-day means, and the smaller |t| (verdict t)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, ts = x[ok], pd.DatetimeIndex(ts)[ok]
    if len(x) < 10:
        return {"n": len(x), "mean": float("nan"), "t_day": float("nan"), "t_nw": float("nan"),
                "t": float("nan")}
    t_day = lf.day_clustered_t(pd.Series(x), pd.Series(ts))
    pdm = _per_day(x, ts)
    full = pdm.reindex(pd.date_range(pdm.index[0], pdm.index[-1], freq="1D"))
    t_nw = nw_mean_t(full.dropna().to_numpy(), lags)
    return {"n": len(x), "mean": float(x.mean()), "t_day": t_day, "t_nw": t_nw,
            "t": _smaller(t_day, t_nw)}


def diff_stats(xa, tsa, xb, tsb, lags: int = 5) -> dict[str, float]:
    """mean(a) - mean(b): cluster-by-day OLS t (``level_flow.cluster_diff_t``) and a NW t on the
    per-day differences (days with both); the verdict uses the smaller |t|."""
    xa, xb = np.asarray(xa, float), np.asarray(xb, float)
    ma, mb = np.isfinite(xa), np.isfinite(xb)
    xa, tsa, xb, tsb = xa[ma], pd.DatetimeIndex(tsa)[ma], xb[mb], pd.DatetimeIndex(tsb)[mb]
    if len(xa) < 10 or len(xb) < 10:
        return {"n_a": len(xa), "n_b": len(xb), "a": float("nan"), "b": float("nan"),
                "diff": float("nan"), "t_cl": float("nan"), "t_nw": float("nan"), "t": float("nan")}
    diff, t_cl = lf.cluster_diff_t(xa, tsa, xb, tsb)
    da, db = _per_day(xa, tsa), _per_day(xb, tsb)
    both = (da - db).dropna()
    t_nw = nw_mean_t(both.to_numpy(), lags)
    return {"n_a": len(xa), "n_b": len(xb), "a": float(xa.mean()), "b": float(xb.mean()),
            "diff": diff, "t_cl": t_cl, "t_nw": t_nw, "t": _smaller(t_cl, t_nw)}


def _smaller(a: float, b: float) -> float:
    if not np.isfinite(a):
        return b
    if not np.isfinite(b):
        return a
    return a if abs(a) <= abs(b) else b


#: verdict thresholds (pre-registered 2026-10-29)
MIN_EFFECT, T_DISCOVERY, T_CONFIRM = 0.05, 3.0, 2.0


def cell_verdict(disc: dict, conf: dict, conf_by_market: dict[str, float], *,
                 effect: str = "mean") -> bool:
    """Confirmed iff 2024 |effect| >= 5 pp with |t| >= 3; 2023 same sign, |t| >= 2, effect >=
    half of 2024; same sign in every market on 2023."""
    e1, t1, e2, t2 = disc[effect], disc["t"], conf[effect], conf["t"]
    if not all(np.isfinite(v) for v in (e1, t1, e2, t2)):
        return False
    sign = np.sign(e1)
    return bool(abs(e1) >= MIN_EFFECT and abs(t1) >= T_DISCOVERY
                and np.sign(e2) == sign and abs(t2) >= T_CONFIRM and abs(e2) >= abs(e1) / 2
                and all(np.isfinite(v) and np.sign(v) == sign for v in conf_by_market.values()))
