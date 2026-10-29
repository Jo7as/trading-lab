"""S-002: Fibonacci retracement in the trend - with a built-in control of "wrong" levels.

Spec (agreed with the user 2026-10-28): ``docs/specs/fib_retracement.yaml``. Every rule is
a small function with its own test in ``tests/test_fib_retracement.py``.

Long (short mirrored):
 1. impulse: an external BOS/CHoCH up on the impulse timeframe (1D for ``swing``, 1h for
    ``intraday``). A = protected low of the breaking leg (fib 1.0), B = highest high since
    A (fib 0.0), B moves up with new highs;
 2. impulse >= 3 x ATR(14) of the impulse timeframe; 3. no close below A;
 4. level L = B - level x (B - A): 0.5 / 0.618 / 0.705, or the CONTROL - a random level in
    [0.40, 0.75] that keeps 0.03 away from the fib numbers (fixed seed per impulse);
 5. valid for 10 impulse bars; a new break re-anchors;
 entry: ``limit`` at L (fills only on a trade-through) or ``bestaetigung`` (price reached L,
 then the first bullish confirmation bar closing above L, market at its close);
 stop: A - 0.1 ATR (``ursprung``) or fib 0.786 - 0.1 ATR(confirm tf) (``fib_0786``);
 targets: old high B, extensions -0.272 / -0.618, fixed 2R / 3R, or 50 % at B + BE + rest
 at -0.618; exits: time stop (15 daily bars) / Friday 21:00 UTC (standard account) /
 session end (intraday) / an external CHoCH against the trade; swaps per rollover.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..indicators import atr as _atr
from ..market_structure import CHOCH_DN, CHOCH_UP, StructureConfig, analyze, mid_ohlc
from ..mr_common import resample_ohlc
from .smc_ict2022 import FLAT, flat_time_ns

NY = "America/New_York"
FIB_LEVELS = (0.5, 0.618, 0.705)
CONTROL_RANGE = (0.40, 0.75)
CONTROL_GAP = 0.03
COMBOS = {"swing": ("1D", "4h"), "intraday": ("1h", "15min")}
FIN_RATE = {"index": 0.025, "metal": 0.025, "fx": 0.010, "other": 0.025}


@dataclass(frozen=True)
class FibConfig:
    asset: str = "US100"
    zeitrahmen: str = "swing"          # swing | intraday
    level: str = "0.618"               # 0.5 | 0.618 | 0.705 | kontrolle
    einstieg: str = "limit"            # limit | bestaetigung
    stop: str = "ursprung"             # ursprung | fib_0786
    ziel: str = "altes_hoch"           # altes_hoch | ext_0272 | ext_0618 | fest_2r | fest_3r | teil_hoch_ext0618
    konto: str = "swing"               # swing | standard (only for zeitrahmen=swing)
    min_impulse_atr: float = 3.0
    valid_bars: int = 10
    sl_buffer_atr: float = 0.1
    max_hold_bars: int = 15            # daily bars (swing)
    slippage_bp: float = 0.5
    seed: int = 7


# ── rule 4: levels, control, targets ───────────────────────────────────────

def fib_price(a: float, b: float, level: float) -> float:
    """Price of a retracement ``level`` of the leg A (1.0) -> B (0.0); negative = extension."""
    return b - level * (b - a)


def control_level(seed: int, key: int) -> float:
    """A 'wrong' level in [0.40, 0.75], at least 0.03 away from every fib level -
    deterministic per impulse (``key``)."""
    rng = np.random.default_rng([seed, key])
    while True:
        x = float(rng.uniform(*CONTROL_RANGE))
        if all(abs(x - f) > CONTROL_GAP for f in FIB_LEVELS):
            return x


def entry_level(cfg: FibConfig, key: int) -> float:
    return control_level(cfg.seed, key) if cfg.level == "kontrolle" else float(cfg.level)


def targets(cfg: FibConfig, a: float, b: float, entry: float, risk: float, d: int) -> tuple[float, float]:
    """(T1, T2); T1 = NaN means one exit at T2."""
    z = cfg.ziel
    if z == "altes_hoch":
        return np.nan, b
    if z == "ext_0272":
        return np.nan, fib_price(a, b, -0.272)
    if z == "ext_0618":
        return np.nan, fib_price(a, b, -0.618)
    if z == "fest_2r":
        return np.nan, entry + d * 2.0 * risk
    if z == "fest_3r":
        return np.nan, entry + d * 3.0 * risk
    if z == "teil_hoch_ext0618":
        return b, fib_price(a, b, -0.618)
    raise ValueError(z)


# ── data: daily bars at 17:00 New York ───────────────────────────────────

def daily_ny17(m1: pd.DataFrame) -> pd.DataFrame:
    """Mid OHLC per trading day ending 17:00 New York (DST-aware); index = bar START (UTC)."""
    mid = mid_ohlc(m1)
    ny = mid.index.tz_convert(NY)
    day = (ny - pd.Timedelta(hours=17)).normalize()          # 17:00 local of the previous day
    g = mid.groupby(day.tz_localize(None) if day.tz is not None else day)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last()})
    start = pd.DatetimeIndex(out.index).tz_localize(NY) + pd.Timedelta(hours=17)
    out.index = start.tz_convert("UTC")
    out = out[out.index.tz_convert(NY).dayofweek != 5]        # no Saturday bars (Fri 17:00 -> Sun)
    return out


def _bars(m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    return daily_ny17(m1) if tf == "1D" else mid_ohlc(resample_ohlc(m1, tf))


def _close_ns(df: pd.DataFrame, tf: str) -> np.ndarray:
    start = df.index.as_unit("ns").asi8
    if tf != "1D":
        return start + pd.Timedelta(tf).value
    # a daily bar ends at the NEXT 17:00 NY wall-clock time (DST-aware) - adding 24 h to an
    # aware timestamp would be one hour off across a DST switch
    naive = df.index.tz_convert(NY).tz_localize(None).normalize() + pd.Timedelta(days=1, hours=17)
    return naive.tz_localize(NY).tz_convert("UTC").as_unit("ns").asi8


# ── rules 1-3, 5: the impulse ──────────────────────────────────────────────

@dataclass
class Impulse:
    direction: int
    a: float
    b: float
    start: int                          # impulse bar of the break (known at its close)
    a_idx: int
    valid_until: int
    traded: bool = False
    alive: bool = True


def track_impulses(high, low, close, atr_, ext_event, prot_low, prot_high, cfg: FibConfig) -> list[Impulse | None]:
    """The impulse active at the CLOSE of each impulse bar (None if none). Re-anchors on
    every new external break; B follows new extremes; dies on a close beyond A or expiry.
    Size (>= min_impulse_atr x ATR) is checked on the live leg each bar."""
    n = len(close)
    out: list[Impulse | None] = [None] * n
    cur: Impulse | None = None
    for k in range(n):
        ev = ext_event[k]
        if ev > 0 and np.isfinite(prot_low[k]):
            a = float(prot_low[k])
            ai = int(np.flatnonzero(low[: k + 1] == a)[-1]) if (low[: k + 1] == a).any() else k
            cur = Impulse(1, a, float(high[ai:k + 1].max()), k, ai, k + cfg.valid_bars)
        elif ev < 0 and np.isfinite(prot_high[k]):
            a = float(prot_high[k])
            ai = int(np.flatnonzero(high[: k + 1] == a)[-1]) if (high[: k + 1] == a).any() else k
            cur = Impulse(-1, a, float(low[ai:k + 1].min()), k, ai, k + cfg.valid_bars)
        elif cur is not None:
            if cur.direction > 0:
                cur.b = max(cur.b, float(high[k]))
                if close[k] < cur.a:
                    cur.alive = False
            else:
                cur.b = min(cur.b, float(low[k]))
                if close[k] > cur.a:
                    cur.alive = False
            if k > cur.valid_until:
                cur.alive = False
        if cur is not None and not cur.alive:
            cur = None
        big = cur is not None and np.isfinite(atr_[k]) and abs(cur.b - cur.a) >= cfg.min_impulse_atr * atr_[k]
        # a SNAPSHOT per bar: storing the live object would let bar k see the B of later
        # bars (look-ahead - caught by test_impulse_anchors_and_moving_b)
        out[k] = replace(cur) if big else None
    return out


# ── preparation ────────────────────────────────────────────────────────────

@dataclass
class Prepared:
    asset: str
    m1_ns: np.ndarray
    bid: dict
    ask: dict
    combos: dict


def prepare(m1: pd.DataFrame, asset: str, combos: tuple[str, ...] = tuple(COMBOS)) -> Prepared:
    p = Prepared(asset=asset, m1_ns=m1.index.as_unit("ns").asi8,
                 bid={k: m1[k].to_numpy(float) for k in ("open", "high", "low", "close")},
                 ask={k: m1[f"ask_{k}"].to_numpy(float) for k in ("open", "high", "low", "close")},
                 combos={})
    for name in combos:
        itf, ctf = COMBOS[name]
        imp, con = _bars(m1, itf), _bars(m1, ctf)
        res = analyze(imp, StructureConfig())
        b = res.bars
        p.combos[name] = {
            "imp": imp, "itf": itf, "ctf": ctf, "i_close": _close_ns(imp, itf),
            "i_atr": _atr(imp["high"], imp["low"], imp["close"], 14).to_numpy(),
            "ext_event": b["ext_event"].to_numpy(),
            "prot_low": b["ext_protected_low"].to_numpy(), "prot_high": b["ext_protected_high"].to_numpy(),
            "con": con, "c_close": _close_ns(con, ctf),
            "c_atr": _atr(con["high"], con["low"], con["close"], 14).to_numpy(),
        }
    return p


# ── execution on 1 minute ─────────────────────────────────────────────────

def rollover_weights(entry_ns: int, exit_ns: int, triple_dow: int = 4) -> float:
    """Financing days between entry and exit: one per 17:00 NY rollover crossed, three for
    the rollover that covers the weekend - Friday for index/metal CFDs (default), Wednesday
    (``triple_dow=2``) for spot FX, whose value date jumps the weekend there."""
    if exit_ns <= entry_ns:
        return 0.0
    a = pd.Timestamp(entry_ns, tz="UTC").tz_convert(NY)
    b = pd.Timestamp(exit_ns, tz="UTC").tz_convert(NY)
    days = pd.date_range(a.normalize(), b.normalize(), freq="D", tz=NY)
    w = 0.0
    for d in days:
        r = d + pd.Timedelta(hours=17)
        if a < r <= b and d.dayofweek < 5:
            w += 3.0 if d.dayofweek == triple_dow else 1.0
    return w


def _friday_cut_ns(t_ns: int) -> int:
    t = pd.Timestamp(t_ns, tz="UTC")
    days = (4 - t.dayofweek) % 7
    cut = (t.normalize() + pd.Timedelta(days=days) + pd.Timedelta(hours=21))
    if cut.value <= t_ns:
        cut += pd.Timedelta(days=7)
    return int(cut.value)


def manage(prep: Prepared, cfg: FibConfig, d: int, fill_k: int, entry: float, stop0: float,
           t1: float, t2: float, deadline_ns: int, choch_ns: np.ndarray) -> dict:
    """Run an open position on 1m bid/ask. Stop first each minute; in the fill minute only the
    stop; T1 -> half + break-even; hard exits at ``deadline_ns`` and at an opposite CHoCH."""
    t, bid, ask = prep.m1_ns, prep.bid, prep.ask
    n = len(t)
    slip = cfg.slippage_bp * 1e-4
    ex_h, ex_l, ex_o, ex_c = ((bid["high"], bid["low"], bid["open"], bid["close"]) if d > 0
                              else (ask["high"], ask["low"], ask["open"], ask["close"]))
    stop, rest, pnl, t1_done, reason = stop0, 1.0, 0.0, not np.isfinite(t1), "data_end"
    ci = int(np.searchsorted(choch_ns, t[fill_k], side="right"))
    next_choch = choch_ns[ci] if ci < len(choch_ns) else np.iinfo(np.int64).max
    hard = min(deadline_ns, next_choch)
    k = fill_k
    while k < n:
        if t[k] >= hard:
            if hard == next_choch:                              # opposite CHoCH: market after that close
                px, reason = ex_o[k], "gegenbruch"
            else:                                               # time / Friday cut: last price BEFORE it
                px, reason = (ex_c[k - 1] if k > fill_k else ex_o[k]), "zeit"
                k = max(k - 1, fill_k)
            pnl += rest * d * (px * (1 - d * slip) - entry)
            break
        adverse = ex_l[k] if d > 0 else ex_h[k]
        favour = ex_h[k] if d > 0 else ex_l[k]
        if d * (adverse - stop) <= 0:
            px = (min(stop, ex_o[k]) if d > 0 else max(stop, ex_o[k])) if k > fill_k else stop
            pnl += rest * d * (px * (1 - d * slip) - entry)
            reason = "breakeven" if t1_done and np.isfinite(t1) and stop == entry else "stop"
            break
        if k == fill_k:
            k += 1
            continue
        if not t1_done and d * (favour - t1) >= 0:
            pnl += 0.5 * d * (t1 - entry)
            rest, stop, t1_done = 0.5, entry, True
            k += 1
            continue
        if d * (favour - t2) >= 0:
            pnl += rest * d * (t2 - entry)
            reason = "ziel"
            break
        k += 1
    else:
        k = n - 1
        pnl += rest * d * (ex_c[k] - entry)
    risk = d * (entry - stop0)
    from ..costs import asset_class
    fin = FIN_RATE[asset_class(prep.asset)] / 365.0 * rollover_weights(int(t[fill_k]), int(t[k])) * entry
    return {"entry_time": pd.Timestamp(t[fill_k], tz="UTC"), "exit_time": pd.Timestamp(t[k], tz="UTC"),
            "direction": d, "entry": entry, "stop": stop0, "target": t2, "t1": t1,
            "r": (pnl - fin) / risk, "swap_r": fin / risk, "exit_reason": reason}


def friday_entry_allowed(t_ns: int) -> bool:
    """No entry from Friday 21:00 UTC until the weekend is over: on an account that must be
    flat for the weekend such a position could not be closed in time - the Friday cut would
    point to the NEXT Friday and the trade would be held over the weekend."""
    ts = pd.Timestamp(t_ns, tz="UTC")
    return not (ts.dayofweek >= 5 or (ts.dayofweek == 4 and ts.hour >= 21))


def _may_enter(prep: Prepared, cfg: FibConfig, t_ns: int) -> bool:
    """Intraday: no entry at or after the session end of that day (it would be closed in the
    same minute - pure cost). Swing on a standard account: not after the Friday cut."""
    if cfg.zeitrahmen == "intraday":
        return t_ns < flat_time_ns(pd.Timestamp(t_ns, tz="UTC"), prep.asset)
    if cfg.konto == "standard":
        return friday_entry_allowed(t_ns)
    return True


def run_backtest(prep: Prepared, cfg: FibConfig) -> dict:
    """Walk impulse bars; between two impulse closes the order lives on 1-minute data."""
    c = prep.combos[cfg.zeitrahmen]
    imp = c["imp"]
    hi, lo, cl = imp["high"].to_numpy(), imp["low"].to_numpy(), imp["close"].to_numpy()
    imps = track_impulses(hi, lo, cl, c["i_atr"], c["ext_event"], c["prot_low"], c["prot_high"], cfg)
    ev = c["ext_event"]
    choch = {1: c["i_close"][ev == CHOCH_DN], -1: c["i_close"][ev == CHOCH_UP]}
    t, bid, ask = prep.m1_ns, prep.bid, prep.ask
    con, c_close, c_atr = c["con"], c["c_close"], c["c_atr"]
    c_open, c_cl, c_lo, c_hi = (con[x].to_numpy() for x in ("open", "close", "low", "high"))
    trades: list[dict] = []
    busy_until = -1
    touched: dict[int, bool] = {}
    traded: set[int] = set()                     # impulse starts that already had their trade
    for k in range(len(imp) - 1):
        im = imps[k]
        if im is None or im.start in traded:
            continue
        a0, b0, d = im.a, im.b, im.direction
        lvl = entry_level(cfg, im.start)
        L = fib_price(a0, b0, lvl)
        win_lo, win_hi = c["i_close"][k], c["i_close"][k + 1]
        if win_lo < busy_until:
            continue
        fill_k, entry = None, np.nan
        if cfg.einstieg == "limit":
            k0, k1 = np.searchsorted(t, win_lo, "left"), np.searchsorted(t, win_hi, "left")
            thr = ask["low"][k0:k1] < L if d > 0 else bid["high"][k0:k1] > L
            for h in np.flatnonzero(thr):
                if _may_enter(prep, cfg, int(t[k0 + h])):
                    fill_k, entry = int(k0 + h), L
                    break
        else:
            j0, j1 = np.searchsorted(c_close, win_lo, "right"), np.searchsorted(c_close, win_hi, "right")
            for j in range(j0, j1):
                if (c_lo[j] <= L) if d > 0 else (c_hi[j] >= L):
                    touched[im.start] = True
                if touched.get(im.start) and (c_cl[j] > c_open[j] and c_cl[j] > L if d > 0
                                              else c_cl[j] < c_open[j] and c_cl[j] < L):
                    fk = int(np.searchsorted(t, c_close[j], "left"))
                    if fk < len(t) and _may_enter(prep, cfg, int(t[fk])):
                        fill_k = fk
                        side = ask if d > 0 else bid
                        entry = side["open"][fk] * (1 + d * cfg.slippage_bp * 1e-4)
                    break
        if fill_k is None:
            continue
        # stop
        if cfg.stop == "ursprung":
            stop = a0 - d * cfg.sl_buffer_atr * c["i_atr"][k]
        else:
            jc = max(int(np.searchsorted(c_close, t[fill_k], "right")) - 1, 0)
            stop = fib_price(a0, b0, 0.786) - d * cfg.sl_buffer_atr * c_atr[jc]
        risk = d * (entry - stop)
        if not np.isfinite(risk) or risk <= 0:
            traded.add(im.start)
            continue
        t1, t2 = targets(cfg, a0, b0, entry, risk, d)
        if d * (t2 - entry) <= 0:
            traded.add(im.start)
            continue
        # deadlines
        if cfg.zeitrahmen == "intraday":
            deadline = flat_time_ns(pd.Timestamp(t[fill_k], tz="UTC"), prep.asset)
        else:
            kk = min(k + cfg.max_hold_bars, len(imp) - 1)
            deadline = int(c["i_close"][kk])
            if cfg.konto == "standard":
                deadline = min(deadline, _friday_cut_ns(int(t[fill_k])))
        tr = manage(prep, cfg, d, fill_k, entry, stop, t1, t2, deadline, choch[d])
        tr.update({"level": lvl, "a": a0, "b": b0, "impulse_start": im.start})
        trades.append(tr)
        traded.add(im.start)
        busy_until = int(tr["exit_time"].value)
    return {"trades": trades}
