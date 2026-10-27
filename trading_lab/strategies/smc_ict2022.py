"""S-001: the ICT 2022 model on the shared market-structure framework.

Spec (agreed with the user 2026-10-27): ``docs/specs/smc_ict2022.yaml``. Every rule below
is a small function with its own test in ``tests/test_smc_ict2022.py``.

Long (short mirrored):
 1. bias: external structure of the higher timeframe is up - only CLOSED HTF bars;
 2. killzone: London 02:00-05:00 / NY 08:30-11:00 New York time (per market);
 3. sweep of sell-side liquidity on the LTF (wick below an unbroken swing low or an EQL
    pool, close back above) inside the killzone;
 4. within ``mss_window`` LTF bars the first internal break up - CHoCH or BOS (MSS);
 5. the leg from the sweep to the MSS left a bullish FVG (>= 0.5 ATR, displacement body
    >= 0.8 ATR) - or, in the ``ifvg`` variant, a bullish inverse FVG;
 6. limit at the FVG CE (or its proximal edge), stop = sweep extreme - 0.1 ATR,
    T1 = nearest unbroken internal LTF swing high (50 % + break-even), T2 = nearest
    unbroken external HTF swing high (or a fixed 3 R); T2 must be >= ``min_rr`` R away;
 7. one setup per market and killzone; flat at the session end.

Fills on 1-minute bid/ask: a long limit fills only when the ASK trades THROUGH it, exits
on the bid; stops get ``slippage_bp``. Within one minute the stop is checked before any
target (conservative).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..indicators import atr as _atr
from ..market_structure import (BOS_DN, BOS_UP, CHOCH_DN, CHOCH_UP, StructureConfig, analyze, mid_ohlc,
                                raw_pivots)
from ..mr_common import resample_ohlc

NY = "America/New_York"
KILLZONES = {"london": ("02:00", "05:00"), "ny": ("08:30", "11:00")}
MARKET_ZONES = {"US100": ("ny",), "SP500": ("ny",), "GER40": ("london",),
                "XAUUSD": ("london", "ny"), "EURUSD": ("london", "ny")}
FLAT = {"US100": (NY, "15:55"), "SP500": (NY, "15:55"), "GER40": ("Europe/Berlin", "17:25"),
        "XAUUSD": (NY, "16:55"), "EURUSD": (NY, "16:55")}
COMBOS = {"1h/5m": ("1h", "5min"), "4h/15m": ("4h", "15min")}
LTF_STRUCTURE = StructureConfig(fvg_min_atr=0.5, displacement_atr=0.8)
HTF_STRUCTURE = StructureConfig()


@dataclass(frozen=True)
class SmcIct2022Config:
    asset: str = "US100"
    zeitrahmen: str = "1h/5m"        # 1h/5m | 4h/15m
    einstieg: str = "ce"             # ce | fvg_kante
    min_rr: float = 2.0
    ziel: str = "liquiditaet"        # liquiditaet | fest_3r
    setup: str = "fvg"               # fvg | ifvg
    bias: str = "an"                 # an | aus (control)
    mss_window: int = 12             # LTF bars from sweep to MSS
    order_valid: int = 12            # LTF bars the limit stays
    sl_buffer_atr: float = 0.1
    min_stop_atr: float = 0.5
    max_stop_atr: float = 4.0
    t1_close: float = 0.5
    slippage_bp: float = 0.5


# ── 2. killzones, session end ─────────────────────────────────────────────

def _hm(s: str) -> int:
    return int(s[:2]) * 60 + int(s[3:])


def killzone_ids(index: pd.DatetimeIndex, asset: str) -> np.ndarray:
    """0 = none, 1 = London, 2 = NY - by the bar's START time in New York (DST-aware)."""
    ny = index.tz_convert(NY)
    mins = ny.hour * 60 + ny.minute
    out = np.zeros(len(index), dtype=int)
    for zid, name in ((1, "london"), (2, "ny")):
        if name not in MARKET_ZONES.get(asset, ("london", "ny")):
            continue
        a, b = (_hm(x) for x in KILLZONES[name])
        out[(mins >= a) & (mins < b)] = zid
    return out


def flat_time_ns(ts: pd.Timestamp, asset: str) -> int:
    """Epoch-ns of the session end on the local trading day of ``ts``."""
    tz, hhmm = FLAT.get(asset, (NY, "16:55"))
    loc = ts.tz_convert(tz)
    end = loc.normalize() + pd.Timedelta(minutes=_hm(hhmm))
    return int(end.tz_convert("UTC").value)


# ── 1. bias from closed HTF bars only ─────────────────────────────────────

def htf_bias_at(ltf_close_ns: np.ndarray, htf_close_ns: np.ndarray, htf_trend: np.ndarray) -> np.ndarray:
    """Trend of the last HTF bar that CLOSED at or before each LTF bar's close (0 if none)."""
    k = np.searchsorted(htf_close_ns, ltf_close_ns, side="right") - 1
    out = np.zeros(len(ltf_close_ns), dtype=int)
    ok = k >= 0
    out[ok] = htf_trend[k[ok]]
    return out


# ── 6. targets ─────────────────────────────────────────────────────────────

TARGET_LOOKBACK = 300                # bars back a target pivot may lie


def nearest_unbroken(pivot_idx: np.ndarray, pivot_conf: np.ndarray, pivot_price: np.ndarray,
                     close: np.ndarray, asof: int, entry: float, direction: int,
                     lookback: int = TARGET_LOOKBACK) -> float:
    """Nearest pivot beyond ``entry`` (above for longs), confirmed by ``asof``, no older than
    ``lookback`` bars, that no close in (pivot, asof] has broken. NaN if there is none.
    Pivots are ordered by confirmation, so the window is found by bisection."""
    best = np.nan
    hi = int(np.searchsorted(pivot_conf, asof, side="right"))
    lo = int(np.searchsorted(pivot_conf, asof - lookback, side="left"))
    for j in range(lo, hi):
        p = pivot_price[j]
        if direction > 0 and p <= entry or direction < 0 and p >= entry:
            continue
        seg = close[pivot_idx[j] + 1: asof + 1]
        broken = (seg > p).any() if direction > 0 else (seg < p).any()
        if broken:
            continue
        if not np.isfinite(best) or (direction > 0 and p < best) or (direction < 0 and p > best):
            best = p
    return float(best)


def choose_targets(entry: float, risk: float, direction: int, t1: float, t2: float,
                   cfg: SmcIct2022Config) -> tuple[float, float] | None:
    """(T1, T2) or None if the trade does not pay ``min_rr``. T1 = NaN means one exit at T2."""
    if cfg.ziel == "fest_3r":
        return np.nan, entry + direction * 3.0 * risk
    if not np.isfinite(t2):
        return None
    if direction * (t2 - entry) / risk < cfg.min_rr:
        return None
    if not np.isfinite(t1) or direction * (t1 - t2) >= 0:
        t1 = np.nan
    return t1, t2


# ── 3.-6. setups on the LTF ───────────────────────────────────────────────

@dataclass
class Setup:
    direction: int
    sweep_idx: int
    mss_idx: int
    extreme: float
    entry: float
    stop: float
    t1: float
    t2: float
    zone_key: tuple
    ready_idx: int = -1                  # LTF bar whose close makes the setup known (>= mss)


@dataclass
class Prepared:
    asset: str
    m1_ns: np.ndarray
    bid: dict[str, np.ndarray]
    ask: dict[str, np.ndarray]
    combos: dict[str, dict] = field(default_factory=dict)


def prepare(m1: pd.DataFrame, asset: str, combos: tuple[str, ...] = tuple(COMBOS)) -> Prepared:
    """Structure on mid OHLC for every timeframe combination, computed once."""
    p = Prepared(asset=asset, m1_ns=m1.index.as_unit("ns").asi8,
                 bid={k: m1[k].to_numpy(float) for k in ("open", "high", "low", "close")},
                 ask={k: m1[f"ask_{k}"].to_numpy(float) for k in ("open", "high", "low", "close")})
    for name in combos:
        htf_tf, ltf_tf = COMBOS[name]
        ltf = mid_ohlc(resample_ohlc(m1, ltf_tf))
        htf = mid_ohlc(resample_ohlc(m1, htf_tf))
        lres, hres = analyze(ltf, LTF_STRUCTURE), analyze(htf, HTF_STRUCTURE)
        ldur, hdur = pd.Timedelta(ltf_tf).value, pd.Timedelta(htf_tf).value
        l_close = ltf.index.as_unit("ns").asi8 + ldur
        h_close = htf.index.as_unit("ns").asi8 + hdur
        ip = raw_pivots(ltf["high"].to_numpy(), ltf["low"].to_numpy(), LTF_STRUCTURE.int_left, LTF_STRUCTURE.int_right)
        ep = raw_pivots(htf["high"].to_numpy(), htf["low"].to_numpy(), HTF_STRUCTURE.ext_left, HTF_STRUCTURE.ext_right)
        piv = lambda lst, k: tuple(np.array([getattr(s, a) for s in lst if s.kind == k], dtype=float if a == "price" else int)  # noqa: E731
                                   for a in ("idx", "confirmed_idx", "price"))
        p.combos[name] = {
            "ltf": ltf, "res": lres, "ldur": ldur, "l_close": l_close,
            "atr": _atr(ltf["high"], ltf["low"], ltf["close"], LTF_STRUCTURE.atr_period).to_numpy(),
            "kz": killzone_ids(ltf.index, asset),
            "day": ltf.index.tz_convert(NY).normalize().asi8,
            "bias": htf_bias_at(l_close, h_close, hres.bars["ext_trend"].to_numpy()),
            "int_hi": piv(ip, 1), "int_lo": piv(ip, -1),
            "htf_hi": piv(ep, 1), "htf_lo": piv(ep, -1),
            "htf_close": htf["close"].to_numpy(float), "h_close_ns": h_close,
        }
    return p


def _pick_zone(res, kind: str, direction: int, lo: int, hi: int):
    cands = [z for z in res.zones if z.kind == kind and z.direction == direction
             and lo <= z.created_idx <= hi and z.active_at(hi)]
    return max(cands, key=lambda z: z.created_idx) if cands else None


def find_setups(prep: Prepared, cfg: SmcIct2022Config) -> list[Setup]:
    """All valid setups. The FVG of the displacement leg may be confirmed one bar AFTER the
    MSS: when the MSS candle is the impulse (the FVG's middle candle), the gap is only known
    at the close of the next bar. "One trade per killzone" is applied in ``run_backtest``
    (to FILLED trades), not here."""
    d = prep.combos[cfg.zeitrahmen]
    res, ltf = d["res"], d["ltf"]
    b = res.bars
    # spec rule 3: a wick below ANY unbroken 5m swing low (external or internal) or an EQL pool
    sw_lo = np.maximum(b["sweep_low"].to_numpy(), b["int_swing_sweep_low"].to_numpy())
    sw_hi = np.maximum(b["sweep_high"].to_numpy(), b["int_swing_sweep_high"].to_numpy())
    iev = b["int_event"].to_numpy()
    lo, hi, cl = ltf["low"].to_numpy(), ltf["high"].to_numpy(), ltf["close"].to_numpy()
    out: list[Setup] = []
    state: dict | None = None
    for i in range(len(ltf)):
        if state is not None:
            if "mss" in state:                          # second chance: FVG confirmed at mss + 1
                s = _build(prep, cfg, d, state, state["mss"], i, cl)
                if s is not None:
                    out.append(s)
                state = None
            elif i - state["i"] > cfg.mss_window:
                state = None
            else:
                state["x"] = min(state["x"], lo[i]) if state["d"] > 0 else max(state["x"], hi[i])
                # spec rule 4 (changed 2026-10-27 with the user): the FIRST internal break in
                # the trade direction after the sweep - CHoCH or BOS
                want = (CHOCH_UP, BOS_UP) if state["d"] > 0 else (CHOCH_DN, BOS_DN)
                if iev[i] in want:
                    s = _build(prep, cfg, d, state, i, i, cl)
                    if s is not None:
                        out.append(s)
                        state = None
                    else:
                        state["mss"] = i
                continue
        if state is not None:
            continue
        kz = d["kz"][i]
        if not kz:
            continue
        key = (int(d["day"][i]), int(kz))
        bias = d["bias"][i]
        dl = sw_lo[i] == 1 and (bias > 0 or cfg.bias == "aus")
        ds = sw_hi[i] == 1 and (bias < 0 or cfg.bias == "aus")
        if dl == ds:
            continue
        dirn = 1 if dl else -1
        state = {"d": dirn, "i": i, "x": lo[i] if dirn > 0 else hi[i], "key": key}
    return out


def _build(prep: Prepared, cfg: SmcIct2022Config, d: dict, st: dict, mss: int, ready: int,
           cl: np.ndarray) -> Setup | None:
    """Setup known at the close of ``ready`` (mss or mss + 1). Zone created in
    [sweep, ready]; ATR and targets as of ``ready`` - all known by then."""
    dirn = st["d"]
    z = _pick_zone(d["res"], "fvg" if cfg.setup == "fvg" else "ifvg", dirn, st["i"], ready)
    if z is None:
        return None
    a = d["atr"][ready]
    if not np.isfinite(a) or a <= 0:
        return None
    if cfg.einstieg == "ce":
        entry = (z.top + z.bottom) / 2
    else:
        entry = z.top if dirn > 0 else z.bottom
    stop = st["x"] - dirn * cfg.sl_buffer_atr * a
    risk = dirn * (entry - stop)
    if not (cfg.min_stop_atr * a <= risk <= cfg.max_stop_atr * a):
        return None
    side = "hi" if dirn > 0 else "lo"
    t1 = nearest_unbroken(*d[f"int_{side}"], cl, ready, entry, dirn)
    k = int(np.searchsorted(d["h_close_ns"], d["l_close"][ready], side="right") - 1)
    t2 = nearest_unbroken(*d[f"htf_{side}"], d["htf_close"], k, entry, dirn) if k >= 0 else np.nan
    tg = choose_targets(entry, risk, dirn, t1, t2, cfg)
    if tg is None:
        return None
    return Setup(dirn, st["i"], mss, st["x"], entry, stop, tg[0], tg[1], st["key"], ready_idx=ready)


# ── 6.-7. execution on 1-minute bid/ask ────────────────────────────────────

def simulate(s: Setup, prep: Prepared, start_ns: int, valid_until_ns: int, cfg: SmcIct2022Config) -> dict | None:
    """Limit order, then the trade. Returns a trade dict or None (not filled)."""
    t, bid, ask = prep.m1_ns, prep.bid, prep.ask
    k = int(np.searchsorted(t, start_ns, side="left"))
    n = len(t)
    if k >= n:
        return None
    flat = flat_time_ns(pd.Timestamp(t[k], tz="UTC"), prep.asset)
    d, L, stop0 = s.direction, s.entry, s.stop
    slip = cfg.slippage_bp * 1e-4
    first_target = s.t1 if np.isfinite(s.t1) else s.t2
    # exit side: longs sell the bid, shorts buy the ask; entry side the opposite
    ex_h, ex_l, ex_o = (bid["high"], bid["low"], bid["open"]) if d > 0 else (ask["high"], ask["low"], ask["open"])
    fill_k = None
    while k < n and t[k] < valid_until_ns and t[k] < flat:
        through = ask["low"][k] < L if d > 0 else bid["high"][k] > L
        if through:
            fill_k = k
            break
        mid_x = (bid["low"][k] + ask["low"][k]) / 2 if d > 0 else (bid["high"][k] + ask["high"][k]) / 2
        if d * (ex_h[k] if d > 0 else ex_l[k]) >= d * first_target or d * (mid_x - s.extreme) < 0:
            return None                       # target reached / sweep extreme broken before the fill
        k += 1
    if fill_k is None:
        return None
    stop, rest, pnl = stop0, 1.0, 0.0
    t1_done, reason = not np.isfinite(s.t1), "time"
    k = fill_k
    while k < n:
        if t[k] >= flat:
            px = ex_o[k] * (1 - d * slip)
            pnl += rest * d * (px - L)
            reason = "session"
            break
        adverse = ex_l[k] if d > 0 else ex_h[k]
        favour = ex_h[k] if d > 0 else ex_l[k]
        if d * (adverse - stop) <= 0:
            px = (min(stop, ex_o[k]) if d > 0 else max(stop, ex_o[k])) if k > fill_k else stop
            px *= (1 - d * slip)
            pnl += rest * d * (px - L)
            reason = "breakeven" if t1_done and np.isfinite(s.t1) and stop == L else "stop"
            break
        if k == fill_k:                        # order of events inside the fill minute is unknown:
            k += 1                             # only the stop may hit there, never a target
            continue
        if not t1_done and d * (favour - s.t1) >= 0:
            pnl += cfg.t1_close * d * (s.t1 - L)
            rest -= cfg.t1_close
            stop, t1_done = L, True
            k += 1
            continue
        if d * (favour - s.t2) >= 0:
            pnl += rest * d * (s.t2 - L)
            reason = "t2"
            break
        k += 1
    else:                                      # data ended with the position open
        k = n - 1
        pnl += rest * d * ((bid if d > 0 else ask)["close"][k] - L)
        reason = "data_end"
    risk = d * (L - stop0)
    return {"entry_time": pd.Timestamp(t[fill_k], tz="UTC"), "exit_time": pd.Timestamp(t[min(k, n - 1)], tz="UTC"),
            "direction": d, "entry": L, "stop": stop0, "target": s.t2, "t1": s.t1,
            "r": pnl / risk, "exit_reason": reason, "sweep_idx": s.sweep_idx, "mss_idx": s.mss_idx}


def run_backtest(prep: Prepared, cfg: SmcIct2022Config) -> dict:
    d = prep.combos[cfg.zeitrahmen]
    trades = []
    busy_until = -1
    traded_zones: set = set()
    for s in find_setups(prep, cfg):
        if s.zone_key in traded_zones:
            continue                          # one TRADE per market and killzone (spec rule 7)
        start = int(d["l_close"][s.ready_idx])
        if start < busy_until:
            continue                          # one position per market at a time
        tr = simulate(s, prep, start, start + cfg.order_valid * d["ldur"], cfg)
        if tr is not None:
            trades.append(tr)
            traded_zones.add(s.zone_key)
            busy_until = int(tr["exit_time"].value)
    return {"trades": trades}
