#!/usr/bin/env python3
"""H12: mine token combinations before London / NY moves (track A) and session sentences
Asia -> London -> NY (track B). Implements ``docs/specs/h12_pattern_mining_PREREG.md``
including its amendments of 2026-10-30 - read it first; nothing here may deviate from it.

Periods: mining 2018-05..2020 (XAUUSD also 2016..2018-04), validation 2021, confirmation
2022-2024 (``--confirm``, once). Holdout 2025+ never.

  uv run python studies/h12_mining_study.py --feasibility   # counts only, no outcomes
  uv run python studies/h12_mining_study.py                 # mining + validation
  uv run python studies/h12_mining_study.py --confirm       # once, after validation
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from studies.cost_gate import load_cached  # noqa: E402
from trading_lab.chart_tokens import to_ns  # noqa: E402
from trading_lab.data_audit import build_study_frame  # noqa: E402
from trading_lab.indicators import atr, ema  # noqa: E402
from trading_lab.pattern_mining import (clustered_z, first_passage_moves, mixed_neighbors,  # noqa: E402
                                        nonoverlap_count, sax_codes, sax_windows)

MARKETS = ("US100", "SP500", "US30", "GER40", "UK100", "XAUUSD")
LDN, NYC, TKY = "Europe/London", "America/New_York", "Asia/Tokyo"
MIN = 60_000_000_000
NORM = 20
OUT = Path("artifacts/h12")
CACHE = OUT / "rows.pkl"

# mining / validation / confirmation periods (prereg amendment 1)
MINING_START = pd.Timestamp("2018-05-01")
MINING_START_GOLD = pd.Timestamp("2016-01-01")
MINING_END = pd.Timestamp("2020-12-31")
VALID = (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"))
CONF = (pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31"))

# track A
HORIZON = 60
GROUPS_A = ("asia", "prelondon", "londonam", "ny", "b5", "b15", "b60", "trend", "prevday", "newhl", "quiet", "clock")
TOKEN_RADICES = (3, 3, 3, 3, 3)                 # SAX W=3 A=3, size, pos/vol
PHASE_MUTABLE = (True, True, True, True, False)  # position letter is not mutated
BARS_MUTABLE = (True, True, True, True, True)
CARD = {"asia": 243, "prelondon": 243, "londonam": 243, "ny": 243, "b5": 243, "b15": 243, "b60": 243,
        "trend": 2, "prevday": 3, "newhl": 3, "quiet": 2, "clock": 3}
MUT = {g: PHASE_MUTABLE for g in ("asia", "prelondon", "londonam", "ny")} | \
      {g: BARS_MUTABLE for g in ("b5", "b15", "b60")}
A_LIFT, A_Z = 1.3, 4.0
A_FREQ_PER_WEEK = 2.0

# track B
B_WORD_RADICES = (3, 3, 3, 3, 3, 3)            # SAX W=4 A=3, size, position - all mutable
B_MIN_OCC = 150
B_LIFT, B_Z = 1.3, 4.0
CLASSES = [f"{d}_{s}" for d in ("up", "down", "flat") for s in ("small", "normal", "big")] + ["reversal"]

V_LIFT, V_Z, V_MARKETS, V_TOP = 1.1, 2.0, 4, 10
C_Z = 2.58


def period_of(day: pd.Timestamp, market: str) -> str | None:
    start = MINING_START_GOLD if market == "XAUUSD" else MINING_START
    if start <= day <= MINING_END:
        return "mining"
    if VALID[0] <= day <= VALID[1]:
        return "validation"
    if CONF[0] <= day <= CONF[1]:
        return "confirmation"
    return None


def _local(day: pd.Timestamp, hhmm: str, tz: str) -> int:
    return pd.Timestamp(f"{day.date()} {hhmm}", tz=tz).tz_convert("UTC").value


def prior_median(s: pd.Series) -> pd.Series:
    """Median of the previous NORM *valid* values (a gap day does not void the next 20 days)."""
    v = s.dropna()
    return v.rolling(NORM, min_periods=NORM).median().shift(1).reindex(s.index)


def same_clock_ratio(x: np.ndarray) -> np.ndarray:
    """(days, slots) -> x / median of the previous NORM valid days at the same slot."""
    df = pd.DataFrame(x)
    return (df / df.apply(prior_median)).to_numpy()


def tercile(r: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """0/1/2 by tercile edges fitted on ``fit_mask`` rows (mining); NaN -> -1."""
    v = r[fit_mask & ~np.isnan(r)]
    edges = np.quantile(v, [1 / 3, 2 / 3]) if len(v) else np.array([np.inf, np.inf])
    out = np.searchsorted(edges, r, side="right").astype(np.int64)
    out[np.isnan(r)] = -1
    return out


# ---------------------------------------------------------------------------
# per-market features
# ---------------------------------------------------------------------------

def market_rows(market: str, years: tuple[int, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_cached(market, *years)
    m1, _ = build_study_frame(raw, market)
    ts = to_ns(m1.index)
    o = ((m1["open"] + m1["ask_open"]) / 2).to_numpy()
    h = ((m1["high"] + m1["ask_high"]) / 2).to_numpy()
    lo = ((m1["low"] + m1["ask_low"]) / 2).to_numpy()
    c = ((m1["close"] + m1["ask_close"]) / 2).to_numpy()
    v = m1["volume"].to_numpy(dtype=float)
    ref = float(np.median(c))
    mid = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": v}, index=m1.index)

    # hourly ATR / EMA (UTC hours), known at the bar end
    h1 = mid.resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    h1_end = to_ns(h1.index) + 60 * MIN
    h1_atr = atr(h1["high"], h1["low"], h1["close"], 14).to_numpy()
    h1_tr = (ema(h1["close"], 20) - ema(h1["close"], 50)).to_numpy()

    # 5 / 15-min grids (UTC aligned) with minute counts
    def grid(rule: str) -> pd.DataFrame:
        g = mid.resample(rule, label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        g["n"] = mid["close"].resample(rule, label="left", closed="left").count()
        return g
    g5, g15 = grid("5min"), grid("15min")
    g5_t0, g15_t0 = to_ns(g5.index)[0], to_ns(g15.index)[0]
    G = {"5": {k: g5[k].to_numpy(dtype=float) for k in ("high", "low", "close", "volume", "n")},
         "15": {k: g15[k].to_numpy(dtype=float) for k in ("high", "low", "close", "volume", "n")}}
    win_ph = []    # (row, phase, i0, i1) -> SAX after the loop
    win_bar = {"5": [], "15": [], "60": []}  # (row, closes)

    days = sorted({d for d in m1.index.tz_convert("UTC").normalize().tz_localize(None)})
    days = [d for d in days if d.dayofweek < 5 and period_of(d, market)]
    day_hl = mid.resample("1D").agg({"high": "max", "low": "min"})  # UTC days
    day_hl = day_hl[(day_hl.index.dayofweek < 5) & day_hl["high"].notna()]  # previous weekday
    day_hl_idx = to_ns(day_hl.index)

    nslot = 53
    rows = []
    # per (day, slot) arrays for same-clock ratios
    shape = (len(days), nslot)
    acc = {k: np.full(shape, np.nan) for k in
           ("r5", "v5", "r15", "v15", "r60", "v60", "r2h",
            "rph0", "rph1", "rph2", "rph3")}
    tok = {k: np.full(shape, -1, dtype=np.int64) for k in ("s5", "s15", "s60", "sph0", "sph1", "sph2", "sph3")}
    pos = {k: np.full(shape, -1, dtype=np.int64) for k in ("pph0", "pph1", "pph2", "pph3")}
    base = []
    for di, day in enumerate(days):
        t_open = _local(day, "08:00", LDN)
        t_last = _local(day, "15:00", NYC)
        ny_open = _local(day, "09:30", NYC)
        ldn_close = _local(day, "16:30", LDN)
        phases = [(day.value, day.value + 6 * 60 * MIN),                    # Asia: Tokyo 09-15 = 00-06 UTC
                  (day.value + 6 * 60 * MIN, t_open), (t_open, ny_open), (ny_open, None)]
        slots = np.arange(t_open, t_last + 1, 15 * MIN)[:nslot]
        # previous trading UTC day high/low
        k = int(np.searchsorted(day_hl_idx, day.value)) - 1
        pd_hi, pd_lo = (day_hl["high"].iloc[k], day_hl["low"].iloc[k]) if k >= 0 else (np.nan, np.nan)
        for si, t in enumerate(slots):
            ip = int(np.searchsorted(ts, t)) - 1
            if ip < 0 or ts[ip] < t - 5 * MIN:
                continue
            p = c[ip]
            jh = int(np.searchsorted(h1_end, t, side="right")) - 1
            if jh < 0 or np.isnan(h1_atr[jh]) or h1_atr[jh] <= 0:
                continue
            # last 1 h / 2 h ranges from 1-min bars
            a1 = int(np.searchsorted(ts, t - 60 * MIN))
            a2 = int(np.searchsorted(ts, t - 120 * MIN))
            if ip + 1 - a1 < 30 or ip + 1 - a2 < 60:
                continue
            r1h = h[a1:ip + 1].max() - lo[a1:ip + 1].min()
            acc["r2h"][di, si] = h[a2:ip + 1].max() - lo[a2:ip + 1].min()
            # outcome window (t, t + 60]
            b = int(np.searchsorted(ts, t + HORIZON * MIN))
            nxt = np.arange(ip + 1, ip + 1 + HORIZON)
            ok = nxt < b
            nxt = np.minimum(nxt, len(ts) - 1)
            fh = np.where(ok, h[nxt], np.nan)
            fl = np.where(ok, lo[nxt], np.nan)
            if ok.sum() < HORIZON // 2:
                continue
            exc = max(np.nanmax(fh) - p, p - np.nanmin(fl)) / h1_atr[jh]
            # last bars on 5 / 15 / 60 min
            g5i = int((t - g5_t0) // (5 * MIN))
            g15i = int((t - g15_t0) // (15 * MIN))
            for key, gi, n_b, need in (("5", g5i, 6, 3), ("15", g15i, 4, 8)):
                if gi - n_b < 0:
                    continue
                g = G[key]
                sl = slice(gi - n_b, gi)
                if (g["n"][sl] < need).any():
                    continue
                win_bar[key].append(((di, si), g["close"][sl]))
                acc["r" + key][di, si] = g["high"][sl].max() - g["low"][sl].min()
                acc["v" + key][di, si] = g["volume"][sl].sum()
            if g15i - 16 >= 0:
                g = G["15"]
                sl = slice(g15i - 16, g15i)
                cnt = g["n"][sl].reshape(4, 4)
                if (cnt.sum(axis=1) >= 30).all() and (cnt > 0).all():
                    win_bar["60"].append(((di, si), g["close"][sl].reshape(4, 4)[:, -1]))
                    acc["r60"][di, si] = g["high"][sl].max() - g["low"][sl].min()
                    acc["v60"][di, si] = g["volume"][sl].sum()
            # phases from Asia start up to t
            for ph, (ps, pe) in enumerate(phases):
                if ps >= t:
                    continue
                end = t if pe is None else min(pe, t)
                i0, i1 = int(np.searchsorted(ts, ps)), int(np.searchsorted(ts, end))
                if i1 - i0 < 30:
                    continue
                win_ph.append((di, si, ph, i0, i1))
                ph_hi, ph_lo = h[i0:i1].max(), lo[i0:i1].min()
                acc[f"rph{ph}"][di, si] = ph_hi - ph_lo
                pos[f"pph{ph}"][di, si] = 0 if p < ph_lo else (2 if p > ph_hi else 1)
            # new session high / low on the 15-min bar ending at t (session = since 00:00 UTC)
            i_s = int(np.searchsorted(ts, day.value))
            i_q = int(np.searchsorted(ts, t - 15 * MIN))
            newhl = -1
            if i_q - i_s > 0 and ip + 1 - i_q > 0:
                if h[i_q:ip + 1].max() > h[i_s:i_q].max():
                    newhl = 1
                elif lo[i_q:ip + 1].min() < lo[i_s:i_q].min():
                    newhl = 2
            prevday = -1 if np.isnan(pd_hi) else (0 if p < pd_lo else (2 if p > pd_hi else 1))
            clock = 0 if t < ny_open else (1 if t < ldn_close else 2)
            base.append((di, si, day, t, p, h1_atr[jh], exc, r1h / h1_atr[jh],
                         int(h1_tr[jh] > 0) if not np.isnan(h1_tr[jh]) else -1, prevday, newhl, clock, fh, fl))

    if not base:
        return pd.DataFrame(), pd.DataFrame()
    for key, lst in win_bar.items():
        if lst:
            codes = sax_codes(np.stack([x for _, x in lst]), 3, 3)
            for ((a_, b_), cd) in zip([k for k, _ in lst], codes):
                tok["s" + key][a_, b_] = cd
    if win_ph:
        w = np.array(win_ph)
        codes = sax_windows(c - ref, w[:, 3], w[:, 4], 3, 3)
        for (a_, b_, ph, _, _), cd in zip(win_ph, codes):
            tok[f"sph{ph}"][a_, b_] = cd
    di = np.array([b[0] for b in base])
    si = np.array([b[1] for b in base])
    per = np.array([period_of(b[2], market) for b in base])
    mining = per == "mining"

    def letter_from(key: str) -> np.ndarray:
        r = same_clock_ratio(acc[key])[di, si]
        return tercile(r, mining)

    df = pd.DataFrame({"market": market, "day": [b[2] for b in base], "t": [b[3] for b in base],
                       "period": per, "p": [b[4] for b in base], "atr": [b[5] for b in base],
                       "exc": [b[6] for b in base], "r1h_atr": [b[7] for b in base]})
    df["trend"] = [b[8] for b in base]
    df["prevday"] = [b[9] for b in base]
    df["newhl"] = [b[10] for b in base]
    df["clock"] = [b[11] for b in base]
    fh = np.stack([b[12] for b in base])
    fl = np.stack([b[13] for b in base])

    def token(word: np.ndarray, size: np.ndarray, last: np.ndarray) -> np.ndarray:
        out = (word * 3 + size) * 3 + last
        out[(word < 0) | (size < 0) | (last < 0)] = -1
        return out

    for key, g in (("5", "b5"), ("15", "b15"), ("60", "b60")):
        df[g] = token(tok["s" + key][di, si], letter_from("r" + key), letter_from("v" + key))
    for ph, g in enumerate(("asia", "prelondon", "londonam", "ny")):
        df[g] = token(tok[f"sph{ph}"][di, si], letter_from(f"rph{ph}"), pos[f"pph{ph}"][di, si])
    q2 = letter_from("r2h")
    df["quiet"] = np.where(q2 == 0, 1, -1)
    df["newhl"] = np.where(df["newhl"] > 0, df["newhl"], -1)
    # move threshold (90th pct of excursion / ATR on mining rows), first passage
    thr = float(np.quantile(df.loc[mining, "exc"], 0.9))
    df["move"] = first_passage_moves(fh, fl, df["p"].to_numpy(), thr * df["atr"].to_numpy())
    qe = np.quantile(df.loc[mining, "r1h_atr"], [0.2, 0.4, 0.6, 0.8])
    df["volq"] = np.searchsorted(qe, df["r1h_atr"].to_numpy(), side="right")
    df["thr"] = thr
    return df, track_b_rows(market, ts, o, h, lo, c, ref, days)


# ---------------------------------------------------------------------------
# track B rows: one per (market, day)
# ---------------------------------------------------------------------------

def track_b_rows(market: str, ts, o, h, lo, c, ref, days) -> pd.DataFrame:
    rec = []
    for day in days:
        win = {"asia": (_local(day, "09:00", TKY), _local(day, "15:00", TKY)),
               "london": (_local(day, "08:00", LDN), _local(day, "16:30", LDN)),
               "ny": (_local(day, "09:30", NYC), _local(day, "16:00", NYC))}
        r = {"day": day}
        for s, (a, b) in win.items():
            i0, i1 = int(np.searchsorted(ts, a)), int(np.searchsorted(ts, b))
            if i1 - i0 < 60:
                r[s] = None
                continue
            seg_h, seg_l = h[i0:i1], lo[i0:i1]
            # first hour
            f1 = int(np.searchsorted(ts, a + 60 * MIN))
            if f1 - i0 < 30 or i1 - f1 < 30:   # first hour / rest of the session too thin
                r[s] = None
                continue
            fh1, fl1 = seg_h[:f1 - i0].max(), seg_l[:f1 - i0].min()
            rest_h, rest_l = seg_h[f1 - i0:], seg_l[f1 - i0:]
            up_i = np.flatnonzero(rest_h > fh1)
            dn_i = np.flatnonzero(rest_l < fl1)
            first = 0
            if len(up_i) or len(dn_i):
                first = 1 if (len(up_i) and (not len(dn_i) or up_i[0] < dn_i[0])) else -1
            close = c[i1 - 1]
            rev = int((first == 1 and close < fl1) or (first == -1 and close > fh1))
            r[s] = {"i0": i0, "i1": i1,
                    "hi": seg_h.max(), "lo": seg_l.min(), "open": o[i0], "close": close, "rev": rev}
        rec.append(r)
    wins = [(r[s]["i0"], r[s]["i1"]) for r in rec for s in ("asia", "london", "ny") if r[s]]
    if wins:
        w = np.array(wins)
        codes = iter(sax_windows(c - ref, w[:, 0], w[:, 1], 4, 3).tolist())
        for r in rec:
            for s in ("asia", "london", "ny"):
                if r[s]:
                    r[s]["sax"] = next(codes)
    seq = []  # time-ordered sessions
    for r in rec:
        for s in ("asia", "london", "ny"):
            seq.append((r["day"], s, r[s]))
    df = pd.DataFrame([{"day": d, "sess": s, **(x or {})} for d, s, x in seq])
    df["market"] = market
    df["valid"] = df["hi"].notna()
    df["range"] = df["hi"] - df["lo"]
    df["period"] = [period_of(d, market) for d in df["day"]]
    mining = (df["period"] == "mining").to_numpy()
    for s in ("asia", "london", "ny"):
        m = (df["sess"] == s).to_numpy()
        rr = df.loc[m, "range"]
        med = prior_median(rr)
        df.loc[m, "ratio"] = (rr / med).to_numpy()
        df.loc[m, "med"] = med.to_numpy()
    df["size"] = -1
    for s in ("asia", "london", "ny"):
        m = (df["sess"] == s).to_numpy()
        df.loc[m, "size"] = tercile(df.loc[m, "ratio"].to_numpy(), mining[m])
    prev_hi, prev_lo = df["hi"].shift(1), df["lo"].shift(1)
    df["pos"] = np.where(df["close"] < prev_lo, 0, np.where(df["close"] > prev_hi, 2, 1))
    df.loc[prev_hi.isna() | df["close"].isna(), "pos"] = -1
    word = df["sax"].fillna(-1).astype(np.int64)
    df["word"] = np.where((word >= 0) & (df["size"] >= 0) & (df["pos"] >= 0),
                          (word * 3 + df["size"]) * 3 + df["pos"], -1)
    # outcome classes of the session itself
    flat = (df["close"] - df["open"]).abs() < 0.25 * df["med"]
    dirn = np.where(flat, "flat", np.where(df["close"] > df["open"], "up", "down"))
    sz = np.array(["small", "normal", "big"])[np.clip(df["size"].to_numpy(), 0, 2)]
    df["cls"] = [f"{a}_{b}" for a, b in zip(dirn, sz)]
    df.loc[~df["valid"] | (df["size"] < 0) | df["med"].isna(), "cls"] = None
    df["prevq"] = df["ratio"].shift(1)
    return df


def build(years: tuple[int, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    a_parts, b_parts = [], []
    for mk in MARKETS:
        t0 = time.time()
        a, b = market_rows(mk, years)
        a_parts.append(a)
        b_parts.append(b)
        print(f"  {mk}: {len(a)} decision rows, {len(b)} session rows ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(a_parts, ignore_index=True), pd.concat(b_parts, ignore_index=True)


# ---------------------------------------------------------------------------
# mining machinery (shared): units = (group tuple, set of member codes)
# ---------------------------------------------------------------------------

def control_p(df: pd.DataFrame, y: np.ndarray, strata: list[str]) -> np.ndarray:
    """Expected rate per stratum, computed on the same rows (prereg amendment 4)."""
    key = df[strata].astype(str).agg("|".join, axis=1)
    return pd.Series(y).groupby(key.to_numpy()).transform("mean").to_numpy()


def combo_codes(df: pd.DataFrame, groups: tuple[str, ...], card: dict) -> np.ndarray:
    code = np.zeros(len(df), dtype=np.int64)
    ok = np.ones(len(df), dtype=bool)
    for g in groups:
        x = df[g].to_numpy()
        ok &= x >= 0
        code = code * card[g] + np.maximum(x, 0)
    code[~ok] = -1
    return code


def family_neighbors(codes: np.ndarray, groups: tuple[str, ...], card: dict, mut: dict,
                     radices: dict) -> np.ndarray:
    """One-step neighbours of combination codes: mutate one allowed letter of one item."""
    cols = []
    place = 1
    places = {}
    for g in reversed(groups):
        places[g] = place
        place *= card[g]
    for g in groups:
        if g not in mut:
            continue
        item = (codes // places[g]) % card[g]
        nb = mixed_neighbors(item, radices[g], mut[g])
        for k in range(nb.shape[1]):
            ok = (nb[:, k] >= 0) & (codes >= 0)
            cols.append(np.where(ok, codes + (nb[:, k] - item) * places[g], -1))
    return np.stack(cols, axis=1) if cols else np.empty((len(codes), 0), dtype=np.int64)


def mine_groups(code: np.ndarray, y_list: list[np.ndarray], p_list: list[np.ndarray],
                neighbors_fn) -> dict:
    """Per unique code: counts and observed / expected sums, for exact combos and star families."""
    v = code >= 0
    uniq, inv = np.unique(code[v], return_inverse=True)
    n = np.bincount(inv, minlength=len(uniq)).astype(float)
    stats = {"n": n}
    for j, (y, p) in enumerate(zip(y_list, p_list)):
        stats[f"o{j}"] = np.bincount(inv, weights=y[v], minlength=len(uniq))
        stats[f"e{j}"] = np.bincount(inv, weights=p[v], minlength=len(uniq))
        stats[f"w{j}"] = np.bincount(inv, weights=(p * (1 - p))[v], minlength=len(uniq))
    nb = neighbors_fn(uniq)
    pos = np.searchsorted(uniq, nb)
    hit = (nb >= 0) & (pos < len(uniq))
    hit &= np.take(uniq, np.minimum(pos, len(uniq) - 1)) == nb
    fam = {}
    for k, s in stats.items():
        add = np.where(hit, s[np.minimum(pos, len(uniq) - 1)], 0.0).sum(axis=1)
        fam[k] = s + add
    return {"uniq": uniq, "exact": stats, "family": fam, "nb_pos": np.where(hit, pos, -1)}


def unit_rows(code: np.ndarray, res: dict, idx: int, family: bool) -> np.ndarray:
    members = [res["uniq"][idx]]
    if family:
        members += [res["uniq"][q] for q in res["nb_pos"][idx] if q >= 0]
    return np.isin(code, members)


# ---------------------------------------------------------------------------
# track A
# ---------------------------------------------------------------------------

A_RADICES = {g: TOKEN_RADICES for g in ("asia", "prelondon", "londonam", "ny", "b5", "b15", "b60")}


def track_a(df: pd.DataFrame, feasibility: bool) -> dict:
    mine = df[df["period"] == "mining"].reset_index(drop=True)
    weeks = sum(mine.loc[mine["market"] == m, "day"].nunique() / 5 for m in MARKETS)
    min_occ = A_FREQ_PER_WEEK * weeks
    y_up = (mine["move"] == 1).to_numpy(float)
    y_dn = (mine["move"] == -1).to_numpy(float)
    if feasibility:  # counts only - no outcome enters
        y_up = y_dn = np.zeros(len(mine))
    strata = ["market", "clock", "volq"]
    p_up = control_p(mine, y_up, strata)
    p_dn = control_p(mine, y_dn, strata)
    t_ns = mine["t"].to_numpy(np.int64)
    mk = mine["market"].to_numpy()
    cluster = (mine["market"] + "|" + mine["day"].astype(str)).to_numpy()
    tried, cands = 0, []
    group_sets = [g for k in (2, 3) for g in itertools.combinations(GROUPS_A, k)]
    for gi, groups in enumerate(group_sets):
        code = combo_codes(mine, groups, CARD)
        res = mine_groups(code, [y_up, y_dn], [p_up, p_dn],
                          lambda u: family_neighbors(u, groups, CARD, MUT, A_RADICES))
        for kind in ("exact", "family"):
            s = res[kind]
            freq_ok = s["n"] >= min_occ        # necessary for the non-overlap rule
            for j in np.flatnonzero(freq_ok):
                rows = unit_rows(code, res, j, kind == "family")
                occ = sum(nonoverlap_count(np.sort(t_ns[rows & (mk == m)]), HORIZON * MIN) for m in MARKETS)
                if occ < min_occ:
                    continue
                tried += 1
                if feasibility:
                    continue
                lift_u = s["o0"][j] / s["e0"][j] if s["e0"][j] > 0 else 0.0
                lift_d = s["o1"][j] / s["e1"][j] if s["e1"][j] > 0 else 0.0
                d = 0 if lift_u >= lift_d else 1
                lift = (lift_u, lift_d)[d]
                if lift < A_LIFT:
                    continue
                y, p = (y_up, p_up) if d == 0 else (y_dn, p_dn)
                z = clustered_z((y - p)[rows], cluster[rows])
                if z >= A_Z:
                    cands.append({"track": "A", "groups": groups, "kind": kind,
                                  "members": [int(x) for x in ([res["uniq"][j]] + ([res["uniq"][q] for q in res["nb_pos"][j] if q >= 0] if kind == "family" else []))],
                                  "dir": ("up", "down")[d], "lift": float(lift), "z": float(z),
                                  "n": int(rows.sum()), "occ": int(occ)})
        if (gi + 1) % 50 == 0:
            print(f"    A: {gi + 1}/{len(group_sets)} group sets, units tried {tried}, candidates {len(cands)}", flush=True)
    return {"min_occ": min_occ, "weeks": weeks, "tried": tried, "candidates": cands}


def directional_stats(y_d: np.ndarray, y_o: np.ndarray, p_d: np.ndarray, p_o: np.ndarray,
                      cluster: np.ndarray, market: np.ndarray) -> dict:
    """Prereg amendment: does D beat O beyond the control? s = (y_D - y_O) - (p_D - p_O)."""
    s = (y_d - y_o) - (p_d - p_o)
    by_m = {str(m): float(s[market == m].mean()) for m in np.unique(market)}
    return {"dir_excess": float(s.mean()), "dir_z": clustered_z(s, cluster), "dir_by_market": by_m,
            "dir_markets_up": sum(v > 0 for v in by_m.values())}


def evaluate_a(df: pd.DataFrame, period: str, units: list[dict]) -> list[dict]:
    d = df[df["period"] == period].reset_index(drop=True)
    y = {"up": (d["move"] == 1).to_numpy(float), "down": (d["move"] == -1).to_numpy(float)}
    p = {k: control_p(d, v, ["market", "clock", "volq"]) for k, v in y.items()}
    cluster = (d["market"] + "|" + d["day"].astype(str)).to_numpy()
    out = []
    for u in units:
        code = combo_codes(d, tuple(u["groups"]), CARD)
        rows = np.isin(code, u["members"])
        yy, pp = y[u["dir"]], p[u["dir"]]
        res = {"n": int(rows.sum())}
        if rows.sum() == 0:
            out.append(u | {period: res})
            continue
        res["lift"] = float(yy[rows].sum() / pp[rows].sum()) if pp[rows].sum() > 0 else float("nan")
        res["z"] = clustered_z((yy - pp)[rows], cluster[rows])
        per_m = {}
        for m in MARKETS:
            r = rows & (d["market"] == m).to_numpy()
            if r.sum():
                per_m[m] = float(yy[r].sum() / pp[r].sum()) if pp[r].sum() > 0 else float("nan")
        res["lift_by_market"] = per_m
        res["markets_up"] = sum(v > 1 for v in per_m.values())
        other = "down" if u["dir"] == "up" else "up"
        res |= directional_stats(yy[rows], y[other][rows], pp[rows], p[other][rows], cluster[rows],
                                 d["market"].to_numpy()[rows])
        out.append(u | {period: res})
    return out


# ---------------------------------------------------------------------------
# track B
# ---------------------------------------------------------------------------

def motif_table(b: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    """(question, target session, frame with columns w0..w{k-1}, cls, rev, prevq, market, day,
    period) for every motif type of the prereg."""
    out = []
    for mk, g in b.groupby("market", sort=False):
        g = g.reset_index(drop=True)
        idx = {s: g.index[g["sess"] == s].to_numpy() for s in ("london", "ny")}
        types = {
            "asia>london": ("london", [(-1,)]), "nyprev+asia>london": ("london", [(-2, -1)]),
            "asia>ny": ("ny", [(-2,)]), "nyprev+asia>ny": ("ny", [(-3, -2)]),
            "asia+london>ny": ("ny", [(-2, -1)]), "nyprev+asia+london>ny": ("ny", [(-3, -2, -1)]),
            "last3>london": ("london", [(-3, -2, -1)]), "last4>london": ("london", [(-4, -3, -2, -1)]),
            "last4>ny": ("ny", [(-4, -3, -2, -1)]),
        }
        for q, (tgt, offs) in types.items():
            for off in offs:
                ti = idx[tgt]
                ti = ti[ti + min(off) >= 0]
                fr = pd.DataFrame({f"w{k}": g["word"].to_numpy()[ti + o] for k, o in enumerate(off)})
                for col in ("cls", "rev", "prevq", "market", "day", "period"):
                    fr[col] = g[col].to_numpy()[ti]
                fr["prevq"] = g["ratio"].to_numpy()[ti - 1]
                out.append((q, tgt, fr))
    merged = {}
    for q, tgt, fr in out:
        merged.setdefault((q, tgt), []).append(fr)
    return [(q, t, pd.concat(v, ignore_index=True)) for (q, t), v in merged.items()]


def b_outcomes(fr: pd.DataFrame) -> dict[str, np.ndarray]:
    ys = {c: (fr["cls"] == c).to_numpy(float) for c in CLASSES[:-1]}
    ys["reversal"] = (fr["rev"] == 1).to_numpy(float)
    return ys


def b_strata(fr: pd.DataFrame) -> pd.DataFrame:
    """market x quintile of the previous session's range ratio (edges per market, all rows given)."""
    q = fr.groupby("market")["prevq"].transform(lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
    return pd.DataFrame({"market": fr["market"], "pq": q.fillna(-1).astype(int)})


def track_b(b: pd.DataFrame, feasibility: bool) -> dict:
    tried, cands = 0, []
    for q, tgt, fr in motif_table(b):
        fr = fr[(fr["period"] == "mining") & fr["cls"].notna()].reset_index(drop=True)
        k = sum(c.startswith("w") for c in fr.columns)
        groups = tuple(f"w{i}" for i in range(k))
        card = {g: 729 for g in groups}
        radices = {g: B_WORD_RADICES for g in groups}
        mut = {g: (True,) * 6 for g in groups}
        code = combo_codes(fr, groups, card)
        ys = b_outcomes(fr)
        if feasibility:  # counts only - no outcome enters
            ys = {c: np.zeros(len(fr)) for c in ys}
        st = b_strata(fr)
        ps = {c: control_p(st, y, ["market", "pq"]) for c, y in ys.items()}
        cluster = (fr["market"] + "|" + fr["day"].astype(str)).to_numpy()
        names = list(ys)
        res = mine_groups(code, [ys[c] for c in names], [ps[c] for c in names],
                          lambda u: family_neighbors(u, groups, card, mut, radices))
        for kind in ("exact", "family"):
            s = res[kind]
            for j in np.flatnonzero(s["n"] >= B_MIN_OCC):
                tried += 1
                if feasibility:
                    continue
                rows = None
                for ci, c in enumerate(names):
                    if s[f"e{ci}"][j] <= 0 or s[f"o{ci}"][j] / s[f"e{ci}"][j] < B_LIFT:
                        continue
                    if rows is None:
                        rows = unit_rows(code, res, j, kind == "family")
                    z = clustered_z((ys[c] - ps[c])[rows], cluster[rows])
                    if z >= B_Z:
                        mem = [int(res["uniq"][j])] + ([int(res["uniq"][x]) for x in res["nb_pos"][j] if x >= 0]
                                                       if kind == "family" else [])
                        cands.append({"track": "B", "question": q, "target": tgt, "groups": groups,
                                      "kind": kind, "members": mem, "dir": c,
                                      "lift": float(s[f"o{ci}"][j] / s[f"e{ci}"][j]), "z": float(z),
                                      "n": int(rows.sum())})
    return {"min_occ": B_MIN_OCC, "tried": tried, "candidates": cands}


def evaluate_b(b: pd.DataFrame, period: str, units: list[dict]) -> list[dict]:
    tables = {q: fr for q, _t, fr in motif_table(b)}
    out = []
    for u in units:
        fr = tables[u["question"]]
        fr = fr[(fr["period"] == period) & fr["cls"].notna()].reset_index(drop=True)
        ys = b_outcomes(fr)
        y = ys[u["dir"]]
        p = control_p(b_strata(fr), y, ["market", "pq"])
        code = combo_codes(fr, tuple(u["groups"]), {g: 729 for g in u["groups"]})
        rows = np.isin(code, u["members"])
        res = {"n": int(rows.sum())}
        if rows.sum():
            cluster = (fr["market"] + "|" + fr["day"].astype(str)).to_numpy()
            res["lift"] = float(y[rows].sum() / p[rows].sum()) if p[rows].sum() > 0 else float("nan")
            res["z"] = clustered_z((y - p)[rows], cluster[rows])
            per_m = {}
            for m in MARKETS:
                r = rows & (fr["market"] == m).to_numpy()
                if r.sum():
                    per_m[m] = float(y[r].sum() / p[r].sum()) if p[r].sum() > 0 else float("nan")
            res["lift_by_market"] = per_m
            res["markets_up"] = sum(v > 1 for v in per_m.values())
        out.append(u | {period: res})
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def passes(res: dict, lift_min: float, z_min: float, markets_min: int) -> bool:
    return (res.get("n", 0) > 0 and res.get("lift", 0) > lift_min and res.get("z", -9) >= z_min
            and res.get("markets_up", 0) >= markets_min)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feasibility", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.confirm:
        val = json.loads((OUT / "validation.json").read_text())
        if (OUT / "confirmation.json").exists():
            raise SystemExit("confirmation exists - it is run once")
        a, b = build((2016, 2024))
        out = {}
        for tr, ev in (("A", evaluate_a), ("B", evaluate_b)):
            units = val[tr]["selected"]
            if not units:
                out[tr] = {"selected": [], "note": "nothing to confirm"}
                continue
            res = ev(a if tr == "A" else b, "confirmation", units)
            for r in res:
                c = r["confirmation"]
                r["old_rule_passed"] = passes(c, 1.0, C_Z, V_MARKETS)
                r["confirmed"] = (c.get("dir_excess", -1) > 0 and c.get("dir_z", -9) >= C_Z
                                  and c.get("dir_markets_up", 0) >= V_MARKETS)
            out[tr] = res
            for r in res:
                c = r["confirmation"]
                print(f"{tr} {'+'.join(r['groups'])} {r['kind']} {r['dir']} n {c.get('n', 0)} lift "
                      f"{c.get('lift', float('nan')):.2f} z {c.get('z', float('nan')):.2f} (old rule "
                      f"{'pass' if r['old_rule_passed'] else 'fail'}) | direction excess "
                      f"{c.get('dir_excess', float('nan')):+.3f} z {c.get('dir_z', float('nan')):.2f} "
                      f"markets {c.get('dir_markets_up', 0)} -> {'CONFIRMED' if r['confirmed'] else 'fail'}")
        (OUT / "confirmation.json").write_text(json.dumps(out, indent=1, default=str))
        return 0

    if CACHE.exists() and not args.rebuild:
        a, b = pickle.loads(CACHE.read_bytes())
    else:
        a, b = build((2016, 2021))
        CACHE.write_bytes(pickle.dumps((a, b)))
    print(f"rows: A {len(a)} (mining {int((a['period'] == 'mining').sum())}), "
          f"B {len(b)}  ({time.time() - t0:.0f}s)", flush=True)

    ra = track_a(a, args.feasibility)
    rb = track_b(b, args.feasibility)
    print(f"A: units meeting the frequency rule {ra['tried']} (min {ra['min_occ']:.0f} non-overlapping)")
    print(f"B: units meeting the frequency rule {rb['tried']} (min {rb['min_occ']})")
    if args.feasibility:
        (OUT / "feasibility.json").write_text(json.dumps({"A": {k: v for k, v in ra.items() if k != "candidates"},
                                                          "B": {k: v for k, v in rb.items() if k != "candidates"}},
                                                         indent=1, default=str))
        return 0

    report = {}
    for tr, r, ev, data in (("A", ra, evaluate_a, a), ("B", rb, evaluate_b, b)):
        cands = sorted(r["candidates"], key=lambda u: -u["z"])
        val = ev(data, "validation", cands)
        ok = [u for u in val if passes(u["validation"], V_LIFT, V_Z, V_MARKETS)]
        ok.sort(key=lambda u: -u["validation"]["z"])
        report[tr] = {"tried": r["tried"], "min_occ": r["min_occ"], "mining_candidates": len(cands),
                      "validated": len(ok), "selected": ok[:V_TOP], "all_candidates": val}
        print(f"\nTrack {tr}: tried {r['tried']}, mining candidates {len(cands)}, validated {len(ok)}")
        for u in val[:20]:
            v = u["validation"]
            print(f"  {u.get('question', '+'.join(u['groups']))} {u['kind']} {u['dir']}  mining lift "
                  f"{u['lift']:.2f} z {u['z']:.1f} n {u['n']}  | 2021 n {v.get('n', 0)} lift "
                  f"{v.get('lift', float('nan')):.2f} z {v.get('z', float('nan')):.2f} markets>1 {v.get('markets_up', 0)}")
    (OUT / "validation.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"-> {OUT / 'validation.json'}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
