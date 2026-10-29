#!/usr/bin/env python3
"""H11 stage 1: do chart tokens announce a big London / NY move beyond the pre-session range?

Implements ``docs/specs/h11_tokens_PREREG.md`` (released 2026-10-30) - read it first; nothing
here may deviate from it. Short version:

* 7 asset-sessions, target ``big`` = top quartile of the 4 h session range / its
  20-session median (threshold frozen on the discovery years), baseline ``x_rel``.
* 200 token readings (SAX N/W/A or candles C) x TF {5,15,30,60} x volume letter on/off,
  each with a size letter; score = shrunk big rate of the token, pooled over markets.
* Primary statistic: stratified AUC inside the ``x_rel`` quintiles of each asset-session.
* Discovery 2016-2021: leave-one-year-out scores, family-wise p from a max-statistic
  permutation (labels shuffled inside asset-session x year x quintile, all 200 readings
  re-scored, maximum kept). <= 3 candidates with p_fw <= 0.05 go on.
* Confirmation 2022-2024 (``--confirm``): only those candidates, scores learned on all
  discovery years, run once.

  uv run python studies/h11_tokens_study.py              # discovery
  uv run python studies/h11_tokens_study.py --confirm    # after discovery, once
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from studies.cost_gate import load_cached  # noqa: E402
from studies.h11_move_forecast_quick import PLAN, SESSIONS, session_table  # noqa: E402
from trading_lab.chart_tokens import (bars_before, candle_str, candle_word, sax_str, sax_word,  # noqa: E402
                                      tercile_letter, to_ns)
from trading_lab.data_audit import build_study_frame  # noqa: E402

DISC = (2016, 2021)
CONF = (2022, 2024)
TFS = (5, 15, 30, 60)
NMAX = 16
SAX_N, SAX_W, SAX_A = (4, 6, 8, 12, 16), (3, 4), (3, 4)
CANDLE_C = (1, 2, 3, 4, 5)
SHRINK = 20.0
NORM = 20
OUT_DISC = Path("artifacts/h11_tokens_discovery.json")
OUT_CONF = Path("artifacts/h11_tokens_confirmation.json")


@dataclass(frozen=True)
class Reading:
    fam: str        # "S" or "K"
    tf: int
    n: int          # bars covered (S: N, K: C)
    w: int = 0      # S only
    a: int = 0      # S only
    vol: bool = False

    @property
    def key(self) -> str:
        core = f"S{self.n}w{self.w}a{self.a}" if self.fam == "S" else f"K{self.n}"
        return f"{core}@{self.tf}m{'+v' if self.vol else ''}"


def grid() -> list[Reading]:
    out = []
    for tf in TFS:
        for vol in (False, True):
            out += [Reading("S", tf, n, w, a, vol) for n in SAX_N for w in SAX_W for a in SAX_A]
            out += [Reading("K", tf, c, vol=vol) for c in CANDLE_C]
    assert len(out) == 200
    return out


# ---------------------------------------------------------------------------
# data -> one row per (asset-session, day)
# ---------------------------------------------------------------------------

def asset_session_frames(years: tuple[int, int]) -> list[dict]:
    """Per asset-session: day table (y, x_rel) and the TF bars before the open."""
    out = []
    for asset, sessions in PLAN.items():
        raw = load_cached(asset, years[0], years[1])
        m1, _ = build_study_frame(raw, asset)
        mid = {k: ((m1[k] + m1[f"ask_{k}"]) / 2).to_numpy() for k in ("open", "high", "low", "close")}
        vol = m1["volume"].to_numpy(dtype=float)
        ts = to_ns(m1.index)
        for sess in sessions:
            tz, op = SESSIONS[sess]["tz"], SESSIONS[sess]["open"]
            t = session_table(m1, tz, op)
            t0s = [pd.Timestamp(f"{d.date()} {op}", tz=tz).tz_convert("UTC").value for d in t.index]
            bars = {tf: np.stack([bars_before(ts, mid["open"], mid["high"], mid["low"], mid["close"],
                                              vol, t0, tf, NMAX) for t0 in t0s]) for tf in TFS}
            out.append({"name": f"{asset}:{sess}", "table": t, "bars": bars})
            print(f"  {asset}:{sess} {len(t)} days", flush=True)
    return out


def _ratio(x: np.ndarray) -> np.ndarray:
    """x / median of the previous NORM values (prior only)."""
    s = pd.Series(x)
    return (s / s.rolling(NORM, min_periods=NORM).median().shift(1)).to_numpy()


def build_rows(frames: list[dict], readings: list[Reading]) -> dict:
    """Pooled arrays: labels, strata, and one token-code array per reading (-1 = missing).

    Frozen on the discovery rows of each asset-session: big threshold (q75 of y), x_rel
    quintile edges, size / volume tercile edges."""
    cols = {k: [] for k in ("as_id", "year", "big", "quint", "y", "x_rel", "day")}
    codes = {r.key: [] for r in readings}
    frozen = {}
    for as_id, f in enumerate(frames):
        t = f["table"]
        year = t.index.year.to_numpy()
        disc = year <= DISC[1]
        q75 = float(np.quantile(t["y"].to_numpy()[disc], 0.75))
        qe = np.quantile(t["x_rel"].to_numpy()[disc], [0.2, 0.4, 0.6, 0.8])
        quint = np.searchsorted(qe, t["x_rel"].to_numpy(), side="right")
        frozen[f["name"]] = {"big_q75": q75, "x_rel_quintile_edges": qe.tolist()}
        cols["as_id"].append(np.full(len(t), as_id))
        cols["year"].append(year)
        cols["big"].append((t["y"].to_numpy() >= q75).astype(float))
        cols["quint"].append(quint)
        cols["y"].append(t["y"].to_numpy())
        cols["x_rel"].append(t["x_rel"].to_numpy())
        cols["day"].append(np.array([str(d.date()) for d in t.index]))

        letters = {}  # (tf, n, kind) -> tercile letter per day
        for tf in TFS:
            b = f["bars"][tf]
            for n in sorted(set(SAX_N) | set(CANDLE_C)):
                seg = b[:, NMAX - n:, :]
                rng = np.nanmax(seg[:, :, 1], axis=1) - np.nanmin(seg[:, :, 2], axis=1)
                ok = ~np.isnan(seg).any(axis=(1, 2))
                rng[~ok] = np.nan
                vsum = np.where(ok, seg[:, :, 4].sum(axis=1), np.nan)
                for kind, x in (("size", rng), ("vol", vsum)):
                    r = _ratio(x)
                    edges = tuple(np.nanquantile(r[disc], [1 / 3, 2 / 3]))
                    letters[(tf, n, kind)] = tercile_letter(r, edges)
        for r in readings:
            b = f["bars"][r.tf][:, NMAX - r.n:, :]
            size = letters[(r.tf, r.n, "size")]
            vl = letters[(r.tf, r.n, "vol")] if r.vol else np.zeros(len(t), dtype=int)
            code = np.full(len(t), -1, dtype=np.int64)
            for i in range(len(t)):
                if size[i] < 0 or vl[i] < 0 or np.isnan(b[i]).any():
                    continue
                word = (sax_word(b[i, :, 3], r.w, r.a) if r.fam == "S"
                        else candle_word(b[i, :, 0], b[i, :, 1], b[i, :, 2], b[i, :, 3]))
                code[i] = (word * 3 + size[i]) * 3 + vl[i]
            codes[r.key].append(code)
    rows = {k: np.concatenate(v) for k, v in cols.items()}
    rows["codes"] = {k: np.concatenate(v) for k, v in codes.items()}
    rows["frozen"] = frozen
    rows["names"] = [f["name"] for f in frames]
    return rows


def token_text(r: Reading, code: int) -> str:
    rest, vl = divmod(int(code), 3)
    word, size = divmod(rest, 3)
    w = sax_str(word, r.w, r.a) if r.fam == "S" else candle_str(word, r.n)
    return f"{w}/{'sml'[size]}" + (f"/v{vl + 1}" if r.vol else "")


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def strat_auc(score: np.ndarray, big: np.ndarray, strat: np.ndarray, n_strata: int
              ) -> tuple[float, np.ndarray]:
    """AUC inside each stratum (ties averaged), weighted by stratum size.

    Returns (weighted AUC, per-stratum AUC with NaN where a class is missing)."""
    ok = ~np.isnan(score)
    s, b, g = score[ok], big[ok], strat[ok]
    n = len(s)
    if n == 0:
        return float("nan"), np.full(n_strata, np.nan)
    order = np.lexsort((s, g))
    gs, ss, bs = g[order], s[order], b[order]
    change = np.r_[True, (gs[1:] != gs[:-1]) | (ss[1:] != ss[:-1])]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], n]
    avg = (starts + ends - 1) / 2.0
    tie = np.cumsum(change) - 1
    pos = np.arange(n)
    gstart = np.maximum.accumulate(np.where(np.r_[True, gs[1:] != gs[:-1]], pos, 0))
    rank = avg[tie] - gstart + 1.0
    cnt = np.bincount(gs, minlength=n_strata).astype(float)
    n1 = np.bincount(gs, weights=bs, minlength=n_strata)
    r1 = np.bincount(gs, weights=rank * bs, minlength=n_strata)
    n0 = cnt - n1
    good = (n1 > 0) & (n0 > 0)
    per = np.full(n_strata, np.nan)
    per[good] = (r1[good] - n1[good] * (n1[good] + 1) / 2) / (n1[good] * n0[good])
    w = cnt * good
    return float((np.nan_to_num(per) * w).sum() / w.sum()) if w.sum() else float("nan"), per


def plain_auc(score: np.ndarray, big: np.ndarray) -> float:
    return strat_auc(score, big, np.zeros(len(score), dtype=int), 1)[0]


class LoyoScorer:
    """Leave-one-year-out shrunk big rate per token; dense ids precomputed once."""

    def __init__(self, code: np.ndarray, year: np.ndarray):
        self.valid = code >= 0
        self.ids = np.full(len(code), -1)
        uniq, inv = np.unique(code[self.valid], return_inverse=True)
        self.ids[self.valid] = inv
        self.k = len(uniq)
        yrs, self.yi = np.unique(year, return_inverse=True)
        self.ny = len(yrs)
        self.cell = self.yi * self.k + np.maximum(self.ids, 0)

    def score(self, big: np.ndarray) -> np.ndarray:
        v = self.valid
        m = self.ny * self.k
        c = np.bincount(self.cell[v], minlength=m).reshape(self.ny, self.k)
        b = np.bincount(self.cell[v], weights=big[v], minlength=m).reshape(self.ny, self.k)
        ct, bt = c.sum(0), b.sum(0)
        ny_n = c.sum(1)
        ny_b = b.sum(1)
        base = (ny_b.sum() - ny_b) / np.maximum(ny_n.sum() - ny_n, 1)  # per held-out year
        out = np.full(len(big), np.nan)
        yi, ids = self.yi[v], self.ids[v]
        ce = ct[ids] - c[yi, ids]
        be = bt[ids] - b[yi, ids]
        out[v] = (be + SHRINK * base[yi]) / (ce + SHRINK)
        return out


def fit_score(code_tr: np.ndarray, big_tr: np.ndarray, code_te: np.ndarray) -> np.ndarray:
    """Shrunk big rate learned on one sample, applied to another (unseen token -> base)."""
    v = code_tr >= 0
    base = big_tr[v].mean()
    s = pd.DataFrame({"c": code_tr[v], "b": big_tr[v]}).groupby("c")["b"].agg(["sum", "count"])
    sc = (s["sum"] + SHRINK * base) / (s["count"] + SHRINK)
    out = pd.Series(code_te).map(sc).to_numpy(dtype=float, copy=True)
    out[np.isnan(out)] = base
    out[code_te < 0] = np.nan
    return out


def shuffle_within(big: np.ndarray, group: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    o0 = np.lexsort((np.arange(len(big)), group))
    o1 = np.lexsort((rng.random(len(big)), group))
    out = np.empty_like(big)
    out[o0] = big[o1]
    return out


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def discovery(perms: int, seed: int) -> int:
    readings = grid()
    t0 = time.time()
    frames = asset_session_frames(DISC)
    rows = build_rows(frames, readings)
    n_as = len(rows["names"])
    strat = rows["as_id"] * 5 + rows["quint"]
    ns = n_as * 5
    big, year = rows["big"], rows["year"]
    pgroup = (rows["as_id"] * 100 + (year - 2000)) * 5 + rows["quint"]
    scorers = {r.key: LoyoScorer(rows["codes"][r.key], year) for r in readings}
    print(f"rows {len(big)}  base {big.mean():.3f}  built in {time.time() - t0:.0f}s", flush=True)

    res = []
    for r in readings:
        sc = scorers[r.key].score(big)
        a, per = strat_auc(sc, big, strat, ns)
        per_as = {rows["names"][i]: round(float(np.nanmean(per[i * 5:(i + 1) * 5])), 3) for i in range(n_as)}
        by_year = {int(y): round(strat_auc(sc[year == y], big[year == y], strat[year == y], ns)[0], 3)
                   for y in np.unique(year)}
        res.append({"reading": r.key, "n": int(scorers[r.key].valid.sum()), "tokens": scorers[r.key].k,
                    "strat_auc": a, "plain_auc": plain_auc(sc, big), "auc_by_asset_session": per_as,
                    "auc_by_year": by_year})

    rng = np.random.default_rng(seed)
    null_max = np.empty(perms)
    t1 = time.time()
    for p in range(perms):
        bp = shuffle_within(big, pgroup, rng)
        null_max[p] = max(strat_auc(scorers[r.key].score(bp), bp, strat, ns)[0] for r in readings)
        if p in (9, 99) or (p + 1) % 250 == 0:
            print(f"  perm {p + 1}/{perms}  {time.time() - t1:.0f}s", flush=True)
    for e in res:
        e["p_fw"] = float((null_max >= e["strat_auc"]).mean())
    res.sort(key=lambda e: -e["strat_auc"])
    cands = [e["reading"] for e in res if e["p_fw"] <= 0.05][:3]

    # top tokens of the best readings (scores on all discovery years)
    by_key = {r.key: r for r in readings}
    for e in res[:5]:
        r = by_key[e["reading"]]
        code = rows["codes"][r.key]
        v = code >= 0
        g = pd.DataFrame({"c": code[v], "b": big[v]}).groupby("c")["b"].agg(["sum", "count"])
        g = g[g["count"] >= 30]
        g["score"] = (g["sum"] + SHRINK * big[v].mean()) / (g["count"] + SHRINK)
        g = g.sort_values("score", ascending=False).head(10)
        e["top_tokens"] = [{"token": token_text(r, c), "n": int(x["count"]), "big_rate": round(x["sum"] / x["count"], 3)}
                           for c, x in g.iterrows()]

    x_rel_auc = plain_auc(rows["x_rel"], big)
    report = {"years": DISC, "rows": int(len(big)), "base_rate": float(big.mean()),
              "x_rel_plain_auc": x_rel_auc, "perms": perms, "seed": seed,
              "null_max_quantiles": {q: float(np.quantile(null_max, q)) for q in (0.5, 0.95, 0.99)},
              "candidates": cands, "frozen": rows["frozen"], "readings": res}
    OUT_DISC.write_text(json.dumps(report, indent=1, default=str))

    print(f"\nbaseline x_rel plain AUC {x_rel_auc:.3f}   null max stratified AUC: "
          f"median {report['null_max_quantiles'][0.5]:.3f}  95% {report['null_max_quantiles'][0.95]:.3f}")
    print(f"{'reading':22s} {'n':>5s} {'tok':>5s} {'stratAUC':>8s} {'plainAUC':>8s} {'p_fw':>6s}")
    for e in res[:15]:
        print(f"{e['reading']:22s} {e['n']:5d} {e['tokens']:5d} {e['strat_auc']:8.3f} "
              f"{e['plain_auc']:8.3f} {e['p_fw']:6.3f}")
    print(f"... worst: {res[-1]['reading']} {res[-1]['strat_auc']:.3f}")
    print(f"candidates (p_fw <= 0.05, max 3): {cands or 'none -> H11 closed'}")
    print(f"-> {OUT_DISC}  ({time.time() - t0:.0f}s)")
    return 0


def confirmation(perms: int, seed: int) -> int:
    disc = json.loads(OUT_DISC.read_text())
    cands = disc["candidates"]
    if not cands:
        print("no discovery candidate - nothing to confirm (H11 closed)")
        return 0
    if OUT_CONF.exists():
        raise SystemExit(f"{OUT_CONF} exists - the confirmation is run once")
    readings = [r for r in grid() if r.key in cands]
    frames = asset_session_frames((DISC[0], CONF[1]))
    rows = build_rows(frames, readings)
    n_as = len(rows["names"])
    year = rows["year"]
    tr, te = year <= DISC[1], (year >= CONF[0]) & (year <= CONF[1])
    big = rows["big"]
    strat = rows["as_id"] * 5 + rows["quint"]
    pgroup = (rows["as_id"] * 100 + (year - 2000)) * 5 + rows["quint"]
    rng = np.random.default_rng(seed)
    alpha = 0.05 / len(readings)
    out = []
    for r in readings:
        code = rows["codes"][r.key]
        sc = fit_score(code[tr], big[tr], code[te])
        a, per = strat_auc(sc, big[te], strat[te], n_as * 5)
        null = np.array([strat_auc(sc, shuffle_within(big[te], pgroup[te], rng), strat[te], n_as * 5)[0]
                         for _ in range(perms)])
        p = float((null >= a).mean())
        per_as = {rows["names"][i]: float(np.nanmean(per[i * 5:(i + 1) * 5])) for i in range(n_as)}
        n_pos = sum(v > 0.5 for v in per_as.values())
        ok = a > 0.5 and p <= alpha and n_pos >= 5
        out.append({"reading": r.key, "strat_auc": a, "p": p, "alpha": alpha,
                    "asset_sessions_above_half": n_pos, "auc_by_asset_session": per_as, "passes": ok})
        print(f"{r.key:22s} stratAUC {a:.3f}  p {p:.4f} (alpha {alpha:.4f})  "
              f"{n_pos}/7 asset-sessions > 0.5  -> {'PASS' if ok else 'fail'}")
    OUT_CONF.write_text(json.dumps({"years": CONF, "results": out}, indent=1, default=str))
    print(f"-> {OUT_CONF}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--perms", type=int, default=None)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    if args.confirm:
        return confirmation(args.perms or 2000, args.seed)
    return discovery(args.perms or 1000, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
