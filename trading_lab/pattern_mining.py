"""Vectorised building blocks for token-combination mining (H12, ``docs/specs/h12_pattern_mining_PREREG.md``).

* ``sax_codes`` / ``sax_windows`` - SAX words for many paths at once (fixed length from a 2-D
  array, or variable windows of one long series via cumulative sums). Same definition as
  ``chart_tokens.sax_word`` (z-normalise, ``np.array_split`` PAA, normal breakpoints).
* ``first_passage_moves`` - +1 / -1 / 0: which of p+m / p-m is touched first on the next bars
  (a bar touching both, or neither within the horizon, gives 0).
* ``mixed_neighbors`` - one-step mutations of mixed-radix token codes (the "similar, not
  identical" families).
* ``clustered_z`` / ``nonoverlap_count`` - the statistic and the frequency rule.
"""
from __future__ import annotations

import numpy as np

from trading_lab.chart_tokens import BREAKPOINTS


def _letters(z: np.ndarray, a: int) -> np.ndarray:
    return np.searchsorted(BREAKPOINTS[a], z, side="right")


def _split_bounds(n: np.ndarray, w: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Segment [start, end) offsets of ``np.array_split(range(n), w)``, vectorised over n."""
    q, r = np.divmod(n, w)
    out, start = [], np.zeros_like(n)
    for k in range(w):
        size = q + (k < r)
        out.append((start, start + size))
        start = start + size
    return out


def sax_codes(paths: np.ndarray, w: int, a: int) -> np.ndarray:
    """SAX word (base-``a`` integer, first segment most significant) per row; NaN row -> -1."""
    x = np.asarray(paths, dtype=float)
    n = x.shape[1]
    bad = np.isnan(x).any(axis=1)
    mu = np.nanmean(np.where(bad[:, None], 0.0, x), axis=1, keepdims=True)
    sd = np.where(bad, 1.0, x.std(axis=1))
    code = np.zeros(len(x), dtype=np.int64)
    for s, e in _split_bounds(np.array(n), w):
        seg = x[:, int(s):int(e)].mean(axis=1)
        z = np.where(sd > 0, (seg - mu[:, 0]) / np.where(sd > 0, sd, 1.0), 0.0)
        code = code * a + _letters(z, a)
    code[bad] = -1
    return code


def sax_windows(values: np.ndarray, i: np.ndarray, j: np.ndarray, w: int, a: int,
                min_len: int = 1) -> np.ndarray:
    """SAX word of ``values[i:j]`` for many windows at once (cumulative sums); -1 if too short.

    ``values`` should be roughly centred (e.g. minus its median) to keep the sums exact."""
    v = np.asarray(values, dtype=float)
    cs = np.r_[0.0, np.cumsum(v)]
    cs2 = np.r_[0.0, np.cumsum(v * v)]
    i, j = np.asarray(i, dtype=np.int64), np.asarray(j, dtype=np.int64)
    n = j - i
    ok = n >= max(min_len, w)
    nn = np.where(ok, n, w)
    ii = np.where(ok, i, 0)
    mu = (cs[ii + nn] - cs[ii]) / nn
    var = np.maximum((cs2[ii + nn] - cs2[ii]) / nn - mu * mu, 0.0)
    sd = np.sqrt(var)
    flat = sd <= 1e-12 * np.maximum(np.abs(mu), 1.0)
    code = np.zeros(len(i), dtype=np.int64)
    for s, e in _split_bounds(nn, w):
        seg = (cs[ii + e] - cs[ii + s]) / (e - s)
        z = np.where(flat, 0.0, (seg - mu) / np.where(flat, 1.0, sd))
        code = code * a + _letters(z, a)
    code[~ok] = -1
    return code


def first_passage_moves(hi: np.ndarray, lo: np.ndarray, p: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Per row: +1 if p+m is touched before p-m, -1 the other way, 0 if neither or both in the
    first bar that touches anything. ``hi``/``lo`` are (rows, bars), NaN = no bar."""
    up = hi >= (p + m)[:, None]
    dn = lo <= (p - m)[:, None]
    n = hi.shape[1]
    iu = np.where(up.any(axis=1), up.argmax(axis=1), n)
    idn = np.where(dn.any(axis=1), dn.argmax(axis=1), n)
    out = np.zeros(len(p), dtype=np.int8)
    out[iu < idn] = 1
    out[idn < iu] = -1
    return out


def mixed_neighbors(codes: np.ndarray, radices: tuple[int, ...], mutable: tuple[bool, ...]) -> np.ndarray:
    """All codes that differ by +-1 in exactly one mutable digit (most significant digit first).

    Returns (len(codes), 2 x number of mutable digits), -1 where the step leaves the range."""
    codes = np.asarray(codes, dtype=np.int64)
    place = np.cumprod((1,) + tuple(reversed(radices[1:])))[::-1].astype(np.int64)
    cols = []
    for k, (r, mu) in enumerate(zip(radices, mutable)):
        if not mu:
            continue
        d = (codes // place[k]) % r
        for step in (-1, 1):
            nd = d + step
            cols.append(np.where((nd >= 0) & (nd < r) & (codes >= 0), codes + step * place[k], -1))
    return np.stack(cols, axis=1) if cols else np.empty((len(codes), 0), dtype=np.int64)


def clustered_z(resid: np.ndarray, cluster: np.ndarray) -> float:
    """sum(resid) / sqrt(sum over clusters of (cluster sum)^2) - (observed - expected) over a
    cluster-robust standard error. NaN if there is no variance."""
    _, inv = np.unique(cluster, return_inverse=True)
    s = np.bincount(inv, weights=resid)
    se = np.sqrt((s * s).sum())
    return float(resid.sum() / se) if se > 0 else float("nan")


def nonoverlap_count(t_ns: np.ndarray, gap_ns: int) -> int:
    """Greedy count of occurrences at least ``gap_ns`` apart (times sorted ascending)."""
    n, last = 0, None
    for t in t_ns:
        if last is None or t - last >= gap_ns:
            n += 1
            last = t
    return n
