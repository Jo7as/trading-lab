"""Chart tokens: price action as letters (H11, ``docs/specs/h11_tokens_PREREG.md``).

Two token families, both price-invariant (the same shape at another price is the same
word) and - before the size letter is added - size-invariant:

* **S (SAX)** - the shape of a path of closes: z-normalise, PAA to ``W`` segments,
  map each segment mean to one of ``A`` letters with the normal breakpoints
  (Lin, Keogh et al. 2003/2007).
* **K (candles)** - each bar one of 5 shapes, read from its own range only.

A size / volume letter (tercile of a ratio) can be attached to either. Words are
returned as small integers so a study can count them with ``np.bincount``; ``*_str``
turns them back into readable text.

Also here: ``bars_before`` - TF bars aggregated from 1-minute bars, aligned so that
the last bar ends exactly at a given time (the session open). Bucketing is by clock
offset, not by row position, so a missing minute cannot shift a bar.
"""
from __future__ import annotations

import numpy as np

#: normal breakpoints for an alphabet of 3 and 4 letters
BREAKPOINTS = {3: np.array([-0.4307, 0.4307]), 4: np.array([-0.6745, 0.0, 0.6745])}

#: candle shapes: D doji, U strong up, u other up, n other down, N strong down
CANDLE_LETTERS = "DUunN"
DOJI_BODY = 0.1


def sax_word(closes: np.ndarray, w: int, a: int) -> int:
    """SAX word of a path as an integer in base ``a`` (first segment = most significant).

    ``closes`` is split into ``w`` segments as equal as possible (``np.array_split``).
    A flat path (std 0) gives the middle letter everywhere (for a = 4: letter 2)."""
    x = np.asarray(closes, dtype=float)
    if len(x) < w or np.isnan(x).any():
        raise ValueError("need >= w finite closes")
    sd = x.std()
    z = np.zeros_like(x) if sd == 0 else (x - x.mean()) / sd
    paa = np.array([seg.mean() for seg in np.array_split(z, w)])
    letters = np.searchsorted(BREAKPOINTS[a], paa, side="right")
    code = 0
    for k in letters:
        code = code * a + int(k)
    return code


def sax_str(code: int, w: int, a: int) -> str:
    out = []
    for _ in range(w):
        code, k = divmod(code, a)
        out.append("abcd"[k])
    return "".join(reversed(out))


def candle_shape(o: float, h: float, lo: float, c: float) -> int:
    """Index into ``CANDLE_LETTERS`` from the bar's own range (price- and size-free)."""
    r = h - lo
    if r <= 0 or abs(c - o) < DOJI_BODY * r:
        return 0
    po, pc = (o - lo) / r, (c - lo) / r
    if po <= 1 / 3 and pc >= 2 / 3:
        return 1
    if po >= 2 / 3 and pc <= 1 / 3:
        return 4
    return 2 if c > o else 3


def candle_word(o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray) -> int:
    code = 0
    for i in range(len(o)):
        code = code * 5 + candle_shape(o[i], h[i], lo[i], c[i])
    return code


def candle_str(code: int, n: int) -> str:
    out = []
    for _ in range(n):
        code, k = divmod(code, 5)
        out.append(CANDLE_LETTERS[k])
    return "".join(reversed(out))


def tercile_letter(x: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    """0 / 1 / 2 by two frozen edges; NaN -> -1."""
    x = np.asarray(x, dtype=float)
    out = np.searchsorted(np.asarray(edges, dtype=float), x, side="right").astype(int)
    out[np.isnan(x)] = -1
    return out


def to_ns(index) -> np.ndarray:
    """Int64 nanoseconds of a DatetimeIndex - pandas 3 keeps ms/us/ns resolutions apart."""
    return index.as_unit("ns").asi8


def bars_before(ts_ns: np.ndarray, o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray,
                v: np.ndarray, t0_ns: int, tf_min: int, n: int, min_fill: float = 0.5
                ) -> np.ndarray:
    """``n`` TF bars ending exactly at ``t0`` from sorted 1-minute bars (bar-open stamps,
    nanoseconds - use ``to_ns``).

    Returns an (n, 5) array open/high/low/close/volume; a bar with fewer than
    ``min_fill`` x tf_min minutes present is all NaN."""
    step = tf_min * 60_000_000_000
    start = t0_ns - n * step
    a, b = np.searchsorted(ts_ns, start), np.searchsorted(ts_ns, t0_ns)
    out = np.full((n, 5), np.nan)
    if b <= a:
        return out
    k = ((ts_ns[a:b] - start) // step).astype(int)
    for j in range(n):
        m = np.flatnonzero(k == j)
        if len(m) < min_fill * tf_min:
            continue
        m = m + a
        out[j] = (o[m[0]], h[m].max(), lo[m].min(), c[m[-1]], v[m].sum())
    return out
