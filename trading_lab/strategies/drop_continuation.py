"""S-010 drop_continuation: short after a sharp 5-min drop in a 1-h uptrend.

Spec: ``docs/specs/drop_continuation.yaml`` (agreed 2026-10-30). Stage 2 of H12. The token
tables come from ``studies/h12_mining_study.py``; this module holds the trade rules only:
the signal test, the variant filters, the fill / exit walk and the one-position-per-group pick.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: (word code, size letter, volume letter): cca/l/v3, bca/l/v3, cba/l/v3, cca/m/v3, cca/l/v2
#: letters a=0 b=1 c=2; size s=0 m=1 l=2; volume v1=0 v2=1 v3=2
SIGNAL_TOKENS = frozenset({(24, 2, 2), (15, 2, 2), (21, 2, 2), (24, 1, 2), (24, 2, 1)})
VARIANTS = ("basis", "session", "struktur15", "neues_tief", "hohe_vola")
GROUPS = {"US100": "us", "US30": "us", "GER40": "ger"}
SLIPPAGE_FRAC = 0.0001  # 1 bp per market side
HOLD_MIN = 60
MIN_NS = 60_000_000_000


def is_signal(b5_token: int, trend: int) -> bool:
    """5-min token (word*9 + size*3 + volume) in the fixed family and 1-h trend up."""
    if b5_token < 0 or trend != 1:
        return False
    word, rest = divmod(int(b5_token), 9)
    size, vol = divmod(rest, 3)
    return (word, size, vol) in SIGNAL_TOKENS


def variant_ok(variant: str, *, clock: int, bos15: int, newhl: int, volq: int) -> bool:
    if variant == "basis":
        return True
    if variant == "session":
        return clock == 1
    if variant == "struktur15":
        return bos15 == -1
    if variant == "neues_tief":
        return newhl == 2
    if variant == "hohe_vola":
        return volq >= 3
    raise ValueError(variant)


def last_bos(event: np.ndarray, end_ns: np.ndarray, t: int, bos_up: int, bos_dn: int) -> int:
    """Direction of the last BOS on a bar that closed at or before t: +1 / -1 / 0."""
    n = int(np.searchsorted(end_ns, t, side="right"))
    hits = np.flatnonzero((event[:n] == bos_up) | (event[:n] == bos_dn))
    if not len(hits):
        return 0
    return 1 if event[hits[-1]] == bos_up else -1


@dataclass(frozen=True)
class Fill:
    side: int        # -1 short, +1 long
    entry: float
    exit: float
    outcome: str     # tp / stop / time
    r: float


def walk(bid_o, bid_h, bid_l, bid_c, ask_o, ask_h, ask_l, ask_c, i: int, e: int, m: float,
         side: int, slip: float = SLIPPAGE_FRAC) -> Fill:
    """Market entry at bar i's open, stop and target +-m, time exit at bar e-1's close.

    A short sells the bid and buys back on the ask; a long the other way. Stop and time exits
    pay slippage, the resting target does not; stop wins a bar that touches both."""
    if side < 0:
        entry = bid_o[i] * (1 - slip)
        stop, tp = entry + m, entry - m
        for j in range(i, e):
            if ask_h[j] >= stop:
                px = stop * (1 + slip)
                return Fill(side, entry, px, "stop", (entry - px) / m)
            if ask_l[j] <= tp:
                return Fill(side, entry, tp, "tp", 1.0)
        px = ask_c[e - 1] * (1 + slip)
        return Fill(side, entry, px, "time", (entry - px) / m)
    entry = ask_o[i] * (1 + slip)
    stop, tp = entry - m, entry + m
    for j in range(i, e):
        if bid_l[j] <= stop:
            px = stop * (1 - slip)
            return Fill(side, entry, px, "stop", (px - entry) / m)
        if bid_h[j] >= tp:
            return Fill(side, entry, tp, "tp", 1.0)
    px = bid_c[e - 1] * (1 - slip)
    return Fill(side, entry, px, "time", (px - entry) / m)


def pick_trades(signals: list[dict], groups: dict[str, str] = GROUPS) -> list[dict]:
    """Chronological greedy pick: one open position per correlation group.

    Each signal needs ``t``, ``exit_t`` and ``market``."""
    busy: dict[str, int] = {}
    out = []
    for s in sorted(signals, key=lambda x: x["t"]):
        g = groups[s["market"]]
        if s["t"] < busy.get(g, -1):
            continue
        busy[g] = s["exit_t"]
        out.append(s)
    return out
