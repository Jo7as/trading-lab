"""S-010: every rule of docs/specs/drop_continuation.yaml on tiny hand-made paths."""
from __future__ import annotations

import numpy as np
import pytest

from trading_lab.chart_tokens import sax_str
from trading_lab.strategies.drop_continuation import (SIGNAL_TOKENS, is_signal, last_bos, pick_trades,
                                                      variant_ok, walk)


def tok(word: str, size: int, vol: int) -> int:
    d = dict(a=0, b=1, c=2)
    return (d[word[0]] * 9 + d[word[1]] * 3 + d[word[2]]) * 9 + size * 3 + vol


def test_token_family_is_the_confirmed_one():
    got = {(sax_str(w, 3, 3), "sml"[s], v + 1) for w, s, v in SIGNAL_TOKENS}
    assert got == {("cca", "l", 3), ("bca", "l", 3), ("cba", "l", 3), ("cca", "m", 3), ("cca", "l", 2)}


def test_signal_needs_token_and_uptrend():
    assert is_signal(tok("cca", 2, 2), 1)
    assert is_signal(tok("cca", 2, 1), 1)
    assert not is_signal(tok("cca", 2, 2), 0)        # trend down
    assert not is_signal(tok("cca", 0, 2), 1)        # small size
    assert not is_signal(tok("aac", 2, 2), 1)        # rising path
    assert not is_signal(-1, 1)


@pytest.mark.parametrize("variant,kw,want", [
    ("basis", {}, True),
    ("session", {"clock": 1}, True), ("session", {"clock": 0}, False), ("session", {"clock": 2}, False),
    ("struktur15", {"bos15": -1}, True), ("struktur15", {"bos15": 1}, False), ("struktur15", {"bos15": 0}, False),
    ("neues_tief", {"newhl": 2}, True), ("neues_tief", {"newhl": 1}, False), ("neues_tief", {"newhl": -1}, False),
    ("hohe_vola", {"volq": 3}, True), ("hohe_vola", {"volq": 4}, True), ("hohe_vola", {"volq": 2}, False),
])
def test_variant_filters(variant, kw, want):
    args = {"clock": 0, "bos15": 0, "newhl": -1, "volq": 0} | kw
    assert variant_ok(variant, **args) is want


def test_last_bos_only_uses_closed_bars():
    ev = np.array([0, 1, 0, -1, 0])
    end = np.array([10, 20, 30, 40, 50])
    assert last_bos(ev, end, 35, 1, -1) == 1      # the down event closes at 40, not known at 35
    assert last_bos(ev, end, 40, 1, -1) == -1
    assert last_bos(ev, end, 5, 1, -1) == 0
    assert last_bos(np.array([2, -2, 0]), np.array([1, 2, 3]), 3, 1, -1) == 0  # CHoCH is no BOS


def path(bid_h, bid_l, bid_c, spread=1.0, bid_o=None):
    bh, bl, bc = (np.array(x, dtype=float) for x in (bid_h, bid_l, bid_c))
    bo = np.array(bid_o, dtype=float) if bid_o else bc.copy()
    return bo, bh, bl, bc, bo + spread, bh + spread, bl + spread, bc + spread


def test_short_hits_target_on_the_ask():
    p = path([100, 100, 100], [99, 95, 88], [100, 96, 90], bid_o=[100, 100, 96])
    f = walk(*p, 0, 3, 10.0, -1, slip=0.0)
    # entry 100 (bid); target 90 needs ask low <= 90: bar 2 has 88 + 1 = 89
    assert (f.outcome, f.entry, f.exit, f.r) == ("tp", 100.0, 90.0, 1.0)


def test_short_stop_on_the_ask_and_stop_wins_a_tie():
    p = path([100, 110], [99, 85], [100, 90], bid_o=[100, 100])
    f = walk(*p, 0, 2, 10.0, -1, slip=0.0)
    assert (f.outcome, f.r) == ("stop", -1.0)         # ask high 111 >= 110, also ask low 86 <= 90


def test_short_time_exit_pays_the_spread():
    p = path([101, 101], [99, 99], [100, 98], bid_o=[100, 100])
    f = walk(*p, 0, 2, 10.0, -1, slip=0.0)
    assert f.outcome == "time" and f.exit == 99.0 and f.r == pytest.approx(0.1)


def test_slippage_on_entry_stop_and_time_not_on_target():
    p = path([100, 100], [99, 80], [100, 85], bid_o=[100, 100])
    f = walk(*p, 0, 2, 10.0, -1, slip=0.01)
    assert f.entry == pytest.approx(99.0) and f.outcome == "tp" and f.exit == pytest.approx(89.0) and f.r == 1.0
    p = path([100, 130], [99, 99], [100, 100], bid_o=[100, 100])
    f = walk(*p, 0, 2, 10.0, -1, slip=0.01)
    assert f.outcome == "stop" and f.exit == pytest.approx(109.0 * 1.01)


def test_long_side_mirrors_for_the_placebo():
    p = path([100, 120], [99, 99], [100, 110], bid_o=[100, 100])
    f = walk(*p, 0, 2, 10.0, 1, slip=0.0)
    # entry on the ask 101, target 111 needs bid high: 120
    assert (f.outcome, f.entry, f.exit, f.r) == ("tp", 101.0, 111.0, 1.0)


def test_one_position_per_group():
    s = [dict(t=0, exit_t=60, market="US100"), dict(t=30, exit_t=90, market="US30"),
         dict(t=30, exit_t=90, market="GER40"), dict(t=60, exit_t=120, market="US30")]
    got = [(x["t"], x["market"]) for x in pick_trades(s)]
    assert got == [(0, "US100"), (30, "GER40"), (60, "US30")]
