"""Fill and exit sides of the pullback candidate (H-010).

``_resolve`` used to check every order against the bid: a long's resting buy was
tested with the bid *close*, a short's stop and target with the bid high/low, and
a time exit left at the mid. That makes shorts asymmetric to longs and lets an
exit happen at a price the market never traded. These tests pin the sides:

============  =========================  ==========================
order         long                       short
============  =========================  ==========================
entry limit   fills when ask_low <= L    fills when bid_high >= L
stop          stop-sell on the bid       stop-buy on the ask
target        limit-sell on the bid      limit-buy on the ask
time exit     sells the bid              buys the ask
============  =========================  ==========================
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from trading_lab.strategies.pullback import PullbackConfig, _resolve

SIGMA = 0.002          # -> sigma_abs = 0.2 at anchor 100, buffer 0.20 -> 0.04
ANCHOR = 100.0
CFG = PullbackConfig(entry="limit", limit_wait_minutes=5, stop_buffer=0.20,
                     target_r=1.5, max_hold_minutes=10, max_cost_r=2.0)


def _frame(specs, *, setup, spread, max_hold_minutes=10, start="2024-03-05 14:30"):
    """Bars given as (open, high, low, close) of the *bid*; ask = the same + spread."""
    idx = pd.date_range(start, periods=len(specs), freq="1min", tz="UTC")
    df = pd.DataFrame(
        [
            {"open": o, "high": h, "low": l, "close": c,
             "ask_open": o + spread, "ask_high": h + spread,
             "ask_low": l + spread, "ask_close": c + spread}
            for o, h, l, c in specs
        ],
        index=idx,
    )
    local = df.index.tz_convert("America/New_York")
    f = pd.DataFrame(
        {
            "day": local.normalize(),
            "minute": local.hour * 60 + local.minute,
            "mid": (df["close"] + df["ask_close"]) / 2.0,
            "bid": df["close"], "ask": df["ask_close"],
            "bid_open": df["open"], "bid_high": df["high"], "bid_low": df["low"],
            "ask_open": df["ask_open"], "ask_high": df["ask_high"], "ask_low": df["ask_low"],
            "high": df["high"], "low": df["low"], "open": df["open"],
            "sigma": SIGMA, "anchor": ANCHOR, "setup": 0,
            "trend": 1.0, "retrace_long": 0.5, "retrace_short": 0.5,
        },
        index=df.index,
    )
    f.iloc[0, f.columns.get_loc("setup")] = setup
    return f, replace(CFG, max_hold_minutes=max_hold_minutes)


def test_long_limit_does_not_fill_without_an_ask_touch():
    # entry level = signal mid = 100.05, i.e. ask_low must fall to 100.05.
    # Every following bar keeps bid_low = 100.00 (ask_low = 100.10) — the old
    # bid-close test filled on the first of them.
    flat = (100.00, 100.00, 100.00, 100.00)
    f, cfg = _frame([flat, flat, flat, flat], setup=1, spread=0.10)
    assert _resolve(f, 0, cfg) is None


def test_long_limit_fills_on_an_ask_touch_and_exits_at_the_target():
    # signal: bid close 100.20 -> entry level 100.25, bid_low 100.00 -> stop 99.96,
    # i.e. risk 0.29 and target 100.685.
    signal = (100.20, 100.20, 100.00, 100.20)
    touch = (100.20, 100.50, 100.05, 100.45)      # ask_low 100.15 <= 100.25
    run = (100.45, 100.75, 100.45, 100.70)        # bid_high 100.75 >= 100.685
    f, cfg = _frame([signal, touch, run], setup=1, spread=0.10)

    t = _resolve(f, 0, cfg)
    assert t is not None
    assert t["entry"] == pytest.approx(100.25)
    assert t["entry_time"] == f.index[1]
    assert t["stop"] == pytest.approx(99.96)
    assert t["exit_reason"] == "target"
    assert t["exit"] == pytest.approx(100.685)
    assert t["r"] == pytest.approx(1.5)
    assert t["cost_r"] == pytest.approx(0.10 / 0.29, abs=1e-6)


def test_short_target_needs_the_ask_not_the_bid():
    # spread 0.04: signal mid 100.02 -> short entry level, stop = ask_high + 0.04
    # = 100.08, risk 0.06, target 99.93.
    signal = (100.00, 100.00, 100.00, 100.00)
    fill = (100.00, 100.03, 100.00, 100.02)       # bid_high 100.03 >= 100.02
    # bid_low 99.90 would have hit the old bid-based target; ask_low 99.94 does not.
    dip = (100.00, 100.00, 99.90, 99.95)
    flat = (100.00, 100.00, 100.00, 100.00)
    f, cfg = _frame([signal, fill, dip, flat], setup=-1, spread=0.04, max_hold_minutes=2)

    t = _resolve(f, 0, cfg)
    assert t is not None
    assert t["target"] == pytest.approx(99.93)
    assert t["exit_reason"] == "time"              # not the target
    assert t["r"] > -1.0                            # it never "won" a touch it did not get


def test_time_exit_sells_the_bid_for_a_long_and_pays_the_ask_for_a_short():
    signal = (100.20, 100.20, 100.00, 100.20)
    touch = (100.20, 100.25, 100.05, 100.10)
    last = (100.10, 100.12, 100.08, 100.10)

    long_f, long_cfg = _frame([signal, touch, last], setup=1, spread=0.10,
                             max_hold_minutes=1)
    long_t = _resolve(long_f, 0, long_cfg)
    assert long_t is not None and long_t["exit_reason"] == "time"
    assert long_t["exit"] == pytest.approx(100.10)   # the bid, not the mid 100.15

    short_f, short_cfg = _frame([signal, touch, last], setup=-1, spread=0.10,
                               max_hold_minutes=1)
    short_t = _resolve(short_f, 0, short_cfg)
    assert short_t is None or short_t["exit_reason"] != "target"


def test_market_entry_pays_the_ask_open():
    signal = (100.20, 100.20, 100.00, 100.20)     # entry = 100.25
    nxt = (100.30, 100.60, 100.25, 100.55)        # ask_open = 100.40
    flat = (100.55, 100.55, 100.55, 100.55)
    f, _ = _frame([signal, nxt, flat], setup=1, spread=0.10)
    cfg = PullbackConfig(entry="market", max_hold_minutes=10, max_cost_r=2.0)

    t = _resolve(f, 0, cfg)
    assert t is not None
    assert t["entry"] == pytest.approx(100.40)    # ask_open, not the bid open 100.30
