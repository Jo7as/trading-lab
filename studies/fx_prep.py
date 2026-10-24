"""FX prep: audit (A1-A6) + S0 gate per session window and timeframe.

One pass per instrument: classify once (the expensive part), then report
- the audit summary (are the FX data usable?),
- the S0 cost gate per timeframe (5m/15m/30m/1h/4h) on the audit-correct frame,
- the S0 gate per session window (Asia / London / NY-overlap / NY-late) at 15m/1h.

  .venv/bin/python studies/fx_prep.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "studies")
from pass_speed import load_cached  # noqa: E402

from trading_lab.cost_gate import horizon_gate  # noqa: E402
from trading_lab.data_audit import SESSIONS, build_study_frame, classify, dst_check  # noqa: E402

FX = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD")
TF = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}
# session windows in New York minutes
WINDOWS = {"Asia 00-03": (0, 180), "London 03-08": (180, 480),
           "Overlap 08-11": (480, 660), "NY-late 11-17": (660, 1020)}


def main() -> None:
    status = {}
    print("### FX audit (A1-A6)\n")
    frames = {}
    for a in FX:
        df = load_cached(a)
        label, rep = classify(df, a)
        rep.dst_ok, rep.dst_notes = dst_check(df, a)
        from trading_lab.data_audit import CROSS_CHECK_MAX, THIN_SHARE_MAX
        rep.passed = (rep.dst_ok and rep.consistency["ask_lt_bid"] == 0
                      and rep.consistency["ohlc_violations"] == 0
                      and rep.thin_days <= THIN_SHARE_MAX * max(rep.days, 1))
        status[a] = bool(rep.passed)
        print(rep.summary())
        frames[a], info = build_study_frame(df, a)
        print(f"        study frame: kept {info['kept']:,} / dropped {info['dropped']:,} "
              f"(thin days {info['thin_days']})")

    print("\n### S0 gate per timeframe (audit-correct frame, taker-exit ratio)\n")
    print(f"{'pair':7s} " + " ".join(f"{tf:>10s}" for tf in TF))
    for a in FX:
        row = [f"{a:7s}"]
        for tf, mins in TF.items():
            bars = frames[a].resample(f"{mins}min").agg(
                {"close": "last", "ask_close": "last"}).dropna()
            if len(bars) < 1000:
                row.append(f"{'n/a':>10s}")
                continue
            g = horizon_gate(bars, 1)          # one bar of this timeframe forward
            row.append(f"{g.ratio_taker:6.2f}{g.verdict()[:4]:>4s}")
        print(" ".join(row))

    print("\n### S0 gate per session window (1h bars, one bar forward)\n")
    print(f"{'pair':7s} " + " ".join(f"{w:>16s}" for w in WINDOWS))
    for a in FX:
        bars = frames[a].resample("60min").agg(
            {"close": "last", "ask_close": "last"}).dropna()
        loc = bars.index.tz_convert("America/New_York")
        mins = loc.hour * 60 + loc.minute
        row = [f"{a:7s}"]
        for w, (lo, hi) in WINDOWS.items():
            sel = bars[(mins >= lo) & (mins < hi)]
            if len(sel) < 500:
                row.append(f"{'n/a':>16s}")
                continue
            g = horizon_gate(sel, 1)
            row.append(f"{g.ratio_taker:6.2f} {g.verdict()[:4]:>4s}   ")
        print(" ".join(row))

    print("\npassed:", [a for a, ok in status.items() if ok] or "—")


if __name__ == "__main__":
    main()
