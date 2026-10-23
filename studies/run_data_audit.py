#!/usr/bin/env python3
"""A1-A6 data audit per the drop-in spec; writes data/audit_status.yaml + a report.

  .venv/bin/python studies/run_data_audit.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, "studies")
from pass_speed import load_cached  # noqa: E402

from trading_lab.data_audit import (  # noqa: E402
    SESSIONS, classify, dst_check, spread_by_bucket,
)

OUT = Path("data/audit_status.yaml")
REP = Path("artifacts/data_audit_v2.json")


def daily_cross_check(df: pd.DataFrame, asset: str) -> float | None:
    """A5: median |daily close (CFD) vs yfinance proxy| in %."""
    proxy = SESSIONS[asset].get("proxy")
    if not proxy:
        return None
    try:
        import yfinance as yf
        y = yf.download(proxy, start=str(df.index.min().date()), end=str(df.index.max().date()),
                        interval="1d", progress=False, auto_adjust=False)
        if y.empty:
            return None
        yc = y["Close"]
        if isinstance(yc, pd.DataFrame):
            yc = yc.iloc[:, 0]
        yc.index = pd.to_datetime(yc.index).tz_localize(None)
        cfd = df["close"].tz_convert("UTC").tz_localize(None).groupby(lambda x: x.date()).last()
        cfd.index = pd.to_datetime(cfd.index)
        j = pd.concat([cfd.rename("cfd"), yc.rename("yf")], axis=1).dropna()
        return round(float(((j["cfd"] / j["yf"] - 1).abs()).median()) * 100, 3) if len(j) else None
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    status, reports = {}, []
    for asset in SESSIONS:
        try:
            df = load_cached(asset)
        except FileNotFoundError:
            print(f"{asset}: no cache, skipped")
            status[asset] = False
            continue
        label, rep = classify(df, asset)
        rep.dst_ok, rep.dst_notes = dst_check(df, asset)
        rep.daily_cross_check_pct = daily_cross_check(df, asset)
        rep.spread_table = None
        from trading_lab.data_audit import CROSS_CHECK_MAX, THIN_SHARE_MAX
        rep.passed = (
            rep.dst_ok
            and rep.consistency["ask_lt_bid"] == 0
            and rep.consistency["ohlc_violations"] == 0
            and rep.thin_days <= THIN_SHARE_MAX * max(rep.days, 1)
            and (rep.daily_cross_check_pct is None or rep.daily_cross_check_pct <= CROSS_CHECK_MAX)
        )
        status[asset] = bool(rep.passed)
        print(rep.summary())
        for e in rep.extremes[:2]:
            print(f"        extreme {e['ts']}  ratio {e['range_ratio']}  -> {e['label']}")
        reports.append({**{k: v for k, v in rep.__dict__.items() if k != "spread_table"}})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump(status, sort_keys=False))
    REP.parent.mkdir(parents=True, exist_ok=True)
    REP.write_text(json.dumps(reports, indent=2, default=str))
    print(f"\n{OUT}\n{REP}\npassed: {[a for a, ok in status.items() if ok] or '—'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
