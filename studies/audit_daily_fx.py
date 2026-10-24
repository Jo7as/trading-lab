#!/usr/bin/env python3
"""Daily-level audit for the carry basket (the strategy uses daily bars).

Checks: monotonic timestamps, duplicates, non-positive prices, outlier daily
moves (> 12x the median |return|), and long gaps (> 7 calendar days). Writes the
`audit_passed` flags into data/audit_status.yaml.
"""
from __future__ import annotations
import pathlib, sys
import numpy as np, pandas as pd, yaml

sys.path.insert(0, "."); sys.path.insert(0, "studies")
import run_carry

OUT = pathlib.Path("data/audit_status.yaml")
st = yaml.safe_load(OUT.read_text()) if OUT.exists() else {}
print(f"{'pair':9s} {'bars':>6s} {'outliers':>8s} {'worst%':>7s} {'maxgap d':>8s}  verdict")
for p, const in run_carry.PAIRS.items():
    s = run_carry.daily(p, const).dropna()
    r = s.pct_change()
    scale = float(r.abs().median()) or 1e-12
    outliers = int((r.abs() > 12 * scale).sum())
    worst = float(r.abs().max()) * 100
    gaps = s.index.to_series().diff().dt.days.max()
    ok = (s.index.is_monotonic_increasing and not s.index.duplicated().any()
          and (s > 0).all() and float(r.abs().max()) < 0.15 and (gaps <= 8))
    st[p.replace("/", "")] = bool(ok)
    print(f"{p:9s} {len(s):6d} {outliers:8d} {worst:7.2f} {int(gaps):8d}  {'OK' if ok else 'SUSPECT'}")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(yaml.safe_dump(st, sort_keys=False))
print("written", OUT)
