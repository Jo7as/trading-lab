#!/usr/bin/env python3
"""Compare the MT5 logger's measured costs with the lab's cost model and write an override.

  .venv/bin/python studies/broker_costs_report.py <MQL5/Files folder> <login> [--name fundednext_demo]

Writes ``artifacts/broker_costs/<name>.json``; then
``studies/tsmom_check.py --referenz --broker artifacts/broker_costs/<name>.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd  # noqa: E402

from trading_lab import broker_costs, rates  # noqa: E402
from trading_lab.strategies.tsmom_multi_asset import GROUP_OF, TsmomConfig, carry_rates  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("login")
    ap.add_argument("--name", default="broker")
    a = ap.parse_args()
    df = broker_costs.summary(a.folder, a.login)
    if df.empty:
        print("keine Logger-Dateien gefunden")
        return 1
    model_cost = json.loads(Path("artifacts/tsmom/costs.json").read_text()) if Path("artifacts/tsmom/costs.json").exists() else {}
    r = rates.load_rates(("USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD")).ffill()
    rows = []
    for lab, x in df.iterrows():
        lo = sh = float("nan")
        if lab in GROUP_OF:
            m_lo, m_sh = carry_rates(lab, r, TsmomConfig())
            lo, sh = float(m_lo.dropna().iloc[-1]), float(m_sh.dropna().iloc[-1])
        rows.append({"Markt": lab, "Kosten/Seite gemessen bp": x["cost_bp_side"], "Modell bp": model_cost.get(lab),
                     "Swap long gemessen %": 100 * x["swap_long_pa"], "Modell long %": 100 * lo,
                     "Swap short gemessen %": 100 * x["swap_short_pa"], "Modell short %": 100 * sh,
                     "Slippage bp": x.get("slip_bp"), "Probes": x.get("n_probes")})
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).set_index("Markt").round(3).to_string())
    out = Path("artifacts/broker_costs") / f"{a.name}.json"
    broker_costs.write_override(df, out, source=f"{a.name} login {a.login}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
