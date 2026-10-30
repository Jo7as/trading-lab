#!/usr/bin/env python3
"""H12 FX check: the one confirmed combination on FX pairs (``docs/specs/h12_fx_check_PREREG.md``).

  uv run python studies/h12_fx_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import studies.h12_mining_study as h12  # noqa: E402
from trading_lab.chart_tokens import sax_str  # noqa: E402
from trading_lab.pattern_mining import clustered_z, nonoverlap_count  # noqa: E402

PAIRS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD")
FIT = (pd.Timestamp("2019-01-01"), pd.Timestamp("2020-12-31"))
TEST = (pd.Timestamp("2021-01-01"), pd.Timestamp("2024-12-31"))
OUT = Path("artifacts/h12/fx_check.json")
LOT, USD_SIDE = 100_000, 2.50


def period_fx(day: pd.Timestamp, market: str) -> str | None:
    if FIT[0] <= day <= FIT[1]:
        return "mining"       # the study's label for the fit period
    return "validation" if TEST[0] <= day <= TEST[1] else None


def mirror(code: int) -> int:
    """cca + trend up -> aac + trend down: SAX letters a <-> c, size / volume kept."""
    tok, tr = divmod(code, h12.CARD["trend"])
    word, rest = divmod(tok, 9)
    digits = [(word // 3 ** k) % 3 for k in range(3)]
    new = sum((2 - d) * 3 ** k for k, d in enumerate(digits))
    return ((new * 9 + rest) * h12.CARD["trend"]) + (1 - tr)


def score(d: pd.DataFrame, members: list[int], direction: str, cl: np.ndarray) -> dict:
    y = {"up": (d["move"] == 1).to_numpy(float), "down": (d["move"] == -1).to_numpy(float)}
    p = {k: h12.control_p(d, v, ["market", "clock", "volq"]) for k, v in y.items()}
    other = "up" if direction == "down" else "down"
    r = np.isin(h12.combo_codes(d, ("b5", "trend"), h12.CARD), members)
    s = (y[direction] - y[other]) - (p[direction] - p[other])
    res = {"n": int(r.sum()), "p_dir": float(y[direction][r].mean()), "ctrl_dir": float(p[direction][r].mean()),
           "p_other": float(y[other][r].mean()), "ctrl_other": float(p[other][r].mean()),
           "excess": float(s[r].mean()), "z": clustered_z(s[r], cl[r])}
    res["by_pair"] = {m: round(float(s[r & (d["market"] == m).to_numpy()].mean()), 4) for m in PAIRS}
    res["by_year"] = pd.Series(s[r]).groupby(pd.to_datetime(d["day"]).dt.year.to_numpy()[r]).mean().round(4).to_dict()
    res["pairs_up"] = sum(v > 0 for v in res["by_pair"].values())
    occ = sum(nonoverlap_count(np.sort(d["t"].to_numpy(np.int64)[r & (d["market"] == m).to_numpy()]), 60 * h12.MIN)
              for m in PAIRS)
    res["per_week_per_pair"] = occ / sum(d.loc[d["market"] == m, "day"].nunique() / 5 for m in PAIRS)
    return res


def cost_atr(m: str, d: pd.DataFrame) -> dict:
    f = h12.load_cached(m, 2021, 2024)
    s = f[(f.index.hour >= 8) & (f.index.hour < 20)]
    spread, price = float((s["ask_close"] - s["close"]).median()), float(s["close"].median())
    # $5 round trip on 100k base; in price units of the quote currency
    comm = 2 * USD_SIDE / LOT * (price if m.startswith("USD") else 1.0)
    atr_ = float(d.loc[d["market"] == m, "atr"].median())
    return {"spread_atr": spread / atr_, "commission_atr": comm / atr_, "total_atr": (spread + comm) / atr_}


def main() -> int:
    h12.period_of = period_fx
    conf = json.loads(Path("artifacts/h12/confirmation.json").read_text())
    u = [r for r in conf["A"] if r["confirmed"]]
    assert len(u) == 1 and tuple(u[0]["groups"]) == ("b5", "trend")
    members = u[0]["members"]
    parts = []
    for m in PAIRS:
        a, _ = h12.market_rows(m, (2019, 2024))
        parts.append(a)
        print(f"  {m}: {len(a)} rows, threshold {a['thr'].iloc[0]:.2f} ATR", flush=True)
    a = pd.concat(parts, ignore_index=True)
    d = a[a["period"] == "validation"].reset_index(drop=True)
    cl = (d["market"] + "|" + d["day"].astype(str)).to_numpy()
    primary = score(d, members, "down", cl)
    second = score(d, [mirror(c) for c in members], "up", cl)
    passed = primary["excess"] > 0 and primary["z"] >= 2.0 and primary["pairs_up"] >= 4
    costs = {m: cost_atr(m, d) for m in PAIRS}
    (OUT).write_text(json.dumps({"primary": primary, "mirror": second, "passed": passed, "costs": costs}, indent=1, default=str))
    for k, r in (("PRIMARY as found", primary), ("mirror (no weight)", second)):
        print(f"\n{k}: n {r['n']}  P(dir) {r['p_dir']:.3f} ctrl {r['ctrl_dir']:.3f} | P(other) {r['p_other']:.3f} "
              f"ctrl {r['ctrl_other']:.3f} | excess {r['excess']:+.4f} z {r['z']:.2f} pairs>0 {r['pairs_up']}/6 "
              f"| {r['per_week_per_pair']:.1f}/week/pair\n  by pair {r['by_pair']}\n  by year {r['by_year']}")
    print(f"\nprimary verdict: {'PASS' if passed else 'FAIL'}")
    for m, c in costs.items():
        print(f"  {m}: spread {c['spread_atr']:.3f} + commission {c['commission_atr']:.3f} = {c['total_atr']:.3f} ATR")
    print("  words:", [sax_str(divmod(divmod(c, 2)[0], 9)[0], 3, 3) for c in members])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
