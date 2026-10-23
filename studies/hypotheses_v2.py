"""Step 2 (v2): S0 gate and the three hypotheses on the AUDIT-CORRECT frame.

The earlier re-run used a blanket mask (drop outliers + the whole roll window),
which also deleted real crisis bars and therefore flattered the results. Here the
frame comes from ``data_audit.build_study_frame``: spike / session_edge / unclear
removed, **news_like kept**, thin days dropped. Cost comes from the frame's own
bid/ask.

  .venv/bin/python studies/hypotheses_v2.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "studies")
from pass_speed import load_cached  # noqa: E402

from trading_lab.cost_gate import horizon_gate  # noqa: E402
from trading_lab.data_audit import SESSIONS, build_study_frame  # noqa: E402

INDICES = ("US100", "US30", "SP500", "GER40", "UK100")


def study_frame(asset: str) -> pd.DataFrame:
    df, _ = build_study_frame(load_cached(asset), asset)
    return df


def day_table(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    tz = SESSIONS[asset]["tz"]
    loc = df.index.tz_convert(tz)
    mins = (loc.hour * 60 + loc.minute).to_numpy()
    tab = pd.Series(df["close"].to_numpy(),
                    index=pd.MultiIndex.from_arrays([loc.date, mins], names=["d", "m"])).unstack("m")
    h = {"US100": (570, 600, 900, 955), "US30": (570, 600, 900, 955), "SP500": (570, 600, 900, 955),
         "GER40": (540, 600, 990, 1015), "UK100": (480, 540, 945, 970)}[asset]
    out = pd.DataFrame({"open": tab.get(h[0]), "c1": tab.get(h[1]),
                        "c2": tab.get(h[2]), "c3": tab.get(h[3])})
    out["prev_close"] = tab.ffill(axis=1).iloc[:, -1].shift(1)
    return out.dropna().reset_index().rename(columns={"d": "day"})


def study(name: str, direction: np.ndarray, move: np.ndarray, cost_pct: float,
          magnitude: np.ndarray, seed: int = 7) -> str:
    n = len(move)
    signed = direction * move
    med = float(np.median(signed))
    n_pos = int((signed > 0).sum())
    z = (n_pos - n / 2) / np.sqrt(n / 4)
    rng = np.random.default_rng(seed)
    placebo = float(np.mean(rng.choice(direction, n) * rng.permutation(move)))
    q = pd.qcut(pd.Series(magnitude).rank(method="first"), 5, labels=False)
    quint = [round(float(np.mean(signed[q == k])) * 100, 3) for k in range(5)]
    eff = abs(med) * 100 / cost_pct if cost_pct else float("inf")
    slope = np.polyfit(range(5), np.array(quint) * np.sign(med or 1), 1)[0]
    verdict = ("PASS" if (z >= 3 and eff >= 2 and abs(placebo) * 100 < 0.5 * abs(med * 100) and slope > 0)
               else "FAIL(sign)" if z < 3 else "FAIL(cost)" if eff < 2
               else "FAIL(dose)" if slope <= 0 else "FAIL(placebo)")
    return (f"  {name:20s} n={n:5d} med {med*100:+.3f}%  z={z:5.2f}  hit {n_pos/n:4.0%}  "
            f"cost {cost_pct:.4f}%  eff/cost {eff:5.1f}  placebo {placebo*100:+.3f}%  {verdict}")


def main() -> None:
    print("### S0 gate on the AUDIT-CORRECT frame (news bars kept)\n")
    frames = {}
    for a in SESSIONS:
        try:
            frames[a] = study_frame(a)
        except Exception as e:  # noqa: BLE001
            print(f"{a}: {e}")
            continue
        row = [f"  {a:7s}"]
        for h in (15, 60, 240):
            g = horizon_gate(frames[a], h)
            row.append(f"{h}m {g.ratio_taker:5.2f} {g.verdict():8s}")
        print("  ".join(row))

    print("\n### the three hypotheses, direct clock, audit-correct frame")
    for a in INDICES:
        if a not in frames:
            continue
        t = day_table(frames[a], a)
        if len(t) < 200:
            print(f"{a}: too few days ({len(t)})")
            continue
        cost = float(((frames[a]["ask_close"] - frames[a]["close"]) /
                      ((frames[a]["close"] + frames[a]["ask_close"]) / 2)).median()) * 100
        print(f"\n{a}  days={len(t)}  median round-trip cost {cost:.4f}%")
        print(study("H1 first-hour mom", np.sign(t["c1"] - t["open"]).to_numpy(),
                    ((t["c2"] - t["c1"]) / t["c1"]).to_numpy(), cost, np.abs(t["c1"] / t["open"] - 1)))
        print(study("H2 last-hour mom", np.sign(t["c2"] - t["open"]).to_numpy(),
                    ((t["c3"] - t["c2"]) / t["c2"]).to_numpy(), cost, np.abs(t["c2"] / t["open"] - 1)))
        gap = (t["open"] / t["prev_close"] - 1).to_numpy()
        print(study("H3 gap fade", -np.sign(gap), ((t["c3"] - t["open"]) / t["open"]).to_numpy(),
                    cost, np.abs(gap)))


if __name__ == "__main__":
    main()
