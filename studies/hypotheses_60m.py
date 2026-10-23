"""Step 2: test three pre-registered hypotheses directly on the minutes clock.

No resampling and no research-engine abstraction (which reported a median of
+2.38 %/day where the same quantity computed directly is +0.08 %). For each day we
read the cleaned 1-minute series at exact ET clock times, so there is nothing to
misalign.

H1 first-hour momentum : sign(close 10:00 - close 09:30) -> move 10:00 .. 15:00
H2 last-hour momentum  : sign(close 15:00 - open 09:30) -> move 15:00 .. 15:55
H3 overnight gap fill  : -sign(open 09:30 / close(prev 15:55) - 1) -> move 09:30 .. 15:55

Gates (pre-registered): sign-test z >= 3, |median move| >= 2x round-trip cost,
placebo (random days) clearly smaller, dose-response monotone in |magnitude|.

  .venv/bin/python studies/hypotheses_60m.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "studies")
from pass_speed import load_cached  # noqa: E402

from trading_lab.data_quality import clean  # noqa: E402

TZ = {"US100": "America/New_York", "SP500": "America/New_York", "GER40": "Europe/Berlin"}
CLOCK = {"US100": (570, 600, 900, 955), "SP500": (570, 600, 900, 955), "GER40": (540, 600, 990, 1015)}


def day_table(asset: str) -> pd.DataFrame:
    """One row per session: open/first-hour/15:00/15:55 closes + prior close (vectorised)."""
    tz = TZ[asset]
    df = clean(load_cached(asset), tz=tz).dropna()
    loc = df.index.tz_convert(tz)
    mins = (loc.hour * 60 + loc.minute).to_numpy()
    mi = pd.MultiIndex.from_arrays([loc.date, mins], names=["d", "m"])
    tab = pd.Series(df["close"].to_numpy(), index=mi).unstack("m")       # date x minute
    open_min, first_min, last_from, last_to = CLOCK[asset]
    out = pd.DataFrame({
        "open": tab.get(open_min), "c1": tab.get(first_min),
        "c2": tab.get(last_from), "c3": tab.get(last_to),
    })
    out["prev_close"] = tab.ffill(axis=1).iloc[:, -1].shift(1)
    return out.dropna().reset_index().rename(columns={"d": "day"})


def study(name: str, r: pd.DataFrame, direction: np.ndarray, move: np.ndarray,
          cost_pct: float, magnitude: np.ndarray, seed: int = 7) -> str:
    n = len(move)
    signed = direction * move
    med = np.median(signed)
    n_pos = int((signed > 0).sum())
    z = (n_pos - n / 2) / np.sqrt(n / 4)
    rng = np.random.default_rng(seed)
    placebo = float(np.mean(rng.choice(direction, n) * rng.permutation(move)))
    q = pd.qcut(pd.Series(magnitude).rank(method="first"), 5, labels=False)
    quint = [round(float(np.mean(signed[q == k])) * 100, 3) for k in range(5)]
    eff_cost = abs(med) * 100 / cost_pct if cost_pct else float("inf")
    slope = np.polyfit(range(5), np.array(quint) * np.sign(med or 1), 1)[0]
    verdict = ("PASS" if (z >= 3 and eff_cost >= 2 and abs(placebo) * 100 < 0.5 * abs(med * 100)
                          and slope > 0) else
               "FAIL(sign)" if z < 3 else
               "FAIL(cost)" if eff_cost < 2 else
               "FAIL(dose)" if slope <= 0 else "FAIL(placebo)")
    return (f"  {name:20s} n={n:5d}  med {med*100:+.3f}%  z={z:5.2f}  hit {n_pos/n:4.0%}  "
            f"cost {cost_pct:.4f}%  eff/cost {eff_cost:5.1f}  placebo {placebo*100:+.3f}%  "
            f"quint {quint}  {verdict}")


def main() -> None:
    print("### Step 2 — three hypotheses, direct clock computation, cleaned 1m data")
    for asset in TZ:
        t = day_table(asset)
        if len(t) < 200:
            print(f"{asset}: too few days")
            continue
        raw = clean(load_cached(asset), tz=TZ[asset])
        cost = float(((raw["ask_close"] - raw["close"]) / ((raw["close"] + raw["ask_close"]) / 2)).median()) * 100
        print(f"\n{asset}  days={len(t)}  median round-trip cost {cost:.4f}%")
        # H1: first-hour sign -> 10:00..15:00
        d1 = np.sign(t["c1"] - t["open"]).to_numpy()
        mv1 = ((t["c2"] - t["c1"]) / t["c1"]).to_numpy()
        print(study("H1 first-hour mom", t, d1, mv1, cost, np.abs(t["c1"] / t["open"] - 1)))
        # H2: 09:30..15:00 sign -> 15:00..15:55
        d2 = np.sign(t["c2"] - t["open"]).to_numpy()
        mv2 = ((t["c3"] - t["c2"]) / t["c2"]).to_numpy()
        print(study("H2 last-hour mom", t, d2, mv2, cost, np.abs(t["c2"] / t["open"] - 1)))
        # H3: overnight gap fill: fade the gap from the open to the close
        gap = (t["open"] / t["prev_close"] - 1).to_numpy()
        d3 = -np.sign(gap)
        mv3 = ((t["c3"] - t["open"]) / t["open"]).to_numpy()
        print(study("H3 gap fade", t, d3, mv3, cost, np.abs(gap)))


if __name__ == "__main__":
    main()
