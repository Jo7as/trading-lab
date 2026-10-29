"""H3 + H5 (docs/research/CRYPTO_ORDERFLOW_PLAN.md, phase 4): trade only extreme quarter-hour
openings, and compare a market entry with a limit entry after the counter-move.

Signal: |OI| of the first 10 s after :00/:15/:30/:45 >= threshold, direction = sign(OI)
(Kim & Hansen continuation). Trades are thinned - a new signal is taken only after the
previous trade's exit - so trades never overlap and the t-statistic is honest.

Entries (all exits at market after ``hold`` from the end of the opening window):
* market:   taker at the last trade of the opening window
* limit Xbp: resting order X bp better than that price (buy below / sell above) for up to
             30 min; filled only if a later bar trades THROUGH the limit (low < limit for a
             buy), at the limit price. Missed signals are reported, because the misses are
             often the winners (adverse selection).

Costs: ``costs.crypto_round_trip_bp`` (Bybit perp: taker 5.5 bp, maker 2 bp, + half spread).
Measurement on 2024 only (train); no parameter is fitted - thresholds and offsets are fixed
here before the run.

    uv run python studies/crypto_opening_entry_study.py
"""
from __future__ import annotations

import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from trading_lab import costs  # noqa: E402
from trading_lab import orderflow as of  # noqa: E402
from trading_lab.adapters import crypto_trades as ct  # noqa: E402

START, END = date(2024, 1, 1), date(2024, 12, 31)
THRESHOLDS = (0.6, 0.8)
LIMIT_BP = (5.0, 10.0, 20.0)
HOLDS = ("4h", "8h")
WAIT = "30min"
PROFILE = "bybit_perp"


def simulate(bars: pd.DataFrame, signals: pd.DataFrame, *, hold: str, limit_bp: float | None,
             wait: str = WAIT) -> pd.DataFrame:
    """One row per signal: filled?, entry/exit price, gross bp. Non-overlapping by construction.

    ``signals``: index = bar label of the last bar of the opening window, column ``side``.
    """
    price = bars["close"].ffill()
    lo, hi = bars["low"].to_numpy(), bars["high"].to_numpy()
    pos = {t: i for i, t in enumerate(bars.index)}
    step = bars.index[1] - bars.index[0]
    n_wait, n_hold = int(pd.Timedelta(wait) / step), int(pd.Timedelta(hold) / step)
    px = price.to_numpy()
    rows, busy_until = [], -1
    for t, side in signals["side"].items():
        i = pos[t]
        if i <= busy_until or i + n_hold >= len(px):
            continue
        ref = px[i]
        if limit_bp is None:
            fill_i, entry = i, ref
        else:
            lim = ref * (1 - side * limit_bp / 1e4)
            seg = slice(i + 1, min(i + 1 + n_wait, len(px)))
            hit = np.flatnonzero(lo[seg] < lim) if side > 0 else np.flatnonzero(hi[seg] > lim)
            if len(hit) == 0:
                exit_ref = px[i + n_hold]
                rows.append({"ts": t, "side": side, "filled": False,
                             "gross_bp": side * np.log(exit_ref / ref) * 1e4})
                busy_until = i + n_wait           # the order rested; no new signal meanwhile
                continue
            fill_i, entry = i + 1 + int(hit[0]), lim
        exit_i = i + n_hold                       # exit time fixed from the signal, not the fill
        rows.append({"ts": t, "side": side, "filled": True,
                     "gross_bp": side * np.log(px[exit_i] / entry) * 1e4})
        busy_until = exit_i
    return pd.DataFrame(rows)


def summary(res: pd.DataFrame, market: str, *, maker_entry: bool) -> dict:
    f = res[res["filled"]] if len(res) else res
    cost = costs.crypto_round_trip_bp(market, PROFILE, maker_entry=maker_entry)
    g = f["gross_bp"]
    t = g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 2 else np.nan
    missed = res[~res["filled"]]["gross_bp"] if len(res) else pd.Series(dtype=float)
    return {"signals": len(res), "filled": len(f), "gross": g.mean(), "t": t, "cost": cost,
            "net": g.mean() - cost, "missed_gross": missed.mean() if len(missed) else np.nan}


def main() -> None:
    lines = ["# H3 + H5: extreme openings, market vs limit entry (2024, Binance prices, Bybit costs)", "",
             "Non-overlapping trades; limit fills only on trade-through; exit at market after the hold.", ""]
    for market in ("BTC", "SOL"):
        bars = ct.load_bars("binance_um", market, START, END, allow_download=False)
        op = of.clock_openings(bars, every="15min", window="10s", by="qty")
        step = bars.index[1] - bars.index[0]
        last_bar = pd.DatetimeIndex(op["known_at"] - step)
        lines += [f"## {market}", "",
                  "| |OI| ≥ | hold | entry | signals | filled | gross bp | t | cost bp | net bp | missed gross bp |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for thr in THRESHOLDS:
            sel = op["imbalance"].abs() >= thr
            sig = pd.DataFrame({"side": np.sign(op["imbalance"][sel]).astype(int).to_numpy()},
                               index=last_bar[sel.to_numpy()])
            for hold in HOLDS:
                for lb in (None, *LIMIT_BP):
                    res = simulate(bars, sig, hold=hold, limit_bp=lb)
                    s = summary(res, market, maker_entry=lb is not None)
                    name = "market" if lb is None else f"limit {lb:.0f} bp"
                    lines.append(f"| {thr} | {hold} | {name} | {s['signals']} | {s['filled']} | "
                                 f"{s['gross']:+.1f} | {s['t']:+.2f} | {s['cost']:.1f} | {s['net']:+.1f} | "
                                 f"{s['missed_gross']:+.1f} |")
        lines.append("")
    text = "\n".join(lines).replace("nan", "-")
    out = ct.CACHE.parent.parent / "crypto" / "opening_entry_study.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
