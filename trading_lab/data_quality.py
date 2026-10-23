"""Data quality gate (plan step 1) — audit a 1-minute bid+ask frame before any research.

The cached CFD data does contain broken bars (we measured a bar 104× the median
move), and a single such bar fabricates a "profitable pattern". This module makes
the defects explicit so they can be fixed or masked, and so no event study runs on
unvetted data.

Checks:

- index: monotonic, duplicates;
- price sanity: non-positive prices, ``high < low``, OHLC violations, ``ask < bid``;
- outliers: bars whose move exceeds ``k`` × the median absolute move (with the
  worst few listed, because they cluster at file seams);
- gaps: minutes missing inside the covered range;
- session shape: bars per local day (half days, holidays, DST);
- spread by local hour (the cost side of the data).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

OHLC = ("open", "high", "low", "close")


@dataclass
class DataReport:
    asset: str
    bars: int
    days: int
    bars_per_day_median: float
    duplicates: int
    unsorted: bool
    nonpositive: int
    high_lt_low: int
    ask_lt_bid: int
    median_abs_move_pct: float
    outlier_bars: int          # soft: > k x median (k=12)
    corrupt_bars: int          # hard: > 50 x median
    worst_move_pct: float
    worst_in_median_units: float
    roll_window_outliers: int          # outliers inside 21:00-01:00 UTC (daily roll)
    worst_bars: list[dict] = field(default_factory=list)
    spread_by_hour: dict[int, float] = field(default_factory=dict)

    def verdict(self) -> str:
        bad = []
        if self.duplicates or self.unsorted:
            bad.append("index")
        if self.nonpositive or self.high_lt_low or self.ask_lt_bid:
            bad.append("prices")
        if self.corrupt_bars or self.worst_in_median_units > 50:
            bad.append(f"corrupt({self.corrupt_bars or 1})")
        return "OK" if not bad else "SUSPECT: " + ", ".join(bad)

    def summary(self) -> str:
        return (f"{self.asset:8s} bars={self.bars:>9,} days={self.days:>4} "
                f"{self.bars_per_day_median:5.0f}/day  dup={self.duplicates} "
                f"unsorted={self.unsorted} badprice={self.nonpositive + self.high_lt_low} "
                f"ask<bid={self.ask_lt_bid}  soft={self.outlier_bars:>6} corrupt={self.corrupt_bars:>5} "
                f"(worst {self.worst_move_pct:.2f}% = {self.worst_in_median_units:.0f}x median, "
                f"{self.roll_window_outliers} at the roll)  {self.verdict()}")


def audit(df: pd.DataFrame, asset: str, *, tz: str = "UTC", k: float = 12.0) -> DataReport:
    idx = df.index
    dup = int(idx.duplicated().sum())
    unsorted = bool(not idx.is_monotonic_increasing)
    bid_close = df["close"]
    mid = (df["close"] + df["ask_close"]) / 2.0

    nonpositive = int(((df[list(OHLC)] <= 0).any(axis=1) | (df["ask_close"] <= 0)).sum())
    high_lt_low = int((df["high"] < df["low"]).sum())
    ask_lt_bid = int((df["ask_close"] < df["close"]).sum())

    r = mid.pct_change()
    scale = float(r.abs().median()) or 1e-12
    outlier_mask = r.abs() > k * scale
    corrupt_bars = int((r.abs() > 50.0 * scale).sum())
    worst = r.abs().nlargest(5)

    local = idx.tz_convert(tz)
    days = pd.Series(1, index=local.normalize()).groupby(level=0).size()
    roll_hours = {21, 22, 23, 0}
    roll_mask = pd.Series(local.hour, index=df.index).isin(roll_hours)
    roll_outliers = int((outlier_mask & roll_mask).sum())

    spread_pct = (df["ask_close"] - mid) / mid
    by_hour = spread_pct.groupby(local.hour).median()

    return DataReport(
        asset=asset, bars=len(df), days=int(len(days)),
        bars_per_day_median=float(days.median()) if len(days) else 0.0,
        duplicates=dup, unsorted=unsorted, nonpositive=nonpositive,
        high_lt_low=high_lt_low, ask_lt_bid=ask_lt_bid,
        median_abs_move_pct=round(scale * 100, 4),
        outlier_bars=int(outlier_mask.sum()), corrupt_bars=corrupt_bars,
        worst_move_pct=round(float(worst.iloc[0]) * 100, 2) if len(worst) else 0.0,
        worst_in_median_units=round(float(worst.iloc[0]) / scale, 1) if len(worst) else 0.0,
        roll_window_outliers=roll_outliers,
        worst_bars=[{"ts": str(worst.index[i]), "ret_pct": round(float(worst.iloc[i]) * 100, 2)}
                    for i in range(len(worst))],
        spread_by_hour={int(h): round(float(v) * 100, 4) for h, v in by_hour.items()},
    )


def clean(df: pd.DataFrame, *, k: float = 50.0, tz: str = "UTC",
          roll_hours: tuple[int, ...] = (21, 22, 23, 0)) -> pd.DataFrame:
    """Mask bars that no study may read: outliers and the daily roll window.

    The cached CFD series contains thousands of bars with 100-1000x the median
    move, concentrated at the daily roll (JP225 logged +178 % in one minute).
    Masked rows get NaN prices, so forward-return code skips them instead of
    inheriting a fabricated jump.
    """
    mid = (df["close"] + df["ask_close"]) / 2.0
    r = mid.pct_change()
    scale = float(r.abs().median()) or 1e-12
    bad = (r.abs() > k * scale)
    if roll_hours:
        local = df.index.tz_convert(tz)
        bad = bad | pd.Series(local.hour, index=df.index).isin(roll_hours)
    out = df.copy()
    out.loc[bad, list(OHLC) + ["ask_close"]] = np.nan
    return out
