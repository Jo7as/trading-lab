"""FX carry factor — a structural (non-predictive) probe.

Carry = the interest-rate differential between a pair's two currencies. The
documented cross-sectional result (Lustig/Verdelhan/Roussanov 2007) is that
high-carry currencies earn positive *excess* returns (and crash in liquidity
shocks, Brunnermeier/Nagel/Pedersen 2008).

What our data can and cannot see:

- **Price returns** of the pairs: yes (daily closes). We test whether the carry
  *ranking* predicts future price returns — an honest, out-of-sample question.
- **Swap income**: not in a price feed. It is *added* as an estimate
  (rate differential / 365 per day held) and flagged as an approximation.

The rate table below is a coarse set of annual averages per currency (rough
policy-rate levels, not exact dates). For a *ranking* factor that is adequate;
the magnitudes are documented as approximate and the result is also reported
with the ranking only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# annual average policy rate in %, per currency (approximate, for ranking only)
POLICY_RATES: dict[str, dict[int, float]] = {
    "USD": {2019: 2.25, 2020: 0.25, 2021: 0.25, 2022: 2.00, 2023: 5.25, 2024: 5.00, 2025: 4.25, 2026: 4.00},
    "EUR": {2019: -0.40, 2020: -0.50, 2021: -0.55, 2022: 1.00, 2023: 4.00, 2024: 3.50, 2025: 2.50, 2026: 2.00},
    "GBP": {2019: 0.75, 2020: 0.10, 2021: 0.10, 2022: 1.75, 2023: 5.25, 2024: 5.00, 2025: 4.00, 2026: 3.50},
    "JPY": {2019: -0.10, 2020: -0.10, 2021: -0.10, 2022: -0.10, 2023: -0.10, 2024: 0.15, 2025: 0.60, 2026: 0.75},
    "CHF": {2019: -0.75, 2020: -0.75, 2021: -0.75, 2022: 0.50, 2023: 1.75, 2024: 1.25, 2025: 0.50, 2026: 0.25},
    "AUD": {2019: 1.00, 2020: 0.25, 2021: 0.10, 2022: 2.50, 2023: 4.10, 2024: 4.35, 2025: 3.60, 2026: 3.10},
    "NZD": {2019: 1.25, 2020: 0.25, 2021: 0.25, 2022: 3.25, 2023: 5.50, 2024: 5.00, 2025: 3.75, 2026: 3.00},
    "CAD": {2019: 1.75, 2020: 0.25, 2021: 0.25, 2022: 3.00, 2023: 5.00, 2024: 4.50, 2025: 2.75, 2026: 2.50},
    "SEK": {2019: -0.25, 2020: 0.00, 2021: 0.00, 2022: 1.75, 2023: 4.00, 2024: 3.25, 2025: 2.00, 2026: 1.75},
    "NOK": {2019: 1.50, 2020: 0.00, 2021: 0.00, 2022: 2.50, 2023: 4.50, 2024: 4.50, 2025: 4.00, 2026: 3.75},
    "PLN": {2019: 1.50, 2020: 0.10, 2021: 0.10, 2022: 6.00, 2023: 5.75, 2024: 5.75, 2025: 5.25, 2026: 5.00},
    "HUF": {2019: 0.90, 2020: 0.60, 2021: 0.60, 2022: 10.00, 2023: 12.00, 2024: 8.00, 2025: 6.50, 2026: 6.00},
    "CZK": {2019: 2.00, 2020: 0.25, 2021: 0.25, 2022: 6.00, 2023: 6.50, 2024: 4.50, 2025: 3.50, 2026: 3.00},
    "MXN": {2019: 8.00, 2020: 4.50, 2021: 4.50, 2022: 8.50, 2023: 11.00, 2024: 10.00, 2025: 8.00, 2026: 7.50},
}


@dataclass(frozen=True)
class CarryConfig:
    top_k: int = 4                # long the k highest-carry pairs
    bottom_k: int = 4             # short the k lowest
    rebalance: str = "ME"         # month end
    include_swap: bool = False    # add the estimated interest accrual
    min_rate_year: int = 2019


def rate(currency: str, year: int) -> float:
    table = POLICY_RATES.get(currency)
    if not table:
        return 0.0
    return table.get(year, table.get(max(table), 0.0))


def pair_carry(pair: str, year: int) -> float:
    base, quote = pair.split("/")
    return rate(base, year) - rate(quote, year)


def monthly_returns(pairs: dict[str, pd.Series], cfg: CarryConfig) -> pd.DataFrame:
    """Equal-weight long-top / short-bottom carry portfolio, monthly rebalance.

    Returns a frame with the price return, the estimated swap accrual and the
    total, per rebalance period.
    """
    prices = pd.DataFrame({p: s for p, s in pairs.items()}).sort_index()
    prices = prices.resample("1D").last().ffill()
    month_ends = prices.resample(cfg.rebalance).last().index
    rows = []
    for a, b in zip(month_ends[:-1], month_ends[1:]):
        pa, pb = prices.loc[a], prices.loc[b]
        year = b.year
        carry = {p: pair_carry(p, year) for p in prices.columns}
        ranked = sorted(carry, key=carry.get)
        shorts, longs = ranked[: cfg.bottom_k], ranked[-cfg.top_k:]
        def pr(p: str) -> float:
            return float(pb[p] / pa[p]) - 1.0
        price_ret = np.mean([pr(p) for p in longs]) - np.mean([pr(p) for p in shorts])
        days = (b - a).days
        swap = 0.0
        if cfg.include_swap:
            swap = (np.mean([carry[p] for p in longs]) - np.mean([carry[p] for p in shorts])) / 100.0 * days / 365.0
        rows.append({"date": b, "price": price_ret, "swap": swap,
                     "total": price_ret + swap, "days": days,
                     "longs": ",".join(longs), "shorts": ",".join(shorts)})
    return pd.DataFrame(rows).set_index("date")


def metrics(daily: pd.Series) -> dict:
    r = daily.dropna()
    if len(r) < 24:
        return {"months": len(r)}
    ann = float((1 + r).prod() ** (12 / len(r)) - 1)
    vol = float(r.std(ddof=1) * np.sqrt(12))
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    neg = r[r < 0]
    return {"months": len(r), "cagr": round(ann * 100, 1), "vol": round(vol * 100, 1),
            "sharpe": round(ann / vol, 2) if vol else 0.0,
            "mdd": round(dd * 100, 1), "monthly_mean": round(r.mean() * 100, 2),
            "pos_months": round(100 * (r > 0).mean(), 1),
            "skew": round(float(r.skew()), 2), "worst": round(float(r.min()) * 100, 1),
            "downside_vol": round(float(neg.std(ddof=1) * np.sqrt(12)) * 100, 1) if len(neg) > 1 else 0.0}
