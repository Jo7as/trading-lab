"""Cost profiles: what a round trip really costs at a given venue.

Why this exists (review 2026-10-27): the repo charged only the Dukascopy bid/ask
spread from the data and **no commission anywhere**. Checked against published
conditions that is wrong in both directions:

* FX majors were too CHEAP - EURUSD shows 0.2 pips (0.19 bp) in the data, but every
  venue charges a commission per lot on FX (FTMO $2.50/lot/side since 2025-09-29,
  IC Markets Raw $3.50/lot/side, Dukascopy $35 per million per side), which is
  3-4x the spread itself;
* AUD/NZD, gold and several crosses were too EXPENSIVE - the Dukascopy feed quotes
  0.9-1.0 pips on AUDUSD/NZDUSD where a raw-ECN account averages 0.02-0.4 pips;
* index CFDs were about right for a prop account (FTMO: no index commission).

A profile has two parts, both applied in one place so every engine sees them:

1. **spread** - ``adjust_frame`` rescales the ask side of the 1-minute frame so the
   liquid-hours median spread equals the venue's typical spread (``None`` = keep the
   data's spread). The intraday *shape* (wide at night, tight at the overlap) is kept.
2. **commission** - ``commission_bp_side`` per side in bp of the traded notional;
   ``optimizer.normalise`` charges it per trade (two sides) in R.

Slippage stays where the engines model it, but in **relative** units (bp), never in
absolute price points (an absolute "1 point" is 0.5 bp on US100 and 2 bp on SP500).

Sources (checked 2026-10-27): FTMO trading update 25 Sep 2025; ic.com raw-spread
table; dukascopy.com commission table; CME micro contract specs. Verify before live
use - venues change their conditions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FX = {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD", "EURGBP",
      "EURJPY", "GBPJPY", "AUDJPY", "EURCHF", "AUDNZD", "AUDCAD"}
METALS = {"XAUUSD", "XAGUSD"}
INDICES = {"US100", "SP500", "US30", "GER40", "UK100", "JP225"}
LOT_UNITS = 100_000          # one FX lot = 100k units of the base currency
LIQUID_UTC_HOURS = (13, 16)  # London/NY overlap, where the "typical" spreads are quoted

# typical raw-ECN spreads in PRICE units during liquid hours (ic.com raw table, Oct 2026;
# AUDUSD set to 0.05 pip although the table says 0.02 - reviews range 0.02-0.17, take the safe side)
_ECN_SPREAD = {"EURUSD": 0.0000015, "GBPUSD": 0.000004, "USDJPY": 0.0003, "AUDUSD": 0.000005,
               "NZDUSD": 0.000038, "USDCHF": 0.000009, "USDCAD": 0.000004, "EURGBP": 0.000027,
               "XAUUSD": 0.09, "US100": 1.8, "SP500": 0.49, "GER40": 1.34}


def asset_class(asset: str) -> str:
    a = asset.replace("/", "").upper()
    if a in FX:
        return "fx"
    if a in METALS:
        return "metal"
    if a in INDICES:
        return "index"
    return "other"


@dataclass(frozen=True)
class CostProfile:
    name: str
    description: str
    #: commission per side: FX in USD per lot, metals/index/other as fraction of notional
    fx_usd_per_lot_side: float = 0.0
    metal_frac_side: float = 0.0
    index_frac_side: float = 0.0
    other_frac_side: float = 0.0
    #: typical liquid-hours spread per asset in price units; missing = keep the data
    spread_override: dict[str, float] = field(default_factory=dict)


PROFILES: dict[str, CostProfile] = {
    "data": CostProfile("data", "nur der Spread aus den Dukascopy-Daten, keine Kommission "
                                "(das alte Modell - nur zum Vergleich)"),
    "ftmo": CostProfile("ftmo", "FTMO: Spread aus den Daten (FTMO veröffentlicht keine Tabelle), "
                                "FX $2,50/Lot/Seite, Metalle 0,0007 %/Seite, Indizes ohne Kommission",
                        fx_usd_per_lot_side=2.50, metal_frac_side=0.000007),
    "ecn_raw": CostProfile("ecn_raw", "Raw-ECN (IC-Markets-artig): typische Raw-Spreads, "
                                      "$3,50/Lot/Seite auf FX und Metalle, Index-CFDs ohne Kommission",
                           fx_usd_per_lot_side=3.50, metal_frac_side=3.50 / (100 * 2300.0),
                           spread_override=dict(_ECN_SPREAD)),
    "dukascopy": CostProfile("dukascopy", "Dukascopy-Konto: Spread aus den Daten + $35 je Mio./Seite "
                                          "(FX) bzw. $52,5 je Mio./Seite (CFDs, Metalle)",
                             fx_usd_per_lot_side=0.0, metal_frac_side=52.5e-6,
                             index_frac_side=52.5e-6, other_frac_side=52.5e-6),
    "cme_micro": CostProfile("cme_micro", "CME-Micro-Futures (MNQ/MES/MGC): 1 Tick Spread, "
                                          "~$0,90 je Kontrakt und Seite inkl. Gebühren",
                             spread_override={"US100": 0.25, "SP500": 0.25, "XAUUSD": 0.10},
                             index_frac_side=0.90 / (2 * 20_000.0), metal_frac_side=0.90 / (10 * 2300.0)),
}
DEFAULT_PROFILE = "ftmo"
#: Dukascopy's FX commission is per million NOTIONAL, not per lot
_DUKA_FX_FRAC_SIDE = 35e-6


def get(name: str | None) -> CostProfile:
    key = (name or DEFAULT_PROFILE).lower()
    if key not in PROFILES:
        raise KeyError(f"unbekanntes Kostenprofil {name!r}; bekannt: {sorted(PROFILES)}")
    return PROFILES[key]


def _base_in_usd(asset: str, price: float) -> float:
    a = asset.replace("/", "").upper()
    if a.endswith("USD") and not a.startswith("USD"):
        return float(price)          # EURUSD 1.08 -> one EUR is 1.08 USD
    if a.startswith("USD"):
        return 1.0                   # USDJPY: base is USD
    return {"EUR": 1.08, "GBP": 1.27, "AUD": 0.66}.get(a[:3], 1.0)   # crosses: rough, documented


def commission_bp_side(asset: str, price: float, profile: str | CostProfile | None = None) -> float:
    """Commission per side in bp of the traded notional."""
    p = profile if isinstance(profile, CostProfile) else get(profile)
    cls = asset_class(asset)
    if cls == "fx":
        if p.name == "dukascopy":
            return _DUKA_FX_FRAC_SIDE * 1e4
        notional_usd = LOT_UNITS * _base_in_usd(asset, price)
        return p.fx_usd_per_lot_side / notional_usd * 1e4
    frac = {"metal": p.metal_frac_side, "index": p.index_frac_side}.get(cls, p.other_frac_side)
    return frac * 1e4


def adjust_frame(m1: pd.DataFrame, asset: str, profile: str | CostProfile | None = None) -> pd.DataFrame:
    """Rescale the ask side so the liquid-hours median spread matches the venue.

    Keeps the intraday shape of the spread; never lets ask fall below bid. With no
    override for this asset the frame is returned unchanged.
    """
    p = profile if isinstance(profile, CostProfile) else get(profile)
    a = asset.replace("/", "").upper()
    target = p.spread_override.get(a)
    if target is None or m1.empty or "ask_close" not in m1:
        return m1
    sp = m1["ask_close"] - m1["close"]
    h = m1.index.hour
    liquid = (h >= LIQUID_UTC_HOURS[0]) & (h < LIQUID_UTC_HOURS[1])
    med = float(np.nanmedian(sp[liquid])) if liquid.any() else float(np.nanmedian(sp))
    if not med > 0:
        return m1
    k = target / med
    out = m1.copy()
    for c in ("open", "high", "low", "close"):
        out[f"ask_{c}"] = out[c] + (m1[f"ask_{c}"] - m1[c]).clip(lower=0) * k
    return out


def summary(asset: str, price: float, profile: str | None = None) -> str:
    p = get(profile)
    return (f"Kostenprofil {p.name}: {p.description}; Kommission {asset} "
            f"{commission_bp_side(asset, price, p):.2f} bp/Seite")


__all__ = ["CostProfile", "PROFILES", "DEFAULT_PROFILE", "get", "asset_class",
           "commission_bp_side", "adjust_frame", "summary"]
