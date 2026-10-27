"""Frozen portfolios: a named set of fixed strategy variants, tested as ONE hypothesis.

Why this exists: the optimiser picks the best of many variants per strategy, so a
single variant on train can never prove itself (S4 deflates it by the search). The
honest next step is to **freeze** a combination - fixed parameters, fixed risk,
pass criteria written down in advance - and test it once on data nobody looked at.
Because it is one pre-registered hypothesis, the holdout bar is the plain one-sided
5 % (t >= 1.645), no deflation.

A portfolio lives in ``configs/portfolios/<ID>.yaml``. Freezing writes a SHA-256 of
the content into ``configs/portfolios/<ID>.lock``; the holdout runner refuses to run
if the file was changed after freezing (the criteria cannot be moved after the fact).

    pf = load("TK8")
    res = evaluate(pf, window="train")      # or "holdout" (only via studies/portfolio_holdout.py)
    print(res.summary())
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from . import prop_sim, screening
from .optimizer import run_variant
from .registry import HOLDOUT_START, REGISTRY, load_cached

PF_DIR = Path(__file__).resolve().parent.parent / "configs" / "portfolios"


@dataclass
class Component:
    strategy: str
    asset: str
    params: dict[str, Any]

    @property
    def label(self) -> str:
        return f"{self.strategy}:{self.asset}"


@dataclass
class Portfolio:
    id: str
    name: str
    risk_per_trade_pct: float
    components: list[Component]
    criteria: dict[str, float]
    raw: dict = field(default_factory=dict)
    path: Path | None = None


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load(pf_id: str, *, directory: Path = PF_DIR) -> Portfolio:
    path = directory / f"{pf_id}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    comps = [Component(c["strategy"], c["asset"], dict(c["params"])) for c in raw["komponenten"]]
    return Portfolio(id=raw["id"], name=raw["name"], risk_per_trade_pct=float(raw["risiko_je_trade_pct"]),
                     components=comps, criteria=dict(raw.get("holdout_kriterien") or {}), raw=raw, path=path)


REPO = Path(__file__).resolve().parent.parent


def code_hashes(pf: "Portfolio") -> dict[str, str]:
    """SHA-256 of every source file the portfolio's results depend on: each
    component's strategy module plus the shared fill/sizing/cost code. Freezing the
    YAML alone would not stop someone from changing *how* a stop is computed."""
    from .rules import MODULES, SHARED
    files = sorted({MODULES[c.strategy] for c in pf.components if c.strategy in MODULES} | set(SHARED))
    return {f: _text_hash((REPO / f).read_text(encoding="utf-8")) for f in files}


def freeze(pf_id: str, *, directory: Path = PF_DIR, with_code: bool = True) -> str:
    """Write the lock (hash of the YAML + of the code). Refuses to overwrite one."""
    path, lock = directory / f"{pf_id}.yaml", directory / f"{pf_id}.lock"
    if lock.exists():
        raise FileExistsError(f"{lock} existiert schon - ein eingefrorenes Portfolio wird nicht neu gesperrt")
    h = _text_hash(path.read_text(encoding="utf-8"))
    rec = {"id": pf_id, "sha256": h, "eingefroren": pd.Timestamp.now().isoformat(timespec="seconds")}
    if with_code:
        rec["code_sha256"] = code_hashes(load(pf_id, directory=directory))
    lock.write_text(json.dumps(rec, indent=1))
    return h


def check(pf_id: str, *, directory: Path = PF_DIR) -> tuple[bool, list[str]]:
    """``(ok, reasons)``: is the YAML and every frozen source file unchanged?"""
    lock = directory / f"{pf_id}.lock"
    if not lock.exists():
        return False, ["nicht eingefroren (keine .lock-Datei)"]
    rec = json.loads(lock.read_text())
    why = []
    if _text_hash((directory / f"{pf_id}.yaml").read_text(encoding="utf-8")) != rec["sha256"]:
        why.append(f"{pf_id}.yaml wurde nach dem Einfrieren geändert")
    frozen = rec.get("code_sha256") or {}
    if frozen:
        now = code_hashes(load(pf_id, directory=directory))
        why += [f"Code geändert: {f}" for f, h in frozen.items() if now.get(f) != h]
    return not why, why


def is_unchanged(pf_id: str, *, directory: Path = PF_DIR) -> bool:
    return check(pf_id, directory=directory)[0]


# ── evaluation ───────────────────────────────────────────────────────────

@dataclass
class PortfolioResult:
    portfolio: Portfolio
    window: str
    daily_r: pd.DataFrame            # per component, calendar business days, R
    months: float

    @property
    def daily_pct(self) -> pd.Series:
        return self.daily_r.sum(axis=1) * self.portfolio.risk_per_trade_pct

    def stats(self) -> dict:
        d = self.daily_pct
        m = d.resample("ME").sum()
        eq = d.cumsum()
        dd = float((eq.cummax().clip(lower=0) - eq).max()) if len(eq) else 0.0
        comp = {c: {"r_summe": round(float(self.daily_r[c].sum()), 2),
                    "trades_tage": int((self.daily_r[c] != 0).sum())} for c in self.daily_r}
        return {
            "fenster": self.window, "von": str(d.index.min().date()) if len(d) else None,
            "bis": str(d.index.max().date()) if len(d) else None, "monate": round(self.months, 1),
            "pct_pro_monat_mittel": round(float(m.mean()), 3) if len(m) else None,
            "pct_pro_monat_median": round(float(m.median()), 3) if len(m) else None,
            "schlechtester_monat_pct": round(float(m.min()), 2) if len(m) else None,
            "positive_monate": round(float((m > 0).mean()), 3) if len(m) else None,
            "max_drawdown_pct": round(dd, 2),
            "schlechtester_tag_pct": round(float(-d.min()), 2) if len(d) else None,
            "t_taeglich": round(screening.t_stat(d), 3),
            "summe_pct": round(float(d.sum()), 2),
            "komponenten": comp,
            "korrelation_monat": self.daily_r.resample("ME").sum().corr().round(2).to_dict(),
        }

    def verdict(self) -> dict:
        """Pre-registered pass/fail (criteria from the frozen YAML)."""
        s, c = self.stats(), self.portfolio.criteria
        checks = {
            f"t_taeglich >= {c.get('t_min', 1.645)}": s["t_taeglich"] >= c.get("t_min", 1.645),
            "pct_pro_monat_mittel > 0": (s["pct_pro_monat_mittel"] or 0) > 0,
            f"max_drawdown_pct <= {c.get('max_dd_pct', 10.0)}": s["max_drawdown_pct"] <= c.get("max_dd_pct", 10.0),
            f"schlechtester_tag_pct <= {c.get('max_tag_pct', 5.0)}": (s["schlechtester_tag_pct"] or 0) <= c.get("max_tag_pct", 5.0),
        }
        return {"bestanden": all(checks.values()), "kriterien": checks}

    def prop(self, firm: str = "ftmo", runs: int = 2000) -> str:
        f = prop_sim.get_firm(firm)
        daily_usd = (self.daily_pct / 100.0 * f.account_size).to_numpy()
        if f.is_instant:
            return prop_sim.simulate_funded(daily_usd, f, runs=runs).summary()
        return prop_sim.evaluate(daily_usd, f, runs=runs).summary()


def _component_trades(c: Component, m1: pd.DataFrame) -> pd.DataFrame:
    spec = REGISTRY[c.strategy][0](c.asset)
    cfg = replace(spec.base, **c.params)
    return run_variant(spec, cfg, m1)


def evaluate(pf: Portfolio, *, window: str, holdout_start: pd.Timestamp = HOLDOUT_START,
             loader: Callable[[str], pd.DataFrame] = load_cached,
             progress: Callable[[str], None] | None = None) -> PortfolioResult:
    """Run every component on ``window`` ("train" or "holdout") and combine.

    The holdout run uses the FULL history (indicators keep their warm-up) and keeps
    only trades that enter on/after ``holdout_start`` - same as ``optimizer``.
    """
    if window not in ("train", "holdout"):
        raise ValueError(window)
    cache: dict[str, pd.DataFrame] = {}
    daily: dict[str, pd.Series] = {}
    lo = hi = None
    for c in pf.components:
        if c.asset not in cache:
            cache[c.asset] = loader(c.asset)
        m1 = cache[c.asset]
        if window == "train":
            m1 = m1[m1.index < holdout_start]
        tr = _component_trades(c, m1)
        if len(tr):
            tr = tr[tr["entry_time"] < holdout_start] if window == "train" else tr[tr["entry_time"] >= holdout_start]
        day = tr["exit_time"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize() if len(tr) else pd.Series(dtype="datetime64[ns]")
        daily[c.label] = tr.groupby(day.to_numpy())["r"].sum() if len(tr) else pd.Series(dtype=float)
        a, b = m1.index[0], m1.index[-1]
        if window == "holdout":
            a = max(a, holdout_start)
        lo, hi = (a if lo is None else min(lo, a)), (b if hi is None else max(hi, b))
        if progress:
            progress(f"{c.label}: {len(tr)} Trades")
    days = pd.bdate_range(lo.tz_convert("UTC").tz_localize(None).normalize(),
                          hi.tz_convert("UTC").tz_localize(None).normalize())
    df = pd.DataFrame({k: v.reindex(days, fill_value=0.0) for k, v in daily.items()}).fillna(0.0)
    return PortfolioResult(portfolio=pf, window=window, daily_r=df, months=max((hi - lo).days / 30.44, 1.0))


__all__ = ["Portfolio", "Component", "PortfolioResult", "load", "freeze", "check", "is_unchanged",
           "code_hashes", "evaluate", "PF_DIR"]
