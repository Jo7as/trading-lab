"""Frozen portfolios: lock detects edits, train/holdout windows never mix, verdict follows the YAML."""
from __future__ import annotations

import pandas as pd
import pytest
import yaml

from trading_lab import portfolio as P
from trading_lab.optimizer import Spec
from tests.test_optimizer import ToyCfg, _m1, _toy_run

HOLD = pd.Timestamp("2022-11-01", tz="UTC")


@pytest.fixture()
def pf_dir(tmp_path, monkeypatch):
    spec = lambda asset: Spec(name="toy", asset=asset, base=ToyCfg(),           # noqa: E731
                              grid={"side": (-1, 1), "hold": (10, 30, 50)},
                              prepare=lambda m1: m1, run=_toy_run)
    monkeypatch.setitem(P.REGISTRY, "toy", (spec, ("SYN",)))
    (tmp_path / "T1.yaml").write_text(yaml.safe_dump({
        "id": "T1", "name": "Test", "risiko_je_trade_pct": 1.0,
        "holdout_kriterien": {"t_min": 1.645, "max_dd_pct": 10.0, "max_tag_pct": 5.0},
        "komponenten": [{"strategy": "toy", "asset": "SYN", "params": {"side": 1, "hold": 50}}]}))
    return tmp_path


def test_freeze_detects_any_later_edit(pf_dir) -> None:
    P.freeze("T1", directory=pf_dir)
    assert P.is_unchanged("T1", directory=pf_dir)
    with pytest.raises(FileExistsError):
        P.freeze("T1", directory=pf_dir)
    f = pf_dir / "T1.yaml"
    f.write_text(f.read_text().replace("t_min: 1.645", "t_min: 0.5"))     # moving the goalposts
    assert not P.is_unchanged("T1", directory=pf_dir)


def test_windows_do_not_overlap(pf_dir) -> None:
    pf = P.load("T1", directory=pf_dir)
    data = _m1()
    tr = P.evaluate(pf, window="train", holdout_start=HOLD, loader=lambda a: data)
    ho = P.evaluate(pf, window="holdout", holdout_start=HOLD, loader=lambda a: data)
    assert tr.daily_r.index.max() < HOLD.tz_localize(None)
    assert ho.daily_r.index.min() >= HOLD.tz_localize(None).normalize()


def test_verdict_uses_the_frozen_criteria(pf_dir) -> None:
    pf = P.load("T1", directory=pf_dir)
    data = _m1()
    res = P.evaluate(pf, window="train", holdout_start=HOLD, loader=lambda a: data)
    v = res.verdict()
    assert v["bestanden"] == all(v["kriterien"].values())
    assert any("1.645" in k for k in v["kriterien"])
    s = res.stats()
    assert s["pct_pro_monat_mittel"] > 0          # the toy edge is long on an up-drift


def test_real_portfolios_are_valid_and_frozen() -> None:
    ids = [f.stem for f in P.PF_DIR.glob("*.yaml")]
    assert any(i.startswith("TK") for i in ids)
    for pid in ids:
        pf = P.load(pid)
        assert pf.components and pf.risk_per_trade_pct > 0
        assert all(c.strategy in P.REGISTRY for c in pf.components)
        # an archived portfolio (holdout spent, result recorded) may have a broken lock;
        # every portfolio still WAITING for its test must be frozen and unchanged
        if not (P.PF_DIR / f"{pid}.result.json").exists():
            assert P.is_unchanged(pid), f"{pid} ist nicht eingefroren oder wurde geaendert"


def test_a_change_in_the_strategy_code_breaks_the_lock(pf_dir, tmp_path, monkeypatch) -> None:
    """Freezing the YAML is not enough: changing HOW a stop is computed must be caught."""
    from trading_lab import rules
    src = tmp_path / "logic.py"
    src.write_text("STOP = 1.5\n")
    monkeypatch.setattr(P, "REPO", tmp_path)
    monkeypatch.setattr(rules, "SHARED", ("logic.py",))
    P.freeze("T1", directory=pf_dir)
    assert P.check("T1", directory=pf_dir) == (True, [])
    src.write_text("STOP = 0.5\n")
    ok, why = P.check("T1", directory=pf_dir)
    assert not ok and any("logic.py" in w for w in why)


def test_rules_text_contains_the_real_numbers() -> None:
    from trading_lab.rules import describe
    from trading_lab.strategies.momentum import MomentumConfig
    d = describe("momentum", MomentumConfig(stop="atr", stop_atr=1.5, exit="time", max_hold_minutes=480))
    assert "1.5 x ATR" in d["stop_loss"]
    assert "480 Minuten" in d["take_profit_und_ausstieg"]
    assert {"entry", "stop_loss", "take_profit_und_ausstieg", "filter", "kosten"} <= set(d)
