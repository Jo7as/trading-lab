"""The guided check: same 12 steps every time, pauses where it should, holdout only on 'HOLDOUT'."""
from __future__ import annotations

from collections import deque

import pandas as pd
import pytest

from trading_lab import guided, registry
from trading_lab.guided import STEPS, GuidedCheck
from tests.test_optimizer import ToyCfg, _m1, _toy_run
from trading_lab.optimizer import Spec


@pytest.fixture()
def toy(monkeypatch, tmp_path):
    spec = lambda asset: Spec(name="toy", asset=asset, base=ToyCfg(),           # noqa: E731
                              grid={"side": (-1, 1), "hold": (10, 30, 50)},
                              prepare=lambda m1: m1, run=_toy_run)
    monkeypatch.setitem(registry.REGISTRY, "toy", (spec, ("SYN",)))
    monkeypatch.setitem(guided.REGISTRY, "toy", (spec, ("SYN",)))
    monkeypatch.setattr(guided, "LEDGER", tmp_path / "ledger.json")
    data = _m1()
    return dict(strategy="toy", asset="SYN", budget=20, out_root=tmp_path, allow_unchecked=True,
                loader=lambda a: data, holdout_start=pd.Timestamp("2022-11-01", tz="UTC"))


def _run(toy, answers):
    out, q = [], deque(answers)
    g = GuidedCheck(say=out.append, ask=lambda p: q.popleft() if q else "", **toy)
    path = g.run()
    return g, "\n".join(out), path


def test_every_run_shows_the_same_twelve_steps(toy) -> None:
    g, text, path = _run(toy, [])
    for i, name in enumerate(STEPS, 1):
        assert f"SCHRITT {i}/{len(STEPS)}: {name}" in text
    assert path is not None and path.exists()
    assert [e["schritt"] for e in g.log] == list(range(1, 13))


def test_holdout_stays_closed_without_typing_holdout(toy) -> None:
    g, text, _ = _run(toy, [])
    assert not g.holdout_used
    assert all("oos_n" not in r for r in g.rows)
    assert not guided.LEDGER.exists()


def test_holdout_opens_on_confirmation_and_is_recorded(toy) -> None:
    out = []
    g = GuidedCheck(say=out.append, ask=lambda p: "HOLDOUT" if "HOLDOUT" in p else "", **toy)
    g.run()
    assert g.holdout_used, "the toy edge should reach S4 and offer the holdout"
    assert guided.LEDGER.exists()
    assert any("oos_n" in r for r in g.rows)
    # a second run on the same strategy x instrument warns that it is no longer independent
    out2 = []
    GuidedCheck(say=out2.append, ask=lambda p: "HOLDOUT" if "HOLDOUT" in p else "", **toy).run()
    assert any("bereits am" in line for line in out2)


def test_abort_writes_the_protocol(toy) -> None:
    g, text, path = _run(toy, ["", "a"])
    assert "Abgebrochen" in text
    assert path is not None and "ABGEBROCHEN" in path.read_text()


def test_without_an_agreed_spec_the_check_stops_by_default(toy) -> None:
    """Every new agent hits this: no docs/specs/<strategy>.yaml -> abort unless chosen."""
    toy = {**toy, "allow_unchecked": None}
    g, text, path = _run(toy, [])
    assert "SPEZIFIKATION FEHLT" in text and "Abgebrochen" in text


def test_a_run_without_spec_is_marked_in_the_report(toy) -> None:
    g, text, path = _run(toy, [])
    assert "UNGEPRÜFTE UMSETZUNG" in path.read_text()
