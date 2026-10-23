"""Hypothesis card (S2 gate): a claim may only be tested if it is pre-registered.

A card is rejected unless it states a claim *with a number*, a mechanism, a
placebo and kill criteria — and unless the S0 cost question is already answered
(``kosten_ratio`` measured with ``studies/cost_gate.py``). This is deliberately
pedantic: every failure in this project came from a claim that was never written
down before the test.

    card = load_hypothesis("docs/hypotheses/H-2026-001.yaml")
    problems = card.validate(cost_ratio_min=2.0)
    if problems: ...            # do not test yet
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REQUIRED = ("id", "titel", "beobachtung", "mechanismus", "behauptung",
            "dosis_wirkung", "placebo", "horizont", "kosten_ratio", "kill_kriterien")
HAS_NUMBER = re.compile(r"\d")
VALID_STATUS = ("offen", "in_test", "bestaetigt", "verworfen")


@dataclass
class Hypothesis:
    data: dict[str, Any]
    path: Path | None = None
    problems: list[str] = field(default_factory=list)

    def validate(self, *, cost_ratio_min: float = 2.0) -> list[str]:
        p: list[str] = []
        for key in REQUIRED:
            if key not in self.data or self.data[key] in (None, "", []):
                p.append(f"missing field: {key}")
        claim = str(self.data.get("behauptung", ""))
        if claim and not HAS_NUMBER.search(claim):
            p.append("claim must contain a number (testable, not 'it tends to')")
        if not str(self.data.get("mechanismus", "")).strip():
            p.append("mechanism required (who trades why)")
        if not self.data.get("kill_kriterien"):
            p.append("kill criteria required before the test")
        ratio = self.data.get("kosten_ratio")
        if isinstance(ratio, (int, float)) and ratio < cost_ratio_min:
            p.append(f"cost gate: E|move|/cost = {ratio} < {cost_ratio_min} — hopeless at this horizon")
        if self.data.get("status") not in (None, *VALID_STATUS):
            p.append(f"status must be one of {VALID_STATUS}")
        self.problems = p
        return p

    @property
    def horizon(self) -> Any:
        return self.data.get("horizont")


def load_hypothesis(path: str | Path) -> Hypothesis:
    p = Path(path)
    return Hypothesis(data=yaml.safe_load(p.read_text(encoding="utf-8")) or {}, path=p)
