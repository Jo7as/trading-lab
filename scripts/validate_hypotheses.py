#!/usr/bin/env python3
"""CLI wrapper for the hypothesis-card validator (S2 gate).

  .venv/bin/python scripts/validate_hypotheses.py docs/hypotheses/H-00*.yaml \
      --audit data/audit_status.yaml
"""
from __future__ import annotations

import sys

from trading_lab.hypothesis_card import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
