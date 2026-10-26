"""Every standalone script under ``studies/`` and ``backtests/`` must be runnable
as a file, i.e. it must put the repo root on ``sys.path`` before importing
``trading_lab``.

The editable install (``.venv``) relies on a ``.pth`` file that carries the macOS
``hidden`` flag, which CPython silently skips — so the editable install is
inactive and ``import trading_lab`` fails with ``ModuleNotFoundError`` when a
script is run directly (``python studies/xyz.py``). The ``sys.path.insert`` line
makes each entry-point work regardless of how the venv was installed.

This test scans ``studies/`` and ``backtests/`` (recursively, skipping
``__pycache__``) and fails for any file that imports ``trading_lab`` at module
level without that ``sys.path`` line.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("studies", "backtests")
INSERT = 'sys.path.insert(0, ".")'


def _imports_trading_lab(path: Path) -> bool:
    """True if the module imports ``trading_lab`` at module level."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:  # module-level statements only
        if isinstance(node, ast.Import):
            if any(
                a.name == "trading_lab" or a.name.startswith("trading_lab.")
                for a in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "trading_lab" or module.startswith("trading_lab."):
                return True
    return False


def test_scripts_insert_repo_root_on_sys_path() -> None:
    offenders: list[str] = []
    for rel in SCAN_DIRS:
        for path in sorted((ROOT / rel).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            if _imports_trading_lab(path) and INSERT not in path.read_text():
                offenders.append(str(path.relative_to(ROOT)))

    assert not offenders, (
        f"scripts import trading_lab at module level but miss {INSERT!r} "
        "(the editable install is inactive because the venv .pth carries the "
        "macOS 'hidden' flag): " + ", ".join(offenders)
    )
