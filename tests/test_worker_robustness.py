"""Regression guards for crash-class bugs found in the pipeline audit (2026-10-26).

Three modules could not run at all:

* ``studies/cost_move_screen.py`` used ``NEAR_K`` without importing it — the
  study printed its whole matrix and then died with a ``NameError``.
* ``trading_lab/st_evolution_worker.py::_baseline_path`` and
  ``studies/compare_live_sim.py::_default_state_dir/_default_strategy`` wrote
  ``env = env(...)``, which makes ``env`` local for the whole function and turns
  the call into an ``UnboundLocalError``.
"""
from __future__ import annotations

import ast
import builtins
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _undefined_globals(path: pathlib.Path) -> list[str]:
    """Names the module loads but never defines or imports (a runtime NameError)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    defined = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            defined.update(node.names)
    return [
        f"{path}:{node.lineno}: {node.id}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in defined
    ]


def test_no_runnable_script_loads_an_undefined_name():
    offenders: list[str] = []
    for folder in ("trading_lab", "studies", "backtests"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            offenders.extend(_undefined_globals(path))
    assert not offenders, offenders


def _load_by_path(relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_st_worker_baseline_path_honours_the_env_override(tmp_path, monkeypatch):
    from trading_lab.st_evolution_worker import STEvolutionWorker

    class _Stub:
        state_dir = tmp_path

    monkeypatch.delenv("LAB_BASELINE_PATH", raising=False)
    assert STEvolutionWorker._baseline_path(_Stub()) == tmp_path / "hf_baseline.yaml"

    override = tmp_path / "custom_baseline.yaml"
    override.write_text("asset: XAU/USD\n")
    monkeypatch.setenv("LAB_BASELINE_PATH", str(override))
    assert STEvolutionWorker._baseline_path(_Stub()) == override


def test_compare_live_sim_default_helpers_return_paths(tmp_path, monkeypatch):
    module = _load_by_path("studies/compare_live_sim.py")

    monkeypatch.delenv("LAB_STATE", raising=False)
    monkeypatch.delenv("LAB_STRATEGY_PATH", raising=False)
    assert isinstance(module._default_state_dir(), pathlib.Path)
    assert isinstance(module._default_strategy(tmp_path), pathlib.Path)

    monkeypatch.setenv("LAB_STATE", str(tmp_path))
    monkeypatch.setenv("LAB_STRATEGY_PATH", str(tmp_path / "s.yaml"))
    assert module._default_state_dir() == tmp_path
    assert module._default_strategy(tmp_path) == tmp_path / "s.yaml"
