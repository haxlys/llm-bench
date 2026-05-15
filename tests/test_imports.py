"""Smoke tests against the kind of regression the P0 review fixes addressed.

Top-level test loops every script and the dashboard, exec'ing them in a fresh
module namespace. A missing `import sys`/`Path` would surface as NameError
during exec.

The dashboard test additionally walks every `from llm_bench... import X`
statement (including lazy ones inside function bodies) via the AST and verifies
each name resolves on the target module — catches stale references like the
removed VARIANT_META / TIER_MAP without needing a Streamlit runtime.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

SCRIPTS = [
    "scripts/run_bench.py",
    "scripts/run_evals.py",
    "scripts/sync_models.py",
    "scripts/compare_quality.py",
    "scripts/aggregate_evals.py",
    "scripts/build_index.py",
    "scripts/plan_eval_catchup.py",
    "scripts/export_site_public_data.py",
    "scripts/import_programbench.py",
    "scripts/run_programbench.py",
    "scripts/run_terminal_bench.py",
]
DASHBOARD = "dashboard/app.py"


def _load(rel_path: str):
    p = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("rel_path", SCRIPTS + [DASHBOARD])
def test_module_top_level_imports(rel_path: str):
    """Every script/dashboard module loads without NameError or ImportError."""
    _load(rel_path)


@pytest.mark.parametrize("rel_path", SCRIPTS + [DASHBOARD])
def test_llm_bench_imports_resolve(rel_path: str):
    """Every `from llm_bench... import X` (top-level or lazy) resolves.

    Walks the AST so lazy imports inside function bodies are checked too —
    that's how the dead VARIANT_META/TIER_MAP imports slipped past CI.
    """
    src = (ROOT / rel_path).read_text()
    tree = ast.parse(src)
    failures: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if not node.module or not node.module.startswith("llm_bench"):
            continue
        target = importlib.import_module(node.module)
        for alias in node.names:
            if not hasattr(target, alias.name):
                failures.append(f"{node.module}.{alias.name} (line {node.lineno})")
    assert not failures, f"Stale imports in {rel_path}: {failures}"
