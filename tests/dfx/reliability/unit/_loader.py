# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Load selected real functions without importing GPU runtime dependencies."""

from __future__ import annotations

import ast
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

RELIABILITY_ROOT = Path(__file__).resolve().parents[1]


def exec_ast_nodes(nodes: list[ast.stmt], source: Path, namespace: dict[str, Any]) -> None:
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)


def load_scenario(name: str) -> Any:
    """Load real scenario bodies with their existing lightweight helpers."""
    from tests.dfx.reliability import helpers

    source = RELIABILITY_ROOT / "scenarios" / f"{name}.py"
    module = ModuleType(f"{name}_under_test")
    module.__dict__.update({key: value for key, value in vars(helpers).items() if not key.startswith("__")})
    module.__dict__["threading"] = threading
    nodes: list[ast.stmt] = [
        node
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]
    exec_ast_nodes(nodes, source, module.__dict__)
    return module
