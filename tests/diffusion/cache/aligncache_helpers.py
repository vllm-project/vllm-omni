# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Execute actual source definitions on CPU with external vLLM/CUDA dependencies stubbed."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def definitions(path, names, namespace=None, class_methods=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = []
    for node in tree.body:
        if getattr(node, "name", None) not in names:
            continue
        if class_methods and isinstance(node, ast.ClassDef):
            node.bases = []
            node.decorator_list = []
            node.body = [
                n
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in class_methods
            ]
        nodes.append(node)
    if not nodes:
        raise ValueError(f"No definitions {names} in {path}")
    env = dict(namespace or {})
    # Annotations must not import model dependencies merely to define a method.
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(ROOT / path), "exec"), env)
    return env
