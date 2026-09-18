# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).parents[3]
_CLIENTS_PACKAGE = "vllm_omni.clients"


def _imports_clients_package(path: Path) -> bool:
    """True if the module imports the client-side package at module level.

    ``vllm_omni.clients`` is client-side by construction: the serving stack
    (imported eagerly by the API server) must never depend on it.
    """
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Import):
            if any(
                alias.name == _CLIENTS_PACKAGE or alias.name.startswith(_CLIENTS_PACKAGE + ".") for alias in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == _CLIENTS_PACKAGE or node.module.startswith(_CLIENTS_PACKAGE + ".")):
                return True
            if node.module == "vllm_omni" and any(alias.name == "clients" for alias in node.names):
                return True
    return False


def test_api_server_does_not_import_client_package() -> None:
    api_server = _REPO_ROOT / "vllm_omni/entrypoints/openai/api_server.py"
    assert not _imports_clients_package(api_server)


def test_duplex_serving_stack_does_not_import_client_package() -> None:
    duplex_pkg = _REPO_ROOT / "vllm_omni/entrypoints/duplex"
    offenders = [path.name for path in sorted(duplex_pkg.glob("*.py")) if _imports_clients_package(path)]
    assert not offenders, f"duplex serving modules import the client package: {offenders}"
