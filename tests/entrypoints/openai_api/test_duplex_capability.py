from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_api_server_does_not_import_experimental_duplex_at_module_load() -> None:
    api_server = Path(__file__).parents[3] / "vllm_omni/entrypoints/openai/api_server.py"
    module = ast.parse(api_server.read_text())

    top_level_imports = {
        node.module for node in module.body if isinstance(node, ast.ImportFrom) and isinstance(node.module, str)
    }

    assert not any(module.startswith("vllm_omni.experimental.fullduplex") for module in top_level_imports)
