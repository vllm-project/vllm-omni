# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_importing_scheduler_output_does_not_eagerly_import_scheduler_implementations() -> None:
    package_init = Path(__file__).parents[3] / "vllm_omni/core/sched/__init__.py"
    module = ast.parse(package_init.read_text())
    eager_modules = {
        node.module
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module in {"omni_ar_scheduler", "omni_generation_scheduler"}
    }

    assert eager_modules == set()
    assert any(isinstance(node, ast.FunctionDef) and node.name == "__getattr__" for node in module.body)
