# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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


def test_cached_request_data_accepts_platform_base_default_fields() -> None:
    """Ascend adds replay_start to the upstream dataclass before Omni imports."""
    from dataclasses import dataclass, field

    @dataclass
    class PlatformCachedRequestData:
        req_ids: list[str]
        replay_start: dict[str, int] = field(default_factory=dict)

    output_path = Path(__file__).parents[3] / "vllm_omni/core/sched/output.py"
    source = ast.parse(output_path.read_text())
    cached_class = next(
        node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "OmniCachedRequestData"
    )
    from typing import Any

    namespace: dict[str, Any] = {
        "dataclass": dataclass,
        "field": field,
        "CachedRequestData": PlatformCachedRequestData,
    }
    exec(compile(ast.Module(body=[cached_class], type_ignores=[]), str(output_path), "exec"), namespace)
    cached = namespace["OmniCachedRequestData"](
        req_ids=["req"],
        replay_start={"req": 8},
        prompt_token_ids={"req": [1, 2]},
        additional_information={"req": None},
    )
    assert cached.replay_start == {"req": 8}
    assert cached.prompt_token_ids == {"req": [1, 2]}
    assert cached.additional_information == {"req": None}
