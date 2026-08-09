# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_ROOT = Path(__file__).parents[4]
_PRODUCTION_FILES = (
    _ROOT / "vllm_omni/diffusion/models/wan2_2/lingbot_world_camera.py",
    _ROOT / "vllm_omni/diffusion/models/wan2_2/lingbot_world_transformer.py",
    _ROOT / "vllm_omni/diffusion/models/wan2_2/pipeline_lingbot_world.py",
)


def test_all_lingbot_production_functions_have_complete_annotations() -> None:
    missing: list[str] = []
    for path in _PRODUCTION_FILES:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
            for argument in arguments:
                if argument.arg not in {"self", "cls"} and argument.annotation is None:
                    missing.append(f"{path.name}:{node.lineno}:{node.name}:{argument.arg}")
            if node.args.vararg is not None and node.args.vararg.annotation is None:
                missing.append(f"{path.name}:{node.lineno}:{node.name}:*{node.args.vararg.arg}")
            if node.args.kwarg is not None and node.args.kwarg.annotation is None:
                missing.append(f"{path.name}:{node.lineno}:{node.name}:**{node.args.kwarg.arg}")
            if node.returns is None:
                missing.append(f"{path.name}:{node.lineno}:{node.name}:return")

    assert missing == []
