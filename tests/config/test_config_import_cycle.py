# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the NPU pytest circular import through DiffusionOutput."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import vllm_omni
import vllm_omni.config as config_pkg
from vllm_omni.config import StageConfigFactory, register_pipeline
from vllm_omni.config.config_factory import StageConfigFactory as DirectStageConfigFactory
from vllm_omni.config.pipeline_registry import register_pipeline as direct_register_pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _top_level_import_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def test_config_package_does_not_eagerly_import_pipeline_registry() -> None:
    imported = _top_level_import_modules(Path(config_pkg.__file__))
    assert "vllm_omni.config.config_factory" not in imported
    assert "vllm_omni.config.pipeline_registry" not in imported


def test_config_package_lazy_exports_still_resolve() -> None:
    assert StageConfigFactory is DirectStageConfigFactory
    assert register_pipeline is direct_register_pipeline


def test_npu_platform_defers_minimax_h3_encoder_patch() -> None:
    src = (
        Path(vllm_omni.__file__)
        .resolve()
        .parent.joinpath("platforms", "npu", "platform.py")
        .read_text(encoding="utf-8")
    )
    init_start = src.index("def __init__(self) -> None:")
    runtime_start = src.index("def init_diffusion_model_runner_runtime")
    init_src = src[init_start:runtime_start]
    runtime_src = src[runtime_start:]
    assert "apply_minimax_h3_qwen3vl_patch" not in init_src
    assert "apply_minimax_h3_qwen3vl_patch" in runtime_src
    assert "apply_minimax_h3_qwen3vl_swiglu_patch" in runtime_src


def test_lora_config_import_does_not_require_diffusion_output() -> None:
    from vllm_omni.config.lora import LoRAConfig
    from vllm_omni.diffusion.data import DiffusionOutput

    assert LoRAConfig is not None
    assert DiffusionOutput is not None
