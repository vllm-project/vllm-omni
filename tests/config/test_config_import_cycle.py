# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Guard the NPU pytest circular import through DiffusionOutput."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
REPO_ROOT = Path(__file__).resolve().parents[2]


def _top_level_import_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def _assert_isolated_import_succeeds(script: str, *, timeout: int = 120) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HIP_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_config_package_does_not_eagerly_import_pipeline_registry() -> None:
    imported = _top_level_import_modules(REPO_ROOT / "vllm_omni" / "config" / "__init__.py")
    assert "vllm_omni.config.config_factory" not in imported
    assert "vllm_omni.config.pipeline_registry" not in imported


def test_config_package_lazy_exports_still_resolve() -> None:
    from vllm_omni.config import StageConfigFactory, register_pipeline
    from vllm_omni.config.config_factory import StageConfigFactory as DirectStageConfigFactory
    from vllm_omni.config.pipeline_registry import register_pipeline as direct_register_pipeline

    assert StageConfigFactory is DirectStageConfigFactory
    assert register_pipeline is direct_register_pipeline


def test_npu_platform_defers_minimax_h3_encoder_patch() -> None:
    src = (REPO_ROOT / "vllm_omni" / "platforms" / "npu" / "platform.py").read_text(encoding="utf-8")
    init_start = src.index("def __init__(self) -> None:")
    runtime_start = src.index("def init_diffusion_model_runner_runtime")
    init_src = src[init_start:runtime_start]
    runtime_src = src[runtime_start:]
    assert "apply_minimax_h3_qwen3vl_patch" not in init_src
    assert "apply_minimax_h3_qwen3vl_patch" in runtime_src
    assert "apply_minimax_h3_qwen3vl_sdpa_patch" not in init_src
    assert "apply_minimax_h3_qwen3vl_sdpa_patch" in runtime_src
    assert "apply_minimax_h3_qwen3vl_swiglu_patch" in runtime_src


def test_lora_config_import_does_not_load_pipeline_registry() -> None:
    """Import LoRAConfig and diffusion.data in a fresh interpreter.

    The NPU failure was ``data.py`` importing LoRAConfig while still loading,
    then ``vllm_omni.config`` re-entering ``pipeline_registry`` → PI0 →
    DiffusionOutput. This child process must not preload those exports.
    """
    _assert_isolated_import_succeeds(
        """
import sys

from vllm_omni.config.lora import LoRAConfig

if "vllm_omni.config.pipeline_registry" in sys.modules:
    raise SystemExit("LoRAConfig import loaded pipeline_registry")
if "vllm_omni.diffusion.models.pi0.pipeline_pi0" in sys.modules:
    raise SystemExit("LoRAConfig import loaded PI0_PIPELINE")

from vllm_omni.diffusion.data import DiffusionOutput

if DiffusionOutput is None or LoRAConfig is None:
    raise SystemExit("expected LoRAConfig and DiffusionOutput")
"""
    )


def test_pipeline_registry_does_not_import_pi0_runtime() -> None:
    """Resolving the pi0 topology must not import its runtime pipeline.

    ``pipeline_pi0`` imports ``diffusion.data``. Importing it while
    ``diffusion.data`` is only partially initialized closes the Ascend plugin's
    circular-import loop, so the registry must depend only on a lightweight
    pipeline-config module.
    """
    _assert_isolated_import_succeeds(
        """
import sys

from vllm_omni.config.pipeline_registry import resolve_pipeline_config

if "vllm_omni.diffusion.models.pi0.pipeline_pi0" in sys.modules:
    raise SystemExit("pipeline_registry imported the pi0 runtime pipeline")

pipeline = resolve_pipeline_config("pi0")
if pipeline is None or pipeline.model_type != "pi0":
    raise SystemExit("expected the registered pi0 pipeline config")
"""
    )


def test_pipeline_registry_imports_without_pyav() -> None:
    """Building stage config must not require PyAV.

    The registry imports Physis pipeline configs. Those packages must not
    re-export the runtime pipelines, which load MiniMax-H3 and then ``av``.
    Non-video ``AsyncOmni`` users hit this on every engine import.
    See https://github.com/vllm-project/vllm-omni/issues/9939.
    """
    _assert_isolated_import_succeeds(
        """
import sys

sys.modules["av"] = None

from vllm_omni.config.pipeline_registry import resolve_pipeline_config

for model_type in ("physis_h3", "physis_ltx", "physis_wan"):
    pipeline = resolve_pipeline_config(model_type)
    if pipeline is None or pipeline.model_type != model_type:
        raise SystemExit(f"expected the registered {model_type} pipeline config")

eager = (
    "vllm_omni.diffusion.models.physis.h3.pipeline_physis_h3",
    "vllm_omni.diffusion.models.physis.ltx.pipeline_physis_ltx",
    "vllm_omni.diffusion.utils.media_utils",
)
loaded = [name for name in eager if name in sys.modules]
if loaded:
    raise SystemExit("pipeline registry eagerly imported " + ", ".join(loaded))

import vllm_omni

if vllm_omni.AsyncOmni is None:
    raise SystemExit("expected AsyncOmni")
""",
        timeout=180,
    )
