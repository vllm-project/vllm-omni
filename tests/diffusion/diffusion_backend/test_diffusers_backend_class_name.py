# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for #6731 / #6733.

Client-side stage-config resolution stamps ``model_class_name`` via
``resolve_model_class_name``. When the diffusers backend is selected, it must
resolve to ``DiffusersAdapterPipeline`` (mirroring ``enrich_config``) — never
to the checkpoint's native ``_class_name``. A native name routes the native
tensor-based post-process funcs onto the adapter's PIL/list outputs, which
crashes postprocessing (e.g. Qwen-Image "We only support pytorch tensor",
Wan2.2 I2V "'list' object has no attribute 'shape'").
"""

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.data import resolve_model_class_name
from vllm_omni.diffusion.registry import get_diffusion_post_process_func

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_diffusers_backend_resolves_adapter_even_with_model_index(mocker):
    """The adapter must win over the pipeline index for the diffusers backend."""
    mocker.patch(
        "vllm_omni.diffusion.utils.hf_utils.get_diffusion_model_index",
        return_value={"_class_name": "QwenImagePipeline"},
    )

    assert resolve_model_class_name("Qwen/Qwen-Image", diffusion_load_format="diffusers") == "DiffusersAdapterPipeline"


def test_default_backend_still_resolves_native_class_from_model_index(mocker):
    """Native pipelines keep resolving from the pipeline index."""
    mocker.patch(
        "vllm_omni.diffusion.utils.hf_utils.get_diffusion_model_index",
        return_value={"_class_name": "WanImageToVideoPipeline"},
    )

    assert (
        resolve_model_class_name("Wan-AI/Wan2.2-I2V-A14B-Diffusers", diffusion_load_format="default")
        == "WanImageToVideoPipeline"
    )


def test_diffusers_backend_resolves_adapter_without_model_index(mocker):
    """Missing/unreadable index still falls back to the adapter."""
    mocker.patch(
        "vllm_omni.diffusion.utils.hf_utils.get_diffusion_model_index",
        side_effect=OSError("no such repo"),
    )

    assert resolve_model_class_name("/models/missing", diffusion_load_format="diffusers") == "DiffusersAdapterPipeline"


def test_adapter_class_has_no_native_post_process_func():
    """The adapter's outputs must pass through: no registered post-process."""
    od_config = SimpleNamespace(model_class_name="DiffusersAdapterPipeline")

    assert get_diffusion_post_process_func(od_config) is None
