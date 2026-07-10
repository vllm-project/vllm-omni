# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_omni.diffusion.runtime_v2.protocol import ArtifactKind, ArtifactLayout, TaskKind
from vllm_omni.diffusion.runtime_v2.registry import (
    get_runtime_v2_adapter,
    supports_runtime_v2_model,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _request(num_steps=9):
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    return OmniDiffusionRequest(
        request_id="r",
        prompt="a cat",
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=num_steps,
            height=512,
            width=512,
        ),
    )


def _pipeline():
    pipeline = MagicMock(supports_step_execution=True)
    for name in ("prepare_encode", "denoise_step", "step_scheduler", "post_decode"):
        setattr(pipeline, name, MagicMock(name=name))
    return pipeline


def test_qwen_compiler_emits_ragged_linear_dag():
    adapter = get_runtime_v2_adapter("QwenImagePipeline")
    compiler = adapter.build_task_compiler(default_denoise_chunk_size=4)
    plan = compiler.compile_request(adapter.normalize_request(_request(), 4))

    prepare = next(task for task in plan.tasks.values() if task.kind == TaskKind.TEXT_ENCODE)
    chunks = sorted(
        (task for task in plan.tasks.values() if task.kind == TaskKind.DIT_STEP_CHUNK),
        key=lambda task: task.step_range.start,
    )
    decode = next(task for task in plan.tasks.values() if task.kind == TaskKind.VAE_DECODE)

    assert [(task.step_range.start, task.step_range.end) for task in chunks] == [
        (0, 4),
        (4, 8),
        (8, 9),
    ]
    assert chunks[0].dependencies == (prepare.task_id,)
    assert chunks[1].dependencies == (chunks[0].task_id,)
    assert chunks[2].dependencies == (chunks[1].task_id,)
    assert decode.dependencies == (chunks[2].task_id,)
    assert plan.terminal_task_ids == (decode.task_id,)
    assert prepare.outputs[0].kind is ArtifactKind.REQUEST_STATE
    assert prepare.outputs[0].layout is ArtifactLayout.WORKER_LOCAL
    assert decode.outputs[0].kind is ArtifactKind.OUTPUT
    assert plan.initial_artifacts[0].handle == prepare.inputs[0]


def test_qwen_compiler_rejects_custom_sigmas():
    adapter = get_runtime_v2_adapter("QwenImagePipeline")
    request = _request()
    request.sampling_params.sigmas = [1.0, 0.5, 0.1]

    with pytest.raises(NotImplementedError, match="custom sigmas"):
        adapter.build_task_compiler(default_denoise_chunk_size=1).compile_request(adapter.normalize_request(request, 1))


def test_qwen_pipeline_validation_and_cache_boundary():
    adapter = get_runtime_v2_adapter("QwenImagePipeline")
    pipeline = _pipeline()
    adapter.validate_pipeline(pipeline, SimpleNamespace(cache_backend=None))

    with pytest.raises(ValueError):
        adapter.validate_pipeline(SimpleNamespace(supports_step_execution=False), None)
    with pytest.raises(ValueError, match="cache_backend"):
        adapter.validate_pipeline(pipeline, SimpleNamespace(cache_backend="cache_dit"))


def test_qwen_adapter_surface_and_registry_boundary():
    adapter = get_runtime_v2_adapter("QwenImagePipeline")
    request = _request(num_steps=4)
    normalized = adapter.normalize_request(request, 2)
    executors = adapter.build_executors(_pipeline())

    assert normalized.denoise_chunk_size == 2
    assert set(executors) == {
        TaskKind.TEXT_ENCODE,
        TaskKind.DIT_STEP_CHUNK,
        TaskKind.VAE_DECODE,
    }
    assert supports_runtime_v2_model("QwenImagePipeline")
    assert not supports_runtime_v2_model("WanPipeline")
    with pytest.raises(KeyError):
        get_runtime_v2_adapter("WanPipeline")
