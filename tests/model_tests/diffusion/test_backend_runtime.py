# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Connect deploy backend selection to two real requests in a tiny FLUX worker.

Random tiny weights exercise native execution, not image quality. Backend and
executor choices come only from deploy YAML, never duplicate engine kwargs.
"""

import json
import os
from pathlib import Path
from typing import cast

import pytest
import yaml
from vllm.model_executor.layers.utils import default_unquantized_gemm
from vllm.platforms import current_platform

from tests.helpers.client import send_async_image_diffusion_request
from tests.helpers.diffusion_backend_probe import BackendRuntimeProbeSnapshot
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import AsyncOmniParams, AsyncOmniRunner
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.fixture
def backend_tiny_flux_path(run_level: str, request: pytest.FixtureRequest, model_name: str) -> str:
    if run_level != "core_model":
        pytest.skip("Backend runtime acceptance uses generated tiny weights at core_model level")
    assert model_name == "FluxPipeline"
    # Resolve lazily: higher run levels must not download the full checkpoint.
    return request.getfixturevalue("tiny_model_paths")[model_name]


@pytest.fixture
def async_omni_resolved_params(backend_tiny_flux_path: str, executor_backend: str, tmp_path: Path) -> AsyncOmniParams:
    deploy_path = tmp_path / "backend-runtime.yaml"
    deploy_path.write_text(
        yaml.safe_dump(
            {
                "dtype": "bfloat16",
                "distributed_executor_backend": executor_backend,
                "stages": [
                    {
                        "stage_id": 0,
                        "devices": "0",
                        "linear_backend": "torch",
                        "diffusion_attention_config": {"default": "torch_sdpa"},
                        "enforce_eager": True,
                        "engine_extras": {
                            "worker_extension_cls": "tests.helpers.diffusion_backend_probe.DiffusionBackendProbe",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return AsyncOmniParams(
        model=backend_tiny_flux_path,
        deploy_config=str(deploy_path),
        extra_omni_kwargs={"stage_init_timeout": 180, "init_timeout": 240},
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Native FLUX CUDA backend runtime acceptance")
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["FluxPipeline"])
@pytest.mark.parametrize("executor_backend", ["mp", "uni"])
async def test_tiny_flux_deploy_backends_execute(
    async_omni_runner_function: AsyncOmniRunner, executor_backend: str
) -> None:
    engine = async_omni_runner_function.engine
    expected_gemm = f"{default_unquantized_gemm.__module__}.{default_unquantized_gemm.__qualname__}"
    expected_attention = f"{SDPAImpl.__module__}.{SDPAImpl.__qualname__}"

    try:
        for request_index in range(2):
            assert await engine.collective_rpc("reset_backend_runtime_probe", stage_ids=[0], timeout=60) == [[True]]
            reset_results = await engine.collective_rpc("get_backend_runtime_probe", stage_ids=[0], timeout=60)
            assert isinstance(reset_results, list) and len(reset_results) == 1
            assert isinstance(reset_results[0], list) and len(reset_results[0]) == 1
            assert isinstance(reset_results[0][0], dict)
            reset = cast(BackendRuntimeProbeSnapshot, reset_results[0][0])

            assert reset["pipeline_type"] == "vllm_omni.diffusion.models.flux.pipeline_flux.FluxPipeline"
            assert reset["linear_backend"] == "torch"
            assert reset["diffusion_attention_backend"] == "torch_sdpa"
            assert reset["executor_backend"] == executor_backend
            assert reset["diffusion_load_format"] == "default"
            assert reset["enforce_eager"] is True
            assert reset["dtype"] == "torch.bfloat16"
            assert reset["rank"] == 0
            assert reset["device"] == "cuda:0"
            if executor_backend == "mp":
                assert reset["pid"] != os.getpid()
            else:
                assert reset["pid"] == os.getpid()
            assert reset["selector_inputs"] and set(reset["selector_inputs"]) == {"torch"}
            assert reset["linear_impls"] and set(reset["linear_impls"].values()) == {expected_gemm}
            assert reset["attention_impls"] and set(reset["attention_impls"].values()) == {expected_attention}
            assert reset["linear_calls"] == dict.fromkeys(reset["linear_impls"], 0)
            assert reset["attention_calls"] == dict.fromkeys(reset["attention_impls"], 0)
            assert reset["transformer_calls"] == 0
            assert reset["transformer_output_shapes"] == []
            assert reset["transformer_outputs_finite"] == []

            request_id = f"backend-{executor_backend}-{request_index}"
            await send_async_image_diffusion_request(
                engine,
                prompt="A small red cube",
                request_id=request_id,
                sampling_params=OmniDiffusionSamplingParams(
                    height=128,
                    width=128,
                    num_inference_steps=2,
                    num_outputs_per_prompt=1,
                    guidance_scale=0.0,
                    max_sequence_length=64,
                    seed=42,
                ),
            )
            results = await engine.collective_rpc("get_backend_runtime_probe", stage_ids=[0], timeout=60)
            assert isinstance(results, list) and len(results) == 1
            assert isinstance(results[0], list) and len(results[0]) == 1
            assert isinstance(results[0][0], dict)
            snapshot = cast(BackendRuntimeProbeSnapshot, results[0][0])
            assert snapshot["pid"] == reset["pid"]
            assert snapshot["selector_inputs"] == reset["selector_inputs"]
            assert snapshot["linear_impls"] == reset["linear_impls"]
            assert snapshot["attention_impls"] == reset["attention_impls"]
            assert snapshot["linear_calls"].keys() == reset["linear_calls"].keys()
            assert snapshot["attention_calls"].keys() == reset["attention_calls"].keys()
            assert all(count > 0 for count in snapshot["linear_calls"].values())
            assert all(count > 0 for count in snapshot["attention_calls"].values())
            assert snapshot["transformer_calls"] > 0
            assert len(snapshot["transformer_output_shapes"]) == snapshot["transformer_calls"]
            assert all(len(shape) == 3 and shape[0] == 1 for shape in snapshot["transformer_output_shapes"])
            assert snapshot["transformer_outputs_finite"] == [True] * snapshot["transformer_calls"]
            print(json.dumps({"request_id": request_id, "backend_runtime": snapshot}, sort_keys=True))
    finally:
        # The shared fixture still shuts the engine down if this RPC fails.
        assert await engine.collective_rpc("stop_backend_runtime_probe", stage_ids=[0], timeout=60) == [[True]]
