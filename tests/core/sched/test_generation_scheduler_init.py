# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise generation constructor policy after actual deploy/platform resolution."""

from types import SimpleNamespace

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler, VLLMScheduler
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def construct_scheduler(monkeypatch, mocker):
    # Only upstream cache allocation and connector construction are omitted.
    # Keep the actual Omni I/O initialization, native decision and constructor.
    mocker.patch("vllm_omni.core.sched.omni_scheduler_mixin.OmniChunkTransferAdapter")

    def upstream_init(self, config):
        self.vllm_config = config
        self.max_num_running_reqs = config.scheduler_config.max_num_seqs

    monkeypatch.setattr(VLLMScheduler, "__init__", upstream_init)
    return OmniGenerationScheduler


def _config(*, native=True, stateful=True, tp=1, pp=1, extras=None, capacity=128):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            use_v2_model_runner=native,
            supports_native_mrv2_data_plane=True,
            retains_state_across_chunks=stateful,
            async_chunk=True,
            stage_id=1,
            stage_connector_config={"extra": extras or {}},
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp, pipeline_parallel_size=pp),
        scheduler_config=SimpleNamespace(max_num_seqs=capacity),
    )


@pytest.mark.parametrize("profile", ["high_concurrency", "low_latency", "default"])
@pytest.mark.parametrize("platform", ["cuda", "npu", "xpu", "rocm", "musa"])
def test_moss_profile_generation_constructor_after_platform_resolution(
    construct_scheduler, monkeypatch, mocker, profile, platform
):
    # merge_pipeline_deploy owns platform resolution. Mock the detected device
    # instead of applying an override first and then resolving the worker's
    # actual platform a second time.
    monkeypatch.setattr(current_omni_platform, "device_name", platform)
    deploy = load_deploy_config(
        get_deploy_config_path("moss_tts_local.yaml" if profile == "default" else f"moss_tts_local_mrv2_{profile}.yaml")
    )
    pipeline = resolve_pipeline_config("moss_tts_local")
    codec = merge_pipeline_deploy(pipeline, deploy)[1]
    extras = deploy.connectors["shm"]["extra"]
    config = _config(
        native=codec.yaml_engine_args["use_v2_model_runner"],
        stateful=pipeline.stages[1].retains_state_across_chunks,
        capacity=codec.yaml_engine_args["max_num_seqs"],
        extras=extras,
    )
    warning = mocker.patch("vllm_omni.core.sched.omni_generation_scheduler.logger.warning")
    scheduler = construct_scheduler(config)
    # The raw extras persist through fallback; the live constructor must cope.
    assert extras["generation_min_batch_size"] == 32 and extras["generation_max_wait_ms"] == 12
    if platform == "cuda":
        assert scheduler._native_data_plane and scheduler.input_coordinator is not None
        assert scheduler._generation_min_batch_size == 32
        assert scheduler._generation_max_wait_s == 0.012
        assert scheduler._generation_max_regular_batch == (16 if profile == "low_latency" else 0)
        warning.assert_not_called()
        if profile == "low_latency":
            assert extras["codec_first_chunk_fast_path"] == extras["codec_first_chunk_gate"] == 1
    else:
        assert not scheduler._native_data_plane and scheduler.chunk_transfer_adapter is not None
        assert scheduler._generation_min_batch_size == 1
        assert scheduler._generation_max_wait_s == scheduler._generation_max_regular_batch == 0
        warning.assert_called_once()


@pytest.mark.parametrize("stateful,tp,pp", [(False, 1, 1), (True, 2, 1), (True, 1, 2)])
def test_native_batch_waiting_still_rejects_unsupported_state_or_topology(construct_scheduler, stateful, tp, pp):
    config = _config(
        stateful=stateful, tp=tp, pp=pp, extras={"generation_min_batch_size": 32, "generation_max_wait_ms": 12}
    )
    with pytest.raises(ValueError, match="stateful native MRV2 TP1/PP1"):
        construct_scheduler(config)


@pytest.mark.parametrize(
    "extras,message",
    [
        ({"generation_min_batch_size": 0}, "generation_min_batch_size"),
        ({"generation_min_batch_size": 129}, "generation_min_batch_size"),
        ({"generation_max_wait_ms": -1}, "generation_max_wait_ms"),
        ({"generation_max_wait_ms": float("nan")}, "generation_max_wait_ms"),
        ({"generation_max_regular_batch": -1}, "generation_max_regular_batch"),
        ({"generation_coalescing_policy": "unknown"}, "generation_coalescing_policy"),
    ],
)
def test_non_native_fallback_still_rejects_invalid_values(construct_scheduler, extras, message):
    with pytest.raises(ValueError, match=message):
        construct_scheduler(_config(native=False, extras=extras))
