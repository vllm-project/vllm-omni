# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline
from vllm_omni.diffusion.models.dreamzero.state_dreamzero import DreamZeroState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _empty_pipeline() -> DreamZeroPipeline:
    pipeline = DreamZeroPipeline.__new__(DreamZeroPipeline)
    pipeline._states = OrderedDict()
    return pipeline


def test_dreamzero_pipeline_state_is_session_keyed() -> None:
    pipeline = _empty_pipeline()

    session_a = pipeline._get_or_create_state("session-a")
    session_b = pipeline._get_or_create_state("session-b")
    session_a.call_count = 7
    session_b.call_count = 3

    assert pipeline._get_or_create_state("session-a") is session_a
    assert pipeline._get_or_create_state("session-b") is session_b
    assert session_a.call_count == 7
    assert session_b.call_count == 3


def test_dreamzero_pipeline_state_follows_runner_lifecycle_notifications() -> None:
    pipeline = _empty_pipeline()

    session_a = pipeline._get_or_create_state("session-a")
    session_b = pipeline._get_or_create_state("session-b")
    pipeline.state = session_b

    pipeline.close_ar_diffusion_session("session-a")
    assert pipeline.state is session_b
    pipeline.reset_ar_diffusion_session("session-b")

    assert not pipeline._states
    assert pipeline.state is None
    assert pipeline._get_or_create_state("session-a") is not session_a
    assert pipeline._get_or_create_state("session-b") is not session_b


def test_close_session_clears_active_state_alias() -> None:
    pipeline = _empty_pipeline()
    session_a = pipeline._get_or_create_state("session-a")
    pipeline.state = session_a

    pipeline.close_ar_diffusion_session("session-a")

    assert not pipeline._states
    assert pipeline.state is None


def test_dreamzero_warmup_provider_builds_session_scoped_requests() -> None:
    provider = SimpleNamespace(
        ar_diffusion_kv_cache_spec=lambda: SimpleNamespace(window_frames=5, frames_per_block=4),
        od_config=SimpleNamespace(
            ar_diffusion_kv_config=None,
            model_config={"policy_server_config": {"image_resolution": [8, 16]}},
        ),
        _ar_warmup_robot_obs=DreamZeroPipeline._ar_warmup_robot_obs,
    )

    requests = list(DreamZeroPipeline.ar_diffusion_warmup_requests(provider, "warmup-session"))

    assert [request.prompt for request in requests] == ["warmup", "warmup"]
    assert all(request.sampling_params.extra_args["session_id"] == "warmup-session" for request in requests)
    assert requests[0].sampling_params.extra_args["robot_obs"]["observation/exterior_image_0_left"].shape == (
        8,
        16,
        3,
    )
    assert requests[1].sampling_params.extra_args["robot_obs"]["observation/exterior_image_0_left"].shape == (
        4,
        8,
        16,
        3,
    )


def test_dreamzero_state_owns_no_kv_caches() -> None:
    """KV (self- and cross-attention) is engine-owned since the AR-Diffusion
    paged backend: the model-local cache accessors must be gone so nothing can
    silently bypass the engine pool."""
    state = DreamZeroState()

    for removed in ("get_kv_caches", "create_kv_caches", "update_kv_cache", "get_crossattn_caches"):
        assert not hasattr(state, removed)
