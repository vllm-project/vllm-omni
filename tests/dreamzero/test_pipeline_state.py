# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.dreamzero import pipeline_dreamzero
from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline
from vllm_omni.diffusion.models.dreamzero.state_dreamzero import DreamZeroState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _empty_pipeline() -> DreamZeroPipeline:
    pipeline = DreamZeroPipeline.__new__(DreamZeroPipeline)
    pipeline._states = OrderedDict()
    pipeline._max_session_states = 2
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


def test_dreamzero_pipeline_state_lru_caps_retained_sessions() -> None:
    pipeline = _empty_pipeline()

    session_a = pipeline._get_or_create_state("session-a")
    pipeline._get_or_create_state("session-b")
    assert pipeline._get_or_create_state("session-a") is session_a

    pipeline._get_or_create_state("session-c")

    assert list(pipeline._states) == ["session-a", "session-c"]
    assert "session-b" not in pipeline._states


def test_dreamzero_state_cache_access_requires_initialization() -> None:
    state = DreamZeroState()

    with pytest.raises(RuntimeError, match="KV caches not initialized"):
        state.get_kv_caches()

    with pytest.raises(RuntimeError, match="Cross-attn caches not initialized"):
        state.get_crossattn_caches()

    with pytest.raises(RuntimeError, match="create_kv_caches first"):
        state.update_kv_cache(0, torch.empty(0))


def test_prefill_kv_cache_allocates_runtime_ulysses_local_heads(monkeypatch) -> None:
    pipeline = _empty_pipeline()
    state = DreamZeroState()
    state.clip_feas = None
    state.ys = None
    pipeline.cfg_scale = 1.0
    pipeline.transformer = SimpleNamespace(
        dim=5120,
        num_heads=40,
        num_layers=1,
        kv_cache_num_heads=lambda ulysses_degree=1: 40 // ulysses_degree,
    )
    seen: dict[str, tuple[int, ...]] = {}

    def record_predict_noise_maybe_with_cfg(**kwargs):
        kv_cache = kwargs["positive_kwargs"]["kv_cache"][0]
        seen["kv_cache_shape"] = tuple(kv_cache.shape)

    monkeypatch.setattr(pipeline, "predict_noise_maybe_with_cfg", record_predict_noise_maybe_with_cfg)
    monkeypatch.setattr(pipeline_dreamzero, "get_current_diffusion_config_or_none", lambda: None)
    monkeypatch.setattr(pipeline_dreamzero, "get_ulysses_parallel_world_size", lambda: 2, raising=False)

    pipeline._prefill_kv_cache(
        image_latents=torch.zeros(1, 1, 16, 2, 2),
        prompt_embeds=torch.zeros(1, 1, 8),
        negative_prompt_embeds=None,
        frame_seqlen=1,
        seq_len=1,
        do_true_cfg=False,
        state=state,
    )

    assert seen["kv_cache_shape"] == (2, 1, 0, 20, 128)
