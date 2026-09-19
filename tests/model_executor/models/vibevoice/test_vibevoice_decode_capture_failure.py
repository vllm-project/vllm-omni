# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU fault injection for cache restoration; not CUDA capture qualification."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.vibevoice.audio_decode import (
    VibeVoiceDecodeGraphExecutor,
    _DecodeCacheRestoreError,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("fail_call", [1, 2])
def test_failed_warmup_or_capture_restores_both_caches(monkeypatch, fail_call):
    caches = [
        SimpleNamespace(layers={"layer": SimpleNamespace(cache=torch.ones(3), is_initialized=True)}) for _ in range(2)
    ]
    calls = 0
    events = []

    def decode(**kwargs):
        nonlocal calls
        calls += 1
        for cache in caches:
            cache.layers["layer"].cache.add_(7)
        if calls == fail_call:
            raise ValueError("injected decode failure")
        return SimpleNamespace(audio=None, semantic_latent=None, next_embedding=None)

    stream = SimpleNamespace(wait_stream=lambda other: events.append("wait"))
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "CUDAGraph", object)
    monkeypatch.setattr(torch.cuda, "graph", lambda graph: nullcontext())
    executor = VibeVoiceDecodeGraphExecutor(SimpleNamespace(decode_audio_token=decode))
    with pytest.raises(ValueError, match="injected decode failure"):
        executor._capture(
            audio_tower=None,
            semantic_encoder=None,
            acoustic_projector=None,
            semantic_connector=None,
            latent_scaling_factor=None,
            latent_bias_factor=None,
            audio_latent=torch.zeros(1),
            acoustic_cache=caches[0],
            semantic_cache=caches[1],
        )
    assert len(events) == 2
    for cache in caches:
        assert torch.equal(cache.layers["layer"].cache, torch.ones(3))
        assert not hasattr(cache, "_vv_decode_graph")


@pytest.mark.parametrize("fatal", [False, True])
@pytest.mark.parametrize("restore_failure", [False, True])
def test_capture_error_fallback_policy(monkeypatch, fatal, restore_failure):
    executor = VibeVoiceDecodeGraphExecutor(None, capture_failure_fatal=fatal)

    def fail(**kwargs):
        if restore_failure:
            raise _DecodeCacheRestoreError("restore failed")
        raise ValueError("capture failed")

    monkeypatch.setattr(executor, "_capture", fail)
    kwargs = dict(
        audio_tower=None,
        semantic_encoder=None,
        acoustic_projector=None,
        semantic_connector=None,
        latent_scaling_factor=None,
        latent_bias_factor=None,
        audio_latent=SimpleNamespace(is_cuda=True),
        acoustic_cache=SimpleNamespace(),
        semantic_cache=SimpleNamespace(),
    )
    if fatal or restore_failure:
        with pytest.raises(RuntimeError):
            executor.decode(**kwargs)
    else:
        assert executor.decode(**kwargs) is None
    assert executor._disabled
    if restore_failure:
        with pytest.raises(RuntimeError, match="disabled"):
            executor.decode(**kwargs)
    assert not hasattr(kwargs["acoustic_cache"], "_vv_decode_graph")
