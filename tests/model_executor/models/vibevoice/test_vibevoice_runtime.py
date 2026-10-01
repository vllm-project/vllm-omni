# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VibeVoice runtime: audio lifecycle, negative branch and cleanup contracts."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.vibevoice.audio_decode import (
    VibeVoiceAudioTokenDecodeOutput,
)
from vllm_omni.model_executor.models.vibevoice.negative_branch import (
    VibeVoiceNegativeBranch,
)
from vllm_omni.model_executor.models.vibevoice.stateful import (
    VibeVoiceStatefulInference,
)
from vllm_omni.model_executor.models.vibevoice.vibevoice import (
    VibeVoiceForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_AUDIO_BOS = 10
_AUDIO_EOS = 11
_AUDIO = 12
_EOS = 13


class _FakeKernel:
    def __init__(self) -> None:
        self.decode_calls = 0

    def sample_audio_latent(
        self, positive_condition, negative_condition, noise, *, guidance_scale, num_inference_steps=None
    ):
        return (positive_condition[:, :2] - negative_condition[:, :2]).unsqueeze(1)

    def decode_audio_token(self, audio_latent, *, acoustic_cache=None, semantic_cache=None):
        self.decode_calls += 1
        v = float(self.decode_calls)
        return VibeVoiceAudioTokenDecodeOutput(
            audio=torch.full((1, 1, 4), v),
            semantic_latent=torch.full((1, 1, 3), v),
            next_embedding=torch.full((1, 1, 4), v + 10),
            acoustic_cache=acoustic_cache or object(),
            semantic_cache=semantic_cache or object(),
        )


class _FakeNegativeBranch:
    def __init__(self) -> None:
        self.reset_ids: list[str] = []
        self.freed_ids: list[str] = []

    def reset_audio_segment(self, request_id: str) -> None:
        self.reset_ids.append(request_id)

    def forward_step(self, request_ids, input_embeddings):
        return [e.clone() for e in input_embeddings]

    def free(self, request_id: str) -> None:
        self.freed_ids.append(request_id)


class _FakeStore:
    def __init__(self) -> None:
        self.name = "negative"
        self.reset_ids: list[str] = []
        self.free_ids: list[str] = []
        self._steps: dict[str, int] = {}

    def reset(self, request_id: str) -> None:
        self.reset_ids.append(request_id)
        self._steps[request_id] = 0

    @contextmanager
    def append_and_enter_batch(self, request_ids):
        pos = torch.tensor([self._steps.get(r, 0) for r in request_ids])
        for r in request_ids:
            self._steps[r] = self._steps.get(r, 0) + 1
        yield SimpleNamespace(position=pos, sequence_length=1)

    def free(self, request_id: str) -> None:
        self.free_ids.append(request_id)


class _FakeQwen(nn.Module):
    def forward(self, input_ids=None, positions=None, intermediate_tensors=None, inputs_embeds=None):
        assert inputs_embeds is not None
        return inputs_embeds + positions.reshape(-1, 1).to(inputs_embeds)


def _stateful() -> VibeVoiceStatefulInference:
    return VibeVoiceStatefulInference(
        audio_bos_token_id=_AUDIO_BOS,
        audio_eos_token_id=_AUDIO_EOS,
        audio_token_id=_AUDIO,
        eos_token_id=_EOS,
        latent_size=2,
        condition_size=4,
        default_guidance_scale=1.3,
        default_num_diffusion_steps=10,
    )


def test_audio_lifecycle_bos_audio_eos_then_request_eos():
    """BOS resets negative segment; AUDIO produces waveform; EOS retains
    negative context; model EOS frees it."""
    stateful = _stateful()
    kernel = _FakeKernel()
    neg = _FakeNegativeBranch()
    stateful.bind_negative_branch(neg)
    emb = torch.arange(4, dtype=torch.float32).reshape(1, 4)

    # BOS
    out, audio = stateful.process_sampled_token(
        request_id="req", token_id=_AUDIO_BOS, token_embedding=emb, kernel=kernel
    )
    assert torch.equal(out, emb) and audio is None
    assert neg.reset_ids == ["req"]

    # AUDIO
    stateful.record_positive_condition("req", torch.ones(1, 4))
    stateful.record_negative_condition("req", torch.zeros(1, 4))
    out, audio = stateful.process_sampled_token(request_id="req", token_id=_AUDIO, token_embedding=emb, kernel=kernel)
    assert audio is not None
    state = stateful.get("req")
    assert state is not None and state.audio_token_count == 1

    # audio_eos — segment ends but negative context survives
    out, audio = stateful.process_sampled_token(
        request_id="req", token_id=_AUDIO_EOS, token_embedding=emb, kernel=kernel
    )
    assert audio is None
    assert neg.freed_ids == []

    # model eos — frees negative branch
    out, audio = stateful.process_sampled_token(request_id="req", token_id=_EOS, token_embedding=emb, kernel=kernel)
    assert audio is None and neg.freed_ids == ["req"]


def test_sparse_waveform_publication_drains_once():
    wrapper = object.__new__(VibeVoiceForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper._stateful = _stateful()
    wrapper._stateful.get_or_create("req").waveform_chunks_cpu.extend(
        [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    )
    hidden = torch.zeros(1, 4)
    kwargs = {"model_intermediate_buffer": [{"_omni_req_id": "req"}]}

    first = VibeVoiceForConditionalGeneration.make_omni_output(wrapper, hidden, **kwargs)
    assert first.multimodal_outputs is not None
    assert torch.equal(first.multimodal_outputs["audio"][0], torch.tensor([1.0, 2.0, 3.0, 4.0]))

    second = VibeVoiceForConditionalGeneration.make_omni_output(wrapper, hidden, **kwargs)
    assert second.multimodal_outputs == {}


def test_request_cleanup_drops_unpublished_waveform_after_abort():
    stateful = _stateful()
    neg = _FakeNegativeBranch()
    stateful.bind_negative_branch(neg)
    state = stateful.get_or_create("aborted")
    state.waveform_chunks_cpu.append(torch.ones(4))
    state.acoustic_cache = object()
    state.semantic_cache = object()

    stateful.on_requests_finished({"aborted"})

    assert stateful.get("aborted") is None
    assert state.waveform_chunks_cpu == []
    assert neg.freed_ids == ["aborted"]


def test_clear_runtime_state_releases_pending_work():
    wrapper = object.__new__(VibeVoiceForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper._stateful = _stateful()
    neg = _FakeNegativeBranch()
    wrapper._stateful.bind_negative_branch(neg)
    wrapper._stateful.get_or_create("req")
    wrapper._pending_request_ids = ["req"]
    wrapper._pending_request_spans = [("req", 0, 1)]
    wrapper._pending_audio_transitions = [("req", 0)]
    wrapper._pending_num_input_rows = 1

    VibeVoiceForConditionalGeneration.clear_runtime_state(wrapper)

    assert wrapper._stateful.active_request_ids == ()
    assert neg.freed_ids == ["req"]
    assert wrapper._pending_request_ids == []


def test_negative_executor_forward_and_free():
    store = _FakeStore()
    branch = VibeVoiceNegativeBranch(
        store=store,
        language_model=_FakeQwen(),
        hidden_size=4,  # type: ignore[arg-type]
    )
    branch.reset_audio_segment("req-a")
    first = branch.forward_step(["req-a"], [torch.ones(1, 4)])
    second = branch.forward_step(["req-a"], [torch.ones(1, 4)])
    assert torch.equal(first[0], torch.ones(1, 4))
    assert torch.equal(second[0], torch.full((1, 4), 2.0))
    branch.free("req-a")
    assert store.free_ids == ["req-a"]
