# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for VibeVoice request-local state transitions."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.vibevoice.audio_decode import (
    VibeVoiceAudioTokenDecodeOutput,
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
        self.sample_calls: list[dict[str, Any]] = []
        self.decode_calls: list[dict[str, Any]] = []

    def sample_audio_latent(
        self,
        positive_condition: torch.Tensor,
        negative_condition: torch.Tensor,
        noise: torch.Tensor,
        *,
        guidance_scale: float,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        self.sample_calls.append(
            {
                "positive": positive_condition.clone(),
                "negative": negative_condition.clone(),
                "noise": noise.clone(),
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
            }
        )
        value = positive_condition[:, :2] - negative_condition[:, :2]
        return value.unsqueeze(1)

    def decode_audio_token(
        self,
        audio_latent: torch.Tensor,
        *,
        acoustic_cache: Any = None,
        semantic_cache: Any = None,
    ) -> VibeVoiceAudioTokenDecodeOutput:
        self.decode_calls.append(
            {
                "latent": audio_latent.clone(),
                "acoustic_cache": acoustic_cache,
                "semantic_cache": semantic_cache,
            }
        )
        next_acoustic_cache = acoustic_cache or object()
        next_semantic_cache = semantic_cache or object()
        value = float(len(self.decode_calls))
        return VibeVoiceAudioTokenDecodeOutput(
            audio=torch.full((1, 1, 4), value),
            semantic_latent=torch.full((1, 1, 3), value),
            next_embedding=torch.full((1, 1, 4), value + 10),
            acoustic_cache=next_acoustic_cache,
            semantic_cache=next_semantic_cache,
        )


class _FakeNegativeBranch:
    def __init__(self) -> None:
        self.reset_ids: list[str] = []
        self.forward_calls: list[tuple[list[str], list[torch.Tensor]]] = []
        self.freed_ids: list[str] = []

    def reset_audio_segment(self, request_id: str) -> None:
        self.reset_ids.append(request_id)

    def forward_step(
        self,
        request_ids: list[str],
        input_embeddings: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        self.forward_calls.append((list(request_ids), [embedding.clone() for embedding in input_embeddings]))
        return [embedding.clone() for embedding in input_embeddings]

    def free(self, request_id: str) -> None:
        self.freed_ids.append(request_id)


class _FakeWrapperKernel(nn.Module, _FakeKernel):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        _FakeKernel.__init__(self)
        self.forward_inputs: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        self.forward_inputs = inputs_embeds.clone()
        return inputs_embeds.clone()


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


def test_stateful_conditions_take_request_owned_copies() -> None:
    stateful = _stateful()
    source = torch.arange(4, dtype=torch.float32).reshape(1, 4)

    stateful.record_positive_condition("request-a", source)
    stateful.record_negative_input_embedding("request-a", source)
    stateful.record_negative_condition("request-a", source)
    state = stateful.get("request-a")
    assert state is not None
    saved = (
        state.positive_condition,
        state.negative_input_embedding,
        state.negative_condition,
    )
    assert all(item is not None for item in saved)
    assert all(item.data_ptr() != source.data_ptr() for item in saved if item is not None)

    source.fill_(99)
    expected = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    for item in saved:
        assert item is not None
        torch.testing.assert_close(item, expected)


def test_runtime_controls_reject_unknown_keys_without_creating_state() -> None:
    stateful = _stateful()
    invalid_key = "guid" + "ence_scale"

    with pytest.raises(ValueError, match=rf"Unsupported VibeVoice runtime controls: \['{invalid_key}'\]"):
        stateful.set_runtime_controls(
            "request-a",
            {invalid_key: 1.3},
        )

    assert stateful.get("request-a") is None


def test_runtime_control_update_is_atomic_when_validation_fails() -> None:
    stateful = _stateful()
    state = stateful.get_or_create("request-a")

    with pytest.raises(ValueError, match="num_diffusion_steps must be a positive integer"):
        stateful.set_runtime_controls(
            "request-a",
            {"guidance_scale": 2.0, "num_diffusion_steps": 1.5},
        )

    assert state.guidance_scale == 1.3
    assert state.num_diffusion_steps == 10


def test_runtime_controls_enforce_request_work_limits() -> None:
    stateful = _stateful()

    with pytest.raises(ValueError, match="guidance_scale must be between 0.0 and 20.0"):
        stateful.set_runtime_controls("request-a", {"guidance_scale": 20.1})
    with pytest.raises(ValueError, match="num_diffusion_steps cannot exceed 50"):
        stateful.set_runtime_controls("request-a", {"num_diffusion_steps": 51})


def test_audio_transition_runs_m4a_m4b_and_threads_per_request_caches() -> None:
    stateful = _stateful()
    kernel = _FakeKernel()
    negative_branch = _FakeNegativeBranch()
    stateful.bind_negative_branch(negative_branch)
    token_embedding = torch.zeros(1, 4)

    bos_embedding, audio = stateful.process_sampled_token(
        request_id="request-a",
        token_id=_AUDIO_BOS,
        token_embedding=token_embedding,
        kernel=kernel,
    )
    assert torch.equal(bos_embedding, token_embedding)
    assert audio is None
    assert negative_branch.reset_ids == ["request-a"]

    stateful.set_runtime_controls(
        "request-a",
        {"guidance_scale": 1.7, "num_diffusion_steps": 7},
    )
    stateful.record_positive_condition(
        "request-a",
        torch.tensor([[4.0, 6.0, 8.0, 10.0]]),
    )
    stateful.record_negative_condition(
        "request-a",
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
    )

    torch.manual_seed(123)
    next_embedding, first_audio = stateful.process_sampled_token(
        request_id="request-a",
        token_id=_AUDIO,
        token_embedding=token_embedding,
        kernel=kernel,
    )
    assert torch.equal(next_embedding, torch.full((1, 4), 11.0))
    assert torch.equal(first_audio, torch.ones(1, 1, 4))
    assert kernel.sample_calls[0]["positive"].shape == (1, 4)
    assert kernel.sample_calls[0]["negative"].shape == (1, 4)
    assert kernel.sample_calls[0]["noise"].shape == (2, 2)
    assert kernel.sample_calls[0]["guidance_scale"] == 1.7
    assert kernel.sample_calls[0]["num_inference_steps"] == 7

    state = stateful.get("request-a")
    assert state is not None
    first_acoustic_cache = state.acoustic_cache
    first_semantic_cache = state.semantic_cache
    assert first_acoustic_cache is not first_semantic_cache
    assert state.audio_token_count == 1
    assert len(state.waveform_chunks_cpu) == 1
    assert state.waveform_chunks_cpu[0].dtype == torch.float32
    assert state.positive_condition is None
    assert state.negative_condition is None

    stateful.record_positive_condition("request-a", torch.full((1, 4), 3.0))
    stateful.record_negative_condition("request-a", torch.full((1, 4), 1.0))
    stateful.process_sampled_token(
        request_id="request-a",
        token_id=_AUDIO,
        token_embedding=token_embedding,
        kernel=kernel,
    )
    assert kernel.decode_calls[1]["acoustic_cache"] is first_acoustic_cache
    assert kernel.decode_calls[1]["semantic_cache"] is first_semantic_cache
    assert state.audio_token_count == 2
    assert len(state.waveform_chunks_cpu) == 2


def test_waveform_chunks_are_drained_once_for_output_ownership_transfer() -> None:
    stateful = _stateful()
    state = stateful.get_or_create("request-a")
    state.waveform_chunks_cpu.extend(
        [
            torch.tensor([1.0, 2.0], dtype=torch.float32),
            torch.tensor([3.0, 4.0], dtype=torch.float32),
        ]
    )

    waveform = stateful.drain_waveform_chunks("request-a")

    assert torch.equal(waveform, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert waveform.dtype == torch.float32
    assert waveform.device.type == "cpu"
    assert state.waveform_chunks_cpu == []
    assert stateful.drain_waveform_chunks("request-a") is None
    assert stateful.drain_waveform_chunks("missing") is None


def test_active_subset_uses_one_batched_official_rng_draw() -> None:
    stateful = _stateful()
    kernel = _FakeKernel()
    for index, request_id in enumerate(("request-a", "request-b"), start=1):
        stateful.start_audio_segment(request_id)
        stateful.record_positive_condition(
            request_id,
            torch.full((1, 4), float(index + 2)),
        )
        stateful.record_negative_condition(
            request_id,
            torch.full((1, 4), float(index)),
        )

    next_embeddings, audio_chunks = stateful.process_audio_tokens_batch(
        request_ids=["request-a", "request-b"],
        token_embeddings=[torch.zeros(1, 4), torch.zeros(1, 4)],
        kernel=kernel,
    )
    assert len(kernel.sample_calls) == 1
    assert kernel.sample_calls[0]["positive"].shape == (2, 4)
    assert kernel.sample_calls[0]["negative"].shape == (2, 4)
    assert kernel.sample_calls[0]["noise"].shape == (4, 2)
    assert len(kernel.decode_calls) == 2
    assert len(next_embeddings) == 2
    assert len(audio_chunks) == 2
    assert stateful.get("request-a").audio_token_count == 1
    assert stateful.get("request-b").audio_token_count == 1


def test_model_forward_batches_negative_branch_and_writes_feedback_rows() -> None:
    wrapper = object.__new__(VibeVoiceForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper._stateful = _stateful()
    wrapper._negative_kv_branch = _FakeNegativeBranch()
    wrapper._stateful.bind_negative_branch(wrapper._negative_kv_branch)
    wrapper._pending_request_ids = ["request-a", "request-b"]
    wrapper._pending_request_spans = [
        ("request-a", 0, 1),
        ("request-b", 1, 2),
    ]
    wrapper._pending_audio_transitions = [
        ("request-a", 0),
        ("request-b", 1),
    ]
    wrapper._pending_num_input_rows = 2
    wrapper.model = _FakeWrapperKernel()

    for index, request_id in enumerate(("request-a", "request-b"), start=1):
        wrapper._stateful.start_audio_segment(request_id)
        wrapper._stateful.record_positive_condition(
            request_id,
            torch.full((1, 4), float(index + 3)),
        )
        wrapper._stateful.record_negative_input_embedding(
            request_id,
            torch.full((1, 4), float(index)),
        )

    output = VibeVoiceForConditionalGeneration.forward(
        wrapper,
        input_ids=torch.tensor([_AUDIO, _AUDIO]),
        positions=torch.tensor([1, 1]),
        inputs_embeds=torch.zeros(2, 4),
        sampling_extra_args=[
            {"guidance_scale": 1.3, "num_diffusion_steps": 10},
            {"guidance_scale": 1.3, "num_diffusion_steps": 10},
        ],
    )

    assert wrapper._negative_kv_branch.forward_calls[0][0] == [
        "request-a",
        "request-b",
    ]
    assert len(wrapper.model.sample_calls) == 1
    assert wrapper.model.sample_calls[0]["noise"].shape == (4, 2)
    assert torch.equal(output[0], torch.full((4,), 11.0))
    assert torch.equal(output[1], torch.full((4,), 12.0))
    assert torch.equal(
        wrapper._stateful.get("request-a").negative_input_embedding,
        torch.full((1, 4), 11.0),
    )
    assert torch.equal(
        wrapper._stateful.get("request-b").negative_input_embedding,
        torch.full((1, 4), 12.0),
    )
    assert not wrapper._pending_request_ids
    assert not wrapper._pending_request_spans
    assert not wrapper._pending_audio_transitions


@pytest.mark.parametrize(
    ("req_ids", "sampling_extra_args", "message"),
    [
        (["request-b"], [{}], "request metadata is misaligned"),
        (["request-a"], [], "sampling controls are misaligned"),
    ],
)
def test_model_preprocess_finalize_rejects_misaligned_runner_metadata(
    req_ids: list[str],
    sampling_extra_args: list[dict],
    message: str,
) -> None:
    wrapper = object.__new__(VibeVoiceForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper._stateful = _stateful()
    wrapper._pending_request_ids = ["request-a"]
    wrapper._pending_request_spans = [("request-a", 0, 1)]
    wrapper._pending_audio_transitions = []
    wrapper._pending_num_input_rows = 1

    with pytest.raises(ValueError, match=message):
        VibeVoiceForConditionalGeneration.preprocess_finalize(
            wrapper,
            input_ids=torch.tensor([_AUDIO]),
            inputs_embeds=None,
            req_ids=req_ids,
            sampling_extra_args=sampling_extra_args,
        )

    assert wrapper._pending_request_ids == []
    assert wrapper._pending_request_spans == []
    assert wrapper._pending_audio_transitions == []
    assert wrapper._pending_num_input_rows == 0


def test_model_preprocess_uses_reserved_omni_request_id() -> None:
    wrapper = object.__new__(VibeVoiceForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper._stateful = _stateful()
    wrapper._pending_request_ids = []
    wrapper._pending_request_spans = []
    wrapper._pending_audio_transitions = []
    wrapper._pending_num_input_rows = 0

    _, _, update = VibeVoiceForConditionalGeneration.preprocess(
        wrapper,
        # The GPU value deliberately disagrees: transition control must come
        # from the runner's already-resident CPU token span.
        input_ids=torch.tensor([999]),
        input_embeds=torch.zeros(1, 4),
        _omni_req_id="internal-request",
        request_id="user-controlled-value",
        _omni_input_token_ids_cpu=(_AUDIO_BOS,),
        _omni_is_prefill=True,
        _omni_num_computed_tokens=0,
        _omni_prompt_len=1,
    )

    assert update == {"_omni_req_id": "internal-request"}
    state = wrapper._stateful.get("internal-request")
    assert state is not None
    assert state.in_audio_segment is True
    assert wrapper._stateful.get("user-controlled-value") is None


def test_model_preprocess_routes_decode_from_cpu_token_metadata() -> None:
    wrapper = object.__new__(VibeVoiceForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper._stateful = _stateful()
    wrapper._pending_request_ids = []
    wrapper._pending_request_spans = []
    wrapper._pending_audio_transitions = []
    wrapper._pending_num_input_rows = 0

    VibeVoiceForConditionalGeneration.preprocess(
        wrapper,
        input_ids=torch.tensor([999]),
        input_embeds=torch.zeros(1, 4),
        _omni_req_id="request-a",
        _omni_input_token_ids_cpu=(_AUDIO,),
        _omni_is_prefill=False,
        _omni_num_computed_tokens=10,
        _omni_prompt_len=1,
    )

    assert wrapper._pending_audio_transitions == [("request-a", 0)]


def test_model_preprocess_rejects_missing_or_unaligned_cpu_token_metadata() -> None:
    wrapper = object.__new__(VibeVoiceForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper._stateful = _stateful()
    wrapper._pending_request_ids = []
    wrapper._pending_request_spans = []
    wrapper._pending_audio_transitions = []
    wrapper._pending_num_input_rows = 0

    with pytest.raises(ValueError, match="requires request-aligned _omni_input_token_ids_cpu"):
        VibeVoiceForConditionalGeneration.preprocess(
            wrapper,
            input_ids=torch.tensor([1, 2]),
            input_embeds=torch.zeros(2, 4),
            _omni_req_id="request-a",
            _omni_input_token_ids_cpu=(1,),
            _omni_is_prefill=True,
            _omni_num_computed_tokens=0,
            _omni_prompt_len=2,
        )


def test_model_omni_output_publishes_sparse_waveform_once() -> None:
    wrapper = object.__new__(VibeVoiceForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper._stateful = _stateful()
    wrapper._stateful.get_or_create("request-a").waveform_chunks_cpu.extend(
        [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    )
    hidden = torch.zeros(2, 4)
    kwargs = {
        "model_intermediate_buffer": [
            {"_omni_req_id": "request-a"},
            {"_omni_req_id": "request-b"},
        ]
    }

    first = VibeVoiceForConditionalGeneration.make_omni_output(
        wrapper,
        hidden,
        **kwargs,
    )
    assert first.text_hidden_states is hidden
    assert first.multimodal_outputs is not None
    assert len(first.multimodal_outputs["audio"]) == 1
    assert torch.equal(
        first.multimodal_outputs["audio"][0],
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )
    assert first.multimodal_outputs["audio"][0].dtype == torch.float32
    assert first.multimodal_outputs["sr"][0].item() == 24_000
    assert first.multimodal_outputs["meta"] == {
        "req_id": ["request-a"],
        "sparse_audio": ["1"],
        "audio_chunk_semantics": ["delta"],
    }

    second = VibeVoiceForConditionalGeneration.make_omni_output(
        wrapper,
        hidden,
        **kwargs,
    )
    assert second.multimodal_outputs == {}


def test_model_forward_flush_preserves_every_scheduled_deferred_request() -> None:
    wrapper = object.__new__(VibeVoiceForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper._stateful = _stateful()
    negative_branch = _FakeNegativeBranch()
    wrapper._stateful.bind_negative_branch(negative_branch)
    wrapper._negative_kv_branch = negative_branch
    wrapper._pending_request_ids = ["request-a", "request-c"]
    wrapper._pending_request_spans = [
        ("request-a", 0, 1),
        ("request-c", 1, 2),
    ]
    wrapper._pending_audio_transitions = []
    wrapper._pending_num_input_rows = 2
    wrapper.model = _FakeWrapperKernel()
    for request_id in ("request-a", "request-c", "idle-abort"):
        wrapper._stateful.get_or_create(request_id)
    wrapper._stateful.on_requests_finished(
        {"request-c", "idle-abort"},
        scheduled_req_ids={"request-c"},
    )

    VibeVoiceForConditionalGeneration.forward(
        wrapper,
        input_ids=torch.tensor([1, 2]),
        positions=torch.tensor([0, 0]),
        inputs_embeds=torch.zeros(2, 4),
    )

    assert wrapper._stateful.get("request-a") is not None
    assert wrapper._stateful.get("request-c") is not None
    assert wrapper._stateful.get("idle-abort") is None
    assert negative_branch.freed_ids == ["idle-abort"]
    wrapper._stateful.finish_postprocess("request-c")
    assert wrapper._stateful.get("request-c") is None


def test_model_clear_runtime_state_releases_request_state_and_pending_work() -> None:
    wrapper = object.__new__(VibeVoiceForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper._stateful = _stateful()
    negative_branch = _FakeNegativeBranch()
    wrapper._stateful.bind_negative_branch(negative_branch)
    wrapper._stateful.get_or_create("request-a")
    wrapper._pending_request_ids = ["request-a"]
    wrapper._pending_request_spans = [("request-a", 0, 1)]
    wrapper._pending_audio_transitions = [("request-a", 0)]
    wrapper._pending_num_input_rows = 1

    VibeVoiceForConditionalGeneration.clear_runtime_state(wrapper)

    assert wrapper._stateful.active_request_ids == ()
    assert negative_branch.freed_ids == ["request-a"]
    assert wrapper._pending_request_ids == []
    assert wrapper._pending_request_spans == []
    assert wrapper._pending_audio_transitions == []
    assert wrapper._pending_num_input_rows == 0


def test_model_terminal_drain_merges_existing_sparse_waveform() -> None:
    wrapper = object.__new__(VibeVoiceForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper._stateful = _stateful()
    wrapper._negative_kv_branch = _FakeNegativeBranch()
    wrapper.model = _FakeWrapperKernel()
    wrapper.get_input_embeddings = lambda: nn.Embedding(32, 4)
    wrapper._stateful.start_audio_segment("request-a")
    wrapper._stateful.record_positive_condition(
        "request-a",
        torch.full((1, 4), 4.0),
    )
    wrapper._stateful.record_negative_input_embedding(
        "request-a",
        torch.full((1, 4), 1.0),
    )
    existing = {
        "audio": [torch.tensor([8.0, 9.0])],
        "sr": [torch.tensor(24_000, dtype=torch.int32)],
        "meta": {"req_id": ["request-a"], "sparse_audio": ["1"]},
    }

    merged = VibeVoiceForConditionalGeneration.drain_terminal_sampled_tokens(
        wrapper,
        request_ids=["request-a"],
        multimodal_outputs=existing,
    )

    assert wrapper._negative_kv_branch.forward_calls[0][0] == ["request-a"]
    assert wrapper._negative_kv_branch.freed_ids[-1] == "request-a"
    assert torch.equal(
        merged["audio"][0],
        torch.tensor([8.0, 9.0, 1.0, 1.0, 1.0, 1.0]),
    )
    assert merged["sr"][0].item() == 24_000
    assert merged["meta"] == {
        "req_id": ["request-a"],
        "sparse_audio": ["1"],
        "audio_chunk_semantics": ["delta"],
    }
    assert wrapper._stateful.get("request-a").audio_token_count == 1
    assert wrapper._stateful.drain_waveform_chunks("request-a") is None


def test_model_terminal_drain_batches_same_controls_and_routes_multiple_requests() -> None:
    wrapper = object.__new__(VibeVoiceForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper._stateful = _stateful()
    wrapper._negative_kv_branch = _FakeNegativeBranch()
    wrapper.model = _FakeWrapperKernel()
    wrapper.get_input_embeddings = lambda: nn.Embedding(32, 4)
    for index, request_id in enumerate(("request-a", "request-b"), start=1):
        wrapper._stateful.start_audio_segment(request_id)
        wrapper._stateful.record_positive_condition(
            request_id,
            torch.full((1, 4), float(index + 3)),
        )
        wrapper._stateful.record_negative_input_embedding(
            request_id,
            torch.full((1, 4), float(index)),
        )
    existing = {
        "audio": [torch.tensor([8.0, 9.0])],
        "sr": [torch.tensor(24_000, dtype=torch.int32)],
        "meta": {"req_id": ["request-a"], "sparse_audio": ["1"]},
    }

    merged = VibeVoiceForConditionalGeneration.drain_terminal_sampled_tokens(
        wrapper,
        request_ids=["request-a", "request-b"],
        multimodal_outputs=existing,
    )

    assert wrapper._negative_kv_branch.forward_calls[0][0] == [
        "request-a",
        "request-b",
    ]
    assert len(wrapper.model.sample_calls) == 1
    assert wrapper.model.sample_calls[0]["noise"].shape == (4, 2)
    assert wrapper._negative_kv_branch.freed_ids[-2:] == [
        "request-a",
        "request-b",
    ]
    assert merged["meta"]["req_id"] == ["request-a", "request-b"]
    assert merged["meta"]["audio_chunk_semantics"] == ["delta", "delta"]
    assert torch.equal(
        merged["audio"][0],
        torch.tensor([8.0, 9.0, 1.0, 1.0, 1.0, 1.0]),
    )
    assert torch.equal(merged["audio"][1], torch.full((4,), 2.0))
    assert all(sample_rate.item() == 24_000 for sample_rate in merged["sr"])
    assert wrapper._stateful.get("request-a").audio_token_count == 1
    assert wrapper._stateful.get("request-b").audio_token_count == 1


def test_model_terminal_drain_groups_different_diffusion_controls() -> None:
    wrapper = object.__new__(VibeVoiceForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper._stateful = _stateful()
    wrapper._negative_kv_branch = _FakeNegativeBranch()
    wrapper.model = _FakeWrapperKernel()
    wrapper.get_input_embeddings = lambda: nn.Embedding(32, 4)
    for index, request_id in enumerate(("request-a", "request-b"), start=1):
        wrapper._stateful.start_audio_segment(request_id)
        wrapper._stateful.set_runtime_controls(
            request_id,
            {
                "guidance_scale": 1.3 + index * 0.1,
                "num_diffusion_steps": 10 + index,
            },
        )
        wrapper._stateful.record_positive_condition(request_id, torch.full((1, 4), 4.0))
        wrapper._stateful.record_negative_input_embedding(request_id, torch.full((1, 4), 1.0))

    VibeVoiceForConditionalGeneration.drain_terminal_sampled_tokens(
        wrapper,
        request_ids=["request-a", "request-b"],
        multimodal_outputs={},
    )

    assert len(wrapper.model.sample_calls) == 2
    assert [call["noise"].shape for call in wrapper.model.sample_calls] == [
        (2, 2),
        (2, 2),
    ]
    assert [call["num_inference_steps"] for call in wrapper.model.sample_calls] == [
        11,
        12,
    ]


def test_audio_transition_refuses_unguided_fallback_without_negative_paged_kv() -> None:
    stateful = _stateful()
    kernel = _FakeKernel()
    stateful.start_audio_segment("request-a")
    stateful.record_positive_condition("request-a", torch.ones(1, 4))

    with pytest.raises(
        RuntimeError,
        match="independent negative Qwen PagedAttention branch",
    ):
        stateful.process_sampled_token(
            request_id="request-a",
            token_id=_AUDIO,
            token_embedding=torch.zeros(1, 4),
            kernel=kernel,
        )
    assert not kernel.sample_calls
    assert not kernel.decode_calls


def test_audio_token_requires_fresh_one_step_conditions() -> None:
    stateful = _stateful()
    kernel = _FakeKernel()
    with pytest.raises(RuntimeError, match="no positive Qwen condition"):
        stateful.process_sampled_token(
            request_id="request-a",
            token_id=_AUDIO,
            token_embedding=torch.zeros(1, 4),
            kernel=kernel,
        )
    assert stateful.get("request-a").in_audio_segment

    stateful.start_audio_segment("request-a")
    stateful.record_positive_condition("request-a", torch.ones(1, 4))
    stateful.record_negative_condition("request-a", torch.zeros(1, 4))
    stateful.process_sampled_token(
        request_id="request-a",
        token_id=_AUDIO,
        token_embedding=torch.zeros(1, 4),
        kernel=kernel,
    )
    with pytest.raises(RuntimeError, match="no positive Qwen condition"):
        stateful.process_sampled_token(
            request_id="request-a",
            token_id=_AUDIO,
            token_embedding=torch.zeros(1, 4),
            kernel=kernel,
        )


def test_audio_eos_retains_negative_context_until_bos_or_model_eos() -> None:
    stateful = _stateful()
    kernel = _FakeKernel()
    negative_branch = _FakeNegativeBranch()
    stateful.bind_negative_branch(negative_branch)
    embedding = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    stateful.start_audio_segment("request-a")

    output, audio = stateful.process_sampled_token(
        request_id="request-a",
        token_id=_AUDIO_EOS,
        token_embedding=embedding,
        kernel=kernel,
    )
    assert torch.equal(output, embedding)
    assert audio is None
    state = stateful.get("request-a")
    assert state is not None and not state.in_audio_segment
    assert torch.equal(state.negative_input_embedding, embedding)
    assert negative_branch.freed_ids == []

    # Match the official generator's robust behavior if model logits produce
    # audio EOS -> audio token without an intervening audio BOS. The existing
    # negative/conv context continues instead of killing the EngineCore.
    stateful.record_positive_condition("request-a", torch.ones(1, 4))
    stateful.record_negative_condition("request-a", torch.zeros(1, 4))
    output, audio = stateful.process_sampled_token(
        request_id="request-a",
        token_id=_AUDIO,
        token_embedding=embedding,
        kernel=kernel,
    )
    assert output.shape == (1, 4)
    assert audio is not None
    assert state.in_audio_segment

    output, audio = stateful.process_sampled_token(
        request_id="request-a",
        token_id=_EOS,
        token_embedding=embedding,
        kernel=kernel,
    )
    assert torch.equal(output, embedding)
    assert audio is None
    assert negative_branch.freed_ids == ["request-a"]
    assert len(kernel.sample_calls) == 1


def test_request_cleanup_drops_unpublished_waveform_after_abort() -> None:
    stateful = _stateful()
    negative_branch = _FakeNegativeBranch()
    stateful.bind_negative_branch(negative_branch)
    state = stateful.get_or_create("aborted")
    state.waveform_chunks_cpu.append(torch.ones(4, dtype=torch.float32))
    state.acoustic_cache = object()
    state.semantic_cache = object()

    stateful.on_requests_finished({"aborted"})

    assert stateful.get("aborted") is None
    assert state.waveform_chunks_cpu == []
    assert state.acoustic_cache is None
    assert state.semantic_cache is None
    assert state._waveform_events == {}
    assert state._pinned_pool == []
    assert negative_branch.freed_ids == ["aborted"]


def test_request_cleanup_recycles_captured_decode_cache_pair() -> None:
    stateful = _stateful()

    class _Layer:
        is_initialized = True

        def __init__(self) -> None:
            self.cache = torch.ones(2)

    class _Cache:
        def __init__(self, *, captured: bool) -> None:
            self.layers = {"layer": _Layer()}
            if captured:
                self._vv_decode_graph = object()

    acoustic_cache = _Cache(captured=True)
    semantic_cache = _Cache(captured=False)
    old_state = stateful.get_or_create("old")
    old_state.acoustic_cache = acoustic_cache
    old_state.semantic_cache = semantic_cache

    stateful.cleanup_request("old")
    assert old_state.acoustic_cache is None
    assert old_state.semantic_cache is None

    stateful.start_audio_segment("new")
    new_state = stateful.get("new")
    assert new_state is not None
    assert new_state.acoustic_cache is acoustic_cache
    assert new_state.semantic_cache is semantic_cache
    assert torch.count_nonzero(acoustic_cache.layers["layer"].cache) == 0
    assert torch.count_nonzero(semantic_cache.layers["layer"].cache) == 0


def test_request_cleanup_waits_for_pending_waveform_copy() -> None:
    stateful = _stateful()
    state = stateful.get_or_create("request-a")
    chunk = torch.ones(4, dtype=torch.float32)
    synchronized: list[str] = []

    class _PendingEvent:
        def synchronize(self) -> None:
            synchronized.append("request-a")

    state.waveform_chunks_cpu.append(chunk)
    state._waveform_events[id(chunk)] = (_PendingEvent(), chunk)
    state._pinned_pool.append(torch.zeros(4, dtype=torch.float32))

    stateful.cleanup_request("request-a")

    assert synchronized == ["request-a"]
    assert stateful.get("request-a") is None
    assert state.waveform_chunks_cpu == []
    assert state._waveform_events == {}
    assert state._pinned_pool == []


def test_request_cleanup_drops_state_when_waveform_event_fails() -> None:
    stateful = _stateful()
    state = stateful.get_or_create("request-a")
    chunk = torch.ones(4, dtype=torch.float32)

    class _FailedEvent:
        def synchronize(self) -> None:
            raise RuntimeError("copy failed")

    state.waveform_chunks_cpu.append(chunk)
    state._waveform_events[id(chunk)] = (_FailedEvent(), chunk)

    with pytest.raises(RuntimeError, match="copy failed"):
        stateful.cleanup_request("request-a")

    assert stateful.get("request-a") is None
    assert state.waveform_chunks_cpu == []
    assert state._waveform_events == {}
    assert state._pinned_pool == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Pinned host allocation requires CUDA")
def test_drain_recycles_pinned_buffers_and_publishes_owning_copies() -> None:
    """Pinned-D2H contract, exercised with host-pinned tensors."""
    stateful = _stateful()
    state = stateful.get_or_create("request-a")

    pinned = torch.empty(4, dtype=torch.float32, pin_memory=True)
    pinned.copy_(torch.arange(4, dtype=torch.float32))
    # Record requires CUDA; emulate an already-completed event with a shim.

    class _CompletedEvent:
        def synchronize(self) -> None:
            return None

    state._waveform_events[id(pinned)] = (_CompletedEvent(), pinned)
    state.waveform_chunks_cpu.append(pinned)

    published = stateful.drain_waveform_chunks("request-a")

    assert published is not None
    assert torch.equal(published, torch.arange(4, dtype=torch.float32))
    # The published tensor must be an owning copy: the pinned buffer returned
    # to the pool may be reused by later tokens.
    assert published.data_ptr() != pinned.data_ptr()
    assert state._pinned_pool == [pinned]
    assert state._waveform_events == {}


def test_request_cleanup_is_deferred_around_the_final_scheduled_forward() -> None:
    stateful = _stateful()
    negative_branch = _FakeNegativeBranch()
    stateful.bind_negative_branch(negative_branch)
    stateful.get_or_create("finished")
    stateful.get_or_create("active")
    stateful.on_requests_finished({"finished"})

    # A different request entering preprocess is a safe point for an aborted
    # request that had no final postprocess callback.
    stateful.flush_deferred_cleanup(exclude_request_ids={"active"})
    assert stateful.get("finished") is None
    assert stateful.get("active") is not None
    assert negative_branch.freed_ids == ["finished"]

    # If the finished request is still scheduled, preserve it until its own
    # postprocess has consumed the final hidden row.
    stateful.on_requests_finished(
        {"active"},
        scheduled_req_ids={"active"},
    )
    stateful.flush_deferred_cleanup(exclude_request_ids={"active"})
    assert stateful.get("active") is not None
    stateful.finish_postprocess("active")
    assert stateful.get("active") is None
    assert negative_branch.freed_ids == ["finished", "active"]
    assert not stateful.deferred_cleanup_ids
