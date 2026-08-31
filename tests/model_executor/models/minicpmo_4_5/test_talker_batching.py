# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-alignment tests for MiniCPM-o 4.5's native Talker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    ConditionalChatTTSConfig,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    _CODEC_PENALTY_WINDOW,
    _DUPLEX_CODEC_TOKENS_PER_CHUNK,
    _OFFLINE_CODEC_MAX_NEW_TOKENS,
    _REPETITION_PENALTY_CHUNK_SIZE,
    MiniCPMO45OmniTTSForConditionalGeneration,
    _apply_batched_repetition_penalty,
    _native_duplex_chunk_budget,
    _restore_weight_norm_weight,
    blank_scheduler_prompt_for_penalties,
)
from vllm_omni.utils.mm_outputs import to_payload_element

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _PenaltyMetadata:
    """Stand-in for the penalty-relevant slice of vLLM's SamplingMetadata."""

    prompt_token_ids: torch.Tensor | None


@dataclass
class _SamplerOutput:
    """Stand-in for vLLM's SamplerOutput."""

    sampled_token_ids: torch.Tensor


@dataclass
class _CodecSamplingMetadata:
    """Stand-in for the codec-penalty slice of vLLM's SamplingMetadata."""

    repetition_penalties: torch.Tensor
    prompt_token_ids: torch.Tensor | None = None
    no_penalties: bool = False


class _FakeNativeTalker(nn.Module):
    has_preprocess = True

    def __init__(self) -> None:
        super().__init__()
        self.forward_kwargs: dict[str, Any] | None = None

    def forward(self, **kwargs: Any) -> torch.Tensor:
        self.forward_kwargs = kwargs
        return torch.ones(2, 4)


def test_tts_config_exposes_codec_vocab_to_vllm() -> None:
    cfg = ConditionalChatTTSConfig(num_audio_tokens=6562)
    assert cfg.vocab_size == 6562
    assert cfg.eos_token_id == 6561


def test_wrapper_always_delegates_talker_to_native_ar_path() -> None:
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "tts"
    talker = _FakeNativeTalker()
    model.talker = talker

    output = model(
        input_ids=torch.tensor([1, 2]),
        positions=torch.arange(2),
        model_intermediate_buffer=[{"request_id": "req"}],
    )

    assert output.shape == (2, 4)
    assert talker.forward_kwargs is not None
    assert talker.forward_kwargs["model_intermediate_buffer"][0]["request_id"] == "req"


def _make_talker() -> MiniCPMO45OmniTTSForConditionalGeneration:
    talker = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    nn.Module.__init__(talker)
    talker._num_audio_tokens = 8
    talker._codec_eos_id = 7
    talker._force_eos_rows = None
    talker._mask_eos_rows = None
    talker._pending_force_eos_rows = None
    talker._penalty_histories = None
    talker._request_audio_states = {}
    talker._request_condition_states = {}
    talker._deferred_cleanup_ids = set()
    talker._tts_config = ConditionalChatTTSConfig()
    talker.head_code = nn.ModuleList([nn.Linear(2, 8, bias=False)])
    with torch.no_grad():
        talker.head_code[0].weight.copy_(torch.eye(8, 2))
    return talker


def _routed(output, index: int):
    return to_payload_element(
        output.multimodal_outputs,
        index,
        index,
        index + 1,
        seq_len=2,
        scheduled_seq_len=2,
    )


def _reference_repetition_penalty(
    logits: torch.Tensor,
    history: torch.Tensor,
    *,
    penalty: float,
    window_size: int,
) -> torch.Tensor:
    if penalty == 1.0 or history.numel() == 0:
        return logits
    recent = history.reshape(-1)[-window_size:].to(device=logits.device, dtype=torch.long)
    frequencies = torch.bincount(recent, minlength=logits.shape[-1]).to(dtype=logits.dtype)
    alpha = torch.pow(torch.as_tensor(penalty, device=logits.device, dtype=logits.dtype), frequencies)
    return torch.where(logits < 0, logits * alpha, logits / alpha)


def test_weight_norm_restore_matches_checkpoint_parametrization_in_bfloat16() -> None:
    generator = torch.Generator().manual_seed(42)
    weight_v = torch.randn(8, 16, generator=generator, dtype=torch.bfloat16)
    weight_g = torch.rand(8, 1, generator=generator, dtype=torch.bfloat16)
    linear = nn.utils.parametrizations.weight_norm(
        nn.Linear(16, 8, bias=False, dtype=torch.bfloat16),
        dim=0,
    )
    with torch.no_grad():
        linear.parametrizations.weight.original0.copy_(weight_g)
        linear.parametrizations.weight.original1.copy_(weight_v)

    restored = _restore_weight_norm_weight(weight_g, weight_v)

    assert torch.equal(restored, linear.weight)


def test_batched_repetition_penalty_matches_request_local_rows() -> None:
    logits = torch.tensor(
        [
            [-2.0, -1.0, 1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0, -1.0, -2.0],
            [1.0, -1.0, 2.0, -2.0, 3.0, -3.0],
        ]
    )
    histories = [
        torch.tensor([1, 1, 2, 5]),
        torch.tensor([0, 4, 4]),
        torch.empty(0, dtype=torch.long),
    ]

    actual = _apply_batched_repetition_penalty(
        logits,
        histories,
        penalty=1.2,
        window_size=3,
    )
    expected = torch.cat(
        [
            _reference_repetition_penalty(
                logits[row : row + 1],
                history,
                penalty=1.2,
                window_size=3,
            )
            for row, history in enumerate(histories)
        ],
        dim=0,
    )

    assert torch.equal(actual, expected)


def test_batched_repetition_penalty_matches_rows_across_chunks(mocker) -> None:
    batch_size = 2 * _REPETITION_PENALTY_CHUNK_SIZE + 1
    vocab_size = 11
    logits = torch.arange(batch_size * vocab_size, dtype=torch.float32).reshape(batch_size, vocab_size) - 100
    histories = [
        torch.tensor([(row + offset) % vocab_size for offset in range(row % 7)], dtype=torch.long)
        for row in range(batch_size)
    ]

    expected = torch.cat(
        [
            _reference_repetition_penalty(
                logits[row : row + 1],
                history,
                penalty=1.2,
                window_size=5,
            )
            for row, history in enumerate(histories)
        ],
        dim=0,
    )
    bincount = mocker.spy(torch, "bincount")
    actual = _apply_batched_repetition_penalty(
        logits,
        histories,
        penalty=1.2,
        window_size=5,
    )

    assert torch.equal(actual, expected)
    assert [call.kwargs["minlength"] for call in bincount.call_args_list] == [
        _REPETITION_PENALTY_CHUNK_SIZE * vocab_size,
        _REPETITION_PENALTY_CHUNK_SIZE * vocab_size,
        vocab_size,
    ]


def test_scheduler_prompt_is_fully_blanked_for_penalties() -> None:
    # The handoff path fills the prompt with 0s and the non-handoff path passes
    # thinker token ids; neither is codec history, so every position is padded.
    prompt = torch.tensor([[0, 0, 0], [12, 0, 340]])

    blanked = blank_scheduler_prompt_for_penalties(prompt, vocab_size=8)

    assert blanked.tolist() == [[8, 8, 8], [8, 8, 8]]
    assert prompt.tolist() == [[0, 0, 0], [12, 0, 340]]


def test_blanked_prompt_leaves_every_codec_logit_unpenalized() -> None:
    from vllm.model_executor.layers.utils import apply_penalties

    vocab_size = 8
    logits = torch.arange(1.0, vocab_size + 1.0).unsqueeze(0)
    prompt = blank_scheduler_prompt_for_penalties(torch.tensor([[0, 0, 3]]), vocab_size)

    penalized = apply_penalties(
        logits.clone(),
        prompt,
        torch.full((1, 1), vocab_size, dtype=torch.long),
        torch.zeros(1),
        torch.zeros(1),
        torch.full((1,), 1.02),
    )

    torch.testing.assert_close(penalized, logits)


def test_talker_sample_blanks_penalty_prompt_without_touching_runner_state(mocker) -> None:
    talker = _make_talker()
    metadata = _PenaltyMetadata(prompt_token_ids=torch.tensor([[0, 2, 0]]))
    captured: dict[str, torch.Tensor] = {}

    class _Recorder:
        def __call__(self, logits, sampling_metadata):
            prompt = sampling_metadata.prompt_token_ids
            assert prompt is not None
            captured["prompt"] = prompt
            return None

    mocker.patch(
        "vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts.Sampler",
        return_value=_Recorder(),
    )
    talker.sample(torch.zeros(1, 8), metadata)

    assert captured["prompt"].tolist() == [[8, 8, 8]]
    assert metadata.prompt_token_ids is not None
    assert metadata.prompt_token_ids.tolist() == [[0, 2, 0]]


def test_compute_logits_returns_codec_vocab() -> None:
    talker = _make_talker()
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    logits = talker.compute_logits(hidden)

    assert logits.shape == (2, 8)
    assert torch.allclose(logits[0], talker.head_code[0](hidden[0]))


def test_compute_logits_forces_eos_for_empty_speech() -> None:
    talker = _make_talker()
    talker._request_audio_states["req-empty"] = {"finished": True}

    output = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[{"request_id": "req-empty", "codes": {"audio": torch.empty(0)}}],
        request_token_spans=[(0, 1)],
    )
    logits = talker.compute_logits(output.text_hidden_states)

    assert output.multimodal_outputs["codes"]["audio"][0].numel() == 0
    assert output.multimodal_outputs["meta"]["finished"][0].item() is True
    assert int(logits.argmax(dim=-1)) == 7


def _capture_sampler_logits(mocker) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def _record(logits, sampling_metadata):
        captured["logits"] = logits.clone()
        captured["metadata"] = sampling_metadata
        return _SamplerOutput(sampled_token_ids=logits.argmax(dim=-1, keepdim=True))

    mocker.patch(
        "vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts.Sampler",
        return_value=_record,
    )
    return captured


def test_sample_scores_the_upstream_windowed_codec_penalty(mocker) -> None:
    talker = _make_talker()
    talker._request_audio_states["req-pen"] = {"finished": False, "step": 0}
    # ``head_code`` is eye(8, 2) here, so only codes 0 and 1 carry a non-zero
    # logit and are the only ones a penalty can visibly move.
    hidden = torch.tensor([[1.0, -2.0]])
    raw = talker.head_code[0](hidden).float()

    output = talker.make_omni_output(
        hidden,
        model_intermediate_buffer=[{"request_id": "req-pen", "codes": {"audio": torch.tensor([[1]])}}],
        request_token_spans=[(0, 1)],
    )
    logits = talker.compute_logits(output.text_hidden_states)
    captured = _capture_sampler_logits(mocker)
    talker.sample(logits, _CodecSamplingMetadata(repetition_penalties=torch.tensor([1.05])))

    expected = _reference_repetition_penalty(raw, torch.tensor([1]), penalty=1.05, window_size=_CODEC_PENALTY_WINDOW)
    torch.testing.assert_close(captured["logits"], expected)
    # A negative logit is pushed further down, so the row really did change.
    assert captured["logits"][0, 1].item() < raw[0, 1].item()
    # Scored once: the Sampler's own whole-stream pass is neutralized.
    assert captured["metadata"].repetition_penalties.tolist() == [1.0]
    # Consumed once, so a later step cannot rescore a stale row.
    assert talker._penalty_histories is None


def test_codec_penalty_forgets_codes_older_than_the_upstream_window(mocker) -> None:
    # vLLM's sampler penalty spans the whole stream, so every code ever sampled
    # stays taxed and a long tail drifts toward never-sampled codes. Upstream
    # scores only the last ``past_window`` frames.
    talker = _make_talker()
    talker._request_audio_states["req-window"] = {"finished": False, "step": 0}
    hidden = torch.tensor([[1.0, -2.0]])
    raw = talker.head_code[0](hidden).float()
    captured = _capture_sampler_logits(mocker)

    def emit(code: int) -> torch.Tensor:
        output = talker.make_omni_output(
            hidden,
            model_intermediate_buffer=[{"request_id": "req-window", "codes": {"audio": torch.tensor([[code]])}}],
            request_token_spans=[(0, 1)],
        )
        logits = talker.compute_logits(output.text_hidden_states)
        talker.sample(logits, _CodecSamplingMetadata(repetition_penalties=torch.tensor([1.05])))
        return captured["logits"]

    assert emit(1)[0, 1].item() < raw[0, 1].item()
    # Push code 1 out of the window with a full window of code 0.
    for _ in range(_CODEC_PENALTY_WINDOW):
        scored = emit(0)

    assert talker._request_audio_states["req-window"]["recent_codes"] == [0] * _CODEC_PENALTY_WINDOW
    torch.testing.assert_close(scored[0, 1], raw[0, 1])


def test_sample_honours_a_request_that_disabled_penalties(mocker) -> None:
    talker = _make_talker()
    talker._request_audio_states["req-none"] = {"finished": False, "step": 0}
    hidden = torch.tensor([[1.0, -2.0]])
    raw = talker.head_code[0](hidden).float()

    output = talker.make_omni_output(
        hidden,
        model_intermediate_buffer=[{"request_id": "req-none", "codes": {"audio": torch.tensor([[1]])}}],
        request_token_spans=[(0, 1)],
    )
    logits = talker.compute_logits(output.text_hidden_states)
    captured = _capture_sampler_logits(mocker)
    talker.sample(logits, _CodecSamplingMetadata(repetition_penalties=torch.tensor([1.0]), no_penalties=True))

    torch.testing.assert_close(captured["logits"], raw)


def test_sample_restores_forced_eos_that_min_tokens_masked_away(mocker) -> None:
    # ``min_tokens`` masks every stage stop_token_id, including the codec EOS
    # compute_logits just forced, so the sampler returns an arbitrary id for a
    # request the model already finished.
    talker = _make_talker()
    talker._request_audio_states["req-empty"] = {"finished": True}

    output = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[{"request_id": "req-empty", "codes": {"audio": torch.empty(0)}}],
        request_token_spans=[(0, 1)],
    )
    logits = talker.compute_logits(output.text_hidden_states)
    logits[0, talker._codec_eos_id] = float("-inf")

    class _MinTokensSampler:
        def __call__(self, logits, sampling_metadata):
            del sampling_metadata
            return _SamplerOutput(sampled_token_ids=logits.argmax(dim=-1, keepdim=True))

    mocker.patch(
        "vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts.Sampler",
        return_value=_MinTokensSampler(),
    )
    sampled = talker.sample(logits, _PenaltyMetadata(prompt_token_ids=None))

    assert sampled.sampled_token_ids.tolist() == [[talker._codec_eos_id]]


def test_sample_leaves_unforced_rows_to_the_sampler(mocker) -> None:
    talker = _make_talker()
    talker._request_audio_states["req-a"] = {"finished": True}
    talker._request_audio_states["req-b"] = {"finished": False, "step": 0}

    output = talker.make_omni_output(
        torch.ones(2, 2),
        model_intermediate_buffer=[
            {"request_id": "req-a", "codes": {"audio": torch.empty(0)}},
            {"request_id": "req-b", "codes": {"audio": torch.tensor([[3]])}},
        ],
        request_token_spans=[(0, 1), (1, 2)],
    )
    talker.compute_logits(output.text_hidden_states)

    mocker.patch(
        "vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts.Sampler",
        return_value=lambda logits, sampling_metadata: _SamplerOutput(sampled_token_ids=torch.tensor([[0], [4]])),
    )
    sampled = talker.sample(torch.zeros(2, 8), _PenaltyMetadata(prompt_token_ids=None))

    assert sampled.sampled_token_ids.tolist() == [[talker._codec_eos_id], [4]]
    # A single compute_logits decision must not leak into the next sample().
    assert talker._pending_force_eos_rows is None


def test_native_duplex_chunk_budget_masks_eos_until_generate_chunk_is_full() -> None:
    assert _native_duplex_chunk_budget({}) == (_DUPLEX_CODEC_TOKENS_PER_CHUNK, _DUPLEX_CODEC_TOKENS_PER_CHUNK)
    assert _native_duplex_chunk_budget({"turn_start": True}) == (_DUPLEX_CODEC_TOKENS_PER_CHUNK, 0)
    assert _native_duplex_chunk_budget({"turn_end": True}) == (_DUPLEX_CODEC_TOKENS_PER_CHUNK, 0)

    talker = _make_talker()
    talker._request_audio_states["req-chunk"] = {
        "finished": False,
        "step": 0,
        "max_tokens": _DUPLEX_CODEC_TOKENS_PER_CHUNK,
        "min_tokens": _DUPLEX_CODEC_TOKENS_PER_CHUNK,
    }

    output = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[{"request_id": "req-chunk", "codes": {"audio": torch.empty(0)}}],
        request_token_spans=[(0, 1)],
    )
    logits = talker.compute_logits(output.text_hidden_states)

    assert logits[0, 7].item() == float("-inf")
    assert torch.isfinite(logits[0, 0])
    assert output.multimodal_outputs["meta"]["finished"][0].item() is False


def test_native_duplex_chunk_budget_forces_eos_after_twenty_five_frames() -> None:
    talker = _make_talker()
    talker._request_audio_states["req-full"] = {
        "finished": False,
        "step": 24,
        "max_tokens": _DUPLEX_CODEC_TOKENS_PER_CHUNK,
        "min_tokens": _DUPLEX_CODEC_TOKENS_PER_CHUNK,
    }

    output = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[{"request_id": "req-full", "codes": {"audio": torch.tensor([[3]])}}],
        request_token_spans=[(0, 1)],
    )
    logits = talker.compute_logits(output.text_hidden_states)

    assert talker._request_audio_states["req-full"]["step"] == 25
    assert talker._request_audio_states["req-full"]["finished"] is True
    assert int(logits.argmax(dim=-1)) == 7
    assert output.multimodal_outputs["meta"]["finished"][0].item() is True


def test_offline_prefill_caps_codec_budget_to_the_remaining_talker_context() -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(1, 2)
    talker.emb_code = nn.ModuleList([nn.Embedding(8, 2)])
    talker._text_eos_id = 0
    talker._tts_bos_id = 0
    # Long enough that the context, not upstream's max_new_token, is the binding
    # limit; the Talker must never plan past its own position embeddings.
    prompt_len = int(talker._tts_config.max_position_embeddings) - _OFFLINE_CODEC_MAX_NEW_TOKENS + 100

    _, _, updates = talker.preprocess(
        torch.zeros(2, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        _omni_prompt_len=prompt_len,
        request_id="req-offline-cap",
        tts_token_ids=torch.empty(0, dtype=torch.long),
        tts_hidden_states=torch.empty(0, 2),
    )

    remaining = int(talker._tts_config.max_position_embeddings) - prompt_len
    assert remaining < _OFFLINE_CODEC_MAX_NEW_TOKENS
    assert updates["audio_state"]["max_tokens"] == remaining
    assert updates["audio_state"]["min_tokens"] is None


def test_talker_masks_eos_until_recorded_min_tokens() -> None:
    talker = _make_talker()
    talker._request_audio_states["req-early"] = {
        "finished": False,
        "step": 0,
        "max_tokens": 8,
        "min_tokens": 4,
    }

    output = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[{"request_id": "req-early", "codes": {"audio": torch.tensor([[3]])}}],
        request_token_spans=[(0, 1)],
    )
    logits = talker.compute_logits(output.text_hidden_states)

    assert talker._request_audio_states["req-early"]["step"] == 1
    assert logits[0, 7].item() == float("-inf")
    assert torch.isfinite(logits[0, 0])
    assert output.multimodal_outputs["meta"]["finished"][0].item() is False


def test_offline_talker_forces_eos_at_max_new_token() -> None:
    talker = _make_talker()
    max_tokens = 8
    talker._request_audio_states["req-cap"] = {
        "finished": False,
        "step": max_tokens - 2,
        "max_tokens": max_tokens,
    }

    output = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[{"request_id": "req-cap", "codes": {"audio": torch.tensor([[3]])}}],
        request_token_spans=[(0, 1)],
    )
    logits = talker.compute_logits(output.text_hidden_states)

    assert talker._request_audio_states["req-cap"]["step"] == max_tokens - 1
    assert talker._request_audio_states["req-cap"]["finished"] is True
    assert int(logits.argmax(dim=-1)) == 7
    assert output.multimodal_outputs["meta"]["finished"][0].item() is True


def test_talker_without_a_recorded_budget_neither_forces_nor_masks_eos() -> None:
    # A state rebuilt from a transported ``audio_state`` can lack budget keys;
    # such rows must fall through to the raw codec head.
    talker = _make_talker()
    talker._request_audio_states["req-nobudget"] = {"finished": False, "step": 0}

    output = talker.make_omni_output(
        torch.tensor([[1.0, 0.0]]),
        model_intermediate_buffer=[{"request_id": "req-nobudget", "codes": {"audio": torch.tensor([[3]])}}],
        request_token_spans=[(0, 1)],
    )
    logits = talker.compute_logits(output.text_hidden_states)

    assert talker._request_audio_states["req-nobudget"]["step"] == 1
    assert torch.allclose(logits[0], talker.head_code[0](output.text_hidden_states[0]).float())


def test_offline_prefill_records_upstream_max_new_token_budget() -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(1, 2)
    talker.emb_code = nn.ModuleList([nn.Embedding(8, 2)])
    talker._text_eos_id = 0
    talker._tts_bos_id = 0

    _, _, updates = talker.preprocess(
        torch.zeros(2, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        request_id="req-offline",
        tts_token_ids=torch.empty(0, dtype=torch.long),
        tts_hidden_states=torch.empty(0, 2),
    )

    # A 2-token boundary prompt leaves far more context than upstream's
    # max_new_token, so MiniCPMTTS.generate's own ceiling is what binds.
    assert int(talker._tts_config.max_position_embeddings) - 2 > _OFFLINE_CODEC_MAX_NEW_TOKENS
    assert updates["audio_state"]["max_tokens"] == _OFFLINE_CODEC_MAX_NEW_TOKENS
    assert updates["audio_state"]["min_tokens"] is None


def test_native_duplex_prefill_records_generate_chunk_budget() -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(1, 2)
    talker.emb_code = nn.ModuleList([nn.Embedding(8, 2)])
    talker._text_eos_id = 0
    talker._tts_bos_id = 0

    _, _, mid_turn = talker.preprocess(
        torch.zeros(2, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        request_id="req-mid",
        native_duplex=True,
        tts_token_ids=torch.empty(0, dtype=torch.long),
        tts_hidden_states=torch.empty(0, 2),
    )
    _, _, boundary = talker.preprocess(
        torch.zeros(2, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        request_id="req-end",
        native_duplex=True,
        meta={"turn_end": True},
        tts_token_ids=torch.empty(0, dtype=torch.long),
        tts_hidden_states=torch.empty(0, 2),
    )

    assert mid_turn["audio_state"]["max_tokens"] == _DUPLEX_CODEC_TOKENS_PER_CHUNK
    assert mid_turn["audio_state"]["min_tokens"] == _DUPLEX_CODEC_TOKENS_PER_CHUNK
    assert boundary["audio_state"]["min_tokens"] == 0


def test_make_omni_output_packs_the_previous_codec_id() -> None:
    talker = _make_talker()
    infos = [
        {"request_id": "req-a", "codes": {"audio": torch.tensor([[2]])}},
        {"request_id": "req-b", "codes": {"audio": torch.tensor([[3]])}},
    ]

    output = talker.make_omni_output(
        torch.ones(2, 2),
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 1), (1, 2)],
    )

    assert _routed(output, 0)["codes"]["audio"].tolist() == [[2]]
    assert _routed(output, 1)["codes"]["audio"].tolist() == [[3]]
    assert _routed(output, 0)["meta"]["finished"].item() is False


def test_talker_projects_request_aligned_duplex_metadata() -> None:
    talker = _make_talker()
    infos = [
        {
            "request_id": "req-a",
            "native_duplex": True,
            "duplex": {"epoch": 3, "turn_id": 7},
            "ids": {"tts": [41]},
            "meta": {"native_duplex_segment_text": "first", "turn_eos_token_id": 99},
            "codes": {"audio": torch.empty(0)},
        },
        {
            "request_id": "req-b",
            "native_duplex": True,
            "duplex": {"epoch": 4, "turn_id": 8},
            "ids": {"tts": [42, 99]},
            "meta": {"native_duplex_segment_text": "second", "turn_eos_token_id": 99},
            "codes": {"audio": torch.empty(0)},
        },
    ]

    output = talker.make_omni_output(
        torch.ones(2, 2),
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 1), (1, 2)],
    )

    meta = output.multimodal_outputs["meta"]
    assert [value.item() for value in meta["native_duplex"]] == [True, True]
    assert [value.item() for value in meta["duplex_epoch"]] == [3, 4]
    assert [value.item() for value in meta["duplex_turn_id"]] == [7, 8]
    assert [bytes(value.tolist()).decode("utf-8") for value in meta["llm_output_text_utf8"]] == [
        "first",
        "second",
    ]
    assert [value.item() for value in meta["turn_end"]] == [False, True]


def test_talker_rejects_native_duplex_without_fence_identity() -> None:
    talker = _make_talker()
    info = {"request_id": "req-missing-fence", "native_duplex": True, "codes": {"audio": torch.empty(0)}}

    with pytest.raises(RuntimeError, match="requires non-negative integer epoch and turn_id"):
        talker.make_omni_output(
            torch.ones(1, 2),
            model_intermediate_buffer=[info],
            request_token_spans=[(0, 1)],
        )


def test_decode_preprocess_embeds_the_sampled_codec_id() -> None:
    talker = _make_talker()
    talker.emb_code = nn.ModuleList([nn.Embedding(8, 4)])
    with torch.no_grad():
        talker.emb_code[0].weight.copy_(torch.arange(32, dtype=torch.float32).reshape(8, 4))

    _, embeds, updates = talker.preprocess(
        torch.tensor([3]),
        None,
        request_id="req-decode",
        audio_state={"finished": False},
    )

    assert torch.equal(embeds, talker.emb_code[0](torch.tensor([3])))
    assert updates["codes"]["audio"].tolist() == [[3]]


def test_decode_preprocess_drops_codec_eos() -> None:
    talker = _make_talker()
    talker.emb_code = nn.ModuleList([nn.Embedding(8, 4)])

    _, _, updates = talker.preprocess(
        torch.tensor([7]),
        None,
        request_id="req-eos",
        audio_state={"finished": False},
    )

    assert updates["codes"]["audio"].numel() == 0


def test_missing_conditioning_fails_clearly() -> None:
    talker = _make_talker()

    with pytest.raises(ValueError, match="tts_token_ids and tts_hidden_states"):
        talker.preprocess(
            torch.tensor([0]),
            None,
            _omni_is_prefill=True,
            request_id="req-invalid",
        )


def test_empty_speech_segment_finishes_without_sampling_codes() -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(8, 4)
    talker.emb_code = nn.ModuleList([nn.Embedding(8, 4)])
    talker._text_eos_id = 5
    talker._tts_bos_id = 6

    _, embeds, updates = talker.preprocess(
        torch.zeros(2, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        request_id="req-empty",
        tts_token_ids=torch.empty(0, dtype=torch.long),
        tts_hidden_states=torch.empty(0, 4),
    )

    assert torch.equal(embeds, talker.emb_text(torch.tensor([5, 6])))
    assert updates["audio_state"]["finished"] is True
    assert updates["codes"]["audio"].numel() == 0

    # min_tokens can keep scheduling leftover decode rows after an empty
    # speech segment. Those rows have no previous codec id to embed.
    _, decode_embeds, decode_updates = talker.preprocess(
        torch.zeros(1, dtype=torch.long),
        None,
        request_id="req-empty",
        audio_state=updates["audio_state"],
    )

    assert decode_embeds.shape == (1, 4)
    assert decode_updates["codes"]["audio"].numel() == 0


def test_chunked_prefill_tail_aligns_condition_with_prompt_length(mocker) -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(1, 2)
    condition = torch.arange(18, dtype=torch.float32).reshape(9, 2)
    mocker.patch.object(talker, "_build_condition_embeddings", return_value=condition)

    _, embeds, updates = talker.preprocess(
        torch.zeros(9, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        _omni_num_computed_tokens=59,
        _omni_prompt_len=68,
        request_id="req-chunked-prefill",
        tts_token_ids=torch.tensor([1]),
        tts_hidden_states=torch.ones(1, 2),
    )

    assert torch.equal(embeds, condition)
    assert updates["codes"]["audio"].numel() == 0
    state = talker._request_audio_states["req-chunked-prefill"]
    assert state["finished"] is False


def test_native_duplex_condition_matches_official_text_plus_audio_bos() -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(16, 2)
    talker.projector_semantic = nn.Identity()
    talker._normalize = False
    talker._text_eos_id = 14
    talker._tts_bos_id = 15
    with torch.no_grad():
        talker.emb_text.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(16, 2))

    token_ids = torch.tensor([2, 3])
    hidden_states = torch.tensor([[0.5, 1.0], [1.5, 2.0]])

    condition = talker._build_condition_embeddings(
        token_ids,
        hidden_states,
        native_duplex=True,
    )

    expected_text = talker.emb_text(token_ids) + hidden_states
    expected = torch.cat(
        [expected_text, talker.emb_text(torch.tensor([talker._tts_bos_id]))],
        dim=0,
    )
    assert torch.equal(condition, expected)
    assert condition.shape[0] == token_ids.shape[0] + 1


def test_native_duplex_rollover_matches_official_sliding_recompute(mocker) -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(1, 2)
    talker.emb_code = nn.ModuleList([nn.Embedding(8, 2)])
    with torch.no_grad():
        talker.emb_code[0].weight.copy_(torch.arange(16, dtype=torch.float32).reshape(8, 2))

    previous_condition = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    current_condition = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    build_condition = mocker.patch.object(
        talker,
        "_build_condition_embeddings",
        side_effect=[previous_condition, current_condition, current_condition],
    )

    talker.preprocess(
        torch.zeros(2, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        _omni_prompt_len=2,
        request_id="req-window",
        native_duplex=True,
        meta={"turn_start": True, "streaming_condition_seq": 7},
        tts_token_ids=torch.tensor([1]),
        tts_hidden_states=torch.ones(1, 2),
    )

    older_codes = [3, 4, 5]
    talker._request_condition_states["req-window"]["base_recent_codes"] = tuple(older_codes)
    previous_codes = [1, 2]
    recompute_meta = {
        "streaming_condition_seq": 8,
        "streaming_prompt_recompute": True,
    }
    expected = torch.cat(
        [
            previous_condition,
            talker.emb_code[0](torch.tensor(previous_codes)),
            current_condition,
        ],
        dim=0,
    )
    _, full_embeds, _ = talker.preprocess(
        torch.zeros(expected.shape[0], dtype=torch.long),
        None,
        _omni_is_prefill=True,
        _omni_prompt_len=expected.shape[0],
        request_id="req-window",
        native_duplex=True,
        ids={"streaming_prompt_previous_codes": previous_codes},
        meta=recompute_meta,
        tts_token_ids=torch.tensor([2]),
        tts_hidden_states=torch.ones(1, 2),
    )

    assert torch.equal(full_embeds, expected)
    assert talker._request_audio_states["req-window"]["recent_codes"] == older_codes + previous_codes

    # Chunked-prefill/preemption retries reuse the immutable recipe for the
    # same condition instead of advancing the one-chunk window twice.
    _, retry_slice, _ = talker.preprocess(
        torch.zeros(3, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        _omni_num_computed_tokens=2,
        _omni_prompt_len=expected.shape[0],
        request_id="req-window",
        native_duplex=True,
        ids={"streaming_prompt_previous_codes": previous_codes},
        meta=recompute_meta,
        tts_token_ids=torch.tensor([2]),
        tts_hidden_states=torch.ones(1, 2),
    )
    assert torch.equal(retry_slice, expected[2:5])
    assert talker._request_audio_states["req-window"]["recent_codes"] == older_codes + previous_codes
    assert build_condition.call_count == 3


def test_native_duplex_condition_advance_requires_recompute_marker(mocker) -> None:
    talker = _make_talker()
    condition = torch.ones(2, 2)
    mocker.patch.object(talker, "_build_condition_embeddings", return_value=condition)
    common = {
        "_omni_is_prefill": True,
        "_omni_prompt_len": 2,
        "request_id": "req-history",
        "native_duplex": True,
        "tts_token_ids": torch.tensor([1]),
        "tts_hidden_states": torch.ones(1, 2),
    }

    talker.preprocess(torch.zeros(2, dtype=torch.long), None, meta={"streaming_condition_seq": 0}, **common)
    with pytest.raises(ValueError, match="advanced without its streaming recompute marker"):
        talker.preprocess(torch.zeros(2, dtype=torch.long), None, meta={"streaming_condition_seq": 1}, **common)


@pytest.mark.parametrize(
    ("state", "previous_codes", "error"),
    [
        (None, [1], "missing the previous Talker condition"),
        (
            {"condition_seq": 0, "condition": torch.ones(2, 2), "base_recent_codes": ()},
            [7],
            "invalid or terminal token",
        ),
    ],
)
def test_native_duplex_rollover_rejects_unverifiable_history(state, previous_codes, error) -> None:
    talker = _make_talker()
    talker.emb_code = nn.ModuleList([nn.Embedding(8, 2)])
    if state is not None:
        talker._request_condition_states["req-invalid-window"] = state

    with pytest.raises(ValueError, match=error):
        talker._build_streaming_recompute_embeddings(
            torch.ones(3, 2),
            request_id="req-invalid-window",
            info_dict={"ids": {"streaming_prompt_previous_codes": previous_codes}},
            meta={
                "streaming_condition_seq": 1,
                "streaming_prompt_recompute": True,
            },
        )


def test_native_duplex_turn_start_resets_rollover_history() -> None:
    talker = _make_talker()
    talker._request_condition_states["req-turn"] = {
        "condition_seq": 4,
        "condition": torch.full((2, 2), 9.0),
        "active_embeddings": torch.full((5, 2), 9.0),
    }
    current = torch.ones(3, 2)

    result = talker._build_streaming_recompute_embeddings(
        current,
        request_id="req-turn",
        info_dict={},
        meta={"turn_start": True, "streaming_condition_seq": 5},
    )

    assert torch.equal(result, current)
    state = talker._request_condition_states["req-turn"]
    assert state["condition_seq"] == 5
    assert torch.equal(state["condition"], current)
    assert "active_embeddings" not in state


def test_request_cleanup_evicts_decode_state() -> None:
    talker = _make_talker()
    talker._request_audio_states["req-done"] = {"finished": False}
    talker._request_condition_states["req-done"] = {
        "condition_seq": 1,
        "condition": torch.ones(2, 2),
    }

    talker.on_requests_finished(["req-done"])
    talker._flush_deferred_cleanup()

    assert "req-done" not in talker._request_audio_states
    assert "req-done" not in talker._request_condition_states
