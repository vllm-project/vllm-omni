# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-alignment tests for MiniCPM-o 4.5's native Talker."""

from __future__ import annotations

from dataclasses import dataclass

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
    _DUPLEX_CODEC_TOKENS_PER_CHUNK,
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


class _FakeNativeTalker(nn.Module):
    has_preprocess = True

    def __init__(self) -> None:
        super().__init__()
        self.forward_kwargs = None

    def forward(self, **kwargs):
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
    model.talker = _FakeNativeTalker()

    output = model(
        input_ids=torch.tensor([1, 2]),
        positions=torch.arange(2),
        model_intermediate_buffer=[{"request_id": "req"}],
    )

    assert output.shape == (2, 4)
    assert model.talker.forward_kwargs["model_intermediate_buffer"][0]["request_id"] == "req"


def _make_talker() -> MiniCPMO45OmniTTSForConditionalGeneration:
    talker = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    nn.Module.__init__(talker)
    talker._num_audio_tokens = 8
    talker._codec_eos_id = 7
    talker._force_eos_rows = None
    talker._mask_eos_rows = None
    talker._request_audio_states = {}
    talker._deferred_cleanup_ids = set()
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
            captured["prompt"] = sampling_metadata.prompt_token_ids
            return None

    mocker.patch(
        "vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts.Sampler",
        return_value=_Recorder(),
    )
    talker.sample(torch.zeros(1, 8), metadata)

    assert captured["prompt"].tolist() == [[8, 8, 8]]
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


def test_simplex_talker_does_not_force_eos_after_a_duplex_sized_chunk() -> None:
    talker = _make_talker()
    talker._request_audio_states["req-simplex"] = {"finished": False, "step": 0}

    output = talker.make_omni_output(
        torch.tensor([[1.0, 0.0]]),
        model_intermediate_buffer=[{"request_id": "req-simplex", "codes": {"audio": torch.tensor([[3]])}}],
        request_token_spans=[(0, 1)],
    )
    logits = talker.compute_logits(output.text_hidden_states)

    assert talker._request_audio_states["req-simplex"]["step"] == 0
    assert torch.allclose(logits[0], talker.head_code[0](output.text_hidden_states[0]).float())


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


def test_request_cleanup_evicts_decode_state() -> None:
    talker = _make_talker()
    talker._request_audio_states["req-done"] = {"finished": False}

    talker.on_requests_finished(["req-done"])
    talker._flush_deferred_cleanup()

    assert "req-done" not in talker._request_audio_states
