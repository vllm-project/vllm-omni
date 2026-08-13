# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-alignment tests for MiniCPM-o 4.5's native Talker."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    _REPETITION_PENALTY_CHUNK_SIZE,
    MiniCPMO45OmniTTSForConditionalGeneration,
    _apply_batched_repetition_penalty,
    _max_audio_tokens,
    _restore_weight_norm_weight,
)
from vllm_omni.utils.mm_outputs import to_payload_element

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeNativeTalker(nn.Module):
    has_preprocess = True

    def __init__(self) -> None:
        super().__init__()
        self.forward_kwargs = None

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        return torch.ones(2, 4)


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
    talker._batch_stop_logits = None
    talker._request_generators = {}
    talker._request_audio_states = {}
    talker._deferred_cleanup_ids = set()
    talker._codec_min_tokens = 50
    talker._codec_seed = 42
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


def _make_sampling_talker() -> MiniCPMO45OmniTTSForConditionalGeneration:
    talker = _make_talker()
    talker._codec_temperature = 0.8
    talker._codec_top_k = 5
    talker._codec_top_p = 0.85
    talker._codec_repetition_penalty = 1.05
    talker._request_audio_states = {
        "req-a": {"min_tokens": 2},
        "req-b": {"min_tokens": 0},
    }
    talker.head_code = nn.ModuleList([nn.Linear(2, 8, bias=False)])
    with torch.no_grad():
        talker.head_code[0].weight.copy_(
            torch.tensor(
                [
                    [0.1, -0.3],
                    [0.2, 0.4],
                    [-0.5, 0.2],
                    [0.7, -0.1],
                    [-0.2, 0.8],
                    [0.6, 0.3],
                    [-0.4, -0.7],
                    [0.3, 0.5],
                ]
            )
        )
    return talker


@pytest.mark.parametrize(
    ("condition_tokens", "expected"),
    [(3, 64), (100, 1000), (1000, 2048)],
)
def test_audio_token_limit_scales_with_condition_length(
    condition_tokens: int,
    expected: int,
) -> None:
    assert _max_audio_tokens(condition_tokens) == expected


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


def test_batched_codec_sampling_is_request_order_independent() -> None:
    talker = _make_sampling_talker()
    histories = [torch.tensor([1, 1, 2]), torch.tensor([3, 4, 4])]
    hidden = torch.tensor([[0.25, -0.75], [-0.5, 0.5]])

    forward_order = talker._sample_audio_codes(
        hidden,
        histories,
        ["req-a", "req-b"],
        [0, 0],
    )
    talker._request_generators = {}
    reverse_order = talker._sample_audio_codes(
        hidden.flip(0),
        list(reversed(histories)),
        ["req-b", "req-a"],
        [0, 0],
    )

    assert torch.equal(forward_order[0], reverse_order[1])
    assert torch.equal(forward_order[1], reverse_order[0])


def test_batched_codec_sampling_preserves_request_rng_after_compaction() -> None:
    talker = _make_sampling_talker()
    history_a = torch.tensor([1, 1, 2])
    history_b = torch.tensor([3, 4, 4])
    hidden_a = torch.tensor([[0.25, -0.75]])
    hidden_b = torch.tensor([[-0.5, 0.5]])

    first_batch = talker._sample_audio_codes(
        torch.cat([hidden_a, hidden_b]),
        [history_a, history_b],
        ["req-a", "req-b"],
        [0, 0],
    )
    compacted_b = talker._sample_audio_codes(
        hidden_b,
        [history_b],
        ["req-b"],
        [1],
    )

    talker._request_generators = {}
    standalone_b_first = talker._sample_audio_codes(
        hidden_b,
        [history_b],
        ["req-b"],
        [0],
    )[0]
    standalone_b_second = talker._sample_audio_codes(
        hidden_b,
        [history_b],
        ["req-b"],
        [1],
    )[0]

    assert torch.equal(first_batch[1], standalone_b_first)
    assert torch.equal(compacted_b[0], standalone_b_second)


def test_talker_emits_request_aligned_codec_deltas_after_compaction(mocker) -> None:
    talker = _make_talker()
    seen: list[tuple[str, list[float], list[int]]] = []

    def sample(hidden, histories, request_ids, steps):
        assert steps == [0, 0]
        seen.extend(
            (request_id, hidden[row].reshape(-1).tolist(), histories[row].tolist())
            for row, request_id in enumerate(request_ids)
        )
        return torch.tensor([2, 3])

    batched_sample = mocker.patch.object(talker, "_sample_audio_codes", side_effect=sample)
    infos = [
        {"request_id": "req-a", "audio_codes": {"accumulated": torch.tensor([1])}},
        {"request_id": "req-b", "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)}},
    ]

    output = talker.make_omni_output(
        torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 2), (2, 3)],
    )

    assert seen == [
        ("req-a", [2.0, 0.0], [1]),
        ("req-b", [3.0, 0.0], []),
    ]
    batched_sample.assert_called_once()
    assert infos[0]["audio_codes"]["accumulated"].tolist() == [1, 2]
    assert infos[1]["audio_codes"]["accumulated"].tolist() == [3]
    assert set(output.multimodal_outputs) == {"codes", "meta"}
    assert "model_outputs" not in output.multimodal_outputs
    assert "sr" not in output.multimodal_outputs
    assert _routed(output, 0)["codes"]["audio"].tolist() == [[2]]
    assert _routed(output, 1)["codes"]["audio"].tolist() == [[3]]
    assert _routed(output, 0)["meta"]["finished"].item() is False
    assert set(output.multimodal_outputs["meta"]) == {"finished"}
    assert talker.compute_logits(output.text_hidden_states).argmax(dim=-1).tolist() == [0, 0]


def test_talker_projects_request_aligned_duplex_metadata(mocker) -> None:
    talker = _make_talker()
    mocker.patch.object(talker, "_sample_audio_codes", return_value=torch.tensor([2, 2]))
    infos = [
        {
            "request_id": "req-a",
            "native_duplex": True,
            "duplex": {"epoch": 3, "turn_id": 7},
            "ids": {"tts": [41]},
            "meta": {
                "native_duplex_segment_text": "first",
                "turn_eos_token_id": 99,
            },
            "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
        },
        {
            "request_id": "req-b",
            "native_duplex": True,
            "duplex": {"epoch": 4, "turn_id": 8},
            "ids": {"tts": [42, 99]},
            "meta": {
                "native_duplex_segment_text": "second",
                "turn_eos_token_id": 99,
            },
            "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
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
    assert "native_duplex_segment_text" not in meta
    assert [bytes(value.tolist()).decode("utf-8") for value in meta["llm_output_text_utf8"]] == [
        "first",
        "second",
    ]
    assert [value.item() for value in meta["turn_end"]] == [False, True]


def test_talker_rejects_native_duplex_without_fence_identity(mocker) -> None:
    talker = _make_talker()
    mocker.patch.object(talker, "_sample_audio_codes", return_value=torch.tensor([2]))
    info = {
        "request_id": "req-missing-fence",
        "native_duplex": True,
        "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
    }

    with pytest.raises(RuntimeError, match="requires non-negative integer epoch and turn_id"):
        talker.make_omni_output(
            torch.ones(1, 2),
            model_intermediate_buffer=[info],
            request_token_spans=[(0, 1)],
        )


def test_incomplete_prefill_emits_no_code_and_does_not_advance_state(mocker) -> None:
    talker = _make_talker()
    sample = mocker.patch.object(talker, "_sample_audio_codes", return_value=torch.tensor([2]))
    infos = [
        {
            "request_id": "req-prefill",
            "audio_state": {"step": 0},
            "audio_codes": {"accumulated": torch.empty(0, dtype=torch.long)},
        },
        {
            "request_id": "req-decode",
            "audio_state": {"step": 4},
            "audio_codes": {"accumulated": torch.tensor([1])},
        },
    ]

    output = talker.make_omni_output(
        torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        model_intermediate_buffer=infos,
        request_token_spans=[(0, 2), (2, 3)],
        request_sample_eligible=[False, True],
    )

    sample.assert_called_once()
    assert sample.call_args.args[2] == ["req-decode"]
    assert infos[0]["audio_state"]["step"] == 0
    assert infos[0]["audio_codes"]["accumulated"].numel() == 0
    assert infos[1]["audio_state"]["step"] == 5
    assert _routed(output, 0)["codes"]["audio"].shape == (0, 1)
    assert _routed(output, 1)["codes"]["audio"].tolist() == [[2]]


def test_eos_is_terminal_once_and_never_enters_codec_history(mocker) -> None:
    talker = _make_talker()
    sample = mocker.patch.object(talker, "_sample_audio_codes", return_value=torch.tensor([7]))
    info = {
        "request_id": "req-stop",
        "audio_state": {"step": 3},
        "audio_codes": {"accumulated": torch.tensor([4, 5])},
    }

    first = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )
    first_logits = talker.compute_logits(first.text_hidden_states)
    second = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )

    sample.assert_called_once()
    assert info["audio_codes"]["accumulated"].tolist() == [4, 5]
    assert first.multimodal_outputs["codes"]["audio"][0].shape == (0, 1)
    assert first.multimodal_outputs["meta"]["finished"][0].item() is True
    assert second.multimodal_outputs["meta"]["finished"][0].item() is False
    assert first_logits.argmax(dim=-1).tolist() == [1]
    assert talker.compute_logits(second.text_hidden_states).argmax(dim=-1).tolist() == [1]


def test_max_token_terminal_drops_unconsumed_codec_delta(mocker) -> None:
    talker = _make_talker()
    mocker.patch.object(talker, "_sample_audio_codes", return_value=torch.tensor([3]))
    info = {
        "request_id": "req-limit",
        "audio_state": {"step": 1, "max_tokens": 2},
        "audio_codes": {"accumulated": torch.tensor([4, 5])},
    }

    output = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[info],
        request_token_spans=[(0, 1)],
    )

    # MiniCPMTTS.generate_chunk samples once at the max-token boundary to
    # advance RNG state, but the sampled code is not fed into KV or returned.
    assert info["audio_codes"]["accumulated"].tolist() == [4, 5]
    assert output.multimodal_outputs["codes"]["audio"][0].shape == (0, 1)
    assert output.multimodal_outputs["meta"]["finished"][0].item() is True
    assert talker.compute_logits(output.text_hidden_states).argmax(dim=-1).tolist() == [1]


def test_request_local_state_survives_missing_runner_buffer_update(mocker) -> None:
    talker = _make_talker()
    mocker.patch.object(talker, "_sample_audio_codes", return_value=torch.tensor([3]))
    first_info = {
        "request_id": "req-local-state",
        "audio_state": {"step": 1, "max_tokens": 3},
        "audio_codes": {"accumulated": torch.tensor([4])},
    }

    talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[first_info],
        request_token_spans=[(0, 1)],
    )
    second = talker.make_omni_output(
        torch.ones(1, 2),
        model_intermediate_buffer=[{"request_id": "req-local-state"}],
        request_token_spans=[(0, 1)],
    )

    assert second.multimodal_outputs["meta"]["finished"][0].item() is True
    assert talker._request_audio_states["req-local-state"]["step"] == 3


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

    # Stage 1's sampling min_tokens keeps scheduling decode steps until the stop
    # token becomes eligible, and those steps have no previous code to embed.
    _, decode_embeds, _ = talker.preprocess(
        torch.zeros(1, dtype=torch.long),
        None,
        request_id="req-empty",
        audio_state=updates["audio_state"],
        audio_codes=updates["audio_codes"],
    )

    assert decode_embeds.shape == (1, 4)


def test_chunked_prefill_tail_aligns_condition_with_prompt_length(mocker) -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(1, 2)
    condition = torch.arange(18, dtype=torch.float32).reshape(9, 2)
    mocker.patch.object(talker, "_build_condition_embeddings", return_value=condition)

    _, embeds, _ = talker.preprocess(
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
    state = talker._request_audio_states["req-chunked-prefill"]
    assert state["min_tokens"] == 50
    assert state["max_tokens"] == 64


@pytest.mark.parametrize(
    ("meta", "expected_min_tokens"),
    [
        ({"turn_start": True}, 0),
        ({}, 26),
        ({"turn_end": True}, 0),
    ],
)
def test_native_duplex_prefill_uses_official_chunk_limits(
    mocker,
    meta,
    expected_min_tokens,
) -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(1, 2)
    mocker.patch.object(
        talker,
        "_build_condition_embeddings",
        return_value=torch.ones(3, 2),
    )

    talker.preprocess(
        torch.zeros(3, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        request_id="req-duplex-chunk",
        native_duplex=True,
        meta=meta,
        tts_token_ids=torch.tensor([1]),
        tts_hidden_states=torch.ones(1, 2),
    )

    state = talker._request_audio_states["req-duplex-chunk"]
    assert state["min_tokens"] == expected_min_tokens
    assert state["max_tokens"] == 26


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


def test_request_cleanup_evicts_ar_rng_and_decode_state() -> None:
    talker = _make_talker()
    talker._request_generators["req-done"] = torch.Generator()
    talker._request_audio_states["req-done"] = {"step": 1}

    talker.on_requests_finished(["req-done"])
    talker._flush_deferred_cleanup()

    assert "req-done" not in talker._request_generators
    assert "req-done" not in talker._request_audio_states
