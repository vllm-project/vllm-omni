# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-alignment tests for MiniCPM-o 4.5's native Talker."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.minicpmo_4_5.handoff import pack_handoff_tensor
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    ConditionalChatTTSConfig,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    _DRAIN_BOUNDARY_MASK,
    _DRAIN_NONE,
    _DRAIN_UNTIL_SILENT,
    _DUPLEX_CODEC_TOKENS_PER_CHUNK,
    _DUPLEX_TURN_END_CODEC_TOKENS,
    _OFFLINE_CODEC_MAX_NEW_TOKENS,
    _QUIET_CODEC_IDS,
    _SILENT_CODEC_IDS,
    MiniCPMO45OmniTTSForConditionalGeneration,
    _native_duplex_chunk_budget,
    _native_duplex_row_meta,
    _restore_weight_norm_weight,
    _trailing_code_runs,
    _turn_end_boundary_eos_masked,
    _turn_end_drain_kind,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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
    talker._request_audio_states = {}
    talker._request_condition_states = {}
    talker._deferred_cleanup_ids = set()
    talker._tts_config = ConditionalChatTTSConfig()
    # Codec sampling contract added by the multi-frame decode work: the fixture
    # runs the deterministic greedy boundary (argmax over head_code) and never
    # arms the multi-frame loop.
    talker._codec_temperature = 0.0
    talker._codec_top_k = 0
    talker._codec_top_p = 1.0
    talker._codec_repetition_penalty = 1.0
    talker._codec_min_tokens = 0
    talker._codec_max_tokens = 4032
    talker._k_step_frames = 0
    talker.supports_multi_frame_decode = False
    talker.head_code = nn.ModuleList([nn.Linear(2, 8, bias=False)])
    with torch.no_grad():
        talker.head_code[0].weight.copy_(torch.eye(8, 2))
    return talker


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


def test_compute_logits_returns_codec_vocab() -> None:
    talker = _make_talker()
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    logits = talker.compute_logits(hidden)

    assert logits.shape == (2, 8)
    assert torch.allclose(logits[0], talker.head_code[0](hidden[0]))


def test_native_duplex_chunk_budget_masks_eos_until_generate_chunk_is_full() -> None:
    assert _native_duplex_chunk_budget({}) == (_DUPLEX_CODEC_TOKENS_PER_CHUNK, _DUPLEX_CODEC_TOKENS_PER_CHUNK)
    assert _native_duplex_chunk_budget({"turn_start": True}) == (_DUPLEX_CODEC_TOKENS_PER_CHUNK, 0)
    # The turn-end chunk drains the owed text: no floor, multi-unit ceiling.
    assert _native_duplex_chunk_budget({"turn_end": True}) == (_DUPLEX_TURN_END_CODEC_TOKENS, 0)
    assert _DUPLEX_TURN_END_CODEC_TOKENS > _DUPLEX_CODEC_TOKENS_PER_CHUNK


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


@pytest.mark.parametrize("as_tensor", [False, True], ids=["transport-list", "tensor"])
@pytest.mark.parametrize(
    "condition_ids,turn_end",
    [([99, 41], False), ([99, 41, 99], True), ([99], True), ([], False)],
)
def test_talker_turn_end_follows_last_condition_token(condition_ids, turn_end, as_tensor) -> None:
    info = {
        "native_duplex": True,
        "duplex": {"epoch": 0, "turn_id": 10},
        "ids": {"tts": torch.tensor(condition_ids, dtype=torch.long) if as_tensor else condition_ids},
        "meta": {"turn_eos_token_id": 99},
    }
    assert _native_duplex_row_meta(info)[-1] is turn_end


def test_venus_delegated_eos_closes_turn_without_a_speech_row():
    info = {
        "native_duplex": True,
        "duplex": {"epoch": 0, "turn_id": 10},
        "ids": {"tts": []},
        "meta": {"turn_eos_token_id": 99, "turn_end": True},
    }
    assert _native_duplex_row_meta(info)[-1] is True


def test_venus_speech_flush_embeds_only_audio_bos_and_runs_one_tail_chunk():
    talker = _make_talker()
    talker.emb_text = nn.Embedding(8, 4)
    talker.emb_code = nn.ModuleList([nn.Embedding(8, 4)])
    talker._text_eos_id = 5
    talker._tts_bos_id = 6
    _, embeds, updates = talker.preprocess(
        torch.zeros(1, dtype=torch.long),
        None,
        _omni_is_prefill=True,
        request_id="req-flush",
        native_duplex=True,
        meta={"speech_flush": True},
        tts_token_ids=torch.empty(0, dtype=torch.long),
        tts_hidden_states=torch.empty(0, 4),
    )
    assert torch.equal(embeds, talker.emb_text(torch.tensor([6])))
    assert updates["audio_state"]["finished"] is False
    assert updates["audio_state"]["min_tokens"] == 0
    assert updates["audio_state"]["max_tokens"] == _DUPLEX_CODEC_TOKENS_PER_CHUNK
    assert updates["audio_state"]["turn_end_drain"] == 0


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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("code_id", [0, 3, 7])
def test_cpu_codec_transport_preserves_embedding_and_owns_delta(device, dtype, code_id) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA embedding")
    talker = _make_talker()
    talker.emb_code = nn.ModuleList([nn.Embedding(8, 2, device=device, dtype=dtype)])
    ids = torch.tensor([code_id], device=device, dtype=torch.int32)
    expected_embeds = talker.emb_code[0](ids.to(dtype=torch.long))
    state = {"finished": False, "step": 0}
    # Factories must explicitly choose CPU even inside a non-CPU default device.
    with torch.device("meta"):
        returned_ids, embeds, updates = talker.preprocess(ids, None, request_id="req", audio_state=state)
    delta = updates["codes"]["audio"]
    assert returned_ids is ids
    assert embeds.device == ids.device
    torch.testing.assert_close(embeds, expected_embeds, rtol=0, atol=0)
    assert delta.device.type == "cpu" and delta.dtype == torch.long
    assert delta.shape == ((0,) if code_id == 7 else (1, 1))
    assert delta.tolist() == ([] if code_id == 7 else [[code_id]])
    assert state == {"finished": code_id == 7, "step": 0}
    ids.fill_(5)
    assert delta.tolist() == ([] if code_id == 7 else [[code_id]])


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
    assert updates["codes"]["audio"].device.type == "cpu"
    assert updates["codes"]["audio"].shape == (0,)

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
    assert decode_updates["codes"]["audio"].device.type == "cpu"
    assert decode_updates["codes"]["audio"].shape == (0,)


@pytest.mark.parametrize("packed", [False, True])
def test_chunked_prefill_tail_aligns_condition_with_prompt_length(mocker, packed) -> None:
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
        tts_hidden_states=pack_handoff_tensor(torch.ones(1, 2)) if packed else torch.ones(1, 2),
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


@pytest.mark.parametrize("mrv2", [False, True])
def test_native_duplex_condition_advance_without_rollover_updates_window_state(mocker, mrv2) -> None:
    talker = _make_talker()
    talker.emb_text = nn.Embedding(1, 2)
    talker.emb_code = nn.ModuleList([nn.Embedding(8, 2)])
    first_condition = torch.ones(2, 2)
    second_condition = torch.full((2, 2), 2.0)
    third_condition = torch.full((2, 2), 3.0)
    mocker.patch.object(
        talker,
        "_build_condition_embeddings",
        side_effect=[first_condition, second_condition, second_condition, third_condition],
    )
    common = {
        "_omni_is_prefill": True,
        "_omni_prompt_len": 2,
        "request_id": "req-history",
        "native_duplex": True,
        "tts_token_ids": torch.tensor([1]),
        "tts_hidden_states": torch.ones(1, 2),
    }

    talker.preprocess(torch.zeros(2, dtype=torch.long), None, meta={"streaming_condition_seq": 0}, **common)
    initial_state = talker._request_condition_states["req-history"]
    assert initial_state["condition_seq"] == 0
    assert torch.equal(initial_state["condition"], first_condition)
    if not mrv2:
        talker._request_audio_states["req-history"]["recent_codes"] = [1, 2, 3]
    else:
        common["ids"] = {"streaming_prompt_previous_codes": [1, 2, 3]}

    _, embeds, _ = talker.preprocess(
        torch.zeros(4, dtype=torch.long),
        None,
        _omni_prompt_len=4,
        meta={"streaming_condition_seq": 1},
        **{key: value for key, value in common.items() if key != "_omni_prompt_len"},
    )

    assert torch.equal(embeds[-2:], second_condition)
    state = talker._request_condition_states["req-history"]
    assert state["condition_seq"] == 1
    assert torch.equal(state["condition"], second_condition)
    assert state["base_recent_codes"] == (1, 2, 3)
    assert talker._request_audio_states["req-history"]["recent_codes"] == [1, 2, 3]

    # A chunked-prefill retry for the same condition is idempotent.
    _, retry_embeds, _ = talker.preprocess(
        torch.zeros(2, dtype=torch.long),
        None,
        _omni_num_computed_tokens=2,
        _omni_prompt_len=4,
        meta={"streaming_condition_seq": 1},
        **{key: value for key, value in common.items() if key != "_omni_prompt_len"},
    )
    assert torch.equal(retry_embeds, second_condition)
    assert talker._request_condition_states["req-history"]["base_recent_codes"] == (1, 2, 3)

    common.pop("ids", None)
    previous_codes = [4, 5]
    expected_rollover = torch.cat(
        [
            second_condition,
            talker.emb_code[0](torch.tensor(previous_codes)),
            third_condition,
        ],
        dim=0,
    )
    _, rollover_embeds, _ = talker.preprocess(
        torch.zeros(expected_rollover.shape[0], dtype=torch.long),
        None,
        _omni_prompt_len=expected_rollover.shape[0],
        ids={"streaming_prompt_previous_codes": previous_codes},
        meta={
            "streaming_condition_seq": 2,
            "streaming_prompt_recompute": True,
        },
        **{key: value for key, value in common.items() if key != "_omni_prompt_len"},
    )
    assert torch.equal(rollover_embeds, expected_rollover)
    rollover_state = talker._request_condition_states["req-history"]
    assert rollover_state["base_recent_codes"] == (1, 2, 3, 4, 5)


def test_native_duplex_sliding_condition_advance_requires_recompute_marker(mocker) -> None:
    talker = _make_talker()
    talker._tts_config.attention_type = "sliding_recompute"
    condition = torch.ones(2, 2)
    mocker.patch.object(talker, "_build_condition_embeddings", return_value=condition)
    common = {
        "_omni_is_prefill": True,
        "_omni_prompt_len": 2,
        "_omni_num_computed_tokens": 0,
        "request_id": "req-sliding",
        "native_duplex": True,
        "tts_token_ids": torch.tensor([1]),
        "tts_hidden_states": torch.ones(1, 2),
    }

    talker.preprocess(torch.zeros(2, dtype=torch.long), None, meta={"streaming_condition_seq": 0}, **common)
    with pytest.raises(ValueError, match="advanced without its streaming recompute marker"):
        talker.preprocess(torch.zeros(2, dtype=torch.long), None, meta={"streaming_condition_seq": 1}, **common)


def test_native_duplex_disabled_recompute_policy_accepts_condition_advance(mocker) -> None:
    talker = _make_talker()
    talker._tts_config.attention_type = "other"
    condition = torch.ones(2, 2)
    mocker.patch.object(talker, "_build_condition_embeddings", return_value=condition)
    common = {
        "_omni_is_prefill": True,
        "_omni_prompt_len": 2,
        "_omni_num_computed_tokens": 0,
        "request_id": "req-disabled-policy",
        "native_duplex": True,
        "tts_token_ids": torch.tensor([1]),
        "tts_hidden_states": torch.ones(1, 2),
    }

    talker.preprocess(torch.zeros(2, dtype=torch.long), None, meta={"streaming_condition_seq": 0}, **common)
    talker.preprocess(torch.zeros(2, dtype=torch.long), None, meta={"streaming_condition_seq": 1}, **common)

    assert talker._request_condition_states["req-disabled-policy"]["condition_seq"] == 1


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


def test_turn_end_drain_masks_only_the_cadence_eos() -> None:
    # Early EOS (text spoken before the first boundary) is allowed.
    assert not _turn_end_boundary_eos_masked(0)
    assert not _turn_end_boundary_eos_masked(18)
    assert not _turn_end_boundary_eos_masked(24)
    # The cadence EOS right after each 25-frame boundary is masked for a short window.
    assert _turn_end_boundary_eos_masked(25)
    assert _turn_end_boundary_eos_masked(29)
    assert not _turn_end_boundary_eos_masked(30)
    assert not _turn_end_boundary_eos_masked(49)
    assert _turn_end_boundary_eos_masked(50)
    assert not _turn_end_boundary_eos_masked(60)


def test_turn_end_drain_kind_follows_sentence_end() -> None:
    assert _turn_end_drain_kind(True, {}) == _DRAIN_NONE
    assert _turn_end_drain_kind(False, {"turn_end": True, "turn_text_ends_sentence": True}) == _DRAIN_NONE
    assert _turn_end_drain_kind(True, {"turn_end": True}) == _DRAIN_BOUNDARY_MASK
    assert _turn_end_drain_kind(True, {"turn_end": True, "turn_text_ends_sentence": False}) == _DRAIN_BOUNDARY_MASK
    assert _turn_end_drain_kind(True, {"turn_end": True, "turn_text_ends_sentence": True}) == _DRAIN_UNTIL_SILENT


def test_trailing_code_runs_count_only_the_current_condition() -> None:
    silent, quiet = _SILENT_CODEC_IDS[0], next(c for c in _QUIET_CODEC_IDS if c not in _SILENT_CODEC_IDS)
    assert _trailing_code_runs(0, [silent, silent]) == (0, 0)
    # Codes retained from the previous condition do not count.
    assert _trailing_code_runs(1, [silent, silent]) == (1, 1)
    assert _trailing_code_runs(3, [1, silent, silent]) == (2, 2)
    assert _trailing_code_runs(3, [silent, quiet, silent]) == (1, 3)
    assert _trailing_code_runs(2, [quiet, 1]) == (0, 0)
    assert _trailing_code_runs(2, [quiet, 1], silent_ids=(1,), quiet_ids=(1,)) == (1, 1)


@pytest.mark.parametrize("legacy_none", [False, True])
@pytest.mark.parametrize("sessions", [2, 4, 8, 16])
def test_mrv2_overlapping_native_conditions_use_distinct_request_ids(mocker, legacy_none, sessions):
    talker = _make_talker()
    talker.emb_text = nn.Embedding(1, 2)
    mocker.patch.object(talker, "_build_condition_embeddings", return_value=torch.ones(2, 2))

    def condition(req_id, seq, turn_start=False):
        return talker.preprocess(
            torch.zeros(2, dtype=torch.long),
            None,
            **({"request_id": None} if legacy_none else {}),
            req_id=req_id,
            native_duplex=True,
            _omni_is_prefill=True,
            _omni_prompt_len=2,
            tts_token_ids=torch.tensor([1]),
            tts_hidden_states=torch.ones(1, 2),
            meta={"streaming_condition_seq": seq, "turn_start": turn_start},
        )

    condition("session-a", 0, True)
    for seq in (1, 2, 3):
        condition("session-a", seq)
    others = [f"session-{i}" for i in range(sessions - 1)]
    for req_id in others:
        condition(req_id, 0, True)
    for seq in range(4, 12):
        condition("session-a", seq)
        for req_id in reversed(others):
            condition(req_id, seq - 3)
    assert talker._request_condition_states["session-a"]["condition_seq"] == 11
    assert all(talker._request_condition_states[req_id]["condition_seq"] == 8 for req_id in others)
    assert set(talker._request_audio_states) == {"session-a", *others}
    talker.on_requests_finished({others[0]})
    talker._flush_deferred_cleanup()
    assert others[0] not in talker._request_condition_states
    assert others[0] not in talker._request_audio_states
    condition("new-session", 0, True)
    condition("session-a", 12)
    assert talker._request_condition_states["new-session"]["condition_seq"] == 0
    assert talker._request_condition_states["session-a"]["condition_seq"] == 12


def test_host_sampler_resumes_only_for_a_new_duplex_condition(mocker):
    talker = _make_talker()
    talker._codec_seed = 0
    talker.emb_text = nn.Embedding(1, 2)
    mocker.patch.object(talker, "_build_condition_embeddings", return_value=torch.ones(2, 2))

    def prefill(request_id, seq):
        return talker.preprocess(
            torch.zeros(2, dtype=torch.long),
            None,
            req_id=request_id,
            native_duplex=True,
            _omni_is_prefill=True,
            _omni_prompt_len=2,
            tts_token_ids=torch.tensor([1]),
            tts_hidden_states=torch.ones(1, 2),
            **({"meta": {"streaming_condition_seq": seq, "turn_start": seq == 0}} if seq is not None else {}),
        )[2]["audio_state"]

    def sample(request_id, state, max_tokens):
        return talker._sample_audio_code_host(
            hidden_state=torch.zeros(1, 2),
            history=torch.empty(0, dtype=torch.long),
            request_id=request_id,
            request_state=state,
            step=0,
            min_tokens=0,
            max_tokens=max_tokens,
            eos_window_masked=False,
            lg_dev=torch.zeros(8),
        )

    first = prefill("a", 0)
    other = prefill("b", 0)
    assert sample("a", first, 1)[1:] == (True, False)
    assert sample("b", other, 1)[1:] == (True, False)
    host_state = talker._request_host_codec_states["a"]
    generator = host_state["generator"]
    host_state["history"] = [3, 4]
    host_state["bins"] = {3: 2}

    # Repeating a prefill for the same condition must not reopen the segment.
    retry = prefill("a", 0)
    assert sample("a", retry, 3)[1:] == (True, False)

    next_condition = prefill("a", 1)
    assert host_state["generator"] is generator
    assert host_state["history"] == [3, 4]
    assert host_state["bins"] == {3: 2}
    assert sample("a", next_condition, 3)[1:] == (False, True)
    assert talker._request_host_codec_states["b"]["finished"] is True

    # An unnumbered prefill retains the other request's finished state.
    prefill("b", None)
    assert talker._request_host_codec_states["b"]["finished"] is True

    talker.on_requests_finished({"a"})
    talker._flush_deferred_cleanup()
    assert "a" not in talker._request_host_codec_states
