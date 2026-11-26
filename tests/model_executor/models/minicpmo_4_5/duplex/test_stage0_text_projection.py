# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage0 text projection for a Thinker-only MiniCPM-o deployment."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from vllm_omni.engine.duplex.plugin import DuplexDataPlaneContext
from vllm_omni.model_executor.models.minicpmo_4_5.duplex import data_plane as data_plane_module
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.data_plane import (
    MiniCPMO45DataPlaneSession,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REQUEST_ID = "duplex-s.c2Vzc2lvbg==.e.0.r.stage0"
LISTEN, SPEAK, CHUNK_EOS, CHUNK_TTS_EOS, TURN_EOS = 7, 8, 5, 6, 9
SPECIAL_TOKEN_IDS = {
    "listen_token_id": LISTEN,
    "speak_token_id": SPEAK,
    "chunk_eos_token_id": CHUNK_EOS,
    "chunk_tts_eos_token_id": CHUNK_TTS_EOS,
    "turn_eos_token_id": TURN_EOS,
}
A, B = 100, 101
SPECIAL_TOKENS = {
    LISTEN: "<|listen|>",
    SPEAK: "<|speak|>",
    CHUNK_EOS: "<|chunk_eos|>",
    CHUNK_TTS_EOS: "<|chunk_tts_eos|>",
    TURN_EOS: "<|turn_eos|>",
}


def _tokenizer(*, byte_level: bool = False) -> PreTrainedTokenizerFast:
    vocabulary = {f"unused_{index}": index for index in range(B + 1)}
    for token_id, token in {**SPECIAL_TOKENS, A: "一", B: "二"}.items():
        del vocabulary[f"unused_{token_id}"]
        vocabulary[token] = token_id
    if byte_level:
        vocabulary.update(
            (token, index) for index, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()), start=B + 1)
        )
        backend = Tokenizer(models.BPE(vocabulary, []))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        backend.decoder = decoders.ByteLevel()
    else:
        backend = Tokenizer(models.WordLevel(vocabulary, unk_token="unused_0"))
        backend.decoder = decoders.Fuse()
    backend.add_special_tokens(list(SPECIAL_TOKENS.values()))
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        additional_special_tokens=[token for token_id, token in SPECIAL_TOKENS.items() if token_id != TURN_EOS],
    )


TOKENIZER = _tokenizer()


def _context(**overrides: object) -> DuplexDataPlaneContext:
    values: dict[str, object] = {
        "auto_responds": True,
        "modalities": ("text",),
    }
    values.update(overrides)
    return DuplexDataPlaneContext(**values)


def _snapshot(
    token_ids: list[int],
    text: str,
    *,
    stop: bool = False,
) -> SimpleNamespace:
    """A Stage0 output as the resumable request reports it: every token so far."""
    multimodal_output: dict[str, object] = {
        "latent": np.ones((1, 16), dtype=np.float32),
        "special_token_ids": dict(SPECIAL_TOKEN_IDS),
    }
    completion = SimpleNamespace(
        token_ids=list(token_ids),
        text=text,
        finish_reason="stop" if stop else None,
        stop_reason=token_ids[-1] if stop and token_ids else None,
        multimodal_output={},
    )
    return SimpleNamespace(
        request_id=REQUEST_ID,
        outputs=[completion],
        finished=False,
        multimodal_output=multimodal_output,
    )


def _project(
    session: MiniCPMO45DataPlaneSession, output: SimpleNamespace, **context: object
) -> list[dict[str, object]]:
    return list(session.project_output(output, context=_context(**context)))


def _session(
    *, tokenizer: Tokenizer | None = TOKENIZER.backend_tokenizer, thinker_only: bool = True
) -> tuple[MiniCPMO45DataPlaneSession, list[object]]:
    encoded: list[object] = []

    def encode(audio: object, *args: object) -> str:
        encoded.append(audio)
        return "pcm"

    session = MiniCPMO45DataPlaneSession(encode)
    session.thinker_only = thinker_only
    session.tokenizer = tokenizer
    return session, encoded


def _texts(results: list[dict[str, object]]) -> str:
    return "".join(str(result.get("text") or "") for result in results)


def test_latent_is_never_encoded_as_audio() -> None:
    session, encoded = _session()
    results = _project(session, _snapshot([SPEAK, A], "一"))
    assert encoded == []
    assert not any(result.get("audio_data") for result in results)
    assert all(result["stage_role"] == "llm" for result in results)
    assert _texts(results) == "一"


def test_listen_only_units_are_listen_decisions_without_text() -> None:
    session, _ = _session()
    first = _project(session, _snapshot([LISTEN], "", stop=True))
    second = _project(session, _snapshot([LISTEN, LISTEN], "", stop=True))
    for results in (first, second):
        assert len(results) == 1
        assert results[0]["is_listen"] is True
        assert results[0]["end_of_turn"] is False
        assert results[0]["model_turn_id"] == 0
        assert not results[0].get("text")


def test_reply_units_emit_text_once_and_close_each_segment() -> None:
    session, _ = _session()
    steps = [
        ([LISTEN], "", True),
        ([LISTEN, SPEAK], "", False),
        ([LISTEN, SPEAK, A], "一", False),
        ([LISTEN, SPEAK, A, B], "一二", False),
        ([LISTEN, SPEAK, A, B, CHUNK_EOS], "一二", True),
        # A repeated snapshot of the same unit end emits nothing.
        ([LISTEN, SPEAK, A, B, CHUNK_EOS], "一二", True),
        ([LISTEN, SPEAK, A, B, CHUNK_EOS, SPEAK, A], "一二一", False),
    ]
    collected: list[list[dict[str, object]]] = []
    for token_ids, text, stop in steps:
        collected.append(_project(session, _snapshot(token_ids, text, stop=stop), active_response_id="resp"))

    assert _texts([r for results in collected for r in results]) == "一二一"
    segment_ends = [r for results in collected for r in results if r.get("segment_complete")]
    assert len(segment_ends) == 1
    assert segment_ends[0]["end_of_turn"] is False
    assert collected[5] == []
    assert not any(r.get("end_of_turn") for results in collected for r in results)


def test_equal_text_in_two_units_is_two_deltas() -> None:
    session, _ = _session()
    first = _project(session, _snapshot([SPEAK, A, CHUNK_EOS], "一", stop=True))
    second = _project(session, _snapshot([SPEAK, A, CHUNK_EOS, SPEAK, A, CHUNK_EOS], "一一", stop=True))
    assert _texts(first) == "一"
    assert _texts(second) == "一"


@pytest.mark.parametrize(
    "unit",
    [
        [SPEAK, A, TURN_EOS, LISTEN],
        [SPEAK, A, TURN_EOS, CHUNK_EOS],
    ],
)
def test_final_reply_unit_emits_trailing_text_before_ending_the_turn(unit: list[int]) -> None:
    session, _ = _session()
    results = _project(session, _snapshot(unit, "一", stop=True), active_response_id="resp", turn_id=3)
    assert [bool(r.get("text")) for r in results] == [True, False]
    assert results[0]["text"] == "一"
    terminal = results[-1]
    assert terminal["end_of_turn"] is True
    assert terminal["is_listen"] is False
    assert terminal["model_turn_id"] == 3


@pytest.mark.parametrize("terminator", [LISTEN, CHUNK_EOS, CHUNK_TTS_EOS])
@pytest.mark.parametrize("tail", [[B], [SPEAK], [SPEAK, B]])
def test_text_or_speak_after_turn_eos_keeps_the_turn_open(terminator: int, tail: list[int]) -> None:
    session, _ = _session()
    tokens = [SPEAK, A, TURN_EOS, *tail, terminator]
    text = TOKENIZER.decode(tokens, skip_special_tokens=True)
    results = _project(session, _snapshot(tokens, text, stop=True), active_response_id="resp", turn_id=3)

    assert _texts(results) == text
    assert not any(result.get("end_of_turn") for result in results)
    assert results[-1].get("is_listen") is (terminator == LISTEN)
    assert bool(results[-1].get("segment_complete")) is (terminator != LISTEN)
    assert all(result["model_turn_id"] == 3 for result in results if result.get("text"))

    tokens.extend([SPEAK, A, TURN_EOS, CHUNK_EOS])
    results = _project(session, _snapshot(tokens, text + "一", stop=True), active_response_id="resp", turn_id=3)
    assert _texts(results) == "一"
    assert [result["model_turn_id"] for result in results if result.get("end_of_turn")] == [3]


def test_turn_eos_from_an_earlier_unit_does_not_end_a_later_listen() -> None:
    session, _ = _session()
    _project(session, _snapshot([SPEAK, A, TURN_EOS, CHUNK_EOS], "一", stop=True))
    results = _project(session, _snapshot([SPEAK, A, TURN_EOS, CHUNK_EOS, LISTEN], "一", stop=True))
    assert len(results) == 1
    assert results[0]["is_listen"] is True
    assert results[0]["end_of_turn"] is False


def test_turns_in_one_snapshot_get_consecutive_ids() -> None:
    session, _ = _session()
    tokens = [SPEAK, A, TURN_EOS, LISTEN, SPEAK, B, TURN_EOS, LISTEN]
    results = _project(session, _snapshot(tokens, "一二", stop=True), turn_id=4)
    ends = [r["model_turn_id"] for r in results if r.get("end_of_turn")]
    assert ends == [4, 5]
    assert [(r["model_turn_id"], r["text"]) for r in results if r.get("text")] == [(4, "一"), (5, "二")]


def test_coalesced_snapshot_keeps_the_next_turns_partial_text() -> None:
    session, _ = _session()
    tokens = [SPEAK, A, TURN_EOS, LISTEN, SPEAK, B]
    results = _project(session, _snapshot(tokens, "一二"), turn_id=4)
    assert [(r["model_turn_id"], r["text"]) for r in results if r.get("text")] == [(4, "一"), (5, "二")]
    assert [r["model_turn_id"] for r in results if r.get("end_of_turn")] == [4]
    assert _project(session, _snapshot(tokens, "一二"), turn_id=5) == []

    session.begin_request(REQUEST_ID)
    results = _project(session, _snapshot([*tokens, TURN_EOS, LISTEN], "一二", stop=True), turn_id=5)
    assert _texts(results) == ""
    assert [r["model_turn_id"] for r in results if r.get("end_of_turn")] == [5]


def test_partial_unit_before_a_coalesced_turn_boundary_preserves_text_ownership() -> None:
    session, _ = _session()
    first = _project(session, _snapshot([SPEAK, A], "一"), turn_id=4)
    assert [(r["model_turn_id"], r["text"]) for r in first if r.get("text")] == [(4, "一")]
    tokens = [SPEAK, A, B, TURN_EOS, LISTEN, SPEAK, A]
    results = _project(session, _snapshot(tokens, "一二一"), turn_id=4)
    assert [(r["model_turn_id"], r["text"]) for r in results if r.get("text")] == [(4, "二"), (5, "一")]
    assert [r["model_turn_id"] for r in results if r.get("end_of_turn")] == [4]


def test_segments_and_repeated_text_remain_in_their_model_turn() -> None:
    session, _ = _session()
    tokens = [SPEAK, A, CHUNK_EOS, SPEAK, B, TURN_EOS, LISTEN, SPEAK, A, TURN_EOS, LISTEN]
    results = _project(session, _snapshot(tokens, "一二一", stop=True), turn_id=4)
    by_turn = {
        turn_id: _texts([result for result in results if result.get("model_turn_id") == turn_id]) for turn_id in (4, 5)
    }
    assert by_turn == {4: "一二", 5: "一"}
    assert [r["model_turn_id"] for r in results if r.get("segment_complete")] == [4]
    assert [r["model_turn_id"] for r in results if r.get("end_of_turn")] == [4, 5]


def test_backend_special_tokens_missing_from_hf_special_ids_are_hidden() -> None:
    session, _ = _session()
    assert TURN_EOS not in TOKENIZER.all_special_ids
    output = _snapshot([SPEAK, A, TURN_EOS, LISTEN], "一", stop=True)
    assert _texts(_project(session, output)) == "一"


@pytest.mark.parametrize("snapshot_text", ["", "unrelated text"])
def test_model_turn_text_is_decoded_from_tokens_instead_of_snapshot_text(snapshot_text: str) -> None:
    session, _ = _session()
    tokens = [SPEAK, A, TURN_EOS, LISTEN, SPEAK, B, TURN_EOS, LISTEN]
    results = _project(session, _snapshot(tokens, snapshot_text, stop=True), turn_id=4)
    assert [(r["model_turn_id"], r["text"]) for r in results if r.get("text")] == [(4, "一"), (5, "二")]


@pytest.mark.parametrize("leading_listen", [False, True])
def test_coalesced_turns_attach_stage_metrics_to_text_only_once_per_snapshot(leading_listen: bool) -> None:
    session, _ = _session()
    tokens = [SPEAK, A, TURN_EOS, LISTEN, SPEAK, B, TURN_EOS, LISTEN]
    if leading_listen:
        tokens.insert(0, LISTEN)
    output = _snapshot(tokens, "一二", stop=True)
    metrics = {"0": {"num_tokens_out": len(tokens), "stage_gen_time_ms": 120.0}}
    output.metrics = {"stage_metrics": metrics}
    results = _project(session, output, turn_id=4)
    assert [result["stage_metrics"] for result in results if "stage_metrics" in result] == [metrics]
    assert [(result.get("model_turn_id"), result.get("text")) for result in results if "stage_metrics" in result] == [
        (4, "一")
    ]
    assert [r["model_turn_id"] for r in results if r.get("end_of_turn")] == [4, 5]


def test_leading_listen_does_not_consume_partial_reply_stage_metrics() -> None:
    session, _ = _session()
    output = _snapshot([LISTEN, SPEAK, A], "一")
    metrics = {"0": {"num_tokens_out": 3, "stage_gen_time_ms": 120.0}}
    output.metrics = {"stage_metrics": metrics}
    results = _project(session, output)
    assert results[0]["is_listen"] is True
    assert "stage_metrics" not in results[0]
    assert [result["text"] for result in results if "stage_metrics" in result] == ["一"]
    assert [result["stage_metrics"] for result in results if "stage_metrics" in result] == [metrics]


def test_thinker_only_without_a_tokenizer_fails_without_leaking_latent_or_text() -> None:
    session, encoded = _session(tokenizer=None)
    results = _project(session, _snapshot([SPEAK, A, CHUNK_EOS], "一", stop=True))
    assert [result.get("error_code") for result in results] == ["runtime_data_plane_missing_tokenizer"]
    assert _texts(results) == ""
    assert encoded == []


def test_close_request_releases_its_text_decoder_and_cursors() -> None:
    session, _ = _session()
    output = _snapshot([SPEAK, A], "一")
    assert _texts(_project(session, output)) == "一"
    session.close_request(REQUEST_ID)
    assert _texts(_project(session, output)) == "一"


def test_interleaved_requests_keep_independent_byte_level_decoder_state() -> None:
    tokenizer = _tokenizer(byte_level=True)
    session, _ = _session(tokenizer=tokenizer.backend_tokenizer)
    first_ids = tokenizer.encode("你", add_special_tokens=False)
    second_ids = tokenizer.encode("好", add_special_tokens=False)
    assert len(first_ids) > 1
    first = _project(session, _snapshot([SPEAK, first_ids[0]], ""))
    assert _texts(first) == ""

    second_output = _snapshot([SPEAK, *second_ids, TURN_EOS, LISTEN], "好", stop=True)
    second_output.request_id = "other-request"
    second = _project(session, second_output)
    assert _texts(second) == "好"
    first = _project(
        session,
        _snapshot([SPEAK, *first_ids, TURN_EOS, LISTEN], "你", stop=True),
    )
    assert _texts(first) == "你"


@pytest.mark.parametrize("one_token_per_snapshot", [False, True])
def test_byte_level_text_and_emoji_survive_snapshot_and_turn_boundaries(one_token_per_snapshot: bool) -> None:
    tokenizer = _tokenizer(byte_level=True)
    session, _ = _session(tokenizer=tokenizer.backend_tokenizer)
    first_text, second_text = "你好🙂", "，世界！"
    first_ids = tokenizer.encode(first_text, add_special_tokens=False)
    second_ids = tokenizer.encode(second_text, add_special_tokens=False)
    tokens = [SPEAK, *first_ids, TURN_EOS, LISTEN, SPEAK, *second_ids, TURN_EOS, LISTEN]
    assert len(first_ids) > len(first_text), "fixture must split Unicode characters into byte tokens"
    collected: list[dict[str, object]] = []
    turn_id = 4
    cuts = range(1, len(tokens) + 1) if one_token_per_snapshot else [len(tokens)]
    for cut in cuts:
        prefix = tokens[:cut]
        text = tokenizer.decode(prefix, skip_special_tokens=True).rstrip("\ufffd")
        results = _project(
            session,
            _snapshot(prefix, text, stop=cut == len(tokens)),
            turn_id=turn_id,
        )
        collected.extend(results)
        turn_id += sum(bool(result.get("end_of_turn")) for result in results)
    assert "\ufffd" not in _texts(collected)
    assert {
        turn_id: _texts([result for result in collected if result.get("model_turn_id") == turn_id])
        for turn_id in (4, 5)
    } == {4: first_text, 5: second_text}
    assert [r["model_turn_id"] for r in collected if r.get("end_of_turn")] == [4, 5]


@pytest.mark.parametrize("tail", [[SPEAK, A, TURN_EOS], [SPEAK, A, B]])
def test_length_stop_without_a_terminator_fails_instead_of_completing(tail: list[int]) -> None:
    # vLLM checks stop tokens before the length cap, so a capped tail is unterminated.
    session, _ = _session()
    text = TOKENIZER.decode(tail, skip_special_tokens=True)
    output = _snapshot(tail, text, stop=True)
    output.outputs[0].finish_reason = "length"
    output.outputs[0].stop_reason = None
    results = _project(session, output, active_response_id="resp")
    assert _texts(results) == text
    assert not any(r.get("end_of_turn") or r.get("segment_complete") or r.get("is_listen") for r in results)
    assert [r.get("error_code") for r in results if r.get("error_code")] == ["runtime_data_plane_unterminated_unit"]
    # The failed tail is consumed: a repeated snapshot does not fail again.
    assert _project(session, output, active_response_id="resp") == []


def test_stop_string_without_a_unit_terminator_fails_instead_of_completing() -> None:
    session, _ = _session()
    output = _snapshot([SPEAK, A, B], "一二", stop=True)
    output.outputs[0].stop_reason = "END"
    results = _project(session, output, active_response_id="resp", turn_id=2)
    assert _texts(results) == "一二"
    assert not any(r.get("end_of_turn") or r.get("segment_complete") for r in results)
    assert results[-1]["error_code"] == "runtime_data_plane_unterminated_unit"
    assert _project(session, output, active_response_id="resp", turn_id=2) == []


@pytest.mark.parametrize("request_id", [None, "", 123])
def test_output_without_a_request_id_is_dropped(request_id: object) -> None:
    session, _ = _session()
    output = _snapshot([SPEAK, A, CHUNK_EOS], "一", stop=True)
    output.request_id = request_id
    assert _project(session, output) == []
    output.outputs[0].text = "一二"
    output.outputs[0].token_ids.extend([SPEAK, B, CHUNK_EOS])
    assert _project(session, output) == []


def test_speech_deployment_keeps_the_audio_path() -> None:
    session, encoded = _session(thinker_only=False)
    _project(session, _snapshot([SPEAK, A], "一"))
    assert len(encoded) == 1


@pytest.mark.parametrize(("snapshot", "text"), [([SPEAK, A], "一"), ([], "")])
def test_a_queued_snapshot_reads_its_own_token_ids_not_the_live_detokenizer_list(
    snapshot: list[int], text: str
) -> None:
    session, _ = _session()
    output = _snapshot(snapshot, text)
    live = output.outputs[0].token_ids
    output.outputs[0].cumulative_token_ids = list(snapshot)
    live.extend([B, TURN_EOS, LISTEN])

    results = _project(session, output)

    assert _texts(results) == text
    assert not any(result.get("end_of_turn") for result in results), "a later turn end leaked in"
    assert not any(result.get("is_listen") for result in results), "a later listen leaked in"


@pytest.mark.parametrize("stale", [{"meta.duplex_turn_id": 0}, {"meta.duplex_epoch": 0}], ids=["turn", "epoch"])
def test_skipped_stale_snapshot_does_not_open_the_next_reply(stale: dict[str, int]) -> None:
    session, _ = _session()
    context = {"active_response_id": "response", "active_response_turn_id": 1, "turn_id": 1, "epoch": 1}
    old = _snapshot([SPEAK, A, CHUNK_EOS], "一")
    old.multimodal_output.update({"meta.duplex_turn_id": 1, "meta.duplex_epoch": 1, **stale})
    assert _project(session, old, **context) == []

    current = _snapshot([SPEAK, A, CHUNK_EOS, SPEAK, B, CHUNK_EOS], "一二")
    current.multimodal_output.update({"meta.duplex_turn_id": 1, "meta.duplex_epoch": 1})
    assert _texts(_project(session, current, **context)) == "二"


def test_skipped_stale_open_unit_does_not_end_the_current_turn() -> None:
    session, _ = _session()
    context = {"active_response_id": "response", "active_response_turn_id": 1, "turn_id": 1}
    old = _snapshot([SPEAK, A, TURN_EOS], "一")
    old.multimodal_output["meta.duplex_turn_id"] = 0
    assert _project(session, old, **context) == []

    # The stale unit closes here; its TURN_EOS must not end turn 1.
    current = _snapshot([SPEAK, A, TURN_EOS, LISTEN, SPEAK, B, CHUNK_EOS], "一二")
    current.multimodal_output["meta.duplex_turn_id"] = 1
    results = _project(session, current, **context)
    assert [(r["model_turn_id"], r["text"]) for r in results if r.get("text")] == [(1, "二")]
    assert not any(r.get("end_of_turn") or r.get("model_listen") for r in results)


@pytest.mark.parametrize("ids", [[1, 2, 3, 4, 5], (1, 2, 3, 4, 5), np.array([1, 2, 3, 4, 5])], ids=type)
def test_snapshot_ids_coerce_only_the_unconsumed_tail(monkeypatch, ids) -> None:
    coerced: list[int] = []
    coerce = data_plane_module._coerce_int_list

    def counting_coerce(value: object) -> list[int]:
        result = coerce(value)
        coerced.append(len(result))
        return result

    monkeypatch.setattr(data_plane_module, "_coerce_int_list", counting_coerce)
    assert data_plane_module._token_ids_from(ids, 3) == (5, [4, 5])
    if not isinstance(ids, np.ndarray):
        assert coerced == [2], "a resident request must not re-coerce its consumed prefix"
