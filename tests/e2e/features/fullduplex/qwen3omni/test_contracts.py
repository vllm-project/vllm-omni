# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Contract tests for the Qwen3-Omni duplex integration.

These exercise the framework's real duck-type gates -- the same functions the
engine and API server call at startup -- rather than mocks of them. They do
NOT establish that the integration works end to end: the worker-side stage-0
audio embedding path is unimplemented (see
``vllm_omni/experimental/fullduplex/qwen3omni/stage0.py``) and no test here
runs a model.
"""

from __future__ import annotations

import base64

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.experimental.fullduplex.engine.contracts import (
    DuplexAppendPlan,
    DuplexInputMode,
)
from vllm_omni.experimental.fullduplex.engine.duplex_runtime import (
    validate_duplex_runtime_extension,
)
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.openai.runtime_adapter import (
    validate_serving_runtime_adapter,
)
from vllm_omni.experimental.fullduplex.qwen3omni.input import Qwen3OmniPcmAppendBuffer
from vllm_omni.experimental.fullduplex.qwen3omni.policy import Qwen3OmniDuplexPolicy
from vllm_omni.experimental.fullduplex.qwen3omni.runtime import (
    Qwen3OmniDuplexRuntimeExtension,
    parse_tool_calls,
    Qwen3OmniDuplexRuntimeExtension,
    build_duplex_prompt_token_ids,
    duplex_audio_token_count,
)
from vllm_omni.experimental.fullduplex.qwen3omni.serving_adapter import (
    Qwen3OmniServingRuntimeAdapter,
)
from vllm_omni.experimental.fullduplex.qwen3omni.stage0 import (
    Qwen3OmniStage0DuplexRuntime,
)

CHUNK_MS = Qwen3OmniDuplexPolicy.CHUNK_PERIOD_MS


def _fence(seq: int = 1) -> DuplexFence:
    return DuplexFence(
        session_id="sess-1",
        epoch=0,
        turn_id=0,
        response_seq=seq,
        incarnation=1,
    )


def _pcm_payload(num_samples: int) -> dict[str, object]:
    data = b"\x00\x00\x00\x00" * num_samples
    return {
        "audio": base64.b64encode(data).decode("ascii"),
        "format": Qwen3OmniDuplexPolicy.PCM_FORMAT,
        "sample_rate_hz": Qwen3OmniDuplexPolicy.SAMPLE_RATE_HZ,
        "num_samples": num_samples,
    }


# --- framework gates -------------------------------------------------------


def test_runtime_extension_passes_engine_gate() -> None:
    """The engine's own validator accepts the extension for a 3-stage pipeline."""
    defaults = (SamplingParams(), SamplingParams(), SamplingParams())
    validate_duplex_runtime_extension(
        Qwen3OmniDuplexRuntimeExtension(),
        sampling_defaults=defaults,
    )


def test_serving_adapter_passes_api_server_gate() -> None:
    """``load_serving_runtime_adapter``'s validator accepts the adapter."""
    adapter = Qwen3OmniServingRuntimeAdapter(lambda *_args: "")
    validate_serving_runtime_adapter(adapter)


def test_pipeline_declares_duplex_paths_that_import() -> None:
    """The dotted paths on the pipeline config resolve to the real classes."""
    from importlib import import_module

    from vllm_omni.model_executor.models.qwen3_omni.pipeline import (
        QWEN3_OMNI_PIPELINE,
    )

    for path, expected in (
        (QWEN3_OMNI_PIPELINE.duplex_runtime_extension, Qwen3OmniDuplexRuntimeExtension),
        (QWEN3_OMNI_PIPELINE.duplex_serving_adapter, Qwen3OmniServingRuntimeAdapter),
    ):
        assert isinstance(path, str) and path
        module_name, _, attribute = path.rpartition(".")
        assert getattr(import_module(module_name), attribute) is expected


# --- append plan -----------------------------------------------------------


def test_plan_append_emits_only_surviving_prompt_keys() -> None:
    """Guards the silent key-drop in build_engine_core_request_from_tokens.

    Only prompt_token_ids / prompt_embeds / cache_salt /
    additional_information / model_intermediate_buffer survive
    ``orchestrator.py:118-158``. A key added here that is not in that set
    would be discarded with no error, so assert the prompt carries nothing
    that would be lost.
    """
    surviving = {
        "prompt_token_ids",
        "prompt_embeds",
        "cache_salt",
        "additional_information",
        "model_intermediate_buffer",
    }
    plan = Qwen3OmniDuplexRuntimeExtension().plan_append(
        request_id="req-1",
        fence=_fence(),
        session_config={},
        runtime_config={},
        seq=1,
        turn_seq=1,
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        payload=_pcm_payload(Qwen3OmniDuplexPolicy.CHUNK_SAMPLES),
        final=False,
        sampling_params=None,
    )
    assert isinstance(plan, DuplexAppendPlan)
    assert set(plan.prompt) <= surviving, f"prompt keys would be dropped: {set(plan.prompt) - surviving}"


def test_plan_append_sets_the_data_plane_gate() -> None:
    """``duplex.data_plane`` must be True or every worker branch no-ops."""
    plan = Qwen3OmniDuplexRuntimeExtension().plan_append(
        request_id="req-1",
        fence=_fence(),
        session_config={},
        runtime_config={},
        seq=1,
        turn_seq=1,
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        payload=_pcm_payload(Qwen3OmniDuplexPolicy.CHUNK_SAMPLES),
        final=False,
        sampling_params=None,
    )
    duplex = plan.prompt["model_intermediate_buffer"]["duplex"]
    assert duplex["data_plane"] is True
    assert duplex["session_id"] == "sess-1"


def test_plan_append_emits_all_duplex_keys_every_time() -> None:
    """The worker-side merge is additive, so omitted keys would go stale.

    ``gpu_model_runner.py:2035-2042`` overlays the new ``duplex`` dict onto
    the previous one. A conditionally-omitted key keeps its old value instead
    of clearing, so every append must carry the full set.
    """
    extension = Qwen3OmniDuplexRuntimeExtension()
    keysets = []
    for seq, final in ((1, False), (2, False), (3, True)):
        plan = extension.plan_append(
            request_id="req-1",
            fence=_fence(seq),
            session_config={},
            runtime_config={},
            seq=seq,
            turn_seq=1,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
            payload=_pcm_payload(Qwen3OmniDuplexPolicy.CHUNK_SAMPLES),
            final=final,
            sampling_params=None,
        )
        keysets.append(set(plan.prompt["model_intermediate_buffer"]["duplex"]))
    assert keysets[0] == keysets[1] == keysets[2]


def test_plan_append_rejects_unsupported_input_mode() -> None:
    with pytest.raises(ValueError, match="does not support input mode"):
        Qwen3OmniDuplexRuntimeExtension().plan_append(
            request_id="req-1",
            fence=_fence(),
            session_config={},
            runtime_config={},
            seq=1,
            turn_seq=1,
            mode=DuplexInputMode.ROLLBACK_TO_CHECKPOINT,
            payload=_pcm_payload(16),
            final=False,
            sampling_params=None,
        )


def test_token_budget_matches_worker_expected_embedding_count() -> None:
    """The reservation and the worker's embedding count must agree.

    A mismatch is absorbed silently by the model runner (truncate/pad), so
    pin the two calculations together.
    """
    for num_samples in (1600, 8000, 16000, 24000, 80000):
        payload = _pcm_payload(num_samples)
        assert duplex_audio_token_count(payload) == Qwen3OmniStage0DuplexRuntime.expected_embedding_count(num_samples)


def test_audio_token_geometry_matches_vllm_thinker_formula() -> None:
    """Pin our integer reimplementation to vLLM's own conv length arithmetic.

    Mirrors ``_get_feat_extract_output_lengths`` in vLLM's
    ``qwen3_omni_moe_thinker.py``. If that function changes upstream this
    test must fail -- the serving layer reserves scheduler slots from it, and
    under-reserving truncates embeddings with no error.

    Derived from Qwen3-Omni-30B-A3B-Instruct: WhisperFeatureExtractor,
    hop_length=160 @ 16 kHz => 100 mel frames per second => 13 tokens/second.
    A linear samples-per-token approximation gives 10 and is wrong.
    """

    def reference(mel_frames: int) -> int:
        leave = mel_frames % 100
        feat_lengths = (leave - 1) // 2 + 1
        return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (mel_frames // 100) * 13

    for mel_frames in range(1, 1001):
        assert Qwen3OmniDuplexPolicy.audio_tokens_for_mel_frames(mel_frames) == reference(mel_frames)

    # Anchor the headline cases so a silent drift is obvious in the diff.
    assert Qwen3OmniDuplexPolicy.audio_tokens_for_samples(16000) == 13
    assert Qwen3OmniDuplexPolicy.audio_tokens_for_samples(8000) == 7
    assert Qwen3OmniDuplexPolicy.audio_tokens_for_samples(80000) == 65
    assert Qwen3OmniDuplexPolicy.tokens_per_chunk() == 13


# --- sampling params -------------------------------------------------------


def test_configure_sampling_params_is_per_stage_and_length_preserving() -> None:
    defaults = (SamplingParams(max_tokens=1), SamplingParams(max_tokens=2), SamplingParams(max_tokens=3))
    configured = Qwen3OmniDuplexRuntimeExtension().configure_sampling_params(
        runtime_config={"duplex_stage_max_tokens": {"0": 128, "1": 4096}},
        defaults=defaults,
    )
    assert len(configured) == len(defaults)
    assert configured[0].max_tokens == 128
    assert configured[1].max_tokens == 4096
    # Untouched stage keeps the checkpoint's own value.
    assert configured[2].max_tokens == 3


def test_decide_output_never_short_circuits() -> None:
    """Qwen3-Omni has no model-native turn signal; None for every stage."""
    extension = Qwen3OmniDuplexRuntimeExtension()
    for stage_id in (0, 1, 2):
        assert (
            extension.decide_output(
                stage_id=stage_id,
                final_stage_id=2,
                segment_finished=True,
                segment_token_ids=(1, 2, 3),
                segment_output_metadata={"meta.listen_token_id": 151645},
                output=object(),
            )
            is None
        )


# --- PCM buffer ------------------------------------------------------------


def test_appends_never_emit_mid_turn() -> None:
    """Audio is held until commit so the thinker gets one well-formed turn.

    Emitting mid-turn makes the framework append a partial user turn with no
    <|audio_end|> and no assistant generation prompt, and auto_response then
    asks the model to continue it. Observed live: only the first second of a
    4 s utterance reached the model and it emitted ' 1000000...'.
    """
    buffer = Qwen3OmniPcmAppendBuffer()
    for i in range(4):
        assert (
            buffer.prepare_append(
                _pcm_payload(Qwen3OmniDuplexPolicy.CHUNK_SAMPLES),
                operation_id=f"op-{i}",
                chunk_period_ms=CHUNK_MS,
                allow_emit=True,
            )
            is None
        ), "no append may emit mid-turn"
    assert buffer.has_pending(), "audio is retained for the commit"


def test_buffer_rollback_restores_audio_in_wire_order() -> None:
    buffer = Qwen3OmniPcmAppendBuffer()
    buffer.prepare_append(
        _pcm_payload(Qwen3OmniDuplexPolicy.CHUNK_SAMPLES),
        operation_id="op-0",
        chunk_period_ms=CHUNK_MS,
        allow_emit=True,
    )
    reservation = buffer.prepare_commit(operation_id="op-1", chunk_period_ms=CHUNK_MS)
    assert reservation is not None
    assert buffer.pending_byte_count == 0
    reservation.rollback()
    assert buffer.pending_byte_count == Qwen3OmniDuplexPolicy.CHUNK_SAMPLES * 4
    assert not buffer.has_reserved()


def test_buffer_commit_consumes_audio() -> None:
    buffer = Qwen3OmniPcmAppendBuffer()
    buffer.prepare_append(
        _pcm_payload(Qwen3OmniDuplexPolicy.CHUNK_SAMPLES),
        operation_id="op-0",
        chunk_period_ms=CHUNK_MS,
        allow_emit=True,
    )
    reservation = buffer.prepare_commit(operation_id="op-1", chunk_period_ms=CHUNK_MS)
    assert reservation is not None
    reservation.commit()
    assert buffer.pending_byte_count == 0
    assert not buffer.has_reserved()


def test_prepare_commit_pads_partial_chunk() -> None:
    buffer = Qwen3OmniPcmAppendBuffer()
    buffer.prepare_append(
        _pcm_payload(10),
        operation_id="op-1",
        chunk_period_ms=CHUNK_MS,
        allow_emit=False,
    )
    reservation = buffer.prepare_commit(operation_id="op-2", chunk_period_ms=CHUNK_MS)
    assert reservation.payload is not None
    assert reservation.payload["num_samples"] == Qwen3OmniDuplexPolicy.CHUNK_SAMPLES


def test_buffer_rejects_unsupported_format() -> None:
    buffer = Qwen3OmniPcmAppendBuffer()
    with pytest.raises(ValueError, match="unsupported duplex audio format"):
        buffer.prepare_append(
            {"audio": "", "format": "pcm_s16le"},
            operation_id="op-1",
            chunk_period_ms=CHUNK_MS,
            allow_emit=True,
        )


# --- serving policy --------------------------------------------------------


def test_private_runtime_config_keys_are_rejected_from_clients() -> None:
    adapter = Qwen3OmniServingRuntimeAdapter(lambda *_args: "")
    with pytest.raises(ValueError, match="server-owned"):
        adapter.validate_client_extra_body({"duplex_stage_max_tokens": {"0": 1}})


def test_capabilities_do_not_claim_model_native_turn_policy() -> None:
    """Qwen3-Omni has no listen/speak token; claiming otherwise misleads clients."""
    adapter = Qwen3OmniServingRuntimeAdapter(lambda *_args: "")
    capabilities = adapter.capabilities(max_sessions=1)
    assert capabilities.supports_model_native_turn_policy is False
    assert capabilities.chunk_period_ms == Qwen3OmniDuplexPolicy.CHUNK_PERIOD_MS


def test_runtime_config_update_preserves_server_owned_state() -> None:
    adapter = Qwen3OmniServingRuntimeAdapter(lambda *_args: "")
    current = {
        "duplex_chunk_period_ms": 1000,
        "duplex_stage_max_tokens": {"0": 64, "1": 8192},
    }

    class _Config:
        instructions = "be brief"
        max_tokens = 99
        temperature = None

    updated = adapter.runtime_config_for_update(_Config(), current)
    assert updated["duplex_chunk_period_ms"] == 1000
    assert updated["duplex_stage_max_tokens"]["0"] == 99
    assert updated["duplex_stage_max_tokens"]["1"] == 8192
    assert updated["instructions"] == "be brief"
    # The caller's mapping must not be mutated in place.
    assert current["duplex_stage_max_tokens"]["0"] == 64


def test_session_state_satisfies_committed_audio_contract() -> None:
    adapter = Qwen3OmniServingRuntimeAdapter(lambda *_args: "")
    state = adapter.session_state("sess-1")
    state.retain_committed_audio({"audio": ""}, operation_id="op-1", reserved_bytes=128)
    assert state.committed_audio_reserved_bytes == 128
    assert state.clear_committed_audio() == 128
    assert state.committed_audio_payload is None
    assert adapter.session_state("sess-1") is state
    adapter.remove_session_state("sess-1")
    assert adapter.session_state("sess-1") is not state


# --- stage-0 boundary ------------------------------------------------------


def test_stage0_fails_loudly_without_a_usable_thinker_stage() -> None:
    """Never silently no-ops: a bad stage raises rather than dropping audio."""
    runtime = Qwen3OmniStage0DuplexRuntime(object())
    with pytest.raises(RuntimeError, match="Qwen3-Omni duplex stage 0"):
        runtime.build_append_embeddings(
            duplex={"session_id": "s", "incarnation": 0, "payload": _pcm_payload(Qwen3OmniDuplexPolicy.CHUNK_SAMPLES)},
            token_offset=0,
            prompt_len=13,
        )


# --- conversation scaffolding ----------------------------------------------

_SCAFFOLD = {
    Qwen3OmniDuplexPolicy.SESSION_PREFIX_IDS_KEY: [1, 2, 3],
    Qwen3OmniDuplexPolicy.TURN_PREFIX_IDS_KEY: [4, 5],
    Qwen3OmniDuplexPolicy.TURN_SUFFIX_IDS_KEY: [6, 7, 8],
    Qwen3OmniDuplexPolicy.NEWLINE_IDS_KEY: [9],
}
#: A later turn opens by closing the assistant's previous turn.
_CLOSE_PREV = [Qwen3OmniDuplexPolicy.IM_END_TOKEN_ID, 9]
_PAD = Qwen3OmniDuplexPolicy.AUDIO_PAD_TOKEN_ID


def _prompt(*, seq: int, turn_seq: int, final: bool, samples: int = Qwen3OmniDuplexPolicy.CHUNK_SAMPLES):
    return build_duplex_prompt_token_ids(
        runtime_config=dict(_SCAFFOLD),
        payload=_pcm_payload(samples),
        seq=seq,
        turn_seq=turn_seq,
        final=final,
    )


def test_session_opener_carries_system_block_and_user_turn() -> None:
    ids, audio_offset, audio_tokens = _prompt(seq=1, turn_seq=1, final=False)
    assert ids[:5] == [1, 2, 3, 4, 5], "session prefix then turn prefix"
    assert ids[5] == Qwen3OmniDuplexPolicy.AUDIO_START_TOKEN_ID
    assert audio_offset == 6
    assert audio_tokens == 13
    assert ids[6:] == [_PAD] * 13


def test_mid_turn_append_has_no_scaffolding() -> None:
    """Only audio: the turn is already open and not yet closed."""
    ids, audio_offset, audio_tokens = _prompt(seq=2, turn_seq=2, final=False)
    assert audio_offset == 0
    assert ids == [_PAD] * audio_tokens


def test_later_turn_reopens_user_without_repeating_system_block() -> None:
    ids, audio_offset, _ = _prompt(seq=9, turn_seq=1, final=False)
    assert ids[:2] == _CLOSE_PREV, "must close the assistant's previous turn"
    assert ids[2:4] == [4, 5]
    assert ids[4] == Qwen3OmniDuplexPolicy.AUDIO_START_TOKEN_ID
    assert audio_offset == 5, "system block belongs to the session, not each turn"


def test_final_append_closes_user_and_opens_assistant() -> None:
    """This suffix is what actually prompts a reply.

    A closing append carries the whole turn, so it is framed on both sides:
    user opener + <|audio_start|> ... <|audio_end|> + assistant opener.
    """
    ids, audio_offset, audio_tokens = _prompt(seq=3, turn_seq=3, final=True)
    assert ids[-3:] == [6, 7, 8]
    assert ids == _CLOSE_PREV + [4, 5, Qwen3OmniDuplexPolicy.AUDIO_START_TOKEN_ID] + [_PAD] * audio_tokens + [
        Qwen3OmniDuplexPolicy.AUDIO_END_TOKEN_ID,
        6,
        7,
        8,
    ]
    assert audio_offset == 5


def test_prompt_length_equals_reservation() -> None:
    """Budget must equal produced embeddings; the runner truncates silently."""
    for seq, turn_seq, final in ((1, 1, False), (2, 2, False), (3, 3, True), (9, 1, True)):
        ids, audio_offset, audio_tokens = _prompt(seq=seq, turn_seq=turn_seq, final=final)
        assert audio_offset + audio_tokens <= len(ids)
        assert ids.count(_PAD) == audio_tokens, "audio span is exactly the pad tokens"
        trailing = len(ids) - audio_offset - audio_tokens
        assert trailing == (4 if final else 0), "audio_end + turn suffix when closing"
        if seq > 1 and audio_offset >= 2:
            assert ids[:2] == _CLOSE_PREV, "later turns close the assistant first"


def test_audio_span_is_contiguous_and_located_by_audio_offset() -> None:
    ids, audio_offset, audio_tokens = _prompt(seq=1, turn_seq=1, final=True)
    assert ids[audio_offset : audio_offset + audio_tokens] == [_PAD] * audio_tokens
    assert _PAD not in ids[:audio_offset]
    assert _PAD not in ids[audio_offset + audio_tokens :]


def test_missing_scaffolding_degrades_to_audio_only() -> None:
    """No tokenizer at config time must not crash the append path."""
    ids, audio_offset, audio_tokens = build_duplex_prompt_token_ids(
        runtime_config={},
        payload=_pcm_payload(Qwen3OmniDuplexPolicy.CHUNK_SAMPLES),
        seq=1,
        turn_seq=1,
        final=True,
    )
    # Delimiters are model constants, not scaffolding, so they survive.
    assert audio_offset == 1
    assert ids == [Qwen3OmniDuplexPolicy.AUDIO_START_TOKEN_ID] + [_PAD] * audio_tokens + [
        Qwen3OmniDuplexPolicy.AUDIO_END_TOKEN_ID
    ]


def test_commit_payload_closes_the_turn_without_the_final_flag() -> None:
    """The framework never passes final=True; the commit rides on the payload.

    session_runner.py:1184,1434 hard-code final=False because MiniCPM decides
    listen/speak natively. Qwen3-Omni needs the assistant generation prompt,
    so a commit is signalled on the payload instead.
    """
    payload = _pcm_payload(Qwen3OmniDuplexPolicy.CHUNK_SAMPLES)
    payload[Qwen3OmniDuplexPolicy.TURN_FINAL_KEY] = True
    ids, _, audio_tokens = build_duplex_prompt_token_ids(
        runtime_config=dict(_SCAFFOLD), payload=payload, seq=2, turn_seq=2, final=False
    )
    assert ids[-3:] == [6, 7, 8], "assistant generation prompt must be appended"
    assert ids == _CLOSE_PREV + [4, 5, Qwen3OmniDuplexPolicy.AUDIO_START_TOKEN_ID] + [_PAD] * audio_tokens + [
        Qwen3OmniDuplexPolicy.AUDIO_END_TOKEN_ID,
        6,
        7,
        8,
    ], "a later turn is framed too, not just the first"


def test_prepare_commit_marks_the_payload_turn_final() -> None:
    buffer = Qwen3OmniPcmAppendBuffer()
    buffer.prepare_append(
        _pcm_payload(Qwen3OmniDuplexPolicy.CHUNK_SAMPLES),
        operation_id="o1",
        chunk_period_ms=CHUNK_MS,
        allow_emit=False,
    )
    reservation = buffer.prepare_commit(operation_id="o2", chunk_period_ms=CHUNK_MS)
    assert reservation.payload is not None
    assert reservation.payload[Qwen3OmniDuplexPolicy.TURN_FINAL_KEY] is True


def test_plain_append_yields_no_reservation() -> None:
    """Only a commit produces a payload, and it is always turn-final."""
    buffer = Qwen3OmniPcmAppendBuffer()
    assert (
        buffer.prepare_append(
            _pcm_payload(Qwen3OmniDuplexPolicy.CHUNK_SAMPLES),
            operation_id="o1",
            chunk_period_ms=CHUNK_MS,
            allow_emit=True,
        )
        is None
    )


def test_sample_count_measured_from_audio_not_a_stale_key() -> None:
    """_merge_native_audio_payloads rebuilds `audio` but copies num_samples.

    A concatenated payload therefore carries the tail's num_samples. Trusting
    it would under-reserve slots and the runner would truncate silently.
    """
    payload = _pcm_payload(Qwen3OmniDuplexPolicy.CHUNK_SAMPLES * 3)
    payload["num_samples"] = Qwen3OmniDuplexPolicy.CHUNK_SAMPLES  # stale, as after a merge
    assert duplex_audio_token_count(payload) == Qwen3OmniDuplexPolicy.audio_tokens_for_samples(
        Qwen3OmniDuplexPolicy.CHUNK_SAMPLES * 3
    )


def test_data_plane_reads_omni_request_output_objects() -> None:
    """collect_registered_outputs yields objects, not dicts.

    Guarding projection on isinstance(..., Mapping) silently discards every
    stage output, which presents as "the model never replies" with no error
    anywhere.
    """
    from types import SimpleNamespace

    from vllm_omni.experimental.fullduplex.qwen3omni.data_plane import (
        Qwen3OmniDataPlaneContext,
        Qwen3OmniDataPlaneSession,
    )

    completion = SimpleNamespace(text="hello there")
    output = SimpleNamespace(
        request_id="r1",
        finished=False,
        stage_id=0,
        request_output=SimpleNamespace(outputs=[completion]),
        multimodal_output={},
    )
    dp = Qwen3OmniDataPlaneSession(lambda *_a: "enc")
    dp.begin_request("r1")
    events = list(dp.project({"data_plane_outputs": [output]}, context=Qwen3OmniDataPlaneContext()))
    assert events, "object-shaped outputs must not be dropped"
    assert events[0]["text"] == "hello there"
    assert events[0]["stage_role"] == "llm"


# --- stop condition and transcript ------------------------------------------


def test_stage0_sampling_stops_on_im_end() -> None:
    """Without an EOS stop token the thinker runs to max_tokens.

    Observed live: a 4 s input produced ~385 s of audio because generation
    only ended at the cap. Every extra token becomes synthesized speech.
    """
    import asyncio

    adapter = Qwen3OmniServingRuntimeAdapter(lambda *_a: "")

    class _Config:
        instructions = None
        max_tokens = None
        temperature = 0.7
        extra_body: dict = {}

    cfg = asyncio.run(adapter.prepare_runtime_config(_Config(), model_config=None))
    stage0 = cfg["duplex_stage_sampling_params"]["0"]
    assert Qwen3OmniDuplexPolicy.IM_END_TOKEN_ID in stage0["stop_token_ids"]
    assert cfg["duplex_stage_max_tokens"]["0"] <= 256, "a spoken turn should be short"


def test_runtime_config_update_keeps_the_stop_token() -> None:
    adapter = Qwen3OmniServingRuntimeAdapter(lambda *_a: "")

    class _Config:
        instructions = None
        max_tokens = None
        temperature = 0.5

    updated = adapter.runtime_config_for_update(_Config(), {})
    assert Qwen3OmniDuplexPolicy.IM_END_TOKEN_ID in updated["duplex_stage_sampling_params"]["0"]["stop_token_ids"]


def test_decide_output_returns_none_so_audio_still_flows() -> None:
    """A direct response would surface text but kill the audio.

    orchestrator.py:1284-1295 returns immediately after emitting a direct
    response, skipping _forward_to_next_stage. Verified live: marking thinker
    output as a direct response delivered the transcript and zero audio,
    because stage 2 never ran.
    """
    from types import SimpleNamespace

    ext = Qwen3OmniDuplexRuntimeExtension()
    output = SimpleNamespace(outputs=[SimpleNamespace(text="hello there")])
    for stage_id in (0, 1, 2):
        assert (
            ext.decide_output(
                stage_id=stage_id,
                final_stage_id=2,
                segment_finished=True,
                segment_token_ids=(1, 2),
                segment_output_metadata={},
                output=output,
            )
            is None
        )


def test_data_plane_reads_text_from_raw_request_output() -> None:
    """Stage 0 sends a raw vllm RequestOutput, not a wrapped OmniRequestOutput."""
    from types import SimpleNamespace

    from vllm_omni.experimental.fullduplex.qwen3omni.data_plane import (
        Qwen3OmniDataPlaneContext,
        Qwen3OmniDataPlaneSession,
    )

    raw = SimpleNamespace(
        request_id="r1",
        finished=False,
        stage_id=0,
        outputs=[SimpleNamespace(text="unwrapped text")],
        multimodal_output={},
    )
    dp = Qwen3OmniDataPlaneSession(lambda *_a: "enc")
    dp.begin_request("r1")
    events = list(dp.project({"data_plane_outputs": [raw]}, context=Qwen3OmniDataPlaneContext()))
    assert events and events[0]["text"] == "unwrapped text"


def test_audio_span_is_delimited_for_the_model() -> None:
    """Missing <|audio_start|>/<|audio_end|> makes the thinker emit garbage.

    Observed live without them: ' a i \\n\\n\\nuser\\n\\n\\nuser\\n...' -- the model
    does not read the embeddings as audio and leaks chat role markers, which
    the talker then synthesizes.
    """
    ids, audio_offset, audio_tokens = _prompt(seq=1, turn_seq=1, final=True)
    assert ids[audio_offset - 1] == Qwen3OmniDuplexPolicy.AUDIO_START_TOKEN_ID
    assert ids[audio_offset + audio_tokens] == Qwen3OmniDuplexPolicy.AUDIO_END_TOKEN_ID
    assert ids[audio_offset : audio_offset + audio_tokens] == [_PAD] * audio_tokens


def test_every_turn_is_framed_not_just_the_first() -> None:
    """turn_seq counts turns, not appends within a turn.

    Gating the opener on turn_seq <= 1 framed only the session's first turn;
    later turns arrived as bare audio with no <|im_start|>user and no
    <|audio_start|>, and produced nothing usable. Observed live as
    prefix=0 on turns 2 and 3.
    """
    for turn_seq in (1, 2, 5):
        ids, audio_offset, audio_tokens = build_duplex_prompt_token_ids(
            runtime_config=dict(_SCAFFOLD),
            payload={**_pcm_payload(Qwen3OmniDuplexPolicy.CHUNK_SAMPLES), Qwen3OmniDuplexPolicy.TURN_FINAL_KEY: True},
            seq=turn_seq,
            turn_seq=turn_seq,
            final=False,
        )
        assert 4 in ids and 5 in ids, f"turn {turn_seq} lost its user opener"
        assert ids[audio_offset - 1] == Qwen3OmniDuplexPolicy.AUDIO_START_TOKEN_ID
        assert ids[audio_offset + audio_tokens] == Qwen3OmniDuplexPolicy.AUDIO_END_TOKEN_ID
        # Only the first turn of the session repeats the system block.
        assert (1 in ids) == (turn_seq == 1)


def test_cumulative_audio_is_sliced_to_the_new_tail() -> None:
    """Code2Wav emits the whole waveform each time; only the tail is new.

    Forwarding each output in full makes the client replay the reply from the
    start repeatedly -- heard as "I. I. I. I'm doing great". Measured delta
    sizes for one reply: 14250, 110250, 206250, 217770 bytes.
    """
    from types import SimpleNamespace

    from vllm_omni.experimental.fullduplex.qwen3omni.data_plane import (
        Qwen3OmniDataPlaneContext,
        Qwen3OmniDataPlaneSession,
    )

    seen: list[int] = []

    def encode(audio, sr, fmt, speed):
        seen.append(len(audio))
        return "x" * len(audio)

    dp = Qwen3OmniDataPlaneSession(encode)
    dp.begin_request("r1")
    ctx = Qwen3OmniDataPlaneContext(modalities=("audio",))

    # Cumulative: 100 samples, then 250, then 400.
    for total in (100, 250, 400):
        out = SimpleNamespace(
            request_id="r1",
            finished=False,
            stage_id=2,
            multimodal_output={"audio": list(range(total)), "sr": 24000},
        )
        list(dp.project({"data_plane_outputs": [out]}, context=ctx))

    assert seen == [100, 150, 150], f"expected new tails only, got {seen}"


def test_repeated_identical_audio_emits_nothing_new() -> None:
    from types import SimpleNamespace

    from vllm_omni.experimental.fullduplex.qwen3omni.data_plane import (
        Qwen3OmniDataPlaneContext,
        Qwen3OmniDataPlaneSession,
    )

    dp = Qwen3OmniDataPlaneSession(lambda a, *_: "y" * len(a))
    dp.begin_request("r1")
    ctx = Qwen3OmniDataPlaneContext(modalities=("audio",))
    out = SimpleNamespace(
        request_id="r1",
        finished=False,
        stage_id=2,
        multimodal_output={"audio": list(range(64)), "sr": 24000},
    )
    first = list(dp.project({"data_plane_outputs": [out]}, context=ctx))
    second = list(dp.project({"data_plane_outputs": [out]}, context=ctx))
    assert first, "first delta must be emitted"
    assert second == [], "an unchanged cumulative buffer has no new audio"


def test_audio_length_uses_the_sample_axis_not_the_batch_axis() -> None:
    """Code2Wav returns [1, samples]; len() would give the batch dim.

    Reading len() made every cumulative output look like length 1, so only the
    first delta was ever emitted and the reply was truncated to 0.30 s.
    """
    from vllm_omni.experimental.fullduplex.qwen3omni.data_plane import (
        _audio_length,
        _audio_tail,
    )

    class _T:
        def __init__(self, n):
            self.shape = (1, n)
            self._n = n

        def __len__(self):
            return 1

        def __getitem__(self, key):
            assert isinstance(key, tuple) and key[0] is Ellipsis, "must slice the sample axis"
            return _T(self._n - (key[1].start or 0))

    assert _audio_length(_T(7125)) == 7125
    assert _audio_length([0] * 40) == 40
    assert _audio_tail(_T(55125), 7125).shape == (1, 48000)


def test_audio_cursor_survives_a_turn_boundary() -> None:
    """One request id spans the session; Code2Wav accumulates across it.

    Resetting the cursor per turn made each reply resend every previously
    spoken sample before the new audio, so replies grew by repeating the whole
    conversation.
    """
    from types import SimpleNamespace

    from vllm_omni.experimental.fullduplex.qwen3omni.data_plane import (
        Qwen3OmniDataPlaneContext,
        Qwen3OmniDataPlaneSession,
    )

    seen: list[int] = []
    dp = Qwen3OmniDataPlaneSession(lambda a, *_: (seen.append(len(a)), "x" * len(a))[1])
    ctx = Qwen3OmniDataPlaneContext(modalities=("audio",))

    def turn(total: int) -> None:
        out = SimpleNamespace(
            request_id="r1",
            finished=False,
            stage_id=2,
            multimodal_output={"audio": list(range(total)), "sr": 24000},
        )
        list(dp.project({"data_plane_outputs": [out]}, context=ctx))

    dp.begin_request("r1")
    turn(100)
    dp.begin_request("r1")  # next turn, same request id
    turn(260)

    assert seen == [100, 160], f"turn 2 must send only its own 160 samples, got {seen}"


# --- text turns -------------------------------------------------------------
#
# A text turn is the same chat turn as an audio one with real token ids where
# the audio span would be. Two things must hold or the thinker reads the turn
# as empty and answers as if the user said nothing:
#   * zero `<|audio_pad|>` slots are reserved, since stage 0 fills none;
#   * no `<|audio_start|>` / `<|audio_end|>`, which tell the thinker to expect
#     an audio span and make it treat the words as a transcription task.


def _text_prompt(*, seq: int, turn_seq: int, final: bool = True, text: str = "hello there", ids=(41, 42)):
    import vllm_omni.experimental.fullduplex.qwen3omni.runtime as runtime_mod

    original = runtime_mod._encode_text
    runtime_mod._encode_text = lambda value: list(ids) if value == text else []
    try:
        return build_duplex_prompt_token_ids(
            runtime_config=dict(_SCAFFOLD),
            payload=text,
            seq=seq,
            turn_seq=turn_seq,
            final=final,
        )
    finally:
        runtime_mod._encode_text = original


def test_text_turn_reserves_no_audio_slots() -> None:
    ids, _audio_offset, audio_tokens = _text_prompt(seq=1, turn_seq=1)
    assert audio_tokens == 0
    assert _PAD not in ids


def test_text_turn_omits_the_audio_delimiters() -> None:
    ids, _audio_offset, _audio_tokens = _text_prompt(seq=1, turn_seq=1)
    assert Qwen3OmniDuplexPolicy.AUDIO_START_TOKEN_ID not in ids
    assert Qwen3OmniDuplexPolicy.AUDIO_END_TOKEN_ID not in ids


def test_text_turn_is_a_well_formed_first_turn() -> None:
    ids, _audio_offset, _audio_tokens = _text_prompt(seq=1, turn_seq=1)
    # system block, user opener, the text itself, then close-user/open-assistant
    assert ids == [1, 2, 3, 4, 5, 41, 42, 6, 7, 8]


def test_later_text_turn_closes_the_assistant_turn_first() -> None:
    ids, _audio_offset, _audio_tokens = _text_prompt(seq=2, turn_seq=2)
    assert ids == [*_CLOSE_PREV, 4, 5, 41, 42, 6, 7, 8]


def test_text_turn_without_a_tokenizer_is_refused() -> None:
    """Encoding to nothing must fail loudly.

    Sending an empty user turn would prompt the model to answer a question the
    user never asked, which reads to them as being ignored -- the same failure
    mode as the unfilled-placeholder bug.
    """
    with pytest.raises(ValueError, match="tokenizer"):
        _text_prompt(seq=1, turn_seq=1, ids=())


def test_audio_turn_is_unchanged_by_the_text_path() -> None:
    ids, audio_offset, audio_tokens = _prompt(seq=1, turn_seq=1, final=False)
    assert audio_tokens == 13
    assert ids[5] == Qwen3OmniDuplexPolicy.AUDIO_START_TOKEN_ID
    assert audio_offset == 6


# --- tool calls -------------------------------------------------------------
#
# Stage 0 is one resumable request for the whole session, so its text
# accumulates across every turn. Re-reporting a call that was already
# dispatched is self-sustaining -- the result is itself a turn, whose reply
# still carries the original call -- and it ran away for real: 1204 turns and
# 105 s of audio repeating one sentence.


class _Completion:
    def __init__(self, text: str) -> None:
        self.text = text


class _StageOutput:
    def __init__(self, text: str, request_id: str = "req-1") -> None:
        self.request_id = request_id
        self.outputs = [_Completion(text)]


_CALL = '<tool_call>\n{"name": "lookup_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
_CALL2 = '<tool_call>\n{"name": "check_inventory", "arguments": {"item": "spoons"}}\n</tool_call>'


def test_parse_tool_calls_reads_the_template_format() -> None:
    assert parse_tool_calls(_CALL) == [{"name": "lookup_weather", "arguments": {"city": "Paris"}}]


def test_parse_tool_calls_ignores_an_unterminated_block() -> None:
    """The thinker streams; a call without its closing tag is still arriving."""
    assert parse_tool_calls('<tool_call>\n{"name": "lookup_weather"') == []


def test_a_tool_call_is_reported_once_even_as_text_accumulates() -> None:
    extension = Qwen3OmniDuplexRuntimeExtension()
    first = extension._new_tool_calls(_StageOutput(f"{_CALL}"))
    assert [call["name"] for call in first] == ["lookup_weather"]

    # Later turns append to the same cumulative text. The original call is
    # still in there and must not be dispatched again.
    again = extension._new_tool_calls(_StageOutput(f"{_CALL}It is 18 degrees in Paris."))
    assert again == []


def test_a_genuinely_new_call_is_still_reported() -> None:
    extension = Qwen3OmniDuplexRuntimeExtension()
    extension._new_tool_calls(_StageOutput(_CALL))
    later = extension._new_tool_calls(_StageOutput(f"{_CALL}{_CALL2}"))
    assert [call["name"] for call in later] == ["check_inventory"]


def test_a_turn_carrying_speech_is_not_a_tool_call() -> None:
    """Speaking wins when the turn has something to say.

    Claiming the turn produces a direct response, which suppresses its audio --
    right for a bare tool call, fatal when the model appends a redundant call to
    a real answer. Observed live: after a tool result the model replied "The
    weather in Barcelona is 18C with light rain." *and* re-emitted the same
    call. Intercepting that threw away the answer and dead-ended the
    conversation in silence, because the client rightly refuses to re-run a call
    it has already run.
    """
    extension = Qwen3OmniDuplexRuntimeExtension()
    mixed = extension._new_tool_calls(_StageOutput(f"The weather in Barcelona is 18C.{_CALL}"))
    assert mixed == []


def test_an_ignored_mixed_call_is_not_re_examined_later() -> None:
    """A call skipped for carrying speech must still advance the cursor."""
    extension = Qwen3OmniDuplexRuntimeExtension()
    extension._new_tool_calls(_StageOutput(f"Some spoken answer.{_CALL}"))
    again = extension._new_tool_calls(_StageOutput(f"Some spoken answer.{_CALL} and more"))
    assert again == []


def test_a_partially_streamed_call_is_not_skipped() -> None:
    """The cursor may only advance past a closed block.

    Advancing past a half-arrived call would consume its opening tag, so the
    closing tag would never match and the call would be lost silently.
    """
    extension = Qwen3OmniDuplexRuntimeExtension()
    assert extension._new_tool_calls(_StageOutput('<tool_call>\n{"name": "lookup_weather"')) == []
    complete = extension._new_tool_calls(_StageOutput(_CALL))
    assert [call["name"] for call in complete] == ["lookup_weather"]


def test_scan_state_is_per_request() -> None:
    extension = Qwen3OmniDuplexRuntimeExtension()
    extension._new_tool_calls(_StageOutput(_CALL, request_id="req-a"))
    other = extension._new_tool_calls(_StageOutput(_CALL, request_id="req-b"))
    assert [call["name"] for call in other] == ["lookup_weather"]


def test_a_reused_request_id_rescans_from_the_start() -> None:
    extension = Qwen3OmniDuplexRuntimeExtension()
    extension._new_tool_calls(_StageOutput(f"{_CALL}some more text here"))
    # Shorter text under the same id means a new request took the id over.
    fresh = extension._new_tool_calls(_StageOutput(_CALL))
    assert [call["name"] for call in fresh] == ["lookup_weather"]


def test_tool_decision_releases_the_prewarmed_downstream_stages() -> None:
    """The talker and code2wav were warmed for a reply that is not coming.

    Without the release flag they park waiting for chunks that never arrive,
    hold their slots, and make `has_work()` read false for later sessions --
    the same wedge as the streaming-input counter leak, reached from the
    direct-response path.
    """
    extension = Qwen3OmniDuplexRuntimeExtension()
    decision = extension.decide_output(
        stage_id=0,
        final_stage_id=2,
        segment_finished=True,
        segment_token_ids=(),
        segment_output_metadata={},
        output=_StageOutput(_CALL),
    )
    assert decision is not None
    assert decision.metadata["duplex_direct_response"] is True
    assert decision.metadata["duplex_release_downstream"] is True
    assert [call["name"] for call in decision.metadata["tool_calls"]] == ["lookup_weather"]


def test_a_plain_reply_still_reaches_the_talker() -> None:
    extension = Qwen3OmniDuplexRuntimeExtension()
    assert (
        extension.decide_output(
            stage_id=0,
            final_stage_id=2,
            segment_finished=True,
            segment_token_ids=(),
            segment_output_metadata={},
            output=_StageOutput("The capital of France is Paris."),
        )
        is None
    )
