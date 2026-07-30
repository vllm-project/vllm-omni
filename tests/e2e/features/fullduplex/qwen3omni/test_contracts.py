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


def test_buffer_emits_only_whole_chunks() -> None:
    buffer = Qwen3OmniPcmAppendBuffer()
    half = Qwen3OmniDuplexPolicy.CHUNK_SAMPLES // 2
    assert (
        buffer.prepare_append(
            _pcm_payload(half),
            operation_id="op-1",
            chunk_period_ms=CHUNK_MS,
            allow_emit=True,
        )
        is None
    )
    reservation = buffer.prepare_append(
        _pcm_payload(half),
        operation_id="op-2",
        chunk_period_ms=CHUNK_MS,
        allow_emit=True,
    )
    assert reservation is not None
    assert reservation.payload is not None
    assert reservation.payload["num_samples"] == Qwen3OmniDuplexPolicy.CHUNK_SAMPLES


def test_buffer_rollback_restores_audio_in_wire_order() -> None:
    buffer = Qwen3OmniPcmAppendBuffer()
    reservation = buffer.prepare_append(
        _pcm_payload(Qwen3OmniDuplexPolicy.CHUNK_SAMPLES),
        operation_id="op-1",
        chunk_period_ms=CHUNK_MS,
        allow_emit=True,
    )
    assert reservation is not None
    assert buffer.pending_byte_count == 0
    reservation.rollback()
    assert buffer.pending_byte_count == Qwen3OmniDuplexPolicy.CHUNK_SAMPLES * 4
    assert not buffer.has_reserved()


def test_buffer_commit_consumes_audio() -> None:
    buffer = Qwen3OmniPcmAppendBuffer()
    reservation = buffer.prepare_append(
        _pcm_payload(Qwen3OmniDuplexPolicy.CHUNK_SAMPLES),
        operation_id="op-1",
        chunk_period_ms=CHUNK_MS,
        allow_emit=True,
    )
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
}
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
    assert audio_offset == 5
    assert audio_tokens == 13
    assert ids[5:] == [_PAD] * 13


def test_mid_turn_append_has_no_scaffolding() -> None:
    """Only audio: the turn is already open and not yet closed."""
    ids, audio_offset, audio_tokens = _prompt(seq=2, turn_seq=2, final=False)
    assert audio_offset == 0
    assert ids == [_PAD] * audio_tokens


def test_later_turn_reopens_user_without_repeating_system_block() -> None:
    ids, audio_offset, _ = _prompt(seq=9, turn_seq=1, final=False)
    assert ids[:2] == [4, 5]
    assert audio_offset == 2, "system block belongs to the session, not each turn"


def test_final_append_closes_user_and_opens_assistant() -> None:
    """This suffix is what actually prompts a reply."""
    ids, _, audio_tokens = _prompt(seq=3, turn_seq=3, final=True)
    assert ids[-3:] == [6, 7, 8]
    assert ids == [_PAD] * audio_tokens + [6, 7, 8]


def test_prompt_length_equals_reservation() -> None:
    """Budget must equal produced embeddings; the runner truncates silently."""
    for seq, turn_seq, final in ((1, 1, False), (2, 2, False), (3, 3, True), (9, 1, True)):
        ids, audio_offset, audio_tokens = _prompt(seq=seq, turn_seq=turn_seq, final=final)
        scaffold = len(ids) - audio_tokens
        assert audio_offset + audio_tokens <= len(ids)
        assert ids.count(_PAD) == audio_tokens, "audio span is exactly the pad tokens"
        assert scaffold == audio_offset + (3 if final else 0)


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
    assert audio_offset == 0
    assert ids == [_PAD] * audio_tokens
