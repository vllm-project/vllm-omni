# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from vllm_omni.model_executor.stage_input_processors import raon as stage_proc


@dataclass
class _DummyChoice:
    multimodal_output: dict
    finish_reason: str | None = None


@dataclass
class _DummyReqOutput:
    request_id: str
    outputs: list[_DummyChoice]


@dataclass
class _DummyStage:
    engine_outputs: list[_DummyReqOutput]


def _build_stage(req_id: str, codes, *, finish_reason: str | None, chunk_key: str = "codec_codes") -> list[_DummyStage]:
    return [
        _DummyStage(
            engine_outputs=[
                _DummyReqOutput(
                    request_id=req_id,
                    outputs=[
                        _DummyChoice(
                            multimodal_output={chunk_key: codes},
                            finish_reason=finish_reason,
                        )
                    ],
                )
            ]
        )
    ]


def test_stage0_to_stage1_no_prompt_before_finish():
    """Stage-1 only emits a prompt once Stage-0 has finished the request."""
    stage_list = _build_stage("req-a", [[1, 2, 3], [4, 5, 6]], finish_reason=None)
    prompts = stage_proc.stage0_to_stage1(stage_list=stage_list, engine_input_source=[0])
    assert prompts == []


def test_stage0_to_stage1_emits_on_finish_with_full_payload():
    """On finish, the full `codec_codes` payload is flattened into the prompt."""
    stage_list = _build_stage("req-a", [[1, 2, 3], [4, 5, 6], [7, 8, 9]], finish_reason="stop")
    prompts = stage_proc.stage0_to_stage1(stage_list=stage_list, engine_input_source=[0])
    assert len(prompts) == 1
    assert prompts[0]["prompt_token_ids"] == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_stage0_to_stage1_accepts_runtime_source_outputs_signature():
    """The orchestrator calls custom processors with source outputs directly."""
    stage_list = _build_stage("req-runtime", [[1, 2, 3], [4, 5, 6]], finish_reason="stop")
    source_outputs = stage_list[0].engine_outputs

    prompts = stage_proc.stage0_to_stage1(
        source_outputs,
        {"additional_information": {"continuation_silence_frames": 2}},
        False,
    )

    assert len(prompts) == 1
    assert prompts[0]["prompt_token_ids"] == [1, 2, 3, 4, 5, 6]
    assert prompts[0]["additional_information"]["continuation_silence_frames"] == 2


def test_stage0_to_stage1_does_not_treat_runtime_token_prompt_as_legacy_source():
    """Runtime source-output calls may pass token-id prompts as list[int]."""
    stage_list = _build_stage("req-token-prompt", [[1, 2, 3], [4, 5, 6]], finish_reason="stop")
    source_outputs = stage_list[0].engine_outputs

    prompts = stage_proc.stage0_to_stage1(source_outputs, [99], False)

    assert len(prompts) == 1
    assert prompts[0]["prompt_token_ids"] == [1, 2, 3, 4, 5, 6]


def test_stage0_to_stage1_falls_back_to_chunk_when_full_absent():
    """When only `codec_codes_chunk` is present on finish, use it as the payload."""
    stage_list = _build_stage(
        "req-b", [[10, 11, 12], [13, 14, 15]], finish_reason="stop", chunk_key="codec_codes_chunk"
    )
    prompts = stage_proc.stage0_to_stage1(stage_list=stage_list, engine_input_source=[0])
    assert len(prompts) == 1
    assert prompts[0]["prompt_token_ids"] == [10, 11, 12, 13, 14, 15]


def test_stage0_to_stage1_handles_flat_1d_payload():
    """1D flat payloads are normalized to [T, G] before flattening."""
    stage_list = _build_stage("req-c", [1, 2, 3, 4, 5, 6], finish_reason="stop")
    prompts = stage_proc.stage0_to_stage1(stage_list=stage_list, engine_input_source=[0])
    assert len(prompts) == 1
    assert prompts[0]["prompt_token_ids"] == [1, 2, 3, 4, 5, 6]


def test_stage0_to_stage1_prefers_full_payload_over_chunk():
    """When both `codec_codes` and `codec_codes_chunk` are present, full wins.

    Sync path always receives the completed request output, so `codec_codes`
    (full) is authoritative and `codec_codes_chunk` is only used as fallback.
    """
    stage_list = [
        _DummyStage(
            engine_outputs=[
                _DummyReqOutput(
                    request_id="req-d",
                    outputs=[
                        _DummyChoice(
                            multimodal_output={
                                "codec_codes": [[1, 2, 3], [4, 5, 6]],
                                "codec_codes_chunk": [[7, 8, 9]],
                            },
                            finish_reason="stop",
                        )
                    ],
                )
            ]
        )
    ]
    prompts = stage_proc.stage0_to_stage1(stage_list=stage_list, engine_input_source=[0])
    assert len(prompts) == 1
    assert prompts[0]["prompt_token_ids"] == [1, 2, 3, 4, 5, 6]


def test_stage0_to_stage1_skips_empty_mm_output():
    """If the finished output has no codec payload, no prompt is emitted."""
    stage_list = [
        _DummyStage(
            engine_outputs=[
                _DummyReqOutput(
                    request_id="req-e",
                    outputs=[_DummyChoice(multimodal_output={}, finish_reason="stop")],
                )
            ]
        )
    ]
    prompts = stage_proc.stage0_to_stage1(stage_list=stage_list, engine_input_source=[0])
    assert prompts == []


def test_stage0_to_stage1_no_module_global_state():
    """Regression: the module holds no per-request buffer state.

    Prior to the stateless refactor a module-global `_REQUEST_CODEC_CHUNKS`
    defaultdict could accumulate entries that were never cleaned up on
    abort/error. After the refactor those symbols no longer exist.
    """
    assert not hasattr(stage_proc, "_REQUEST_CODEC_CHUNKS")
    assert not hasattr(stage_proc, "_append_request_chunk")
