# SPDX-License-Identifier: Apache-2.0
"""Unit tests for speech token-usage accounting (issue #4646).

These exercise the pure usage logic without a model/server: input-token
breakdown (text + reference-audio gating) and output-token accumulation.
"""

from dataclasses import dataclass

import pytest

from vllm_omni.entrypoints.openai.speech_usage import (
    SpeechOutputTokenCounter,
    build_speech_usage,
    gate_audio_tokens,
    qwen3_tts_input_token_details,
)


# A deterministic stand-in for a real tokenizer: 1 token per whitespace word.
def word_count(text: str) -> int:
    return len(text.split())


# --- input breakdown: gating ------------------------------------------------


def test_gate_audio_tokens_base_in_context_counts_ref_frames():
    assert (
        gate_audio_tokens(task_type="Base", x_vector_only_mode=False, icl_mode_override=None, ref_code_length=120)
        == 120
    )


def test_gate_audio_tokens_customvoice_is_zero():
    assert (
        gate_audio_tokens(
            task_type="CustomVoice", x_vector_only_mode=False, icl_mode_override=None, ref_code_length=120
        )
        == 0
    )


def test_gate_audio_tokens_x_vector_only_is_zero_even_with_ref():
    # x-vector cloning inserts NO codec frames, so audio_tokens must be 0
    # even though ref_code_length is populated.
    assert (
        gate_audio_tokens(task_type="Base", x_vector_only_mode=True, icl_mode_override=None, ref_code_length=120) == 0
    )


def test_gate_audio_tokens_icl_override_wins():
    assert (
        gate_audio_tokens(task_type="Base", x_vector_only_mode=False, icl_mode_override=False, ref_code_length=120) == 0
    )
    assert (
        gate_audio_tokens(task_type="Base", x_vector_only_mode=True, icl_mode_override=True, ref_code_length=80) == 80
    )


def test_gate_audio_tokens_icl_override_ignored_for_non_base():
    # icl_mode_override is honored only for the Base task; CustomVoice stays 0
    # even if such a flag leaks into its params (mirrors the estimator).
    assert (
        gate_audio_tokens(
            task_type="CustomVoice", x_vector_only_mode=False, icl_mode_override=True, ref_code_length=120
        )
        == 0
    )


def test_gate_audio_tokens_bad_ref_length_is_zero():
    assert (
        gate_audio_tokens(task_type="Base", x_vector_only_mode=False, icl_mode_override=None, ref_code_length="oops")
        == 0
    )


# --- input breakdown: Qwen3-TTS reproduces issue #4646 observation ----------


def test_customvoice_text_scales_with_input_no_audio():
    short = qwen3_tts_input_token_details(
        input_text="hi there",
        instructions=None,
        tts_params={"task_type": ["CustomVoice"]},
        count_text_tokens=word_count,
    )
    long = qwen3_tts_input_token_details(
        input_text="hi there this is a much longer sentence",
        instructions=None,
        tts_params={"task_type": ["CustomVoice"]},
        count_text_tokens=word_count,
    )
    assert short.audio_tokens == 0 and long.audio_tokens == 0
    assert long.text_tokens > short.text_tokens  # scales with input


def test_base_icl_text_tracks_input_audio_tracks_ref():
    # The core #4646 fix: text_tokens reflect the input (not dropped), and
    # audio_tokens track the reference audio independently.
    params = {"task_type": ["Base"], "x_vector_only_mode": [False], "ref_code_length": [100]}
    short = qwen3_tts_input_token_details(
        input_text="hi", instructions=None, tts_params=params, count_text_tokens=word_count
    )
    long = qwen3_tts_input_token_details(
        input_text="hi there friend", instructions=None, tts_params=params, count_text_tokens=word_count
    )
    # text now varies with input (the bug was that it was dropped for Base)
    assert long.text_tokens > short.text_tokens
    # audio fixed by the reference, independent of input text
    assert short.audio_tokens == long.audio_tokens == 100


def test_instructions_count_toward_text():
    without = qwen3_tts_input_token_details(
        input_text="hello world",
        instructions=None,
        tts_params={"task_type": ["CustomVoice"]},
        count_text_tokens=word_count,
    )
    with_instr = qwen3_tts_input_token_details(
        input_text="hello world",
        instructions="speak slowly and warmly",
        tts_params={"task_type": ["CustomVoice"]},
        count_text_tokens=word_count,
    )
    assert with_instr.text_tokens == without.text_tokens + word_count("speak slowly and warmly")


# --- build usage ------------------------------------------------------------


def test_build_speech_usage_aggregates_and_totals():
    details = qwen3_tts_input_token_details(
        input_text="hi there",
        instructions=None,
        tts_params={"task_type": ["Base"], "x_vector_only_mode": [False], "ref_code_length": [100]},
        count_text_tokens=word_count,
    )
    usage = build_speech_usage(details, output_tokens=250)
    assert usage.input_tokens == details.text_tokens + 100
    assert usage.output_tokens == 250
    assert usage.total_tokens == usage.input_tokens + 250
    assert usage.input_token_details.text_tokens == 2
    assert usage.input_token_details.audio_tokens == 100


def test_batch_item_result_serializes_usage():
    # Guards the batch per-item usage carrier (and that it is optional/None on error).
    from vllm_omni.entrypoints.openai.protocol.audio import SpeechBatchItemResult

    details = qwen3_tts_input_token_details(
        input_text="hello world",
        instructions=None,
        tts_params={"task_type": ["Base"], "x_vector_only_mode": [False], "ref_code_length": [50]},
        count_text_tokens=word_count,
    )
    item = SpeechBatchItemResult(
        index=0,
        status="success",
        audio_data="x",
        media_type="audio/wav",
        usage=build_speech_usage(details, output_tokens=30),
    )
    dumped = item.model_dump()
    assert dumped["usage"]["input_tokens"] == 2 + 50
    assert dumped["usage"]["output_tokens"] == 30
    assert dumped["usage"]["input_token_details"] == {"text_tokens": 2, "audio_tokens": 50}
    # Errored items carry no usage.
    assert SpeechBatchItemResult(index=1, status="error", error="boom").model_dump()["usage"] is None


# --- output token accumulation ----------------------------------------------
# The generated codec-token count is read from per-stage pipeline metrics on the
# final output: res.metrics["stage_metrics"][<stage>]["num_tokens_out"]. Only the
# AR stage (0) is non-zero; code2wav (stage 1) reports 0.


@dataclass
class _FakeRes:
    metrics: dict


def _final_res(stage0_out, stage1_out=0):
    return _FakeRes(
        metrics={
            "stage_metrics": {
                "0": {"num_tokens_in": 110, "num_tokens_out": stage0_out},
                "1": {"num_tokens_in": 0, "num_tokens_out": stage1_out},
            }
        }
    )


def test_output_tokens_from_stage_metrics():
    acc = SpeechOutputTokenCounter()
    # Intermediate outputs carry empty metrics -> contribute nothing.
    acc.observe(_FakeRes(metrics={}))
    acc.observe(_FakeRes(metrics={}))
    # Final output carries stage metrics; stage-0 num_tokens_out is the count.
    acc.observe(_final_res(stage0_out=77))
    assert acc.total() == 77


def test_output_tokens_takes_max_across_stages():
    acc = SpeechOutputTokenCounter()
    acc.observe(_final_res(stage0_out=42, stage1_out=0))
    assert acc.total() == 42


def test_output_tokens_zero_without_metrics():
    acc = SpeechOutputTokenCounter()
    acc.observe(_FakeRes(metrics={}))
    acc.observe(_FakeRes(metrics={"stage_metrics": {}}))
    assert acc.total() == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
