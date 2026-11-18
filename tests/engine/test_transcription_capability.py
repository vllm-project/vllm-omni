# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-only capability checks with model-registry metadata stubbed."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_omni.config.transcription import (
    ASR_ALIGNER_INPUT_PROCESSOR,
    has_transcription_aligner,
    supports_transcription_model,
    supports_transcription_pipeline,
    validate_asr_aligner_stages,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _pipeline(*outputs, model_type="another_asr", supports_transcription=True):
    configs = [
        SimpleNamespace(
            stage_id=i,
            input_sources=() if i == 0 else (i - 1,),
            is_comprehension=True,
            custom_process_input_func=ASR_ALIGNER_INPUT_PROCESSOR if output == "timestamps" else None,
        )
        for i, output in enumerate(outputs)
    ]
    metadata = [
        SimpleNamespace(
            stage_type="llm",
            final_output=output is not None,
            final_output_type=output,
            model_stage="forced_aligner" if output == "timestamps" else "model",
        )
        for output in outputs
    ]
    runtime = [
        SimpleNamespace(
            model_config=SimpleNamespace(
                architectures=["SpeechModel"],
                hf_config=SimpleNamespace(model_type=model_type, architectures=["CheckpointArchitecture"]),
                registry=SimpleNamespace(
                    inspect_model_cls=Mock(
                        return_value=(
                            SimpleNamespace(supports_transcription=supports_transcription, is_pooling_model=False),
                            "SpeechModel",
                        )
                    )
                ),
                runner_type="pooling" if output == "timestamps" else "generate",
            )
        )
        for output in outputs
    ]
    return configs, metadata, runtime


@pytest.mark.parametrize("model_type", ["qwen3_asr", "whisper", "another_asr"])
def test_transcription_is_detected_by_model_capability(model_type):
    pipeline = _pipeline("text", model_type=model_type)
    assert supports_transcription_pipeline(*pipeline)
    model_config = pipeline[2][0].model_config
    model_config.registry.inspect_model_cls.assert_called_once_with(["SpeechModel"], model_config)


def test_checkpoint_name_does_not_override_missing_transcription_capability():
    assert not supports_transcription_pipeline(*_pipeline("text", model_type="qwen3_asr", supports_transcription=False))


@pytest.mark.parametrize("outputs", [(None, "audio"), (None, "audio", "timestamps"), (None, "text"), ("text", "text")])
def test_tts_and_incompatible_text_topologies_are_not_transcription_pipelines(outputs):
    assert not supports_transcription_pipeline(*_pipeline(*outputs))


@pytest.mark.parametrize("extra_output", ["audio", "image"])
def test_generic_transcription_can_coexist_with_other_output_modalities(extra_output):
    assert supports_transcription_pipeline(*_pipeline("text", extra_output))
    assert supports_transcription_pipeline(*_pipeline("text", "timestamps", extra_output))


def test_tts_timestamps_do_not_disable_plain_transcription_or_enable_asr_alignment():
    configs, metadata, runtime = _pipeline("text", "audio", "timestamps")
    configs[
        2
    ].custom_process_input_func = "vllm_omni.model_executor.stage_input_processors.forced_aligner.code2wav2aligner"
    assert supports_transcription_pipeline(configs, metadata, runtime)
    assert not has_transcription_aligner(configs, metadata, runtime)


def test_transcription_aligner_does_not_depend_on_stage_name():
    configs, metadata, runtime = _pipeline("text", "timestamps")
    metadata[1].model_stage = "custom_alignment"
    assert supports_transcription_pipeline(configs, metadata, runtime)
    assert has_transcription_aligner(configs, metadata, runtime)


@pytest.mark.parametrize(
    "invalid_field", ["runner_type", "model_config", "input_processor", "input_sources", "extra_stage"]
)
def test_incompatible_timestamp_stage_disables_only_forced_word_alignment(invalid_field):
    configs, metadata, runtime = _pipeline("text", "timestamps")
    if invalid_field == "runner_type":
        runtime[1].model_config.runner_type = "generate"
    elif invalid_field == "model_config":
        runtime[1] = None
    elif invalid_field == "input_processor":
        configs[1].custom_process_input_func = "tts.code2wav2aligner"
    elif invalid_field == "input_sources":
        configs[1].input_sources = ()
    else:
        configs.append(configs[1])
        metadata.append(metadata[1])
        runtime.append(runtime[1])
    assert supports_transcription_pipeline(configs, metadata, runtime)
    assert not has_transcription_aligner(configs, metadata, runtime)


def test_injected_aligner_does_not_make_a_text_generation_model_transcription_capable():
    assert not supports_transcription_pipeline(*_pipeline("text", "timestamps", supports_transcription=False))


def test_pooling_runner_is_not_asr_even_with_inherited_transcription_methods():
    pipeline = _pipeline("text")
    model_config = pipeline[2][0].model_config
    model_config.runner_type = "pooling"
    assert not supports_transcription_pipeline(*pipeline)
    model_config.registry.inspect_model_cls.assert_not_called()


def test_pooling_class_is_not_asr_even_if_runner_is_generate():
    pipeline = _pipeline("text")
    pipeline[2][0].model_config.registry.inspect_model_cls.return_value[0].is_pooling_model = True
    assert not supports_transcription_pipeline(*pipeline)


def test_frontend_model_config_must_describe_the_asr_entry_stage():
    pipeline = _pipeline("text", "timestamps", model_type="qwen3_asr")
    pipeline[0][0].is_comprehension = False
    assert not supports_transcription_pipeline(*pipeline)


def test_empty_pipeline_does_not_enable_transcription():
    assert not supports_transcription_pipeline(*_pipeline())


def test_missing_entry_model_config_does_not_enable_transcription():
    configs, metadata, runtime = _pipeline("text")
    runtime[0] = None
    assert not supports_transcription_pipeline(configs, metadata, runtime)


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, ImportError])
def test_inspection_failure_disables_transcription_without_breaking_startup(error_type, caplog):
    pipeline = _pipeline("text")
    pipeline[2][0].model_config.registry.inspect_model_cls.side_effect = error_type("inspection failed")
    assert not supports_transcription_pipeline(*pipeline)
    assert "transcription endpoint disabled" in caplog.text


@pytest.mark.parametrize("prefix_stages", [0, 1, 3])
def test_asr_bridge_checks_its_producer_at_any_pipeline_depth(prefix_stages):
    configs, metadata, runtime = _pipeline(*([None] * prefix_stages), "text", "timestamps")
    for config in runtime[:prefix_stages]:
        config.model_config.registry.inspect_model_cls.return_value[0].supports_transcription = False

    validate_asr_aligner_stages(configs, runtime)

    source = runtime[-2].model_config
    source.registry.inspect_model_cls.assert_called_once_with(source.architectures, source)
    for config in runtime[:prefix_stages]:
        config.model_config.registry.inspect_model_cls.assert_not_called()
    if prefix_stages:
        # Valid alignment does not require exposing the transcription HTTP API.
        assert not supports_transcription_pipeline(configs, metadata, runtime)


def test_asr_before_an_ordinary_llm_does_not_make_the_llm_output_a_transcript():
    configs, _, runtime = _pipeline("text", "text", "timestamps")
    runtime[1].model_config.registry.inspect_model_cls.return_value[0].supports_transcription = False
    assert supports_transcription_model(runtime[0].model_config)

    with pytest.raises(ValueError, match="stage 1 does not support transcription"):
        validate_asr_aligner_stages(configs, runtime)


@pytest.mark.parametrize("runner", ["pooling", "generate"])
def test_asr_bridge_rejects_pooling_models_inheriting_transcription_support(runner):
    configs, _, runtime = _pipeline(None, "text", "timestamps")
    source = runtime[1].model_config
    source.runner_type = runner
    source.registry.inspect_model_cls.return_value[0].is_pooling_model = True
    with pytest.raises(ValueError, match="transcription-capable, non-pooling"):
        validate_asr_aligner_stages(configs, runtime)


def test_asr_bridge_requires_source_model_config():
    configs, _, runtime = _pipeline(None, "text", "timestamps")
    runtime[1] = None
    with pytest.raises(ValueError, match="transcription-capable, non-pooling"):
        validate_asr_aligner_stages(configs, runtime)


def test_explicit_asr_alignment_fails_if_capability_cannot_be_inspected():
    configs, _, runtime = _pipeline(None, "text", "timestamps")
    runtime[1].model_config.registry.inspect_model_cls.side_effect = RuntimeError("inspection failed")
    with pytest.raises(RuntimeError, match="inspection failed"):
        validate_asr_aligner_stages(configs, runtime)


def test_tts_alignment_does_not_require_transcription_capability():
    configs, _, runtime = _pipeline(None, "audio", "timestamps", supports_transcription=False)
    configs[
        -1
    ].custom_process_input_func = "vllm_omni.model_executor.stage_input_processors.forced_aligner.code2wav2aligner"
    validate_asr_aligner_stages(configs, runtime)
    for config in runtime:
        config.model_config.registry.inspect_model_cls.assert_not_called()
