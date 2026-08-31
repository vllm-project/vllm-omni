# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.omnivoice.chunking import (
    _split_at_sentence_boundaries,
    join_audio_chunks,
    split_text_into_chunks,
)
from vllm_omni.diffusion.models.omnivoice.pipeline_omnivoice import (
    OmniVoicePipeline,
    _copy_audio_to_cpu,
    _parse_chunking_seconds,
)
from vllm_omni.model_executor.models.omnivoice.omnivoice_generator import OmniVoiceGenerator
from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_chunking_config_uses_upstream_defaults_and_accepts_overrides():
    default_config = OmniVoiceConfig()
    override_config = OmniVoiceConfig(
        audio_chunk_duration=12.0,
        audio_chunk_threshold=24.0,
    )

    assert default_config.audio_chunk_duration == 15.0
    assert default_config.audio_chunk_threshold == 30.0
    assert override_config.audio_chunk_duration == 12.0
    assert override_config.audio_chunk_threshold == 24.0


@pytest.mark.parametrize(
    ("value", "allow_zero"),
    [
        (True, False),
        ("invalid", False),
        (float("nan"), False),
        (float("inf"), False),
        (0, False),
        (-1, True),
    ],
)
def test_chunking_seconds_rejects_invalid_values(value, allow_zero):
    with pytest.raises(ValueError):
        _parse_chunking_seconds("audio_chunk_duration", value, allow_zero=allow_zero)


def test_pipeline_marks_invalid_chunking_values_as_client_errors():
    pipeline = SimpleNamespace(
        config=SimpleNamespace(
            audio_chunk_duration=15.0,
            audio_chunk_threshold=30.0,
        )
    )
    request = SimpleNamespace(
        prompts=["Hello"],
        sampling_params=SimpleNamespace(
            extra_args={"audio_chunk_duration": 0},
            generator=torch.Generator(),
        ),
    )

    output = OmniVoicePipeline.forward(pipeline, request)

    assert output.error == "audio_chunk_duration must be a finite positive number"
    assert output.error_status_code == 400
    assert output.error_type == "BadRequestError"


def test_pipeline_rejects_missing_request_generator():
    pipeline = SimpleNamespace(
        config=SimpleNamespace(
            audio_chunk_duration=15.0,
            audio_chunk_threshold=30.0,
        )
    )
    request = SimpleNamespace(
        prompts=["Hello"],
        sampling_params=SimpleNamespace(extra_args={}, generator=None),
    )

    with pytest.raises(RuntimeError, match="diffusion worker"):
        OmniVoicePipeline.forward(pipeline, request)


def test_split_text_preserves_abbreviations_and_closing_marks():
    text = 'Dr. Smith left. "Next sentence!" Final sentence.'

    chunks = split_text_into_chunks(text, max_characters=20)

    assert chunks == ["Dr. Smith left.", '"Next sentence!"', "Final sentence."]


def test_split_text_handles_multi_period_abbreviation_and_cjk_punctuation():
    text = "Use e.g. this form. 下一句。最后一句！"

    chunks = split_text_into_chunks(text, max_characters=20)

    assert chunks == ["Use e.g. this form.", "下一句。最后一句！"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Meet me at apt. 5. Then leave.", ["Meet me at apt. 5.", " Then leave."]),
        ("Read the D.I.Y. guide. Next.", ["Read the D.I.Y. guide.", " Next."]),
        ("Read the D.I.Y guide. Next.", ["Read the D.I.Y guide.", " Next."]),
        ("Please R.S.V.P. today. Thanks.", ["Please R.S.V.P. today.", " Thanks."]),
        ("Please R.S.V.P today. Thanks.", ["Please R.S.V.P today.", " Thanks."]),
        ("P.S. Please reply. Done.", ["P.S. Please reply.", " Done."]),
        ("P.S Please reply. Done.", ["P.S Please reply.", " Done."]),
        ("Smith et al. reported it. Next.", ["Smith et al. reported it.", " Next."]),
    ],
)
def test_sentence_boundaries_preserve_additional_abbreviations(text, expected):
    assert _split_at_sentence_boundaries(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "alpha beta gamma delta epsilon",
        "alpha, beta gamma delta epsilon",
        "abcdefghijklmnopqrstuvwxyz",
    ],
)
def test_split_text_bounds_oversized_sentences_without_dropping_content(text):
    chunks = split_text_into_chunks(text, max_characters=10)

    assert chunks
    assert all(len(chunk) <= 10 for chunk in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_split_text_keeps_short_final_sentence_when_merge_would_exceed_limit():
    chunks = split_text_into_chunks("Long sentence. X", max_characters=14)

    assert chunks == ["Long sentence.", "X"]


def test_join_audio_chunks_returns_single_chunk_unchanged():
    audio = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4)

    assert join_audio_chunks([audio], sample_rate=20) is audio


def test_join_audio_chunks_fades_boundaries_and_inserts_silence():
    first = torch.ones(1, 1, 4)
    second = torch.full((1, 1, 3), 2.0)

    joined = join_audio_chunks([first, second], sample_rate=20)

    expected = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0]).reshape(1, 1, -1)
    torch.testing.assert_close(joined, expected)
    torch.testing.assert_close(first, torch.ones_like(first))
    torch.testing.assert_close(second, torch.full_like(second, 2.0))


def test_join_audio_chunks_fades_both_edges_of_middle_chunks():
    chunks = [torch.full((1, 1, 4), value) for value in (1.0, 2.0, 3.0)]
    original_chunks = [chunk.clone() for chunk in chunks]

    joined = join_audio_chunks(chunks, sample_rate=20)

    expected = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0]).reshape(
        1, 1, -1
    )
    torch.testing.assert_close(joined, expected)
    for chunk, original_chunk in zip(chunks, original_chunks, strict=True):
        torch.testing.assert_close(chunk, original_chunk)


def test_copy_audio_to_cpu_does_not_copy_cpu_input():
    audio = torch.ones(1, 1, 4)

    assert _copy_audio_to_cpu(audio, copy_stream=None) is audio


class _RecordingPipeline:
    def __init__(self):
        self.config = SimpleNamespace(frame_rate=1)
        self.sample_rate = 10
        self.pin_memory = False
        self.calls = []

    def _estimate_target_length(self, text, ref_text, ref_audio_tokens):
        return len(text)

    def _generate_tokens(
        self,
        *,
        text,
        target_length,
        lang,
        instruct,
        ref_text,
        ref_audio_tokens,
        generator,
    ):
        token_value = len(self.calls) + 1
        tokens = torch.full((1, 8, target_length), token_value, dtype=torch.long)
        self.calls.append(
            {
                "text": text,
                "ref_text": ref_text,
                "ref_audio_tokens": ref_audio_tokens,
                "generator": generator,
                "random_value": torch.rand((), generator=generator).item(),
                "tokens": tokens,
            }
        )
        return tokens

    @staticmethod
    def decoder(tokens):
        return tokens[:, :1].float()


def _generate_audio(
    pipeline,
    text,
    *,
    threshold,
    generator,
    chunk_duration=8.0,
    ref_text=None,
    ref_audio_tokens=None,
):
    return OmniVoicePipeline._generate_audio(
        pipeline,
        text=text,
        lang="None",
        instruct="None",
        ref_text=ref_text,
        ref_audio_tokens=ref_audio_tokens,
        audio_chunk_duration=chunk_duration,
        audio_chunk_threshold=threshold,
        generator=generator,
    )


def test_threshold_is_inclusive_and_one_frame_over_uses_chunking():
    text = "One. Two. Three."
    pipeline = _RecordingPipeline()

    _generate_audio(
        pipeline,
        text,
        threshold=len(text),
        generator=torch.Generator().manual_seed(1),
    )
    assert [call["text"] for call in pipeline.calls] == [text]

    pipeline.calls.clear()
    _generate_audio(
        pipeline,
        text,
        threshold=len(text) - 1,
        generator=torch.Generator().manual_seed(1),
    )
    assert len(pipeline.calls) > 1


@pytest.mark.parametrize(
    ("chunk_duration", "threshold"),
    [(8.0, 1e308), (1e308, 0.0)],
)
def test_extreme_finite_chunking_values_do_not_overflow(chunk_duration, threshold):
    pipeline = _RecordingPipeline()

    _generate_audio(
        pipeline,
        "One. Two. Three.",
        threshold=threshold,
        chunk_duration=chunk_duration,
        generator=torch.Generator().manual_seed(1),
    )

    assert len(pipeline.calls) == 1


def test_explicit_reference_is_reused_for_every_chunk():
    pipeline = _RecordingPipeline()
    ref_audio_tokens = torch.full((8, 4), 9)

    _generate_audio(
        pipeline,
        "First sentence. Second sentence. Third sentence.",
        threshold=0,
        generator=torch.Generator().manual_seed(2),
        ref_text="Reference text.",
        ref_audio_tokens=ref_audio_tokens,
    )

    assert len(pipeline.calls) > 1
    assert all(call["ref_text"] == "Reference text." for call in pipeline.calls)
    assert all(call["ref_audio_tokens"] is ref_audio_tokens for call in pipeline.calls)


def test_auto_voice_uses_first_chunk_as_fixed_reference_and_advances_generator():
    pipeline = _RecordingPipeline()
    generator = torch.Generator().manual_seed(3)

    _generate_audio(
        pipeline,
        "First sentence. Second sentence. Third sentence.",
        threshold=0,
        generator=generator,
    )

    assert len(pipeline.calls) > 2
    first_call = pipeline.calls[0]
    assert first_call["ref_text"] is None
    assert first_call["ref_audio_tokens"] is None
    for call in pipeline.calls[1:]:
        assert call["ref_text"] == first_call["text"]
        torch.testing.assert_close(call["ref_audio_tokens"], first_call["tokens"][0])
        assert call["generator"] is generator

    expected_generator = torch.Generator().manual_seed(3)
    expected_values = [torch.rand((), generator=expected_generator).item() for _ in pipeline.calls]
    assert [call["random_value"] for call in pipeline.calls] == expected_values


def test_generator_uses_and_advances_the_supplied_random_state(monkeypatch):
    config = OmniVoiceConfig(
        audio_vocab_size=5,
        audio_mask_id=4,
        num_audio_codebook=2,
        enable_cuda_graph=False,
        llm_config={
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 32,
            "vocab_size": 32,
            "max_position_embeddings": 32,
            "head_dim": 8,
        },
    )
    model = OmniVoiceGenerator(config)
    monkeypatch.setattr(
        model,
        "_transformer_forward",
        lambda inputs_embeds, attention_mask, cos, sin: torch.zeros_like(inputs_embeds),
    )
    generator = torch.Generator().manual_seed(4)

    def run_generation():
        input_ids = torch.full((2, 2, 3), config.audio_mask_id, dtype=torch.long)
        return model(
            input_ids=input_ids,
            audio_mask=torch.ones(2, 3, dtype=torch.bool),
            attention_mask=torch.zeros(2, 1, 3, 3, dtype=torch.bool),
            target_lens=[3],
            conditional_lens=[3],
            generator=generator,
            num_step=2,
        )

    state_before = generator.get_state()
    run_generation()
    state_after_first_call = generator.get_state()
    run_generation()
    state_after_second_call = generator.get_state()

    assert not torch.equal(state_before, state_after_first_call)
    assert not torch.equal(state_after_first_call, state_after_second_call)


def test_duration_estimate_uses_reference_text_and_token_count_together():
    calls = []
    pipeline = SimpleNamespace(
        duration_estimator=SimpleNamespace(
            estimate_duration=lambda text, ref_text, ref_length: calls.append((text, ref_text, ref_length)) or 42.8
        )
    )
    ref_audio_tokens = torch.zeros(8, 17)

    target_length = OmniVoicePipeline._estimate_target_length(
        pipeline,
        "Target text.",
        "Reference text.",
        ref_audio_tokens,
    )

    assert target_length == 42
    assert calls == [("Target text.", "Reference text.", 17)]
