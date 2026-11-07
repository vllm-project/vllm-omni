# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.models.omnivoice.chunking import (
    _split_at_sentence_boundaries,
    join_audio_chunks,
    split_text_into_chunks,
)
from vllm_omni.diffusion.models.omnivoice.pipeline_omnivoice import (
    _MAX_AUDIO_CHUNKS,
    OmniVoicePipeline,
    _copy_audio_to_cpu,
    _parse_chunking_seconds,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.omnivoice.duration import RuleDurationEstimator
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
        pytest.param(10**1000, False, id="overflow-duration"),
        pytest.param(10**1000, True, id="overflow-threshold"),
        (0, False),
        (-1, True),
    ],
)
def test_chunking_seconds_rejects_invalid_values(value, allow_zero):
    with pytest.raises(ValueError):
        _parse_chunking_seconds("audio_chunk_duration", value, allow_zero=allow_zero)


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


@pytest.mark.parametrize("number", ["0.26", "0,26"])
def test_sentence_boundaries_preserve_numeric_separators(number):
    text = f"Value {number} stays intact."
    assert _split_at_sentence_boundaries(text + " Next.") == [text, " Next."]


@pytest.mark.parametrize("number", ["0.26", "0,26"])
def test_oversized_sentence_keeps_numeric_separator(number):
    text = f"Value {number} stays intact, then we continue."
    chunks = split_text_into_chunks(text, max_characters=9)
    assert any(number in chunk for chunk in chunks)
    assert all(len(chunk) <= 9 for chunk in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_numeric_protection_keeps_sentence_boundaries():
    assert _split_at_sentence_boundaries("Value 0.26. Next, please.") == ["Value 0.26.", " Next, please."]


@pytest.mark.parametrize("punctuation", [",", ";", ":", "，", "；", "："])
def test_split_text_prefers_whole_sentences_over_clauses(punctuation):
    text = f"Done. Alpha{punctuation} beta gamma delta."

    assert split_text_into_chunks(text, max_characters=25) == ["Done.", f"Alpha{punctuation} beta gamma delta."]


@pytest.mark.parametrize("punctuation", [",", ";", ":", "，", "；", "："])
def test_oversized_sentence_falls_back_to_clause_boundaries(punctuation):
    text = f"Alpha{punctuation} beta gamma delta."

    assert split_text_into_chunks(text, max_characters=18) == [f"Alpha{punctuation}", "beta gamma delta."]


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\n\n"])
def test_split_text_prefers_line_boundaries(newline):
    lines = ["First item on this line", "Second item on this line", "Last three words"]
    text = newline.join(lines)

    assert "".join(_split_at_sentence_boundaries(text)) == text
    assert split_text_into_chunks(text, max_characters=30) == lines


def test_short_lines_can_share_a_chunk():
    assert split_text_into_chunks("First\nSecond\nThird", max_characters=30) == ["First\nSecond\nThird"]


def test_sentence_can_end_in_a_number():
    assert split_text_into_chunks("Meet me at apt. 5. Then leave.", max_characters=24) == [
        "Meet me at apt. 5.",
        "Then leave.",
    ]


def test_abbreviation_before_clause_punctuation_stays_intact():
    assert _split_at_sentence_boundaries("Acme Co., based here. Next.") == [
        "Acme Co., based here.",
        " Next.",
    ]


def test_decimal_list_keeps_numbers_whole():
    text = "See releases 0.26, 0.28 and 0.30 now."
    chunks = split_text_into_chunks(text, max_characters=20)
    assert all(any(number in chunk for chunk in chunks) for number in ("0.26", "0.28", "0.30"))
    assert all(len(chunk) <= 20 for chunk in chunks)


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


class _RecordingPipeline(OmniVoicePipeline):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(
            frame_rate=1,
            num_audio_codebook=8,
            audio_mask_id=100,
            audio_chunk_duration=8.0,
            audio_chunk_threshold=30.0,
        )
        self.device = torch.device("cpu")
        self.sample_rate = 10
        self.tokenizer = SimpleNamespace(encode=lambda text: SimpleNamespace(ids=[1]))
        self.duration_estimator = SimpleNamespace(estimate_duration=lambda text, ref_text, ref_len: len(text))
        self.audio_tokenizer = SimpleNamespace(config=SimpleNamespace(sample_rate=10, hop_length=1))
        self._inline_reference_cache = OrderedDict()
        self._initialize_asr(None)
        self.num_step = 2
        self.guidance_scale = 2.0
        self.t_shift = 0.1
        self.layer_penalty_factor = 5.0
        self.position_temperature = 5.0
        self.class_temperature = 0.0
        self.calls = []
        self.draws = []
        self.reference = torch.full((8, 4), 9)

    def _encode_ref_audio(self, audio_signal, sr):
        return self.reference

    def _prepare_chunk_input(
        self, text, lang, instruct, ref_text, ref_audio_tokens, seed, chunks=None, *, reference_rms=None
    ):
        self.calls.append({"text": text, "ref_text": ref_text, "ref_audio_tokens": ref_audio_tokens})
        return super()._prepare_chunk_input(
            text, lang, instruct, ref_text, ref_audio_tokens, seed, chunks, reference_rms=reference_rms
        )

    def generator(self, *, target_lens, generators, **kwargs):
        tokens = []
        for length, generator in zip(target_lens, generators, strict=True):
            self.draws.append((generator, torch.rand((), generator=generator).item()))
            tokens.append(torch.full((1, 8, length), generator.initial_seed(), dtype=torch.long))
        return torch.cat(tokens, dim=-1)

    @staticmethod
    def decoder(tokens):
        return tokens[:, :1].float()


def _request(prompt, *, seed=3, **extra):
    return SimpleNamespace(
        prompt=prompt,
        sampling_params=SimpleNamespace(
            extra_args=extra,
            seed=seed,
            num_inference_steps=2,
            guidance_scale=None,
        ),
    )


def _run(pipeline, prompt, *, threshold=0, chunk_duration=8.0, seed=3):
    return pipeline(
        SimpleNamespace(
            requests=[
                _request(
                    prompt,
                    seed=seed,
                    audio_chunk_duration=chunk_duration,
                    audio_chunk_threshold=threshold,
                )
            ]
        )
    )


def test_pipeline_marks_invalid_chunking_values_as_client_errors():
    pipeline = _RecordingPipeline()
    request = _request("Hello", audio_chunk_duration=0)
    outputs = pipeline(SimpleNamespace(requests=[request]))
    assert len(outputs) == 1
    assert outputs[0].error == "audio_chunk_duration must be a finite positive number"
    assert outputs[0].error_status_code == 400
    assert outputs[0].error_type == "BadRequestError"
    assert not pipeline.draws

    state = SimpleNamespace(prompt=request.prompt, sampling=request.sampling_params, extra={})
    with pytest.raises(OmniClientError) as error:
        pipeline.prepare_encode(state)
    assert error.value.status_code == 400
    assert error.value.error_type == "BadRequestError"


@pytest.mark.parametrize("chunk_count", [_MAX_AUDIO_CHUNKS, _MAX_AUDIO_CHUNKS + 1, 386])
@pytest.mark.parametrize("step_mode", [False, True])
def test_chunk_count_limit_rejects_before_generation(chunk_count, step_mode):
    pipeline = _RecordingPipeline()
    request = _request("A" * chunk_count, audio_chunk_duration=0.001, audio_chunk_threshold=0)
    state = SimpleNamespace(prompt=request.prompt, sampling=request.sampling_params, extra={})
    if chunk_count <= _MAX_AUDIO_CHUNKS:
        prepared = (
            pipeline.prepare_encode(state).extra["prepared"]
            if step_mode
            else pipeline._prepare_request_input(request.prompt, request.sampling_params.extra_args)
        )
        assert len(prepared.chunks.texts) == chunk_count
    else:
        message = f"at most {_MAX_AUDIO_CHUNKS} audio chunks"
        if step_mode:
            with pytest.raises(OmniClientError, match=message) as error:
                pipeline.prepare_encode(state)
            assert error.value.status_code == 400
        else:
            output = pipeline(SimpleNamespace(requests=[request]))[0]
            assert message in output.error
            assert output.error_status_code == 400
        assert not pipeline.calls
    assert not pipeline.draws


@pytest.mark.parametrize("canonical_seed,legacy_seed", [(7, None), (None, 3), (7, 3)])
@pytest.mark.parametrize("step_mode", [False, True])
def test_omnivoice_preserves_model_owned_seed_over_random_default(canonical_seed, legacy_seed, step_mode):
    pipeline = _RecordingPipeline()
    extra = {"seed": legacy_seed} if legacy_seed is not None else {}
    request = OmniDiffusionRequest(
        prompt="Short",
        sampling_params=OmniDiffusionSamplingParams(seed=canonical_seed, extra_args=extra),
        request_id="seed-test",
    )
    expected_seed = legacy_seed if legacy_seed is not None else canonical_seed
    if step_mode:
        state = SimpleNamespace(prompt=request.prompt, sampling=request.sampling_params, extra={})
        pipeline.prepare_encode(state)
        generator = state.extra["generator"]
    else:
        pipeline(SimpleNamespace(requests=[request]))
        generator = pipeline.draws[0][0]
    assert generator.initial_seed() == expected_seed


@pytest.mark.parametrize("seed", [-(2**63), -1])
@pytest.mark.parametrize("legacy", [False, True])
def test_step_generator_accepts_negative_canonical_and_legacy_seeds(seed, legacy):
    pipeline = _RecordingPipeline()
    sampling = OmniDiffusionSamplingParams(seed=None if legacy else seed, extra_args={"seed": seed} if legacy else {})
    request = OmniDiffusionRequest(prompt="Short", sampling_params=sampling, request_id="negative-seed")
    state = SimpleNamespace(prompt=request.prompt, sampling=request.sampling_params, extra={})
    pipeline.prepare_encode(state)
    assert state.extra["generator"].initial_seed() == torch.Generator().manual_seed(seed).initial_seed()


def test_threshold_is_inclusive_and_one_frame_over_uses_chunking():
    text = "One. Two. Three."
    pipeline = _RecordingPipeline()
    _run(pipeline, text, threshold=len(text))
    assert [call["text"] for call in pipeline.calls] == [text]
    pipeline.calls.clear()
    _run(pipeline, text, threshold=len(text) - 1)
    assert len(pipeline.calls) > 1


@pytest.mark.parametrize(
    ("chunk_duration", "threshold"),
    [(8.0, 1e308), (1e308, 0.0)],
)
def test_extreme_finite_chunking_values_do_not_overflow(chunk_duration, threshold):
    pipeline = _RecordingPipeline()
    _run(pipeline, "One. Two. Three.", threshold=threshold, chunk_duration=chunk_duration)
    assert len(pipeline.calls) == 1


def test_explicit_reference_is_reused_for_every_chunk():
    pipeline = _RecordingPipeline()
    _run(
        pipeline,
        {
            "input": "First sentence. Second sentence. Third sentence.",
            "ref_text": "Reference text.",
            "ref_audio": (torch.ones(4), 10),
        },
    )
    assert len(pipeline.calls) > 1
    assert all(call["ref_text"] == "Reference text." for call in pipeline.calls)
    assert all(call["ref_audio_tokens"] is pipeline.reference for call in pipeline.calls)


def test_auto_voice_uses_first_chunk_as_fixed_reference_and_advances_generator():
    pipeline = _RecordingPipeline()
    _run(pipeline, "First sentence. Second sentence. Third sentence.", seed=3)
    assert len(pipeline.calls) > 2
    first = pipeline.calls[0]
    assert first["ref_text"] is None
    assert first["ref_audio_tokens"] is None
    reference = pipeline.calls[1]["ref_audio_tokens"]
    torch.testing.assert_close(reference, torch.full((8, len(first["text"])), 3, dtype=torch.long))
    for call in pipeline.calls[1:]:
        assert call["ref_text"] == first["text"]
        assert call["ref_audio_tokens"] is reference
    generator = pipeline.draws[0][0]
    expected = torch.Generator().manual_seed(3)
    assert all(draw[0] is generator for draw in pipeline.draws)
    assert [draw[1] for draw in pipeline.draws] == [torch.rand((), generator=expected).item() for _ in pipeline.draws]


def test_mixed_short_and_long_batch_preserves_output_order():
    pipeline = _RecordingPipeline()
    requests = [
        _request("First sentence. Second sentence. Third sentence.", seed=3, audio_chunk_threshold=10),
        _request("Short", seed=7),
        _request("Another long sentence. And one more sentence.", seed=5, audio_chunk_threshold=10),
    ]
    outputs = pipeline(SimpleNamespace(requests=requests))
    assert len(outputs) == 3
    for output, seed in zip(outputs, (3, 7, 5), strict=True):
        assert output.error is None
        assert output.output.max().item() == seed
    torch.testing.assert_close(outputs[1].output, torch.full((1, 1, 5), 7.0))
    assert sum(generator.initial_seed() == 7 for generator, _ in pipeline.draws) == 1


def test_step_decode_resets_schedule_and_retains_generator_until_final_chunk():
    pipeline = _RecordingPipeline()
    request = _request("One. Two. Three.", audio_chunk_threshold=0)
    state = SimpleNamespace(prompt=request.prompt, sampling=request.sampling_params, extra={}, step_index=0)
    pipeline.prepare_encode(state)
    generator = state.extra["generator"]
    count = len(state.extra["prepared"].chunks.texts)
    assert count > 1
    for index in range(count):
        schedule = state.extra["schedules"]
        assert all(isinstance(count, int) for count in schedule)
        assert schedule == state.timesteps.tolist()
        assert sum(schedule) == state.extra["target_len"] * 8
        state.extra["tokens"].fill_(index + 1)
        state.step_index = len(state.timesteps)
        output = pipeline.post_decode(state)
        assert state.extra["generator"] is generator
        if index < count - 1:
            assert output is None
            assert state.step_index == 0
            assert torch.all(state.extra["tokens"] == pipeline.config.audio_mask_id)
        else:
            assert output.output.shape[-1] > state.extra["target_len"]
            assert state.step_index == len(state.timesteps)


def test_generator_uses_and_advances_the_supplied_random_state(monkeypatch):
    config = OmniVoiceConfig(
        audio_vocab_size=5,
        audio_mask_id=4,
        num_audio_codebook=2,
        enable_cuda_graph=False,
        llm_config={
            "hidden_size": 16,
            "num_hidden_layers": 0,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 32,
            "vocab_size": 32,
            "max_position_embeddings": 32,
            "head_dim": 8,
        },
    )
    model = OmniVoiceGenerator(config, od_config=SimpleNamespace())
    monkeypatch.setattr(
        model,
        "_transformer_forward",
        lambda inputs_embeds, cu_seqs, max_seqlen=None, rope_table=None: torch.zeros_like(inputs_embeds),
    )
    generator = torch.Generator().manual_seed(4)

    def run_generation():
        input_ids = torch.full((6, 2), config.audio_mask_id, dtype=torch.long)
        return model(
            input_ids=input_ids,
            audio_mask=torch.ones(6, dtype=torch.bool),
            cond_lens=[3],
            target_lens=[3],
            generators=[generator],
            num_step=2,
        )

    state_before = generator.get_state()
    run_generation()
    state_after_first_call = generator.get_state()
    run_generation()
    state_after_second_call = generator.get_state()

    assert not torch.equal(state_before, state_after_first_call)
    assert not torch.equal(state_after_first_call, state_after_second_call)


def test_generator_rejects_wrong_number_of_supplied_generators():
    model = SimpleNamespace(config=SimpleNamespace(audio_mask_id=4, num_audio_codebook=2))
    with pytest.raises(ValueError, match="one generator per request"):
        OmniVoiceGenerator.forward(
            model,
            input_ids=torch.full((6, 2), 4, dtype=torch.long),
            audio_mask=torch.ones(6, dtype=torch.bool),
            cond_lens=[3],
            target_lens=[3],
            generators=[],
        )


def test_duration_estimate_uses_reference_text_and_token_count_together():
    calls = []

    def estimate_duration(text, ref_text, ref_length):
        calls.append((text, ref_text, ref_length))
        return 42.8

    pipeline = SimpleNamespace(duration_estimator=SimpleNamespace(estimate_duration=estimate_duration))
    ref_audio_tokens = torch.zeros(8, 17)

    target_length = OmniVoicePipeline._estimate_target_len(
        pipeline,
        "Target text.",
        "Reference text.",
        ref_audio_tokens,
    )

    assert target_length == 42
    assert calls == [("Target text.", "Reference text.", 17)]


def test_short_clone_uses_reference_duration_before_chunking_threshold():
    pipeline = _RecordingPipeline()
    pipeline.duration_estimator = RuleDurationEstimator()
    pipeline.config.frame_rate = 25
    text = "One short sentence."
    prompt = {"input": text, "ref_text": text, "ref_audio": (torch.ones(4), 10)}
    for ref_length in (100, 200):
        pipeline._inline_reference_cache.clear()
        pipeline.reference = torch.zeros(8, ref_length, dtype=torch.long)
        prepared = pipeline._prepare_request_input(prompt, {})
        assert prepared.chunks is None
        assert prepared.target_len == ref_length


@pytest.mark.parametrize("ref_text,has_audio", [(None, True), ("Reference text.", False), (None, False)])
def test_duration_without_complete_reference_retains_default_estimate(ref_text, has_audio):
    pipeline = _RecordingPipeline()
    pipeline.duration_estimator = RuleDurationEstimator()
    text = "One short sentence."
    tokens = torch.zeros(8, 200) if has_audio else None
    assert pipeline._estimate_target_len(text, ref_text, tokens) == pipeline._estimate_target_len(text, None, None)


def test_reference_duration_changes_whether_chunking_starts():
    pipeline = _RecordingPipeline()
    pipeline.duration_estimator = RuleDurationEstimator()
    pipeline.config.frame_rate = 25
    text = "First sentence. Second sentence."
    prompt = {"input": text, "ref_text": text, "ref_audio": (torch.ones(4), 10)}
    extra = {"audio_chunk_threshold": 6.0, "audio_chunk_duration": 3.0}
    pipeline.reference = torch.zeros(8, 100, dtype=torch.long)
    assert pipeline._prepare_request_input(prompt, extra).chunks is None
    pipeline._inline_reference_cache.clear()
    pipeline.reference = torch.zeros(8, 200, dtype=torch.long)
    assert len(pipeline._prepare_request_input(prompt, extra).chunks.texts) > 1


@pytest.mark.parametrize("step_mode", [False, True])
@pytest.mark.parametrize("use_asr", [False, True])
def test_chunked_clone_keeps_reference_and_postprocesses_joined_audio_once(step_mode, use_asr):
    pipeline = _RecordingPipeline()
    pipeline._asr_pipeline = Mock(return_value={"text": "Reference transcript"})
    pipeline._postprocess_audio = Mock(wraps=pipeline._postprocess_audio)
    prompt = {
        "input": "First sentence. Second sentence. Third sentence.",
        "ref_audio": (torch.full((4,), 0.05), 10),
    }
    if not use_asr:
        prompt["ref_text"] = "Reference transcript"
    request = _request(prompt, audio_chunk_threshold=0)

    if step_mode:
        state = SimpleNamespace(prompt=request.prompt, sampling=request.sampling_params, extra={}, step_index=0)
        pipeline.prepare_encode(state)
        chunk_count = len(state.extra["prepared"].chunks.texts)
        for index in range(chunk_count):
            state.extra["tokens"].fill_(3)
            state.step_index = len(state.timesteps)
            output = pipeline.post_decode(state)
            if index < chunk_count - 1:
                assert output is None
                pipeline._postprocess_audio.assert_not_called()
    else:
        output = pipeline(SimpleNamespace(requests=[request]))[0]

    assert output.error is None
    assert len(pipeline.calls) > 1
    assert all(call["ref_text"] == "Reference transcript." for call in pipeline.calls)
    assert all(call["ref_audio_tokens"] is pipeline.reference for call in pipeline.calls)
    assert pipeline._asr_pipeline.call_count == int(use_asr)
    pipeline._postprocess_audio.assert_called_once()
    joined_audio, reference_rms = pipeline._postprocess_audio.call_args.args
    assert reference_rms == pytest.approx(0.05)
    assert joined_audio.shape[-1] > max(len(call["text"]) for call in pipeline.calls)
    assert output.output.max().item() == pytest.approx(1.5)
