# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Kimi-Audio multimodal processing through vLLM's native ``apply``."""

from types import SimpleNamespace

import numpy as np
import pytest
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.processing import ProcessorInputs, TimingContext

from vllm_omni.model_executor.models.kimi_audio.audio_processing import SAMPLES_PER_TOKEN, prepare_kimi_audio_inputs
from vllm_omni.model_executor.models.kimi_audio.processor import (
    KimiAudioDummyInputsBuilder,
    KimiAudioMultiModalProcessor,
    KimiAudioProcessingInfo,
)
from vllm_omni.model_executor.models.kimi_audio.prompt import (
    KimiAudioPreparedInput,
    KimiAudioPromptBuilder,
    KimiAudioSpecialTokens,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.omni]

OFFSET = 256


class _Context:
    """The parts of InputProcessingContext that Kimi processing reads."""

    def __init__(self):
        self.model_config = SimpleNamespace(
            model="kimi-audio-test",
            hf_config=SimpleNamespace(kimia_token_offset=OFFSET, use_whisper_feature=False),
            model_stage=None,
            multimodal_config=None,
        )
        # The base class requires a tokenizer. Prepared prompts are already
        # tokenized, so processing must never call it.
        self.tokenizer = SimpleNamespace()

    def get_hf_config(self):
        return self.model_config.hf_config

    def get_mm_config(self):
        return SimpleNamespace(mm_hasher_algorithm="blake3")


@pytest.fixture
def processor():
    info = KimiAudioProcessingInfo(_Context())
    return KimiAudioMultiModalProcessor(info, KimiAudioDummyInputsBuilder(info))


@pytest.fixture
def prompt_builder():
    return KimiAudioPromptBuilder(
        encode_text=lambda text: list(text.encode("utf-8")),
        special_tokens=KimiAudioSpecialTokens(
            msg_end=0,
            media_begin=1,
            media_end=2,
            kimia_text_blank=3,
            kimia_text_eos=4,
            kimia_user_msg_start=5,
            kimia_assistant_msg_start=6,
            kimia_speech_ct_id=7,
            kimia_speech_ctd_id=8,
        ),
        audio_token_offset=OFFSET,
        audio_vocab_size=128,
        audio_delay=2,
        continuous_feature_size=4,
    )


def _prepare(prompt_builder, code_counts):
    """A prepared prompt with one audio history item per code count."""
    messages, audio = [], {}
    for count in code_counts:
        messages.append({"role": "user", "message_type": "text", "content": "Hi"})
        audio[len(messages)] = np.ones(count * SAMPLES_PER_TOKEN, dtype=np.float32)
        messages.append({"role": "assistant", "message_type": "audio-text", "content": (None, "a")})
    prepared = prepare_kimi_audio_inputs(messages, prompt_builder, audio_inputs=audio, output_type="both")
    layout = KimiAudioPreparedInput.from_wire(prepared["model_intermediate_buffer"]["kimi_audio_input"])
    spans = [(start, stop - start) for _, start, stop in layout.audio_spans]
    assert [length for _, length in spans] == list(code_counts)
    return prepared, spans


def _inputs(processor, prepared, cache=None):
    items = processor.info.parse_mm_data({"audio": prepared["multi_modal_data"]["audio"]}, validate=False)
    return ProcessorInputs(prompt=list(prepared["prompt_token_ids"]), mm_data_items=items, cache=cache)


def _apply(processor, prepared, cache=None):
    return processor.apply(_inputs(processor, prepared, cache), TimingContext(enabled=False))


# A shorter item after a longer one would match the longer span's prefix if
# vLLM re-inserted placeholders; the prepared spans must only be located.
@pytest.mark.parametrize("code_counts", [(3, 2), (2, 3)])
def test_prepared_audio_spans_are_located_without_rewriting(processor, prompt_builder, code_counts):
    prepared, spans = _prepare(prompt_builder, code_counts)

    result = _apply(processor, prepared)

    assert result["prompt_token_ids"] == prepared["prompt_token_ids"]
    assert [(p.offset, p.length) for p in result["mm_placeholders"]["audio"]] == spans
    waveforms = [item["kimi_waveform"].data for item in result["mm_kwargs"]["audio"]]
    assert [len(waveform) // SAMPLES_PER_TOKEN for waveform in waveforms] == list(code_counts)


def test_processor_cache_hits_keep_prepared_spans(processor, prompt_builder):
    # Serving enables the processor cache by default. Cached items return
    # their stored updates, which must locate the spans of the new prompt.
    config = SimpleNamespace(get_multimodal_config=lambda: SimpleNamespace(mm_processor_cache_gb=1))
    cache = MultiModalProcessorOnlyCache(config)
    # All misses, all hits, then a new item before a cached one.
    for code_counts, cached in [((3, 2), [False, False]), ((3, 2), [True, True]), ((4, 3), [False, True])]:
        prepared, spans = _prepare(prompt_builder, code_counts)
        inputs = _inputs(processor, prepared, cache)
        assert [cache.is_cached_item(h) for h in processor._get_mm_hashes(inputs)["audio"]] == cached

        result = processor.apply(inputs, TimingContext(enabled=False))

        assert result["prompt_token_ids"] == prepared["prompt_token_ids"]
        assert [(p.offset, p.length) for p in result["mm_placeholders"]["audio"]] == spans


def test_profiling_dummy_inputs_keep_their_span_layout(processor):
    seq_len = 4
    inputs = processor.get_dummy_inputs(seq_len, {"audio": 2}, {})
    result = processor.apply(inputs, TimingContext(enabled=False))

    assert result["prompt_token_ids"] == [OFFSET] * (2 * seq_len)
    assert [(p.offset, p.length) for p in result["mm_placeholders"]["audio"]] == [(0, seq_len), (seq_len, seq_len)]
