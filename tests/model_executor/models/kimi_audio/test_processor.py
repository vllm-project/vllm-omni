# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native CPU MM cache, placeholder and profiling-input boundaries."""

from types import SimpleNamespace

import pytest
import torch
from transformers import WhisperFeatureExtractor
from vllm.exceptions import VLLMValidationError
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.processing import ProcessorInputs, TimingContext

from tests.model_executor.models.kimi_audio.runtime import cpu_pp_group as cpu_pp_group
from tests.model_executor.models.kimi_audio.runtime import kimi_mm_processor as kimi_mm_processor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.omni]


def test_native_profile_honors_model_encoder_skip(cpu_pp_group):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    seen = []
    runner = SimpleNamespace(
        supports_mm_inputs=True,
        model_config=SimpleNamespace(multimodal_config=SimpleNamespace(skip_mm_profiling=True)),
        max_num_tokens=2,
        is_pooling_model=False,
        encoder_cache={},
        _dummy_run=lambda *args, **kwargs: (seen.append("backbone") or torch.zeros(2, 4), torch.zeros(1, 4)),
        _dummy_sampler_run=lambda hidden: seen.append("heads"),
        _sync_device=lambda: None,
    )
    # No encoder or MM budget: any attempted encoder profiling fails. The
    # actual native profile_run must still warm up the backbone on each rank.
    cpu_pp_group.is_first_rank, cpu_pp_group.is_last_rank = False, False
    GPUModelRunner.profile_run(runner)
    cpu_pp_group.is_last_rank = True
    GPUModelRunner.profile_run(runner)
    assert seen == ["backbone", "backbone", "heads"]


def test_partial_processor_cache_preserves_audio_order_and_history_mode(kimi_mm_processor):
    # Identical waveform, but one item uses Whisper and the history item does
    # not. Prime one cache entry, then reorder it around a missing entry.
    processor = kimi_mm_processor(offset=152064)
    processor.cache = MultiModalProcessorOnlyCache(processor.info.ctx.model_config)
    waveform = torch.ones(2 * 1280)
    audio = dict(waveform=waveform, whisper_features=torch.ones(1, 128, 3000), whisper_lengths=torch.tensor([2]))
    history = dict(
        waveform=waveform, whisper_features=torch.empty(0, 128, 3000), whisper_lengths=torch.empty(0, dtype=torch.long)
    )

    def process(items):
        # A non-audio separator makes adjacent audio items unambiguous.
        prompt = [152064, 152064, 0] * len(items)
        output = processor.apply(
            ProcessorInputs(prompt, processor.info.parse_mm_data({"audio": items})), TimingContext(enabled=False)
        )
        assert output["prompt_token_ids"] == prompt
        assert [(r.offset, r.length) for r in output["mm_placeholders"]["audio"]] == [
            (i * 3, 2) for i in range(len(items))
        ]
        data = output["mm_kwargs"].get_data()
        for received, expected in zip(data["kimi_whisper_features"], items, strict=True):
            torch.testing.assert_close(received, expected["whisper_features"])
        return output["mm_hashes"]["audio"]

    first = process([audio])[0]
    mixed = process([history, audio])
    assert mixed[0] != first and mixed[1] == first
    assert process([audio, history]) == mixed[::-1]
    assert process([history, audio]) == mixed  # all-hit path, no new fields


@pytest.mark.parametrize("use_whisper", [False, True])
def test_native_dummy_inputs_cover_long_audio_and_declared_budget(kimi_mm_processor, monkeypatch, use_whisper):
    from vllm.transformers_utils import processor as hf_processor

    extractor = WhisperFeatureExtractor(feature_size=128)
    monkeypatch.setattr(hf_processor, "cached_feature_extractor_from_config", lambda *args, **kwargs: extractor)
    processor = kimi_mm_processor(offset=152064, use_whisper=use_whisper)
    config = processor.info.ctx.model_config
    config.max_model_len = 376  # 30 seconds plus one audio token
    result = MULTIMODAL_REGISTRY.get_dummy_mm_inputs(config, {"audio": 1}, processor=processor)
    assert processor.info.get_mm_max_tokens_per_item(config.max_model_len, {"audio": 1}) == {"audio": 376}
    assert [(r.offset, r.length) for r in result["mm_placeholders"]["audio"]] == [(0, 376)]
    data = result["mm_kwargs"].get_data()
    assert data["kimi_waveform"].shape == (1, 376 * 1280)
    assert data["kimi_whisper_lengths"][0].tolist() == ([375, 1] if use_whisper else [])
    assert data["kimi_whisper_features"].shape == (1, 2 if use_whisper else 0, 128, 3000)
    assert all(field.field.keep_on_cpu for field in result["mm_kwargs"]["audio"][0].values())


def test_decoder_has_no_input_audio_and_native_limits_are_enforced(kimi_mm_processor):
    decoder = kimi_mm_processor(offset=152064, stage="kimi_audio_decoder")
    assert decoder.info.get_supported_mm_limits() == {}
    assert decoder.info.get_mm_max_tokens_per_item(8192, {}) == {}
    assert decoder.dummy_inputs.get_dummy_mm_data(8192, {}) == {}
    disabled = kimi_mm_processor(offset=152064, audio_limit=0)
    item = dict(
        waveform=torch.ones(1),
        whisper_features=torch.empty(0, 128, 3000),
        whisper_lengths=torch.empty(0, dtype=torch.long),
    )
    with pytest.raises(VLLMValidationError, match="At most 0"):
        disabled.info.parse_mm_data({"audio": [item]})
