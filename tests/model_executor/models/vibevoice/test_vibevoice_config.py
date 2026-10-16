# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Portable CPU contracts."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoConfig, PreTrainedTokenizerFast
from transformers.models.vibevoice_acoustic_tokenizer.feature_extraction_vibevoice_acoustic_tokenizer import (
    VibeVoiceAcousticTokenizerFeatureExtractor,
)
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.processing import InputProcessingContext
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.config.stage_config import _DEPLOY_DIR, load_deploy_config, merge_pipeline_deploy
from vllm_omni.model_executor.models.registry import OmniModelRegistry
from vllm_omni.model_executor.models.vibevoice.pipeline import VIBEVOICE_PIPELINE
from vllm_omni.model_executor.models.vibevoice.prompt import (
    AUDIO_BOS,
    AUDIO_EOS,
    AUDIO_TOKEN,
    AUDIO_TOKEN_ID,
    check_reference_prompt,
    render_vibevoice_prompt,
)
from vllm_omni.transformers_utils.configs.vibevoice import VibeVoiceConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
ARCH = "VibeVoiceForConditionalGeneration"


@pytest.fixture
def processor(tmp_path, mocker):
    config = VibeVoiceConfig(
        text_config={"hidden_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": 8},
        audio_bos_token_id=1,
        audio_eos_token_id=2,
        audio_token_id=3,
        eos_token_id=4,
        pad_token_id=4,
    )
    config.save_pretrained(tmp_path)
    VibeVoiceAcousticTokenizerFeatureExtractor(pad_to_multiple_of=3200).save_pretrained(tmp_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"<unk>": 0, AUDIO_BOS: 1, AUDIO_EOS: 2, AUDIO_TOKEN: 3, "<|endoftext|>": 4}, unk_token="<unk>")
        )
    )
    mm_config = MultiModalConfig(limit_per_prompt={"audio": 1})
    model_config = mocker.Mock(
        spec=ModelConfig,
        model=str(tmp_path),
        revision=None,
        hf_config=AutoConfig.from_pretrained(tmp_path),
        multimodal_config=mm_config,
    )
    model_config.get_multimodal_config.return_value = mm_config
    model_cls = OmniModelRegistry._try_load_model_cls(ARCH)
    assert model_cls is not None
    return model_cls._processor_factory.build_processor(InputProcessingContext(model_config, tokenizer))


def _apply(processor, ids, audio, cache=None):
    items = processor.info.parse_mm_data({"audio": audio})
    return processor.apply(ProcessorInputs(prompt=ids, mm_data_items=items, cache=cache), TimingContext(enabled=False))


def test_native_config_and_legacy_rejection():
    data = {"text_config": {"model_type": "qwen2", "hidden_size": 32}, "diffusion_head_config": {"hidden_size": 32}}
    original = deepcopy(data)
    config = VibeVoiceConfig(**data)
    assert data == original
    assert config.get_text_config().hidden_size == config.diffusion_head_config.hidden_size == 32
    with pytest.raises(ValueError, match="native HF schema"):
        VibeVoiceConfig(decoder_config={})
    with pytest.raises(ValueError, match="hidden_size"):
        VibeVoiceConfig(text_config={"hidden_size": 32}, diffusion_head_config={"hidden_size": 64})


def test_registration_and_deploy():
    model_cls = OmniModelRegistry._try_load_model_cls(ARCH)
    assert model_cls is not None and model_cls.get_placeholder_str("audio", 0) == AUDIO_TOKEN
    stage = merge_pipeline_deploy(VIBEVOICE_PIPELINE, load_deploy_config(_DEPLOY_DIR / "vibevoice.yaml"))[0]
    args = stage.yaml_engine_args
    # VibeVoice is eager-only: graph, async scheduling and prefix caching must stay off.
    assert args["enforce_eager"] and not args["async_scheduling"] and not args["enable_prefix_caching"]
    assert not args["use_v2_model_runner"] and args["max_num_seqs"] == 4


def test_prompt():
    prompt = render_vibevoice_prompt("Hello.")
    assert prompt.count(AUDIO_TOKEN) == 1 and prompt.endswith(AUDIO_BOS)
    assert "Speaker 0: Hello.\n" in prompt
    no_voice = render_vibevoice_prompt("Hello.", reference=False)
    assert AUDIO_TOKEN not in no_voice and "Voice input" not in no_voice and no_voice.endswith(AUDIO_BOS)
    for invalid in ("", "  ", AUDIO_TOKEN):
        with pytest.raises(ValueError):
            render_vibevoice_prompt(invalid)
    # vLLM skips the multimodal processor without audio, so the stage-0 transform rejects stray placeholders.
    audio = {"audio": (np.zeros(3200, dtype=np.float32), 24000)}
    for ok in (no_voice, {"prompt": no_voice}, {"prompt": prompt, "multi_modal_data": audio}):
        assert check_reference_prompt(ok, []) is ok
    for bad in (prompt, {"prompt": prompt, "multi_modal_data": {}}, {"prompt_token_ids": [1, AUDIO_TOKEN_ID, 2]}):
        with pytest.raises(ValueError, match="no audio"):
            check_reference_prompt(bad, [])


def test_reference_lengths_and_cache_hit(processor, mocker):
    ids = [1, 3, 2, 1]
    cache = MultiModalProcessorOnlyCache(processor.info.ctx.model_config)
    extract = mocker.spy(processor, "_call_hf_processor")
    for n in (3200, 3201):
        audio = (np.linspace(-0.1, 0.1, n, dtype=np.float32), 24000)
        first = _apply(processor, ids, audio, cache)
        calls = extract.call_count
        cached = _apply(processor, ids, audio, cache)
        assert extract.call_count == calls  # A real cache hit, not another extraction.
        count = (n + 3199) // 3200
        data = cached["mm_kwargs"].get_data()
        assert data["audio_num_tokens"].tolist() == [count] and data["audio_num_tokens"].device.type == "cpu"
        assert data["input_values"].shape == (1, 1, count * 3200)
        assert cached["prompt_token_ids"] == first["prompt_token_ids"]
        assert cached["mm_placeholders"]["audio"][0].length == count
        torch.testing.assert_close(data["input_values"], first["mm_kwargs"].get_data()["input_values"], atol=0, rtol=0)
    assert ids == [1, 3, 2, 1]


def test_reference_resampling_and_rejection(processor):
    ids = [1, 3, 2, 1]
    # 4800 samples at 16 kHz -> 7200 at 24 kHz -> 3 tokens (2 without resampling).
    result = _apply(processor, ids, (np.ones(4800, dtype=np.float32), 16000))
    assert result["mm_kwargs"].get_data()["audio_num_tokens"].tolist() == [3]
    # Any channel layout is mixed to mono before resampling: 0.3 s -> 3 tokens.
    for sr in (16000, 24000, 48000):
        mono = np.linspace(-0.1, 0.1, int(0.3 * sr), dtype=np.float32)
        for audio in (np.stack([mono, mono], axis=1), np.stack([mono, mono]), mono[None], mono.tolist()):
            result = _apply(processor, ids, (audio, sr))
            assert result["mm_kwargs"].get_data()["audio_num_tokens"].tolist() == [3], (sr, np.shape(audio))
    good = (np.ones(3200, dtype=np.float32), 24000)
    for bad_ids, audio in [(ids + [3], good), ([1], good), (ids, [])]:
        with pytest.raises(ValueError):
            _apply(processor, bad_ids, audio)
    with pytest.raises(VLLMValidationError, match="At most 1"):
        _apply(processor, ids, [good, good])
    for array in (np.empty(0), np.array([np.nan]), np.zeros(60 * 24000 + 1)):
        with pytest.raises(ValueError):
            _apply(processor, ids, (array, 24000))


def test_startup_rejects_unsupported_runtime(mocker):
    from vllm_omni.model_executor.models.vibevoice.vibevoice import _check_positive_kv_capacity, _check_supported_config

    def config(cache_dtype="auto", v2=False, num_gpu_blocks=4097) -> VllmConfig:
        runtime = VllmConfig()
        # Only checkpoint loading is replaced; use the real runtime configs and MRv2 property.
        runtime.model_config = mocker.Mock(
            spec=OmniModelConfig, enforce_eager=True, max_model_len=16384, use_v2_model_runner=v2
        )
        runtime.scheduler_config = SchedulerConfig(
            max_model_len=16384, is_encoder_decoder=False, async_scheduling=False, max_num_seqs=4
        )
        runtime.cache_config = CacheConfig(enable_prefix_caching=False, cache_dtype=cache_dtype, block_size=16)
        runtime.cache_config.num_gpu_blocks = num_gpu_blocks
        return runtime

    _check_supported_config(config())
    for bad, match in ((config(cache_dtype="fp8"), "quantized KV"), (config(v2=True), "runner V2")):
        with pytest.raises(ValueError, match=match):
            _check_supported_config(bad)
    # 4 requests x 1024 blocks, plus the null block.
    _check_positive_kv_capacity(config())
    with pytest.raises(ValueError, match="cannot be preempted"):
        _check_positive_kv_capacity(config(num_gpu_blocks=4096))
