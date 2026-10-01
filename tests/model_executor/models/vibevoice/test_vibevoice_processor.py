# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VibeVoice config normalization, processor registration and deploy defaults."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import AutoConfig, PretrainedConfig, PreTrainedTokenizerFast, Qwen2Config
from transformers.models.vibevoice_acoustic_tokenizer.configuration_vibevoice_acoustic_tokenizer import (
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceAcousticTokenizerEncoderConfig,
)
from vllm.config.multimodal import MultiModalConfig
from vllm.multimodal.processing import InputProcessingContext
from vllm.sampling_params import SamplingParams

from vllm_omni.config.omni_config import _stage_sampling_params
from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.model_executor.models.vibevoice.pipeline import (
    VIBEVOICE_PIPELINE,
    VIBEVOICE_VALID_TOKEN_IDS,
)
from vllm_omni.model_executor.models.vibevoice.processing_vibevoice import (
    AUDIO_BOS_TOKEN,
    AUDIO_EOS_TOKEN,
    AUDIO_TOKEN,
    MAX_AUDIO_SAMPLES,
    SAMPLE_RATE,
    VibeVoiceDummyInputsBuilder,
    VibeVoiceMultiModalProcessor,
    VibeVoiceProcessingInfo,
)
from vllm_omni.transformers_utils.configs.vibevoice import VibeVoiceConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _official_checkpoint_config() -> dict:
    """Structural fields from Microsoft's original 1.5B checkpoint."""
    return {
        "acoustic_vae_dim": 64,
        "acoustic_tokenizer_config": {
            "causal": True,
            "channels": 1,
            "conv_bias": True,
            "conv_norm": "none",
            "encoder_depths": "3-3-3-3-3-3-8",
            "encoder_n_filters": 32,
            "encoder_ratios": [8, 5, 5, 4, 2, 2],
            "fix_std": 0.5,
            "layer_scale_init_value": 1e-6,
            "layernorm": "RMSNorm",
            "layernorm_elementwise_affine": True,
            "layernorm_eps": 1e-5,
            "mixer_layer": "depthwise_conv",
            "model_type": "vibevoice_acoustic_tokenizer",
            "pad_mode": "constant",
            "std_dist_type": "gaussian",
            "vae_dim": 64,
            "weight_init_value": 0.01,
        },
        "architectures": ["VibeVoiceForConditionalGeneration"],
        "decoder_config": {
            "hidden_size": 1536,
            "max_position_embeddings": 65536,
            "model_type": "qwen2",
            "num_attention_heads": 12,
            "num_hidden_layers": 28,
            "num_key_value_heads": 2,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
            "vocab_size": 151936,
        },
        "diffusion_head_config": {"model_type": "vibevoice_diffusion_head"},
        "model_type": "vibevoice",
        "semantic_tokenizer_config": {"model_type": "vibevoice_semantic_tokenizer"},
        "torch_dtype": "bfloat16",
        "custom_checkpoint_field": "preserved",
    }


@pytest.fixture
def checkpoint_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_official_checkpoint_config()), encoding="utf-8")
    return tmp_path


def test_config_normalizes_official_checkpoint(checkpoint_dir):
    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=False)

    assert isinstance(config, VibeVoiceConfig)
    assert config.model_type == "vibevoice"
    assert isinstance(config.audio_config, VibeVoiceAcousticTokenizerConfig)
    assert isinstance(config.semantic_model_config, VibeVoiceAcousticTokenizerEncoderConfig)
    assert isinstance(config.get_text_config(), Qwen2Config)
    assert config.get_text_config().hidden_size == 1536


def test_deploy_defaults_match_generation_contract():
    deploy_path = Path(__file__).parents[4] / "vllm_omni" / "deploy" / "vibevoice.yaml"
    stage = load_deploy_config(deploy_path).stages[0]

    assert stage.tensor_parallel_size == 1
    assert stage.enforce_eager is False
    assert stage.max_model_len == 65536
    assert stage.max_num_seqs == 4
    assert stage.engine_extras["limit_mm_per_prompt"] == {"audio": 8}
    assert stage.engine_extras["additional_config"]["vibevoice_runtime_config"] == {
        "negative_kv_cache_memory_bytes": 8 * 1024**3,
        "negative_kv_activation_margin_bytes": 512 * 1024**2,
        "diffusion_cuda_graph": True,
        "decode_cuda_graph": True,
    }

    effective = _stage_sampling_params(stage, VIBEVOICE_PIPELINE.stages[0])
    sampling_params = SamplingParams(**effective)
    assert sampling_params.allowed_token_ids == VIBEVOICE_VALID_TOKEN_IDS
    assert sampling_params.stop_token_ids == [151643]
    assert sampling_params.detokenize is False


def _make_processor(*, user_audio_limit: int = 8):
    backend = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "Speaker": 1,
                "0:": 2,
                "1:": 3,
                "then": 4,
                AUDIO_BOS_TOKEN: 5,
                AUDIO_EOS_TOKEN: 6,
                AUDIO_TOKEN: 7,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        additional_special_tokens=[AUDIO_BOS_TOKEN, AUDIO_EOS_TOKEN, AUDIO_TOKEN],
    )
    hf_config = PretrainedConfig()
    hf_config.audio_bos_token_id = tokenizer.convert_tokens_to_ids(AUDIO_BOS_TOKEN)
    hf_config.audio_eos_token_id = tokenizer.convert_tokens_to_ids(AUDIO_EOS_TOKEN)
    hf_config.audio_token_id = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN)
    mm_config = MultiModalConfig(limit_per_prompt={"audio": user_audio_limit})
    model_config = SimpleNamespace(
        model="test-vibevoice",
        hf_config=hf_config,
        multimodal_config=mm_config,
        dtype=torch.float32,
        encoder_config=None,
        max_model_len=4096,
        get_multimodal_config=lambda: mm_config,
    )
    ctx = InputProcessingContext(model_config, tokenizer=tokenizer)
    info = VibeVoiceProcessingInfo(ctx)
    return VibeVoiceMultiModalProcessor(info, VibeVoiceDummyInputsBuilder(info), cache=None), info


def test_processor_preserves_multi_reference_order():
    processor, info = _make_processor()
    tokenizer = info.get_tokenizer()
    prompt = (
        f"Speaker 0: {AUDIO_BOS_TOKEN}{AUDIO_TOKEN}{AUDIO_EOS_TOKEN} "
        f"then Speaker 1: {AUDIO_BOS_TOKEN}{AUDIO_TOKEN}{AUDIO_EOS_TOKEN}"
    )
    first_audio = np.linspace(-0.25, 0.25, 3_201, dtype=np.float32)
    second_audio = np.sin(np.linspace(0, 20 * np.pi, 8_001, dtype=np.float32)).astype(np.float32)
    mm_items = info.parse_mm_data({"audio": [(first_audio, SAMPLE_RATE), (second_audio, SAMPLE_RATE)]})

    result = processor(prompt, mm_items=mm_items)

    expected = (
        f"Speaker 0: {AUDIO_BOS_TOKEN}{AUDIO_TOKEN * 2}{AUDIO_EOS_TOKEN} "
        f"then Speaker 1: {AUDIO_BOS_TOKEN}{AUDIO_TOKEN * 3}{AUDIO_EOS_TOKEN}"
    )
    assert result["prompt_token_ids"] == tokenizer.encode(expected, add_special_tokens=False)
    assert [item.length for item in result["mm_placeholders"]["audio"]] == [2, 3]


def test_sixty_second_audio_expands_and_longer_is_rejected():
    processor, info = _make_processor()
    prompt = f"{AUDIO_BOS_TOKEN}{AUDIO_TOKEN}{AUDIO_EOS_TOKEN}"
    ok = np.zeros(MAX_AUDIO_SAMPLES, dtype=np.float32)
    result = processor(prompt, mm_items=info.parse_mm_data({"audio": [(ok, SAMPLE_RATE)]}))
    assert result["mm_kwargs"].get_data()["audio_num_tokens"].item() == 450

    too_long = np.zeros(MAX_AUDIO_SAMPLES + 1, dtype=np.float32)
    with pytest.raises(ValueError, match=r"60\.00s; the maximum is 60s"):
        processor(prompt, mm_items=info.parse_mm_data({"audio": [(too_long, SAMPLE_RATE)]}))
