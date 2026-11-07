# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The config's constants, and the variant switch that every deploy file relies on."""

import pytest

from tests.model_executor.models.chatterbox.helpers import SEED
from vllm_omni.config.model import OmniModelArchConfigConvertor
from vllm_omni.model_executor.models.chatterbox.chatterbox_s3gen import SAMPLES_PER_FRAME, S3GenDecoder
from vllm_omni.transformers_utils.configs.chatterbox import VARIANT_FIELDS, ChatterboxConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("variant", ["turbo", "original"])
def test_checkpoint_constants_agree_with_what_reads_them(variant: str) -> None:
    """Relations the two stages rely on, each of which fails a load or the audio when broken."""
    cfg = ChatterboxConfig(variant)
    # The decoder's mel is the output audio's, and HiFT turns one frame into
    # the 480 samples the decoder's seam caches are sized in.
    assert cfg.mel["sampling_rate"] == cfg.sample_rate
    assert cfg.mel["hop_size"] == SAMPLES_PER_FRAME
    assert cfg.mel["hop_size"] * cfg.token_mel_ratio * cfg.token_rate == cfg.sample_rate
    # Stage 0's conditioning is the encoder window's tokens.
    assert cfg.cond_prompt_len == cfg.enc_cond_seconds * cfg.token_rate
    # Silence is a speech token; the control ids sit above every speech
    # token and inside the speech vocabulary, which is the sampler's.
    assert cfg.silence_token < cfg.speech_token_limit <= cfg.start_speech_token
    assert cfg.start_speech_token < cfg.stop_speech_token < cfg.speech_vocab_size == cfg.vocab_size
    # The engine stops on the stop token.
    assert cfg.eos_token_id == cfg.stop_speech_token


@pytest.mark.parametrize("variant", ["turbo", "original"])
def test_the_engine_reads_the_backbones_attention_geometry(variant: str) -> None:
    """vLLM sizes the KV cache from this, not from the backbone stage 0 builds."""
    cfg = ChatterboxConfig(variant)

    arch = OmniModelArchConfigConvertor(cfg, cfg.get_text_config()).convert()

    assert (arch.total_num_attention_heads, arch.total_num_kv_heads, arch.head_size) == (
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.hidden_size // cfg.num_attention_heads,
    )


def test_variant_and_flow_settings_survive_serialization() -> None:
    """vLLM serializes the config into each stage process and rebuilds it there."""
    config = ChatterboxConfig("original", flow_dtype="float32")
    restored = ChatterboxConfig.from_dict(config.to_dict())
    assert restored.to_dict() == config.to_dict()
    assert {name: getattr(restored, name) for name in VARIANT_FIELDS["original"]} == VARIANT_FIELDS["original"]


@pytest.mark.parametrize("variant", ["turbo", "original"])
def test_variant_override_after_construction_switches_every_checkpoint_field(variant: str) -> None:
    """vLLM applies a deploy file's hf_overrides through update, after construction."""
    config = ChatterboxConfig("original" if variant == "turbo" else "turbo")
    config.update({"variant": variant})
    expected = ChatterboxConfig(variant)
    assert {name: getattr(config, name) for name in VARIANT_FIELDS[variant]} == {
        name: getattr(expected, name) for name in VARIANT_FIELDS[variant]
    }
    assert config.to_dict() == expected.to_dict()


def test_unknown_variant_fails_on_first_use() -> None:
    with pytest.raises(ValueError, match="variant"):
        ChatterboxConfig("multilingual")
    config = ChatterboxConfig()
    config.update({"variant": "multilingual"})
    with pytest.raises(ValueError, match="multilingual"):
        _ = config.num_hidden_layers


def test_checkpoint_fields_cannot_be_overridden_one_by_one() -> None:
    """A deploy file switches the checkpoint with variant, never a field of it."""
    config = ChatterboxConfig()
    with pytest.raises(AttributeError):
        config.update({"num_hidden_layers": 30})


def test_the_decoder_refuses_a_flow_precision_it_cannot_run() -> None:
    """A deploy file's hf_overrides reach the config after construction, so stage 1 checks it when it is built."""
    config = ChatterboxConfig()
    config.update({"flow_dtype": "float16"})
    with pytest.raises(ValueError, match="flow_dtype"):
        S3GenDecoder(config, config.max_new_tokens, SEED)
