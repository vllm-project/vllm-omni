# SPDX-License-Identifier: Apache-2.0
from vllm_omni.model_executor.models.minimind_o.config import (
    MiniMindOConfig,
    MiniMindOThinkerConfig,
    MiniMindOTalkerConfig,
)


def test_minimind_o_nested_configs():
    cfg = MiniMindOConfig()
    assert isinstance(cfg.thinker_config, MiniMindOThinkerConfig)
    assert isinstance(cfg.talker_config, MiniMindOTalkerConfig)
    assert cfg.thinker_config.text_config.vocab_size == 6400
    assert cfg.talker_config.text_config.vocab_size == cfg.audio_vocab_size
    assert cfg.talker_config.audio_pad_token == 2049
    assert cfg.thinker_config.audio_token_index == 16
    assert cfg.thinker_config.image_token_index == 12


def test_minimind_o_hf_model_type():
    cfg = MiniMindOConfig.from_pretrained("jingyaogong/minimind-3o", trust_remote_code=True)
    assert cfg.model_type == "minimind-o"
    assert cfg.architectures == ["MiniMindOmni"]
    assert cfg.bridge_layer == 3
    assert cfg.use_moe is False
