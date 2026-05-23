# SPDX-License-Identifier: Apache-2.0
from vllm_omni.model_executor.models.minimind_o.minimind_omni_config import (
    MiniMindConfig,
    MiniMindOmniAudioConfig,
    MiniMindOmniConfig,
    MiniMindOmniTalkerConfig,
    MiniMindOmniVisionConfig,
)


def test_minimind_omni_nested_configs():
    cfg = MiniMindOmniConfig()
    assert isinstance(cfg.text_config, MiniMindConfig)
    assert isinstance(cfg.talker_config, MiniMindOmniTalkerConfig)
    assert isinstance(cfg.vision_config, MiniMindOmniVisionConfig)
    assert isinstance(cfg.audio_config, MiniMindOmniAudioConfig)
    assert cfg.text_config.vocab_size == 6400
    # Talker LM vocab follows text_config; codec/MTP use audio_vocab_size.
    assert cfg.talker_config.vocab_size == cfg.text_config.vocab_size
    assert cfg.talker_config.audio_vocab_size == cfg.audio_vocab_size
    assert cfg.audio_pad_token == 2049
    assert cfg.audio_ids[0] == 16
    assert cfg.image_ids[0] == 12
    assert cfg.bridge_layer == cfg.text_config.num_hidden_layers // 2 - 1


def test_minimind_omni_hf_model_type():
    cfg = MiniMindOmniConfig.from_pretrained("jingyaogong/minimind-3o", trust_remote_code=True)
    assert cfg.model_type == "minimind-o"
    assert cfg.architectures == ["MiniMindOmni"]
    assert cfg.text_config.use_moe is False
    assert cfg.codec_num_code_layers == 8
