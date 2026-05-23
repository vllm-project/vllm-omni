# SPDX-License-Identifier: Apache-2.0
from vllm_omni.model_executor.models.minimind_o.minimind_omni_config import MiniMindOmniConfig


def test_minimind_omni_multimodal_marker_ids():
    cfg = MiniMindOmniConfig()
    assert cfg.audio_ids == [16]
    assert cfg.image_ids == [12]
    assert cfg.audio_pad_token == 2049
    assert cfg.codec_pad_token == 2049
    assert cfg.bridge_layer == cfg.text_config.num_hidden_layers // 2 - 1
