# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.config.model import OmniModelArchConfigConvertor
from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_attention_geometry_matches_qwen_backbone() -> None:
    config = CosyVoice3Config()
    model_arch_config = OmniModelArchConfigConvertor(config, config.get_text_config()).convert()

    assert model_arch_config.total_num_attention_heads == 14
    assert model_arch_config.total_num_kv_heads == 2
    assert model_arch_config.head_size == 64
