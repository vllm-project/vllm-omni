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


def test_single_stage_selects_async_codec_scheduler():
    from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
    from vllm_omni.model_executor.models.cosyvoice3.async_scheduler import CosyVoice3AsyncScheduler
    from vllm_omni.model_executor.models.cosyvoice3.pipeline import COSYVOICE3_SINGLE_STAGE_PIPELINE

    stage = COSYVOICE3_SINGLE_STAGE_PIPELINE.stages[0]
    assert stage.scheduler_cls == (
        "vllm_omni.model_executor.models.cosyvoice3.async_scheduler.CosyVoice3AsyncScheduler"
    )
    assert issubclass(CosyVoice3AsyncScheduler, OmniARAsyncScheduler)
