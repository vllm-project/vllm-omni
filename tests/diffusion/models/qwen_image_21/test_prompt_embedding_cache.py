# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def pipeline():
    result = QwenImage21Pipeline.__new__(QwenImage21Pipeline)
    nn.Module.__init__(result)
    result.text_encoder = SimpleNamespace(dtype=torch.float32, training=False)
    result.processor = object()
    result.od_config = SimpleNamespace(lora_config=None, enable_cpu_offload=False)
    result.device = torch.device("cpu")
    result.prompt_template_t2i = "{}"
    result._drop_idx = 0
    result._get_qwen_prompt_embeds = Mock(
        side_effect=lambda *args, **kwargs: (
            torch.arange(8, dtype=torch.float32).reshape(1, 2, 4),
            torch.ones(1, 2, dtype=torch.long),
            torch.zeros(1, 2, dtype=torch.bool),
        )
    )
    return result


@torch.no_grad()
def test_cache_returns_request_owned_expansion_and_masks(pipeline):
    first = pipeline.encode_prompt("A", num_images_per_prompt=2)
    first[0].fill_(999)
    second = pipeline.encode_prompt("A", num_images_per_prompt=1)
    assert pipeline._get_qwen_prompt_embeds.call_count == 1
    torch.testing.assert_close(second[0], torch.arange(8, dtype=torch.float32).reshape(1, 2, 4))
    assert second[1].shape == (1, 2)
    assert first[0].shape == (2, 2, 4)


@torch.no_grad()
def test_changed_prompt_limit_role_or_encoder_cannot_hit(pipeline):
    pipeline.encode_prompt("A")
    pipeline.encode_prompt("B")
    pipeline.encode_prompt("A", max_sequence_length=10)
    pipeline.encode_prompt("A", prompt_name="negative_prompt")
    pipeline.text_encoder = SimpleNamespace(dtype=torch.float32, training=False)
    pipeline.encode_prompt("A")
    assert pipeline._get_qwen_prompt_embeds.call_count == 5


@torch.no_grad()
def test_edit_batch_and_training_bypass_and_invalidate(pipeline):
    pipeline.encode_prompt("A")
    pipeline.encode_prompt("A", images_per_prompt=[[object()]])
    pipeline.encode_prompt("A", images_per_prompt=[[object()]])
    pipeline.encode_prompt(["A", "B"])
    pipeline.text_encoder.training = True
    pipeline.encode_prompt("A")
    pipeline.text_encoder.training = False
    pipeline.encode_prompt("A")
    assert pipeline._get_qwen_prompt_embeds.call_count == 6


@torch.no_grad()
def test_entry_limit_evicts_oldest_and_sleep_releases_tensors(pipeline):
    for index in range(17):
        pipeline.encode_prompt(str(index))
    assert len(pipeline._prompt_embedding_cache) == 16
    pipeline.encode_prompt("0")
    assert pipeline._get_qwen_prompt_embeds.call_count == 18
    pipeline.release_captured_graphs()
    assert not pipeline._prompt_embedding_cache


@torch.no_grad()
def test_failed_limit_validation_is_not_cached(pipeline):
    pipeline._get_qwen_prompt_embeds.side_effect = ValueError("invalid sequence limit")
    for _ in range(2):
        with pytest.raises(ValueError):
            pipeline.encode_prompt("A", max_sequence_length=-1)
    assert pipeline._get_qwen_prompt_embeds.call_count == 2
    assert not pipeline._prompt_embedding_cache
