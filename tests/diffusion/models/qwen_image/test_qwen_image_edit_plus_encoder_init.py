# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest
import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_image import pipeline_qwen_image_edit_plus as pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _StopAfterEncoderError(Exception):
    """Stop before loading unrelated VAE and DiT weights."""


@pytest.mark.parametrize("loader_device", ["cpu", "meta"])
def test_edit_plus_encoder_preserves_cpu_rope_initialization(tmp_path: Path, mocker, loader_device: str):
    config = Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
            "fullatt_block_indexes": [0],
        },
    )
    with set_default_torch_dtype(torch.bfloat16):
        reference = Qwen2_5_VLForConditionalGeneration(config)
    reference.save_pretrained(tmp_path / "text_encoder")
    expected = {name: buffer for name, buffer in reference.named_buffers() if "inv_freq" in name}
    assert expected

    mocker.patch.object(pipeline, "get_local_device", return_value=torch.device("cpu"))
    mocker.patch.object(pipeline.FlowMatchEulerDiscreteScheduler, "from_pretrained")
    mocker.patch.object(pipeline.AutoencoderKLQwenImage, "from_pretrained", side_effect=_StopAfterEncoderError)
    loaded = mocker.spy(pipeline.Qwen2_5_VLForConditionalGeneration, "from_pretrained")
    od_config = OmniDiffusionConfig(model=str(tmp_path))

    # A non-CPU outer context represents the loader's accelerator context
    # without requiring a GPU. RoPE buffers must still be computed on CPU.
    with torch.device(loader_device), set_default_torch_dtype(torch.bfloat16):
        with pytest.raises(_StopAfterEncoderError):
            pipeline.QwenImageEditPlusPipeline(od_config=od_config)

    encoder = loaded.spy_return
    assert encoder.dtype == torch.bfloat16
    actual = dict(encoder.named_buffers())
    for name, buffer in expected.items():
        assert actual[name].device.type == "cpu"
        assert actual[name].dtype == torch.float32
        torch.testing.assert_close(actual[name], buffer, rtol=0, atol=0)
