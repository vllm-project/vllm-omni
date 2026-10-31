# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for HunyuanVideo-1.5 quant_config propagation through transformer creation.

Tests cover:
- HunyuanVideo15Pipeline passes quant_config to HunyuanVideo15Transformer3DModel
- HunyuanVideo15I2VPipeline passes quant_config to HunyuanVideo15Transformer3DModel
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import PIL.Image
import pytest
import torch

import vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5 as t2v_module
import vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5_i2v as i2v_module
from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import (
    _gather_valid_encoder_hidden_states,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_conditioning_indices_preserve_token_order():
    streams = torch.arange(24, dtype=torch.float32).reshape(3, 1, 4, 2)
    masks = torch.tensor([[True, False, True, False], [False, True, True, False], [True, True, False, True]])[:, None]
    indices = t2v_module._get_encoder_hidden_states_indices(*masks)
    assert indices is not None
    expected = torch.cat([stream[:, mask[0]] for stream, mask in zip(streams, masks)], dim=1)
    assert torch.equal(_gather_valid_encoder_hidden_states(*streams, indices), expected)
    assert t2v_module._get_encoder_hidden_states_indices(*masks.repeat(1, 2, 1)) is None


def test_i2v_pre_process_uses_single_request_prompt():
    preprocess = i2v_module.get_hunyuan_video_15_i2v_pre_process_func(SimpleNamespace())
    image = PIL.Image.new("RGB", (640, 320))
    request = SimpleNamespace(
        prompt={"prompt": "turn this into a video", "multi_modal_data": {"image": image}},
        sampling_params=SimpleNamespace(height=None, width=None),
    )

    result = preprocess(request)

    assert result is request
    assert request.sampling_params.height == 448
    assert request.sampling_params.width == 896


class TestHunyuanVideoQuantConfigPropagation:
    """Verify quant_config is propagated to the transformer model in HunyuanVideo-1.5 pipelines."""

    @patch(
        "vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5.get_local_device",
        return_value="cpu",
    )
    @patch("vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5.prefetch_subfolders")
    @patch(
        "vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5.from_pretrained_with_prefetch",
        return_value=MagicMock(),
    )
    @patch(
        "vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5.Qwen2Tokenizer.from_pretrained",
        return_value=MagicMock(),
    )
    @patch(
        "vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5.ByT5Tokenizer.from_pretrained",
        return_value=MagicMock(),
    )
    @patch(
        "vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5.AutoConfig.from_pretrained",
        return_value=MagicMock(),
    )
    @patch(
        "vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5.T5EncoderModel",
        return_value=MagicMock(),
    )
    @patch(
        "vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5.FlowMatchEulerDiscreteScheduler.from_pretrained",
        return_value=MagicMock(),
    )
    @patch(
        "vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5.get_transformer_config_kwargs",
        return_value={},
    )
    def test_t2v_quant_config_passed_through(
        self,
        mock_get_kwargs,
        mock_scheduler,
        mock_t5_encoder,
        mock_auto_config,
        mock_byt5_tok,
        mock_qwen2_tok,
        mock_from_pretrained,
        mock_prefetch,
        mock_get_device,
    ):
        captured_kwargs = {}

        class FakeTransformer:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        with patch.object(t2v_module, "HunyuanVideo15Transformer3DModel", FakeTransformer):
            fake_qc = MagicMock()
            od_config = SimpleNamespace(
                model="fake-t2v-model",
                dtype=torch.bfloat16,
                flow_shift=None,
                tf_model_config=SimpleNamespace(),
                quantization_config=fake_qc,
                enable_diffusion_pipeline_profiler=False,
            )

            t2v_module.HunyuanVideo15Pipeline(od_config=od_config)

            assert captured_kwargs.get("quant_config") is fake_qc

    @patch(
        "vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5_i2v.get_local_device",
        return_value="cpu",
    )
    @patch("vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5_i2v.prefetch_subfolders")
    @patch(
        "vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5_i2v.from_pretrained_with_prefetch",
        return_value=MagicMock(),
    )
    @patch(
        "vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5_i2v.Qwen2Tokenizer.from_pretrained",
        return_value=MagicMock(),
    )
    @patch(
        "vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5_i2v.ByT5Tokenizer.from_pretrained",
        return_value=MagicMock(),
    )
    @patch(
        "vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5_i2v.AutoConfig.from_pretrained",
        return_value=MagicMock(),
    )
    @patch(
        "vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5_i2v.T5EncoderModel",
        return_value=MagicMock(),
    )
    @patch(
        "vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5_i2v.SiglipImageProcessor.from_pretrained",
        return_value=MagicMock(),
    )
    @patch(
        "vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5_i2v.FlowMatchEulerDiscreteScheduler.from_pretrained",
        return_value=MagicMock(),
    )
    @patch(
        "vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5_i2v.get_transformer_config_kwargs",
        return_value={},
    )
    def test_i2v_quant_config_passed_through(
        self,
        mock_get_kwargs,
        mock_scheduler,
        mock_siglip_processor,
        mock_t5_encoder,
        mock_auto_config,
        mock_byt5_tok,
        mock_qwen2_tok,
        mock_from_pretrained,
        mock_prefetch,
        mock_get_device,
    ):
        captured_kwargs = {}

        class FakeTransformer:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        with patch.object(i2v_module, "HunyuanVideo15Transformer3DModel", FakeTransformer):
            fake_qc = MagicMock()
            od_config = SimpleNamespace(
                model="fake-i2v-model",
                dtype=torch.bfloat16,
                flow_shift=None,
                tf_model_config=SimpleNamespace(),
                quantization_config=fake_qc,
                enable_diffusion_pipeline_profiler=False,
            )

            i2v_module.HunyuanVideo15I2VPipeline(od_config=od_config)

            assert captured_kwargs.get("quant_config") is fake_qc
