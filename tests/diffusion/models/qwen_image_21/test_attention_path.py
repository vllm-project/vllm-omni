# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request attention choices reach both Qwen-Image 2.1 execution protocols."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline
from vllm_omni.model_extras.qwen_image_21 import QWEN_IMAGE_21_EXTRA_BODY_PARAMS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("protocol", ["forward", "prepare_encode"])
@pytest.mark.parametrize("path", [None, "auto", "piecewise", "mask", "flex"])
def test_request_attention_path_reaches_preparation(monkeypatch, protocol, path):
    pipeline = object.__new__(QwenImage21Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.vae_scale_factor = 8
    pipeline._max_length = 512
    monkeypatch.setattr(pipeline, "_extract_prompts", lambda prompts: (["prompt"], None))
    monkeypatch.setattr(
        pipeline, "_extract_request_images", lambda prompts: [{"vae_images": None, "input_image_sizes": None}]
    )
    captured = []

    class PreparationReachedError(Exception):
        pass

    def capture(**kwargs):
        captured.append(kwargs["attention_kwargs"])
        raise PreparationReachedError

    monkeypatch.setattr(pipeline, "_prepare_generation_context", capture)
    sampling = SimpleNamespace(
        height=256,
        width=256,
        num_inference_steps=2,
        sigmas=None,
        max_sequence_length=512,
        num_outputs_per_prompt=1,
        true_cfg_scale=1.0,
        output_type="latent",
        generator=None,
        extra_args={} if path is None else {"attn_path": path},
    )
    with pytest.raises(PreparationReachedError):
        if protocol == "forward":
            pipeline.forward(
                SimpleNamespace(
                    prompts=["prompt"],
                    sampling_params_list=[sampling],
                    collate_request_generators=lambda *args: None,
                    collate_request_tensors=lambda *args: None,
                )
            )
        else:
            pipeline.prepare_encode(SimpleNamespace(sampling=sampling, prompt="prompt"))
    assert captured == [{"attn_path": path or "auto"}]
    assert "attn_path" in QWEN_IMAGE_21_EXTRA_BODY_PARAMS
