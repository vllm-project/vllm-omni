# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

from vllm_omni.diffusion.cache.cachedit import CacheDiTBackend
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import HunyuanVideo15Transformer3DModel
from vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5 import HunyuanVideo15Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def cpu_runtime(tmp_path, monkeypatch):
    platform = type(current_omni_platform)
    monkeypatch.setattr(
        platform,
        "get_diffusion_attn_backend_cls",
        Mock(return_value="vllm_omni.diffusion.attention.backends.sdpa.SDPABackend"),
    )
    monkeypatch.setattr(platform, "is_out_of_tree", Mock(return_value=True))
    monkeypatch.setattr(platform, "is_available", Mock(return_value=False), raising=False)

    rendezvous = (tmp_path / "gloo").as_uri()
    torch.distributed.init_process_group("gloo", init_method=rendezvous, world_size=1, rank=0)
    try:
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, distributed_init_method=rendezvous, backend="gloo"
        )
        initialize_model_parallel()
        config = OmniDiffusionConfig.from_kwargs(
            model=str(tmp_path),
            dtype=torch.float32,
            diffusion_attention_config={"default": "TORCH_SDPA"},
            linear_backend="TORCH",
        )
        with (
            set_forward_context(omni_diffusion_config=config),
            set_current_diffusion_config(config),
            set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))),
            torch.random.fork_rng(devices=[]),
            torch.no_grad(),
        ):
            torch.manual_seed(19)
            yield config
    finally:
        destroy_distributed_env()


def test_t2v_cache_dit_counts_steps_when_cfg_changes(cpu_runtime):
    pipeline = HunyuanVideo15Pipeline.__new__(HunyuanVideo15Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = cpu_runtime
    pipeline.device = torch.device("cpu")
    pipeline.vision_num_semantic_tokens = 2
    pipeline.vision_states_dim = 16
    pipeline.use_meanflow = False
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler()
    pipeline.set_progress_bar_config(disable=True)

    model = HunyuanVideo15Transformer3DModel(
        cpu_runtime,
        in_channels=5,
        out_channels=2,
        num_attention_heads=2,
        attention_head_dim=32,
        num_layers=2,
        num_refiner_layers=1,
        mlp_ratio=2.0,
        text_embed_dim=32,
        text_embed_2_dim=16,
        image_embed_dim=16,
        rope_axes_dim=(8, 12, 12),
        task_type="t2v",
    ).eval()
    for parameter in model.parameters():
        parameter.normal_(mean=0.0, std=0.05)
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            quant_method.process_weights_after_loading(module)
    pipeline.transformer = model

    text = torch.randn(1, 4, 32)
    byt5 = torch.randn(1, 4, 16)
    mask = torch.ones(1, 4, dtype=torch.bool)
    positive = (text, mask, byt5, mask)
    negative = (torch.zeros_like(text), mask, torch.zeros_like(byt5), mask)

    def encode_prompt(*, do_classifier_free_guidance, **_kwargs):
        return positive + (negative if do_classifier_free_guidance else (None,) * 4)

    pipeline.encode_prompt = encode_prompt
    backend = CacheDiTBackend({"max_warmup_steps": 4})
    backend.enable(pipeline)
    try:
        for index, (guidance_scale, expected_forwards) in enumerate(((1.0, 4), (4.0, 8), (1.0, 4))):
            backend.refresh(pipeline, 4, verbose=False)
            request = OmniDiffusionRequest(
                prompt="a short video",
                request_id=f"hunyuan-cfg-{index}",
                sampling_params=OmniDiffusionSamplingParams(
                    guidance_scale=guidance_scale,
                    num_inference_steps=4,
                    latents=torch.randn(1, 2, 2, 4, 4),
                    seed=0,
                ),
            )
            result = pipeline(DiffusionRequestBatch(requests=[request]), output_type="latent")
            assert result.output.shape == (1, 2, 2, 4, 4)
            assert torch.isfinite(result.output).all()
            context = model._context_manager.get_context()
            assert context.transformer_executed_steps == expected_forwards
            assert context.executed_steps == 4
    finally:
        backend.disable(pipeline)
