# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

from vllm_omni.diffusion.cache.cachedit import CacheDiTBackend
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.longcat_image.longcat_image_transformer import LongCatImageTransformer2DModel
from vllm_omni.diffusion.models.longcat_image.pipeline_longcat_image import LongCatImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def cpu_diffusion_config(tmp_path, monkeypatch):
    platform = type(current_omni_platform)
    monkeypatch.setattr(
        platform,
        "get_diffusion_attn_backend_cls",
        lambda *_args, **_kwargs: "vllm_omni.diffusion.attention.backends.sdpa.SDPABackend",
    )
    monkeypatch.setattr(platform, "is_out_of_tree", lambda *_args: True)

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
            enforce_eager=True,
            diffusion_attention_config={"default": "TORCH_SDPA"},
            linear_backend="TORCH",
        )
        config.set_tf_model_config(
            TransformerConfig.from_dict(
                {
                    "patch_size": 1,
                    "in_channels": 64,
                    "num_layers": 2,
                    "num_single_layers": 2,
                    "attention_head_dim": 16,
                    "num_attention_heads": 2,
                    "joint_attention_dim": 24,
                    "pooled_projection_dim": 32,
                    "axes_dims_rope": [4, 4, 8],
                }
            )
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


def test_t2i_cache_dit_counts_both_block_lists_when_cfg_changes(cpu_diffusion_config, monkeypatch):
    pipeline = LongCatImagePipeline.__new__(LongCatImagePipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = cpu_diffusion_config
    pipeline.device = torch.device("cpu")
    pipeline.vae_scale_factor = 8
    pipeline.tokenizer_max_length = 512
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)

    model = LongCatImageTransformer2DModel(cpu_diffusion_config).eval()
    for parameter in model.parameters():
        parameter.normal_(mean=0.0, std=0.05)
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            quant_method.process_weights_after_loading(module)
    pipeline.transformer = model

    positive = torch.randn(1, 4, 24)
    negative = torch.zeros_like(positive)
    monkeypatch.setattr(pipeline, "_encode_prompt", lambda prompts: positive if prompts[0] else negative)
    initial_latents = torch.randn(1, 4, 64)
    denoise_steps = 4

    def run_request(guidance_scale):
        request = OmniDiffusionRequest(
            prompt="a cat on a chair",
            request_id="longcat-t2i",
            sampling_params=OmniDiffusionSamplingParams(
                height=32,
                width=32,
                guidance_scale=guidance_scale,
                num_inference_steps=denoise_steps,
                latents=initial_latents.clone(),
                output_type="latent",
                extra_args={"enable_prompt_rewrite": False},
            ),
        )
        result = pipeline(DiffusionRequestBatch(requests=[request]))
        assert pipeline.scheduler.step_index == denoise_steps
        return result.output

    uncached_outputs = {scale: run_request(scale) for scale in (1.0, 4.0)}
    backend = CacheDiTBackend({"max_warmup_steps": denoise_steps})
    backend.enable(pipeline)
    try:
        assert len(model._context_names) == 2
        for guidance_scale, expected_forwards in ((1.0, 4), (4.0, 8), (1.0, 4)):
            backend.refresh(pipeline, denoise_steps, verbose=False)
            torch.testing.assert_close(run_request(guidance_scale), uncached_outputs[guidance_scale])
            for context_name in model._context_names:
                context = model._context_manager.get_context(context_name)
                assert context.transformer_executed_steps == expected_forwards
                assert context.executed_steps == denoise_steps
    finally:
        backend.disable(pipeline)
