# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run a tiny native FLUX transformer, checking dispatch rather than saved flags.

This is a model-component smoke test, not image-quality or serving validation.
Random weights and synthetic embeddings keep it offline and small enough for
one CUDA GPU. No dispatch or numerical implementation is mocked.
"""

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.utils import default_unquantized_gemm
from vllm.platforms import current_platform

from vllm_omni.config.omni_config import VllmOmniDiffusionStageConfig, extract_diffusion_stage_config_kwargs
from vllm_omni.config.stage_config import StageExecutionType, StagePipelineConfig
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.flux.flux_transformer import FluxTransformer2DModel
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.engine.stage_init_utils import build_engine_args_dict_from_omni_stage_config

pytestmark = [pytest.mark.local_model, pytest.mark.diffusion, pytest.mark.cuda]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA dispatch smoke test")
def test_tiny_flux_runs_requested_backends(tmp_path, mocker):
    stage = VllmOmniDiffusionStageConfig(
        stage_pipeline_config=StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            final_output=True,
        ),
        diffusion_config={
            "linear_backend": "TORCH",
            "diffusion_attention_config": {"default": "TORCH_SDPA"},
        },
    )
    args = build_engine_args_dict_from_omni_stage_config(stage, model="unused")
    od_config = OmniDiffusionConfig.from_kwargs(
        **extract_diffusion_stage_config_kwargs(
            args,
            stage_id=0,
            include_engine_adapter_metadata=True,
        )
    )
    config = create_diffusion_vllm_config(torch.device("cuda", 0), od_config)
    try:
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{tmp_path / 'distributed'}",
        )
        initialize_model_parallel(sequence_parallel_size=1)
        # Observe inputs to the real selector without replacing its behavior.
        # On Ampere, "torch" and "auto" can resolve to the same callable.
        dispatch = mocker.spy(linear, "dispatch_unquantized_gemm")
        with set_current_vllm_config(config), set_current_diffusion_config(od_config):
            model = (
                FluxTransformer2DModel(
                    num_layers=1,
                    num_single_layers=1,
                    num_attention_heads=2,
                    attention_head_dim=32,
                    joint_attention_dim=64,
                    pooled_projection_dim=64,
                    axes_dims_rope=(8, 12, 12),
                    guidance_embeds=False,
                )
                .to(device="cuda", dtype=torch.bfloat16)
                .eval()
            )
        generator = torch.Generator(device="cuda").manual_seed(42)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(mean=0.0, std=0.02, generator=generator)

        calls = []
        handles = []
        linears = []
        attentions = []
        for name, layer in model.named_modules():
            if isinstance(getattr(layer, "quant_method", None), UnquantizedLinearMethod):
                assert layer.quant_method._gemm_impl is default_unquantized_gemm
                linears.append(name)
                handles.append(layer.register_forward_hook(lambda module, inputs, output: calls.append(module)))
            if isinstance(layer, Attention):
                assert isinstance(layer.attention, SDPAImpl)
                attentions.append(name)
        assert linears and attentions
        assert dispatch.call_count == len(linears)
        assert all(call.args == ("torch",) for call in dispatch.call_args_list)

        def random_tensor(*shape):
            return torch.randn(*shape, device="cuda", dtype=torch.bfloat16, generator=generator)

        inputs = dict(
            hidden_states=random_tensor(1, 4, 64),
            encoder_hidden_states=random_tensor(1, 4, 64),
            pooled_projections=random_tensor(1, 64),
            timestep=torch.tensor([0.5], device="cuda"),
            img_ids=torch.zeros(4, 3, device="cuda"),
            txt_ids=torch.zeros(4, 3, device="cuda"),
        )
        with torch.inference_mode(), set_forward_context(vllm_config=config, omni_diffusion_config=od_config):
            output = model(**inputs).sample
        for handle in handles:
            handle.remove()
        assert len(calls) >= len(linears)
        assert output.shape == (1, 4, 64)
        assert torch.isfinite(output).all()
        print(
            f"Tiny FLUX: {len(linears)} real linear layers executed; GEMM=default_unquantized_gemm; "
            f"{len(attentions)} attention layers=SDPAImpl; output={tuple(output.shape)} finite"
        )
    finally:
        destroy_distributed_env()
