# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Deploy-selected SenseNova-U1-A3B FusedMoE vs. the official Qwen3-MoE loop.

Collected by the ready ``Diffusion · Model Test`` job
(``tests/diffusion/models/ -m 'core_model and cuda'``), not by the CPU simple lane.
"""

import json

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.config.omni_config import VllmOmniConfig, extract_diffusion_stage_config_kwargs
from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
    load_deploy_config,
)
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.engine.stage_init_utils import build_engine_args_dict_from_omni_stage_config
from vllm_omni.transformers_utils.configs.sensenova_u1 import SenseNovaU1MoELLMConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _tiny_moe_llm(**overrides) -> SenseNovaU1MoELLMConfig:
    kwargs = dict(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        norm_topk_prob=True,
    )
    kwargs.update(overrides)
    return SenseNovaU1MoELLMConfig(**kwargs)


def _reference_qwen3_moe(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Per-expert loop matching the official Qwen3MoeSparseMoeBlock used by SenseNova-U1.

    Softmax over the router logits in fp32, top-k, renormalize, then each selected
    expert's SwiGLU weighted by its routing probability. Accumulates in fp32.
    """
    probs = torch.softmax(router_logits.float(), dim=-1)
    top_p, top_i = torch.topk(probs, top_k, dim=-1)
    top_p = top_p / top_p.sum(dim=-1, keepdim=True)
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for expert in range(gate_w.shape[0]):
        token_idx, slot_idx = torch.where(top_i == expert)
        if token_idx.numel() == 0:
            continue
        xe = x[token_idx].float()
        h = F.silu(xe @ gate_w[expert].float().t()) * (xe @ up_w[expert].float().t())
        out.index_add_(0, token_idx, (h @ down_w[expert].float().t()) * top_p[token_idx, slot_idx, None])
    return out


@pytest.fixture
def cuda_vllm_env(monkeypatch, tmp_path):
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "29617")
    deploy_path = tmp_path / "diffusion-moe.yaml"
    deploy_path.write_text(
        "dtype: bfloat16\n"
        "stages:\n"
        "  - stage_id: 0\n"
        "    linear_backend: TORCH\n"
        "    moe_backend: TRITON\n"
        "    enforce_eager: true\n",
        encoding="utf-8",
    )
    pipeline = PipelineConfig(
        model_type="sensenova_u1",
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="model",
                execution_type=StageExecutionType.DIFFUSION,
                final_output=True,
            ),
        ),
    )
    stage = VllmOmniConfig.from_pipeline_config(
        pipeline, user_deploy_config=load_deploy_config(str(deploy_path))
    ).stage_configs[0]
    args = build_engine_args_dict_from_omni_stage_config(stage, model=str(tmp_path))
    diffusion_config = OmniDiffusionConfig.from_kwargs(
        **extract_diffusion_stage_config_kwargs(args, stage_id=0, include_engine_adapter_metadata=True)
    )
    vllm_config = create_diffusion_vllm_config(torch.device("cuda"), diffusion_config)
    assert vllm_config.kernel_config.linear_backend == "torch"
    assert vllm_config.kernel_config.moe_backend == "triton"
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
        initialize_model_parallel()
        # The modular FusedMoE kernel draws scratch buffers from here; the
        # diffusion worker initializes it at startup.
        init_workspace_manager(torch.device("cuda"))
        try:
            yield vllm_config
        finally:
            reset_workspace_manager()
            cleanup_dist_env_and_memory()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FusedMoE")
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_sparse_moe_block_matches_reference_after_checkpoint_loading(cuda_vllm_env, mocker):
    """Checkpoint-format expert shards, loaded through the production loader, must
    reproduce the reference Qwen3-MoE output. Covers the w13 gate/up order, the
    expert-id mapping and the routing math in one pass."""
    import vllm.forward_context as vllm_forward_context
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import UnquantizedMoeBackend
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    from vllm_omni.diffusion.layers.fused_moe import FusedMoE
    from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import SenseNovaU1SparseMoeBlock

    num_experts, top_k, hidden, inter, num_tokens = 16, 4, 128, 64, 32
    torch.manual_seed(0)
    # Checkpoint layout: experts.{e}.gate_proj / up_proj are [inter, hidden], down_proj is [hidden, inter].
    gate_w = (torch.randn(num_experts, inter, hidden) / hidden**0.5).to("cuda", torch.bfloat16)
    up_w = (torch.randn(num_experts, inter, hidden) / hidden**0.5).to("cuda", torch.bfloat16)
    down_w = (torch.randn(num_experts, hidden, inter) / inter**0.5).to("cuda", torch.bfloat16)
    router_w = (torch.randn(num_experts, hidden) / hidden**0.5).to("cuda", torch.bfloat16)

    config = _tiny_moe_llm(
        hidden_size=hidden, num_experts=num_experts, num_experts_per_tok=top_k, moe_intermediate_size=inter
    )
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            block = SenseNovaU1SparseMoeBlock(
                config,
                num_experts=num_experts,
                num_experts_per_tok=top_k,
                moe_intermediate_size=inter,
                prefix="model.layers.0.mlp",
            )
    finally:
        torch.set_default_dtype(default_dtype)

    params = dict(block.named_parameters())
    mapping = FusedMoE.make_expert_params_mapping(
        block,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=num_experts,
    )
    loaded: set[str] = set()
    for expert in range(num_experts):
        for proj, weights in (("gate_proj", gate_w), ("up_proj", up_w), ("down_proj", down_w)):
            assert SenseNovaU1Pipeline._load_expert_weight(
                f"experts.{expert}.{proj}.weight", weights[expert], params, mapping, loaded
            )
    with torch.no_grad():
        params["gate.weight"].copy_(router_w)
    for module in block.modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            quant_method.process_weights_after_loading(module)

    linear_method = block.gate.quant_method
    assert isinstance(linear_method, UnquantizedLinearMethod)
    assert linear_method._gemm_impl is default_unquantized_gemm
    expert_method = block.experts.routed_experts.quant_method
    assert expert_method.moe.moe_backend == "triton"
    assert expert_method.unquantized_backend is UnquantizedMoeBackend.TRITON
    assert isinstance(expert_method.moe_kernel.fused_experts, TritonExperts)
    # Spies delegate to the selected implementations, retaining real expert math.
    gemm = mocker.spy(linear_method, "_gemm_impl")
    expert_apply = mocker.spy(expert_method.moe_kernel, "apply")
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), vllm_forward_context.set_forward_context(None, cuda_vllm_env):
        router_logits, _ = block.gate(x)
        gemm.reset_mock()
        actual = block(x).float()
        assert vllm_forward_context.get_forward_context().num_tokens == num_tokens

    gemm.assert_called_once()
    expert_apply.assert_called_once()
    assert actual.shape == x.shape
    assert torch.isfinite(actual).all()

    expected = _reference_qwen3_moe(x, router_logits, gate_w, up_w, down_w, top_k)
    rel_err = ((actual - expected).norm() / expected.norm()).item()
    # Measured 0.5% on H20 in BF16; the slack covers kernel tiling differences across GPUs.
    assert rel_err < 2e-2, f"FusedMoE deviates from the reference: relative error {rel_err:.4f}"

    # The tolerance must be able to fail: swapping gate and up breaks SwiGLU.
    swapped = _reference_qwen3_moe(x, router_logits, up_w, gate_w, down_w, top_k)
    swapped_rel_err = ((actual - swapped).norm() / swapped.norm()).item()
    assert swapped_rel_err > 0.1
    print(
        json.dumps(
            {
                "linear_backend": cuda_vllm_env.kernel_config.linear_backend,
                "moe_backend": cuda_vllm_env.kernel_config.moe_backend,
                "moe_impl": type(expert_method.moe_kernel.fused_experts).__qualname__,
                "gemm_calls": gemm.call_count,
                "expert_calls": expert_apply.call_count,
                "output_shape": list(actual.shape),
                "relative_error": rel_err,
                "swapped_gate_up_relative_error": swapped_rel_err,
            },
            sort_keys=True,
        )
    )
