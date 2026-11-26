# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise both fused FLUX.2 attention blocks through native model forwards."""

import importlib
from unittest.mock import patch

import pytest
import torch
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not current_platform.is_cuda() or not torch.cuda.is_available(), reason="NVIDIA CUDA required"),
    pytest.mark.skipif(not HAS_TRITON, reason="Triton required"),
]

_MIN_TOKENS = "VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS"


@pytest.fixture
def _diffusion_context(tmp_path):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.platforms import current_omni_platform

    vllm_config = VllmConfig()
    config = OmniDiffusionConfig(
        model=None,
        enforce_eager=True,
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
    )
    with (
        set_current_vllm_config(vllm_config),
        current_omni_platform.get_default_ir_op_priority(vllm_config).set_priority(),
        set_current_diffusion_config(config),
        set_forward_context(omni_diffusion_config=config),
    ):
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, distributed_init_method=f"file://{tmp_path / 'rendezvous'}"
        )
        try:
            initialize_model_parallel()
            yield config
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.parametrize("model_name", ["flux2", "flux2_klein"])
@torch.no_grad()
def test_flux2_model_fused_rope_eager_and_compiled(_diffusion_context, monkeypatch, model_name):
    from torch._inductor.compile_fx import compile_fx
    from vllm import ir

    from vllm_omni.diffusion.compile import regionally_compile

    module = importlib.import_module(f"vllm_omni.diffusion.models.{model_name}.{model_name}_transformer")
    torch.manual_seed(17)
    with torch.device("cuda"):
        model = (
            module.Flux2Transformer2DModel(
                in_channels=16,
                joint_attention_dim=32,
                num_layers=1,
                num_single_layers=1,
                num_attention_heads=4,
                attention_head_dim=128,
                timestep_guidance_channels=16,
                guidance_embeds=False,
                od_config=_diffusion_context,
            )
            .to(torch.bfloat16)
            .eval()
        )

    # vLLM's parallel linears allocate uninitialized weights.
    for name, parameter in model.named_parameters():
        if parameter.ndim > 1:
            parameter.normal_(0, 0.02)
        elif name.endswith("weight"):
            parameter.uniform_(0.5, 1.5)
        else:
            parameter.zero_()

    batch, txt, img = 2, 7, 17
    img_ids = torch.zeros(img, 4, device="cuda")
    img_ids[:, 1] = torch.arange(img, device="cuda")
    txt_ids = torch.zeros(txt, 4, device="cuda")
    txt_ids[:, 0] = torch.arange(txt, device="cuda")
    inputs = dict(
        hidden_states=torch.randn(batch, img, 16, device="cuda", dtype=torch.bfloat16),
        encoder_hidden_states=torch.randn(batch, txt, 32, device="cuda", dtype=torch.bfloat16),
        timestep=torch.full((batch,), 0.5, device="cuda"),
        img_ids=img_ids,
        txt_ids=txt_ids,
        return_dict=False,
    )
    attention = model.transformer_blocks[0].attn
    probe = inputs["hidden_states"].new_empty((batch, img, 4, 128))
    norm = attention.norm_q
    assert ir.ops.rms_norm.dispatch(probe, norm.weight, norm.variance_epsilon, None).provider == "vllm_c"

    disabled = str(batch * (txt + img) + 1)
    with (
        patch.object(module, "fused_joint_qkv_norm_rope", wraps=module.fused_joint_qkv_norm_rope) as joint,
        patch.object(module, "fused_qk_norm_rope", wraps=module.fused_qk_norm_rope) as single,
    ):
        monkeypatch.setenv(_MIN_TOKENS, disabled)
        eager_reference = model(**inputs)[0]
        joint.assert_not_called()
        single.assert_not_called()
        monkeypatch.setenv(_MIN_TOKENS, "0")
        eager_fused = model(**inputs)[0]
        joint.assert_called_once()
        single.assert_called_once()
    assert torch.isfinite(eager_reference).all()
    assert torch.equal(eager_fused.view(torch.int16), eager_reference.view(torch.int16))

    fused_ops = {"vllm_omni.fused_joint_qkv_norm_rope", "vllm_omni.fused_qk_norm_rope"}
    captured_ops: set[str] = set()

    def recording_inductor(graph, example_inputs):
        captured_ops.update(
            str(node.target).removesuffix(".default") for node in graph.graph.nodes if node.op == "call_function"
        )
        return compile_fx(graph, example_inputs)

    torch._dynamo.reset()
    try:
        regionally_compile(model, backend=recording_inductor, dynamic=True)
        monkeypatch.setenv(_MIN_TOKENS, disabled)
        compiled_reference = model(**inputs)[0]
        assert captured_ops and not (captured_ops & fused_ops)
        captured_ops.clear()
        monkeypatch.setenv(_MIN_TOKENS, "0")
        compiled_fused = model(**inputs)[0]
        assert fused_ops <= captured_ops
    finally:
        torch._dynamo.reset()

    # Inductor can change BF16 rounding in the surrounding block. Keep a
    # bounded comparison for the compiled arms while enforcing eager parity above.
    torch.testing.assert_close(compiled_reference, eager_reference, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(compiled_fused, compiled_reference, rtol=1e-2, atol=1e-2)
