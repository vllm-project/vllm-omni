# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real H3 QKV/output MXFP4 kernels followed by the real MindIE-SD SLA."""

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"npu": "A5"}, num_cards=1)]


@pytest.fixture(scope="module")
def h3_npu_parallel(tmp_path_factory):
    torch_npu = pytest.importorskip("torch_npu")
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    if not torch.npu.is_available() or "Ascend950" not in torch.npu.get_device_name(0):
        pytest.skip("H3 MXFP4/SLA integration requires Ascend A5")
    torch.npu.set_device(0)
    rendezvous = tmp_path_factory.mktemp("h3-qat-hccl") / "rendezvous"
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0, distributed_init_method=rendezvous.as_uri(), backend="hccl"
    )
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        yield torch_npu
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize(
    "serialized,scale_alg,fake",
    [
        (False, 0, None),
        (True, 0, None),
        (False, 2, None),
        (True, 2, None),
        (False, 0, "e2m1_rne"),
        (False, 0, "mxfp4_c7"),
    ],
    ids=["ocp-online", "ocp-serialized", "uos-online", "uos-serialized", "fake-reference", "fake-c7"],
)
@torch.inference_mode()
def test_h3_mxfp4_linears_and_sla_match_representable_reference(
    h3_npu_parallel, monkeypatch, serialized, scale_alg, fake
):
    pytest.importorskip(
        "vllm_omni.diffusion.attention.backends.sla_attn",
        reason="Real SLA composition requires the SLA backend from PR #11089",
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_blocks as blocks
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config, NPUMxfp4LinearMethod
    from vllm_omni.quantization.mxfp4_fake_config import DiffusionMXFP4FakeConfig, MXFP4FakeLinearMethod

    torch_npu = h3_npu_parallel
    config = (
        DiffusionMXFP4FakeConfig(recipe=fake)
        if fake
        else DiffusionMXFP4Config(
            is_checkpoint_mxfp4_serialized=serialized, require_smooth_scale=serialized, mxfp4_scale_alg=scale_alg
        )
    )
    od_config = OmniDiffusionConfig(
        diffusion_attention_config=AttentionConfig(
            default={"backend": "FLASH_ATTN"},
            per_role={"self": {"backend": "SLA_ATTN", "sla": {"topk": 0.05}}},
        )
    )
    arch = blocks.MiniMaxH3DiTArchConfig(num_attention_heads=2, attention_head_dim=128, hidden_size=256)
    # No constructor or projection doubles: these are real vLLM parallel
    # linears, their real weight loaders and their real NPU quant methods.
    with set_current_diffusion_config(od_config), set_default_torch_dtype(torch.bfloat16), torch.device("npu"):
        attention = blocks.MiniMaxH3Attention(arch, config, prefix="blocks.0.attn")
    model = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    block = nn.Module()
    block.attn = attention
    model.blocks = nn.ModuleList([block])
    model.arch = arch
    model._serialized_mxfp4 = serialized

    values = {}
    # Native rows: [head0 Q,K,V, head1 Q,K,V]. All powers of two are
    # representable in both tested MX algorithms, so no QAT recipe is assumed.
    qkv = torch.tensor([1, 2, 4, 8, 16, 32], dtype=torch.float32).repeat_interleave(128)[:, None] / 64
    qkv = qkv.expand(768, 256).to(torch.bfloat16)
    for name, param in model.named_parameters():
        value = torch.ones(param.shape, dtype=param.dtype, device="cpu")
        if "sla_module.proj_l" in name:
            value.zero_()
        elif name.endswith("qkv_proj.weight"):
            value = torch.full_like(qkv, 4) if serialized else qkv
        elif name.endswith("out_proj.weight"):
            value.fill_(4 if serialized else 1 / 64)
        elif name.endswith("qkv_proj.weight_scale"):
            value = torch.arange(119, 125, dtype=torch.uint8).repeat_interleave(128)[:, None].expand(768, 8)
        elif name.endswith("out_proj.weight_scale"):
            value.fill_(119)
        values[name] = value
    loaded = model.load_weights(values.items())
    assert values.keys() <= loaded
    for linear in (attention.qkv_proj, attention.out_proj):
        assert isinstance(linear.quant_method, MXFP4FakeLinearMethod if fake else NPUMxfp4LinearMethod)
        linear.quant_method.process_weights_after_loading(linear)
        # torch-npu exposes FP4 as a kernel dtype enum; storage is packed
        # bytes, rather than a torch.dtype with one element per value.
        if fake:
            assert linear.weight.dtype == torch.bfloat16
        else:
            assert linear.weight.dtype in (torch.uint8, torch.int8)
            assert linear.weight.numel() * 2 == linear.output_size_per_partition * linear.input_size_per_partition
            assert linear.weight_scale.numel() > 0
    # MindIE-SD owns the projection dtype (its linear branch may use FP32).
    assert attention.sla_module.proj_l.weight.is_floating_point()
    assert isinstance(attention.sla_module.proj_l, nn.Linear)

    kernel = torch_npu.npu_quant_matmul
    calls = []

    def record_kernel(*args, **kwargs):
        calls.append((kwargs["x1_dtype"], kwargs["x2_dtype"]))
        return kernel(*args, **kwargs)

    monkeypatch.setattr(torch_npu, "npu_quant_matmul", record_kernel)
    x = torch.full((1536, 256), 1 / 64, dtype=torch.bfloat16, device="npu")
    lengths = torch.tensor([0, 1280, 1536], dtype=torch.int32, device="npu")
    output = attention(x, rope_table=None, cu_seqlens=lengths, max_seqlen=1280, packed_total=1536)
    torch.npu.synchronize()
    assert calls == ([] if fake else [(torch_npu.float4_e2m1fn_x2, torch_npu.float4_e2m1fn_x2)] * 2)
    assert output.shape == x.shape and output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    # With constant rows, every selected KV block has identical V. V for
    # the two heads is 1/4 and 2; output projection gives 128*(1/4+2)/64=4.5.
    torch.testing.assert_close(
        output[:1280].cpu(), torch.full((1280, 256), 4.5, dtype=torch.bfloat16), atol=0.04, rtol=0
    )
    assert torch.count_nonzero(output[1280:]) == 0


@pytest.mark.parametrize("recipe", ["e2m1_rne", "mxfp4_c7"])
@torch.inference_mode()
def test_fake_reference_rounding_matches_cpu_on_npu(h3_npu_parallel, recipe):
    from vllm_omni.quantization.mxfp4_fake_config import fake_quantize_mxfp4

    x = torch.zeros(4, 64, dtype=torch.bfloat16)
    x[:, :9] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6, 0.7])
    x[2, 8] = 0.2490234375
    x[3, 7] = 7.5
    x[1] *= -8
    x[:, 32:] = x[:, :32] / 16
    expected = fake_quantize_mxfp4(x, recipe)
    actual = fake_quantize_mxfp4(x.to("npu"), recipe).cpu()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.count_nonzero(actual != x) > 0


@pytest.mark.parametrize("recipe,expected", [("e2m1_rne", 36.0), ("mxfp4_c7", 36.25)])
@torch.inference_mode()
def test_h3_fake_linears_load_and_execute_without_sla(h3_npu_parallel, recipe, expected):
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_blocks as blocks
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
    from vllm_omni.quantization.mxfp4_fake_config import DiffusionMXFP4FakeConfig, MXFP4FakeLinearMethod

    config = DiffusionMXFP4FakeConfig(recipe=recipe)
    od_config = OmniDiffusionConfig(diffusion_attention_config=AttentionConfig(default={"backend": "FLASH_ATTN"}))
    arch = blocks.MiniMaxH3DiTArchConfig(num_attention_heads=2, attention_head_dim=128, hidden_size=256)
    with set_current_diffusion_config(od_config), set_default_torch_dtype(torch.bfloat16), torch.device("npu"):
        attention = blocks.MiniMaxH3Attention(arch, config, prefix="blocks.0.attn")
    model = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    block = nn.Module()
    block.attn = attention
    model.blocks = nn.ModuleList([block])
    model.arch = arch
    model._serialized_mxfp4 = False
    values = {}
    for name, param in model.named_parameters():
        value = torch.ones(param.shape, dtype=param.dtype, device="cpu")
        if name.endswith(("qkv_proj.weight", "out_proj.weight")):
            value.zero_()
            value[:, :2] = torch.tensor([6, 0.25])
        elif name.endswith("bias"):
            value.zero_()
        values[name] = value
    assert values.keys() <= model.load_weights(values.items())
    x = torch.zeros(2, 256, device="npu", dtype=torch.bfloat16)
    x[:, :2] = torch.tensor([6, 0.25], device="npu")
    for linear in (attention.qkv_proj, attention.out_proj):
        assert isinstance(linear.quant_method, MXFP4FakeLinearMethod)
        linear.quant_method.process_weights_after_loading(linear)
        output, _ = linear(x)
        torch.testing.assert_close(output.cpu(), torch.full_like(output.cpu(), expected), atol=0, rtol=0)
        assert linear.weight.dtype == torch.bfloat16
