# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bitwise checks of the thinker HF-numerics ops against the transformers modules."""

import os

import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3OmniMoeTextConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextModel,
    Qwen3OmniMoeThinkerTextRMSNorm,
    Qwen3OmniMoeThinkerTextRotaryEmbedding,
    Qwen3OmniMoeThinkerTextSparseMoeBlock,
    apply_rotary_pos_emb,
)
from vllm.config import KernelConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe import fused_moe as fused_moe_module
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.utils.torch_utils import set_default_torch_dtype

from tests.helpers.mark import hardware_marks, hardware_test
from vllm_omni.model_executor.models.qwen3_omni.hf_numerics import (
    HFNumericsDecoderLayer,
    HFNumericsRMSNorm,
    HFNumericsRotaryEmbedding,
    hf_add_deepstack,
    hf_add_rms_norm,
    hf_moe,
    hf_mrope,
    hf_rms_norm,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model]

_DTYPES = [torch.float32, torch.bfloat16]
_requires_cuda = pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="needs CUDA")
_DEVICES = [
    pytest.param("cpu", marks=pytest.mark.cpu),
    pytest.param("cuda", marks=[*hardware_marks(res={"cuda": "L4"}, num_cards=1), _requires_cuda]),
]


_TEXT_CONFIG = dict(
    hidden_size=32,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=8,
    norm_topk_prob=True,
    hidden_act="silu",
    rms_norm_eps=1e-6,
)


def _text_config() -> Qwen3OmniMoeTextConfig:
    return Qwen3OmniMoeTextConfig(
        **_TEXT_CONFIG,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [3, 3, 2]},
    )


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_rms_norm_matches_hf(device: str, dtype: torch.dtype):
    torch.manual_seed(0)
    hf = Qwen3OmniMoeThinkerTextRMSNorm(32, eps=1e-6).to(device=device, dtype=dtype)
    hf.weight.data.uniform_(0.5, 1.5)
    x = torch.randn(5, 32, dtype=dtype, device=device) * 3
    torch.testing.assert_close(hf_rms_norm(x, hf.weight, 1e-6), hf(x), rtol=0, atol=0)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_add_rms_norm_normalises_the_rounded_sum(device: str, dtype: torch.dtype):
    torch.manual_seed(0)
    hf = Qwen3OmniMoeThinkerTextRMSNorm(32, eps=1e-6).to(device=device, dtype=dtype)
    hf.weight.data.uniform_(0.5, 1.5)
    x = torch.randn(5, 32, dtype=dtype, device=device)
    residual = torch.randn(5, 32, dtype=dtype, device=device) * 40
    out, new_residual = hf_add_rms_norm(x, residual, hf.weight, 1e-6)
    torch.testing.assert_close(new_residual, x + residual, rtol=0, atol=0)
    torch.testing.assert_close(out, hf(x + residual), rtol=0, atol=0)


@pytest.mark.parametrize("device", _DEVICES)
def test_add_deepstack_matches_hf_deepstack_process(device: str):
    torch.manual_seed(0)
    n, dtype = 64, torch.bfloat16
    x = torch.randn(n, 32, dtype=dtype, device=device)
    residual = torch.randn(n, 32, dtype=dtype, device=device) * 40
    mask = torch.arange(n, device=device) % 3 != 0
    visual = torch.randn(int(mask.sum()), 32, dtype=dtype, device=device) * 5
    # The thinker passes a dense per-token tensor with zeros off the visual positions.
    deepstack = torch.zeros_like(x)
    deepstack[mask] = visual
    out = hf_add_deepstack(x, residual, deepstack)
    ref = Qwen3OmniMoeThinkerTextModel._deepstack_process(None, residual + x, mask, visual)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert not torch.equal((x + deepstack) + residual, ref)


@pytest.mark.cpu
def test_rms_norm_module_dispatches_to_the_ops():
    norm = HFNumericsRMSNorm(32, eps=1e-6)
    assert isinstance(norm, RMSNorm)
    x = torch.randn(3, 32)
    torch.testing.assert_close(norm(x), hf_rms_norm(x, norm.weight, 1e-6), rtol=0, atol=0)
    out, residual = norm(x, torch.ones_like(x))
    torch.testing.assert_close(residual, x + 1, rtol=0, atol=0)
    torch.testing.assert_close(out, hf_rms_norm(x + 1, norm.weight, 1e-6), rtol=0, atol=0)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("positions_dim", [1, 2])
def test_mrope_matches_hf_rotary(device: str, dtype: torch.dtype, positions_dim: int):
    torch.manual_seed(0)
    config = _text_config()
    n, heads, kv_heads, d = 7, 4, 2, 16
    q = torch.randn(n, heads * d, dtype=dtype, device=device)
    k = torch.randn(n, kv_heads * d, dtype=dtype, device=device)
    if positions_dim == 1:
        positions = torch.arange(n, device=device)
        hf_positions = positions[None]
    else:
        positions = torch.stack([torch.arange(n), torch.arange(n) // 2, torch.arange(n) % 3]).to(device)
        hf_positions = positions[:, None, :]
    hf_rope = Qwen3OmniMoeThinkerTextRotaryEmbedding(config, device=device)
    cos, sin = hf_rope(q, hf_positions)
    q_ref, k_ref = apply_rotary_pos_emb(
        q.view(1, n, heads, d).transpose(1, 2), k.view(1, n, kv_heads, d).transpose(1, 2), cos, sin
    )
    rope = HFNumericsRotaryEmbedding(config).to(device)
    q_out, k_out = rope(positions, q, k)
    torch.testing.assert_close(q_out.view(n, heads, d), q_ref[0].transpose(0, 1), rtol=0, atol=0)
    torch.testing.assert_close(k_out.view(n, kv_heads, d), k_ref[0].transpose(0, 1), rtol=0, atol=0)
    torch.testing.assert_close(hf_mrope(positions, q, k, rope.inv_freq, [3, 3, 2], d)[0], q_out, rtol=0, atol=0)


_MROPE_UNDER_BATCH_INVARIANCE = """
import os, torch
os.environ["VLLM_BATCH_INVARIANT"] = "1"
from vllm.model_executor.determinism.batch_invariant import init_batch_invariance
init_batch_invariance()
from transformers import Qwen3OmniMoeTextConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import apply_rotary_pos_emb
from vllm_omni.model_executor.models.qwen3_omni.hf_numerics import HFNumericsRotaryEmbedding
torch.manual_seed(0)
config = Qwen3OmniMoeTextConfig(hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [3, 3, 2]})
n, heads, kv_heads, d = 64, 4, 2, 16
q = torch.randn(n, heads * d, dtype=torch.bfloat16, device="cuda")
k = torch.randn(n, kv_heads * d, dtype=torch.bfloat16, device="cuda")
positions = torch.arange(3000, 3000 + n, device="cuda")
rope = HFNumericsRotaryEmbedding(config).cuda()
freqs = positions[:, None].float() * rope.inv_freq
freqs = torch.cat((freqs, freqs), dim=-1)
cos, sin = freqs.cos().to(q.dtype)[None], freqs.sin().to(q.dtype)[None]
q_ref, _ = apply_rotary_pos_emb(q.view(1, n, heads, d).transpose(1, 2), k.view(1, n, kv_heads, d).transpose(1, 2), cos, sin)
q_out, _ = rope(positions, q, k)
torch.testing.assert_close(q_out.view(n, heads, d), q_ref[0].transpose(0, 1), rtol=0, atol=0)
"""


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@_requires_cuda
def test_mrope_matches_hf_rotary_under_batch_invariance():
    # vLLM's batch-invariant mode overrides bmm with a Triton kernel that runs fp32 matmuls at TF32 precision; the
    # frequency product must stay an exact elementwise multiply. The mode is process-global, hence the subprocess.
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-c", _MROPE_UNDER_BATCH_INVARIANCE], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@_requires_cuda
def test_rotary_inv_freq_matches_hf_when_built_on_cuda():
    # vLLM builds the model under a CUDA default device, while Transformers computes inv_freq on the CPU; the two
    # pow implementations differ in the last bit for some frequencies at the thinker's head_dim and rope_theta.
    config = Qwen3OmniMoeTextConfig(
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0, "mrope_section": [24, 20, 20]},
    )
    with torch.device("cuda"):
        rope = HFNumericsRotaryEmbedding(config)
    hf_inv_freq = Qwen3OmniMoeThinkerTextRotaryEmbedding(config).inv_freq
    assert rope.inv_freq.device.type == "cuda"
    torch.testing.assert_close(rope.inv_freq.cpu(), hf_inv_freq, rtol=0, atol=0)


@pytest.mark.cpu
def test_rotary_rejects_non_default_rope_type():
    config = _text_config()
    config.rope_parameters["rope_type"] = "yarn"
    with pytest.raises(ValueError, match="rope_type"):
        HFNumericsRotaryEmbedding(config)


def _hf_moe_block(dtype: torch.dtype, experts_implementation: str = "grouped_mm"):
    config = _text_config()
    config._experts_implementation = experts_implementation
    block = Qwen3OmniMoeThinkerTextSparseMoeBlock(config).to(device="cuda", dtype=dtype)
    for p in block.parameters():
        p.data.normal_(0, 0.2)
    return block


def _hf_router_then_fused_experts(block, x: torch.Tensor) -> torch.Tensor:
    """What a trainer computes with the HF router and ``fused_experts`` registered as its experts."""
    _, top_k_weights, top_k_index = block.gate(x)
    return fused_experts(
        x, block.experts.gate_up_proj, block.experts.down_proj, top_k_weights.float(), top_k_index.to(torch.int32)
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@_requires_cuda
def test_moe_matches_hf_router_with_fused_experts():
    # bf16 only: the default fused_experts config for fp32 needs more shared memory than an L4 has.
    dtype = torch.bfloat16
    torch.manual_seed(0)
    block = _hf_moe_block(dtype)
    x = torch.randn(9, 32, dtype=dtype, device="cuda")
    out = hf_moe(x, block.gate.weight, block.experts.gate_up_proj, block.experts.down_proj, 2, True)
    torch.testing.assert_close(out, _hf_router_then_fused_experts(block, x), rtol=0, atol=0)
    torch.testing.assert_close(out, block(x[None])[0], rtol=2e-2, atol=2e-2)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@_requires_cuda
def test_moe_rows_do_not_depend_on_the_batch(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    fused_moe_module.get_moe_configs.cache_clear()
    torch.manual_seed(0)
    block = _hf_moe_block(torch.bfloat16)
    x = torch.randn(256, 32, dtype=torch.bfloat16, device="cuda")

    def moe(rows):
        return hf_moe(rows, block.gate.weight, block.experts.gate_up_proj, block.experts.down_proj, 2, True)

    whole = moe(x)
    for chunk in (1, 8, 32):
        torch.testing.assert_close(
            torch.cat([moe(x[i : i + chunk]) for i in range(0, 256, chunk)]), whole, rtol=0, atol=0
        )


@pytest.fixture(scope="module")
def _dist():
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29531")
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
        initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


def _build_layer(tmp_path, moe_backend: str = "triton", dtype: torch.dtype = torch.bfloat16, enforce_eager=False):
    Qwen3MoeConfig(
        **_TEXT_CONFIG,
        architectures=["Qwen3MoeForCausalLM"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [3, 3, 2]},
    ).save_pretrained(tmp_path)
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path), dtype=dtype, skip_tokenizer_init=True, enforce_eager=enforce_eager
        ),
        kernel_config=KernelConfig(moe_backend=moe_backend),
    )
    with set_current_vllm_config(vllm_config), set_default_torch_dtype(dtype), torch.device("cuda"):
        return HFNumericsDecoderLayer(vllm_config, prefix="model.layers.0")


def _load_hf_experts(layer: HFNumericsDecoderLayer, hf_block) -> None:
    layer.mlp.gate.weight.data.copy_(hf_block.gate.weight)
    experts = layer.mlp.experts.routed_experts
    gate_up, down = hf_block.experts.gate_up_proj, hf_block.experts.down_proj
    inter = down.shape[-1]
    for e in range(gate_up.shape[0]):
        experts.weight_loader(experts.w13_weight, gate_up[e, :inter], "experts.w13_weight", "w1", e)
        experts.weight_loader(experts.w13_weight, gate_up[e, inter:], "experts.w13_weight", "w3", e)
        experts.weight_loader(experts.w2_weight, down[e], "experts.w2_weight", "w2", e)
    experts.quant_method.process_weights_after_loading(experts)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@_requires_cuda
def test_triton_loaded_moe_matches_hf_router_with_fused_experts(_dist, tmp_path):
    torch.manual_seed(0)
    layer = _build_layer(tmp_path, enforce_eager=True)
    hf_block = _hf_moe_block(torch.bfloat16)
    _load_hf_experts(layer, hf_block)
    x = torch.randn(9, 32, dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(layer.mlp(x), _hf_router_then_fused_experts(hf_block, x), rtol=0, atol=0)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@_requires_cuda
def test_flashinfer_cutlass_moe_backend_is_rejected(_dist, tmp_path, monkeypatch):
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import UnquantizedMoeBackend

    # FlashInfer CUTLASS is not available on every GPU, so report it as the selected backend.
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method.select_unquantized_moe_backend",
        lambda moe_config: (UnquantizedMoeBackend.FLASHINFER_CUTLASS, TritonExperts),
    )
    with pytest.raises(ValueError, match="needs the Triton MoE backend, got FlashInfer CUTLASS"):
        _build_layer(tmp_path, enforce_eager=True)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@_requires_cuda
def test_moe_capture_replay_matches_eager(_dist, tmp_path):
    torch.manual_seed(0)
    layer = _build_layer(tmp_path)
    _load_hf_experts(layer, _hf_moe_block(torch.bfloat16))
    static_x = torch.randn(9, 32, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        layer.mlp(static_x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = layer.mlp(static_x)
    for _ in range(3):
        x = torch.randn(9, 32, dtype=torch.bfloat16, device="cuda")
        static_x.copy_(x)
        graph.replay()
        torch.testing.assert_close(static_out, layer.mlp(x), rtol=0, atol=0)
