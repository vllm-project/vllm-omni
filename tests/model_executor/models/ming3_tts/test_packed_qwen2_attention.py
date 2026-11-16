# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
import types

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2Model

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.ming3_tts.packed_qwen2 import PackedQwen2, PagedKV

pytestmark = [pytest.mark.core_model, pytest.mark.tts]


def _tiny_qwen2(*, sliding: bool) -> Qwen2Model:
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        use_sliding_window=sliding,
        sliding_window=4 if sliding else None,
        layer_types=(["sliding_attention", "full_attention"] if sliding else ["full_attention", "full_attention"]),
    )
    config._attn_implementation = "sdpa"
    return Qwen2Model(config).eval()


def _install_cpu_kernels(monkeypatch):
    import vllm_omni.model_executor.models.ming3_tts.packed_qwen2 as packed

    def rms_norm(out, hidden, weight, variance_epsilon):
        variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        out.copy_(hidden * torch.rsqrt(variance + variance_epsilon) * weight)

    def fused_add_rms_norm(hidden, residual, weight, variance_epsilon):
        residual.add_(hidden)
        variance = residual.float().pow(2).mean(dim=-1, keepdim=True)
        hidden.copy_(residual * torch.rsqrt(variance + variance_epsilon) * weight)

    class _C:
        @staticmethod
        def silu_and_mul(out, gate_up):
            gate, up = gate_up.split(out.shape[-1], dim=-1)
            out.copy_(F.silu(gate) * up)

    monkeypatch.setattr(packed.ops, "rms_norm", rms_norm)
    monkeypatch.setattr(packed.ops, "rotary_embedding", lambda *args, **kwargs: None)
    monkeypatch.setattr(packed.ops, "fused_add_rms_norm", fused_add_rms_norm)
    monkeypatch.setattr(packed.ops, "reshape_and_cache_flash", lambda *args, **kwargs: None)
    monkeypatch.setattr(packed.torch.ops, "_C", _C)


class _Recorder:
    def __init__(self) -> None:
        self.seen: list[dict] = []

    def __call__(self, **kwargs):
        self.seen.append(kwargs)
        return torch.zeros_like(kwargs["q"])


def _run_both_paths(packed: PackedQwen2, attn: _Recorder):
    tokens = 4
    hidden = torch.randn(tokens, packed.model.config.hidden_size)
    positions = torch.arange(tokens)
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32)
    packed(hidden, positions, cu_seqlens, tokens)
    keys, values = packed.new_paged_caches(blocks=1, block_size=16)
    packed(
        hidden,
        positions,
        cu_seqlens,
        tokens,
        PagedKV(
            key_caches=keys,
            value_caches=values,
            slot_mapping=torch.arange(tokens),
            seqused_k=torch.tensor([tokens], dtype=torch.int32),
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            max_seqlen_k=tokens,
        ),
    )
    assert len(attn.seen) == 4
    for call in attn.seen[:2]:
        assert set(call) >= {"q", "k", "v", "max_seqlen_q", "cu_seqlens_q", "max_seqlen_k", "cu_seqlens_k"}
    for call in attn.seen[2:]:
        assert call["block_table"] is not None
        assert call["num_splits"] == 1
        assert "cu_seqlens_k" not in call
    return attn.seen[2:]


@pytest.mark.cpu
def test_xpu_selection_rejects_cuda_flash_attn_and_passes_keywords(monkeypatch):
    from vllm.platforms import current_platform

    attn = _Recorder()
    module = types.ModuleType("vllm._xpu_ops")
    setattr(module, "xpu_ops", types.SimpleNamespace(flash_attn_varlen_func=attn))
    monkeypatch.setitem(sys.modules, "vllm._xpu_ops", module)
    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)

    class _RejectCudaFlashAttn:
        def find_spec(self, fullname, path, target=None):
            if fullname == "vllm.vllm_flash_attn" or fullname.startswith("vllm.vllm_flash_attn."):
                raise ImportError("CUDA flash attention")
            return None

    sys.modules.pop("vllm.vllm_flash_attn", None)
    sys.meta_path.insert(0, _RejectCudaFlashAttn())
    try:
        _install_cpu_kernels(monkeypatch)
        packed = PackedQwen2(_tiny_qwen2(sliding=True), max_positions=32)
        assert packed._attn is attn
        assert packed._fa_version == 2
        paged = _run_both_paths(packed, attn)
        assert paged[0]["max_seqlen_q"] == 1
        assert attn.seen[0]["window_size"] == [3, 0]
        assert attn.seen[1]["window_size"] == [-1, -1]
    finally:
        sys.meta_path.pop(0)


@pytest.mark.cpu
def test_cuda_selection_passes_keywords(monkeypatch):
    from vllm.platforms import current_platform

    attn = _Recorder()
    module = types.ModuleType("vllm.vllm_flash_attn")
    setattr(module, "flash_attn_varlen_func", attn)
    setattr(module, "is_fa_version_supported", lambda version: version == 3)
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", module)
    monkeypatch.setattr(current_platform, "is_xpu", lambda: False)
    _install_cpu_kernels(monkeypatch)
    packed = PackedQwen2(_tiny_qwen2(sliding=False), max_positions=32)
    assert packed._attn is attn
    assert packed._fa_version == 3
    paged = _run_both_paths(packed, attn)
    assert paged[0]["max_seqlen_q"] == 4


@hardware_test(res={"xpu": "B60"}, num_cards=1)
@pytest.mark.skipif(not hasattr(torch, "xpu") or not torch.xpu.is_available(), reason="XPU required")
@torch.inference_mode()
def test_packed_qwen2_matches_transformers_on_xpu():
    # AudioVAE Qwen2 shape. One-token paged steps stay on the decode path.
    torch.manual_seed(0)
    device = torch.device("xpu")
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=2,
        num_attention_heads=14,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=64,
        use_sliding_window=True,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
    )
    config._attn_implementation = "sdpa"
    model = Qwen2Model(config).eval().to(device=device, dtype=torch.bfloat16)
    packed = PackedQwen2(model, max_positions=64)
    tokens = 6
    hidden = torch.randn(tokens, config.hidden_size, device=device, dtype=torch.bfloat16)
    expected = model(inputs_embeds=hidden.unsqueeze(0), use_cache=False).last_hidden_state[0]
    uncached = packed(
        hidden,
        torch.arange(tokens, device=device),
        torch.tensor([0, tokens], device=device, dtype=torch.int32),
        tokens,
    )
    uncached_diff = (uncached - expected).abs().max().item()

    keys, values = packed.new_paged_caches(blocks=1, block_size=16)
    paged_rows = []
    for index in range(tokens):
        paged_rows.append(
            packed(
                hidden[index : index + 1],
                torch.tensor([index], device=device),
                torch.tensor([0, 1], device=device, dtype=torch.int32),
                1,
                PagedKV(
                    key_caches=keys,
                    value_caches=values,
                    slot_mapping=torch.tensor([index], device=device),
                    seqused_k=torch.tensor([index + 1], device=device, dtype=torch.int32),
                    block_table=torch.zeros(1, 1, device=device, dtype=torch.int32),
                    max_seqlen_k=index + 1,
                ),
            )
        )
    paged_diff = (torch.cat(paged_rows) - expected).abs().max().item()
    print(f"max_abs_uncached={uncached_diff:.6f} max_abs_paged={paged_diff:.6f}")
    torch.testing.assert_close(uncached, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(torch.cat(paged_rows), expected, atol=2e-2, rtol=2e-2)
