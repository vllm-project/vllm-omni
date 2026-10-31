# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real CUDA graph/FA varlen parity and reuse across different B=1 lengths."""

from dataclasses import dataclass

import pytest
import torch
from torch import nn
from vllm.triton_utils import HAS_TRITON

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.yue2 import yue2 as y
from vllm_omni.model_executor.models.yue2.cuda_graph_wrapper import PrefillGraphWrapper
from vllm_omni.model_executor.models.yue2.kv_slot_pool import KVSlotPool, round_token_capacity

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required"),
]


@dataclass
class _Config:
    num_attention_heads: int = 2
    num_key_value_heads: int = 1
    head_dim: int = 32


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(64)
        self.post_attention_layernorm = nn.RMSNorm(64)
        self.self_attn = nn.Module()
        self.self_attn.num_heads = 2
        self.self_attn.num_kv_heads = 1
        self.self_attn.head_dim = 32
        self.self_attn.qkv_proj = nn.Linear(64, 128, bias=False)
        self.self_attn.q_norm = nn.RMSNorm(32)
        self.self_attn.k_norm = nn.RMSNorm(32)
        self.self_attn.o_proj = nn.Linear(64, 64, bias=False)
        self.mlp = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 64))


class _Model(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.config = _Config()
        self.max_position_embeddings = 128
        self._sampler_capacity = 1
        self.vae2llm = nn.Linear(64, 64)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(128, 64)
        self.model.layers = nn.ModuleList([_Layer() for _ in range(3)])
        self.rotary_emb = y._RotaryEmbedding(32, 10000.0)
        self.to(device="cuda", dtype=dtype)

    def rotary(self, positions):
        return self.rotary_emb(positions)

    def ar_project(self, layer, x, cos, sin):
        return y.Yue2ForCausalLM.ar_project(self, layer, x, cos, sin)

    def prefill_graph_wrapper(self):
        return None  # The oracle always uses the original eager prefill.


@pytest.mark.parametrize("end_only", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_graph_reuses_bucket_for_different_actual_lengths(end_only, dtype, monkeypatch):
    with torch.cuda.stream(torch.cuda.Stream()):
        torch.manual_seed(17)
        model = _Model(dtype)
        wrapper = PrefillGraphWrapper.create(model)
        if wrapper is None:
            pytest.skip("vLLM FlashAttention unavailable")
        pool = KVSlotPool(
            layers=3, kv_heads=1, head_dim=32, maximum=128, max_slots=1, device=torch.device("cuda"), dtype=dtype
        )
        saved_graph = None
        saved_slot = None
        for length in [9, 13]:
            tokens = list(range(1, length)) + [127]
            reference = object.__new__(y.CachedNAR)
            reference.model = model
            reference.device, reference.dtype, reference.fa3 = torch.device("cuda"), dtype, None
            reference.ar_length, reference.nar_length = length, 10
            reference.chunk = y.Chunk(tokens, torch.empty(8, 64))
            reference._prefill()
            lease = pool.acquire(length + 10)
            assert lease is not None
            lease.slot.keys.fill_(float("nan"))
            lease.slot.values.fill_(float("nan"))
            start = length - 1 if end_only else 0
            if end_only:
                lease.slot.keys[:, :start].copy_(reference.keys[:, :start])
                lease.slot.values[:, :start].copy_(reference.values[:, :start])
            bucket = round_token_capacity(length - start, wrapper.maximum)
            if saved_graph is not None:
                assert lease.slot is saved_slot
                assert wrapper.graphs[lease.slot][bucket].graph is saved_graph
                # A hit must replay the old graph without executing Python forward.
                with monkeypatch.context() as patch:
                    patch.setattr(wrapper, "_forward", lambda *args: pytest.fail("unexpected recapture"))
                    wrapper.prefill(lease, tokens[start:], start)
            else:
                wrapper.prefill(lease, tokens[start:], start)
            saved_slot = lease.slot
            saved_graph = wrapper.graphs[lease.slot][bucket].graph
            keys, values = lease.views(length + 10)
            torch.testing.assert_close(keys[:, :length], reference.keys[:, :length], rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(values[:, :length], reference.values[:, :length], rtol=1e-2, atol=1e-2)
            assert torch.isnan(keys[:, length:]).all()  # Padding must never overwrite NAR.
            assert torch.isnan(values[:, length:]).all()
            torch.cuda.current_stream().synchronize()
            lease.release()
