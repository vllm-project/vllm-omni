# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real FA3 graph replay must preserve the decode GQA layout."""

from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch
from transformers import Qwen2Config
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.utils import AttentionGroup

from tests.helpers.mark import hardware_test
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

pytestmark = [pytest.mark.core_model]


@dataclass
class _AttentionScales:
    _q_scale: torch.Tensor
    _k_scale: torch.Tensor
    _v_scale: torch.Tensor


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("graph_mode", [CUDAGraphMode.FULL_AND_PIECEWISE, CUDAGraphMode.FULL_DECODE_ONLY])
def test_full_decode_graph_matches_fa3_aot_schedule(graph_mode, tmp_path):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires FA3 on Hopper or newer")
    torch.manual_seed(17)
    device = torch.device("cuda:0")
    c = VllmConfig()
    Qwen2Config(
        architectures=["Qwen2ForCausalLM"],
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=2,
        max_position_embeddings=2048,
    ).save_pretrained(tmp_path)
    c.model_config = ModelConfig(model=str(tmp_path), skip_tokenizer_init=True, dtype="bfloat16")
    c.scheduler_config.max_num_seqs = 64
    c.compilation_config.cudagraph_mode = graph_mode
    c.compilation_config.max_cudagraph_capture_size = 128
    c.attention_config.flash_attn_version = 3
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.bfloat16)
    kvconfig = KVCacheConfig(num_blocks=1024, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["attn"], spec)])
    state = OmniModelState.__new__(OmniModelState)
    state.vllm_config = c
    state.model_config = c.model_config
    state.supports_mm_inputs = False
    with set_current_vllm_config(c):
        group = AttentionGroup(FlashAttentionBackend, ["attn"], spec, 0)
        group.create_metadata_builders(c, device)
        impl = FlashAttentionImpl(16, 64, 64**-0.5, 2, None, None, "auto")
        assert impl.vllm_flash_attn_version == 3
        buffers = InputBuffers(64, 128, device)
        batch = InputBatch.make_dummy(8, 8, buffers, max_query_len=None)
        tables = torch.arange(1024, dtype=torch.int32, device=device).reshape(8, 128)
        slots = torch.zeros((1, 8), dtype=torch.int64, device=device)
        cache = torch.randn(1024, 2, 16, 128, device=device, dtype=torch.bfloat16)
        q = torch.randn(8, 16, 64, device=device, dtype=torch.bfloat16)
        k = v = torch.empty(8, 2, 64, device=device, dtype=torch.bfloat16)
        layer = _AttentionScales(
            _q_scale=torch.ones((), device=device),
            _k_scale=torch.ones((), device=device),
            _v_scale=torch.ones((), device=device),
        )
        md = state.prepare_attn(batch, CUDAGraphMode.FULL, (tables,), slots, [[group]], kvconfig, for_capture=True)[
            "attn"
        ]
        out = torch.full_like(q, float("nan"))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                impl.forward(layer, q, k, v, cache, md, out)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            impl.forward(layer, q, k, v, cache, md, out)
        lengths = np.arange(37, 45, dtype=np.int32)
        batch.seq_lens.copy_(torch.tensor(lengths, device=device))
        runtime = replace(batch, max_query_len=1, seq_lens_cpu_upper_bound=torch.tensor(lengths))
        eager_md = state.prepare_attn(runtime, CUDAGraphMode.FULL, (tables,), slots, [[group]], kvconfig)["attn"]
        expected = torch.empty_like(q)
        impl.forward(layer, q, k, v, cache, eager_md, expected)
        state.prepare_attn(runtime, CUDAGraphMode.FULL, (tables,), slots, [[group]], kvconfig)
        graph.replay()
        torch.accelerator.synchronize()
        assert md.max_query_len == 1
        torch.testing.assert_close(out, expected, atol=0.002, rtol=0.002)
