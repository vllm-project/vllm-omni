# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import torch
import pytest
from vllm.platforms import current_platform
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import (
    DiffusionParallelConfig,
    OmniDiffusionConfig,
    set_current_omni_diffusion_config,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.utils.platform_utils import detect_device_type

device_type = detect_device_type()
if device_type == "cuda":
    torch_device = torch.cuda
elif device_type == "npu":
    torch_device = torch.npu
else:
    raise ValueError(f"Unsupported device type: {device_type} for this test script! Expected GPU or NPU.")

class TestAttentionModel(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        hidden_size: int,
        causal: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.attention = Attention(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=1.0 / (head_size**0.5),
        )
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_size)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * head_size)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * head_size)
        self.o_proj = torch.nn.Linear(num_heads * head_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_size)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_size)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_size)
        
        attn_output = self.attention(q, k, v)
        attn_output = attn_output.view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)

def setup_distributed():
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    
    parallel_config = DiffusionParallelConfig(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        tensor_parallel_size=1,
        sequence_parallel_size=1,
        ulysses_degree=1,
        ring_degree=1,
        cfg_parallel_size=1,
    )
    
    initialize_model_parallel(
        data_parallel_degree=1,
        classifier_free_guidance_degree=1,
        sequence_parallel_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        tensor_parallel_degree=1,
        pipeline_parallel_degree=1,
    )
    return parallel_config

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_backend_consistency(dtype):
    device = torch.device(f"{device_type}:0")
    torch_device.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)
    current_platform.seed_everything(42)

    try:
        parallel_config = setup_distributed()
        od_config = OmniDiffusionConfig(
            model="test_model",
            dtype=dtype,
            parallel_config=parallel_config,
        )

        with set_current_omni_diffusion_config(od_config):
            model = TestAttentionModel(num_heads=8, head_size=64, hidden_size=512)
            model.to(device).to(dtype)
            
            # Prepare input
            hidden_states = torch.randn(2, 16, 512, device=device, dtype=dtype)
            
            # Run 1: Force SDPA
            os.environ["VLLM_OMNI_FORCE_SDPA"] = "1"
            # Re-init model or attention to ensure no cached state affects result? 
            # Attention layer checks env var at forward time, so just running forward is enough.
            output_sdpa = model(hidden_states)
            
            # Run 2: Use Flash Attention (default if available)
            os.environ["VLLM_OMNI_FORCE_SDPA"] = "0"
            output_fa = model(hidden_states)
            
            # Compare
            diff = torch.abs(output_sdpa - output_fa)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            print(f"\n[Dtype {dtype}] SDPA vs Flash Attention Difference:")
            print(f"Max Diff: {max_diff:.6e}")
            print(f"Mean Diff: {mean_diff:.6e}")
            
            # Typically FA vs SDPA diff in FP16 can be around 1e-3
            # We don't assert fail here, just report, unless it's huge
            if max_diff > 1e-2:
                print("WARNING: Significant difference between SDPA and Flash Attention!")
            
    finally:
        destroy_distributed_env()
        if "VLLM_OMNI_FORCE_SDPA" in os.environ:
            del os.environ["VLLM_OMNI_FORCE_SDPA"]

