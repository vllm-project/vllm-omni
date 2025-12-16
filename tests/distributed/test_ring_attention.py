# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import pytest
import torch
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


def update_environment_variables(envs_dict: dict[str, str]):
    """Update multiple environment variables with logging."""
    for k, v in envs_dict.items():
        os.environ[k] = v


class MockAttentionModel(torch.nn.Module):
    """Test model using Attention layer."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        hidden_size: int,
        causal: bool = False,
        num_kv_heads: int | None = None,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_sync: bool = False,
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
            num_kv_heads=num_kv_heads,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            use_sync=use_sync,
        )
        # Linear projection layers for Q, K, V
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_size)
        self.k_proj = torch.nn.Linear(hidden_size, (num_kv_heads or num_heads) * head_size)
        self.v_proj = torch.nn.Linear(hidden_size, (num_kv_heads or num_heads) * head_size)
        self.o_proj = torch.nn.Linear(num_heads * head_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention layer."""
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to (batch_size, seq_len, num_heads, head_size)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_size)
        k = k.view(batch_size, seq_len, k.shape[-1] // self.head_size, self.head_size)
        v = v.view(batch_size, seq_len, v.shape[-1] // self.head_size, self.head_size)

        # Apply attention
        print(f"Rank {torch.distributed.get_rank()}: q shape before attention: {q.shape}")
        attn_output = self.attention(q, k, v)
        print(f"Rank {torch.distributed.get_rank()}: attn_output shape: {attn_output.shape}")
        # Reshape back and project
        # attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_size)
        print(f"Rank {torch.distributed.get_rank()}: attn_output reshaped: {attn_output.shape}")
        output = self.o_proj(attn_output)

        return output


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len", [16])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_size", [32])
@pytest.mark.parametrize("causal", [False]) # Ring attention typically non-causal for DiT, but supports causal
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_sync", [True, False])
@pytest.mark.parametrize("ring_degree", [2])
@pytest.mark.parametrize("ulysses_degree", [2])
def test_ring_attention(
    dtype: torch.dtype,
    causal: bool,
    use_sync: bool,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    ring_degree: int,
    ulysses_degree: int,
):
    """Test Ring attention."""
    num_processes = ring_degree * ulysses_degree
    if torch.cuda.device_count() < num_processes:
        pytest.skip(f"Need {num_processes} GPUs")
        
    sequence_parallel_size = num_processes

    def run_torch_spawn(fn, nprocs):
        torch.multiprocessing.spawn(
            fn,
            args=(
                num_processes,
                batch_size,
                seq_len,
                num_heads,
                head_size,
                dtype,
                causal,
                use_sync,
                ulysses_degree,
                ring_degree,
                sequence_parallel_size,
            ),
            nprocs=nprocs,
        )

    run_torch_spawn(ring_attention_on_test_model, num_processes)


def ring_attention_on_test_model(
    local_rank: int,
    world_size: int,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    causal: bool,
    use_sync: bool,
    ulysses_degree: int,
    ring_degree: int,
    sequence_parallel_size: int,
):
    """Run Ring attention test on a test model."""
    current_platform.seed_everything(42)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12346", # Different port from other tests
        }
    )
    # Initialize distributed environment
    init_distributed_environment()

    # Set up OmniDiffusionConfig
    parallel_config = DiffusionParallelConfig(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        tensor_parallel_size=1,
        sequence_parallel_size=sequence_parallel_size,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        cfg_parallel_size=1,
    )

    od_config = OmniDiffusionConfig(
        model="test_model",
        dtype=dtype,
        parallel_config=parallel_config,
    )

    # Initialize model parallel
    initialize_model_parallel(
        data_parallel_degree=1,
        classifier_free_guidance_degree=1,
        sequence_parallel_degree=sequence_parallel_size,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        tensor_parallel_degree=1,
        pipeline_parallel_degree=1,
    )

    # Set the config so Attention can access it
    with set_current_omni_diffusion_config(od_config):
        hidden_size = num_heads * head_size
        model = MockAttentionModel(
            num_heads=num_heads,
            head_size=head_size,
            hidden_size=hidden_size,
            causal=causal,
            use_sync=use_sync,
        )

        model = model.to(device).to(dtype)

        # Create input
        # In sequence parallel, each rank gets seq_len / sequence_parallel_size
        local_seq_len = seq_len // sequence_parallel_size
        hidden_states = torch.randn(
            (batch_size, local_seq_len, hidden_size),
            dtype=dtype,
            device=device,
        )
        hidden_states.requires_grad = True

        # Run forward pass
        output = model(hidden_states)

        # Verify output shape
        assert output.shape == (batch_size, local_seq_len, hidden_size), (
            f"Output shape mismatch: expected {(batch_size, local_seq_len, hidden_size)}, got {output.shape}"
        )

        # Verify attributes
        if ring_degree > 1:
            assert model.attention.use_ring, "Attention should be using Ring"
        if ulysses_degree > 1:
            assert model.attention.use_ulysses, "Attention should be using Ulysses"

        # Run backward pass to ensure gradients work
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        assert hidden_states.grad is not None
        assert not torch.isnan(hidden_states.grad).any()

        print(
            f"Rank {local_rank}: Test passed with "
            f"ring_degree={ring_degree}, ulysses_degree={ulysses_degree}"
        )
        destroy_distributed_env()


