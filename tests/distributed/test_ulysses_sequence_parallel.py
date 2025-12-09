# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm.platforms import current_platform

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import (
    DiffusionParallelConfig,
    OmniDiffusionConfig,
    set_current_vllm_config,
)
from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
from vllm_omni.utils.system_utils import update_environment_variables


class TestAttentionModel(torch.nn.Module):
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
        attn_output = self.attention(q, k, v)

        # Reshape back and project
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)

        return output


class TestMultiLayerAttentionModel(torch.nn.Module):
    """Test model with multiple attention layers."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_size: int,
        hidden_size: int,
        causal: bool = True,
        num_kv_heads: int | None = None,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_sync: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList(
            [
                TestAttentionModel(
                    num_heads=num_heads,
                    head_size=head_size,
                    hidden_size=hidden_size,
                    causal=causal,
                    num_kv_heads=num_kv_heads,
                    scatter_idx=scatter_idx,
                    gather_idx=gather_idx,
                    use_sync=use_sync,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through multiple attention layers."""
        for layer in self.layers:
            hidden_states = hidden_states + layer(hidden_states)
        return hidden_states


@pytest.mark.parametrize(
    "test_model_cls",
    [
        TestAttentionModel,
        TestMultiLayerAttentionModel,
    ],
)
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("seq_len", [16, 32])
# @pytest.mark.parametrize("num_heads", [4, 8])
# @pytest.mark.parametrize("head_size", [32, 64])
# @pytest.mark.parametrize("causal", [True, False])
# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("use_sync", [True, False])
# @pytest.mark.parametrize("dynamic", [False, True])
# @pytest.mark.parametrize("use_compile", [False, True])
def test_ulysses_attention(
    test_model_cls: type[torch.nn.Module],
    batch_size: int,
    seq_len: int = 16,
    num_heads: int = 4,
    head_size: int = 32,
    dtype: torch.dtype = torch.float16,
    causal: bool = False,
    use_sync: bool = False,
    dynamic: bool = False,
    use_compile: bool = False,
):
    """Test Ulysses attention with various parameter combinations."""
    num_processes = 2
    ulysses_degree = 2  # Must match num_processes for this test
    ring_degree = 1
    sequence_parallel_size = ulysses_degree * ring_degree

    def run_torch_spawn(fn, nprocs):
        # need to use torch.mp.spawn otherwise will have problems with
        # torch.distributed and cuda
        torch.multiprocessing.spawn(
            fn,
            args=(
                num_processes,
                test_model_cls,
                batch_size,
                seq_len,
                num_heads,
                head_size,
                dtype,
                causal,
                use_sync,
                dynamic,
                use_compile,
                ulysses_degree,
                ring_degree,
                sequence_parallel_size,
            ),
            nprocs=nprocs,
        )

    run_torch_spawn(ulysses_attention_on_test_model, num_processes)


def ulysses_attention_on_test_model(
    local_rank: int,
    world_size: int,
    test_model_cls: type[torch.nn.Module],
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    causal: bool,
    use_sync: bool,
    dynamic: bool,
    use_compile: bool,
    ulysses_degree: int,
    ring_degree: int,
    sequence_parallel_size: int,
):
    """Run Ulysses attention test on a test model."""
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
            "MASTER_PORT": "12345",
        }
    )
    # Initialize distributed environment
    init_distributed_environment()

    # Set up OmniDiffusionConfig with Ulysses parallel config
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

    # Initialize model parallel with Ulysses
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
    with set_current_vllm_config(od_config):
        # Create model
        hidden_size = num_heads * head_size

        # Create model with appropriate parameters
        model_kwargs = {
            "num_heads": num_heads,
            "head_size": head_size,
            "hidden_size": hidden_size,
            "causal": causal,
            "num_kv_heads": None,
            "scatter_idx": 2,
            "gather_idx": 1,
            "use_sync": use_sync,
        }

        if test_model_cls == TestMultiLayerAttentionModel:
            model_kwargs["num_layers"] = 2

        model = test_model_cls(**model_kwargs)

        model = model.to(device).to(dtype)

        # Create input
        # In sequence parallel, each rank gets seq_len / sequence_parallel_size
        local_seq_len = seq_len // sequence_parallel_size
        hidden_states = torch.randn(
            (batch_size, local_seq_len, hidden_size),
            dtype=dtype,
            device=device,
        )

        if dynamic:
            torch._dynamo.mark_dynamic(hidden_states, 0)
            torch._dynamo.mark_dynamic(hidden_states, 1)

        # Compile model if requested
        if use_compile:
            model = torch.compile(model)

        # Run forward pass
        output = model(hidden_states)

        # Verify output shape
        assert output.shape == (batch_size, local_seq_len, hidden_size), (
            f"Output shape mismatch: expected {(batch_size, local_seq_len, hidden_size)}, got {output.shape}"
        )

        # Verify that Attention is using Ulysses
        if hasattr(model, "attention"):
            assert hasattr(model.attention, "use_ulysses"), "Attention should have use_ulysses attribute"
            assert model.attention.use_ulysses, "Attention should be using Ulysses"
        elif hasattr(model, "layers"):
            for i, layer in enumerate(model.layers):
                assert hasattr(layer.attention, "use_ulysses"), f"Layer {i} attention should have use_ulysses attribute"
                assert layer.attention.use_ulysses, f"Layer {i} attention should be using Ulysses"

        # Run backward pass to ensure gradients work
        loss = output.sum()
        loss.backward()

        print(
            f"Rank {local_rank}: Test passed with "
            f"batch_size={batch_size}, seq_len={seq_len}, "
            f"num_heads={num_heads}, head_size={head_size}, "
            f"dtype={dtype}, causal={causal}, use_sync={use_sync}, "
            f"dynamic={dynamic}, use_compile={use_compile}"
        )
