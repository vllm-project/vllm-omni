# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for Ulysses + Ring sequence-parallel attention correctness.

What is tested
--------------
* ``test_sequence_parallel`` verifies that the ``Attention`` layer produces
  numerically equivalent results whether the sequence is processed on a single
  rank (baseline, SP=1) or sharded across multiple ranks via Ulysses/Ring SP.
  The test spawns two separate multi-process runs with ``torch.multiprocessing.spawn``:
  1. Baseline   – world_size=1, ulysses_degree=1, ring_degree=1.
  2. SP run     – world_size=ulysses_degree*ring_degree, each rank holds a
                  contiguous slice of the full sequence.
  After both runs, rank-0 output tensors are compared element-wise with a
  tolerance appropriate for the dtype (bfloat16).

SP-plan hooks are NOT applied in this test
------------------------------------------
``ForwardContext.sp_plan_hooks_applied`` remains ``False``.  As a result,
``ForwardContext.sp_active`` falls back to the "no hooks" branch: SP is
considered active whenever ``parallel_config.sequence_parallel_size > 1``.
This makes the test self-contained and suitable for standalone CI runs that
do not exercise the full model-registry pipeline.
"""

import os
import pickle
import tempfile

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.parallel.allgather_kv import (
    AllGatherKVParallelAttention,
    _AllGatherKVCtx,
)
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import (
    DiffusionParallelConfig,
    OmniDiffusionConfig,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.platforms import current_omni_platform


def update_environment_variables(envs_dict: dict[str, str]):
    """Update multiple environment variables with logging."""
    for k, v in envs_dict.items():
        os.environ[k] = v


def seed_everything(seed: int):
    torch.manual_seed(seed)
    current_omni_platform.manual_seed(seed)


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


class _MockAllGatherSPGroup:
    def __init__(self, *, rank: int, gather_chunks: list[list[torch.Tensor]]) -> None:
        self.ulysses_group = object()
        self.ulysses_world_size = len(gather_chunks[0])
        self.ulysses_rank = rank
        self._gather_chunks = list(gather_chunks)
        self.gathered_input_shapes: list[tuple[int, ...]] = []

    def all_gather(self, input_: torch.Tensor, dim: int = 0, separate_tensors: bool = False):
        assert not separate_tensors
        self.gathered_input_shapes.append(tuple(input_.shape))
        chunks = self._gather_chunks.pop(0)
        return torch.cat(chunks, dim=dim)


def test_allgather_kv_slices_full_dense_mask_to_local_query_rows():
    rank = 1
    joint_len = 1
    img_seq_local = 2
    img_seq_full = 4
    key_chunks = [
        torch.full((1, img_seq_local, 1, 1), 10.0),
        torch.full((1, img_seq_local, 1, 1), 20.0),
    ]
    value_chunks = [
        torch.full((1, img_seq_local, 1, 1), 30.0),
        torch.full((1, img_seq_local, 1, 1), 40.0),
    ]
    strategy = AllGatherKVParallelAttention(
        _MockAllGatherSPGroup(rank=rank, gather_chunks=[key_chunks, value_chunks]),
    )

    query = torch.zeros((1, img_seq_local, 1, 1))
    key = key_chunks[rank]
    value = value_chunks[rank]
    joint = torch.ones((1, joint_len, 1, 1))
    mask = torch.arange((joint_len + img_seq_full) * (joint_len + img_seq_full))
    mask = (mask % 2 == 0).view(1, 1, joint_len + img_seq_full, joint_len + img_seq_full)
    metadata = AttentionMetadata(
        attn_mask=mask,
        joint_query=joint,
        joint_key=joint,
        joint_value=joint,
        joint_strategy="front",
    )

    q_local, k_full, v_full, metadata_local, _ = strategy.pre_attention(query, key, value, metadata)

    expected_rows = torch.cat(
        [
            mask[..., :joint_len, :],
            mask[
                ...,
                joint_len + rank * img_seq_local : joint_len + (rank + 1) * img_seq_local,
                :,
            ],
        ],
        dim=-2,
    )
    assert q_local.shape[1] == joint_len + img_seq_local
    assert k_full.shape[1] == joint_len + img_seq_full
    assert v_full.shape[1] == joint_len + img_seq_full
    torch.testing.assert_close(k_full[:, joint_len:], torch.cat(key_chunks, dim=1))
    torch.testing.assert_close(v_full[:, joint_len:], torch.cat(value_chunks, dim=1))
    assert metadata_local is not None
    assert torch.equal(metadata_local.attn_mask, expected_rows)


def test_allgather_kv_maps_full_attn_spans_to_local_query_rows():
    strategy = AllGatherKVParallelAttention(
        _MockAllGatherSPGroup(
            rank=1,
            gather_chunks=[
                [torch.zeros((1, 2, 1, 1)), torch.zeros((1, 2, 1, 1))],
                [torch.zeros((1, 2, 1, 1)), torch.zeros((1, 2, 1, 1))],
            ],
        ),
    )
    query = torch.zeros((1, 2, 1, 1))
    key = torch.zeros((1, 2, 1, 1))
    value = torch.zeros((1, 2, 1, 1))
    joint = torch.ones((1, 1, 1, 1))
    mask = torch.ones((1, 1, 5, 5), dtype=torch.bool)
    metadata = AttentionMetadata(
        attn_mask=mask,
        joint_query=joint,
        joint_key=joint,
        joint_value=joint,
        full_attn_spans=[
            [
                (0, 1),  # joint span: kept on every rank.
                (2, 5),  # image span crossing rank 0 and rank 1 image shards.
            ]
        ],
    )

    _, _, _, metadata_out, _ = strategy.pre_attention(query, key, value, metadata)

    assert metadata_out is not None
    assert metadata_out.full_attn_spans == [[(0, 1), (1, 3)]]


def test_allgather_kv_allows_empty_full_attn_spans():
    strategy = AllGatherKVParallelAttention(
        _MockAllGatherSPGroup(
            rank=0,
            gather_chunks=[
                [torch.zeros((1, 2, 1, 1)), torch.zeros((1, 2, 1, 1))],
                [torch.zeros((1, 2, 1, 1)), torch.zeros((1, 2, 1, 1))],
            ],
        ),
    )
    query = torch.zeros((1, 2, 1, 1))
    key = torch.zeros((1, 2, 1, 1))
    value = torch.zeros((1, 2, 1, 1))
    metadata = AttentionMetadata(full_attn_spans=[[]])

    _, _, _, metadata_out, _ = strategy.pre_attention(query, key, value, metadata)

    assert metadata_out is not None
    assert metadata_out.full_attn_spans == [[]]


def test_allgather_kv_gathers_compressed_kv_then_repeats_locally():
    rank = 0
    img_seq_local = 2
    kv_heads = 2
    repeat_num = 2
    q_heads = kv_heads * repeat_num
    key_chunks = [
        torch.arange(1 * img_seq_local * kv_heads * 1, dtype=torch.float32).view(1, img_seq_local, kv_heads, 1),
        torch.arange(100, 100 + 1 * img_seq_local * kv_heads * 1, dtype=torch.float32).view(
            1, img_seq_local, kv_heads, 1
        ),
    ]
    value_chunks = [chunk + 1000 for chunk in key_chunks]
    sp_group = _MockAllGatherSPGroup(rank=rank, gather_chunks=[key_chunks, value_chunks])
    strategy = AllGatherKVParallelAttention(sp_group)

    query = torch.zeros((1, img_seq_local, q_heads, 1))
    metadata = AttentionMetadata(extra={"kv_repeat_num": repeat_num})

    _, k_full, v_full, _, _ = strategy.pre_attention(query, key_chunks[rank], value_chunks[rank], metadata)

    assert sp_group.gathered_input_shapes == [
        (1, img_seq_local, kv_heads, 1),
        (1, img_seq_local, kv_heads, 1),
    ]
    assert k_full.shape == (1, img_seq_local * 2, q_heads, 1)
    assert v_full.shape == (1, img_seq_local * 2, q_heads, 1)
    expected_key = torch.cat(key_chunks, dim=1).repeat_interleave(repeat_num, dim=2)
    expected_value = torch.cat(value_chunks, dim=1).repeat_interleave(repeat_num, dim=2)
    torch.testing.assert_close(k_full, expected_key)
    torch.testing.assert_close(v_full, expected_value)


def test_allgather_kv_post_attention_returns_local_shard_no_joint():
    """post_attention with joint_len=0 returns the attention output unchanged."""
    strategy = AllGatherKVParallelAttention(
        _MockAllGatherSPGroup(
            rank=0,
            gather_chunks=[
                [torch.zeros((1, 2, 1, 1)), torch.zeros((1, 2, 1, 1))],
                [torch.zeros((1, 2, 1, 1)), torch.zeros((1, 2, 1, 1))],
            ],
        ),
    )
    attn_output = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).view(1, 2, 3, 4)
    ctx = _AllGatherKVCtx(
        name=strategy.name,
        joint_len=0,
        joint_strategy="front",
        img_seq_local=2,
    )

    out = strategy.post_attention(attn_output, ctx)

    assert out.shape == attn_output.shape
    torch.testing.assert_close(out, attn_output)


def test_allgather_kv_post_attention_returns_local_shard_front_joint():
    """post_attention with joint_strategy='front' keeps joint at the front.

    The attention output arrives as [joint, img_local]; post_attention splits
    it back into the same order so downstream layers see the joint prefix
    preserved.
    """
    strategy = AllGatherKVParallelAttention(
        _MockAllGatherSPGroup(
            rank=0,
            gather_chunks=[
                [torch.zeros((1, 2, 1, 1)), torch.zeros((1, 2, 1, 1))],
                [torch.zeros((1, 2, 1, 1)), torch.zeros((1, 2, 1, 1))],
            ],
        ),
    )
    joint_len = 1
    img_seq_local = 2
    joint_slice = torch.full((1, joint_len, 3, 4), 7.0)
    img_slice = torch.full((1, img_seq_local, 3, 4), 9.0)
    attn_output = torch.cat([joint_slice, img_slice], dim=1)
    ctx = _AllGatherKVCtx(
        name=strategy.name,
        joint_len=joint_len,
        joint_strategy="front",
        img_seq_local=img_seq_local,
    )

    out = strategy.post_attention(attn_output, ctx)

    assert out.shape == attn_output.shape
    torch.testing.assert_close(out, attn_output)
    torch.testing.assert_close(out[:, :joint_len], joint_slice)
    torch.testing.assert_close(out[:, joint_len:], img_slice)


def test_allgather_kv_post_attention_returns_local_shard_rear_joint():
    """post_attention with joint_strategy='rear' keeps joint at the back.

    The attention output arrives as [img_local, joint]; post_attention splits
    it back into the same order so the joint suffix is preserved.
    """
    strategy = AllGatherKVParallelAttention(
        _MockAllGatherSPGroup(
            rank=0,
            gather_chunks=[
                [torch.zeros((1, 2, 1, 1)), torch.zeros((1, 2, 1, 1))],
                [torch.zeros((1, 2, 1, 1)), torch.zeros((1, 2, 1, 1))],
            ],
        ),
    )
    joint_len = 1
    img_seq_local = 2
    joint_slice = torch.full((1, joint_len, 3, 4), 7.0)
    img_slice = torch.full((1, img_seq_local, 3, 4), 9.0)
    attn_output = torch.cat([img_slice, joint_slice], dim=1)
    ctx = _AllGatherKVCtx(
        name=strategy.name,
        joint_len=joint_len,
        joint_strategy="rear",
        img_seq_local=img_seq_local,
    )

    out = strategy.post_attention(attn_output, ctx)

    assert out.shape == attn_output.shape
    torch.testing.assert_close(out, attn_output)
    torch.testing.assert_close(out[:, :img_seq_local], img_slice)
    torch.testing.assert_close(out[:, img_seq_local:], joint_slice)


@pytest.mark.parametrize(
    "test_model_cls",
    [
        TestMultiLayerAttentionModel,
    ],
)
@pytest.mark.parametrize("ulysses_degree", [2])
@pytest.mark.parametrize("ring_degree", [2])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len", [16])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_size", [8])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("use_sync", [False])
@pytest.mark.parametrize("dynamic", [False])
@pytest.mark.parametrize("use_compile", [False])
def test_sequence_parallel(
    ulysses_degree: int,
    ring_degree: int,
    test_model_cls: type[torch.nn.Module],
    dtype: torch.dtype,
    causal: bool,
    use_sync: bool,
    dynamic: bool,
    use_compile: bool,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
):
    """Test Ulysses attention by comparing with and without SP enabled."""
    sequence_parallel_size = ulysses_degree * ring_degree

    # Skip if not enough GPUs available
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < sequence_parallel_size:
        pytest.skip(f"Test requires {sequence_parallel_size} GPUs but only {available_gpus} available")

    # Create temporary files to share results between processes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        baseline_output_file = f.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        sp_output_file = f.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        model_state_file = f.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        input_data_file = f.name

    try:
        # Step 1: Run without SP (baseline with ulysses_degree=1, ring_degree=1)
        print("\n[Baseline] Running without SP (ulysses_degree=1, ring_degree=1)...")
        torch.multiprocessing.spawn(
            ulysses_attention_on_test_model,
            args=(
                1,  # num_processes = 1 for baseline
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
                1,  # ulysses_degree = 1
                1,  # ring_degree = 1
                1,  # allgather_degree = 1 (baseline: no AllGather-KV SP)
                1,  # sequence_parallel_size = 1
                baseline_output_file,
                model_state_file,
                input_data_file,
                True,  # is_baseline
            ),
            nprocs=1,
        )

        # Step 2: Run with SP enabled
        print(f"\n[SP Test] Running with SP (ulysses_degree={ulysses_degree}, ring_degree={ring_degree})...")
        torch.multiprocessing.spawn(
            ulysses_attention_on_test_model,
            args=(
                sequence_parallel_size,  # num_processes
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
                1,  # allgather_degree = 1 (Ulysses/Ring test does not use AllGather-KV)
                sequence_parallel_size,
                sp_output_file,
                model_state_file,
                input_data_file,
                False,  # is_baseline
            ),
            nprocs=sequence_parallel_size,
        )

        # Step 3: Verify input consistency and compare outputs
        print(f"\n{'=' * 80}")
        print("Verifying input data consistency...")
        with open(input_data_file, "rb") as f:
            input_data = pickle.load(f)
        input_checksum = hash(input_data.tobytes())
        print(f"  Input data shape: {input_data.shape}")
        print(f"  Input data checksum: {input_checksum}")
        print("  ✓ Both baseline and SP used the same input data")

        print(f"\n{'=' * 80}")
        print("Comparing outputs between baseline and SP...")
        with open(baseline_output_file, "rb") as f:
            baseline_output = pickle.load(f)
        with open(sp_output_file, "rb") as f:
            sp_output = pickle.load(f)

        # Convert to tensors for comparison
        baseline_tensor = torch.tensor(baseline_output)
        sp_tensor = torch.tensor(sp_output)

        print(f"  Baseline output shape: {baseline_tensor.shape}")
        print(f"  SP output shape: {sp_tensor.shape}")
        assert baseline_tensor.shape == sp_tensor.shape, "Output shapes must match!"

        # Calculate differences
        abs_diff = torch.abs(baseline_tensor - sp_tensor)
        max_abs_diff = abs_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()

        # Calculate relative difference (avoid division by zero)
        baseline_abs = torch.abs(baseline_tensor)
        relative_diff = abs_diff / (baseline_abs + 1e-8)
        max_relative_diff = relative_diff.max().item()
        mean_relative_diff = relative_diff.mean().item()

        print(f"\n{'=' * 80}")
        print("Output Difference Analysis:")
        print(f"  - Max absolute difference: {max_abs_diff:.6e}")
        print(f"  - Mean absolute difference: {mean_abs_diff:.6e}")
        print(f"  - Max relative difference: {max_relative_diff:.6e}")
        print(f"  - Mean relative difference: {mean_relative_diff:.6e}")
        print(f"  - Baseline output range: [{baseline_tensor.min().item():.6e}, {baseline_tensor.max().item():.6e}]")
        print(f"  - SP output range: [{sp_tensor.min().item():.6e}, {sp_tensor.max().item():.6e}]")
        print(f"{'=' * 80}\n")

        # Assert that differences are within acceptable tolerance
        # For FP16/BF16, we expect some numerical differences due to different computation order under parallelism.
        # If we use the same backend (e.g. Flash Attention) for both baseline and SP, differences should be smaller.
        if dtype == torch.float16:
            atol, rtol = 5e-2, 5e-2  # Increased tolerance for Ring Attention
        elif dtype == torch.bfloat16:
            atol, rtol = 5e-2, 5e-2  # Increased tolerance for Ring Attention
        else:
            atol, rtol = 1e-5, 1e-4

        assert max_abs_diff < atol or max_relative_diff < rtol, (
            f"Output difference too large: max_abs_diff={max_abs_diff:.6e}, "
            f"max_relative_diff={max_relative_diff:.6e}, "
            f"tolerance: atol={atol}, rtol={rtol}"
        )

        print("✓ Test passed: SP output matches baseline within tolerance")

    finally:
        # Clean up temporary files
        for f in [baseline_output_file, sp_output_file, model_state_file, input_data_file]:
            if os.path.exists(f):
                os.remove(f)


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
    allgather_degree: int,
    sequence_parallel_size: int,
    output_file: str,
    model_state_file: str,
    input_data_file: str,
    is_baseline: bool,
):
    """Run Ulysses attention test on a test model and save results for comparison."""
    # Use fixed seed for reproducibility across baseline and SP runs
    RANDOM_SEED = 42
    seed_everything(RANDOM_SEED)

    if allgather_degree > 1:
        sp_kind = f"allgather={allgather_degree}"
    else:
        sp_kind = f"ulysses={ulysses_degree}, ring={ring_degree}"
    mode_str = "Baseline (no SP)" if is_baseline else f"SP ({sp_kind})"
    print(f"\n[{mode_str}] Rank {local_rank}/{world_size} - Random seed set to {RANDOM_SEED}")

    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
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

    # Set up OmniDiffusionConfig with parallel config
    parallel_config = DiffusionParallelConfig(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        tensor_parallel_size=1,
        sequence_parallel_size=sequence_parallel_size,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        allgather_degree=allgather_degree,
        cfg_parallel_size=1,
    )

    od_config = OmniDiffusionConfig(
        model="test_model",
        dtype=dtype,
        parallel_config=parallel_config,
    )

    # Initialize model parallel
    initialize_model_parallel(
        data_parallel_size=1,
        cfg_parallel_size=1,
        sequence_parallel_size=sequence_parallel_size,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        allgather_degree=allgather_degree,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    # Set the config so Attention can access it during init and forward
    with set_forward_context(omni_diffusion_config=od_config), set_current_diffusion_config(od_config):
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

        # For baseline: Generate and save model state and input data
        # This ensures both baseline and SP use exactly the same initialization
        if is_baseline and local_rank == 0:
            # Save model state for reuse (before any computation)
            model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            with open(model_state_file, "wb") as f:
                pickle.dump(model_state, f)

            full_hidden_states = torch.randn(
                (batch_size, seq_len, hidden_size),
                dtype=dtype,
                device="cpu",
            )
            with open(input_data_file, "wb") as f:
                pickle.dump(full_hidden_states.detach().cpu().float().numpy(), f)

            print("[Baseline] Saved model state and input data")

        # Synchronize to ensure baseline has saved data before SP loads it
        if world_size > 1:
            torch.distributed.barrier()

        # IMPORTANT: Both baseline and SP load the same model state and input data
        # This ensures exact same initialization and input for fair comparison
        with open(model_state_file, "rb") as f:
            model_state = pickle.load(f)
        model.load_state_dict({k: v.to(device).to(dtype) for k, v in model_state.items()})

        with open(input_data_file, "rb") as f:
            full_hidden_states_np = pickle.load(f)
        full_hidden_states = torch.from_numpy(full_hidden_states_np).to(device).to(dtype)

        print(f"[Rank {local_rank}] Loaded model state and full input data with shape {full_hidden_states.shape}")

        # Split input sequence according to sequence parallel BEFORE model forward
        # Each rank gets a contiguous chunk of the sequence dimension
        local_seq_len = seq_len // sequence_parallel_size
        start_idx = local_rank * local_seq_len
        end_idx = start_idx + local_seq_len
        hidden_states = full_hidden_states[:, start_idx:end_idx, :].contiguous()

        print(
            f"[Rank {local_rank}] Split input: local_seq_len={local_seq_len}, "
            f"indices=[{start_idx}:{end_idx}], local_shape={hidden_states.shape}"
        )

        if dynamic:
            torch._dynamo.mark_dynamic(hidden_states, 0)
            torch._dynamo.mark_dynamic(hidden_states, 1)

        # Compile model if requested
        if use_compile:
            model = torch.compile(model)

        # Run forward pass with local sequence chunk
        print(f"[Rank {local_rank}] Running forward pass...")
        output = model(hidden_states)
        print(f"[Rank {local_rank}] Forward pass completed, output shape: {output.shape}")

        # Verify output shape
        assert output.shape == (batch_size, local_seq_len, hidden_size), (
            f"Output shape mismatch: expected {(batch_size, local_seq_len, hidden_size)}, got {output.shape}"
        )

        # Gather outputs from all ranks AFTER computation
        if world_size > 1:
            print(f"[Rank {local_rank}] Gathering outputs from all {world_size} ranks...")
            # Gather all outputs to rank 0
            gathered_outputs = [torch.zeros_like(output) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_outputs, output)
            if local_rank == 0:
                # Concatenate along sequence dimension to reconstruct full sequence
                full_output = torch.cat(gathered_outputs, dim=1)
                print(f"[Rank 0] Gathered and concatenated outputs: {full_output.shape}")
                # Verify the full output shape matches expected
                assert full_output.shape == (batch_size, seq_len, hidden_size), (
                    f"Gathered output shape mismatch: expected {(batch_size, seq_len, hidden_size)}, "
                    f"got {full_output.shape}"
                )
            else:
                full_output = None
        else:
            # For baseline (world_size=1), output is already complete
            full_output = output
            print(f"[Rank 0] No gather needed (world_size=1), output shape: {full_output.shape}")

        # Save output from rank 0 for comparison
        if local_rank == 0:
            output_np = full_output.detach().cpu().float().numpy()
            with open(output_file, "wb") as f:
                pickle.dump(output_np, f)

            if allgather_degree > 1:
                sp_kind = f"allgather={allgather_degree}"
            else:
                sp_kind = f"ulysses={ulysses_degree}, ring={ring_degree}"
            mode_str = "baseline (no SP)" if is_baseline else f"SP ({sp_kind})"
            print(
                f"\n[{mode_str}] ✓ Saved output with shape {full_output.shape}:\n"
                f"  - batch_size={batch_size}, seq_len={seq_len}\n"
                f"  - num_heads={num_heads}, head_size={head_size}\n"
                f"  - dtype={dtype}, causal={causal}, use_sync={use_sync}\n"
            )

        destroy_distributed_env()


@pytest.mark.parametrize(
    "test_model_cls",
    [
        TestMultiLayerAttentionModel,
    ],
)
@pytest.mark.parametrize("allgather_degree", [2])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len", [16])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_size", [8])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("use_sync", [False])
@pytest.mark.parametrize("dynamic", [False])
@pytest.mark.parametrize("use_compile", [False])
def test_allgather_kv_sequence_parallel(
    allgather_degree: int,
    test_model_cls: type[torch.nn.Module],
    dtype: torch.dtype,
    causal: bool,
    use_sync: bool,
    dynamic: bool,
    use_compile: bool,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
):
    """Test AllGather-KV sequence-parallel attention by comparing with single-rank.

    AllGather-KV SP: each rank holds 1/P of Q (pre_processor sequence-split);
    AllGather collects full K/V on every rank; a single local attention kernel
    computes Q_local x K_full. Only valid for causal=False. v1 is mutually
    exclusive with Ulysses/Ring (ulysses_degree=1, ring_degree=1).

    The test spawns two separate multi-process runs:
    1. Baseline – world_size=1, allgather_degree=1 (single-rank, no SP).
    2. SP run  – world_size=allgather_degree, each rank holds a contiguous
                 slice of the sequence; full K/V is AllGathered on every rank.
    After both runs, rank-0 outputs are compared element-wise with bfloat16
    tolerance.
    """
    assert causal is False, "AllGather-KV SP is only valid for causal=False"
    sequence_parallel_size = allgather_degree

    # Skip if not enough GPUs available
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < sequence_parallel_size:
        pytest.skip(f"Test requires {sequence_parallel_size} GPUs but only {available_gpus} available")

    # Create temporary files to share results between processes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        baseline_output_file = f.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        sp_output_file = f.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        model_state_file = f.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        input_data_file = f.name

    try:
        # Step 1: Run without SP (baseline with allgather_degree=1)
        print("\n[Baseline] Running without SP (allgather_degree=1)...")
        torch.multiprocessing.spawn(
            ulysses_attention_on_test_model,
            args=(
                1,  # num_processes = 1 for baseline
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
                1,  # ulysses_degree = 1
                1,  # ring_degree = 1
                1,  # allgather_degree = 1 (baseline: no AllGather-KV SP)
                1,  # sequence_parallel_size = 1
                baseline_output_file,
                model_state_file,
                input_data_file,
                True,  # is_baseline
            ),
            nprocs=1,
        )

        # Step 2: Run with AllGather-KV SP enabled
        print(f"\n[SP Test] Running with AllGather-KV SP (allgather_degree={allgather_degree})...")
        torch.multiprocessing.spawn(
            ulysses_attention_on_test_model,
            args=(
                sequence_parallel_size,  # num_processes
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
                1,  # ulysses_degree = 1 (mutually exclusive with AllGather-KV)
                1,  # ring_degree = 1 (mutually exclusive with AllGather-KV)
                allgather_degree,
                sequence_parallel_size,
                sp_output_file,
                model_state_file,
                input_data_file,
                False,  # is_baseline
            ),
            nprocs=sequence_parallel_size,
        )

        # Step 3: Verify input consistency and compare outputs
        print(f"\n{'=' * 80}")
        print("Verifying input data consistency...")
        with open(input_data_file, "rb") as f:
            input_data = pickle.load(f)
        input_checksum = hash(input_data.tobytes())
        print(f"  Input data shape: {input_data.shape}")
        print(f"  Input data checksum: {input_checksum}")
        print("  ✓ Both baseline and SP used the same input data")

        print(f"\n{'=' * 80}")
        print("Comparing outputs between baseline and AllGather-KV SP...")
        with open(baseline_output_file, "rb") as f:
            baseline_output = pickle.load(f)
        with open(sp_output_file, "rb") as f:
            sp_output = pickle.load(f)

        # Convert to tensors for comparison
        baseline_tensor = torch.tensor(baseline_output)
        sp_tensor = torch.tensor(sp_output)

        print(f"  Baseline output shape: {baseline_tensor.shape}")
        print(f"  SP output shape: {sp_tensor.shape}")
        assert baseline_tensor.shape == sp_tensor.shape, "Output shapes must match!"

        # Calculate differences
        abs_diff = torch.abs(baseline_tensor - sp_tensor)
        max_abs_diff = abs_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()

        # Calculate relative difference (avoid division by zero)
        baseline_abs = torch.abs(baseline_tensor)
        relative_diff = abs_diff / (baseline_abs + 1e-8)
        max_relative_diff = relative_diff.max().item()
        mean_relative_diff = relative_diff.mean().item()

        print(f"\n{'=' * 80}")
        print("Output Difference Analysis:")
        print(f"  - Max absolute difference: {max_abs_diff:.6e}")
        print(f"  - Mean absolute difference: {mean_abs_diff:.6e}")
        print(f"  - Max relative difference: {max_relative_diff:.6e}")
        print(f"  - Mean relative difference: {mean_relative_diff:.6e}")
        print(f"  - Baseline output range: [{baseline_tensor.min().item():.6e}, {baseline_tensor.max().item():.6e}]")
        print(f"  - SP output range: [{sp_tensor.min().item():.6e}, {sp_tensor.max().item():.6e}]")
        print(f"{'=' * 80}\n")

        # Assert that differences are within acceptable tolerance
        # AllGather-KV computes attention on full K/V on every rank, so the
        # only numerical difference vs baseline comes from AllGather reordering
        # and the local FA kernel's reduction order.
        if dtype == torch.float16:
            atol, rtol = 5e-2, 5e-2
        elif dtype == torch.bfloat16:
            atol, rtol = 5e-2, 5e-2
        else:
            atol, rtol = 1e-5, 1e-4

        assert max_abs_diff < atol or max_relative_diff < rtol, (
            f"Output difference too large: max_abs_diff={max_abs_diff:.6e}, "
            f"max_relative_diff={max_relative_diff:.6e}, "
            f"tolerance: atol={atol}, rtol={rtol}"
        )

        print("✓ Test passed: AllGather-KV SP output matches baseline within tolerance")

    finally:
        # Clean up temporary files
        for f in [baseline_output_file, sp_output_file, model_state_file, input_data_file]:
            if os.path.exists(f):
                os.remove(f)
