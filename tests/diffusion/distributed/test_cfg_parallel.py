# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CFG (Classifier-Free Guidance) parallel functionality.

This test verifies that predict_noise_maybe_with_cfg produces numerically
equivalent results with and without CFG parallel using fixed random inputs.
"""

import os

import pytest
import torch

from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.platforms import current_omni_platform


def update_environment_variables(envs_dict: dict[str, str]):
    """Update multiple environment variables."""
    for k, v in envs_dict.items():
        os.environ[k] = v


class SimpleTransformer(torch.nn.Module):
    """Simple transformer model for testing with random initialization.

    Contains:
    - Input projection (conv to hidden_dim)
    - QKV projection layers
    - Self-attention layer
    - Output projection
    """

    def __init__(self, in_channels: int = 4, hidden_dim: int = 128, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Input projection: (B, C, H, W) -> (B, hidden_dim, H, W)
        self.input_proj = torch.nn.Conv2d(in_channels, hidden_dim, 1)

        # QKV projection layers
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim)

        # Output projection after attention
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim)

        # Final output projection: (B, hidden_dim, H, W) -> (B, C, H, W)
        self.final_proj = torch.nn.Conv2d(hidden_dim, in_channels, 1)

        # Layer norm
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor]:
        """Forward pass with self-attention.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Input projection
        x = self.input_proj(x)  # (B, hidden_dim, H, W)

        # Reshape to sequence: (B, hidden_dim, H, W) -> (B, H*W, hidden_dim)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, hidden_dim)

        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)

        # QKV projection
        q = self.q_proj(x)  # (B, H*W, hidden_dim)
        k = self.k_proj(x)  # (B, H*W, hidden_dim)
        v = self.v_proj(x)  # (B, H*W, hidden_dim)

        # Reshape for multi-head attention: (B, H*W, hidden_dim) -> (B, num_heads, H*W, head_dim)
        seq_len = H * W
        q = q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim**-0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, num_heads, H*W, H*W)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, self.hidden_dim)

        attn_output = self.out_proj(attn_output)

        x = residual + attn_output
        residual = x
        x = self.norm2(x)
        x = residual + x
        x = x.transpose(1, 2).view(B, self.hidden_dim, H, W)

        out = self.final_proj(x)

        return (out,)


class TestCFGPipeline(CFGParallelMixin):
    """Test pipeline using CFGParallelMixin."""

    def __init__(self, in_channels: int = 4, hidden_dim: int = 128, seed: int = 42):
        # Set seed BEFORE creating transformer to ensure consistent layer initialization
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.transformer = SimpleTransformer(in_channels, hidden_dim)

        # Re-initialize all parameters with fixed seed for full reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        for param in self.transformer.parameters():
            torch.nn.init.normal_(param, mean=0.0, std=0.02)


def _test_cfg_parallel_worker(
    local_rank: int,
    world_size: int,
    cfg_parallel_size: int,
    dtype: torch.dtype,
    test_config: dict,
    result_queue: torch.multiprocessing.Queue,
):
    """Worker function for CFG parallel test."""
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29502",
        }
    )

    init_distributed_environment()
    initialize_model_parallel(cfg_parallel_size=cfg_parallel_size)

    cfg_rank = get_classifier_free_guidance_rank()
    cfg_world_size = get_classifier_free_guidance_world_size()

    assert cfg_world_size == cfg_parallel_size

    # Create pipeline with same seed to ensure identical model weights across all ranks
    # Note: model_seed is set inside TestCFGPipeline.__init__
    pipeline = TestCFGPipeline(
        in_channels=test_config["channels"],
        hidden_dim=test_config["hidden_dim"],
        seed=test_config["model_seed"],
    )
    pipeline.transformer = pipeline.transformer.to(device=device, dtype=dtype)
    pipeline.transformer.eval()  # Set to eval mode for deterministic behavior

    # Create fixed inputs with explicit seed setting for reproducibility
    # Set both CPU and CUDA seeds to ensure identical inputs across all ranks
    torch.manual_seed(test_config["input_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(test_config["input_seed"])

    batch_size = test_config["batch_size"]
    channels = test_config["channels"]
    height = test_config["height"]
    width = test_config["width"]

    # Positive input
    positive_input = torch.randn(batch_size, channels, height, width, dtype=dtype, device=device)

    # Negative input with different seed
    torch.manual_seed(test_config["input_seed"] + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(test_config["input_seed"] + 1)
    negative_input = torch.randn(batch_size, channels, height, width, dtype=dtype, device=device)

    # Prepare kwargs for predict_noise_maybe_with_cfg
    positive_kwargs = {"x": positive_input}
    negative_kwargs = {"x": negative_input}

    with torch.no_grad():
        # Call predict_noise_maybe_with_cfg
        noise_pred = pipeline.predict_noise_maybe_with_cfg(
            do_true_cfg=True,
            true_cfg_scale=test_config["cfg_scale"],
            positive_kwargs=positive_kwargs,
            negative_kwargs=negative_kwargs,
            cfg_normalize=test_config["cfg_normalize"],
        )

    # Only rank 0 has valid output in CFG parallel mode
    if cfg_rank == 0:
        assert noise_pred is not None
        result_queue.put(noise_pred.cpu())
    else:
        assert noise_pred is None

    destroy_distributed_env()


def _test_cfg_sequential_worker(
    local_rank: int,
    world_size: int,
    dtype: torch.dtype,
    test_config: dict,
    result_queue: torch.multiprocessing.Queue,
):
    """Worker function for sequential CFG test (baseline)."""
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29503",
        }
    )

    init_distributed_environment()
    initialize_model_parallel(cfg_parallel_size=1)  # No CFG parallel

    cfg_world_size = get_classifier_free_guidance_world_size()
    assert cfg_world_size == 1

    # Create pipeline with same seed to ensure identical model weights as CFG parallel
    # Note: model_seed is set inside TestCFGPipeline.__init__
    pipeline = TestCFGPipeline(
        in_channels=test_config["channels"],
        hidden_dim=test_config["hidden_dim"],
        seed=test_config["model_seed"],
    )
    pipeline.transformer = pipeline.transformer.to(device=device, dtype=dtype)
    pipeline.transformer.eval()

    # Create fixed inputs (same seed as CFG parallel to ensure identical inputs)
    # Set both CPU and CUDA seeds for full reproducibility
    torch.manual_seed(test_config["input_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(test_config["input_seed"])

    batch_size = test_config["batch_size"]
    channels = test_config["channels"]
    height = test_config["height"]
    width = test_config["width"]

    # Positive input
    positive_input = torch.randn(batch_size, channels, height, width, dtype=dtype, device=device)

    # Negative input with different seed
    torch.manual_seed(test_config["input_seed"] + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(test_config["input_seed"] + 1)
    negative_input = torch.randn(batch_size, channels, height, width, dtype=dtype, device=device)

    positive_kwargs = {"x": positive_input}
    negative_kwargs = {"x": negative_input}

    with torch.no_grad():
        noise_pred = pipeline.predict_noise_maybe_with_cfg(
            do_true_cfg=True,
            true_cfg_scale=test_config["cfg_scale"],
            positive_kwargs=positive_kwargs,
            negative_kwargs=negative_kwargs,
            cfg_normalize=test_config["cfg_normalize"],
        )

    # Sequential CFG always returns output
    assert noise_pred is not None
    result_queue.put(noise_pred.cpu())

    destroy_distributed_env()


@pytest.mark.parametrize("cfg_parallel_size", [2])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("cfg_normalize", [False, True])
def test_predict_noise_maybe_with_cfg(cfg_parallel_size: int, dtype: torch.dtype, batch_size: int, cfg_normalize: bool):
    """
    Test that predict_noise_maybe_with_cfg produces identical results
    with and without CFG parallel.

    Args:
        cfg_parallel_size: Number of GPUs for CFG parallel
        dtype: Data type for computation
        batch_size: Batch size for testing
        cfg_normalize: Whether to normalize CFG output
    """
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < cfg_parallel_size:
        pytest.skip(f"Test requires {cfg_parallel_size} GPUs but only {available_gpus} available")

    test_config = {
        "batch_size": batch_size,
        "channels": 4,
        "height": 16,
        "width": 16,
        "hidden_dim": 128,
        "cfg_scale": 7.5,
        "cfg_normalize": cfg_normalize,
        "model_seed": 42,  # Fixed seed for model initialization
        "input_seed": 123,  # Fixed seed for input generation
    }

    mp_context = torch.multiprocessing.get_context("spawn")

    manager = mp_context.Manager()
    baseline_queue = manager.Queue()
    cfg_parallel_queue = manager.Queue()

    # Run baseline (sequential CFG) on single GPU
    torch.multiprocessing.spawn(
        _test_cfg_sequential_worker,
        args=(1, dtype, test_config, baseline_queue),
        nprocs=1,
    )

    # Run CFG parallel on multiple GPUs
    torch.multiprocessing.spawn(
        _test_cfg_parallel_worker,
        args=(cfg_parallel_size, cfg_parallel_size, dtype, test_config, cfg_parallel_queue),
        nprocs=cfg_parallel_size,
    )

    # Get results from queues
    baseline_output = baseline_queue.get()
    cfg_parallel_output = cfg_parallel_queue.get()

    # Verify shapes match
    assert baseline_output.shape == cfg_parallel_output.shape, (
        f"Shape mismatch: baseline {baseline_output.shape} vs CFG parallel {cfg_parallel_output.shape}"
    )

    # Verify numerical equivalence with appropriate tolerances
    if dtype == torch.float32:
        rtol, atol = 1e-5, 1e-5
    elif dtype == torch.bfloat16:
        rtol, atol = 1e-2, 1e-2
    else:
        rtol, atol = 1e-3, 1e-3

    torch.testing.assert_close(
        cfg_parallel_output,
        baseline_output,
        rtol=rtol,
        atol=atol,
        msg=(
            f"CFG parallel output differs from sequential CFG\n"
            f"  dtype={dtype}, batch_size={batch_size}, cfg_normalize={cfg_normalize}\n"
            f"  Max diff: {(cfg_parallel_output - baseline_output).abs().max().item():.6e}"
        ),
    )

    print(
        f"✓ Test passed: cfg_size={cfg_parallel_size}, dtype={dtype}, "
        f"batch_size={batch_size}, cfg_normalize={cfg_normalize}"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_predict_noise_without_cfg(dtype: torch.dtype):
    """
    Test predict_noise_maybe_with_cfg when do_true_cfg=False.

    When CFG is disabled, only the positive branch should be computed.
    This test runs on a single GPU without distributed environment.
    """
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < 1:
        pytest.skip("Test requires at least 1 GPU")

    device = torch.device(f"{current_omni_platform.device_type}:0")
    current_omni_platform.set_device(device)

    # Create pipeline without distributed environment
    pipeline = TestCFGPipeline(in_channels=4, hidden_dim=128, seed=42)
    pipeline.transformer = pipeline.transformer.to(device=device, dtype=dtype)
    pipeline.transformer.eval()

    # Set seed for input generation
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
    positive_input = torch.randn(1, 4, 16, 16, dtype=dtype, device=device)

    with torch.no_grad():
        noise_pred = pipeline.predict_noise_maybe_with_cfg(
            do_true_cfg=False,  # No CFG
            true_cfg_scale=7.5,
            positive_kwargs={"x": positive_input},
            negative_kwargs=None,
            cfg_normalize=False,
        )

    # Should always return output when do_true_cfg=False
    assert noise_pred is not None
    assert noise_pred.shape == (1, 4, 16, 16)

    print(f"✓ Test passed: predict_noise without CFG (dtype={dtype})")
