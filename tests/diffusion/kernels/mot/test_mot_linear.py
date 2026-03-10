# ruff: noqa: N803, E741
"""Layer-level correctness & performance test for MoT parallel linear layers.

Compares two equivalent computation paths:
  - Reference: 2x standard vLLM parallel linear layers + PyTorch index
    scatter/gather (text_linear(x[text_idx]) + vae_linear(x[vae_idx]))
  - Target:    1x MoT fused parallel linear layer
    (mot_linear(x, text_indices, vae_indices))

The reference path uses cuBLAS GEMM (always auto-tuned by cuBLAS).
The MoT path uses a fused Triton kernel whose tile config is loaded
from a JSON file matched by ``device + dtype``.  If no tuned
config is found for the current GPU, the kernel falls back to a conservative
default and a warning is printed — the correctness test still passes
but the performance comparison is NOT representative.

Usage::
    pytest tests/diffusion/kernels/mot/test_mot_linear.py -v -s
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)

from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from vllm_omni.diffusion.layers.mot.mot_qkv_parallel_linear import (
    MoTQKVParallelLinear,
)
from vllm_omni.diffusion.layers.mot.mot_row_parallel_linear import (
    MoTRowParallelLinear,
)
from vllm_omni.diffusion.layers.mot.ops.mot_gemm import (
    get_best_mot_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.gpu]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# BAGEL-7B-MoT architecture parameters
_BAGEL_HEAD_SIZE = 128
_BAGEL_TOTAL_NUM_HEADS = 28
_BAGEL_TOTAL_NUM_KV_HEADS = 4
_VAE_CHUNK_SIZE = 1024  # the token number of one image
_IMAGE_NUM = [1, 2, 4, 8]


@pytest.fixture(scope="module", autouse=True)
def _init_single_rank_tp_env():
    """Initialize single-rank distributed/TP env for vLLM linear params."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")

    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0)

    if not model_parallel_is_initialized():
        initialize_model_parallel(
            data_parallel_size=1,
            cfg_parallel_size=1,
            sequence_parallel_size=1,
            ulysses_degree=1,
            ring_degree=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )

    yield
    destroy_distributed_env()


# ---------------------------------------------------------------------------
# DType configuration — extensible for future quantized types
# ---------------------------------------------------------------------------


@dataclass
class DTypeConfig:
    """Parsed dtype configuration for a test case."""

    torch_dtype: torch.dtype
    use_fp8_w8a8: bool = False
    use_int8_w8a16: bool = False
    use_int4_w4a16: bool = False


def _parse_dtype(dtype_str: str) -> DTypeConfig:
    """Parse a dtype string into quantization flags and torch dtype.

    Supported now:
        "w16a16_bf16"  — BF16 weights & activations (no quantization)
        "w16a16_fp16"  — FP16 weights & activations (no quantization)
    Reserved for future:
        "fp8_w8a8"     — FP8 W8A8 quantization
        "int8_w8a16"   — INT8 weight-only quantization
        "int4_w4a16"   — INT4 weight-only quantization
    """
    supported: dict[str, DTypeConfig] = {
        "w16a16_bf16": DTypeConfig(torch_dtype=torch.bfloat16),
        "w16a16_fp16": DTypeConfig(torch_dtype=torch.float16),
    }
    if dtype_str in supported:
        return supported[dtype_str]
    pytest.skip(f"Quantized dtype '{dtype_str}' not yet implemented in layer test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _report_mot_config(K: int, N: int, M: int):
    """Print which Triton tile config the MoT kernel will use."""
    loaded_m_key, config = get_best_mot_config(M, N, K, None)
    if loaded_m_key == -1:
        print(
            "  [config] WARNING: No tuned config found — "
            "using conservative default. "
            "Performance numbers are NOT representative. "
            "Run mot_linear_benchmarks.py --tune to generate configs."
        )
    else:
        print(f"  [config] Tuned config loaded (actual M={M}, loaded M={loaded_m_key}) config = {config})")


def _make_indices(image_num: int, vae_chunk_size: int, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Simulate exact Bagel-MoT distributions for image generation:
    Pattern per image like: [1 Text] + [4096 VAE] + [1 Text]
    Returns text_indices, vae_indices, and the exact total M.
    """
    text_idx_list = []
    vae_idx_list = []

    current_idx = 0
    for _ in range(image_num):
        text_idx_list.append(current_idx)
        current_idx += 1

        vae_idx_list.extend(range(current_idx, current_idx + vae_chunk_size))
        current_idx += vae_chunk_size

        text_idx_list.append(current_idx)
        current_idx += 1

    text_indices = torch.tensor(text_idx_list, dtype=torch.long, device=device)
    vae_indices = torch.tensor(vae_idx_list, dtype=torch.long, device=device)

    exact_M = current_idx  # exact_M = image_num * (vae_chunk_size + 2)

    return text_indices, vae_indices, exact_M


def _benchmark(fn, warmup: int = 20, iters: int = 100) -> float:
    """Return mean latency in milliseconds."""
    cache_flusher = torch.empty(int(256 * 1024 * 1024 / 4), dtype=torch.int32, device="cuda")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    t_flush_start = time.perf_counter()
    for _ in range(iters):
        cache_flusher.zero_()
    torch.cuda.synchronize()
    flush_time_total = time.perf_counter() - t_flush_start

    #  3. Measure the total time of "flush cache + operator execution"
    torch.cuda.synchronize()
    t_total_start = time.perf_counter()
    for _ in range(iters):
        cache_flusher.zero_()
        fn()
    torch.cuda.synchronize()
    total_time = time.perf_counter() - t_total_start

    # 4. Asynchronous subtraction separation
    # The E2E time of the pure operator = total time - flush time
    pure_fn_time_total = total_time - flush_time_total
    avg_ms = (pure_fn_time_total / iters) * 1000.0

    del cache_flusher
    return max(avg_ms, 0.001)


def _sync_weights(ref_text, ref_vae, mot_layer):
    """Assign the same random weights to reference layers and MoT layer."""
    with torch.no_grad():
        W_text = torch.randn_like(ref_text.weight) * 0.02
        W_vae = torch.randn_like(ref_vae.weight) * 0.02
        ref_text.weight.copy_(W_text)
        ref_vae.weight.copy_(W_vae)
        mot_layer.weight.copy_(W_text)
        mot_layer.gen_exp.weight.copy_(W_vae)
        if ref_text.bias is not None and mot_layer.bias is not None:
            b_text = torch.randn_like(ref_text.bias) * 0.02
            ref_text.bias.copy_(b_text)
            mot_layer.bias.copy_(b_text)
        if ref_vae.bias is not None and mot_layer.gen_exp.bias is not None:
            b_vae = torch.randn_like(ref_vae.bias) * 0.02
            ref_vae.bias.copy_(b_vae)
            mot_layer.gen_exp.bias.copy_(b_vae)


def _reference_forward(x, text_indices, vae_indices, text_linear, vae_linear):
    """Reference path: index-gather → 2x standard linear → index-scatter."""
    M = x.size(0)
    out_text = text_linear(x[text_indices])
    out_vae = vae_linear(x[vae_indices])
    if isinstance(out_text, tuple):
        out_text = out_text[0]
    if isinstance(out_vae, tuple):
        out_vae = out_vae[0]
    N = out_text.size(-1)
    output = torch.empty(M, N, dtype=x.dtype, device=x.device)
    output[text_indices] = out_text
    output[vae_indices] = out_vae
    return output


def _check_and_report(ref: torch.Tensor, mot: torch.Tensor, tag: str):
    """Compare outputs, print metrics, and assert correctness.

    Both ``ref`` and ``mot`` are in the original compute dtype (e.g. bf16).
    We upcast to fp32 solely for computing error metrics with higher
    arithmetic precision — the actual layer outputs remain bf16.
    """
    # Upcast for numerically stable error computation only
    ref_hp = ref.float()
    mot_hp = mot.float()

    abs_err = (ref_hp - mot_hp).abs()
    max_abs = abs_err.max().item()

    # Mixed metric: relative error where |ref| >= 1, absolute error otherwise
    denom = ref_hp.abs().clamp(min=1.0)
    max_rel = (abs_err / denom).max().item()

    cos_sim = (
        torch.nn.functional.cosine_similarity(
            ref_hp,
            mot_hp,
            dim=-1,
        )
        .min()
        .item()
    )

    print(f"\n  [{tag}]  max_abs={max_abs:.4e}  max_rel={max_rel:.4e}  min_cos_sim={cos_sim:.6f}")

    # Cosine similarity is the primary correctness gate: robust to scale
    # and accumulation-order differences between cuBLAS and Triton.
    # For bf16 GEMM with K up to ~20k, cos_sim > 0.99 is easily achieved.
    assert cos_sim > 0.98, f"Cosine similarity too low: {cos_sim:.6f}"
    # Supplementary per-element check (generous to avoid flaky failures
    # on extreme K dimensions like 18944)
    assert max_rel < 0.1, f"Max relative error too large: {max_rel:.4e}"


def _run_timing(
    ref_fn,
    mot_fn,
    tag: str,
    warmup: int = 20,
    iters: int = 100,
):
    """Benchmark both paths and print timing comparison."""
    ref_ms = _benchmark(ref_fn, warmup=warmup, iters=iters)
    mot_ms = _benchmark(mot_fn, warmup=warmup, iters=iters)
    speedup = ref_ms / mot_ms if mot_ms > 0 else float("inf")
    print(f"  [{tag}]  Ref(2x linear): {ref_ms:.3f} ms  |  MoT(fused): {mot_ms:.3f} ms  |  Speedup: {speedup:.2f}x")


# =========================================================================
#  Test: qkv proj
# =========================================================================


@pytest.mark.parametrize(
    "image_num, K, N, dtype",
    [(num, 3584, 4608, "w16a16_bf16") for num in _IMAGE_NUM],
    ids=lambda val: "",
)
def test_mot_qkv_parallel(image_num: int, K: int, N: int, dtype: str):
    dcfg = _parse_dtype(dtype)
    torch.manual_seed(42)

    with set_current_vllm_config(VllmConfig()):
        text_linear = QKVParallelLinear(
            hidden_size=K,
            head_size=_BAGEL_HEAD_SIZE,
            total_num_heads=_BAGEL_TOTAL_NUM_HEADS,
            total_num_kv_heads=_BAGEL_TOTAL_NUM_KV_HEADS,
            bias=True,
            params_dtype=dcfg.torch_dtype,
            disable_tp=True,
        ).cuda()
        vae_linear = QKVParallelLinear(
            hidden_size=K,
            head_size=_BAGEL_HEAD_SIZE,
            total_num_heads=_BAGEL_TOTAL_NUM_HEADS,
            total_num_kv_heads=_BAGEL_TOTAL_NUM_KV_HEADS,
            bias=True,
            params_dtype=dcfg.torch_dtype,
            disable_tp=True,
        ).cuda()
        mot_linear = MoTQKVParallelLinear(
            hidden_size=K,
            head_size=_BAGEL_HEAD_SIZE,
            total_num_heads=_BAGEL_TOTAL_NUM_HEADS,
            total_num_kv_heads=_BAGEL_TOTAL_NUM_KV_HEADS,
            bias=True,
            vae_bias=True,
            params_dtype=dcfg.torch_dtype,
            disable_tp=True,
        ).cuda()

        assert text_linear.output_size_per_partition == N, (
            f"Expected output_size_per_partition={N}, "
            f"got {text_linear.output_size_per_partition}. "
            f"Check head parameters."
        )

        _sync_weights(text_linear, vae_linear, mot_linear)

        text_idx, vae_idx, M = _make_indices(image_num, _VAE_CHUNK_SIZE)
        x = torch.randn(M, K, dtype=dcfg.torch_dtype, device="cuda")

        tag = f"QKVParallel M={M} K={K} N={N}"
        _report_mot_config(K, N, M)

        with torch.no_grad():
            ref = _reference_forward(
                x,
                text_idx,
                vae_idx,
                text_linear,
                vae_linear,
            )
            mot_out, _ = mot_linear(x, text_idx, vae_idx)

        _check_and_report(ref, mot_out, tag)

        with torch.no_grad():
            _run_timing(
                lambda: _reference_forward(
                    x,
                    text_idx,
                    vae_idx,
                    text_linear,
                    vae_linear,
                ),
                lambda: mot_linear(x, text_idx, vae_idx),
                tag,
            )


# =========================================================================
#  Test: o proj
# =========================================================================


@pytest.mark.parametrize(
    "image_num, K, N, dtype",
    [(num, 3584, 3584, "w16a16_bf16") for num in _IMAGE_NUM],
    ids=lambda val: "",
)
def test_mot_o_proj(
    image_num: int,
    K: int,
    N: int,
    dtype: str,
):
    dcfg = _parse_dtype(dtype)
    torch.manual_seed(42)

    with set_current_vllm_config(VllmConfig()):
        text_linear = RowParallelLinear(
            K,
            N,
            bias=False,
            input_is_parallel=True,
            params_dtype=dcfg.torch_dtype,
            disable_tp=True,
        ).cuda()
        vae_linear = RowParallelLinear(
            K,
            N,
            bias=False,
            input_is_parallel=True,
            params_dtype=dcfg.torch_dtype,
            disable_tp=True,
        ).cuda()
        mot_linear = MoTRowParallelLinear(
            K,
            N,
            bias=False,
            input_is_parallel=True,
            params_dtype=dcfg.torch_dtype,
            disable_tp=True,
        ).cuda()

        _sync_weights(text_linear, vae_linear, mot_linear)

        text_idx, vae_idx, M = _make_indices(image_num, _VAE_CHUNK_SIZE)
        x = torch.randn(M, K, dtype=dcfg.torch_dtype, device="cuda")

        tag = f"O Proj M={M} K={K} N={N}"
        _report_mot_config(K, N, M)

        # Correctness (also warms up Triton JIT compilation)
        with torch.no_grad():
            ref = _reference_forward(
                x,
                text_idx,
                vae_idx,
                text_linear,
                vae_linear,
            )
            mot_out, _ = mot_linear(x, text_idx, vae_idx)

        _check_and_report(ref, mot_out, tag)

        # Performance
        with torch.no_grad():
            _run_timing(
                lambda: _reference_forward(
                    x,
                    text_idx,
                    vae_idx,
                    text_linear,
                    vae_linear,
                ),
                lambda: mot_linear(x, text_idx, vae_idx),
                tag,
            )
