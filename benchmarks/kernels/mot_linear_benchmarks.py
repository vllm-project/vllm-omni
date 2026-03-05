# ruff: noqa: N803

"""MoT (Mixture-of-Tokens) GEMM kernel benchmark and auto-tuning.

Generates optimal Triton kernel configurations for MoT GEMM operations
across different batch sizes, model shapes, TP configurations, and hardware.

Usage:
    # Auto-tune and save configs:
    python benchmarks/kernels/mot_linear_benchmarks.py \
        --model ByteDance-Seed/BAGEL-7B-MoT \
        --tp-size 1 --dtype w16a16 --tune \
        --save-dir vllm_omni/diffusion/layers/mot/configs/

    # Auto-tune with local model path (offline clusters):
    python benchmarks/kernels/mot_linear_benchmarks.py \
        --model /data/models/BAGEL-7B-MoT \
        --tp-size 2 --tune

    # Benchmark only (measure with default configs, no search):
    python benchmarks/kernels/mot_linear_benchmarks.py \
        --model ByteDance-Seed/BAGEL-7B-MoT \
        --tp-size 1 --dtype w16a16
"""

import argparse
import gc
import json
import logging
import math
import os
import time
from datetime import datetime
from itertools import product
from typing import Any

import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_config
from vllm.triton_utils import triton
from vllm.utils.torch_utils import set_random_seed

# NOTE: you should use the same naming syetem for the kernel to load properly
from vllm_omni.diffusion.layers.mot.ops.mot_gemm import build_config_filename, get_device_name

# clear the triton cache from time to time, usaully no need to change
_CACHE_CLEAR_INTERVAL_ENV = "VLLM_MOT_TUNE_CACHE_CLEAR_INTERVAL"
TRITON_CACHE_CLEAR_INTERVAL = int(os.environ.get(_CACHE_CLEAR_INTERVAL_ENV, "50"))

# represent the token number of each generated image
_VAE_CHUNK_SIZE_ENV = "VAE_CHUNK_SIZE"
VAE_CHUNK_SIZE = int(os.environ.get(_VAE_CHUNK_SIZE_ENV, "1024"))

logger = logging.getLogger(__name__)

# =====================================================================
#  Utility Functions
# =====================================================================


def clear_triton_cache():
    """Clear Triton JIT compilation cache and Python/CUDA memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        if hasattr(triton, "runtime") and hasattr(triton.runtime, "cache") and hasattr(triton.runtime.cache, "clear"):
            triton.runtime.cache.clear()
    except Exception:
        pass
    gc.collect()


# TODO: check rocm/npus support
# based on https://docs.nvidia.com/cuda/cuda-runtime-api/
# structcudaDeviceProp.html#structcudaDeviceProp_16cede1829516e86917f0842a5f6498c8
def get_max_shared_memory() -> int:
    """Return the maximum shared memory per block in bytes."""
    props = torch.cuda.get_device_properties(0)
    if hasattr(props, "shared_memory_per_block_option"):
        return props.shared_memory_per_block_option
    return getattr(props, "shared_memory_per_block", 49152)


def get_max_regs() -> int:
    """Return the maximum registers per block in bytes."""
    props = torch.cuda.get_device_properties(0)
    if hasattr(props, "regs_per_block"):
        return props.regs_per_block
    return 65536


def get_sm_count() -> int:
    """Get the number of physical SMs on the target GPU (A100 = 108, H100 = 132)."""
    return torch.cuda.get_device_properties(0).multi_processor_count


def get_ab_element_bytes(dtype_str: str) -> tuple[int, int]:
    """Return ``(activation_bytes, weight_bytes)`` for a dtype config."""
    if dtype_str == "w16a16":
        return 2, 2
    elif dtype_str == "fp8_w8a8":
        return 1, 1
    elif dtype_str == "int8_w8a16":
        return 2, 1
    return 2, 2


def build_regular_indices(
    image_num: int,
    vae_chunk_size: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build deterministic MoT indices with per-image [Text][VAE...][Text]."""
    if image_num <= 0:
        raise ValueError(f"image_num must be > 0, got {image_num}")
    if vae_chunk_size <= 0:
        raise ValueError(f"{_VAE_CHUNK_SIZE_ENV} must be > 0, got {vae_chunk_size}")

    text_idx_list: list[int] = []
    vae_idx_list: list[int] = []

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
    exact_M = current_idx
    return text_indices, vae_indices, exact_M


def get_exact_m(image_num: int, vae_chunk_size: int) -> int:
    if image_num <= 0:
        raise ValueError(f"image_num must be > 0, got {image_num}")
    if vae_chunk_size <= 0:
        raise ValueError(f"{_VAE_CHUNK_SIZE_ENV} must be > 0, got {vae_chunk_size}")
    return image_num * (vae_chunk_size + 2)


# =====================================================================
#  Model Shape Extraction
# =====================================================================


class MoTShape:
    """One unique (K, N) GEMM shape in a MoT model layer."""

    def __init__(self, K: int, N: int, comment: str):
        self.K = K
        self.N = N
        self.comment = comment

    def config_key(self) -> str:
        return f"{self.K}_{self.N}"

    def __repr__(self) -> str:
        return f"MoTShape(K={self.K}, N={self.N}, comment='{self.comment}')"


def get_mot_shapes(
    model: str,
    tp_size: int,
    trust_remote_code: bool = False,
) -> tuple[list[MoTShape], str]:
    """Extract MoT GEMM shapes from a HuggingFace model config.

    Supports both remote HuggingFace model IDs and local checkpoint paths.

    Returns
    -------
    shapes : list[MoTShape]
        De-duplicated GEMM shapes (K, N) with TP applied.
    model_name : str
        Cleaned model name for the config filename.
    """
    config = get_config(model=model, trust_remote_code=trust_remote_code)
    model_name = model.rstrip("/").split("/")[-1]

    text_config = getattr(config, "text_config", config)

    hidden_size: int = text_config.hidden_size
    num_attention_heads: int = text_config.num_attention_heads
    num_kv_heads: int = getattr(text_config, "num_key_value_heads", num_attention_heads)
    head_dim: int = getattr(text_config, "head_dim", hidden_size // num_attention_heads)
    intermediate_size: int = text_config.intermediate_size

    # ---- Compute per-TP shapes ----

    # QKV_PROJ  (QKVParallelLinear, output partitioned by TP)
    q_out = num_attention_heads * head_dim
    kv_out = 2 * num_kv_heads * head_dim
    qkv_total = q_out + kv_out
    assert qkv_total % tp_size == 0, f"QKV output {qkv_total} not divisible by tp {tp_size}"
    qkv_N = qkv_total // tp_size

    # O_PROJ  (RowParallelLinear, input partitioned by TP)
    assert q_out % tp_size == 0, f"Q output {q_out} not divisible by tp {tp_size}"
    o_K = q_out // tp_size
    o_N = hidden_size

    # FFN gate+up  (MergedColumnParallelLinear, output partitioned by TP)
    gate_up_total = 2 * intermediate_size
    assert gate_up_total % tp_size == 0, f"Gate-up output {gate_up_total} not divisible by tp {tp_size}"
    gate_up_N = gate_up_total // tp_size

    # FFN down  (RowParallelLinear, input partitioned by TP)
    assert intermediate_size % tp_size == 0, f"Intermediate size {intermediate_size} not divisible by tp {tp_size}"
    down_K = intermediate_size // tp_size
    down_N = hidden_size

    shapes = [
        MoTShape(K=o_K, N=o_N, comment="O_PROJ"),
        MoTShape(K=hidden_size, N=qkv_N, comment="QKV_PROJ"),
        MoTShape(K=hidden_size, N=gate_up_N, comment="FFN_GATE_UP_PROJ"),
        MoTShape(K=down_K, N=down_N, comment="FFN_DOWN_PROJ"),
    ]

    seen: dict[str, MoTShape] = {}
    unique: list[MoTShape] = []
    for s in shapes:
        key = s.config_key()
        if key not in seen:
            seen[key] = s
            unique.append(s)
        else:
            seen[key].comment += f" / {s.comment}"

    return unique, model_name


# =====================================================================
#  Search Space Generation & Pruning
# =====================================================================


def estimate_sram_bytes(config: dict[str, int], dtype_str: str) -> int:
    """Estimate SRAM (shared memory) usage for a Triton tile config.

    Formula:
        (BLOCK_M * BLOCK_K * a_bytes + BLOCK_N * BLOCK_K * b_bytes)
        * num_stages
    """
    bm = config["BLOCK_SIZE_M"]
    bn = config["BLOCK_SIZE_N"]
    bk = config["BLOCK_SIZE_K"]
    stages = config["num_stages"]
    a_bytes, b_bytes = get_ab_element_bytes(dtype_str)
    return (bm * bk * a_bytes + bn * bk * b_bytes) * stages


# TODO:
# granule_size= 256 for nvdia gpus,
# warp_size=32 for nvdia gpus,
# not sure if it is true for rocm/other npus
def estimate_register_pressure(
    config: dict[str, int],
    dtype_str: str,
    max_regs_per_block: int = 65536,
    max_regs_per_thread: int = 255,
    granule_size: int = 256,
    warp_size: int = 32,
) -> bool:
    """
    Evaluate register pressure for MoT GEMM based on kernel structure and datatypes.

    Args:
        config: Triton tile configuration.
        dtype_str: for now only support:"w16a16", "fp8_w8a8","int8_w8a16"
        max_regs_per_block: Hardware limit (usually 65536).
        max_regs_per_thread: PTX limit  (usually 255).
        granule_size: register allocation size for one warp.
        warp_size: number of threads per warp.
    Returns:
        True if the config is safe to compile and run efficiently, False if it should be pruned.
    """
    bm = config["BLOCK_SIZE_M"]
    bn = config["BLOCK_SIZE_N"]
    bk = config["BLOCK_SIZE_K"]
    warps = config["num_warps"]
    num_threads = warps * warp_size

    a_bytes, b_bytes = get_ab_element_bytes(dtype_str)

    # Physical register standard: 1 register = 32-bit (4 bytes)

    # [Accumulator C]
    # Triton uses fp32/int32 by default as the accumulator for fp16/int8 to prevent overflow
    regs_c = (bm * bn) / num_threads * 1.0

    # [MMA slices A and B]
    # Data is loaded into registers from SRAM to participate in Tensor Core operations
    regs_a = ((bm * bk) / num_threads) * (a_bytes / 4.0)
    regs_b = ((bk * bn) / num_threads) * (b_bytes / 4.0)

    # [MoT specific routing overhead]
    # real_row_idxs is tl.int64 (8 bytes), each element needs 2 32-bit registers
    regs_routing = (bm / num_threads) * 2.0

    # [Quantization specific Epilogue overhead]
    # W8A8 needs to load scale_a and scale_b after the loop for de-quantization
    regs_epilogue = 0.0
    if dtype_str == "fp8_w8a8":
        # fp8*token-wise quant scenario: scale_a length is bm, scale_b length is bn
        regs_epilogue = ((bm + bn) / num_threads) * 1.0
    elif dtype_str == "int8_w8a16":
        # Weight-Only*token-wise quant scenario:  usually only scale_b is needed
        regs_epilogue = (bn / num_threads) * 1.0

    # [Control flow and base pointer constant overhead]
    # Includes: loop counter(k), pointer addressing,
    # Mask predicate calculation, TMA state machine, etc.
    constant_overhead = 35
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Summary and red line intercept
    # ---------------------------------------------------------
    estimated_regs_per_thread = math.ceil(regs_c + regs_a + regs_b + regs_routing + regs_epilogue + constant_overhead)
    # Hardware red line 1: single thread physical limit
    # (PTX ISA specifies a maximum of 255 registers per thread,
    # leaving 10% as a compiler buffer)
    if estimated_regs_per_thread > max_regs_per_thread * 0.9:
        return False

    # Hardware red line 2: single block total physical limit
    # (each warp allocated registers are rounded up to the nearest multiple of 256)
    regs_per_warp_raw = estimated_regs_per_thread * warp_size
    regs_per_warp_actual = math.ceil(regs_per_warp_raw / granule_size) * granule_size

    # Calculate the actual physical register consumption for the current block
    estimated_regs_per_block = regs_per_warp_actual * warps
    if estimated_regs_per_block > max_regs_per_block:
        return False

    return True


def get_mot_search_space(
    M: int,
    K: int,
    N: int,
    dtype_str: str,
    max_sram: int,
    max_regs: int,
    num_sms: int,
) -> list[dict[str, int]]:
    """Generate a pruned search space of Triton tile configs for MoT GEMM."""

    param_ranges = {
        "BLOCK_SIZE_M": [32, 64, 128, 256],
        "BLOCK_SIZE_N": [32, 64, 128, 256],
        "BLOCK_SIZE_K": [32, 64, 128],
        "GROUP_SIZE_M": [4, 8, 16],
        "num_warps": [4, 8],
        "num_stages": [2, 3, 4, 5],
    }

    def next_power_of_2(n):
        return 1 if n == 0 else 2 ** (n - 1).bit_length()

    padded_M = next_power_of_2(M)
    padded_N = next_power_of_2(N)
    padded_K = next_power_of_2(K)

    keys, values = zip(*param_ranges.items())
    configs: list[dict[str, int]] = []

    for vals in product(*values):
        cfg = dict(zip(keys, vals))
        bm = cfg["BLOCK_SIZE_M"]
        bn = cfg["BLOCK_SIZE_N"]
        bk = cfg["BLOCK_SIZE_K"]

        # --- Dimension-based pruning ---
        if bm > max(32, padded_M):
            continue
        if bn > max(32, padded_N):
            continue
        if bk > max(32, padded_K):
            continue
        if bm * bn < 64:
            continue

        # --- Occupancy-based pruning ---
        grid_m = (M + bm - 1) // bm
        grid_n = (N + bn - 1) // bn
        total_blocks = grid_m * grid_n

        if total_blocks < num_sms // 4:
            continue

        # --- SRAM capacity check ---
        if estimate_sram_bytes(cfg, dtype_str) > max_sram * 0.9:
            continue

        # --- register spilling check ---
        if not estimate_register_pressure(cfg, dtype_str, max_regs):
            continue

        configs.append(cfg)

    return configs


# =====================================================================
#  Single-Config Benchmark
# =====================================================================


def benchmark_config(
    config: dict[str, int],
    image_num: int,
    K: int,
    N: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    num_iters: int = 10,
    cache_flusher: torch.Tensor | None = None,
) -> float:
    """Run a MoT GEMM with the given tile config and return avg latency (us)."""
    from vllm_omni.diffusion.layers.mot.ops.mot_gemm import invoke_mot_gemm

    text_indices, vae_indices, M = build_regular_indices(
        image_num=image_num,
        vae_chunk_size=VAE_CHUNK_SIZE,
        device="cuda",
    )

    # ---- Allocate tensors on the current CUDA device ----
    A_scale: torch.Tensor | None = None
    B_text_scale: torch.Tensor | None = None
    B_vae_scale: torch.Tensor | None = None

    if use_fp8_w8a8:
        fp8_dtype = current_platform.fp8_dtype()
        A = torch.randn(M, K, dtype=torch.float16, device="cuda").to(fp8_dtype)
        B_text = torch.randn(K, N, dtype=torch.float16, device="cuda").to(fp8_dtype)
        B_vae = torch.randn(K, N, dtype=torch.float16, device="cuda").to(fp8_dtype)
        A_scale = torch.ones(M, dtype=torch.float32, device="cuda")
        B_text_scale = torch.ones(1, dtype=torch.float32, device="cuda")
        B_vae_scale = torch.ones(1, dtype=torch.float32, device="cuda")
        C = torch.empty(M, N, dtype=dtype, device="cuda")
    elif use_int8_w8a16:
        A = torch.randn(M, K, dtype=dtype, device="cuda")
        B_text = torch.randint(-127, 127, (K, N), dtype=torch.int8, device="cuda")
        B_vae = torch.randint(-127, 127, (K, N), dtype=torch.int8, device="cuda")
        B_text_scale = torch.ones(N, dtype=torch.float32, device="cuda")
        B_vae_scale = torch.ones(N, dtype=torch.float32, device="cuda")
        C = torch.empty(M, N, dtype=dtype, device="cuda")
    else:
        A = torch.randn(M, K, dtype=dtype, device="cuda")
        B_text = torch.randn(K, N, dtype=dtype, device="cuda")
        B_vae = torch.randn(K, N, dtype=dtype, device="cuda")
        C = torch.empty(M, N, dtype=dtype, device="cuda")

    def run():
        invoke_mot_gemm(
            A=A,
            B_text=B_text,
            B_vae=B_vae,
            C=C,
            bias_text=None,
            bias_vae=None,
            text_indices=text_indices,
            vae_indices=vae_indices,
            A_scale=A_scale,
            B_text_scale=B_text_scale,
            B_vae_scale=B_vae_scale,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=False,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=False,
            A_per_channel_quant=use_fp8_w8a8,
            B_per_channel_quant=use_int8_w8a16,
            config=config,
        )

    # JIT warmup
    run()
    torch.cuda.synchronize()

    # Capture 1 invocations with CUDA Graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    torch.cuda.synchronize()

    # Warmup replays
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies: list[float] = []
    for _ in range(num_iters):
        if cache_flusher is not None:
            cache_flusher.zero_()
        torch.cuda.synchronize()

        start_event.record()
        graph.replay()
        end_event.record()
        end_event.synchronize()

        latencies.append(start_event.elapsed_time(end_event))

    latencies.sort()
    valid_latencies = latencies[1:-1] if len(latencies) > 2 else latencies

    avg_us = sum(valid_latencies) / len(valid_latencies) * 1000  # ms → us
    graph.reset()

    return avg_us


# =====================================================================
#  Ray Worker
# =====================================================================


@ray.remote(num_gpus=1)
class BenchmarkWorker:
    def __init__(self, seed: int) -> None:
        # Ray will automatically set CUDA_VISIBLE_DEVICES,
        # so the GPU seen by the worker is always the logical 0
        self.logical_device_id = 0
        torch.set_default_device(f"cuda:{self.logical_device_id}")

        set_random_seed(seed)
        self.seed = seed

    # ---- Benchmark (use default config, report latency) ----

    def benchmark(
        self,
        image_num: int,
        K: int,
        N: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
    ) -> tuple[dict[str, int], float]:
        set_random_seed(self.seed)
        from vllm_omni.diffusion.layers.mot.ops.mot_gemm import (
            get_best_mot_config,
        )

        M = get_exact_m(image_num, VAE_CHUNK_SIZE)
        loaded_m_key, config = get_best_mot_config(M, N, K)
        if loaded_m_key == -1:
            print(
                "  [config] WARNING: No tuned config found — "
                "using conservative default. "
                "Performance numbers are NOT representative. "
                "Run mot_linear_benchmarks.py --tune to generate configs."
            )
        else:
            print(f"  [config] Tuned config loaded (actual M={M}, loaded M={loaded_m_key}) config = {config})")
        kernel_time = benchmark_config(
            config,
            image_num,
            K,
            N,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a16,
            num_iters=10,
        )
        return config, kernel_time

    # ---- Tune (search over all configs, return best) ----

    def tune(
        self,
        image_num: int,
        K: int,
        N: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
        search_space: list[dict[str, int]],
    ) -> dict[str, int] | None:
        set_random_seed(self.seed)
        M = get_exact_m(image_num, VAE_CHUNK_SIZE)

        best_config: dict[str, int] | None = None
        best_time = float("inf")

        # Diagnosis counters
        total_configs = len(search_space)
        err_oom = 0
        err_triton_resources = 0
        err_other = 0

        with torch.cuda.device(self.logical_device_id):
            cache_flusher = torch.empty(int(256 * 1024 * 1024 / 4), dtype=torch.int32, device="cuda")

            for idx, config in enumerate(tqdm(search_space)):
                try:
                    kernel_time = benchmark_config(
                        config,
                        image_num,
                        K,
                        N,
                        dtype,
                        use_fp8_w8a8,
                        use_int8_w8a16,
                        num_iters=10,
                        cache_flusher=cache_flusher,
                    )
                except triton.runtime.autotuner.OutOfResources:
                    err_triton_resources += 1
                    continue
                except torch.cuda.OutOfMemoryError:
                    err_oom += 1
                    clear_triton_cache()
                    continue
                except Exception:
                    err_other += 1
                    logger.exception("Config %s failed unexpectedly", config)
                    clear_triton_cache()
                    continue

                if kernel_time < best_time:
                    best_time = kernel_time
                    best_config = config

                if TRITON_CACHE_CLEAR_INTERVAL > 0 and idx > 0 and idx % TRITON_CACHE_CLEAR_INTERVAL == 0:
                    clear_triton_cache()

        del cache_flusher
        clear_triton_cache()

        if best_config is None:
            diag_msg = (
                f"\n🚨 [CRITICAL] TUNING FAILED for M={M}, K={K}, N={N}\n"
                f"   Total configs tested: {total_configs}\n"
                f"   - Triton OutOfResources (SRAM/Regs): {err_triton_resources}\n"
                f"   - CUDA OOM: {err_oom}\n"
                f"   - Other Errors: {err_other}\n"
                f"   💡 DIAGNOSIS:\n"
                f"   1. If total configs is 0, your 'get_mot_search_space' pruning is too aggressive.\n"
                f"   2. If Triton/OOM errors == total configs, hardware limits (SRAM/Regs) in pruning are too loose.\n"
                f"   3. If Other Errors is high, check benchmark_config logic or Triton kernel runtime bugs."
            )
            print(diag_msg)
            return None

        now = datetime.now()
        print(f"[{now.ctime()}] Tuning done: M={M}, K={K}, N={N}, best_time={best_time:.2f} us")
        return best_config

    # ---- Device info helpers (called from driver) ----

    def get_device_name(self) -> str:
        return get_device_name()

    def get_max_shared_memory(self) -> int:
        return get_max_shared_memory()

    def get_sm_count(self) -> int:
        return get_sm_count()

    def get_max_regs(self) -> int:
        return get_max_regs()


# =====================================================================
#  Config I/O
# =====================================================================


def sort_config(config: dict[str, int]) -> dict[str, int]:
    return {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": config["GROUP_SIZE_M"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
    }


def save_configs(
    results: dict[str, dict[int, dict[str, int]]],
    shapes: list[MoTShape],
    model_name: str,
    tp_size: int,
    device_name: str,
    dtype_str: str,
    save_dir: str,
) -> str:
    """Merge tuned configs into ``device_name=...,dtype=....json``.

    Behavior:
      - Create a new file if it does not exist.
      - If it exists, merge by shape key (``K_N``) and M key.
      - Existing entries are preserved unless overwritten by current results.
    """
    shape_map = {s.config_key(): s for s in shapes}

    current_output: dict[str, Any] = {}
    for config_key, m_configs in results.items():
        shape = shape_map[config_key]
        # Make comments self-descriptive across mixed model/tp runs.
        comment = f"model={model_name}|tp={tp_size}|op={shape.comment}"
        entry: dict[str, Any] = {"_comment": comment}
        for m_val in sorted(m_configs.keys()):
            entry[str(m_val)] = sort_config(m_configs[m_val])
        current_output[config_key] = entry

    filename = f"device_name={device_name},dtype={dtype_str}.json"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    merged_output: dict[str, Any] = {}
    if os.path.isfile(filepath):
        try:
            with open(filepath) as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                merged_output = existing
            else:
                print(f"WARNING: Existing config is not a JSON object: {filepath}. Overwrite with newly tuned configs.")
        except Exception as exc:
            print(f"WARNING: Failed to read existing config {filepath}: {exc}. Overwrite with newly tuned configs.")

    # Merge on two levels: shape key -> M key
    for config_key, new_entry in current_output.items():
        old_entry = merged_output.get(config_key, {})
        if not isinstance(old_entry, dict):
            old_entry = {}
        merged_entry = dict(old_entry)
        old_comment = merged_entry.get("_comment")
        new_comment = new_entry.get("_comment")
        merged_entry.update(new_entry)
        if old_comment and new_comment and old_comment != new_comment:
            merged_entry["_comment"] = f"{old_comment} / {new_comment}"
        merged_output[config_key] = merged_entry

    print(f"Saving merged config to {filepath}")
    with open(filepath, "w") as f:
        json.dump(merged_output, f, indent=2)
        f.write("\n")

    return filepath


# =====================================================================
#  Main
# =====================================================================


def main(args: argparse.Namespace):
    print(args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not current_platform.is_cuda():
        raise NotImplementedError(
            "Non-CUDA environment detected!"
            "This benchmark script has not been fully tested on"
            "AMD GPUs and may produce errors or suboptimal results."
        )

    # ---- 1. Extract model shapes ----
    shapes, model_name = get_mot_shapes(args.model, args.tp_size, args.trust_remote_code)
    print(f"\nModel: {model_name}  |  TP: {args.tp_size}")
    print(f"Detected {len(shapes)} unique GEMM shape(s):")
    for s in shapes:
        print(f"  {s}")

    # ---- 2. Determine dtype ----
    dtype_str: str = args.dtype
    use_fp8_w8a8 = dtype_str == "fp8_w8a8"
    use_int8_w8a16 = dtype_str == "int8_w8a16"
    dtype = torch.bfloat16

    # ---- 3. Image counts ----
    image_nums: list[int] = args.batch_size if args.batch_size is not None else [1, 2, 4, 8, 16]

    # ---- 4. Initialize Ray workers ----
    ray.init()
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [BenchmarkWorker.remote(args.seed) for _ in range(num_gpus)]
    print(f"\nRay initialized with {num_gpus} GPU worker(s)")

    device_name = ray.get(workers[0].get_device_name.remote())
    max_sram = ray.get(workers[0].get_max_shared_memory.remote())
    max_regs = ray.get(workers[0].get_max_regs.remote())
    sm_count = ray.get(workers[0].get_sm_count.remote())

    print(
        f"Device: {device_name}  |  Max SRAM/Block: {max_sram} bytes\n"
        f" Max Regs/Block: {max_regs} 32-bit regs\n"
        f" SM on GPU: {sm_count} \n"
    )

    # ---- Helper: round-robin distribute tasks to workers ----
    def distribute(method: str, inputs: list[tuple[Any, ...]]) -> list[Any]:
        futures = []
        for i, input_args in enumerate(inputs):
            worker = workers[i % num_gpus]
            futures.append(getattr(worker, method).remote(*input_args))
        return ray.get(futures)

    # ---- 5. TUNE mode ----
    if args.tune:
        start = time.time()

        # 1) Checkpoint loading and resuming
        filename = build_config_filename(device_name, dtype_str)
        filepath = os.path.join(args.save_dir, filename)

        existing_history: dict[str, Any] = {}
        if os.path.isfile(filepath):
            try:
                with open(filepath) as f:
                    existing_history = json.load(f)
                print(f"Loaded existing checkpoint from {filepath}, resuming...")
            except Exception as e:
                print(f"WARNING: Failed to load existing checkpoint: {e}")

        # 2) Build task queue and execute checkpoint filtering
        pending_futures = {}
        task_counter = 0

        for shape in shapes:
            shape_key = shape.config_key()
            for image_num in image_nums:
                exact_M = get_exact_m(image_num, VAE_CHUNK_SIZE)

                if shape_key in existing_history and str(exact_M) in existing_history[shape_key]:
                    print(f"Skipping image_num={image_num} (M={exact_M}), Shape={shape_key} (Already tuned)")
                    continue

                # Only tune parameters that have not been tuned yet
                search_space = get_mot_search_space(
                    M=exact_M,
                    K=shape.K,
                    N=shape.N,
                    dtype_str=dtype_str,
                    max_sram=max_sram,
                    max_regs=max_regs,
                    num_sms=sm_count,
                )
                if len(search_space) == 0:
                    print(
                        f"WARNING: empty search space for "
                        f"{shape.config_key()} image_num={image_num} (M={exact_M}), "
                        "skipping"
                    )
                    continue

                # Round-robin assign to Worker
                worker = workers[task_counter % num_gpus]
                future = worker.tune.remote(
                    image_num, shape.K, shape.N, dtype, use_fp8_w8a8, use_int8_w8a16, search_space
                )

                # Bind future with its corresponding metadata
                pending_futures[future] = (shape, image_num, exact_M)
                task_counter += 1

        print(f"Starting tuning: {len(pending_futures)} new tasks pending...")

        # 3）Async streaming collect results and incremental checkpoint (Streaming Checkpoint)
        results: dict[str, dict[int, dict[str, int]]] = {}

        # ray.wait will return when any task is completed
        # file I/O is executed serially
        while pending_futures:
            done_refs, not_done_refs = ray.wait(list(pending_futures.keys()), num_returns=1)

            for ready_future in done_refs:
                shape, image_num, exact_M = pending_futures.pop(ready_future)
                config_key = shape.config_key()
                try:
                    best_config = ray.get(ready_future)

                    if best_config is None:
                        print(
                            f"⚠️ SKIPPING CHECKPOINT for image_num={image_num}, "
                            f"M={exact_M}, Shape={config_key} due to tuning failure. "
                            "Please review the worker diagnostics above."
                        )
                        continue

                    # Put the temporary results of this run into the result set
                    results.setdefault(config_key, {})[exact_M] = best_config
                    save_configs(
                        results={config_key: {exact_M: best_config}},
                        shapes=shapes,
                        model_name=model_name,
                        tp_size=args.tp_size,
                        device_name=device_name,
                        dtype_str=dtype_str,
                        save_dir=args.save_dir,
                    )
                    print(f"Checkpoint saved for image_num={image_num}, M={exact_M}, Shape={config_key}")

                except Exception as e:
                    print(
                        f"🚨 CRITICAL ERROR: Task failed for image_num={image_num}, "
                        f"M={exact_M}, Shape={config_key}. Error: {e}"
                    )

        elapsed = time.time() - start
        print(f"\nTuning completed in {elapsed:.1f}s")
        print(f"Complete Configs saved to: {filepath}")

    # ---- 6. BENCHMARK mode ----
    else:
        all_tasks = []
        task_keys = []

        for shape in shapes:
            for image_num in image_nums:
                exact_M = get_exact_m(image_num, VAE_CHUNK_SIZE)
                all_tasks.append(
                    (
                        image_num,
                        shape.K,
                        shape.N,
                        dtype,
                        use_fp8_w8a8,
                        use_int8_w8a16,
                    )
                )
                task_keys.append((shape.config_key(), image_num, exact_M))

        all_results = distribute("benchmark", all_tasks)

        current_key = None
        for (config_key, image_num, exact_M), (config, kernel_time) in zip(task_keys, all_results):
            if config_key != current_key:
                current_key = config_key
                shape = next(s for s in shapes if s.config_key() == config_key)
                print(f"\n{'=' * 60}")
                print(f"Shape: {shape}")
                print(f"{'=' * 60}")
            print(f"  image_num={image_num:>4d}  M={exact_M:>6d}  {kernel_time:>8.2f} us  config={config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MoT GEMM kernel benchmark and auto-tuning",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or local checkpoint path",
    )
    parser.add_argument(
        "--tp-size",
        "-tp",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="w16a16",
        choices=["w16a16", "fp8_w8a8", "int8_w8a16"],
        help="Weight/activation dtype (default: w16a16)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=None,
        help="Image counts to tune/benchmark, note M=batch_size*(VAE_CHUNK_SIZE+2) (default: 1 2 4 8 16)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable auto-tuning mode (search for best configs)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./",
        help="Directory to save tuned config JSON (default: ./)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading HuggingFace config",
    )

    args = parser.parse_args()
    main(args)
