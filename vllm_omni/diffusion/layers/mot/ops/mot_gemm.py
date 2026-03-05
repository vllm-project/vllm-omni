# ruff: noqa: N803, E741
import functools
import json
import logging
import os
import pathlib
import re
from typing import Any

import torch
from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)

# =====================================================================
#  MoT GEMM Config Loading (3-tier: env → built-in → default)
#
#  Usage pattern (mirrors vLLM fused_moe):
#    1. Layer invokes invoke_mot_gemm(...) without extra config context.
#    2. invoke_mot_gemm lazily calls
#         get_mot_configs(K, N, dtype_str)
#       which is @lru_cache'd and lazily loads the JSON on first hit.
#    3. If get_mot_configs returns None, fall back to
#         get_mot_default_config(M, N, K, ...)
# =====================================================================

_CONFIGS_DIR = pathlib.Path(__file__).resolve().parent.parent / "configs"
_ENV_CONFIG_FOLDER = "VLLM_TUNED_CONFIG_FOLDER"


def get_device_name() -> str:
    """Sanitized GPU device name, matching mot_linear_benchmarks.py output."""
    raw = torch.cuda.get_device_name(0)
    name = re.sub(r"[^a-zA-Z0-9]", "", raw.replace(" ", ""))
    for prefix in ("NVIDIA", "AMD"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    # --- Device Aliasing Patch ---
    alias_map = {
        "A800": "A100",
        "H800": "H100",
    }
    # map A800/H800 to A100/H100 for chinese market
    for key, target in alias_map.items():
        if key in name:
            name = name.replace(key, target)
            break
    return name


def build_config_filename(device_name: str, dtype_str: str) -> str:
    return f"device_name={device_name},dtype={dtype_str}.json"


def _try_load_json(filepath: str) -> dict | None:
    if os.path.isfile(filepath):
        with open(filepath) as f:
            return json.load(f)
    return None


@functools.lru_cache
def _load_mot_config_file(dtype_str: str) -> dict | None:
    """Load and cache the full MoT config JSON (one file per device/dtype).

    Search order:
      1. ``$VLLM_TUNED_CONFIG_FOLDER/device_name=...,dtype=....json``
      2. ``vllm_omni/.../mot/configs/device_name=...,dtype=....json``
      3. Return ``None`` (caller falls back to ``get_mot_default_config``).
    """
    device_name = get_device_name()
    filename = build_config_filename(device_name, dtype_str)

    config_file_paths: list[str] = []
    env_dir = os.environ.get(_ENV_CONFIG_FOLDER)
    if env_dir:
        config_file_paths.append(str(pathlib.Path(env_dir) / filename))
    config_file_paths.append(str(_CONFIGS_DIR / filename))

    for path in config_file_paths:
        data = _try_load_json(path)
        if data is not None:
            logger.info("MoT config loaded from %s", path)
            return data

    logger.warning(
        f"\n{'=' * 80}\n"
        f" ⚠️  [WARNING] No tuned MoT config found.\n"
        f" Searched paths: {', '.join(config_file_paths)}\n"
        f" Using conservative default configs which are NOT optimal.\n"
        f" Run `python benchmarks/kernels/mot_linear_benchmarks.py --tune` \n"
        f" to generate hardware-specific optimal configs.\n"
        f"{'=' * 80}\n"
    )
    return None


@functools.lru_cache
def get_mot_configs(
    K: int,
    N: int,
    dtype_str: str | None = None,
) -> dict[int, dict[str, int]] | None:
    """Return ``{M: tile_config}`` for a given (K, N) shape, or ``None``.

    The return value maps an irregular grid of batch sizes (M) to Triton
    tile configurations.  The caller should pick the entry whose M is
    closest to the actual batch size.

    Config file is selected by ``device_name + dtype``.
    """
    file_data = _load_mot_config_file(dtype_str or "w16a16")
    if file_data is None:
        return None

    shape_entry = file_data.get(f"{K}_{N}")
    if shape_entry is None:
        logger.warning(
            f"\n{'=' * 80}\n"
            f" ⚠️  [WARNING] MoT config file found, but NO tuned entry for shape K={K}, N={N}.\n"
            f" Using conservative default configs which are NOT optimal for this specific shape.\n"
            f" Run `python benchmarks/kernels/mot_linear_benchmarks.py --tune` \n"
            f" to generate hardware-specific optimal configs.\n"
            f"{'=' * 80}\n"
        )
        return None

    return {int(k): dict(v) for k, v in shape_entry.items() if k != "_comment"}


def get_mot_default_config(
    M: int,
    N: int,
    K: int,
    dtype: str | None = None,
    block_quant_shape: list[int] | None = None,
) -> dict[str, int]:
    """Conservative fallback config guaranteed to compile on all hardware.

    Trades peak performance for universal compatibility (T4 / V100 / A100 /
    H100, CUDA & ROCm).
    """
    # FP8 block-wise quantization requires strict alignment
    if dtype == "fp8_w8a8" and block_quant_shape is not None:
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_quant_shape[0],
            "BLOCK_SIZE_K": block_quant_shape[1],
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 2,
        }

    # Very small M (tail batches, final feature concat, etc.)
    if M <= 64:
        return {
            "BLOCK_SIZE_M": 16 if M <= 16 else 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 2,
        }

    # Standard fallback for typical image-generation M (2048 / 4096 / …)
    # SRAM usage: (64*32 + 64*32) * 2 * 2 = 16 KB — safe on decade-old GPUs
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 2,
    }


def get_best_mot_config(M: int, N: int, K: int, dtype_str: str | None = None) -> tuple[int, dict[str, int]]:
    configs = get_mot_configs(K, N, dtype_str)
    if configs:
        loaded_m_key = min(configs.keys(), key=lambda x: abs(x - M))
        return loaded_m_key, configs[loaded_m_key]
    else:
        return -1, get_mot_default_config(M, N, K, dtype=dtype_str)


# =================================================================
#  Part 1: The Router (Routing Component)
#  Responsibilities: Handle PID mapping, Text/VAE distribution, indirect index loading, pointer calculation
# =================================================================
@triton.jit
def _get_mot_pointers(
    # System Inputs
    pid,
    # Matrix Pointers
    a_ptr,
    b_text_ptr,
    b_vae_ptr,
    bias_text_ptr,
    bias_vae_ptr,
    scale_a_ptr,
    scale_b_text_ptr,
    scale_b_vae_ptr,
    # Indices & Meta
    text_indices_ptr,
    vae_indices_ptr,
    M_text,
    M_vae,
    N,
    # Strides (need to select based on Text/VAE)
    stride_bk_text,
    stride_bn_text,
    stride_bk_vae,
    stride_bn_vae,
    # Block Config
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # 1. Calculate Text/VAE task boundaries
    num_pid_m_text = tl.cdiv(M_text, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # 2. MoT Routing
    # Initialize VAE variables first and overwrite for Text

    # VAE Path
    cur_pid = pid - (num_pid_m_text * num_pid_n)
    # Select VAE Pointers
    cur_b_ptr = b_vae_ptr
    cur_bias_ptr = bias_vae_ptr
    cur_scale_b_ptr = scale_b_vae_ptr
    cur_indices_ptr = vae_indices_ptr
    # Select VAE Strides / Limits
    cur_stride_bk = stride_bk_vae
    cur_stride_bn = stride_bn_vae
    M_limit = M_vae

    # Text Path
    if pid < num_pid_m_text * num_pid_n:
        cur_pid = pid
        # Select Text Pointers
        cur_b_ptr = b_text_ptr
        cur_bias_ptr = bias_text_ptr
        cur_scale_b_ptr = scale_b_text_ptr
        cur_indices_ptr = text_indices_ptr
        # Select Text Strides / Limits
        cur_stride_bk = stride_bk_text
        cur_stride_bn = stride_bn_text
        M_limit = M_text

    # 3. Calculate Grid coordinates(grouping)
    cur_num_pid_m = tl.cdiv(M_limit, BLOCK_SIZE_M)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = cur_pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m_adj = tl.minimum(cur_num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (cur_pid % group_size_m_adj)
    pid_n = (cur_pid % num_pid_in_group) // group_size_m_adj

    # 4. Load indirect indices (Indirect Indexing for A)
    # Calculate the M range covered by current Block [0, BLOCK_SIZE_M]
    offs_m_idx = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    m_mask = offs_m_idx < M_limit
    # Load real row indices from the index array
    # for very big K (eg. ffn down proj*TP=1,K=16k), if M is also huge(M>130K)
    # loading int32 indices may result in integer overflow when we compute offs_m
    real_row_idxs = tl.load(cur_indices_ptr + offs_m_idx, mask=m_mask, other=0).to(tl.int64)

    # 5. Calculate N-dimension Offsets
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    n_mask = offs_n < N

    return (
        pid_m,
        pid_n,  # Grid Coordinates
        real_row_idxs,
        m_mask,  # M-dim info (Indirect)
        offs_n,
        n_mask,  # N-dim info
        M_limit,  # Boundary
        cur_b_ptr,
        cur_bias_ptr,  # Selected Pointers
        cur_scale_b_ptr,  # Selected Scale Pointer
        cur_stride_bk,
        cur_stride_bn,  # Selected Strides
    )


# =================================================================
#  Part 2: Compute Cores
#  Responsibilities: Execute specific Loop structures based on QUANT_TYPE
# =================================================================


# Core A: Standard GEMM (for BF16/FP16 and W8A8)
# Feature: No dequantization inside Loop, Scale is applied after Loop ends
@triton.jit
def _core_standard_gemm(
    # Pointers
    a_ptr,
    b_ptr,
    # Offsets & Masks
    real_row_idxs,
    m_mask,
    offs_n,
    n_mask,
    offs_k,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    # Quant-Related
    scale_a_ptr,
    scale_b_ptr,
    stride_scale_a,
    stride_scale_b,
    # Loop Info
    K,
    BLOCK_SIZE_K: tl.constexpr,
    # Configs
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACCUMULATOR_DTYPE: tl.constexpr,
    IS_W8A8: tl.constexpr,
    # Accelerate Configs
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    STRIDE_AK_IS_1: tl.constexpr,
    STRIDE_BK_IS_1: tl.constexpr,
    STRIDE_BN_IS_1: tl.constexpr,
):
    # 1. Stride optimizations (Bypassing compiler limitations for unit strides)
    _stride_ak = 1 if STRIDE_AK_IS_1 else stride_ak
    _stride_bk = 1 if STRIDE_BK_IS_1 else stride_bk
    _stride_bn = 1 if STRIDE_BN_IS_1 else stride_bn

    # Pointer initialization (A uses indirect indexing, B uses standard striding)
    a_ptrs = a_ptr + (stride_am * real_row_idxs[:, None] + _stride_ak * offs_k[None, :])
    b_ptrs = b_ptr + (_stride_bk * offs_k[:, None] + _stride_bn * offs_n[None, :])

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUMULATOR_DTYPE)

    # 2. Unified Main Loop
    # Triton evaluates `constexpr` conditions at compile time, ensuring zero runtime overhead
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute K-mask only if needed
        if not EVEN_K:
            mask_k = offs_k < K - k * BLOCK_SIZE_K

        # Load A
        if EVEN_K:
            a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        else:
            a = tl.load(a_ptrs, mask=m_mask[:, None] & mask_k[None, :], other=0.0)

        # Load B
        if EVEN_K and EVEN_N:
            b = tl.load(b_ptrs)
        elif not EVEN_K and EVEN_N:
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        elif EVEN_K and not EVEN_N:
            b = tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)
        else:
            b = tl.load(b_ptrs, mask=mask_k[:, None] & n_mask[None, :], other=0.0)

        # Compute & Advance
        accumulator = tl.dot(a, b, accumulator, out_dtype=ACCUMULATOR_DTYPE)
        a_ptrs += BLOCK_SIZE_K * _stride_ak
        b_ptrs += BLOCK_SIZE_K * _stride_bk

    # 3. Epilogue (only needed for W8A8)
    if IS_W8A8:
        accumulator = accumulator.to(tl.float32)
        # Load Scale A
        scale_a_ptrs = scale_a_ptr + real_row_idxs * stride_scale_a
        sa = tl.load(scale_a_ptrs, mask=m_mask, other=1.0)
        accumulator = accumulator * sa[:, None]

        # Load Scale B
        scale_b_ptrs = scale_b_ptr + offs_n * stride_scale_b
        sb = tl.load(scale_b_ptrs, mask=n_mask, other=1.0)
        accumulator = accumulator * sb[None, :]

    return accumulator


# Core B: Weight Only GEMM (for W4A16 / W8A16)
# Feature: Dequantization inside Loop (Dequantize-on-the-fly)
# Currently only supports W8A16
@triton.jit
def _core_weight_only_gemm(
    # Pointers
    a_ptr,
    b_ptr,
    # Offsets & Masks
    real_row_idxs,
    m_mask,
    offs_n,
    n_mask,
    offs_k,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    # Quant-Related
    scale_b_ptr,
    stride_scale_b,  # 1(per-channel) or 0(per-tensor)
    # Loop Info
    K,
    BLOCK_SIZE_K: tl.constexpr,
    # Configs
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACCUMULATOR_DTYPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,  # bf16 or fp16
    WEIGHT_BITS: tl.constexpr,  # 4 or 8
    # Accelerate Configs
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    STRIDE_AK_IS_1: tl.constexpr,
    STRIDE_BK_IS_1: tl.constexpr,
    STRIDE_BN_IS_1: tl.constexpr,
):
    # Compile-time Check
    tl.static_assert(WEIGHT_BITS == 8, "For weight-only, we only support W8A16 at this point")

    # 1. Stride optimizations
    _stride_ak = 1 if STRIDE_AK_IS_1 else stride_ak
    _stride_bk = 1 if STRIDE_BK_IS_1 else stride_bk
    _stride_bn = 1 if STRIDE_BN_IS_1 else stride_bn

    a_ptrs = a_ptr + (stride_am * real_row_idxs[:, None] + _stride_ak * offs_k[None, :])
    b_ptrs = b_ptr + (_stride_bk * offs_k[:, None] + _stride_bn * offs_n[None, :])

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUMULATOR_DTYPE)

    # 2. Unified Main Loop
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if not EVEN_K:
            mask_k = offs_k < K - k * BLOCK_SIZE_K

        # Load A
        if EVEN_K:
            a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        else:
            a = tl.load(a_ptrs, mask=m_mask[:, None] & mask_k[None, :], other=0.0)

        # Load B (Int)
        if EVEN_K and EVEN_N:
            b_int = tl.load(b_ptrs)
        elif not EVEN_K and EVEN_N:
            b_int = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        elif EVEN_K and not EVEN_N:
            b_int = tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)
        else:
            b_int = tl.load(b_ptrs, mask=mask_k[:, None] & n_mask[None, :], other=0.0)

        # --- Dequantize Logic (Type Cast Only) ---
        # No multiplication here, correct as long as per-tensor/per-channel scaling (scale_b.shape=(N,))
        b_compute = b_int.to(COMPUTE_DTYPE)

        # Compute & Advance
        accumulator = tl.dot(a, b_compute, accumulator, out_dtype=ACCUMULATOR_DTYPE)

        a_ptrs += BLOCK_SIZE_K * _stride_ak
        b_ptrs += BLOCK_SIZE_K * _stride_bk

    # 3. Epilogue: Apply Scale B safely outside the loop
    # Load Scale B
    scale_b_ptrs = scale_b_ptr + offs_n * stride_scale_b
    sb = tl.load(scale_b_ptrs, mask=n_mask, other=1.0)  # shape=(N，)

    accumulator = accumulator * sb[None, :]
    return accumulator


# =================================================================
#  Part 3: Unified Entry Kernel
#  Responsibilities: Call Router, statically dispatch Core, store results
# =================================================================
@triton.jit
def mot_unified_gemm_kernel(
    # Inputs
    a_ptr,
    b_text_ptr,
    b_vae_ptr,
    c_ptr,
    bias_text_ptr,
    bias_vae_ptr,
    text_indices_ptr,
    vae_indices_ptr,
    # Dimensions & Strides
    M_text,
    M_vae,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk_text,
    stride_bn_text,
    stride_bk_vae,
    stride_bn_vae,
    stride_cm,
    stride_cn,
    # Scales (pass 0 if None)
    scale_a_ptr,
    scale_b_text_ptr,
    scale_b_vae_ptr,
    stride_scale_a,
    stride_scale_b,  # 1(per-channel) or 0(per-tensor)
    # Metas
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    # accelerate config
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    STRIDE_AK_IS_1: tl.constexpr,
    STRIDE_BK_IS_1: tl.constexpr,
    STRIDE_BN_IS_1: tl.constexpr,
    # Quant-related dtypes
    ACCUMULATOR_DTYPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    # Quant Control
    # 0=None, 1=W8A8, 2=W8A16, 3=W4A16
    QUANT_TYPE: tl.constexpr,
    HAS_BIAS: tl.constexpr = False,
):
    pid = tl.program_id(axis=0)

    # -----------------------------------------------------------
    # 1. Routing Phase (General)
    # -----------------------------------------------------------
    (
        pid_m,
        pid_n,
        real_row_idxs,
        m_mask,
        offs_n,
        n_mask,
        M_limit,
        cur_b_ptr,
        cur_bias_ptr,
        cur_scale_b_ptr,
        cur_stride_bk,
        cur_stride_bn,
    ) = _get_mot_pointers(
        pid,
        a_ptr,
        b_text_ptr,
        b_vae_ptr,
        bias_text_ptr,
        bias_vae_ptr,
        scale_a_ptr,
        scale_b_text_ptr,
        scale_b_vae_ptr,
        text_indices_ptr,
        vae_indices_ptr,
        M_text,
        M_vae,
        N,
        stride_bk_text,
        stride_bn_text,
        stride_bk_vae,
        stride_bn_vae,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        GROUP_SIZE_M,
    )

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # -----------------------------------------------------------
    # 2. Compute Phase (Static Dispatch)
    # -----------------------------------------------------------
    if QUANT_TYPE == 0:  # FP16 / BF16 Standard
        c = _core_standard_gemm(
            a_ptr,
            cur_b_ptr,
            real_row_idxs,
            m_mask,
            offs_n,
            n_mask,
            offs_k,
            stride_am,
            stride_ak,
            cur_stride_bk,
            cur_stride_bn,
            0,
            0,
            0,
            0,
            K,
            BLOCK_SIZE_K,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            ACCUMULATOR_DTYPE,
            IS_W8A8=False,
            EVEN_K=EVEN_K,
            EVEN_N=EVEN_N,
            STRIDE_AK_IS_1=STRIDE_AK_IS_1,
            STRIDE_BK_IS_1=STRIDE_BK_IS_1,
            STRIDE_BN_IS_1=STRIDE_BN_IS_1,
        )
    elif QUANT_TYPE == 1:  # W8A8
        c = _core_standard_gemm(
            a_ptr,
            cur_b_ptr,
            real_row_idxs,
            m_mask,
            offs_n,
            n_mask,
            offs_k,
            stride_am,
            stride_ak,
            cur_stride_bk,
            cur_stride_bn,
            scale_a_ptr,
            cur_scale_b_ptr,
            stride_scale_a,
            stride_scale_b,
            K,
            BLOCK_SIZE_K,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            ACCUMULATOR_DTYPE,
            IS_W8A8=True,
            EVEN_K=EVEN_K,
            EVEN_N=EVEN_N,
            STRIDE_AK_IS_1=STRIDE_AK_IS_1,
            STRIDE_BK_IS_1=STRIDE_BK_IS_1,
            STRIDE_BN_IS_1=STRIDE_BN_IS_1,
        )
    elif QUANT_TYPE == 2 or QUANT_TYPE == 3:  # Weight Only
        bits = 8 if QUANT_TYPE == 2 else 4
        c = _core_weight_only_gemm(
            a_ptr,
            cur_b_ptr,
            real_row_idxs,
            m_mask,
            offs_n,
            n_mask,
            offs_k,
            stride_am,
            stride_ak,
            cur_stride_bk,
            cur_stride_bn,
            cur_scale_b_ptr,
            stride_scale_b,
            K,
            BLOCK_SIZE_K,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            ACCUMULATOR_DTYPE,
            COMPUTE_DTYPE,
            WEIGHT_BITS=bits,
            EVEN_K=EVEN_K,
            EVEN_N=EVEN_N,
            STRIDE_AK_IS_1=STRIDE_AK_IS_1,
            STRIDE_BK_IS_1=STRIDE_BK_IS_1,
            STRIDE_BN_IS_1=STRIDE_BN_IS_1,
        )

    # -----------------------------------------------------------
    # 3. Store Phase (General)
    # -----------------------------------------------------------

    # Bias Add
    if HAS_BIAS:
        bias = tl.load(cur_bias_ptr + offs_n, mask=n_mask, other=0.0)
        c = c + bias[None, :]  # reshape to (1, BLOCK_SIZE_N)

    # Cast C into the output dtype
    c = c.to(OUTPUT_DTYPE)

    # Store
    offs_cm = real_row_idxs
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_n[None, :]

    # Need to be careful with the Mask here: real_row_idxs may contain 0
    # (out-of-bounds padding), but c_mask needs the real boundary
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, c, mask=store_mask)


# Define is_weak_contiguous
def is_weak_contiguous(x: torch.Tensor):
    strides = x.stride()
    sizes = x.shape
    is_not_transpose = strides[0] == 1 and (strides[1] >= max(1, sizes[0]))
    is_transpose = strides[1] == 1 and (strides[0] >= max(1, sizes[1]))
    return is_transpose or is_not_transpose


def invoke_mot_gemm(
    # Inputs
    A: torch.Tensor,
    B_text: torch.Tensor,
    B_vae: torch.Tensor,
    C: torch.Tensor,
    bias_text: torch.Tensor | None,
    bias_vae: torch.Tensor | None,
    # Indices
    text_indices: torch.Tensor,
    vae_indices: torch.Tensor,
    # Quant Scales (None if disabled)
    A_scale: torch.Tensor | None,
    B_text_scale: torch.Tensor | None,
    B_vae_scale: torch.Tensor | None,
    # Quant Flags
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    # Quant Type
    A_per_channel_quant: bool,
    B_per_channel_quant: bool,
    # Config
    config: dict[str, Any] | None = None,
):
    # ------ 1. Basic Assertions ------
    M = A.size(0)
    K = B_text.size(0)
    N = B_text.size(1)

    if config is None:
        if use_fp8_w8a8:
            _dtype_str: str | None = "fp8_w8a8"
        elif use_int8_w8a16:
            _dtype_str = "int8_w8a16"
        else:
            _dtype_str = None

        loaded_m_key, config = get_best_mot_config(M, N, K, _dtype_str)

    assert len(A.shape) == 2 and len(C.shape) == 2, (
        "The input tensor and output tensor should be flattened to (batch_size*seq_len, hidden_dim)"
    )

    assert K == A.size(1), "the weights' first dimension should matchinputtensor's last dimension (hidden_dim)"

    assert K == B_vae.size(0) and N == B_vae.size(1), (
        "the weights dimension for text expert andimage expert should be the same"
    )

    assert C.size(0) == M and C.size(1) == N, "the output tensor shape is not correct"

    M_text = text_indices.size(0)
    M_vae = vae_indices.size(0)
    assert M_text + M_vae == M, "the length sum of text and image indices should match input tensor's first dimension"
    if bias_text is not None:
        assert bias_text.dtype == C.dtype, "the bias tensor dtype should match the output tensor dtype"
    if bias_vae is not None:
        assert bias_vae.dtype == C.dtype, "the bias tensor dtype should match the output tensor dtype"

    assert is_weak_contiguous(A)
    assert is_weak_contiguous(B_text)
    assert is_weak_contiguous(B_vae)

    # --- 2. Quantization Logic Translation ---

    def triton_dtype(torch_dtype):
        if torch_dtype == torch.float8_e4m3fn:
            return getattr(tl, "float8e4m3fn", tl.float8e4nv)
        elif torch_dtype == torch.float8_e5m2:
            return tl.float8e5
        return {
            torch.float16: tl.float16,
            torch.bfloat16: tl.bfloat16,
            torch.float32: tl.float32,
            torch.int8: tl.int8,
            torch.float8_e4m3fn: tl.float8e4nv,
            torch.float8_e5m2: tl.float8e5,
        }[torch_dtype]

    # Determine QUANT_TYPE
    # 0=None, 1=W8A8, 2=W8A16, 3=W4A16
    quant_type = 0
    ACCUMULATOR_DTYPE = tl.float32
    COMPUTE_DTYPE = triton_dtype(A.dtype)
    OUTPUT_DTYPE = triton_dtype(C.dtype)

    if use_int8_w8a8 or use_fp8_w8a8:
        quant_type = 1
        assert A_scale is not None, "W8A8 requires A_scale"
        assert B_text_scale is not None and B_vae_scale is not None, "W8A8 requires B_text_scale and B_vae_scale"
        if use_int8_w8a8:
            ACCUMULATOR_DTYPE = tl.int32
            assert (
                A.dtype == torch.int8
                and B_text.dtype == torch.int8
                and B_vae.dtype == torch.int8
                and C.dtype in [torch.float16, torch.bfloat16]
            ), "if you want to use INT8_W8A8, A should be INT8, B should be INT8, C should be FP16/BF16"
        else:
            ACCUMULATOR_DTYPE = tl.float32
            assert (
                A.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
                and B_text.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
                and B_vae.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
                and C.dtype in [torch.float16, torch.bfloat16]
            ), "if you want to use FP8_W8A8, A should be FP8, B should be FP8, C should be FP16/BF16"
    elif use_int8_w8a16:
        quant_type = 2
        assert B_text_scale is not None and B_vae_scale is not None, "W8A16 requires B_text_scale and B_vae_scale"
        ACCUMULATOR_DTYPE = tl.float32
        assert (
            A.dtype in [torch.float16, torch.bfloat16]
            and B_text.dtype == torch.int8
            and B_vae.dtype == torch.int8
            and C.dtype in [torch.float16, torch.bfloat16]
        ), "if you want to use INT8_W8A16, A should be FP16/BF16, B should be INT8, C should be FP16/BF16"

    elif use_int4_w4a16:
        raise NotImplementedError("For weight-only, we only support W8A16 at this point")
        # quant_type = 3
        # ACCUMULATOR_DTYPE=tl.float32

    # accelerate config
    EVEN_K = K % config["BLOCK_SIZE_K"] == 0
    EVEN_N = N % config["BLOCK_SIZE_N"] == 0
    STRIDE_AK_IS_1 = A.stride(1) == 1
    STRIDE_BK_IS_1 = (B_text.stride(0) == 1) and (B_vae.stride(0) == 1)
    STRIDE_BN_IS_1 = (B_text.stride(1) == 1) and (B_vae.stride(1) == 1)

    # bias check
    assert (bias_text is None) == (bias_vae is None), (
        "Bias must be provided for both Text and VAE simultaneously, or neither."
    )
    has_bias = bias_text is not None

    # --- 3. Grid Calculation ---
    def grid(META):
        return (
            (triton.cdiv(M_text, META["BLOCK_SIZE_M"]) + triton.cdiv(M_vae, META["BLOCK_SIZE_M"]))
            * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    # --- 4. Launch ---
    run_config = config.copy()
    run_config.update(
        {
            "QUANT_TYPE": quant_type,
            "ACCUMULATOR_DTYPE": ACCUMULATOR_DTYPE,
            "COMPUTE_DTYPE": COMPUTE_DTYPE,
            "OUTPUT_DTYPE": OUTPUT_DTYPE,
            "HAS_BIAS": has_bias,
            "EVEN_K": EVEN_K,
            "EVEN_N": EVEN_N,
            "STRIDE_AK_IS_1": STRIDE_AK_IS_1,
            "STRIDE_BK_IS_1": STRIDE_BK_IS_1,
            "STRIDE_BN_IS_1": STRIDE_BN_IS_1,
        }
    )

    # Pointers (Handle None -> 0)
    p_a_scale = A_scale if A_scale is not None else 0
    p_b_text_scale = B_text_scale if B_text_scale is not None else 0
    p_b_vae_scale = B_vae_scale if B_vae_scale is not None else 0
    p_bias_text = bias_text if bias_text is not None else 0
    p_bias_vae = bias_vae if bias_vae is not None else 0

    # Quantization granularity
    stride_scale_a = 1 if A_per_channel_quant else 0
    stride_scale_b = 1 if B_per_channel_quant else 0

    mot_unified_gemm_kernel[grid](
        # Inputs
        A,
        B_text,
        B_vae,
        C,
        p_bias_text,
        p_bias_vae,
        text_indices,
        vae_indices,
        # Dimensions
        M_text,
        M_vae,
        N,
        K,
        # Strides
        A.stride(0),
        A.stride(1),
        B_text.stride(0),
        B_text.stride(1),
        B_vae.stride(0),
        B_vae.stride(1),
        C.stride(0),
        C.stride(1),
        # Scales
        p_a_scale,
        p_b_text_scale,
        p_b_vae_scale,
        stride_scale_a,
        stride_scale_b,
        # Config
        **run_config,
    )
