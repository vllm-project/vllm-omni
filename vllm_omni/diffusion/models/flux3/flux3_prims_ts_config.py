# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured PrimsTS GEMM configurations and optional autotuning for Flux3."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import os
from pathlib import Path
from typing import TYPE_CHECKING

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from torch import Tensor, device


AUTOTUNE_ENV = "VLLM_OMNI_FLUX3_PRIMS_TS_AUTOTUNE"
TUNING_BUCKETS_ENV = "VLLM_OMNI_FLUX3_PRIMS_TS_TUNING_BUCKETS"
_M_BUCKETS = (1024, 8192, 10240, 32768, 40960)

# (format, epilogue, K, packed N) -> M bucket ->
# (cluster_m, cluster_n, tile_k, overlap, stages, mma_k, use_clc,
# epilogue_warps, use_tma_store).
_TACTICS = {
    ("fp8_e4m3", "linear", 3072, 9216): {
        1024: (2, 2, 128, False, 6, 64, True, 8, False),
        8192: (2, 2, 128, False, 5, 64, True, 4, False),
        10240: (2, 2, 128, False, 6, 64, True, 4, False),
        32768: (2, 2, 128, False, 6, 64, True, 4, False),
        40960: (2, 2, 128, False, 6, 64, True, 8, False),
    },
    ("fp8_e4m3", "linear", 3072, 3072): {
        1024: (2, 2, 128, False, 6, 64, True, 8, True),
        8192: (2, 2, 128, False, 6, 64, True, 4, True),
        10240: (2, 2, 128, False, 6, 64, True, 4, True),
        32768: (4, 4, 128, False, 6, 64, True, 4, True),
        40960: (4, 4, 128, False, 6, 64, True, 4, True),
    },
    ("fp8_e4m3", "swiglu", 3072, 18432): {
        1024: (2, 2, 128, False, 6, 64, True, 8, False),
        8192: (4, 4, 128, False, 6, 64, True, 8, False),
        10240: (4, 4, 128, False, 5, 64, True, 8, False),
        32768: (2, 2, 128, False, 6, 64, True, 8, False),
        40960: (2, 2, 128, False, 5, 64, True, 8, False),
    },
    ("fp8_e4m3", "linear", 9216, 3072): {
        1024: (2, 2, 128, False, 6, 64, True, 4, True),
        8192: (4, 4, 128, False, 6, 64, True, 4, True),
        10240: (2, 2, 128, False, 6, 64, True, 4, True),
        32768: (2, 2, 128, False, 6, 64, True, 4, False),
        40960: (4, 4, 128, False, 6, 64, True, 4, False),
    },
    ("fp8_e4m3", "qkv_qknorm_rope", 3072, 9216): {
        1024: (2, 2, 128, False, 4, 64, True, 8, True),
        8192: (2, 2, 128, False, 4, 64, True, 8, True),
        10240: (2, 2, 128, False, 4, 64, True, 8, True),
        32768: (4, 4, 128, False, 4, 64, True, 8, True),
        40960: (4, 4, 128, False, 4, 64, True, 8, True),
    },
    ("nvfp4_e2m1", "linear", 3072, 9216): {
        1024: (2, 2, 256, True, 5, 64, True, 8, False),
        8192: (2, 2, 256, True, 5, 64, True, 8, False),
        10240: (2, 2, 256, True, 5, 64, True, 8, False),
        32768: (2, 2, 256, True, 5, 64, True, 8, False),
        40960: (4, 4, 256, True, 5, 64, True, 8, False),
    },
    ("nvfp4_e2m1", "linear", 3072, 3072): {
        1024: (2, 2, 256, False, 5, 64, True, 8, True),
        8192: (4, 4, 256, True, 5, 64, True, 8, False),
        10240: (2, 2, 256, True, 5, 64, True, 8, False),
        32768: (4, 4, 256, True, 5, 64, True, 8, False),
        40960: (4, 4, 256, True, 5, 64, True, 8, False),
    },
    ("nvfp4_e2m1", "linear", 9216, 3072): {
        1024: (4, 4, 256, False, 5, 64, True, 4, True),
        8192: (2, 2, 256, True, 5, 64, True, 8, False),
        10240: (2, 2, 256, True, 5, 64, True, 8, False),
        32768: (4, 4, 256, True, 5, 64, True, 8, False),
        40960: (4, 4, 256, True, 5, 64, True, 8, False),
    },
    ("nvfp4_e2m1", "swiglu", 3072, 18432): {
        1024: (4, 4, 256, False, 5, 64, True, 8, False),
        8192: (2, 2, 256, False, 5, 64, True, 8, False),
        10240: (4, 4, 256, False, 5, 64, True, 8, False),
        32768: (4, 4, 256, False, 5, 64, True, 8, False),
        40960: (2, 2, 256, False, 5, 64, True, 8, False),
    },
    ("nvfp4_e2m1", "qkv_qknorm_rope", 3072, 9216): {
        1024: (2, 2, 256, False, 4, 64, True, 8, True),
        8192: (2, 2, 256, False, 4, 64, True, 8, True),
        10240: (2, 2, 256, False, 4, 64, True, 8, True),
        32768: (4, 4, 256, False, 4, 64, True, 8, True),
        40960: (4, 4, 256, False, 4, 64, True, 8, True),
    },
}


def prims_ts_autotune_enabled() -> bool:
    value = os.environ.get(AUTOTUNE_ENV, "0").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{AUTOTUNE_ENV} must be a boolean, got {value!r}")


def flux3_prims_ts_gemm_config(
    device: device,
    *,
    operand_format: str,
    epilogue: str,
    k: int,
    n: int,
    m: int,
    output_format: str = "bf16",
    head_dim: int | None = None,
    qkv_scale: bool = False,
):
    """Return a measured config, or ``None`` to use FlashInfer autotuning."""
    if prims_ts_autotune_enabled():
        return None

    capability = current_platform.get_device_capability(device.index or 0)
    if capability is None:
        raise RuntimeError("Unable to determine the CUDA capability for a PrimsTS launch")

    bucket = next((candidate for candidate in _M_BUCKETS if m <= candidate), _M_BUCKETS[-1])
    tactic = _TACTICS.get((operand_format, epilogue, k, n), {}).get(bucket)
    common = dict(
        arch=capability.to_int(),
        operand_format=operand_format,
        output_format=output_format,
        epilogue=epilogue,
        has_bias=False,
        head_dim=head_dim,
        is_neox=False if epilogue == "qkv_qknorm_rope" else None,
        has_qkv_scale=qkv_scale,
    )
    if tactic is None:
        return _config_type()(**common)

    cluster_m, cluster_n, tile_k, overlap, stages, mma_k, use_clc, warps, use_tma_store = tactic
    config = _config_type()(
        **common,
        tile_m=256,
        tile_n=256,
        tile_k=tile_k,
        cluster_shape=(cluster_m, cluster_n, 1),
        scheduler="clc_dynamic" if use_clc else "static",
        nvfp4_mma_k=mma_k,
        epilogue_warps=warps,
        tmem_overlap=overlap,
        ab_stages=stages,
        use_tma_store=use_tma_store,
    )
    return _with_nvfp4_epilogue_overlap(config)


@functools.cache
def _overlap_epilogues() -> frozenset[str]:
    # FlashInfer adds tmem_overlap to these prepare functions together with allowing
    # it in the kernel config (SwiGLU, then QKV/QKNorm/RoPE); older builds reject it.
    from flashinfer.prims_ts import gemm

    prepare_fns = {
        "swiglu": gemm.prepare_fp4_linear_swiglu,
        "qkv_qknorm_rope": gemm.prepare_fp4_qkv_qknorm_rope,
    }
    return frozenset(
        epilogue
        for epilogue, prepare_fn in prepare_fns.items()
        if "tmem_overlap" in inspect.signature(prepare_fn).parameters
    )


def _with_nvfp4_epilogue_overlap(config):
    # The next tile's MMA runs during this tile's epilogue (bit-identical). The table
    # was tuned without SwiGLU/QKV overlap, so add it where FlashInfer accepts it.
    if (
        getattr(config, "operand_format", None) != "nvfp4_e2m1"
        or config.tmem_overlap
        or config.epilogue not in _overlap_epilogues()
        # Kernel constraints for overlap; QKV also needs MMA-K 64.
        or (config.tile_n, config.tile_k, config.epilogue_warps) != (256, 256, 8)
        or (config.epilogue == "qkv_qknorm_rope" and config.nvfp4_mma_k != 64)
    ):
        return config
    # Overlap and the TMA output store cannot share tensor memory.
    return dataclasses.replace(config, tmem_overlap=True, use_tma_store=False)


def flux3_fp8_gemm_config(x: Tensor, weight: Tensor, epilogue: str, *, head_dim: int | None = None):
    """Build the Flux3 FP8 tactic for the input and projection shape."""
    return flux3_prims_ts_gemm_config(
        x.device,
        operand_format="fp8_e4m3",
        epilogue=epilogue,
        k=weight.shape[1],
        n=weight.shape[0],
        m=x.numel() // x.shape[-1],
        head_dim=head_dim,
    )


def flux3_nvfp4_gemm_config(
    device: device,
    *,
    epilogue: str,
    k: int,
    n: int,
    m: int,
    head_dim: int | None = None,
    qkv_scale: bool = False,
):
    """Build the Flux3 NVFP4 tactic for one prepared projection."""
    return flux3_prims_ts_gemm_config(
        device,
        operand_format="nvfp4_e2m1",
        epilogue=epilogue,
        k=k,
        n=n,
        m=m,
        output_format="nvfp4_e2m1" if epilogue == "swiglu" else "bf16",
        head_dim=head_dim,
        qkv_scale=qkv_scale,
    )


def _config_type():
    from flashinfer.prims_ts.gemm import PrimsTsGemmConfig

    return PrimsTsGemmConfig


def flux3_prims_ts_autotune_context(model_path: str, *, tune_mode: bool = True):
    """Build a persistent FlashInfer tuner context scoped to model and rank."""
    import torch
    from flashinfer.autotuner import autotune

    cache_root = Path(os.environ.get("VLLM_CACHE_ROOT", "~/.cache/vllm")).expanduser()
    model_key = hashlib.sha256(str(model_path).encode()).hexdigest()[:16]
    rank = int(os.environ.get("RANK", "0"))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    device_index = torch.accelerator.current_device_index()
    device_name = torch.cuda.get_device_name(device_index).replace(" ", "_")
    cache_path = (
        cache_root / "flashinfer" / "autotune" / "flux3_prims_ts" / device_name / model_key / f"rank{rank}.json"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    raw_buckets = os.environ.get(TUNING_BUCKETS_ENV)
    buckets = tuple(int(value.strip()) for value in raw_buckets.split(",")) if raw_buckets else None
    if buckets is not None and (not buckets or any(bucket <= 0 for bucket in buckets)):
        raise ValueError(f"{TUNING_BUCKETS_ENV} must contain positive comma-separated M buckets")
    return autotune(tune_mode, cache=str(cache_path), tuning_buckets=buckets, round_up=True)
