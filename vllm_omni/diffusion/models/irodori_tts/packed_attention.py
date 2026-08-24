# SPDX-License-Identifier: Apache-2.0
"""Packed attention backend selection for Irodori-TTS."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import torch
from packaging.version import InvalidVersion, Version
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.utils.fa import flash_attn_varlen_func

logger = init_logger(__name__)

try:
    from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

    HAS_FLASHINFER = True
except Exception:
    BatchPrefillWithRaggedKVCacheWrapper = None
    HAS_FLASHINFER = False


def _has_unsafe_sm120_fa4(device: torch.device) -> bool:
    if torch.cuda.get_device_capability(device) != (12, 0):
        return False
    module = getattr(flash_attn_varlen_func, "__module__", "")
    if not module.startswith("flash_attn.cute"):
        return False
    try:
        return Version(version("flash-attn-4")) <= Version("4.0.0b26")
    except (PackageNotFoundError, InvalidVersion):
        return True


def resolve_packed_attention_backend(device: torch.device, dtype: torch.dtype | None) -> str | None:
    """Prefer FA4/FA3, using FlashInfer when the selected FA kernel is unsafe."""
    if device.type != "cuda" or dtype not in (torch.bfloat16, torch.float16):
        return None
    major, _minor = torch.cuda.get_device_capability(device)
    has_flash_attention = flash_attn_varlen_func is not None
    if has_flash_attention and (major < 10 or not _has_unsafe_sm120_fa4(device)):
        return "flash-attn"
    if HAS_FLASHINFER:
        return "flashinfer"
    return None


class _FlashInferRaggedRunner:
    def __init__(self, device: torch.device) -> None:
        assert BatchPrefillWithRaggedKVCacheWrapper is not None
        self.device = device
        self.workspace = torch.empty(128 * 1024 * 1024, device=device, dtype=torch.uint8)
        self.wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace,
            kv_layout="NHD",
            backend="auto",
        )
        self.plan_key: tuple[object, ...] | None = None

    @torch.compiler.disable
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        key = (
            cu_seqlens_q.data_ptr(),
            cu_seqlens_k.data_ptr(),
            cu_seqlens_q.shape[0],
            q.shape[1],
            q.shape[2],
            q.dtype,
            k.dtype,
            v.dtype,
            softmax_scale,
        )
        if key != self.plan_key:
            self.wrapper.plan(
                cu_seqlens_q,
                cu_seqlens_k,
                q.shape[1],
                k.shape[1],
                q.shape[2],
                head_dim_vo=v.shape[2],
                causal=False,
                sm_scale=softmax_scale,
                q_data_type=q.dtype,
                kv_data_type=k.dtype,
                o_data_type=q.dtype,
            )
            self.plan_key = key
        return self.wrapper.run(q, k, v)


_FLASHINFER_RUNNERS: dict[int, _FlashInferRaggedRunner] = {}


@torch.compiler.disable
def _run_flashinfer_ragged(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    device_index = q.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    runner = _FLASHINFER_RUNNERS.get(device_index)
    if runner is None:
        runner = _FlashInferRaggedRunner(q.device)
        _FLASHINFER_RUNNERS[device_index] = runner
        logger.info_once("Using FlashInfer ragged attention for Irodori packed batches")
    return runner(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale)


def run_packed_attention(
    backend: str,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
) -> torch.Tensor:
    if backend == "flashinfer":
        return _run_flashinfer_ragged(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale)
    if backend != "flash-attn" or flash_attn_varlen_func is None:
        raise RuntimeError("Irodori packed attention has no available backend.")
    output = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=False,
        softmax_scale=softmax_scale,
    )
    return output[0] if isinstance(output, tuple) else output
