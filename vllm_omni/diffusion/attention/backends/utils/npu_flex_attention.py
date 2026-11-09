# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Adapt request-owned block-causal metadata to MindIE-SD NPU FlexAttention.

Omni owns model selection and request lifetime. MindIE owns mask construction,
compilation and local execution. This adapter does not support sequence parallelism.
This module never imports a vendor runtime for another device or FA path.
"""

from __future__ import annotations

import importlib
from functools import cache
from types import ModuleType
from typing import TYPE_CHECKING, NamedTuple

import torch

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

NPU_FLEX_VALID_TOKENS = "npu_flex_valid_tokens"


class _FlexRequest(NamedTuple):
    """Device-local state shared only by layers of one immutable prefill."""

    signature: tuple[object, ...]
    mask: object


@cache
def load_mindie_flex_runtime() -> ModuleType:
    """Load the required API lazily, distinguishing missing and broken installs.

    Missing packages or entry points produce actionable ImportErrors. An error
    importing a transitive dependency retains its original type and cause.
    Successful API discovery does not compile or execute the kernel.
    """
    try:
        runtime = importlib.import_module("mindiesd")
    except ModuleNotFoundError as exc:
        if exc.name != "mindiesd":
            raise
        raise ImportError(
            "MindIE-SD with CATLASS DSL APIs is required for NPU FlexAttention. "
            "Install a compatible MindIE-SD build before starting NPU FlexAttention."
        ) from exc
    required = ("flex_attention", "make_block_causal_mask", "check_flex_attention_device")
    missing = [name for name in required if not callable(getattr(runtime, name, None))]
    if missing:
        raise ImportError(
            "Upgrade your MindIE-SD build to use CATLASS DSL FlexAttention; "
            f"the installed package lacks these APIs: {', '.join(missing)}."
        )
    return runtime


def _can_use_flex(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: AttentionMetadata | None,
) -> bool:
    """Check the shared adapter contract before loading MindIE-SD.

    This adapter selects BF16 single-sample block-causal attention with complete local Q/K/V.
    The public MindIE API has its own broader tensor contract.
    """
    if (
        query.device.type != "npu"
        or metadata is None
        or query.ndim != 4
        or key.ndim != 4
        or value.ndim != 4
        or query.shape[0] != 1
        or key.shape[0] != 1
        or key.shape != value.shape
        or query.shape[1] != key.shape[1]
        or query.shape[-1] != key.shape[-1]
        or min(query.shape) <= 0
        or min(key.shape) <= 0
        or query.shape[2] % key.shape[2] != 0
        or query.dtype != torch.bfloat16
        or key.dtype != query.dtype
        or value.dtype != query.dtype
        or key.device != query.device
        or value.device != query.device
        or metadata.full_attn_spans is None
        or len(metadata.full_attn_spans) != 1
        or metadata.attn_mask is not None
        or metadata.query_ranges is not None
        or metadata.joint_key is not None
        or metadata.joint_value is not None
        or metadata.extra.get("kv_cache_dtype") not in (None, "float", "auto")
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
    ):
        return False
    cursor = 0
    for start, end in metadata.full_attn_spans[0]:
        if not (cursor <= start < end <= key.shape[1]):
            return False
        cursor = end
    return True


def try_npu_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: AttentionMetadata | None,
    softmax_scale: float,
) -> torch.Tensor | None:
    """Execute local MindIE-SD NPU FlexAttention for supported metadata and tensors.

    Unsupported model contracts return None; import, compilation and execution
    failures propagate.
    """
    if not _can_use_flex(query, key, value, metadata):
        return None
    assert metadata is not None and metadata.full_attn_spans is not None
    runtime = load_mindie_flex_runtime()
    runtime.check_flex_attention_device(query.device)
    spans = tuple(metadata.full_attn_spans[0])
    seq_len = key.shape[1]
    valid_tokens = metadata.extra.get(NPU_FLEX_VALID_TOKENS)
    # The validity tensor and callback captures are immutable through this
    # prefill. Never retain them in a process-wide cache.
    validity_version = None if valid_tokens is None or valid_tokens.is_inference() else valid_tokens._version
    signature = (
        spans,
        str(query.device),
        query.shape,
        key.shape,
        id(valid_tokens),
        validity_version,
    )
    shared = metadata.extra.setdefault("_npu_flex_cache", {})
    request = shared.get("request")
    if request is None or request.signature != signature:
        mask = runtime.make_block_causal_mask(spans, seq_len, device=query.device, valid_tokens=valid_tokens)
        request = _FlexRequest(signature, mask)
        shared["request"] = request
    return runtime.flex_attention(
        query,
        key,
        value,
        mask=request.mask,
        scale=softmax_scale,
        layout="BSND",
        enable_gqa=query.shape[2] != key.shape[2],
    )
