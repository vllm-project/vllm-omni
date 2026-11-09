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


def run_npu_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: AttentionMetadata,
    softmax_scale: float,
) -> torch.Tensor:
    """Execute local MindIE FlexAttention with request-owned span metadata.

    Omni restricts this integration to single-sample BF16 prefill with complete
    local Q/K/V. MindIE validates tensor compatibility and span geometry.
    Unsupported inputs raise ValueError; dependency and execution errors propagate.
    """
    if query.device.type != "npu":
        raise ValueError("NPU FlexAttention requires NPU inputs.")
    if any(tensor.ndim != 4 for tensor in (query, key, value)):
        raise ValueError("NPU FlexAttention requires BSND inputs.")
    if query.shape[0] != 1 or query.dtype != torch.bfloat16:
        raise ValueError("NPU FlexAttention requires one BF16 sample.")
    if query.shape[1] != key.shape[1]:
        raise ValueError("NPU FlexAttention requires equal Q/K sequence lengths.")
    if metadata.full_attn_spans is None or len(metadata.full_attn_spans) != 1:
        raise ValueError("NPU FlexAttention requires single-sample block-causal spans.")
    if any(
        field is not None
        for field in (metadata.attn_mask, metadata.query_ranges, metadata.joint_key, metadata.joint_value)
    ):
        raise ValueError(
            "NPU FlexAttention requires local span metadata without dense masks, query ranges or joint K/V."
        )
    if metadata.extra.get("kv_cache_dtype") not in (None, "float", "auto"):
        raise ValueError("NPU FlexAttention does not support attention quantization.")
    # Reject outer compilation before constructing masks or mutating the request cache.
    if torch.compiler.is_compiling():
        raise ValueError("NPU FlexAttention must run outside an outer torch.compile region.")

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
