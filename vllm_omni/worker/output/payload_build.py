# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request multimodal payload builders.

State-free extraction helpers that turn a step's `multimodal_outputs`
(combined prefix-cache-merged form, or the direct `mm_cpu` form) into one
request's payload dict. Sparse-routing context (`audio_sparse_output`,
`sparse_mm_index`) is resolved upstream by
`vllm_omni.worker.sparse_audio.resolve_sparse_mm_routing` and arrives here
as plain arguments.
"""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.utils.mm_outputs import to_payload_element

logger = init_logger(__name__)

_SPARSE_AUDIO_PROTOCOL_KEYS = frozenset({"meta.req_id", "meta.sparse_audio"})


def unwrap_combined_payload_value(
    value: Any,
    *,
    rid: str,
    list_idx: int,
    mm_key: str,
) -> tuple[bool, Any]:
    """Unwrap one value from the prefix-cache-merged payload.

    Returns ``(keep, unwrapped)``: ``keep`` is False when a non-singleton
    per-request list is misaligned and must be dropped. A singleton list is
    request-invariant passthrough data and is shared across the batch.
    """
    if isinstance(value, list):
        if len(value) == 1:
            # Prefix-cache passthrough intentionally preserves list
            # containers and copies the whole list under every request id.
            # Qwen3-Omni uses this shape for its request-invariant TTS
            # BOS/EOS/PAD embeddings, so every batch index unwraps the same
            # singleton. Sparse one-request lists are safe here too: routing
            # limits them to the declared request at sparse index zero.
            return True, value[0]
        if list_idx < len(value):
            return True, value[list_idx]
        # A non-singleton list shorter than the request's index is per-request
        # data that lost alignment. Never substitute another request's
        # element (`v[0]` used to ship request 0's payload as `rid`'s); drop
        # the key loudly instead — the SAME protocol-break policy as the
        # sparse mismatch in
        # `build_omni_mm_payload`. Not a raise: v1 runners do not catch
        # per-request exceptions, so raising would turn a per-request
        # protocol break into a whole-stage outage.
        logger.error(
            "Combined prefix-cache mm payload misaligned for request %s: index %d out of "
            "range for per-request list of length %d; dropping key %s.",
            rid,
            list_idx,
            len(value),
            mm_key,
        )
        return False, None
    if isinstance(value, dict):
        nested = {}
        for k, sv in value.items():
            keep, unwrapped = unwrap_combined_payload_value(sv, rid=rid, list_idx=list_idx, mm_key=mm_key)
            if keep:
                nested[k] = unwrapped
        return True, nested
    return True, value


def build_combined_prefix_cache_mm_payload(
    combined_multimodal_outputs: dict,
    *,
    rid: str,
    list_idx: int,
) -> dict[str, object]:
    """Unwrap one request's entry from the prefix-cache-merged payload.

    Sparse-agnostic by design: ``list_idx`` arrives pre-resolved by the
    caller (`build_omni_mm_payload`, which owns the sparse-routing context)
    — batch index normally, sparse index under sparse routing.
    """
    payload: dict[str, object] = {}
    for mm_key in combined_multimodal_outputs.keys():
        if mm_key in _SPARSE_AUDIO_PROTOCOL_KEYS:
            continue
        keep, unwrapped = unwrap_combined_payload_value(
            combined_multimodal_outputs[mm_key][rid],
            rid=rid,
            list_idx=list_idx,
            mm_key=mm_key,
        )
        if keep:
            payload[mm_key] = unwrapped
    return payload


def build_omni_mm_payload(
    *,
    combined_multimodal_outputs: dict | None,
    mm_cpu: dict[str, object] | None,
    rid: str,
    idx: int,
    start: int,
    end: int,
    audio_sparse_output: bool,
    sparse_mm_index: dict[str, int],
    hidden_seq_len: int,
    scheduled_seq_len: int,
) -> dict[str, object]:
    """Build one request's multimodal payload for the step."""
    if combined_multimodal_outputs:
        # Sparse-audio producers emit per-request lists aligned to the
        # sparse request order (`meta.req_id`), not the batch order; under
        # sparse routing the request's sparse index selects its element.
        list_idx = sparse_mm_index.get(rid, idx) if audio_sparse_output else idx
        return build_combined_prefix_cache_mm_payload(
            combined_multimodal_outputs,
            rid=rid,
            list_idx=list_idx,
        )

    mm_payload: dict[str, object] = {}
    if not mm_cpu:
        return mm_payload

    for mm_key, mm_val in mm_cpu.items():
        if mm_key in _SPARSE_AUDIO_PROTOCOL_KEYS:
            continue
        if audio_sparse_output and isinstance(mm_val, list):
            sparse_idx = sparse_mm_index.get(rid)
            if sparse_idx is None:
                continue
            if sparse_idx >= len(mm_val):
                # Misalignment means the sparse-marker protocol broke; the
                # request's output for this key is dropped, so be loud.
                logger.error(
                    "Sparse multimodal payload mismatch for request %s: index %d >= %d; dropping key %s.",
                    rid,
                    sparse_idx,
                    len(mm_val),
                    mm_key,
                )
                continue
            sparse_val = mm_val[sparse_idx]
            mm_payload[mm_key] = sparse_val.clone() if isinstance(sparse_val, torch.Tensor) else sparse_val
            continue
        mm_payload[mm_key] = to_payload_element(
            element=mm_val,
            idx=idx,
            start=start,
            end=end,
            pass_lists_through=False,
            seq_len=hidden_seq_len,
            scheduled_seq_len=scheduled_seq_len,
        )
    return mm_payload
