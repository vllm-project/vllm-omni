"""Utilities for handling multimodal outputs / building multimodal output
payloads, most of which are shared by the prefix cache / no prefix cache path.
"""

from collections.abc import Mapping

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Flat payload keys partitioned at worker output into inter-stage connector
# payloads vs client-facing multimodal outputs.  Only final output roots are
# listed here; everything else remains available for stage-to-stage transport.
_CLIENT_MM_ROOT_KEYS: frozenset[str] = frozenset(
    {
        "model_outputs",
        "sr",
        "audio",
        "image",
        "images",
        "video",
        "videos",
        "trajectory_latents",
        "latents",
    }
)


def partition_flat_payload(
    payload: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Split a flattened per-request payload into inter-stage vs client mm dicts."""
    if not payload:
        return {}, {}
    inter_stage: dict[str, object] = {}
    client_mm: dict[str, object] = {}
    for key, value in payload.items():
        root = key.split(".", 1)[0]
        if root in _CLIENT_MM_ROOT_KEYS:
            client_mm[key] = value
        else:
            inter_stage[key] = value
    return inter_stage, client_mm


def partition_payload_list(
    payloads: list[dict[str, object]],
) -> tuple[list[dict[str, object] | None] | None, list[dict[str, object] | None] | None]:
    inter_stage_list: list[dict[str, object] | None] = []
    client_mm_list: list[dict[str, object] | None] = []
    for payload in payloads:
        inter_stage, client_mm = partition_flat_payload(payload)
        inter_stage_list.append(inter_stage or None)
        client_mm_list.append(client_mm or None)
    return (
        None if all(item is None for item in inter_stage_list) else inter_stage_list,
        None if all(item is None for item in client_mm_list) else client_mm_list,
    )


def build_mm_cpu(multimodal_outputs: dict) -> dict[str, object]:
    """Pre-copies multimodal tensor to CPU once (not per-request) to avoid
    redundant D2H transfers when gpu_resident_buffer_keys keeps them on GPU.

    In the case of prefix caching, the multimodal outputs provided will
    only contain the passthrough data.

    Args:
        multimodal_outputs: Multimodal dict mapping strings to objects.
    """
    if not multimodal_outputs:
        return {}

    # Pre-copy multimodal tensors to CPU once (not per-request) to avoid
    # redundant D2H transfers when gpu_resident_buffer_keys keeps them on GPU.
    mm_cpu: dict[str, object] = {}
    # Currently there are some cases where this is true at the
    # moment, which should be fixed.
    if not isinstance(multimodal_outputs, Mapping):
        logger.warning("Multimodal outputs are not a dict and will not be passed")

    for k, v in multimodal_outputs.items():
        cpu_v = _to_cpu(v)
        if cpu_v is not None:
            mm_cpu[k] = cpu_v
    return mm_cpu


def _to_cpu(value):
    """Recursively detach + move tensors to CPU; preserve dict/list nesting."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").contiguous()
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            cpu_v = _to_cpu(v)
            if cpu_v is not None:
                out[k] = cpu_v
        return out or None
    if isinstance(value, list):
        if not value:
            return value
        return [_to_cpu(v) for v in value]
    return value


def to_payload_element(
    element: object, idx: int, start: int, end: int, pass_lists_through: bool = False, seq_len: int | None = None
):
    """Build an mm payload element corresponding to one request index
    from an element containing 0 or more CPU tensors.

    Args:
        element: The object to be added to the payload.
        idx: The index of the request.
        start: The start index corresponding to the request idx.
        end: The end index corresponding to the request idx.
        pass_lists_through: bool Whether or not lists should be treated as
            passthrough data; this should be False in normal cases, but True
            if we need to avoid splitting nonempty lists prior to calling
            postprocess, which is the case for prefix cache.
        seq_len: Optional sequence length (i.e., dim 0 of hidden states).
            When set, a tensor whose first dimension equals seq_len is
            sliced per request. The prefix cache passthrough also passes
            the total scheduled token count here so 1D (seq_len,) metadata
            that is intentionally not cached is still split per request.
    """
    # Cached per-token tensors are merged elsewhere; here a first dim
    # equal to seq_len means a per-request slice is required.
    if seq_len is not None and isinstance(element, torch.Tensor) and element.shape[0] == seq_len:
        return element[start:end].contiguous()
    # Every other case is shared between prefix cache (passthrough data)
    # and running a model without prefix caching.
    elif isinstance(element, dict):
        return {
            sk: to_payload_element(sv, idx, start, end, pass_lists_through=pass_lists_through, seq_len=seq_len)
            for sk, sv in element.items()
        }
    elif isinstance(element, list):
        # For lists, clone tensors to avoid cross-request aliasing
        if pass_lists_through:
            return [elem.clone() if isinstance(elem, torch.Tensor) else elem for elem in element]
        element = element[idx] if idx < len(element) else element[0]
        if isinstance(element, torch.Tensor):
            element = element.clone()
        return element
    elif isinstance(element, torch.Tensor):
        # List-derived tensor payloads are request-invariant; clone to
        # avoid accidental cross-request aliasing on downstream mutation.
        return element.clone()
    return element
