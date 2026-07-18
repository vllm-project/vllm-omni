"""Multimodal output data structures for vLLM-Omni.

This module defines structured types for multimodal outputs.

"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.outputs import CompletionOutput

from vllm_omni.outputs.output_modality import TensorAccumulationStrategy
from vllm_omni.outputs.utils import _is_tensor_list, _to_cpu

logger = init_logger(__name__)

# Keys whose values are metadata scalars (e.g. audio sample rate) but may
# arrive as 0-d torch.Tensors — from_dict routes all tensors into .tensors,
# so we relocate them to .metadata before consolidation to avoid a bogus
# torch.cat attempt and its warn-and-keep-last fallback.
_METADATA_TENSOR_KEYS: frozenset[str] = frozenset({"sr", "sample_rate", "audio_sample_rate"})


def _cat_tensors(
    tensors: list[torch.Tensor],
    strategy: TensorAccumulationStrategy,
) -> torch.Tensor:
    """Concatenate a list of tensors according to *strategy*."""
    if strategy == TensorAccumulationStrategy.CONCAT_LAST:
        return torch.cat(tensors, dim=-1)
    if strategy == TensorAccumulationStrategy.REPLACE:
        return tensors[-1]
    # CONCAT_DIM0 / APPEND_LIST / default
    return torch.cat(tensors, dim=0)


def _consolidate_tensor_list(
    key: str,
    tensor_list: list[torch.Tensor],
    strategy: TensorAccumulationStrategy,
) -> torch.Tensor:
    """Concatenate a deferred tensor list, with fallbacks on shape mismatch."""
    try:
        return _cat_tensors(tensor_list, strategy)
    except RuntimeError:
        # [TODO: this part is for async chunk, not for audio only. can we come up with a design
        # that identify this by whether using async chunk instead of modality?]
        if key != "audio":
            logger.warning("Error concatenating tensor for key %s; keeping last tensor", key)
            return tensor_list[-1]
        # Audio chunks may have mismatched shapes; retry along the last dim,
        # then fall back to flattening each chunk.
        try:
            return torch.cat(tensor_list, dim=-1)
        except RuntimeError:
            return torch.cat([chunk.reshape(-1) for chunk in tensor_list], dim=0)


def _append_entries(store: dict[str, Any], incoming: dict[str, Any]) -> None:
    """Append *incoming* entries into *store*.

    Tensor values are collected into lists (deferred concatenation, avoiding
    O(n²) repeated torch.cat); non-tensor values replace the existing entry.
    """
    for key, new_value in incoming.items():
        existing = store.get(key)
        if isinstance(existing, list) and isinstance(new_value, torch.Tensor):
            existing.append(new_value)
        elif isinstance(existing, torch.Tensor) and isinstance(new_value, torch.Tensor):
            store[key] = [existing, new_value]
        else:
            store[key] = new_value


@dataclass(eq=False)
class MultimodalPayload(Mapping):
    """Structured multimodal output payload.

    Implements ``collections.abc.Mapping`` so that ``isinstance(payload, dict)``
    style checks in downstream code can be replaced with duck-typing, and
    ``payload.get(key)``, ``payload[key]``, ``key in payload``, ``len(payload)``
    all work seamlessly for both tensors and metadata.

    Attributes:
        tensors: Dictionary mapping modality/key names to their tensors.
        metadata: Optional dictionary for non-tensor metadata
            (e.g., sample rate for audio, image dimensions).
    """

    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_tensor(self) -> torch.Tensor | None:
        """Return the first tensor in the payload, or None if empty."""
        if self.tensors:
            return next(iter(self.tensors.values()))
        return None

    @property
    def is_empty(self) -> bool:
        """Return True if the payload has no tensors and no metadata."""
        return len(self.tensors) == 0 and len(self.metadata) == 0

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key, searching tensors first then metadata."""
        if key in self.tensors:
            return self.tensors[key]
        return self.metadata.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.tensors or key in self.metadata

    def __getitem__(self, key: str) -> Any:
        """Dict-like indexing: search tensors first, then metadata."""
        if key in self.tensors:
            return self.tensors[key]
        if key in self.metadata:
            return self.metadata[key]
        raise KeyError(key)

    def __len__(self) -> int:
        return len(self.tensors) + len(self.metadata)

    def __iter__(self) -> Iterator[str]:
        yield from self.tensors
        yield from self.metadata

    def __bool__(self) -> bool:
        return bool(self.tensors) or bool(self.metadata)

    def __eq__(self, other: object) -> bool:
        """Support equality with plain dicts and other Mappings."""
        if isinstance(other, MultimodalPayload):
            return self.tensors == other.tensors and self.metadata == other.metadata
        if isinstance(other, Mapping):
            return dict(self) == dict(other)
        return NotImplemented

    def merged_with(self, incoming: MultimodalPayload) -> MultimodalPayload:
        """Merge *incoming* onto this payload and return the result.

        Tensor values accumulate into lists for deferred concatenation;
        non-tensor values are replaced with the latest. When this payload
        is empty, *incoming* is returned as-is, so callers should use the
        return value: ``accumulated = accumulated.merged_with(incoming)``.
        """
        if self.is_empty:
            return incoming
        _append_entries(self.tensors, incoming.tensors)
        _append_entries(self.metadata, incoming.metadata)
        return self

    def consolidate_tensors(self, strategy: TensorAccumulationStrategy) -> None:
        """Concatenate deferred tensor lists into single tensors.

        Tensors are generated content accumulated as chunks, so lists are
        concatenated according to *strategy* (e.g. audio chunks along the
        time dimension, latent frames along the batch dimension).
        """
        # Relocate scalar metadata tensors (e.g. sample rate) that from_dict
        # routed into .tensors, so they take the REPLACE path via .metadata
        # instead of failing torch.cat as 0-d tensors.
        for key in _METADATA_TENSOR_KEYS:
            value = self.tensors.pop(key, None)
            if value is None:
                continue
            self.metadata[key] = value[-1] if isinstance(value, list) else value

        for key, value in list(self.tensors.items()):
            if _is_tensor_list(value):
                self.tensors[key] = _consolidate_tensor_list(key, value, strategy)

    def consolidate_metadata(self) -> None:
        """Resolve deferred tensor lists in metadata by keeping the latest value.

        Metadata values are per-step snapshots (e.g. sample rate), not
        content deltas, so the latest value supersedes earlier ones.
        Nested dicts (unflattened payloads) are resolved one level down.
        """
        for key, value in list(self.metadata.items()):
            if _is_tensor_list(value):
                self.metadata[key] = value[-1]
            elif isinstance(value, dict):
                for sub_key, sub_value in list(value.items()):
                    if _is_tensor_list(sub_value):
                        value[sub_key] = sub_value[-1]

    def to_dict(self) -> dict[str, Any]:
        """Convert back to a plain dict (tensors + metadata merged)."""
        result: dict[str, Any] = dict(self.tensors)
        result.update(self.metadata)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> MultimodalPayload | None:
        """Create a MultimodalPayload from a raw dictionary.

        Separates torch.Tensor values into tensors and everything
        else into metadata.
        """
        if not data:
            return None
        tensors: dict[str, torch.Tensor] = {}
        metadata: dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                tensors[k] = v
            else:
                metadata[k] = v
        if not tensors and not metadata:
            return None
        return cls(tensors=tensors, metadata=metadata)

    @classmethod
    def from_raw(cls, payload: Any, modality_key: str) -> MultimodalPayload | None:
        """Create a MultimodalPayload from a raw producer payload.

        Accepts a MultimodalPayload (returned as-is), a dict, or a bare
        tensor (stored under *modality_key*). Tensors are moved to CPU.
        Producer-specific dict keys are remapped to the semantic modality
        key (e.g. "audio", "latent"): AR runners produce {"hidden": ...}
        and generation runners produce {"model_outputs": ...}.
        """
        if isinstance(payload, MultimodalPayload):
            return payload

        if not isinstance(payload, dict):
            return cls.from_dict({modality_key: _to_cpu(payload)})

        remapped: dict[str, Any] = {}
        for key, value in payload.items():
            is_producer_key = key == "model_outputs" or (key == "hidden" and modality_key != "hidden")
            remapped[modality_key if is_producer_key else key] = _to_cpu(value)
        return cls.from_dict(remapped)


@dataclass
class MultimodalCompletionOutput(CompletionOutput):
    """CompletionOutput with multimodal support.

    Inherits all CompletionOutput fields and adds multimodal_output.
    As a CompletionOutput subclass, compatible with all existing vLLM consumers.
    """

    def __init__(
        self,
        multimodal_output: MultimodalPayload | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.multimodal_output = multimodal_output

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base[:-1]}, multimodal_output={self.multimodal_output!r})"
