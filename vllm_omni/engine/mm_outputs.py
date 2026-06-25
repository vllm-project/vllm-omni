"""Multimodal output data structures for vLLM-Omni.

This module defines structured types for multimodal outputs.

"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.outputs import CompletionOutput


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
