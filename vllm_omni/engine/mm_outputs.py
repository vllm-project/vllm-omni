"""Multimodal output data structures for vLLM-Omni.

This module defines structured types for multimodal outputs.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.outputs import CompletionOutput


@dataclass
class MultimodalPayload:
    """Structured multimodal output payload.

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
        """Return True if the payload has no tensors."""
        return len(self.tensors) == 0

    def get(self, key: str) -> torch.Tensor | None:
        """Get a tensor by key, returning None if not found."""
        return self.tensors.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self.tensors

    def __len__(self) -> int:
        return len(self.tensors)

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
