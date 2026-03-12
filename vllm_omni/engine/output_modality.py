"""Output modality types for vLLM-Omni.

This module defines the OutputModality enum and TensorAccumulationStrategy
for type-safe multimodal output routing and tensor merging.

Part of RFC #1601: Decouple Multimodal Output Channel & Simplify Output Processor.
"""

from __future__ import annotations

import re
from enum import Enum, Flag, auto

_MODALITY_ALIASES: dict[str, str] = {
    "speech": "audio",
    "images": "image",
    "latents": "latent",
    "wav": "audio",
    "waveform": "audio",
    "pixel_values": "image",
    "pixels": "image",
}


class OutputModality(Flag):
    """Bit-flag enum for output modalities.

    Compose freely with ``|`` — no need to enumerate every combination.

    Single:   ``OutputModality.TEXT``, ``OutputModality.IMAGE``, ...
    Compound: ``OutputModality.TEXT | OutputModality.IMAGE``  (text+image)

    Note: POOLING is intentionally excluded. Pooling/embedding is vLLM's
    native path (``pooling_output → PoolingRequestOutput``), handled entirely
    by the base OutputProcessor. vLLM-Omni's layer does not participate.
    """

    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    LATENT = auto()

    @classmethod
    def from_string(cls, s: str | None) -> OutputModality:
        """Parse a free-text modality string into an ``OutputModality`` flag.

        Handles common aliases and compound strings separated by ``+`` or ``,``.

        Examples::

            OutputModality.from_string("text+image")
            # → OutputModality.TEXT | OutputModality.IMAGE

            OutputModality.from_string("speech")
            # → OutputModality.AUDIO

            OutputModality.from_string(None)
            # → OutputModality.TEXT

            OutputModality.from_string("")
            # → OutputModality.TEXT

        Args:
            s: Free-text modality string, or None.

        Returns:
            The corresponding ``OutputModality`` flag (possibly compound).

        Raises:
            ValueError: If any part of the string is not a recognized modality.
        """
        if not s or not s.strip():
            return cls.TEXT

        parts = [p.strip().lower() for p in re.split(r"[+,]", s.strip())]
        result = cls(0)
        for p in parts:
            p = _MODALITY_ALIASES.get(p, p)
            try:
                result |= cls[p.upper()]
            except KeyError:
                raise ValueError(f"Unknown modality: {p!r}. Supported: {[m.name.lower() for m in cls]}")
        return result

    @property
    def has_text(self) -> bool:
        """Return True if this modality includes text output."""
        return OutputModality.TEXT in self

    @property
    def has_multimodal(self) -> bool:
        """Return True if this modality includes any non-text output."""
        return bool(self & ~OutputModality.TEXT)


class TensorAccumulationStrategy(Enum):
    """Strategy for merging incremental multimodal tensors.

    Different modalities have different tensor shape semantics and
    require different merge strategies when accumulating across steps.
    """

    CONCAT_DIM0 = "concat_dim0"
    """Concatenate along dimension 0. Used for image/latent tensors."""

    CONCAT_LAST = "concat_last"
    """Concatenate along the last dimension. Used for audio waveforms."""

    APPEND_LIST = "append_list"
    """Append to a list (no tensor concatenation)."""

    REPLACE = "replace"
    """Replace previous tensor entirely with the latest one."""


def get_accumulation_strategy(modality: OutputModality) -> TensorAccumulationStrategy:
    """Determine tensor merge strategy from the multimodal flags.

    Uses Flag bit checks — no need to enumerate combinations.

    Args:
        modality: The output modality flag.

    Returns:
        The appropriate ``TensorAccumulationStrategy``.
    """
    if OutputModality.AUDIO in modality:
        return TensorAccumulationStrategy.CONCAT_LAST
    if OutputModality.IMAGE in modality or OutputModality.LATENT in modality:
        return TensorAccumulationStrategy.CONCAT_DIM0
    return TensorAccumulationStrategy.CONCAT_DIM0  # default
