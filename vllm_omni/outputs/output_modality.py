"""Output modality types for vLLM-Omni.

This module defines the OutputModality enum and TensorAccumulationStrategy
for type-safe multimodal output routing and tensor merging.

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
    "token_ids": "text",
    "tokens": "text",
}


class OutputModalityNames(str, Enum):
    """Keys for output modalities.

    TODO: (Alex) Integrate this with the big-flag enum below + throughout the code
    for better type safety (currently only used for output processor).
    """

    # Keep the string conversion provided by ``enum.StrEnum`` while using the
    # Python 3.10-compatible ``str, Enum`` form.
    __str__ = str.__str__

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    LATENT = "latent"


# Specify which output modalities may be drained when handling delta messages.
# For some types, e.g., latents, we need to be careful to ensure the full context
# is passed as the stream yields due to assumptions in the I/O processing and model
# when async chunk isn't enabled.
NON_DRAINABLE_MODALITIES = {OutputModalityNames.TEXT, OutputModalityNames.LATENT}
DRAINABLE_MODALITIES = {mod for mod in OutputModalityNames if mod not in NON_DRAINABLE_MODALITIES}


class OutputModality(Flag):
    """Bit-flag enum for output modalities.

    Compose freely with ``|`` — no need to enumerate every combination.

    Single:   ``OutputModality.TEXT``, ``OutputModality.IMAGE``, ...
    Compound: ``OutputModality.TEXT | OutputModality.IMAGE``  (text+image)

    Note: POOLING is intentionally excluded. Pooling/embedding is vLLM's
    native path (pooling_output → PoolingRequestOutput), handled entirely
    by the base OutputProcessor. vLLM-Omni's layer does not participate.
    """

    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    LATENT = auto()

    @classmethod
    def from_string(cls, s: str | None) -> OutputModality:
        """Parse a free-text modality string into an OutputModality flag.

        Handles common aliases and compound strings separated by + or ,.

        Examples::

            OutputModality.from_string("text+image")
            # → OutputModality.TEXT | OutputModality.IMAGE
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
        return OutputModality.TEXT in self

    @property
    def has_multimodal(self) -> bool:
        return bool(self & ~OutputModality.TEXT)


class TensorAccumulationStrategy(Enum):
    """Strategy for merging incremental multimodal tensors."""

    CONCAT_DIM0 = "concat_dim0"
    """Concatenate along dimension 0. Used for image/latent tensors."""

    CONCAT_LAST = "concat_last"
    """Concatenate along the last dimension. Used for audio waveforms."""

    APPEND_LIST = "append_list"
    """Append to a list (no tensor concatenation)."""

    REPLACE = "replace"
    """Replace previous tensor entirely with the latest one."""


# Per-tensor-key overrides for the modality-wide default below. A modality
# is a coarse hint (e.g. AUDIO tensors are usually waveform chunks that grow
# along the last dimension), but some models emit tensors under that same
# modality whose accumulation semantics differ -- e.g. discrete codec-frame
# matrices that grow along dim 0, or a constant reference/context tensor
# re-emitted unchanged at every step. Pipelines that produce such keys
# register the correct strategy here via ``register_key_accumulation_strategy``
# instead of forcing every tensor of the modality through one default.
_KEY_ACCUMULATION_OVERRIDES: dict[str, TensorAccumulationStrategy] = {}


def register_key_accumulation_strategy(key: str, strategy: TensorAccumulationStrategy) -> None:
    """Register a per-key override for ``get_accumulation_strategy``.

    Args:
        key: The flattened multimodal output key this override applies to,
            e.g. ``"codes.audio"`` as produced by
            ``vllm_omni.data_entry_keys.flatten_payload`` from a nested
            ``{"codes": {"audio": tensor}}`` payload.
        strategy: The accumulation strategy to use for this key, regardless
            of the modality its request is otherwise associated with.
    """
    _KEY_ACCUMULATION_OVERRIDES[key] = strategy


def get_accumulation_strategy(
    modality: OutputModality,
    key: str | None = None,
) -> TensorAccumulationStrategy:
    """Determine the tensor merge strategy for one output key.

    A registered per-key override (see ``register_key_accumulation_strategy``)
    always wins; otherwise the strategy falls back to the modality-wide
    default.
    """
    if key is not None and key in _KEY_ACCUMULATION_OVERRIDES:
        return _KEY_ACCUMULATION_OVERRIDES[key]
    if OutputModality.AUDIO in modality:
        return TensorAccumulationStrategy.CONCAT_LAST
    if OutputModality.IMAGE in modality or OutputModality.LATENT in modality:
        return TensorAccumulationStrategy.CONCAT_DIM0
    return TensorAccumulationStrategy.CONCAT_DIM0  # default
