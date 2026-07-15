# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
import PIL.Image
import torch

# Final user-facing output modality returned by pipeline stages.
FinalOutputModalityType: TypeAlias = Literal["text", "image", "audio", "video"]

# Scalar values that may appear in either payload or metadata.
# Examples: text_output="done", fps=16, action_only_output=True.
DiffusionScalar: TypeAlias = str | int | float | bool | None

# Shape-like metadata. Examples: [1, 16, 3, 480, 832], (16, 480, 832).
DiffusionShape: TypeAlias = list[int] | tuple[int, ...]

# Named tensor controls used by transfer-style metadata.
# Example: {"pose": pose_tensor, "depth": depth_tensor}.
DiffusionTensorMap: TypeAlias = Mapping[str, torch.Tensor]

# Values allowed under metadata groups such as video/audio/actions/transfer.
# Examples: {"fps": 16}, {"shape": [1, 16, 3, 480, 832]},
# {"controls": {"pose": pose_tensor}}, {"hints": ["pose"]}.
DiffusionMetadataValue: TypeAlias = (
    DiffusionScalar
    | DiffusionShape
    | DiffusionTensorMap
    | torch.Tensor
    | list["DiffusionMetadataValue"]
    | tuple["DiffusionMetadataValue", ...]
    | Mapping[str, "DiffusionMetadataValue"]
)

# Mutable metadata envelope side. Example:
# {"video": {"fps": 16}, "actions": {"raw_action_dim": 14}}.
DiffusionMetadata: TypeAlias = dict[str, DiffusionMetadataValue]

# Read-only metadata view for validators and serving readers.
DiffusionMetadataMapping: TypeAlias = Mapping[str, DiffusionMetadataValue]


class DiffusionTrajectoryPayload(TypedDict, total=False):
    # Optional denoising trajectory payload for RL/debug consumers.
    latents: torch.Tensor | dict[str, Any] | None
    timesteps: torch.Tensor | dict[str, Any] | None
    log_probs: torch.Tensor | dict[str, Any] | None
    decoded: list[PIL.Image.Image] | None


# Concrete payload values produced by post_process.
# Examples: PIL image, video tensor, audio bytes/tensor, action tensor.
DiffusionPayloadValue: TypeAlias = (
    DiffusionScalar
    | bytes
    | torch.Tensor
    | np.ndarray
    | PIL.Image.Image
    | DiffusionTrajectoryPayload
    | list["DiffusionPayloadValue"]
    | tuple["DiffusionPayloadValue", ...]
    | Mapping[str, "DiffusionPayloadValue"]
)

# Mutable payload side. Example: {"video": video_tensor, "action": action_tensor}.
DiffusionPayload: TypeAlias = dict[str, DiffusionPayloadValue]

# Read-only payload view for code that consumes but does not mutate payload.
DiffusionPayloadMapping: TypeAlias = Mapping[str, DiffusionPayloadValue]

# Serving-facing multimodal output. Non-metadata keys hold payload values.
# Example: {"video": video_tensor, "metadata": {"video": {"fps": 16}}}.
DiffusionMultimodalOutput: TypeAlias = dict[str, DiffusionPayloadValue | DiffusionMetadata]


class DiffusionOutputEnvelope(TypedDict, total=False):
    # Canonical post_process return shape:
    # {"payload": {"video": video_tensor}, "metadata": {"video": {"fps": 16}}}
    payload: DiffusionPayload
    metadata: DiffusionMetadata


# Raw post_process return before formatter normalization. This keeps backward
# compatibility with direct payload returns while accepting the new envelope.
# Examples: image, {"video": video_tensor}, or
# {"payload": {"video": video_tensor}, "metadata": {"video": {"fps": 16}}}.
DiffusionPostprocessRawOutput: TypeAlias = DiffusionPayloadValue | DiffusionPayload | DiffusionOutputEnvelope


def strip_internal_metadata(metadata: DiffusionMetadataMapping) -> DiffusionMetadata:
    """Return public metadata, dropping model-private postprocess state."""
    return {key: value for key, value in metadata.items() if key != "internal"}


def validate_public_diffusion_metadata(metadata: DiffusionMetadataMapping) -> None:
    """Validate metadata that is safe to expose outside model postprocess."""
    if "internal" in metadata:
        raise ValueError("Diffusion metadata field 'internal' must not escape public formatting.")
    validate_diffusion_metadata(metadata)


def validate_diffusion_metadata(metadata: DiffusionMetadataMapping) -> None:
    """Validate canonical diffusion output metadata fields.

    Unknown model-specific groups and keys are intentionally tolerated. This
    validator only guards fields that are part of the shared output protocol or
    consumed by serving code.
    """
    _validate_video_metadata(_optional_mapping(metadata, "video"))
    _validate_audio_metadata(_optional_mapping(metadata, "audio"))
    _validate_actions_metadata(_optional_mapping(metadata, "actions"))
    _validate_transfer_metadata(_optional_mapping(metadata, "transfer"))
    _validate_text_metadata(_optional_mapping(metadata, "text"))
    _validate_common_metadata(_optional_mapping(metadata, "common"))


def _validate_video_metadata(metadata: DiffusionMetadataMapping | None) -> None:
    if metadata is None:
        return
    _check_optional_positive_number(metadata, "fps", group="video")
    _check_optional_int(metadata, "video_fps_multiplier", group="video", min_value=1)
    _check_optional_shape(metadata, "shape", group="video")


def _validate_audio_metadata(metadata: DiffusionMetadataMapping | None) -> None:
    if metadata is None:
        return
    _check_optional_int(metadata, "sample_rate", group="audio", min_value=1)
    _check_optional_non_empty_str(metadata, "audiox_task", group="audio")


def _validate_actions_metadata(metadata: DiffusionMetadataMapping | None) -> None:
    if metadata is None:
        return
    _check_optional_int(metadata, "raw_action_dim", group="actions", min_value=1)
    _check_optional_non_empty_str(metadata, "action_mode", group="actions")
    _check_optional_int(metadata, "domain_id", group="actions")


def _validate_transfer_metadata(metadata: DiffusionMetadataMapping | None) -> None:
    if metadata is None:
        return
    controls = metadata.get("controls")
    if controls is not None and not _is_tensor_mapping(controls):
        raise TypeError("Diffusion metadata field 'transfer.controls' must be a mapping of str to torch.Tensor.")
    hints = metadata.get("hints")
    if hints is not None and not _is_str_list(hints):
        raise TypeError("Diffusion metadata field 'transfer.hints' must be a list of strings.")


def _validate_text_metadata(metadata: DiffusionMetadataMapping | None) -> None:
    if metadata is None:
        return
    _check_optional_str(metadata, "text_output", group="text")
    _check_optional_str(metadata, "think_text", group="text")


def _validate_common_metadata(metadata: DiffusionMetadataMapping | None) -> None:
    if metadata is None:
        return
    value = metadata.get("action_only_output")
    if value is not None and not isinstance(value, bool):
        raise TypeError("Diffusion metadata field 'common.action_only_output' must be bool.")


def _optional_mapping(metadata: DiffusionMetadataMapping, group: str) -> DiffusionMetadataMapping | None:
    value = metadata.get(group)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"Diffusion metadata group '{group}' must be a mapping.")
    return value


def _check_optional_positive_number(metadata: DiffusionMetadataMapping, key: str, *, group: str) -> None:
    value = metadata.get(key)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Diffusion metadata field '{group}.{key}' must be a positive number.")
    if value <= 0:
        raise ValueError(f"Diffusion metadata field '{group}.{key}' must be > 0.")


def _check_optional_int(
    metadata: DiffusionMetadataMapping,
    key: str,
    *,
    group: str,
    min_value: int | None = None,
) -> None:
    value = metadata.get(key)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Diffusion metadata field '{group}.{key}' must be int.")
    if min_value is not None and value < min_value:
        raise ValueError(f"Diffusion metadata field '{group}.{key}' must be >= {min_value}.")


def _check_optional_str(metadata: DiffusionMetadataMapping, key: str, *, group: str) -> None:
    value = metadata.get(key)
    if value is not None and not isinstance(value, str):
        raise TypeError(f"Diffusion metadata field '{group}.{key}' must be str.")


def _check_optional_non_empty_str(metadata: DiffusionMetadataMapping, key: str, *, group: str) -> None:
    value = metadata.get(key)
    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"Diffusion metadata field '{group}.{key}' must be str.")
    if not value:
        raise ValueError(f"Diffusion metadata field '{group}.{key}' must be non-empty.")


def _check_optional_shape(metadata: DiffusionMetadataMapping, key: str, *, group: str) -> None:
    value = metadata.get(key)
    if value is None:
        return
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"Diffusion metadata field '{group}.{key}' must be a list or tuple of positive ints.")
    if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in value):
        raise ValueError(f"Diffusion metadata field '{group}.{key}' must contain positive ints.")


def _is_tensor_mapping(value: object) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(key, str) and isinstance(tensor, torch.Tensor) for key, tensor in value.items()
    )


def _is_str_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)
