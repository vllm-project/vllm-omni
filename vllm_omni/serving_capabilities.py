# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-owned serving declarations, independent of engines and HTTP servers.

These describe implemented request contracts, not device capacity or health.
PipelineConfig and DiffusionModelMetadata own instances of these declarations.
The serving layer adds deployment routes and execution observations separately.
"""

from dataclasses import dataclass
from typing import Literal

OperationId = Literal["chat", "image_generation", "image_edit", "video_generation", "speech", "audio_generation"]
Support = Literal["supported", "unsupported", "unknown"]
ConstraintValue = str | int | float | bool | tuple[str, ...]

OPERATIONS: tuple[OperationId, ...] = (
    "chat",
    "image_generation",
    "image_edit",
    "video_generation",
    "speech",
    "audio_generation",
)


@dataclass(frozen=True)
class CapabilityProfile:
    """One request contract; required inputs are in addition to its protocol."""

    id: str
    operation: OperationId
    required_inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    support: Support = "supported"
    reason_code: str = "model_declared"
    constraints: tuple[tuple[str, ConstraintValue], ...] = ()


TEXT_RESPONSE = CapabilityProfile("text_response", "chat", ("text",), ("text",))
IMAGE_RESPONSE = CapabilityProfile("image_response", "chat", ("text",), ("image",))
AUDIO_RESPONSE = CapabilityProfile("audio_response", "chat", ("text",), ("audio",))
TEXT_TO_IMAGE = CapabilityProfile("text_to_image", "image_generation", ("text",), ("image",))
IMAGE_TO_IMAGE = CapabilityProfile("image_to_image", "image_edit", ("text", "image"), ("image",))
TEXT_TO_VIDEO = CapabilityProfile("text_to_video", "video_generation", ("text",), ("video",))
IMAGE_TO_VIDEO = CapabilityProfile("image_to_video", "video_generation", ("text", "image"), ("video",))
VIDEO_TO_VIDEO = CapabilityProfile("video_to_video", "video_generation", ("video",), ("video",))
TEXT_TO_SPEECH = CapabilityProfile("text_to_speech", "speech", ("text",), ("audio",))
LYRICS_TO_MUSIC = CapabilityProfile("lyrics_to_music", "speech", ("text", "instructions"), ("audio",))
CAPTION_TO_AUDIO = CapabilityProfile("caption_to_audio", "speech", ("text",), ("audio",))
TEXT_TO_AUDIO = CapabilityProfile("text_to_audio", "audio_generation", ("text",), ("audio",))

STANDARD_PROFILES = (
    TEXT_RESPONSE,
    IMAGE_RESPONSE,
    AUDIO_RESPONSE,
    TEXT_TO_IMAGE,
    IMAGE_TO_IMAGE,
    TEXT_TO_VIDEO,
    IMAGE_TO_VIDEO,
    VIDEO_TO_VIDEO,
    TEXT_TO_SPEECH,
    LYRICS_TO_MUSIC,
    CAPTION_TO_AUDIO,
    TEXT_TO_AUDIO,
)


@dataclass(frozen=True)
class OperationCapability:
    id: OperationId
    support: Support
    profiles: tuple[CapabilityProfile, ...] = ()
    reason_code: str = "model_declared"


@dataclass(frozen=True)
class ServingCapabilities:
    """A declaration for the complete selected model, including its topology.

    Missing operations/profiles are unknown. ``sources`` are repository-relative
    implementation or documentation references for maintainers, not response
    fields; local checkpoint paths never belong here.
    """

    operations: tuple[OperationCapability, ...] = ()
    sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        operation_ids = [operation.id for operation in self.operations]
        if len(set(operation_ids)) != len(operation_ids):
            raise ValueError("Serving operations must have unique IDs")
        for operation in self.operations:
            if operation.id not in OPERATIONS:
                raise ValueError(f"Unknown serving operation: {operation.id}")
            profile_ids = [profile.id for profile in operation.profiles]
            if len(set(profile_ids)) != len(profile_ids):
                raise ValueError(f"Duplicate profiles for {operation.id}")
            for profile in operation.profiles:
                if profile.operation != operation.id:
                    raise ValueError(f"Profile {profile.id} belongs to {profile.operation}, not {operation.id}")
                if operation.support == "unsupported" and profile.support == "supported":
                    raise ValueError(f"Unsupported operation {operation.id} has supported profile {profile.id}")

    def get(self, operation_id: OperationId) -> OperationCapability:
        for operation in self.operations:
            if operation.id == operation_id:
                return operation
        return OperationCapability(operation_id, "unknown", reason_code="declaration_missing")


def declare_capabilities(
    *profiles: CapabilityProfile,
    sources: tuple[str, ...],
    unknown_operations: tuple[OperationId, ...] = (),
    unknown_reason: str = "request_contract_unverified",
    unsupported_reason: str = "model_not_supported",
) -> ServingCapabilities:
    """Declare a *closed* set of the six standard request contracts.

    Callers explicitly assert that omitted standard profiles are unsupported.
    Use an unknown profile/operation for checkpoint-dependent or unverified
    contracts. Plugins with partial declarations should instantiate
    ServingCapabilities directly, where omission remains unknown.
    """
    supplied = {profile.id: profile for profile in profiles}
    if len(supplied) != len(profiles):
        raise ValueError("Serving profiles must have unique IDs")
    standard_ids = {profile.id for profile in STANDARD_PROFILES}
    if supplied.keys() - standard_ids:
        raise ValueError("Closed declarations require standard profile IDs")
    if set(unknown_operations) - set(OPERATIONS):
        raise ValueError("Unknown serving operation")
    operations = []
    for operation_id in OPERATIONS:
        operation_profiles = []
        for standard in STANDARD_PROFILES:
            if standard.operation != operation_id:
                continue
            profile = supplied.get(standard.id)
            if profile is None:
                unknown = operation_id in unknown_operations
                profile = CapabilityProfile(
                    id=standard.id,
                    operation=operation_id,
                    required_inputs=(),
                    outputs=(),
                    support="unknown" if unknown else "unsupported",
                    reason_code=unknown_reason if unknown else unsupported_reason,
                )
            operation_profiles.append(profile)
        states = {profile.support for profile in operation_profiles}
        support: Support = "supported" if "supported" in states else "unknown" if "unknown" in states else "unsupported"
        reason = (
            "model_declared"
            if support == "supported"
            else unknown_reason
            if support == "unknown"
            else unsupported_reason
        )
        operations.append(OperationCapability(operation_id, support, tuple(operation_profiles), reason))
    return ServingCapabilities(tuple(operations), sources)
