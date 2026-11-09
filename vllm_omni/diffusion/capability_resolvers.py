# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lightweight, model-owned declarations for checkpoint-dependent contracts.

These functions consume the resolved deployment fields already in memory.
Worker-only checkpoint facts are deliberately not reconstructed from model
names, paths, filesystem probes, or another Hugging Face configuration load.
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace

from vllm_omni.serving_capabilities import (
    IMAGE_RESPONSE,
    IMAGE_TO_VIDEO,
    TEXT_TO_IMAGE,
    TEXT_TO_VIDEO,
    VIDEO_TO_VIDEO,
    ServingCapabilities,
    declare_capabilities,
)

_H3_SOURCES = (
    "vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py",
    "vllm_omni/model_executor/models/minimax_h3/checkpoint.py",
)
MINIMAX_H3_CAPABILITIES = declare_capabilities(
    *(
        replace(
            profile, outputs=("video", "audio"), support="unknown", reason_code="worker_checkpoint_tasks_unavailable"
        )
        for profile in (TEXT_TO_VIDEO, IMAGE_TO_VIDEO, VIDEO_TO_VIDEO)
    ),
    sources=_H3_SOURCES,
    unknown_reason="worker_checkpoint_tasks_unavailable",
)
MINIMAX_H3_CAPABILITIES_RESOLVER = "vllm_omni.diffusion.capability_resolvers.resolve_minimax_h3_capabilities"

# Both distilled and non-distilled checkpoints implement T2I and I2V.
# T2V and V2V depend on the worker's scheduler/backbone configuration.
COSMOS3_CAPABILITIES = declare_capabilities(
    IMAGE_RESPONSE,
    TEXT_TO_IMAGE,
    replace(IMAGE_TO_VIDEO, constraints=(("optional_outputs", ("audio",)),)),
    *(
        replace(
            profile,
            support="unknown",
            reason_code="worker_scheduler_mode_unavailable",
            constraints=(("optional_outputs", ("audio",)),),
        )
        for profile in (TEXT_TO_VIDEO, VIDEO_TO_VIDEO)
    ),
    sources=("vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py",),
    unknown_reason="worker_scheduler_mode_unavailable",
)


def _field(value: object, name: str) -> object:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def resolve_minimax_h3_capabilities(declaration: ServingCapabilities, stages: Sequence[object]) -> ServingCapabilities:
    """Apply explicit partition exclusions; release task lists stay unknown.

    ``task_type`` chooses FL2VA/Ref2VA components. The worker subsequently
    reads the release's task list, which can further narrow those components.
    An explicit partition therefore proves exclusions, but not admission.
    """
    task_types = {
        str(task).lower() for stage in stages if (task := _field(_field(stage, "engine_args"), "task_type")) is not None
    }
    # Conflicting stage overrides cannot establish a single public contract.
    if len(task_types) != 1:
        return declaration
    task_type = task_types.pop()
    excluded = (
        {"text_to_video"}
        if task_type == "ref2va"
        else ({"video_to_video"} if task_type in {"t2va", "fl2va"} else set())
    )
    profiles = [
        profile
        for operation in declaration.operations
        for profile in operation.profiles
        if profile.support != "unsupported"
    ]
    return declare_capabilities(
        *(
            replace(profile, support="unsupported", reason_code="startup_partition_excludes_profile")
            if profile.id in excluded
            else profile
            for profile in profiles
        ),
        sources=declaration.sources,
        unknown_reason="worker_checkpoint_tasks_unavailable",
    )
