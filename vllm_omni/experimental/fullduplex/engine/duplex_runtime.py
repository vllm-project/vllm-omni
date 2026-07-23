# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental duplex runtime loading and compatibility exports."""

from __future__ import annotations

from importlib import import_module

from vllm_omni.experimental.fullduplex.engine.contracts import (
    DuplexAppendPlan,
    DuplexInputMode,
    DuplexOutputAction,
    DuplexOutputDecision,
    DuplexRuntimeCapabilities,
    DuplexRuntimeExtension,
    SessionMode,
    duplex_data_plane_request_info,
    duplex_resource_request_belongs_to_session,
    duplex_resource_request_id,
)


def load_duplex_runtime_extension(path: str | None) -> DuplexRuntimeExtension | None:
    if not path:
        return None
    module_name, separator, attribute_name = path.rpartition(".")
    if not separator:
        raise ValueError(f"Invalid duplex runtime extension path: {path!r}")
    extension_type = getattr(import_module(module_name), attribute_name)
    return validate_duplex_runtime_extension(extension_type())


def validate_duplex_runtime_extension(
    extension: object,
    *,
    sampling_defaults: tuple[object, ...] | None = None,
) -> DuplexRuntimeExtension:
    required_methods = (
        "configure_sampling_params",
        "plan_append",
        "decide_output",
    )
    missing = [name for name in required_methods if not callable(getattr(extension, name, None))]
    if missing:
        raise TypeError(f"Duplex runtime extension is missing callable method(s): {', '.join(missing)}")
    typed_extension = extension  # type: ignore[assignment]
    if sampling_defaults is not None:
        configured = typed_extension.configure_sampling_params(
            runtime_config={},
            defaults=sampling_defaults,
        )
        if not isinstance(configured, tuple):
            raise TypeError("Duplex runtime extension must return sampling parameters as a tuple")
        if len(configured) != len(sampling_defaults):
            raise ValueError("Duplex runtime extension must return one sampling parameter per stage")
        for stage_id, (value, default) in enumerate(zip(configured, sampling_defaults, strict=True)):
            if default is not None and not isinstance(value, type(default)):
                raise TypeError(
                    "Duplex runtime extension sampling parameter type mismatch "
                    f"for stage {stage_id}: expected {type(default).__name__}, got {type(value).__name__}"
                )
    return typed_extension


from vllm_omni.experimental.fullduplex.engine.duplex_session import (  # noqa: E402, F401
    DuplexAppendReservation,
    DuplexCompletedAppend,
    DuplexFenceMismatchError,
    DuplexInputAppend,
    DuplexRequestResource,
    DuplexSessionRuntimeManager,
    DuplexSessionRuntimeState,
    DuplexStageBinding,
)
from vllm_omni.experimental.fullduplex.engine.lease import (  # noqa: E402, F401
    DuplexLeaseActivity,
    DuplexLeaseConfig,
    DuplexLeaseState,
    DuplexSessionExpiry,
)

__all__ = [
    "DuplexAppendPlan",
    "DuplexAppendReservation",
    "DuplexCompletedAppend",
    "DuplexFenceMismatchError",
    "DuplexInputAppend",
    "DuplexInputMode",
    "DuplexLeaseActivity",
    "DuplexLeaseConfig",
    "DuplexLeaseState",
    "DuplexOutputAction",
    "DuplexOutputDecision",
    "DuplexRequestResource",
    "DuplexRuntimeCapabilities",
    "DuplexRuntimeExtension",
    "DuplexSessionExpiry",
    "DuplexSessionRuntimeManager",
    "DuplexSessionRuntimeState",
    "DuplexStageBinding",
    "SessionMode",
    "duplex_data_plane_request_info",
    "duplex_resource_request_belongs_to_session",
    "duplex_resource_request_id",
    "load_duplex_runtime_extension",
    "validate_duplex_runtime_extension",
]
