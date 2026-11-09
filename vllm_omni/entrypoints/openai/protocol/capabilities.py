# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Read-only, versioned discovery for a running Omni deployment."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from vllm_omni.serving_capabilities import OPERATIONS, OperationId, Support

Exposure = Literal["enabled", "disabled", "unknown"]


class CapabilityRouteReference(BaseModel):
    transport: Literal["http", "websocket"] = "http"
    method: str
    path: str


class CapabilityRoute(CapabilityRouteReference):
    exposure: Exposure


class CapabilityExecution(BaseModel):
    """A frontend execution gate observation, never a worker health claim."""

    state: Literal["unpaused", "paused", "unknown"] = "unknown"
    scope: Literal["deployment", "worker"] = "worker"
    observed_at: datetime


class CapabilityInputProfile(BaseModel):
    id: str
    support: Support
    required_inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    routes: list[CapabilityRouteReference] = Field(default_factory=list)
    reason_code: str
    constraints: dict[str, str | int | float | bool | tuple[str, ...]] = Field(default_factory=dict)


class CapabilityOperation(BaseModel):
    support: Support
    exposure: Exposure
    reason_code: str
    profiles: list[CapabilityInputProfile] = Field(default_factory=list)
    execution: CapabilityExecution


class CapabilityModel(BaseModel):
    id: str
    stage_ids: list[str]
    operations: dict[OperationId, CapabilityOperation]

    @model_validator(mode="after")
    def complete_operations(self) -> "CapabilityModel":
        if set(self.operations) != set(OPERATIONS):
            raise ValueError("A capabilities response must describe all six operations")
        return self


class CapabilityStage(BaseModel):
    id: str
    type: str
    depends_on: list[str] = Field(default_factory=list)


class OmniCapabilitiesResponse(BaseModel):
    object: Literal["omni.capabilities"] = "omni.capabilities"
    schema_version: Literal[1] = 1
    observed_at: datetime
    models: list[CapabilityModel]
    stages: list[CapabilityStage]
    routes: list[CapabilityRoute]
