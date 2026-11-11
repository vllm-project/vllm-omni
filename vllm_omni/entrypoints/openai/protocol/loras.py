# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""OpenAI-compatible protocol models for runtime LoRA inventory queries."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LoRAAdapterInfo(BaseModel):
    """Identity of one adapter currently registered in a runtime cache."""

    id: int = Field(description="Runtime LoRA adapter identifier")
    name: str | None = Field(default=None, description="Adapter name when provided by the backend")


class LoRAReplicaInventory(BaseModel):
    """Runtime LoRA adapters owned by one physical stage replica."""

    replica_id: int = Field(ge=0)
    adapters: list[LoRAAdapterInfo] = Field(default_factory=list)


class LoRAStageInventory(BaseModel):
    """Runtime LoRA adapters grouped under one logical stage."""

    stage_id: int = Field(ge=0)
    model_stage: str | None = None
    replicas: list[LoRAReplicaInventory] = Field(default_factory=list)


class LoRAInventoryResponse(BaseModel):
    """Stable response envelope for ``GET /v1/omni/loras``."""

    object: Literal["omni.lora_inventory"] = "omni.lora_inventory"
    data: list[LoRAStageInventory] = Field(default_factory=list)


__all__ = [
    "LoRAAdapterInfo",
    "LoRAInventoryResponse",
    "LoRAReplicaInventory",
    "LoRAStageInventory",
]
