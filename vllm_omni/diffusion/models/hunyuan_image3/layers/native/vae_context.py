# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Per-call context for an already initialized VAE device group."""

from contextvars import ContextVar
from dataclasses import dataclass

import torch.distributed as dist


@dataclass(frozen=True)
class VaeSpatialContext:
    group: dist.ProcessGroup


VAE_SPATIAL_CONTEXT: ContextVar[VaeSpatialContext | None] = ContextVar("hunyuan_image3_vae_spatial", default=None)
