# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .pipeline_cosmos3 import (
    Cosmos3OmniDiffusersPipeline,
    get_cosmos3_post_process_func,
    get_cosmos3_pre_process_func,
)
from .transformer_cosmos3 import Cosmos3VFMTransformer
from .transformer_cosmos3_edge import Cosmos3EdgeVFMTransformer

__all__ = [
    "Cosmos3OmniDiffusersPipeline",
    "Cosmos3MultiviewPipeline",
    "get_cosmos3_post_process_func",
    "get_cosmos3_pre_process_func",
    "Cosmos3VFMTransformer",
    "Cosmos3EdgeVFMTransformer",
    "Cosmos3MultiviewVFMTransformer",
]


def __getattr__(name: str):
    # Keep the optional FlexAttention variant lazy, matching the registry's
    # import behavior and avoiding extra startup work for regular Cosmos3.
    if name == "Cosmos3MultiviewPipeline":
        from .pipeline_cosmos3_multiview import Cosmos3MultiviewPipeline

        return Cosmos3MultiviewPipeline
    if name == "Cosmos3MultiviewVFMTransformer":
        from .transformer_cosmos3_multiview import Cosmos3MultiviewVFMTransformer

        return Cosmos3MultiviewVFMTransformer
    raise AttributeError(name)
