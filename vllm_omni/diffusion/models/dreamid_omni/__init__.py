# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

__all__ = [
    "DreamIDOmniPipeline",
]


def __getattr__(name: str):
    if name == "DreamIDOmniPipeline":
        from .pipeline_dreamid_omni import DreamIDOmniPipeline

        return DreamIDOmniPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
