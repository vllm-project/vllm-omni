# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
vLLM-Omni entrypoints module.
"""


def __getattr__(name: str):
    # Lazy imports to avoid eagerly loading heavy modules (engine,
    # model_loader, pynvml) when the package is imported in lightweight
    # contexts such as model-architecture inspection subprocesses.
    if name == "AsyncOmni":
        from vllm_omni.entrypoints.async_omni import AsyncOmni

        return AsyncOmni
    if name == "Omni":
        from vllm_omni.entrypoints.omni import Omni

        return Omni
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AsyncOmni",
    "Omni",
]
