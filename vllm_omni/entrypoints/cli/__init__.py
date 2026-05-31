"""CLI helpers for vLLM-Omni entrypoints.

Keep this package import lightweight. Multiprocessing ``spawn`` re-imports the
main module's package in stage subprocesses, and eager benchmark imports can
trigger CUDA/NVML probing before the stage has finished bootstrapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand

    from .serve import OmniServeCommand


__all__ = ["OmniServeCommand", "OmniBenchmarkServingSubcommand"]


def __getattr__(name: str):
    if name == "OmniServeCommand":
        from .serve import OmniServeCommand

        return OmniServeCommand
    if name == "OmniBenchmarkServingSubcommand":
        from vllm_omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand

        return OmniBenchmarkServingSubcommand
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
