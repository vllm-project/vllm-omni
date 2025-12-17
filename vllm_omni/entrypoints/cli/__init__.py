"""CLI helpers for vLLM-Omni entrypoints."""

from .serve import OmniServeCommand
from vllm_omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand

__all__ = ["OmniServeCommand", "OmniBenchmarkServingSubcommand"]
