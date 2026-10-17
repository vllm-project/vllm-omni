# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Stage input processors: per-model builders for inter-stage transitions.

Naming convention: ``*_full_payload``/``*_async_chunk`` are producer-side
(worker) builders (``pooling_output``/``multimodal_output`` load-bearing);
``*_token_only`` and no-suffix names are consumer-side (orchestrator) builders,
normalized through ``wrap_orchestrator_processor``.

This package re-exports only the dispatch contract and registry — importing it
never pulls in model-specific code.
"""

from vllm_omni.model_executor.stage_input_processors._dispatch import (
    DiffusionInputBuilder,
    OrchestratorInputContext,
    PlaceholderPromptBuilder,
    invoke_orchestrator_processor,
    wrap_orchestrator_processor,
)
from vllm_omni.model_executor.stage_input_processors._registry import (
    ProcessorKind,
    ProcessorSpec,
    ProcessorValidationError,
    dead_processor_hint,
    infer_kind,
    register_processor,
    resolve_processor,
    validate_processor,
)

__all__ = [
    "OrchestratorInputContext",
    "PlaceholderPromptBuilder",
    "DiffusionInputBuilder",
    "invoke_orchestrator_processor",
    "wrap_orchestrator_processor",
    "ProcessorKind",
    "ProcessorSpec",
    "ProcessorValidationError",
    "resolve_processor",
    "register_processor",
    "infer_kind",
    "validate_processor",
    "dead_processor_hint",
]
