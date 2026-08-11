"""Deprecated compatibility shim for ``StageEngineCoreClient``.

The class was renamed to
:class:`vllm_omni.engine.stage.stage_llm_core_client.StageLLMCoreClient` and moved
into the ``vllm_omni.engine.stage`` subpackage. Because the old module path and
name were part of the documented API (docs/api/README.md), this shim re-exports
the new class under the old name for at least one release to avoid breaking
imports on upgrade. Remove no earlier than the next release.
"""

from __future__ import annotations

import warnings

from vllm_omni.engine.stage.stage_llm_core_client import (
    StageLLMCoreClient as StageEngineCoreClient,
)

warnings.warn(
    "vllm_omni.engine.stage_engine_core_client.StageEngineCoreClient is deprecated "
    "and will be removed in a future release; use "
    "vllm_omni.engine.stage.stage_llm_core_client.StageLLMCoreClient instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["StageEngineCoreClient"]
