"""Backward-compat shims for the ``OmniEngineCore*`` -> ``StageLLMCore*`` rename.

These names/paths were part of the documented API (docs/api/README.md) and are
kept as deprecated aliases for at least one release so upgrades do not break at
import time.
"""

from __future__ import annotations

import warnings

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_engine_module_getattr_aliases_resolve_and_warn():
    import vllm_omni.engine as engine_pkg
    from vllm_omni.engine.stage.stage_core_types import (
        StageLLMCoreOutput,
        StageLLMCoreOutputs,
        StageLLMCoreRequest,
    )

    mapping = {
        "OmniEngineCoreRequest": StageLLMCoreRequest,
        "OmniEngineCoreOutput": StageLLMCoreOutput,
        "OmniEngineCoreOutputs": StageLLMCoreOutputs,
    }
    for old_name, new_cls in mapping.items():
        with pytest.warns(DeprecationWarning):
            obj = getattr(engine_pkg, old_name)
        assert obj is new_cls

    # Unknown attributes still raise AttributeError (not a silent alias).
    with pytest.raises(AttributeError):
        engine_pkg.DefinitelyNotAnAttribute  # noqa: B018


def test_stage_engine_core_client_shim_reexports_and_warns():
    import sys

    # Force a fresh import so the module-level DeprecationWarning fires here
    # regardless of whether another test imported the shim earlier.
    sys.modules.pop("vllm_omni.engine.stage_engine_core_client", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import vllm_omni.engine.stage_engine_core_client as shim

    from vllm_omni.engine.stage.stage_llm_core_client import StageLLMCoreClient

    assert shim.StageEngineCoreClient is StageLLMCoreClient
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_from_request_alias_delegates_and_warns():
    from vllm_omni.engine.stage.stage_core_types import StageLLMCoreRequest

    # ``from_request`` is the deprecated former name of ``from_vllm_request``.
    assert hasattr(StageLLMCoreRequest, "from_request")
    with pytest.warns(DeprecationWarning):
        # Delegation is what we assert; construction details belong to
        # from_vllm_request's own coverage, so a minimal stub is enough.
        with pytest.raises(Exception):  # noqa: B017 - any construction error is fine
            StageLLMCoreRequest.from_request(object())
