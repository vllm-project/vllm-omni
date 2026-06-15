# SPDX-License-Identifier: Apache-2.0
"""Routing tests for BDE engine selection (Phase 1, PR-1).

Covers ``DiffusionEngine.resolve_engine_class`` — the ``engine_backend``
config-field dispatcher that routes a request to ``BDEEngine`` vs the base
``DiffusionEngine``. Resolution is tested without constructing an engine
(construction runs a dummy forward that needs a real model).
"""

from dataclasses import fields
from types import SimpleNamespace

import pytest

from vllm_omni.bde.engine import BDEEngine
from vllm_omni.diffusion.data import (
    OmniDiffusionConfig,
    default_engine_backend_for_model,
)
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine


def _cfg(backend):
    return SimpleNamespace(engine_backend=backend)


def test_default_resolves_to_diffusion_engine():
    assert DiffusionEngine.resolve_engine_class(_cfg("default")) is DiffusionEngine


def test_missing_field_resolves_to_diffusion_engine():
    # Backward compatibility: configs predating the engine_backend field.
    assert DiffusionEngine.resolve_engine_class(SimpleNamespace()) is DiffusionEngine


def test_none_resolves_to_diffusion_engine():
    assert DiffusionEngine.resolve_engine_class(_cfg(None)) is DiffusionEngine


def test_bde_key_resolves_to_bde_engine():
    assert DiffusionEngine.resolve_engine_class(_cfg("bde")) is BDEEngine


def test_subclass_type_is_returned():
    assert DiffusionEngine.resolve_engine_class(_cfg(BDEEngine)) is BDEEngine


def test_qualname_string_resolves():
    cls = DiffusionEngine.resolve_engine_class(_cfg("vllm_omni.bde.engine.BDEEngine"))
    assert cls is BDEEngine


def test_non_engine_type_raises():
    with pytest.raises(TypeError):
        DiffusionEngine.resolve_engine_class(_cfg(dict))


def test_qualname_not_an_engine_raises():
    with pytest.raises(TypeError):
        DiffusionEngine.resolve_engine_class(_cfg("builtins.dict"))


def test_bad_qualname_raises():
    with pytest.raises(ValueError):
        DiffusionEngine.resolve_engine_class(_cfg("not.a.real.module.NoSuchEngine"))


def test_bde_engine_is_diffusion_engine_subclass():
    assert issubclass(BDEEngine, DiffusionEngine)


def test_config_engine_backend_field_defaults_to_default():
    field = {f.name: f for f in fields(OmniDiffusionConfig)}["engine_backend"]
    assert field.default == "default"


# --- DreamZero defaults to the BDE engine -----------------------------------


def test_dreamzero_pipeline_defaults_to_bde():
    assert default_engine_backend_for_model("DreamZeroPipeline") == "bde"


def test_unknown_model_defaults_to_none():
    assert default_engine_backend_for_model("WanS2VPipeline") is None


def test_none_model_defaults_to_none():
    assert default_engine_backend_for_model(None) is None


def test_dreamzero_default_routes_to_bde_engine():
    # End-to-end of the routing: DreamZero's per-model default backend resolves
    # to BDEEngine through the same dispatcher used by make_engine.
    backend = default_engine_backend_for_model("DreamZeroPipeline")
    assert DiffusionEngine.resolve_engine_class(_cfg(backend)) is BDEEngine
