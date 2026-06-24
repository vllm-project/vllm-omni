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
from vllm.utils.import_utils import resolve_obj_by_qualname

from vllm_omni.diffusion.data import (
    OmniDiffusionConfig,
    _bde_routing_enabled,
    default_engine_backend_for_model,
    resolve_default_engine_backend,
)
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.experimental.bde.engine import (
    BDE_MODEL_RUNNER_CLS,
    BDEEngine,
    apply_bde_runner_default,
)


def _cfg(backend):
    return SimpleNamespace(engine_backend=backend)


def test_default_resolves_to_diffusion_engine():
    assert DiffusionEngine.resolve_engine_class(_cfg("default")) is DiffusionEngine


def test_bde_key_resolves_to_bde_engine():
    assert DiffusionEngine.resolve_engine_class(_cfg("bde")) is BDEEngine


def test_subclass_type_is_returned():
    assert DiffusionEngine.resolve_engine_class(_cfg(BDEEngine)) is BDEEngine


def test_qualname_string_resolves():
    cls = DiffusionEngine.resolve_engine_class(_cfg("vllm_omni.experimental.bde.engine.BDEEngine"))
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


# --- BDE routing is gated on BDE_KV_ENABLE (PR #4534 blast-radius review) ----


def test_routing_enabled_reads_env(monkeypatch):
    monkeypatch.setenv("BDE_KV_ENABLE", "1")
    assert _bde_routing_enabled() is True
    monkeypatch.delenv("BDE_KV_ENABLE", raising=False)
    assert _bde_routing_enabled() is False


def test_default_backend_applied_only_when_enabled(monkeypatch):
    # DreamZero + unset ("default") backend: auto-route applies only with BDE on.
    monkeypatch.setenv("BDE_KV_ENABLE", "1")
    assert resolve_default_engine_backend("DreamZeroPipeline", "default") == "bde"
    monkeypatch.delenv("BDE_KV_ENABLE", raising=False)
    assert resolve_default_engine_backend("DreamZeroPipeline", "default") is None


def test_explicit_backend_not_overridden_even_when_enabled(monkeypatch):
    # An explicit engine_backend bypasses the auto-route gate entirely.
    monkeypatch.setenv("BDE_KV_ENABLE", "1")
    assert resolve_default_engine_backend("DreamZeroPipeline", "bde") is None
    assert resolve_default_engine_backend("DreamZeroPipeline", "vllm_omni.experimental.bde.engine.BDEEngine") is None


def test_non_bde_model_never_routed(monkeypatch):
    monkeypatch.setenv("BDE_KV_ENABLE", "1")
    assert resolve_default_engine_backend("WanS2VPipeline", "default") is None


# --- BDE worker/runner wiring -----------------------------------------------


def test_bde_model_runner_cls_resolves():
    cls = resolve_obj_by_qualname(BDE_MODEL_RUNNER_CLS)
    assert cls.__name__ == "BDEModelRunner"


def test_apply_runner_default_sets_when_unset():
    cfg = SimpleNamespace(diffusion_model_runner_cls=None)
    apply_bde_runner_default(cfg)
    assert cfg.diffusion_model_runner_cls == BDE_MODEL_RUNNER_CLS


def test_apply_runner_default_respects_explicit_choice():
    cfg = SimpleNamespace(diffusion_model_runner_cls="my.custom.Runner")
    apply_bde_runner_default(cfg)
    assert cfg.diffusion_model_runner_cls == "my.custom.Runner"


def test_worker_runner_selection_prefers_override():
    # Mirrors the worker hook: an od_config override wins over the platform default.
    over = SimpleNamespace(diffusion_model_runner_cls=BDE_MODEL_RUNNER_CLS)
    assert (getattr(over, "diffusion_model_runner_cls", None) or "PLATFORM") == BDE_MODEL_RUNNER_CLS
    unset = SimpleNamespace(diffusion_model_runner_cls=None)
    assert (getattr(unset, "diffusion_model_runner_cls", None) or "PLATFORM") == "PLATFORM"
