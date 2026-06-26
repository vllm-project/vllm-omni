# SPDX-License-Identifier: Apache-2.0
"""Routing tests for AR-Diffusion engine selection (Phase 1, PR-1).

Covers ``DiffusionEngine.resolve_engine_class`` — the ``engine_backend``
config-field dispatcher that routes a request to ``ARDiffusionEngine`` vs the base
``DiffusionEngine``. Resolution is tested without constructing an engine
(construction runs a dummy forward that needs a real model).
"""

from dataclasses import fields
from types import SimpleNamespace

import pytest
from vllm.utils.import_utils import resolve_obj_by_qualname

from vllm_omni.diffusion.data import (
    OmniDiffusionConfig,
    default_engine_backend_for_model,
    resolve_default_engine_backend,
)
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.experimental.ar_diffusion.engine import (
    AR_DIFFUSION_MODEL_RUNNER_CLS,
    ARDiffusionEngine,
    apply_ar_diffusion_runner_default,
)


def _cfg(backend):
    return SimpleNamespace(engine_backend=backend)


def test_default_resolves_to_diffusion_engine():
    assert DiffusionEngine.resolve_engine_class(_cfg("default")) is DiffusionEngine


def test_bde_key_resolves_to_bde_engine():
    assert DiffusionEngine.resolve_engine_class(_cfg("ar_diffusion")) is ARDiffusionEngine


def test_subclass_type_is_returned():
    assert DiffusionEngine.resolve_engine_class(_cfg(ARDiffusionEngine)) is ARDiffusionEngine


def test_qualname_string_resolves():
    cls = DiffusionEngine.resolve_engine_class(_cfg("vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine"))
    assert cls is ARDiffusionEngine


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
    assert issubclass(ARDiffusionEngine, DiffusionEngine)


def test_config_engine_backend_field_defaults_to_default():
    field = {f.name: f for f in fields(OmniDiffusionConfig)}["engine_backend"]
    assert field.default == "default"


# --- DreamZero defaults to the AR-Diffusion engine -----------------------------------


def test_dreamzero_pipeline_defaults_to_bde():
    assert default_engine_backend_for_model("DreamZeroPipeline") == "ar_diffusion"


def test_unknown_model_defaults_to_none():
    assert default_engine_backend_for_model("WanS2VPipeline") is None


def test_none_model_defaults_to_none():
    assert default_engine_backend_for_model(None) is None


def test_dreamzero_default_routes_to_bde_engine():
    # End-to-end of the routing: DreamZero's per-model default backend resolves
    # to ARDiffusionEngine through the same dispatcher used by make_engine.
    backend = default_engine_backend_for_model("DreamZeroPipeline")
    assert DiffusionEngine.resolve_engine_class(_cfg(backend)) is ARDiffusionEngine


# --- AR-Diffusion routing: unconditional per-model default; explicit arg overrides ---


def test_dreamzero_default_applied_unconditionally():
    # DreamZero + unset ("default") backend -> auto-route to ar_diffusion (no env gate).
    assert resolve_default_engine_backend("DreamZeroPipeline", "default") == "ar_diffusion"


def test_explicit_backend_not_overridden():
    # An explicit engine_backend command arg bypasses the per-model default.
    assert resolve_default_engine_backend("DreamZeroPipeline", "ar_diffusion") is None
    assert (
        resolve_default_engine_backend(
            "DreamZeroPipeline", "vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine"
        )
        is None
    )


def test_non_ar_diffusion_model_not_routed():
    assert resolve_default_engine_backend("WanS2VPipeline", "default") is None


# --- AR-Diffusion worker/runner wiring -----------------------------------------------


def test_bde_model_runner_cls_resolves():
    cls = resolve_obj_by_qualname(AR_DIFFUSION_MODEL_RUNNER_CLS)
    assert cls.__name__ == "ARDiffusionModelRunner"


def test_apply_runner_default_sets_when_unset():
    cfg = SimpleNamespace(diffusion_model_runner_cls=None)
    apply_ar_diffusion_runner_default(cfg)
    assert cfg.diffusion_model_runner_cls == AR_DIFFUSION_MODEL_RUNNER_CLS


def test_apply_runner_default_respects_explicit_choice():
    cfg = SimpleNamespace(diffusion_model_runner_cls="my.custom.Runner")
    apply_ar_diffusion_runner_default(cfg)
    assert cfg.diffusion_model_runner_cls == "my.custom.Runner"


def test_worker_runner_selection_prefers_override():
    # Mirrors the worker hook: an od_config override wins over the platform default.
    over = SimpleNamespace(diffusion_model_runner_cls=AR_DIFFUSION_MODEL_RUNNER_CLS)
    assert (getattr(over, "diffusion_model_runner_cls", None) or "PLATFORM") == AR_DIFFUSION_MODEL_RUNNER_CLS
    unset = SimpleNamespace(diffusion_model_runner_cls=None)
    assert (getattr(unset, "diffusion_model_runner_cls", None) or "PLATFORM") == "PLATFORM"
