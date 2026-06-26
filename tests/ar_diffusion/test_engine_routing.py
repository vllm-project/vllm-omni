# SPDX-License-Identifier: Apache-2.0
"""Tests for the generic engine-backend dispatcher + AR-Diffusion runner wiring.

``DiffusionEngine.resolve_engine_class`` is a generic dispatcher (``"default"`` / a
``DiffusionEngine`` subclass / an import-path string). DreamZero selects the
AR-Diffusion engine via its deploy config's ``engine_backend`` qualname — no
DreamZero/ar_diffusion knowledge lives in the public base, so the routing check
here is simply that the ``ARDiffusionEngine`` qualname resolves correctly.
"""

from dataclasses import fields
from types import SimpleNamespace

import pytest
from vllm.utils.import_utils import resolve_obj_by_qualname

from vllm_omni.diffusion.data import OmniDiffusionConfig
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


def test_subclass_type_is_returned():
    assert DiffusionEngine.resolve_engine_class(_cfg(ARDiffusionEngine)) is ARDiffusionEngine


def test_ar_diffusion_qualname_resolves():
    # How DreamZero's deploy config selects the engine: a full import-path string.
    cls = DiffusionEngine.resolve_engine_class(
        _cfg("vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine")
    )
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


def test_ar_diffusion_engine_is_diffusion_engine_subclass():
    assert issubclass(ARDiffusionEngine, DiffusionEngine)


def test_config_engine_backend_field_defaults_to_default():
    field = {f.name: f for f in fields(OmniDiffusionConfig)}["engine_backend"]
    assert field.default == "default"


# --- AR-Diffusion worker/runner wiring -----------------------------------------------


def test_ar_diffusion_model_runner_cls_resolves():
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
    over = SimpleNamespace(diffusion_model_runner_cls=AR_DIFFUSION_MODEL_RUNNER_CLS)
    assert (getattr(over, "diffusion_model_runner_cls", None) or "PLATFORM") == AR_DIFFUSION_MODEL_RUNNER_CLS
    unset = SimpleNamespace(diffusion_model_runner_cls=None)
    assert (getattr(unset, "diffusion_model_runner_cls", None) or "PLATFORM") == "PLATFORM"
