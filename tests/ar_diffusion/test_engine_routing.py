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
)


def _cfg(backend):
    return SimpleNamespace(engine_backend=backend)


def test_default_resolves_to_diffusion_engine():
    assert DiffusionEngine.resolve_engine_class(_cfg("default")) is DiffusionEngine


def test_subclass_type_is_returned():
    assert DiffusionEngine.resolve_engine_class(_cfg(ARDiffusionEngine)) is ARDiffusionEngine


def test_ar_diffusion_qualname_resolves():
    # How DreamZero's deploy config selects the engine: a full import-path string.
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


def test_ar_diffusion_engine_is_diffusion_engine_subclass():
    assert issubclass(ARDiffusionEngine, DiffusionEngine)


def test_config_engine_backend_field_defaults_to_default():
    field = {f.name: f for f in fields(OmniDiffusionConfig)}["engine_backend"]
    assert field.default == "default"


# --- AR-Diffusion worker/runner wiring -----------------------------------------------


def test_ar_diffusion_model_runner_cls_resolves():
    cls = resolve_obj_by_qualname(AR_DIFFUSION_MODEL_RUNNER_CLS)
    assert cls.__name__ == "ARDiffusionModelRunner"


def test_engine_class_declares_its_runner():
    """Routing lives on the engine class (review: hsliuustc0106) — engines
    never mutate od_config; the worker resolves the runner off the class."""
    assert ARDiffusionEngine.default_diffusion_model_runner_cls == AR_DIFFUSION_MODEL_RUNNER_CLS
    assert DiffusionEngine.default_diffusion_model_runner_cls is None


def _select_runner(od_config, platform_default="PLATFORM"):
    """Mirror of the worker's runner-selection chain (override > engine > platform)."""
    runner_override = getattr(od_config, "diffusion_model_runner_cls", None)
    if isinstance(runner_override, str) and runner_override:
        return runner_override
    engine_cls = DiffusionEngine.resolve_engine_class(od_config)
    engine_runner = getattr(engine_cls, "default_diffusion_model_runner_cls", None)
    if isinstance(engine_runner, str) and engine_runner:
        return engine_runner
    return platform_default


def test_runner_selection_chain():
    ar = "vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine"
    # engine-declared runner wins over platform default
    assert (
        _select_runner(SimpleNamespace(diffusion_model_runner_cls=None, engine_backend=ar))
        == AR_DIFFUSION_MODEL_RUNNER_CLS
    )
    # explicit override wins over the engine's declaration
    assert (
        _select_runner(SimpleNamespace(diffusion_model_runner_cls="my.custom.Runner", engine_backend=ar))
        == "my.custom.Runner"
    )
    # default engine -> platform default
    assert _select_runner(SimpleNamespace(diffusion_model_runner_cls=None, engine_backend="default")) == "PLATFORM"


def test_dreamzero_pipeline_rejects_non_ar_diffusion_engine():
    """Review (hsliuustc0106): a stale config with engine_backend='default'
    must fail at init, not mid-forward on the first KV access."""
    from types import SimpleNamespace

    import pytest

    from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline

    with pytest.raises(ValueError, match="requires the AR-Diffusion engine"):
        DreamZeroPipeline(od_config=SimpleNamespace(engine_backend="default"))
    with pytest.raises(ValueError, match="requires the AR-Diffusion engine"):
        DreamZeroPipeline(od_config=SimpleNamespace())


def test_runner_rejects_multi_sequence_config():
    """Review (hsliuustc0106): the paged path writes batch index 0 only, so
    max_num_seqs > 1 must be rejected at init."""
    from types import SimpleNamespace

    import pytest

    from vllm_omni.experimental.ar_diffusion.runner import ARDiffusionModelRunner

    fake = SimpleNamespace(od_config=SimpleNamespace(max_num_seqs=4))
    with pytest.raises(ValueError, match="max_num_seqs=1"):
        ARDiffusionModelRunner._preallocate_kv_cache(fake)


def test_execute_model_accepts_kv_prefetch_job_kwarg():
    """Regression: the worker passes ``kv_prefetch_job`` to every runner
    (added with the KV prefetch path, #4448); the AR-Diffusion override must
    accept and forward it or DreamZero crashes on the warm-up dummy run."""
    import inspect

    from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
    from vllm_omni.experimental.ar_diffusion.runner import ARDiffusionModelRunner

    for cls in (DiffusionModelRunner, ARDiffusionModelRunner):
        params = inspect.signature(cls.execute_model).parameters
        assert "kv_prefetch_job" in params, cls
