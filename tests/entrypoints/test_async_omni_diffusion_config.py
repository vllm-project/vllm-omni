# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.entrypoints import omni as omni_module
from vllm_omni.entrypoints.async_omni import AsyncOmni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_default_stage_config_includes_cache_backend(monkeypatch):
    """Ensure cache_backend/cache_config are preserved in default diffusion stage."""
    monkeypatch.setattr(omni_module, "load_stage_configs_from_model", lambda model, base_engine_args=None: [])
    monkeypatch.setattr(omni_module, "resolve_model_config_path", lambda model: None)
    monkeypatch.setattr(AsyncOmni, "_start_stages", lambda self, model: None)
    monkeypatch.setattr(AsyncOmni, "_wait_for_stages_ready", lambda self, timeout=0: None)

    omni = AsyncOmni(
        model="dummy-model",
        cache_backend="cache_dit",
        cache_config='{"Fn_compute_blocks": 2}',
        vae_use_slicing=True,
        ulysses_degree=2,
    )

    stage_cfg = omni.stage_configs[0]
    engine_args = stage_cfg.engine_args

    assert engine_args.get("cache_backend") == "cache_dit"
    cache_config = engine_args.get("cache_config")
    assert cache_config["Fn_compute_blocks"] == 2
    assert engine_args.get("vae_use_slicing") is True
    parallel_config = engine_args.get("parallel_config")
    if hasattr(parallel_config, "get"):
        ulysses_degree = parallel_config.get("ulysses_degree")
    else:
        ulysses_degree = getattr(parallel_config, "ulysses_degree", None)
    assert ulysses_degree == 2


def test_default_cache_config_used_when_missing(monkeypatch):
    """Ensure default cache_config is applied when cache_backend is set."""
    monkeypatch.setattr(omni_module, "load_stage_configs_from_model", lambda model, base_engine_args=None: [])
    monkeypatch.setattr(omni_module, "resolve_model_config_path", lambda model: None)
    monkeypatch.setattr(AsyncOmni, "_start_stages", lambda self, model: None)
    monkeypatch.setattr(AsyncOmni, "_wait_for_stages_ready", lambda self, timeout=0: None)

    omni = AsyncOmni(
        model="dummy-model",
        cache_backend="cache_dit",
    )

    engine_args = omni.stage_configs[0].engine_args
    cache_config = engine_args.get("cache_config")
    assert cache_config is not None
    assert cache_config["Fn_compute_blocks"] == 1


def test_default_stage_devices_from_sequence_parallel(monkeypatch):
    """Ensure devices list reflects sequence parallel size when no parallel_config is provided."""
    monkeypatch.setattr(omni_module, "load_stage_configs_from_model", lambda model, base_engine_args=None: [])
    monkeypatch.setattr(omni_module, "resolve_model_config_path", lambda model: None)
    monkeypatch.setattr(AsyncOmni, "_start_stages", lambda self, model: None)
    monkeypatch.setattr(AsyncOmni, "_wait_for_stages_ready", lambda self, timeout=0: None)

    omni = AsyncOmni(
        model="dummy-model",
        ulysses_degree=2,
        ring_degree=2,
    )

    stage_cfg = omni.stage_configs[0]
    runtime = stage_cfg.runtime
    if hasattr(runtime, "get"):
        devices = runtime.get("devices")
    else:
        devices = getattr(runtime, "devices", None)
    assert devices == "0,1,2,3"
