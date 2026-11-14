# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the opt-in MRv2 ngram adapter.

Loaded from file path so the suite runs without a vllm/vllm-ascend install:
the module under test only imports the standard library at module scope.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MODULE_PATH = Path(__file__).resolve().parents[2] / "vllm_omni" / "worker_v2" / "ngram_adapter.py"


@pytest.fixture(scope="module")
def adapter():
    spec = importlib.util.spec_from_file_location("ngram_adapter_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _install_fake_modules(monkeypatch, modules: dict[str, types.ModuleType]) -> None:
    """Install a dotted package chain into sys.modules with parent attributes."""
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent_name, child = name.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            if parent is not None:
                monkeypatch.setattr(parent, child, module, raising=False)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("1", True),
        ("true", True),
        ("YES", True),
        (" on ", True),
    ],
)
def test_env_flag(monkeypatch, adapter, value, expected):
    if value is None:
        monkeypatch.delenv(adapter.MRV2_NGRAM_ENV, raising=False)
    else:
        monkeypatch.setenv(adapter.MRV2_NGRAM_ENV, value)
    assert adapter.mrv2_ngram_enabled() is expected


@pytest.mark.parametrize(
    ("spec", "method", "is_ngram"),
    [
        (None, None, False),
        ({"method": "ngram"}, "ngram", True),
        ({"method": "ngram_gpu"}, "ngram_gpu", False),
        ({"method": "eagle"}, "eagle", False),
        ({"num_speculative_tokens": 7}, None, False),
        (types.SimpleNamespace(method="ngram"), "ngram", True),
        (types.SimpleNamespace(method="dspark"), "dspark", False),
    ],
)
def test_spec_accessors(adapter, spec, method, is_ngram):
    assert adapter.spec_method(spec) == method
    assert adapter.is_ngram_spec(spec) is is_ngram


NGRAM = {"method": "ngram", "num_speculative_tokens": 7}


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # Gate only applies to an NPU stage that asked for V2 and has a spec.
        (dict(use_v2_model_runner=False, is_npu=True, speculative_config=NGRAM, kstep_frames=0), False),
        (dict(use_v2_model_runner=True, is_npu=False, speculative_config=NGRAM, kstep_frames=0), True),
        (dict(use_v2_model_runner=True, is_npu=True, speculative_config=None, kstep_frames=0), True),
        # Talker K-step and DSpark already stay on V2.
        (dict(use_v2_model_runner=True, is_npu=True, speculative_config=NGRAM, kstep_frames=8), True),
        (
            dict(
                use_v2_model_runner=True,
                is_npu=True,
                speculative_config={"method": "dspark"},
                kstep_frames=0,
            ),
            True,
        ),
        # Default: PR #10313 behaviour -- ngram stage falls back to V1.
        (
            dict(
                use_v2_model_runner=True,
                is_npu=True,
                speculative_config=NGRAM,
                kstep_frames=0,
                supports_scheduler_ngram=True,
                ngram_enabled=False,
                ngram_available=True,
            ),
            False,
        ),
        # Opt-in with a V2 speculator available -- stay on MRv2.
        (
            dict(
                use_v2_model_runner=True,
                is_npu=True,
                speculative_config=NGRAM,
                kstep_frames=0,
                supports_scheduler_ngram=True,
                ngram_enabled=True,
                ngram_available=True,
            ),
            True,
        ),
        # Opt-in requested but the vLLM build cannot serve it -- keep V1.
        (
            dict(
                use_v2_model_runner=True,
                is_npu=True,
                speculative_config=NGRAM,
                kstep_frames=0,
                supports_scheduler_ngram=True,
                ngram_enabled=True,
                ngram_available=False,
            ),
            False,
        ),
        # Enabling the adapter does not move unrelated Ngram stages to MRv2.
        (
            dict(
                use_v2_model_runner=True,
                is_npu=True,
                speculative_config=NGRAM,
                kstep_frames=0,
                ngram_enabled=True,
                ngram_available=True,
            ),
            False,
        ),
        # Only the Ngram family participates in the fallback.
        (
            dict(
                use_v2_model_runner=True,
                is_npu=True,
                speculative_config={"method": "eagle"},
                kstep_frames=0,
                supports_scheduler_ngram=True,
                ngram_enabled=True,
                ngram_available=True,
            ),
            True,
        ),
        (
            dict(
                use_v2_model_runner=True,
                is_npu=True,
                speculative_config={"method": "ngram_gpu"},
                kstep_frames=0,
                supports_scheduler_ngram=True,
                ngram_enabled=True,
                ngram_available=True,
            ),
            False,
        ),
    ],
)
def test_resolve_use_v2_model_runner(adapter, kwargs, expected):
    assert adapter.resolve_use_v2_model_runner(**kwargs) is expected


def test_runner_scope_sets_and_restores(monkeypatch, adapter):
    monkeypatch.setenv(adapter.MRV2_NGRAM_ENV, "1")
    runner = object()
    assert adapter._active_runner is None
    with adapter.ngram_runner_scope(runner):
        assert adapter._active_runner is runner
    assert adapter._active_runner is None


def test_runner_scope_is_noop_when_disabled(monkeypatch, adapter):
    monkeypatch.delenv(adapter.MRV2_NGRAM_ENV, raising=False)
    with adapter.ngram_runner_scope(object()):
        assert adapter._active_runner is None


class FakeNgramGPUSpeculator:
    """Mimics the upstream constructor contract (it stores req_states)."""

    def __init__(self, vllm_config, device, req_states):
        self.vllm_config = vllm_config
        self.device = device
        self.req_states = req_states


@pytest.fixture
def fake_vllm(monkeypatch, adapter):
    adapter._speculator_cls = None
    names = [
        "vllm",
        "vllm.v1",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu",
        "vllm.v1.worker.gpu.spec_decode",
        "vllm.v1.worker.gpu.spec_decode.ngram",
    ]
    fake_modules = {name: types.ModuleType(name) for name in names}
    triton_utils = types.ModuleType("vllm.triton_utils")
    setattr(triton_utils, "HAS_TRITON", True)
    fake_modules["vllm.triton_utils"] = triton_utils
    speculator_mod = types.ModuleType("vllm.v1.worker.gpu.spec_decode.ngram.speculator")
    setattr(speculator_mod, "NgramGPUSpeculator", FakeNgramGPUSpeculator)
    fake_modules[speculator_mod.__name__] = speculator_mod
    _install_fake_modules(monkeypatch, fake_modules)
    yield adapter
    adapter._speculator_cls = None


def test_speculator_resolves_live_req_states(fake_vllm):
    """Ascend rebuilds req_states after init_speculator: never capture it."""
    assert fake_vllm.ngram_speculator_available() is True
    runner = types.SimpleNamespace(req_states=object())
    spec = fake_vllm.build_ngram_speculator("cfg", "device", runner)
    assert spec.req_states is runner.req_states
    runner.req_states = object()
    assert spec.req_states is runner.req_states
    assert spec.vllm_config == "cfg"


def test_build_speculator_raises_without_upstream_support(monkeypatch, adapter):
    monkeypatch.setattr(adapter, "ngram_speculator_available", lambda: False)
    with pytest.raises(RuntimeError, match="V1 runner"):
        adapter.build_ngram_speculator("cfg", "device", object())


def test_ascend_blacklist_filters_only_supported_thinker_ngram(monkeypatch, fake_vllm):
    names = ["vllm_ascend"]
    fake_modules = {name: types.ModuleType(name) for name in names}
    mrv2 = types.ModuleType("vllm_ascend.mrv2_utils")

    def _original(vllm_config):
        return ["ngram speculative decoding", "draft_model speculative decoding"]

    setattr(mrv2, "_v2_blacklist", _original)
    fake_modules[mrv2.__name__] = mrv2
    _install_fake_modules(monkeypatch, fake_modules)
    fake_vllm._patch_ascend_v2_blacklist()
    monkeypatch.setenv(fake_vllm.MRV2_NGRAM_ENV, "1")
    cfg = _thinker_config()
    assert getattr(mrv2, "_v2_blacklist")(cfg) == ["draft_model speculative decoding"]
    cfg.model_config.model_stage = "tts"
    assert getattr(mrv2, "_v2_blacklist")(cfg) == _original(cfg)
    other = types.SimpleNamespace(speculative_config=types.SimpleNamespace(method="eagle"))
    assert getattr(mrv2, "_v2_blacklist")(other) == _original(other)


def test_ascend_init_speculator_routes_ngram(monkeypatch, fake_vllm):
    names = [
        "vllm_ascend",
        "vllm_ascend.worker",
        "vllm_ascend.worker.v2",
    ]
    fake_modules = {name: types.ModuleType(name) for name in names}
    leaf = types.ModuleType("vllm_ascend.worker.v2.spec_decode")
    calls = []

    def _original(vllm_config, device, req_states=None):
        calls.append(("original", req_states))
        return "eagle"

    setattr(leaf, "init_speculator", _original)
    fake_modules[leaf.__name__] = leaf
    runner_module = types.ModuleType("vllm_ascend.worker.v2.model_runner")
    vars(runner_module)["init_speculator"] = _original
    fake_modules[runner_module.__name__] = runner_module
    _install_fake_modules(monkeypatch, fake_modules)
    monkeypatch.setenv(fake_vllm.MRV2_NGRAM_ENV, "1")
    built = []

    def _build(vllm_config, device, runner):
        built.append(runner)
        return "ngram"

    monkeypatch.setattr(fake_vllm, "build_ngram_speculator", _build)
    runner = object()
    with fake_vllm.ngram_runner_scope(runner):
        fake_vllm._patch_ascend_init_speculator()
        cfg = _thinker_config()
        # vllm-ascend 0.33 passes req_states as a third positional argument.
        assert getattr(leaf, "init_speculator")(cfg, "device", "req_states") == "ngram"
        assert vars(runner_module)["init_speculator"] is getattr(leaf, "init_speculator")
        assert built == [runner]
        other = types.SimpleNamespace(speculative_config=types.SimpleNamespace(method="eagle"))
        assert getattr(leaf, "init_speculator")(other, "device", "req_states") == "eagle"
        # Keyword form and the older two-argument form must still work.
        assert getattr(leaf, "init_speculator")(vllm_config=other, device="device") == "eagle"
        cfg.model_config.model_stage = "tts"
        assert getattr(leaf, "init_speculator")(cfg, "device", "talker_state") == "eagle"
    assert calls == [("original", "req_states"), ("original", None), ("original", "talker_state")]


def test_vllm_unsupported_features_filters_ngram(monkeypatch, fake_vllm):
    vllm_config_mod = types.ModuleType("vllm.config")

    class VllmConfig:
        def _get_v2_model_runner_unsupported_features(self):
            return ["ngram/ngram_gpu speculative decoding", "another reason"]

    setattr(vllm_config_mod, "VllmConfig", VllmConfig)
    _install_fake_modules(monkeypatch, {"vllm.config": vllm_config_mod})
    monkeypatch.setenv(fake_vllm.MRV2_NGRAM_ENV, "1")
    fake_vllm._patch_vllm_v2_unsupported_features()
    thinker = _thinker_config()
    cfg = VllmConfig()
    vars(cfg).update(vars(thinker))
    assert cfg._get_v2_model_runner_unsupported_features() == ["another reason"]
    thinker.device_config.device_type = "cuda"
    assert cfg._get_v2_model_runner_unsupported_features() == ["ngram/ngram_gpu speculative decoding", "another reason"]


@pytest.mark.parametrize(
    ("device", "stage", "method", "runner_v2", "async_scheduling", "expected"),
    [
        ("npu", "llm", "ngram", True, False, True),
        ("cuda", "llm", "ngram", True, False, False),
        ("npu", "tts", "ngram", True, False, False),
        ("npu", "llm", "ngram_gpu", True, False, False),
        ("npu", "llm", "ngram", False, False, False),
        ("npu", "llm", "ngram", True, True, False),
    ],
)
def test_scheduler_ngram_requires_synchronous_npu_mrv2_thinker(
    adapter, monkeypatch, device, stage, method, runner_v2, async_scheduling, expected
):
    monkeypatch.setenv(adapter.MRV2_NGRAM_ENV, "1")
    config = _thinker_config()
    config.device_config.device_type = device
    config.model_config.model_stage = stage
    config.model_config.use_v2_model_runner = runner_v2
    config.speculative_config.method = method
    config.scheduler_config.async_scheduling = async_scheduling
    assert adapter.use_scheduler_ngram(config) is expected


def _thinker_config():
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            architecture="MiniCPMO45OmniForConditionalGeneration",
            model_stage="llm",
            use_v2_model_runner=True,
        ),
        device_config=types.SimpleNamespace(device_type="npu"),
        scheduler_config=types.SimpleNamespace(async_scheduling=False),
        speculative_config=types.SimpleNamespace(method="ngram"),
        parallel_config=types.SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        kv_transfer_config=None,
    )
