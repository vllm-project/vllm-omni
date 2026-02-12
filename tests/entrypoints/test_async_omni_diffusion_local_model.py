# SPDX-License-Identifier: Apache-2.0
"""
Regression test for GH-573: passing a local model directory should bypass
HuggingFace repo validation and load configs from the filesystem.

This test loads async_omni_diffusion via spec_from_file_location to avoid
bringing in heavy runtime dependencies, and stubs the small set of imports it
needs.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path


def _install_stub(monkeypatch, name: str, module: types.ModuleType) -> None:
    monkeypatch.setitem(sys.modules, name, module)


def _load_async_module():
    """Load vllm_omni.entrypoints.async_omni_diffusion without package imports."""
    sys.modules.pop("vllm_omni.diffusion.model_config_loader", None)
    spec = importlib.util.spec_from_file_location(
        "async_omni_diffusion_under_test",
        Path(__file__).resolve().parents[2] / "vllm_omni" / "entrypoints" / "async_omni_diffusion.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def test_local_model_path_bypasses_hf_validation(monkeypatch, tmp_path):
    # Prepare a fake local model directory with minimal configs
    model_dir = tmp_path / "local_model"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text('{"_class_name": "DummyPipeline"}', encoding="utf-8")
    (model_dir / "transformer").mkdir()
    (model_dir / "transformer" / "config.json").write_text("{}", encoding="utf-8")

    # Stub vllm logger
    vllm_mod = types.ModuleType("vllm")
    vllm_logger_mod = types.ModuleType("vllm.logger")
    vllm_logger_mod.init_logger = lambda name: logging.getLogger(name)
    _install_stub(monkeypatch, "vllm", vllm_mod)
    _install_stub(monkeypatch, "vllm.logger", vllm_logger_mod)

    # Stub huggingface helper to assert it's not invoked
    hf_calls = {"count": 0}
    vllm_tf_utils_mod = types.ModuleType("vllm.transformers_utils")
    vllm_tf_config_mod = types.ModuleType("vllm.transformers_utils.config")

    def _raise_if_called(file_name, model, revision=None):
        hf_calls["count"] += 1
        raise AssertionError("get_hf_file_to_dict should not be called for local paths")

    vllm_tf_config_mod.get_hf_file_to_dict = _raise_if_called  # type: ignore[attr-defined]
    _install_stub(monkeypatch, "vllm.transformers_utils", vllm_tf_utils_mod)
    _install_stub(monkeypatch, "vllm.transformers_utils.config", vllm_tf_config_mod)

    # Stub vllm_omni package and required submodules to avoid importing heavy deps
    repo_root = Path(__file__).resolve().parents[2]
    vo_root = types.ModuleType("vllm_omni")
    vo_root.__path__ = [str(repo_root / "vllm_omni")]
    vo_entrypoints = types.ModuleType("vllm_omni.entrypoints")
    vo_entrypoints.__path__ = [str(repo_root / "vllm_omni" / "entrypoints")]
    vo_diffusion_pkg = types.ModuleType("vllm_omni.diffusion")
    vo_diffusion_pkg.__path__ = [str(repo_root / "vllm_omni" / "diffusion")]
    vo_inputs_pkg = types.ModuleType("vllm_omni.inputs")
    vo_lora_pkg = types.ModuleType("vllm_omni.lora")
    _install_stub(monkeypatch, "vllm_omni", vo_root)
    _install_stub(monkeypatch, "vllm_omni.entrypoints", vo_entrypoints)
    _install_stub(monkeypatch, "vllm_omni.diffusion", vo_diffusion_pkg)
    _install_stub(monkeypatch, "vllm_omni.inputs", vo_inputs_pkg)
    _install_stub(monkeypatch, "vllm_omni.lora", vo_lora_pkg)

    # Stub lightweight diffusion config and engine
    diff_data_mod = types.ModuleType("vllm_omni.diffusion.data")

    class DummyTransformerConfig:
        def __init__(self, params=None):
            self.params = params or {}

        @classmethod
        def from_dict(cls, data):
            return cls(dict(data) if data is not None else {})

    class DummyOmniDiffusionConfig:
        def __init__(self, model, **kwargs):
            self.model = model
            self.omni_kv_config = {}
            self.model_class_name = None
            self.tf_model_config = None

        @classmethod
        def from_kwargs(cls, **kwargs):
            return cls(**kwargs)

        def update_multimodal_support(self):
            pass

    diff_data_mod.TransformerConfig = DummyTransformerConfig
    diff_data_mod.OmniDiffusionConfig = DummyOmniDiffusionConfig
    _install_stub(monkeypatch, "vllm_omni.diffusion.data", diff_data_mod)

    diff_engine_mod = types.ModuleType("vllm_omni.diffusion.diffusion_engine")

    class DummyEngine:
        def __init__(self, cfg=None):
            self.cfg = cfg

    class DummyDiffusionEngine:
        @staticmethod
        def make_engine(cfg):
            return DummyEngine(cfg)

    diff_engine_mod.DiffusionEngine = DummyDiffusionEngine
    _install_stub(monkeypatch, "vllm_omni.diffusion.diffusion_engine", diff_engine_mod)

    # Stub remaining referenced modules
    diff_request_mod = types.ModuleType("vllm_omni.diffusion.request")

    class OmniDiffusionRequest:
        def __init__(self, *args, **kwargs):
            self.prompts = kwargs.get("prompts", [])
            self.request_ids = kwargs.get("request_ids", [])
            self.error = None
            self.output = None
            self.trajectory_latents = None
            self.trajectory_timesteps = None

    diff_request_mod.OmniDiffusionRequest = OmniDiffusionRequest
    _install_stub(monkeypatch, "vllm_omni.diffusion.request", diff_request_mod)

    vo_inputs_data_mod = types.ModuleType("vllm_omni.inputs.data")

    class OmniDiffusionSamplingParams:
        def __init__(self, *args, **kwargs):
            pass

    class OmniPromptType:
        pass

    vo_inputs_data_mod.OmniDiffusionSamplingParams = OmniDiffusionSamplingParams
    vo_inputs_data_mod.OmniPromptType = OmniPromptType
    _install_stub(monkeypatch, "vllm_omni.inputs.data", vo_inputs_data_mod)

    vo_lora_request_mod = types.ModuleType("vllm_omni.lora.request")

    class LoRARequest:
        pass

    vo_lora_request_mod.LoRARequest = LoRARequest
    _install_stub(monkeypatch, "vllm_omni.lora.request", vo_lora_request_mod)

    vo_outputs_mod = types.ModuleType("vllm_omni.outputs")

    class OmniRequestOutput:
        @classmethod
        def from_diffusion(cls, **kwargs):
            return cls()

    vo_outputs_mod.OmniRequestOutput = OmniRequestOutput
    _install_stub(monkeypatch, "vllm_omni.outputs", vo_outputs_mod)

    # Ensure clean module load each time
    sys.modules.pop("async_omni_diffusion_under_test", None)
    async_mod = _load_async_module()

    instance = async_mod.AsyncOmniDiffusion(model=str(model_dir))

    assert isinstance(instance.engine, DummyEngine)
    assert instance.od_config.model_class_name == "DummyPipeline"
    assert hf_calls["count"] == 0  # confirm we bypassed HF repo validation


def test_local_bagel_without_transformer_config(monkeypatch, tmp_path):
    # Prepare local model dir with only config.json describing a bagel model
    model_dir = tmp_path / "bagel_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type": "bagel", "architectures": ["BagelForConditionalGeneration"]}',
        encoding="utf-8",
    )
    # No transformer/config.json on purpose

    # Shared stubs
    vllm_mod = types.ModuleType("vllm")
    vllm_logger_mod = types.ModuleType("vllm.logger")
    vllm_logger_mod.init_logger = lambda name: logging.getLogger(name)
    _install_stub(monkeypatch, "vllm", vllm_mod)
    _install_stub(monkeypatch, "vllm.logger", vllm_logger_mod)

    hf_calls = {"count": 0}
    vllm_tf_utils_mod = types.ModuleType("vllm.transformers_utils")
    vllm_tf_config_mod = types.ModuleType("vllm.transformers_utils.config")

    def _raise_if_called(file_name, model, revision=None):
        hf_calls["count"] += 1
        raise AssertionError("get_hf_file_to_dict should not be called for local paths")

    vllm_tf_config_mod.get_hf_file_to_dict = _raise_if_called  # type: ignore[attr-defined]
    _install_stub(monkeypatch, "vllm.transformers_utils", vllm_tf_utils_mod)
    _install_stub(monkeypatch, "vllm.transformers_utils.config", vllm_tf_config_mod)

    repo_root = Path(__file__).resolve().parents[2]
    vo_root = types.ModuleType("vllm_omni")
    vo_root.__path__ = [str(repo_root / "vllm_omni")]
    _install_stub(monkeypatch, "vllm_omni", vo_root)
    vo_entrypoints = types.ModuleType("vllm_omni.entrypoints")
    vo_entrypoints.__path__ = [str(repo_root / "vllm_omni" / "entrypoints")]
    _install_stub(monkeypatch, "vllm_omni.entrypoints", vo_entrypoints)
    vo_diffusion_pkg = types.ModuleType("vllm_omni.diffusion")
    vo_diffusion_pkg.__path__ = [str(repo_root / "vllm_omni" / "diffusion")]
    _install_stub(monkeypatch, "vllm_omni.diffusion", vo_diffusion_pkg)

    diff_data_mod = types.ModuleType("vllm_omni.diffusion.data")

    class DummyTransformerConfig:
        def __init__(self, params=None):
            self.params = params or {}

        @classmethod
        def from_dict(cls, data):
            return cls(dict(data) if data is not None else {})

    class DummyOmniDiffusionConfig:
        def __init__(self, model, **kwargs):
            self.model = model
            self.omni_kv_config = {}
            self.model_class_name = None
            self.tf_model_config = None

        @classmethod
        def from_kwargs(cls, **kwargs):
            return cls(**kwargs)

        def update_multimodal_support(self):
            pass

    diff_data_mod.TransformerConfig = DummyTransformerConfig
    diff_data_mod.OmniDiffusionConfig = DummyOmniDiffusionConfig
    _install_stub(monkeypatch, "vllm_omni.diffusion.data", diff_data_mod)

    diff_engine_mod = types.ModuleType("vllm_omni.diffusion.diffusion_engine")

    class DummyEngine:
        def __init__(self, cfg=None):
            self.cfg = cfg

    class DummyDiffusionEngine:
        @staticmethod
        def make_engine(cfg):
            return DummyEngine(cfg)

    diff_engine_mod.DiffusionEngine = DummyDiffusionEngine
    _install_stub(monkeypatch, "vllm_omni.diffusion.diffusion_engine", diff_engine_mod)

    diff_request_mod = types.ModuleType("vllm_omni.diffusion.request")
    diff_request_mod.OmniDiffusionRequest = type("OmniDiffusionRequest", (), {})
    _install_stub(monkeypatch, "vllm_omni.diffusion.request", diff_request_mod)

    vo_inputs_data_mod = types.ModuleType("vllm_omni.inputs.data")
    vo_inputs_data_mod.OmniDiffusionSamplingParams = type("OmniDiffusionSamplingParams", (), {})
    vo_inputs_data_mod.OmniPromptType = type("OmniPromptType", (), {})
    _install_stub(monkeypatch, "vllm_omni.inputs.data", vo_inputs_data_mod)
    _install_stub(monkeypatch, "vllm_omni.inputs", types.ModuleType("vllm_omni.inputs"))

    vo_lora_request_mod = types.ModuleType("vllm_omni.lora.request")
    vo_lora_request_mod.LoRARequest = type("LoRARequest", (), {})
    _install_stub(monkeypatch, "vllm_omni.lora.request", vo_lora_request_mod)
    _install_stub(monkeypatch, "vllm_omni.lora", types.ModuleType("vllm_omni.lora"))

    vo_outputs_mod = types.ModuleType("vllm_omni.outputs")
    vo_outputs_mod.OmniRequestOutput = type(
        "OmniRequestOutput", (), {"from_diffusion": classmethod(lambda cls, **_: cls())}
    )
    _install_stub(monkeypatch, "vllm_omni.outputs", vo_outputs_mod)

    # Load and instantiate
    sys.modules.pop("async_omni_diffusion_under_test", None)
    async_mod = _load_async_module()

    instance = async_mod.AsyncOmniDiffusion(model=str(model_dir))

    assert isinstance(instance.engine, DummyEngine)
    assert instance.od_config.model_class_name == "BagelPipeline"
    assert instance.od_config.tf_model_config is not None
    assert hf_calls["count"] == 0
