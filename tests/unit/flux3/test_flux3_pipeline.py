# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract + end-to-end tests for Flux3Pipeline.

Coverage here:
- ``TestOutputModalityDefaults``: pure CPU mode-dispatch logic (no weights/GPU).
- ``TestTextConditioning``: the ``Flux3TextEncoder`` random-output stub shape
  contract + the pipeline's CFG text conditioning (CPU, no weights).
- ``TestForwardE2E``: the real ``forward()`` entry point for text/image conditioning with a
  small random transformer + real image VAE (checkpoint) + random text encoder.

The image VAE encode/decode maths are covered by ``test_flux3_image_vae.py``.

Checkpoint/CUDA-gated tests skip automatically when unavailable. Set
FLUX3_MODEL_PATH to point at a checkpoint:
    FLUX3_MODEL_PATH=/path/to/checkpoint pytest tests/unit/flux3/test_flux3_pipeline.py
"""

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.flux3.flux3_image_vae import Flux3ImageVAE


@dataclass
class _FakeModelOutput:
    """Minimal HuggingFace-style model output with hidden_states."""

    hidden_states: tuple


@dataclass
class _FakeSamplingParams:
    """Minimal sampling params for forward() unit tests."""

    extra_args: dict = field(default_factory=dict)
    height: int = 64
    width: int = 64
    num_inference_steps: int = 2
    guidance_scale: float = 3.5
    guidance_scale_provided: bool = True
    seed: int | None = None
    generator: Any = None


@dataclass
class _FakeRequest:
    """Minimal request batch for forward() unit tests."""

    prompts: list
    sampling_params: _FakeSamplingParams

    def is_dummy_run(self) -> bool:
        return False


MODEL_PATH = os.getenv("FLUX3_MODEL_PATH", "")

requires_checkpoint = pytest.mark.skipif(
    not os.path.isdir(MODEL_PATH),
    reason=f"Checkpoint not found at {MODEL_PATH}. Set FLUX3_MODEL_PATH to run.",
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA",
)


@pytest.mark.parametrize("initial_ops", [["none"], ["none", "+quant_fp8"]])
def test_flux3_enables_native_fp8_only_during_construction(
    monkeypatch: pytest.MonkeyPatch,
    initial_ops: list[str],
) -> None:
    from vllm_omni.diffusion.models.flux3 import pipeline_flux3

    compilation_config = SimpleNamespace(custom_ops=list(initial_ops))
    vllm_config = SimpleNamespace(compilation_config=compilation_config)
    monkeypatch.setattr(pipeline_flux3, "get_current_vllm_config", lambda: vllm_config)

    with pipeline_flux3._enable_native_fp8_quant_for_flux3():
        assert "+quant_fp8" in compilation_config.custom_ops

    assert compilation_config.custom_ops == initial_ops


def test_postprocess_labels_flux3_image_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flux3 image requests retain their image modality."""
    from vllm_omni.diffusion.models.flux3.pipeline_flux3 import (
        Flux3ImageProcessor,
        get_flux3_post_process_func,
    )

    monkeypatch.setattr("vllm_omni.diffusion.data.is_port_available", lambda _port: True)
    with mock.patch.object(Flux3ImageProcessor, "postprocess", return_value="processed-image"):
        postprocess = get_flux3_post_process_func(OmniDiffusionConfig(model=str(tmp_path), master_port=29500))
        output = postprocess({"image": torch.zeros(1)})

    assert output == {"image": "processed-image"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FinalizableTextEncoderStub(torch.nn.Module):
    """Satisfy the pipeline load hook until the E2E fake encoder is installed."""

    def finalize_after_loading(self) -> None:
        pass


def _make_pipeline_stub_with_vaes(device: torch.device) -> object:
    """Construct a partial pipeline and load its VAE weights through the same
    ComponentSource path used in production. The transformer, text encoder, and
    scheduler are intentionally omitted.
    """
    from types import SimpleNamespace

    import torch.nn as nn
    from vllm.config.load import LoadConfig

    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.diffusion.models.flux3.pipeline_flux3 import (
        Flux3ImageProcessor,
        Flux3Pipeline,
    )

    pipe = object.__new__(Flux3Pipeline)
    nn.Module.__init__(pipe)
    pipe._execution_device = device
    # Flux3Pipeline.load_weights() finalizes every production component after
    # loading. This helper loads the VAE weights and installs its fake text encoder
    # afterward, so provide a parameter-free placeholder for that load hook.
    pipe.text_encoder = _FinalizableTextEncoderStub()
    pipe.vae = Flux3ImageVAE(device=device)
    pipe.weights_sources = [
        DiffusersPipelineLoader.ComponentSource(
            model_or_path=MODEL_PATH,
            subfolder=None,
            revision=None,
            prefix="vae._model.",
            allow_patterns_overrides=["ae.safetensors"],
        ),
    ]
    loader_config = SimpleNamespace(
        dtype=torch.bfloat16,
        parallel_config=SimpleNamespace(use_hsdp=False),
        quantization_config=None,
        enable_multithread_weight_load=True,
        num_weight_load_threads=4,
    )
    DiffusersPipelineLoader(LoadConfig(), loader_config).load_weights(pipe)

    image_vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    pipe.image_vae_scale_factor = image_vae_scale_factor
    pipe.latent_channels = pipe.vae.config.latent_channels
    pipe.tokenizer_max_length = 8192
    pipe.default_sample_size = 128
    pipe.image_processor = Flux3ImageProcessor(vae_scale_factor=image_vae_scale_factor * 2)
    pipe.batch_cfg = False
    pipe.od_config = OmniDiffusionConfig(flow_shift=6.93)
    pipe._scheduler_key = None
    pipe._default_num_inference_steps = 50
    pipe._default_guidance_scale = 3.5
    pipe._default_flow_shift = 6.93
    pipe._default_size = (1024, 1024)
    pipe._image_rope_scale = 0.5
    return pipe


# ---------------------------------------------------------------------------
# Construction/loading — CPU, mocked components, no checkpoint required
# ---------------------------------------------------------------------------


class _LoadingComponent(torch.nn.Module):
    """Minimal loadable module used to record pipeline component resolution."""

    def __init__(self, *, image_vae: bool = False) -> None:
        super().__init__()
        if image_vae:
            self.config = type(
                "ImageVAEConfig", (), {"block_out_channels": [128, 256, 512, 512], "latent_channels": 32}
            )()
        else:
            self.config = type("ComponentConfig", (), {"context_in_dim": 64, "vec_in_dim": None})()

    def to(self, *args: Any, **kwargs: Any) -> "_LoadingComponent":
        return self


class _FakeQuantConfig:
    def __init__(
        self,
        name: str,
        *,
        checkpoint_fp8: bool = False,
        checkpoint_nvfp4: bool = False,
    ) -> None:
        self._name = name
        self.is_checkpoint_fp8_serialized = checkpoint_fp8
        self.is_checkpoint_nvfp4_serialized = checkpoint_nvfp4

    def get_name(self) -> str:
        return self._name


class TestLoading:
    """Unit tests for Flux3 native and HF checkpoint component resolution."""

    @staticmethod
    def _f3_package_config(
        *,
        quantization: str | None = None,
        quantization_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "format_version": 1,
            "model": "flux3",
            "quantization": quantization,
            "quantization_config": quantization_config,
            "torch_dtype": "bfloat16",
            "dit": {
                "hidden": 3072,
                "heads": 24,
                "depth": 5,
                "depth_single": 28,
                "mlp_ratio": 3.0,
                "axes_dim": [32, 32, 32, 32],
                "theta": 10000,
                "in_channels": {"image": 128, "image_cond": 128, "text": 20480},
                "out_modes": ["image"],
                "vec_in_dim": None,
                "image_rope_scale": 0.5,
            },
            "text_encoder": {
                "path": "text_encoder",
                "tokenizer": "tokenizer",
                "layers": [4, 8, 12, 16, 20, 24, 28, 32],
                "pad_multiple": 80,
                "max_tokens": 8192,
                "dtype": "bfloat16",
            },
            "vae": {"path": "ae.safetensors", "dtype": "float32"},
            "sampler": {"steps": 50, "guidance": 3.5, "alpha": 6.93, "size": [1024, 1024]},
        }

    @classmethod
    def _write_f3_package_config(
        cls,
        directory: Path,
        *,
        quantization: str | None = None,
        quantization_config: dict[str, Any] | None = None,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(
            json.dumps(
                cls._f3_package_config(
                    quantization=quantization,
                    quantization_config=quantization_config,
                )
            )
        )

    @staticmethod
    def _make_config(model_root: Path, monkeypatch: pytest.MonkeyPatch) -> OmniDiffusionConfig:
        """Build a loading config without opening a TCP socket in a unit test."""
        monkeypatch.setattr("vllm_omni.diffusion.data.is_port_available", lambda _port: True)
        monkeypatch.setattr("vllm_omni.diffusion.data.get_model_path", lambda model, _revision: model)
        return OmniDiffusionConfig(model=str(model_root), master_port=29500)

    def test_config_detects_flux3_package(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from vllm_omni.diffusion.data import resolve_model_class_name

        self._write_f3_package_config(tmp_path)

        assert resolve_model_class_name(str(tmp_path)) == "Flux3Pipeline"

        config = self._make_config(tmp_path, monkeypatch)
        config.enrich_config()

        assert config.model_class_name == "Flux3Pipeline"
        assert config.tf_model_config.params == self._f3_package_config()
        assert config.quantization_config is None

    @pytest.mark.core_model
    @pytest.mark.cpu
    def test_config_detects_flux3_by_architecture(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from vllm_omni.diffusion.data import resolve_model_class_name

        package_config = self._f3_package_config()
        package_config["model"] = "checkpoint"
        (tmp_path / "config.json").write_text(json.dumps(package_config))

        assert resolve_model_class_name(str(tmp_path)) == "Flux3Pipeline"
        config = self._make_config(tmp_path, monkeypatch)
        config.enrich_config()
        assert config.model_class_name == "Flux3Pipeline"
        assert config.tf_model_config.params == self._f3_package_config()

    @pytest.mark.core_model
    @pytest.mark.cpu
    def test_flux3_detection_rejects_unrelated_config(self, tmp_path: Path) -> None:
        from vllm_omni.diffusion.utils.flux3_package import _read_flux3_package_config
        from vllm_omni.diffusion.utils.hf_utils import _looks_like_flux3

        package_config = self._f3_package_config()
        package_config["model"] = "unrelated"
        package_config["dit"].pop("heads")
        (tmp_path / "config.json").write_text(json.dumps(package_config))

        assert not _looks_like_flux3(str(tmp_path), package_config)
        assert _read_flux3_package_config(str(tmp_path)) is None

    def test_config_preserves_explicit_flux3_pipeline_with_package_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._write_f3_package_config(tmp_path)
        config = self._make_config(tmp_path, monkeypatch)
        config.model_class_name = "Flux3Pipeline"
        config.enrich_config()

        assert config.model_class_name == "Flux3Pipeline"
        assert config.tf_model_config.params == self._f3_package_config()

    @staticmethod
    def _make_pipeline(
        config: OmniDiffusionConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[object, dict[str, mock.Mock]]:
        config.enrich_config()
        from vllm_omni.diffusion.models.flux3 import pipeline_flux3

        image_vae = _LoadingComponent(image_vae=True)
        transformer = _LoadingComponent()
        text_encoder = _LoadingComponent()
        text_encoder_config = object()
        mocks = {
            "image": mock.Mock(return_value=image_vae),
            "transformer": mock.Mock(return_value=transformer),
            "text": mock.Mock(return_value=text_encoder),
            "text_config": mock.Mock(return_value=text_encoder_config),
            "processor": mock.Mock(return_value=object()),
        }
        monkeypatch.setattr(pipeline_flux3, "get_local_device", lambda: torch.device("cpu"))
        monkeypatch.setattr(pipeline_flux3, "get_transformer_config_kwargs", lambda *args: {})
        monkeypatch.setattr(pipeline_flux3, "Flux3ImageVAE", mocks["image"])
        monkeypatch.setattr(pipeline_flux3, "Flux3Transformer2DModel", mocks["transformer"])
        monkeypatch.setattr(pipeline_flux3.AutoProcessor, "from_pretrained", mocks["processor"])
        monkeypatch.setattr(pipeline_flux3.AutoConfig, "from_pretrained", mocks["text_config"])
        monkeypatch.setattr(pipeline_flux3.Flux3TextEncoder, "from_config", mocks["text"])
        monkeypatch.setattr(
            pipeline_flux3.Flux3Pipeline, "setup_diffusion_pipeline_profiler", lambda *args, **kwargs: None
        )
        return pipeline_flux3.Flux3Pipeline(od_config=config), mocks

    def test_native_layout_declares_component_sources(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        model_root = tmp_path / "native"
        self._write_f3_package_config(model_root)

        config = self._make_config(model_root, monkeypatch)
        pipe, mocks = self._make_pipeline(config, monkeypatch)

        assert [
            (source.model_or_path, source.subfolder, source.allow_patterns_overrides, source.prefix)
            for source in pipe.weights_sources
        ] == [
            (str(model_root), None, ["model.safetensors"], "transformer."),
            (str(model_root), None, ["ae.safetensors"], "vae._model."),
            (str(model_root), "text_encoder", None, "text_encoder.text_encoder."),
        ]
        mocks["processor"].assert_called_once_with(str(model_root), subfolder="tokenizer", local_files_only=True)
        mocks["text_config"].assert_called_once_with(str(model_root), subfolder="text_encoder", local_files_only=True)

    @pytest.mark.parametrize("variant", [None, "bf16"])
    @pytest.mark.parametrize("remote_tokenizer", [False, True])
    def test_hf_component_references(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str | None, remote_tokenizer: bool
    ) -> None:
        model_root = tmp_path / "flux3"
        self._write_f3_package_config(model_root)
        transformer_root = model_root if variant is None else model_root / "variants" / variant
        transformer_root.mkdir(parents=True, exist_ok=True)
        package_config = self._f3_package_config()
        package_config["text_encoder"]["path"] = "hf://black-forest-labs/flux-3-image-small/text_encoder"
        package_config["vae"]["path"] = "hf://black-forest-labs/flux-3-image-small/vae/ae.safetensors"
        if remote_tokenizer:
            package_config["text_encoder"]["tokenizer"] = "hf://other-owner/processor/tokenizer"
        (transformer_root / "config.json").write_text(json.dumps(package_config))
        config = self._make_config(model_root, monkeypatch)
        if variant is not None:
            config.additional_config["variant"] = variant
        pipe, mocks = self._make_pipeline(config, monkeypatch)

        assert [
            (source.model_or_path, source.subfolder, source.allow_patterns_overrides) for source in pipe.weights_sources
        ] == [
            (str(transformer_root), None, ["model.safetensors"]),
            ("black-forest-labs/flux-3-image-small", None, ["vae/ae.safetensors"]),
            ("black-forest-labs/flux-3-image-small", "text_encoder", None),
        ]
        mocks["text_config"].assert_called_once_with(
            "black-forest-labs/flux-3-image-small", subfolder="text_encoder", local_files_only=False
        )
        mocks["processor"].assert_called_once_with(
            "other-owner/processor" if remote_tokenizer else str(model_root),
            subfolder="tokenizer",
            local_files_only=not remote_tokenizer,
        )

    @pytest.mark.parametrize("path", ["hf://", "hf://owner/repo", "hf://owner//encoder", "hf://owner/repo/../encoder"])
    def test_invalid_hf_component_reference(self, path: str) -> None:
        from vllm_omni.diffusion.utils.flux3_package import _resolve_flux3_component_path

        with pytest.raises(ValueError, match="Invalid Flux3 component reference"):
            _resolve_flux3_component_path("/local/package", path)

    def test_flux3_package_config_drives_components(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        model_root = tmp_path / "flux3"
        self._write_f3_package_config(model_root)

        config = self._make_config(model_root, monkeypatch)
        pipe, mocks = self._make_pipeline(config, monkeypatch)

        assert [
            (source.model_or_path, source.subfolder, source.allow_patterns_overrides, source.prefix)
            for source in pipe.weights_sources
        ] == [
            (str(model_root), None, ["model.safetensors"], "transformer."),
            (str(model_root), None, ["ae.safetensors"], "vae._model."),
            (str(model_root), "text_encoder", None, "text_encoder.text_encoder."),
        ]
        transformer_kwargs = mocks["transformer"].call_args.kwargs
        assert transformer_kwargs["hidden_size"] == 3072
        assert transformer_kwargs["num_heads"] == 24
        assert transformer_kwargs["depth"] == 5
        assert transformer_kwargs["depth_single_blocks"] == 28
        assert transformer_kwargs["context_in_dim"] == 20480
        assert transformer_kwargs["vec_in_dim"] is None
        assert transformer_kwargs["out_modes"] == ("image",)
        mocks["processor"].assert_called_once_with(str(model_root), subfolder="tokenizer", local_files_only=True)
        mocks["text_config"].assert_called_once_with(str(model_root), subfolder="text_encoder", local_files_only=True)
        text_kwargs = mocks["text"].call_args.kwargs
        assert text_kwargs["output_layer"] == [4, 8, 12, 16, 20, 24, 28, 32]
        assert text_kwargs["max_sequence_length"] == 8192
        assert text_kwargs["pad_multiple"] == 80
        assert pipe.tokenizer_max_length == 8192
        assert pipe._default_num_inference_steps == 50
        assert pipe._default_guidance_scale == 3.5
        assert pipe._default_flow_shift == 6.93
        assert pipe._default_size == (1024, 1024)
        assert pipe._image_rope_scale == 0.5

    @pytest.mark.parametrize(
        ("requested", "variant", "quantization", "quant_algo", "method", "runtime_name"),
        [
            ("fp8r", "fp8r", "fp8r", "FP8_PER_CHANNEL_PER_TOKEN", "modelopt", "modelopt"),
            ("nvfp4", "nvfp4", "nvfp4", "NVFP4", "modelopt", "modelopt_fp4"),
        ],
    )
    def test_quantized_package_variant_drives_transformer_source_and_quant_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        requested: str,
        variant: str,
        quantization: str,
        quant_algo: str,
        method: str,
        runtime_name: str,
    ) -> None:
        model_root = tmp_path / "flux3"
        self._write_f3_package_config(model_root)
        variant_dir = model_root / "variants" / variant
        self._write_f3_package_config(variant_dir, quantization=quantization)
        hf_quant_config = {
            "producer": {"name": "flux3"},
            "quantization": {
                "quant_algo": quant_algo,
                "exclude_modules": [
                    "txt_in",
                    "txt_mode_blocks.0.q_proj",
                    "txt_mode_blocks.0.k_proj",
                    "txt_mode_blocks.0.v_proj",
                    "final_layer.image.adaLN_modulation.1",
                ],
            },
        }
        (variant_dir / "hf_quant_config.json").write_text(json.dumps(hf_quant_config))
        built_quant_configs: list[dict[str, Any]] = []

        def fake_build_quant_config(spec: dict[str, Any]) -> _FakeQuantConfig:
            built_quant_configs.append(spec)
            return _FakeQuantConfig(runtime_name)

        monkeypatch.setattr("vllm_omni.diffusion.data.build_quant_config", fake_build_quant_config)
        config = self._make_config(model_root, monkeypatch)
        config.additional_config["variant"] = requested

        config.enrich_config()
        from vllm.config.load import LoadConfig

        from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

        loader = DiffusersPipelineLoader(LoadConfig(), config)
        assert loader.quant_config is config.quantization_config
        assert loader.quant_config.get_name() == runtime_name
        # Pipeline construction must neither rediscover nor replace quantization.
        monkeypatch.setattr(config, "enrich_config", lambda: None)
        (variant_dir / "hf_quant_config.json").unlink()
        pipe, mocks = self._make_pipeline(config, monkeypatch)

        assert [
            (source.model_or_path, source.subfolder, source.allow_patterns_overrides, source.prefix)
            for source in pipe.weights_sources
        ] == [
            (str(variant_dir), None, ["model.safetensors"], "transformer."),
            (str(model_root), None, ["ae.safetensors"], "vae._model."),
            (str(model_root), "text_encoder", None, "text_encoder.text_encoder."),
        ]
        expected_quant_config = {"quant_method": method, **hf_quant_config}
        if quantization == "nvfp4":
            expected_quant_config = {
                **expected_quant_config,
                "quantization": {
                    **hf_quant_config["quantization"],
                    "kv_cache_quant_algo": "FP8",
                },
            }
        assert built_quant_configs == [expected_quant_config]
        assert config.quantization_config.get_name() == runtime_name
        assert config.additional_config["auto_detected_quant_method"] == quantization
        assert config.tf_model_config.params["quantization_config"] == expected_quant_config
        assert mocks["transformer"].call_args.kwargs["quant_config"] is config.quantization_config

    def test_sd_variant_enables_flow_map_and_uses_variant_assets(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        model_root = tmp_path / "flux3"
        self._write_f3_package_config(model_root)
        variant_dir = model_root / "variants" / "sd"
        variant_config = self._f3_package_config()
        variant_config["dit"]["flow_map"] = True
        variant_config["text_encoder"]["tokenizer"] = "text_encoder"
        variant_config["sampler"]["steps"] = 16
        variant_config["sampler"]["guidance"] = None
        variant_dir.mkdir(parents=True)
        (variant_dir / "config.json").write_text(json.dumps(variant_config))

        config = self._make_config(model_root, monkeypatch)
        config.additional_config["variant"] = "sd"
        pipe, mocks = self._make_pipeline(config, monkeypatch)

        assert pipe._flow_map is True
        assert pipe._default_num_inference_steps == 16
        assert pipe._default_guidance_scale is None
        assert pipe.weights_sources[0].model_or_path == str(variant_dir)
        assert mocks["transformer"].call_args.kwargs["flow_map"] is True
        mocks["processor"].assert_called_once_with(str(model_root), subfolder="text_encoder", local_files_only=True)

    @pytest.mark.parametrize(
        ("variant", "quantization", "quant_algo", "runtime_name", "checkpoint_fp8", "checkpoint_nvfp4"),
        [
            ("sd-fp8r", "fp8r", "FP8_PER_CHANNEL_PER_TOKEN", "modelopt", True, False),
            ("sd-nvfp4", "nvfp4", "NVFP4", "modelopt_fp4", False, True),
        ],
    )
    @pytest.mark.parametrize("has_diffusers_index", [False, True])
    def test_sd_quantized_variant_enables_flow_map_and_quantization(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        variant: str,
        quantization: str,
        quant_algo: str,
        runtime_name: str,
        checkpoint_fp8: bool,
        checkpoint_nvfp4: bool,
        has_diffusers_index: bool,
    ) -> None:
        model_root = tmp_path / "flux3"
        self._write_f3_package_config(model_root)
        if has_diffusers_index:
            (model_root / "modular_model_index.json").write_text(
                json.dumps({"_class_name": "Flux3DistilledModularPipeline"})
            )
            transformer_dir = model_root / "transformer"
            transformer_dir.mkdir()
            (transformer_dir / "config.json").write_text(json.dumps({"flow_map": True}))
        variant_dir = model_root / "variants" / variant
        variant_config = self._f3_package_config(quantization=quantization)
        variant_config["dit"]["flow_map"] = True
        variant_config["text_encoder"]["tokenizer"] = "text_encoder"
        variant_config["sampler"].update(steps=16, guidance=None)
        variant_dir.mkdir(parents=True)
        (variant_dir / "config.json").write_text(json.dumps(variant_config))
        (variant_dir / "hf_quant_config.json").write_text(
            json.dumps({"quantization": {"quant_algo": quant_algo, "exclude_modules": ["txt_in"]}})
        )

        monkeypatch.setattr(
            "vllm_omni.diffusion.data.build_quant_config",
            lambda _spec: _FakeQuantConfig(
                runtime_name, checkpoint_fp8=checkpoint_fp8, checkpoint_nvfp4=checkpoint_nvfp4
            ),
        )
        config = self._make_config(model_root, monkeypatch)
        config.additional_config["variant"] = variant
        config.enrich_config()
        pipe, mocks = self._make_pipeline(config, monkeypatch)

        assert pipe._flow_map is True
        assert pipe._default_num_inference_steps == 16
        assert pipe._default_guidance_scale is None
        assert config.quantization_config.get_name() == runtime_name
        assert pipe.weights_sources[0].model_or_path == str(variant_dir)
        assert mocks["transformer"].call_args.kwargs["flow_map"] is True
        mocks["processor"].assert_called_once_with(str(model_root), subfolder="text_encoder", local_files_only=True)

    def test_sampler_config_accepts_absent_guidance(self):
        from vllm_omni.diffusion.models.flux3.pipeline_flux3 import _flux3_sampler_config

        assert _flux3_sampler_config({"sampler": {"guidance": None}})["guidance"] is None

    def test_quantized_package_uses_embedded_quantization_config_without_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model_root = tmp_path / "flux3"
        self._write_f3_package_config(model_root)
        variant_dir = model_root / "variants" / "fp8r"
        embedded_quant_config = {
            "producer": {"name": "flux3"},
            "quantization": {"quant_algo": "FP8_PER_CHANNEL_PER_TOKEN", "exclude_modules": ["txt_in"]},
        }
        expected_quant_config = {"quant_method": "modelopt", **embedded_quant_config}
        self._write_f3_package_config(
            variant_dir,
            quantization="fp8r",
            quantization_config=embedded_quant_config,
        )
        built_quant_configs: list[dict[str, Any]] = []

        def fake_build_quant_config(spec: dict[str, Any]) -> _FakeQuantConfig:
            built_quant_configs.append(spec)
            return _FakeQuantConfig("modelopt")

        monkeypatch.setattr("vllm_omni.diffusion.data.build_quant_config", fake_build_quant_config)
        config = self._make_config(model_root, monkeypatch)
        config.additional_config["variant"] = "fp8r"

        self._make_pipeline(config, monkeypatch)

        assert built_quant_configs == [expected_quant_config]
        assert config.tf_model_config.params["quantization_config"] == expected_quant_config
        assert config.quantization_config.get_name() == "modelopt"

    def test_direct_quantized_package_path_is_rejected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        model_root = tmp_path / "flux3"
        self._write_f3_package_config(model_root)
        variant_dir = model_root / "variants" / "fp8r"
        self._write_f3_package_config(variant_dir, quantization="fp8r")
        config = self._make_config(variant_dir, monkeypatch)

        with pytest.raises(ValueError, match="shared package root"):
            self._make_pipeline(config, monkeypatch)

    @pytest.mark.parametrize(
        ("variant", "quantization", "quant_algo", "runtime_name"),
        [
            ("fp8r", "fp8", "FP8_PER_CHANNEL_PER_TOKEN", "modelopt"),
            ("nvfp4", "nvfp4", "NVFP4", "modelopt_fp4"),
        ],
    )
    def test_quantized_package_uses_variant_config_to_resolve_generic_quantization(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        variant: str,
        quantization: str,
        quant_algo: str,
        runtime_name: str,
    ) -> None:
        model_root = tmp_path / "flux3"
        self._write_f3_package_config(model_root)
        variant_dir = model_root / "variants" / variant
        self._write_f3_package_config(variant_dir, quantization=variant)
        (variant_dir / "hf_quant_config.json").write_text(
            json.dumps(
                {
                    "producer": {"name": "flux3"},
                    "quantization": {"quant_algo": quant_algo, "exclude_modules": ["txt_in"]},
                }
            )
        )

        monkeypatch.setattr(
            "vllm_omni.diffusion.data.build_quant_config",
            lambda spec: _FakeQuantConfig(
                runtime_name, checkpoint_fp8=variant == "fp8r", checkpoint_nvfp4=variant == "nvfp4"
            ),
        )
        config = self._make_config(model_root, monkeypatch)
        config.additional_config["variant"] = variant
        config.quantization_config = quantization

        self._make_pipeline(config, monkeypatch)

        assert config.quantization_config.get_name() == runtime_name

    def test_variant_enrichment_preserves_explicit_quantization_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._write_f3_package_config(tmp_path)
        self._write_f3_package_config(
            tmp_path / "variants" / "fp8r",
            quantization="fp8r",
            quantization_config={"quantization": {"quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"}},
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.data.build_quant_config",
            lambda spec: _FakeQuantConfig("modelopt", checkpoint_fp8=True),
        )
        config = self._make_config(tmp_path, monkeypatch)
        config.additional_config["variant"] = "fp8r"
        override = _FakeQuantConfig("explicit_override")
        config.quantization_config = override

        config.enrich_config()

        assert config.quantization_config is override

    def test_missing_flux3_variant_reports_available_variants(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model_root = tmp_path / "flux3"
        self._write_f3_package_config(model_root)
        self._write_f3_package_config(model_root / "variants" / "fp8r", quantization="fp8r")
        config = self._make_config(model_root, monkeypatch)
        config.additional_config["variant"] = "fp8x"

        with pytest.raises(ValueError, match="available variants: fp8r"):
            self._make_pipeline(config, monkeypatch)

    def test_flux3_variant_must_be_available(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        model_root = tmp_path / "flux3"
        self._write_f3_package_config(model_root)
        self._write_f3_package_config(model_root / "variants" / "fp8r", quantization="fp8r")
        config = self._make_config(model_root, monkeypatch)
        config.additional_config["variant"] = "bf16"

        with pytest.raises(ValueError, match="available variants: fp8r"):
            self._make_pipeline(config, monkeypatch)

    @pytest.mark.parametrize(
        ("cfg_parallel_size", "batch_cfg", "expected_batch_cfg"),
        [
            (1, None, True),
            (2, None, False),
            (1, False, False),
            (1, True, True),
            (2, True, False),
        ],
    )
    def test_batch_cfg_is_fused_without_cfg_parallelism(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        cfg_parallel_size: int,
        batch_cfg: bool | None,
        expected_batch_cfg: bool,
    ) -> None:
        model_root = tmp_path / f"cfg-{cfg_parallel_size}"
        self._write_f3_package_config(model_root)
        config = self._make_config(model_root, monkeypatch)
        config.parallel_config.cfg_parallel_size = cfg_parallel_size
        if batch_cfg is not None:
            config.additional_config["batch_cfg"] = batch_cfg

        pipe, _ = self._make_pipeline(config, monkeypatch)

        assert pipe.batch_cfg is expected_batch_cfg

    def test_batch_cfg_rejects_non_boolean_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model_root = tmp_path / "invalid-batch-cfg"
        self._write_f3_package_config(model_root)
        config = self._make_config(model_root, monkeypatch)
        config.additional_config["batch_cfg"] = "false"

        with pytest.raises(TypeError, match="additional_config\\['batch_cfg'\\] must be a bool or None"):
            self._make_pipeline(config, monkeypatch)


# ---------------------------------------------------------------------------
# Text conditioning — the Flux3TextEncoder random stub, CPU, no weights
# ---------------------------------------------------------------------------


class _FakeProcessor:
    """Minimal processor stub that satisfies Flux3TextEncoder._encode_qwen3()."""

    def apply_chat_template(self, messages, **kwargs) -> str:
        return "fake template"

    @property
    def tokenizer(self):
        return self

    def __call__(self, text, return_tensors=None, padding=None, truncation=None, max_length=None):
        n = max_length or 32
        return {
            "input_ids": torch.zeros(1, n, dtype=torch.long),
            "attention_mask": torch.ones(1, n, dtype=torch.long),
        }


class _FakeLanguageModel(torch.nn.Module):
    """Minimal Qwen language model returning shape-correct hidden states."""

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        *,
        hidden_dim: int,
        n_layers: int,
    ):
        super().__init__()
        self._device = torch.device(device)
        self._dtype = dtype
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers

    def forward(self, input_ids, attention_mask=None, position_ids=None, output_hidden_states=False, use_cache=False):
        B, T = input_ids.shape
        hs = tuple(
            torch.zeros(B, T, self._hidden_dim, device=self._device, dtype=self._dtype) for _ in range(self._n_layers)
        )
        return _FakeModelOutput(hidden_states=hs)


class _FakeTextModel(torch.nn.Module):
    """Minimal stand-in exposing Qwen3-VL's nested language-model layout.

    Args:
        hidden_dim: hidden size per layer.  Must satisfy
            ``len(output_layer) * hidden_dim == context_in_dim``.
        n_layers: number of hidden-state tensors to expose.  Must be
            ``> max(output_layer)``; default 33 covers the default ``[4..32]`` list.
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        *,
        hidden_dim: int = 16,
        n_layers: int = 33,
    ):
        super().__init__()
        self._device = torch.device(device)
        self._dtype = dtype
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self.model = torch.nn.Module()
        self.model.language_model = _FakeLanguageModel(
            device,
            dtype,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def forward(self, input_ids, attention_mask=None, position_ids=None, output_hidden_states=False, use_cache=False):
        return self.model.language_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )


class TestTextConditioning:
    """CPU contract tests for the text conditioning path.

    Uses a ``_FakeTextModel`` stub to pin the encoder's shape contract and the
    pipeline's CFG text conditioning without requiring a checkpoint or GPU.
    """

    CTX_DIM = 128
    VEC_DIM = 16
    N_TOKENS = 32

    def _pipe(self):
        import torch.nn as nn

        from vllm_omni.diffusion.models.flux3.flux3_text_encoder import (
            Flux3TextEncoder,
        )
        from vllm_omni.diffusion.models.flux3.pipeline_flux3 import (
            Flux3Pipeline,
        )

        device = torch.device("cpu")
        pipe = object.__new__(Flux3Pipeline)
        nn.Module.__init__(pipe)
        pipe._execution_device = device
        # 8 default output layers × hidden_dim=16 == CTX_DIM=128
        pipe.text_encoder = Flux3TextEncoder(
            _FakeTextModel(device, hidden_dim=self.CTX_DIM // 8),
            processor=_FakeProcessor(),
            context_in_dim=self.CTX_DIM,
            vec_in_dim=self.VEC_DIM,
            max_sequence_length=self.N_TOKENS,
        )
        return pipe

    def test_encode_shapes(self):
        ctx, vector = self._pipe().text_encoder.encode("a photo of a cat")
        assert ctx.shape == (1, self.N_TOKENS, self.CTX_DIM)
        assert vector.shape == (1, self.VEC_DIM)

    def test_encode_finite(self):
        ctx, vector = self._pipe().text_encoder.encode("a landscape with mountains")
        assert torch.isfinite(ctx.float()).all()
        assert torch.isfinite(vector.float()).all()

    def test_text_conditioning_cfg_doubled_shapes(self):
        pipe = self._pipe()
        negative, positive = pipe._text_conditioning("hello world", torch.device("cpu"))

        for ctx, ctx_ids, vector, ts_ctx in (negative, positive):
            assert ctx.shape == (1, self.N_TOKENS, self.CTX_DIM)
            assert ctx_ids.shape == (1, self.N_TOKENS, 4)
            assert vector.shape == (1, self.VEC_DIM)
            assert ts_ctx.shape == (1, self.N_TOKENS)
            assert torch.equal(ts_ctx, torch.zeros_like(ts_ctx))

    def test_text_conditioning_uses_branch_local_lengths(self):
        import torch.nn as nn

        from vllm_omni.diffusion.models.flux3.pipeline_flux3 import Flux3Pipeline

        class VariableTextEncoder:
            def __init__(self, ctx_dim: int, vec_dim: int):
                self.ctx_dim = ctx_dim
                self.vec_dim = vec_dim

            def encode(self, prompt: str):
                length = 8 if prompt == "" else 16
                ctx = torch.zeros(1, length, self.ctx_dim, dtype=torch.bfloat16)
                vector = torch.zeros(1, self.vec_dim, dtype=torch.bfloat16)
                return ctx, vector

        pipe = object.__new__(Flux3Pipeline)
        nn.Module.__init__(pipe)
        pipe.text_encoder = VariableTextEncoder(self.CTX_DIM, self.VEC_DIM)

        negative, positive = pipe._text_conditioning("a longer positive prompt", torch.device("cpu"))

        assert negative[0].shape == (1, 8, self.CTX_DIM)
        assert negative[1].shape == (1, 8, 4)
        assert positive[0].shape == (1, 16, self.CTX_DIM)
        assert positive[1].shape == (1, 16, 4)

    def test_diffuse_pads_cfg_text_when_lengths_differ(self):
        import torch.nn as nn

        from vllm_omni.diffusion.models.flux3.pipeline_flux3 import Flux3Pipeline

        class FakeTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.precompute_ctx_shapes = []
                self.forward_ctx_shapes = []

            def precompute_static_conditioning(self, ctx, ctx_ids, vector, timesteps_ctx, **_kwargs):
                self.precompute_ctx_shapes.append(tuple(ctx.shape))
                return {}

            def forward(self, **kwargs):
                self.forward_ctx_shapes.append(tuple(kwargs["ctx"].shape))
                self.text_lens = kwargs["precomputed"].get("text_lens")
                return {"x_image": torch.zeros_like(kwargs["x_image"])}

        class NoopProgress:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def update(self):
                pass

        @contextmanager
        def progress_bar(*, total):
            yield NoopProgress()

        def branch(length):
            ctx = torch.zeros(1, length, self.CTX_DIM, dtype=torch.bfloat16)
            ctx_ids = torch.zeros(1, length, 4)
            vector = torch.zeros(1, self.VEC_DIM, dtype=torch.bfloat16)
            timesteps_ctx = torch.zeros(1, length, dtype=torch.bfloat16)
            return ctx, ctx_ids, vector, timesteps_ctx

        pipe = object.__new__(Flux3Pipeline)
        nn.Module.__init__(pipe)
        pipe.od_config = SimpleNamespace(dtype=torch.bfloat16)
        pipe.batch_cfg = True
        fake_transformer = FakeTransformer()
        pipe.transformer = fake_transformer
        pipe.progress_bar = progress_bar

        latents = torch.ones(1, 2, 128, dtype=torch.bfloat16)
        latent_ids = torch.zeros(1, 2, 4)

        out = pipe.diffuse(
            latents,
            latent_ids,
            timesteps=torch.tensor([1.0, 0.0], dtype=torch.bfloat16),
            text_conditioning=(branch(8), branch(16)),
            guidance_scale=3.5,
        )

        assert torch.equal(out, latents)
        # Static text streams per branch at their own length; one fused B=2 forward over
        # text padded to the longer branch, each sample attending to its own length.
        assert fake_transformer.precompute_ctx_shapes == [(1, 8, self.CTX_DIM), (1, 16, self.CTX_DIM)]
        assert fake_transformer.forward_ctx_shapes == [(2, 16, self.CTX_DIM)]
        assert fake_transformer.text_lens == (8, 16)

    def test_diffuse_keeps_separate_forwards_under_cache_dit(self):
        import torch.nn as nn

        from vllm_omni.diffusion.models.flux3.pipeline_flux3 import Flux3Pipeline

        class FakeTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.precompute_ctx_shapes = []
                self.forward_ctx_shapes = []

            def precompute_static_conditioning(self, ctx, ctx_ids, vector, timesteps_ctx, **_kwargs):
                self.precompute_ctx_shapes.append(tuple(ctx.shape))
                return {}

            def forward(self, **kwargs):
                self.forward_ctx_shapes.append(tuple(kwargs["ctx"].shape))
                return {"x_image": torch.zeros_like(kwargs["x_image"])}

        class NoopProgress:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def update(self):
                pass

        @contextmanager
        def progress_bar(*, total):
            yield NoopProgress()

        def branch(length):
            ctx = torch.zeros(1, length, self.CTX_DIM, dtype=torch.bfloat16)
            ctx_ids = torch.zeros(1, length, 4)
            vector = torch.zeros(1, self.VEC_DIM, dtype=torch.bfloat16)
            timesteps_ctx = torch.zeros(1, length, dtype=torch.bfloat16)
            return ctx, ctx_ids, vector, timesteps_ctx

        pipe = object.__new__(Flux3Pipeline)
        nn.Module.__init__(pipe)
        pipe.od_config = SimpleNamespace(dtype=torch.bfloat16)
        pipe.batch_cfg = True
        fake_transformer = FakeTransformer()
        fake_transformer._is_cached = True
        pipe.transformer = fake_transformer
        pipe.progress_bar = progress_bar

        latents = torch.ones(1, 2, 128, dtype=torch.bfloat16)
        latent_ids = torch.zeros(1, 2, 4)

        out = pipe.diffuse(
            latents,
            latent_ids,
            timesteps=torch.tensor([1.0, 0.0], dtype=torch.bfloat16),
            text_conditioning=(branch(8), branch(16)),
            guidance_scale=3.5,
        )

        assert torch.equal(out, latents)
        # Cache-DiT has no per-sample attention lengths: two B=1 forwards as before.
        assert fake_transformer.precompute_ctx_shapes == [(1, 8, self.CTX_DIM), (1, 16, self.CTX_DIM)]
        assert fake_transformer.forward_ctx_shapes == [(1, 16, self.CTX_DIM), (1, 8, self.CTX_DIM)]

    def test_flow_map_diffuse_passes_next_timestep_without_cfg(self):
        import torch.nn as nn

        from vllm_omni.diffusion.models.flux3.pipeline_flux3 import Flux3Pipeline

        class FakeTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, **kwargs):
                self.calls.append(kwargs)
                return {"x_image": torch.ones_like(kwargs["x_image"])}

        @contextmanager
        def progress_bar(*, total):
            yield SimpleNamespace(update=lambda: None)

        pipe = object.__new__(Flux3Pipeline)
        nn.Module.__init__(pipe)
        pipe.od_config = SimpleNamespace(dtype=torch.bfloat16)
        pipe.transformer = FakeTransformer()
        pipe.progress_bar = progress_bar
        ctx = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
        branch = (ctx, torch.zeros(1, 2, 4), torch.zeros(1, 3), torch.zeros(1, 2, dtype=torch.bfloat16))
        latents = torch.zeros(1, 2, 128, dtype=torch.float32)
        result = pipe._diffuse_flow_map(
            latents,
            torch.zeros(1, 2, 4),
            text=branch,
            timesteps=torch.tensor([1.0, 0.5, 0.0]),
        )

        assert len(pipe.transformer.calls) == 2
        assert [call["timesteps_ref"].tolist() for call in pipe.transformer.calls] == [[0.5], [0.0]]
        assert [call["x_image_timesteps"][0, 0].item() for call in pipe.transformer.calls] == [1.0, 0.5]
        assert all(call["timesteps_ref"].dtype == torch.bfloat16 for call in pipe.transformer.calls)
        assert all(call["x_image_timesteps"].dtype == torch.bfloat16 for call in pipe.transformer.calls)
        torch.testing.assert_close(result, torch.full_like(result, -1.0))


# ---------------------------------------------------------------------------
# Forward contract — CPU, no weights
# ---------------------------------------------------------------------------


class TestForwardContract:
    """Unsupported request fields are rejected before generation starts."""

    @staticmethod
    def _pipe():
        import torch.nn as nn

        from vllm_omni.diffusion.models.flux3.pipeline_flux3 import Flux3Pipeline

        pipe = object.__new__(Flux3Pipeline)
        nn.Module.__init__(pipe)
        pipe.od_config = SimpleNamespace(flow_shift=None)
        pipe._flow_map = False
        pipe._execution_device = torch.device("cpu")
        pipe._default_num_inference_steps = 50
        pipe._default_guidance_scale = 3.5
        pipe._default_flow_shift = 6.93
        pipe._default_size = (1024, 1024)
        return pipe

    @staticmethod
    def _request(extra_args: dict[str, Any], multi_modal_data: dict[str, Any] | None = None) -> _FakeRequest:
        prompt = {"prompt": "x", "multi_modal_data": multi_modal_data or {}}
        return _FakeRequest(prompts=[prompt], sampling_params=_FakeSamplingParams(extra_args=extra_args))

    def test_rejects_unsupported_multimodal_data(self):
        with pytest.raises(ValueError, match="supports only image multi_modal_data"):
            self._pipe().forward(
                self._request(
                    {},
                    multi_modal_data={"unsupported": object()},
                )
            )

    @pytest.mark.parametrize("image", [None, torch.zeros(1, 3, 64, 64)])
    def test_passes_optional_image_condition(self, image):
        pipe = self._pipe()
        pipe.generate = mock.Mock(return_value={"image": torch.zeros(1, 3, 64, 64)})
        req = self._request({}, multi_modal_data={} if image is None else {"image": image})
        req.sampling_params.seed = 123

        out = pipe.forward(req)

        assert set(out.output) == {"image"}
        assert pipe.generate.call_args.kwargs["image"] is image

    def test_ignores_unknown_framework_extra_args(self):
        pipe = self._pipe()
        pipe.generate = mock.Mock(return_value={"image": torch.zeros(1, 3, 64, 64)})
        req = self._request({"flow_shift": 6.5, "cfg_text_scale": 4.25, "cfg_img_scale": 1.0, "mode": "i2i"})
        req.sampling_params.seed = 123

        pipe.forward(req, guidance_scale=9.0)

        kwargs = pipe.generate.call_args.kwargs
        assert kwargs["image"] is None
        assert kwargs["guidance_scale"] == 3.5
        assert kwargs["flow_shift"] == 6.5

    def test_flow_map_rejects_explicit_guidance_and_negative_prompt(self):
        pipe = self._pipe()
        pipe._flow_map = True
        pipe._default_guidance_scale = None
        pipe.generate = mock.Mock(return_value={"image": torch.zeros(1, 3, 64, 64)})
        req = self._request({})
        req.sampling_params.guidance_scale_provided = True
        with pytest.raises(ValueError, match="sd variant does not support guidance_scale"):
            pipe.forward(req)

        req = self._request({})
        req.prompts[0]["negative_prompt"] = "avoid this"
        req.sampling_params.guidance_scale_provided = False
        with pytest.raises(ValueError, match="sd variant does not support negative_prompt"):
            pipe.forward(req)

    def test_flow_map_ignores_engine_guidance_during_dummy_warmup(self):
        pipe = self._pipe()
        pipe._flow_map = True
        pipe._default_guidance_scale = None
        pipe.generate = mock.Mock(return_value={"image": torch.zeros(1, 3, 64, 64)})
        req = self._request({})
        req.sampling_params.guidance_scale = 0.0
        req.sampling_params.guidance_scale_provided = True
        req.is_dummy_run = lambda: True

        pipe.forward(req)

        assert pipe.generate.call_args.kwargs["guidance_scale"] is None

    def test_dummy_run_skips_prims_ts_autotune_when_backend_falls_back(self, monkeypatch: pytest.MonkeyPatch):
        from vllm_omni.diffusion.models.flux3 import pipeline_flux3

        pipe = self._pipe()
        pipe.transformer = SimpleNamespace(kernel_adapter=SimpleNamespace(backend="vllm"))
        pipe.generate = mock.Mock(return_value={"image": torch.zeros(1, 3, 64, 64)})
        req = self._request({})
        req.sampling_params.seed = 123
        req.is_dummy_run = lambda: True

        monkeypatch.setattr(pipeline_flux3, "prims_ts_autotune_enabled", lambda: True)
        context = mock.Mock(side_effect=AssertionError("unsupported backend must not enter autotune"))
        monkeypatch.setattr(pipeline_flux3, "flux3_prims_ts_autotune_context", context)

        pipe.forward(req)

        context.assert_not_called()

    def test_uses_package_defaults_when_sampling_params_omit_values(self):
        pipe = self._pipe()
        pipe._default_num_inference_steps = 37
        pipe._default_guidance_scale = 3.25
        pipe._default_flow_shift = 7.0
        pipe._default_size = (768, 512)
        pipe.generate = mock.Mock(return_value={"image": torch.zeros(1, 3, 64, 64)})
        req = self._request({})
        req.sampling_params.height = None
        req.sampling_params.width = None
        req.sampling_params.num_inference_steps = None
        req.sampling_params.guidance_scale_provided = False
        req.sampling_params.seed = 123

        pipe.forward(req)

        kwargs = pipe.generate.call_args.kwargs
        assert kwargs["height"] == 512
        assert kwargs["width"] == 768
        assert kwargs["num_inference_steps"] == 37
        assert kwargs["guidance_scale"] == 3.25
        assert kwargs["flow_shift"] == 7.0


# ---------------------------------------------------------------------------
# End-to-end forward() — the real entry point, per image generation mode.
# Full pipeline = small RANDOM transformer + real image VAE (checkpoint) + random text
# encoder. This exercises the forward -> generate -> per-mode dispatch -> decode
# -> output-dict contract (keys / shape / dtype / finite / determinism).
# ---------------------------------------------------------------------------

# Small DiT config. Image stream in_channels (128) are fixed by the model, so a
# small hidden_size still connects to the real image VAE.
_E2E_DIT = dict(
    hidden_size=256,
    num_heads=4,
    depth=1,
    depth_single_blocks=2,
    depth_late_blocks=0,
    context_in_dim=64,
    vec_in_dim=None,
    mlp_ratio=2.0,
    axes_dim=(16, 16, 16, 16),
    theta=10000,
)


def _start_tp_mocks():
    patches = [
        mock.patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=1),
        mock.patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", return_value=0),
        mock.patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1),
        mock.patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0),
        mock.patch("vllm.model_executor.layers.linear.tensor_model_parallel_all_reduce", side_effect=lambda x: x),
        mock.patch(
            "vllm.model_executor.layers.linear.tensor_model_parallel_all_gather", side_effect=lambda x, dim=-1: x
        ),
        mock.patch(
            "vllm_omni.diffusion.distributed.cfg_parallel.get_classifier_free_guidance_world_size", return_value=1
        ),
    ]
    for p in patches:
        p.start()
    return patches


@requires_checkpoint
@requires_cuda
class TestForwardE2E:
    @pytest.fixture(scope="class")
    def full_pipe(self):
        """Full pipeline stub; keeps the vllm-config + TP mocks live across the
        yield so forward() runs the DiT under the same context it was built in."""
        from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

        from vllm_omni.diffusion.models.flux3.flux3_text_encoder import Flux3TextEncoder
        from vllm_omni.diffusion.models.flux3.flux3_transformer import Flux3Transformer2DModel

        device = torch.device("cuda")
        patches = _start_tp_mocks()
        cfg_ctx = set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cuda")))
        cfg_ctx.__enter__()
        try:
            pipe = _make_pipeline_stub_with_vaes(device)  # real image VAE
            pipe.transformer = Flux3Transformer2DModel(**_E2E_DIT).to(device, torch.bfloat16).eval()
            for p in pipe.transformer.parameters():
                p.data.zero_()
            # 8 default output layers × hidden_dim == context_in_dim
            pipe.text_encoder = Flux3TextEncoder(
                _FakeTextModel(device, hidden_dim=_E2E_DIT["context_in_dim"] // 8),
                processor=_FakeProcessor(),
                context_in_dim=_E2E_DIT["context_in_dim"],
                vec_in_dim=_E2E_DIT["vec_in_dim"],
                max_sequence_length=16,
            )
            yield pipe
        finally:
            cfg_ctx.__exit__(None, None, None)
            for p in patches:
                p.stop()

    @staticmethod
    def _request(mode, *, seed=0):
        hw = 64
        mm = {}
        if mode == "i2i":
            mm["image"] = (torch.rand(1, 3, hw, hw, device="cuda") * 2 - 1).float()
        prompt = {"prompt": "x", "multi_modal_data": mm} if mm else "x"
        sp = _FakeSamplingParams(
            extra_args={},
            height=hw,
            width=hw,
            num_inference_steps=2,
            guidance_scale=3.5,
            guidance_scale_provided=True,
            seed=seed,
            generator=None,
        )
        return _FakeRequest(prompts=[prompt], sampling_params=sp)

    _IMAGE_MODES = {"t2i", "i2i"}

    @pytest.mark.parametrize("mode", ["t2i", "i2i"])
    def test_forward_output_contract(self, full_pipe, mode):
        with torch.no_grad():
            out = full_pipe.forward(self._request(mode)).output
        assert mode in self._IMAGE_MODES
        assert set(out.keys()) == {"image"}, f"{mode}: keys {list(out)} != {{'image'}}"
        for k, v in out.items():
            assert isinstance(v, torch.Tensor) and v.ndim == 4, f"{mode}/{k}: bad tensor {type(v)}"
            assert torch.isfinite(v.float()).all(), f"{mode}/{k}: non-finite output"

    def test_forward_deterministic(self, full_pipe):
        # Same seed -> identical output (RNG seeded up front; text stub is seed-driven).
        with torch.no_grad():
            a = full_pipe.forward(self._request("t2i", seed=0)).output["image"]
            b = full_pipe.forward(self._request("t2i", seed=0)).output["image"]
        assert torch.equal(a, b), "forward() not deterministic under a fixed seed"
