# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import errno
import gc
import json
import stat
import zipfile
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers import DiffusionPipeline  # pyright: ignore[reportPrivateImportUsage]
from PIL import Image

import vllm_omni.diffusion.data as diffusion_data
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient, dummy_messages_from_mix_data
from vllm_omni.diffusion.data import (
    AttentionConfig,
    AttentionSpec,
    DiffusionOutput,
    DiffusionParallelConfig,
    OmniDiffusionConfig,
)
from vllm_omni.diffusion.models.diffusers_adapter import DiffusersAdapterPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.diffusion]


def _make_od_config(**overrides) -> OmniDiffusionConfig:
    od_config = OmniDiffusionConfig(
        model="test/model",
        model_class_name="DiffusersAdapterPipeline",
        dtype=torch.float16,
        diffusion_load_format="diffusers",
        diffusers_load_kwargs={},
        diffusers_call_kwargs={},
        output_type="pil",
        parallel_config=DiffusionParallelConfig(cfg_parallel_size=1, sequence_parallel_size=1),
        cache_backend="none",
    )
    for key, value in overrides.items():
        setattr(od_config, key, value)
    return od_config


def _make_request(**overrides) -> OmniDiffusionRequest:
    prompt = overrides.pop("prompt", "a test prompt")
    negative_prompt = overrides.pop("negative_prompt", None)
    prompt_obj: dict[str, str] = {"prompt": prompt}
    if negative_prompt is not None:
        prompt_obj["negative_prompt"] = negative_prompt

    defaults = {
        "prompt": prompt_obj,
        "sampling_params": OmniDiffusionSamplingParams(
            num_inference_steps=20,
            guidance_scale=7.5,
            height=16,
            width=16,
            num_frames=1,
            num_outputs_per_prompt=1,
            seed=42,
            output_type="pil",
            generator_device="cpu",
        ),
        "request_id": "diffusers-backend",
    }
    defaults.update(overrides)
    return OmniDiffusionRequest(**defaults)


def _patch_fake_diffusers_quantization_backends(mocker):
    class FakePipelineQuantizationConfig:
        def __init__(self, *, quant_mapping=None, quant_backend=None, quant_kwargs=None, components_to_quantize=None):
            self.quant_mapping = quant_mapping
            self.quant_backend = quant_backend
            self.quant_kwargs = quant_kwargs
            self.components_to_quantize = components_to_quantize

    class FakeTorchAoConfig:
        def __init__(self, quant_type, modules_to_not_convert=None):
            self.quant_type = quant_type
            self.modules_to_not_convert = modules_to_not_convert

        @classmethod
        def from_dict(cls, config):
            return cls(quant_type=config["quant_type"])

    class FakeInt8DynamicActivationInt8WeightConfig:
        pass

    mocker.patch(
        "vllm_omni.diffusion.models.diffusers_adapter.quantization_utils.PipelineQuantizationConfig",
        FakePipelineQuantizationConfig,
    )
    mocker.patch(
        "vllm_omni.diffusion.models.diffusers_adapter.quantization_utils.TorchAoConfig",
        FakeTorchAoConfig,
    )
    mocker.patch(
        "vllm_omni.diffusion.models.diffusers_adapter.quantization_utils._get_torchao_quant_type_cls",
        return_value=FakeInt8DynamicActivationInt8WeightConfig,
    )
    mocker.patch(
        "vllm_omni.diffusion.models.diffusers_adapter.quantization_utils._get_diffusers_quantization_config_cls",
        return_value=FakeTorchAoConfig,
    )
    return FakePipelineQuantizationConfig, FakeTorchAoConfig, FakeInt8DynamicActivationInt8WeightConfig


class _TransformerBackedPipeline(DiffusionPipeline):
    def __init__(self, transformer):
        pass


def _make_batch(**overrides) -> DiffusionRequestBatch:
    """Wrap a single request in a DiffusionRequestBatch, matching the forward contract."""
    return DiffusionRequestBatch(requests=[_make_request(**overrides)])


@pytest.mark.core_model
@pytest.mark.cpu
class TestPipelineArgumentsHandling:
    def test_adapter_forward_returns_output(self, mocker):
        od_config = _make_od_config()
        request = _make_request()
        stub_image = Image.new("RGB", (request.sampling_params.width, request.sampling_params.height))  # pyright: ignore[reportArgumentType]

        adapter = DiffusersAdapterPipeline(od_config=od_config)
        MockPipelineOutput = namedtuple("MockPipelineOutput", ["image"])
        MockPipeline = type("MockPipeline", (DiffusionPipeline,), {})
        adapter._pipeline = MockPipeline()

        mocker.patch.object(
            MockPipeline,
            "__call__",
            return_value=MockPipelineOutput(image=stub_image),
        )
        output = adapter.forward(DiffusionRequestBatch(requests=[request]))

        assert isinstance(output, DiffusionOutput)
        assert isinstance(output.output, MockPipelineOutput)
        assert output.output.image is stub_image

    @staticmethod
    def _make_adapter(backend: str, set_attention_backend):
        adapter = DiffusersAdapterPipeline(
            od_config=_make_od_config(
                diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend=backend)),
            )
        )
        adapter._pipeline = SimpleNamespace(
            transformer=SimpleNamespace(set_attention_backend=set_attention_backend),
        )
        return adapter

    @pytest.mark.parametrize("backend", ["CUDNN_ATTN", "FLASHINFER_ATTN", "TRTLLM_ATTN", "SPARGE_ATTN"])
    def test_explicit_unbound_backend_is_rejected(self, backend):
        calls: list[str] = []
        adapter = self._make_adapter(backend, calls.append)

        with pytest.raises(ValueError, match="do not provide a binding"):
            adapter._set_attention_backend()

        assert calls == []

    def test_failed_explicit_backend_does_not_fallback_to_native(self):
        calls: list[str] = []

        def reject_backend(backend):
            calls.append(backend)
            raise RuntimeError(f"{backend} unavailable")

        adapter = self._make_adapter("SAGE_ATTN", reject_backend)

        with pytest.raises(RuntimeError, match="explicitly selected attention backend 'SAGE_ATTN'"):
            adapter._set_attention_backend()

        assert calls == ["sage_hub", "sage", "sage_varlen"]
        assert "native" not in calls

    def test_explicit_flash_attn_3_hub_does_not_fallback_to_fa2_or_local(self, monkeypatch):
        monkeypatch.setattr(DiffusersAdapterPipeline, "_is_blackwell_gpu", lambda self: False)
        calls: list[str] = []

        def accept_later_backends(backend):
            calls.append(backend)
            if backend in ("flash_hub", "flash_varlen_hub", "flash", "flash_varlen", "_native_flash"):
                return
            raise RuntimeError(f"{backend} unavailable")

        adapter = self._make_adapter("FLASH_ATTN_3_HUB", accept_later_backends)

        with pytest.raises(RuntimeError, match="explicitly selected attention backend 'FLASH_ATTN_3_HUB'"):
            adapter._set_attention_backend()

        assert calls == ["_flash_3_hub", "_flash_3_varlen_hub"]

    def test_explicit_flash_attn_hub_does_not_fallback_to_local(self, monkeypatch):
        monkeypatch.setattr(DiffusersAdapterPipeline, "_is_blackwell_gpu", lambda self: False)
        calls: list[str] = []

        def accept_local_flash(backend):
            calls.append(backend)
            if backend in ("flash", "flash_varlen", "_native_flash"):
                return
            raise RuntimeError(f"{backend} unavailable")

        adapter = self._make_adapter("FLASH_ATTN_HUB", accept_local_flash)

        with pytest.raises(RuntimeError, match="explicitly selected attention backend 'FLASH_ATTN_HUB'"):
            adapter._set_attention_backend()

        assert calls == ["flash_hub", "flash_varlen_hub"]

    def test_explicit_torch_sdpa_maps_to_diffusers_native(self):
        calls: list[str] = []
        adapter = self._make_adapter("TORCH_SDPA", calls.append)

        adapter._set_attention_backend()

        assert calls == ["native"]

    @pytest.mark.parametrize(
        "feature_id",
        [
            "compact_offload",
            "cfg_parallel",
            "tensor_parallel",
            "ulysses",
            "ring",
            "teacache",
            "cache_dit",
            "enforce_eager",
            "unsupported_quantization",
        ],
    )
    def test_adapter_guard_unsupported_feature(self, feature_id):
        if feature_id == "compact_offload":
            od_config = _make_od_config(
                diffusion_offload_config={"mode": "module", "components": ["dit"]},
            )
        elif feature_id == "cfg_parallel":
            od_config = _make_od_config(
                parallel_config=DiffusionParallelConfig(cfg_parallel_size=2, sequence_parallel_size=1),
                cache_backend="none",
            )
        elif feature_id == "tensor_parallel":
            od_config = _make_od_config(
                parallel_config=DiffusionParallelConfig(tensor_parallel_size=2, sequence_parallel_size=1),
                cache_backend="none",
            )
        elif feature_id == "ulysses":
            od_config = _make_od_config(
                parallel_config=DiffusionParallelConfig(cfg_parallel_size=1, ulysses_degree=2),
                cache_backend="none",
            )
        elif feature_id == "ring":
            od_config = _make_od_config(
                parallel_config=DiffusionParallelConfig(cfg_parallel_size=1, ring_degree=2),
                cache_backend="none",
            )
        elif feature_id == "teacache":
            od_config = _make_od_config(
                parallel_config=DiffusionParallelConfig(cfg_parallel_size=1, sequence_parallel_size=1),
                cache_backend="tea_cache",
            )
        elif feature_id == "cache_dit":
            od_config = _make_od_config(
                parallel_config=DiffusionParallelConfig(cfg_parallel_size=1, sequence_parallel_size=1),
                cache_backend="cache_dit",
            )
        elif feature_id == "enforce_eager":
            od_config = _make_od_config(enforce_eager=True)
        elif feature_id == "unsupported_quantization":
            od_config = _make_od_config(quantization_config=SimpleNamespace(quant_method="gguf"))
        else:
            raise ValueError(f"Unknown feature ID: {feature_id}")

        with pytest.raises(NotImplementedError):
            DiffusersAdapterPipeline(od_config=od_config)

    def test_adapter_load_weights_injects_supported_quantization_config(self, mocker):
        (
            FakePipelineQuantizationConfig,
            FakeTorchAoConfig,
            FakeInt8DynamicActivationInt8WeightConfig,
        ) = _patch_fake_diffusers_quantization_backends(mocker)

        class MockPipeline:
            def __call__(self, prompt=None):
                return None

            def to(self, device):
                return self

        mock_from_pretrained = mocker.patch(
            "vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter.DiffusionPipeline.from_pretrained",
            return_value=MockPipeline(),
        )
        mocker.patch(
            "vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter.DiffusionPipeline.load_config",
            return_value={
                "_class_name": "TransformerBackedPipeline",
                "scheduler": ["diffusers", "SchedulerMixin"],
                "transformer": ["diffusers", "Transformer2DModel"],
                "transformer_2": ["diffusers", "Transformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            },
        )

        adapter = DiffusersAdapterPipeline(
            od_config=_make_od_config(
                quantization_config=SimpleNamespace(quant_method="int8"),
                diffusers_pipeline_cls=_TransformerBackedPipeline,
            )
        )
        adapter.load_weights()

        pipeline_quant_config = mock_from_pretrained.call_args.kwargs["quantization_config"]
        torchao_config = pipeline_quant_config.quant_mapping["transformer"]
        assert isinstance(pipeline_quant_config, FakePipelineQuantizationConfig)
        assert isinstance(torchao_config, FakeTorchAoConfig)
        assert isinstance(torchao_config.quant_type, FakeInt8DynamicActivationInt8WeightConfig)
        assert isinstance(
            pipeline_quant_config.quant_mapping["transformer_2"].quant_type,
            FakeInt8DynamicActivationInt8WeightConfig,
        )

    @pytest.mark.parametrize(
        ("pipeline_config", "diffusers_load_kwargs", "diffusers_pipeline_cls"),
        [
            (
                {
                    "_class_name": "NoTransformerPipeline",
                    "scheduler": ["diffusers", "SchedulerMixin"],
                    "unet": ["diffusers", "UNet2DConditionModel"],
                    "vae": ["diffusers", "AutoencoderKL"],
                },
                {},
                _TransformerBackedPipeline,
            ),
            (
                {
                    "_class_name": "TransformerPipeline",
                    "scheduler": ["diffusers", "SchedulerMixin"],
                    "transformer": ["diffusers", "Transformer2DModel"],
                    "vae": ["diffusers", "AutoencoderKL"],
                },
                {"transformer": None},
                None,
            ),
        ],
    )
    def test_adapter_load_weights_rejects_converted_quantization_without_transformer_component(
        self,
        pipeline_config,
        diffusers_load_kwargs,
        diffusers_pipeline_cls,
        mocker,
    ):
        _patch_fake_diffusers_quantization_backends(mocker)
        mocker.patch(
            "vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter.DiffusionPipeline.load_config",
            return_value=pipeline_config,
        )
        mock_from_pretrained = mocker.patch(
            "vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter.DiffusionPipeline.from_pretrained"
        )

        adapter = DiffusersAdapterPipeline(
            od_config=_make_od_config(
                quantization_config=SimpleNamespace(quant_method="int8"),
                diffusers_load_kwargs=diffusers_load_kwargs,
                diffusers_pipeline_cls=diffusers_pipeline_cls,
            )
        )

        with pytest.raises(NotImplementedError, match="transformer"):
            adapter.load_weights()
        mock_from_pretrained.assert_not_called()

    def test_adapter_rejects_dduf_file_with_converted_quantization(self):
        with pytest.raises(NotImplementedError, match="dduf_file"):
            DiffusersAdapterPipeline(
                od_config=_make_od_config(
                    quantization_config=SimpleNamespace(quant_method="int8"),
                    diffusers_load_kwargs={"dduf_file": "model.dduf"},
                )
            )

    def test_adapter_load_weights_preserves_diffusers_native_quantization_config(self, mocker):
        mocker.patch(
            "vllm_omni.diffusion.models.diffusers_adapter.dduf.diffusers_version",
            "0.40.0",
        )

        class MockPipeline:
            def __call__(self, prompt=None):
                return None

            def to(self, device):
                return self

        diffusers_native_quantization_config = object()
        mock_from_pretrained = mocker.patch(
            "vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter.DiffusionPipeline.from_pretrained",
            return_value=MockPipeline(),
        )

        adapter = DiffusersAdapterPipeline(
            od_config=_make_od_config(
                quantization_config=SimpleNamespace(quant_method="gguf"),
                diffusers_load_kwargs={
                    "dduf_file": "model.dduf",
                    "quantization_config": diffusers_native_quantization_config,
                },
            )
        )
        adapter.load_weights()

        kwargs = mock_from_pretrained.call_args.kwargs
        assert kwargs["quantization_config"] is diffusers_native_quantization_config
        assert kwargs["dduf_file"] == "model.dduf"

    def test_adapter_load_weights_builds_diffusers_native_quantization_config_from_dict(self, mocker):
        FakePipelineQuantizationConfig, FakeTorchAoConfig, _ = _patch_fake_diffusers_quantization_backends(mocker)

        class MockPipeline:
            def __call__(self, prompt=None):
                return None

            def to(self, device):
                return self

        mock_from_pretrained = mocker.patch(
            "vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter.DiffusionPipeline.from_pretrained",
            return_value=MockPipeline(),
        )

        adapter = DiffusersAdapterPipeline(
            od_config=_make_od_config(
                diffusers_load_kwargs={
                    "quantization_config": {
                        "quant_mapping": {
                            "transformer": {
                                "quant_method": "torchao",
                                "quant_type": "fake-int8-config",
                            },
                        },
                    },
                },
            )
        )
        adapter.load_weights()

        quant_config = mock_from_pretrained.call_args.kwargs["quantization_config"]
        assert isinstance(quant_config, FakePipelineQuantizationConfig)
        assert isinstance(quant_config.quant_mapping["transformer"], FakeTorchAoConfig)
        assert quant_config.quant_mapping["transformer"].quant_type == "fake-int8-config"

    def test_adapter_guard_unknown_output_type(self, mocker):
        """Test that the adapter wraps an unknown output type as-is.
        This is useful when `return_dict=True` and the diffusers pipeline returns an OrderedDict subclass."""

        adapter = DiffusersAdapterPipeline(od_config=_make_od_config())
        raw_output = {"unexpected": "dict-output"}

        MockPipeline = type("MockPipeline", (DiffusionPipeline,), {})
        adapter._pipeline = MockPipeline()

        mocker.patch.object(
            MockPipeline,
            "__call__",
            return_value=raw_output,
        )
        output = adapter.forward(_make_batch())

        assert isinstance(output, DiffusionOutput)
        assert output.output == raw_output

    def test_adapter_build_call_kwargs(self, mocker):
        class MockPipeline:
            def __call__(
                self,
                prompt=None,
                negative_prompt=None,
                num_inference_steps=None,
                # guidance_scale=None, # deliberately not included
                height=None,
                width=None,
                num_frames=None,
                num_images_per_prompt=None,
                num_videos_per_prompt=None,
                output_type=None,
                generator=None,
            ):
                """Need to make the signature match the actual DiffusionPipeline.__call__, because inspect.signature() is used"""
                return None

            def to(self, device):
                return self

        mock_from_pretrained = mocker.patch(
            "vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter.DiffusionPipeline.from_pretrained",
            return_value=MockPipeline(),
        )

        adapter = DiffusersAdapterPipeline(
            od_config=_make_od_config(
                diffusers_call_kwargs={
                    "guidance_scale": 1.25,
                    "eta": 0.3,
                    "output_type": "np",
                }
            )
        )

        adapter.load_weights()
        mock_from_pretrained.assert_called_once()

        req = _make_request(
            prompt="a cat on mars",
            negative_prompt="low quality",
            sampling_params=OmniDiffusionSamplingParams(
                num_inference_steps=9,
                guidance_scale=8.0,
                height=320,
                width=640,
                num_frames=8,
                num_outputs_per_prompt=2,
                seed=123,
                output_type="pil",
            ),
        )

        kwargs = adapter._build_call_kwargs(DiffusionRequestBatch(requests=[req]))

        assert kwargs["prompt"] == "a cat on mars"
        assert kwargs["negative_prompt"] == "low quality"
        assert kwargs["num_inference_steps"] == 9
        assert "guidance_scale" not in kwargs
        assert kwargs["height"] == 320
        assert kwargs["width"] == 640
        assert kwargs["num_frames"] == 8
        assert kwargs["num_images_per_prompt"] == 2
        assert kwargs["num_videos_per_prompt"] == 2
        assert kwargs["output_type"] == "pil"
        assert isinstance(kwargs["generator"], torch.Generator)
        assert kwargs["generator"].device.type == "cpu"
        assert kwargs["generator"].initial_seed() == 123

    def test_adapter_extract_input_reads_negative_prompt_fallback(self):
        # 1. test with diffusers_call_kwargs.negative_prompt is a correct list[str]
        adapter = DiffusersAdapterPipeline(
            od_config=_make_od_config(
                diffusers_call_kwargs={
                    "negative_prompt": [
                        "fallback negative prompt 0",
                        "fallback negative prompt 1",
                        "fallback negative prompt 2",
                    ],
                }
            )
        )
        input_kwargs = adapter._extract_input(
            [
                "a prompt from a string",
                {"prompt": "a prompt from a dict"},
                {
                    "prompt": "a prompt with its own negative prompt",
                    "negative_prompt": "request negative prompt",
                },
            ]
        )
        assert input_kwargs["prompt"] == [
            "a prompt from a string",
            "a prompt from a dict",
            "a prompt with its own negative prompt",
        ]
        assert input_kwargs["negative_prompt"] == [
            "fallback negative prompt 0",
            "fallback negative prompt 1",
            "request negative prompt",
        ]

        # 2. test with diffusers_call_kwargs.negative_prompt is a single str
        adapter = DiffusersAdapterPipeline(
            od_config=_make_od_config(diffusers_call_kwargs={"negative_prompt": "fallback negative prompt"})
        )
        input_kwargs = adapter._extract_input(
            [
                "a prompt from a string",
                {"prompt": "a prompt from a dict"},
                {
                    "prompt": "a prompt with its own negative prompt",
                    "negative_prompt": "request negative prompt",
                },
            ]
        )
        assert input_kwargs["prompt"] == [
            "a prompt from a string",
            "a prompt from a dict",
            "a prompt with its own negative prompt",
        ]
        assert input_kwargs["negative_prompt"] == [
            "fallback negative prompt",
            "fallback negative prompt",
            "request negative prompt",
        ]

        # 3. test with diffusers_call_kwargs.negative_prompt is None
        adapter = DiffusersAdapterPipeline(od_config=_make_od_config(diffusers_call_kwargs={}))
        input_kwargs = adapter._extract_input(
            [
                "a prompt from a string",
                {"prompt": "a prompt from a dict"},
                {
                    "prompt": "a prompt with its own negative prompt",
                    "negative_prompt": "request negative prompt",
                },
            ]
        )
        assert input_kwargs["prompt"] == [
            "a prompt from a string",
            "a prompt from a dict",
            "a prompt with its own negative prompt",
        ]
        assert input_kwargs["negative_prompt"] == ["", "", "request negative prompt"]

        # 4. test with diffusers_call_kwargs.negative_prompt is a list[str] but its length is less than the number of prompts
        adapter = DiffusersAdapterPipeline(
            od_config=_make_od_config(
                diffusers_call_kwargs={
                    "negative_prompt": [
                        "fallback negative prompt 0",
                    ]
                }
            )
        )
        with pytest.raises(ValueError):
            adapter._extract_input(
                [
                    "a prompt from a string",
                    {"prompt": "a prompt from a dict"},
                    {
                        "prompt": "a prompt with its own negative prompt",
                        "negative_prompt": "request negative prompt",
                    },
                ]
            )

        # 5. test with diffusers_call_kwargs.negative_prompt is a list[str] but its length is greater than the number of prompts
        adapter = DiffusersAdapterPipeline(
            od_config=_make_od_config(
                diffusers_call_kwargs={
                    "negative_prompt": [
                        "fallback negative prompt 0",
                        "fallback negative prompt 1",
                        "fallback negative prompt 2",
                        "fallback negative prompt 3",
                    ]
                }
            )
        )
        input_kwargs = adapter._extract_input(
            [
                "a prompt from a string",
                {"prompt": "a prompt from a dict"},
                {
                    "prompt": "a prompt with its own negative prompt",
                    "negative_prompt": "request negative prompt",
                },
            ]
        )
        assert input_kwargs["prompt"] == [
            "a prompt from a string",
            "a prompt from a dict",
            "a prompt with its own negative prompt",
        ]
        assert input_kwargs["negative_prompt"] == [
            "fallback negative prompt 0",
            "fallback negative prompt 1",
            "request negative prompt",
        ]

        # 6. test when no negative prompt is provided
        adapter = DiffusersAdapterPipeline(od_config=_make_od_config())
        input_kwargs = adapter._extract_input(
            [
                "a prompt from a string",
                {"prompt": "a prompt from a dict"},
            ]
        )
        assert input_kwargs["prompt"] == [
            "a prompt from a string",
            "a prompt from a dict",
        ]
        assert "negative_prompt" not in input_kwargs

    def test_adapter_load_weights_uses_registered_pipeline_utils(self, mocker):
        class WanImageToVideoPipeline:
            pass

        class MockPipeline:
            def __call__(self, prompt=None):
                return None

            def to(self, device):
                return self

        od_config = _make_od_config(
            diffusers_pipeline_cls=WanImageToVideoPipeline,
            boundary_ratio=0.875,
        )
        mock_from_pretrained = mocker.patch(
            "vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter.DiffusionPipeline.from_pretrained",
            return_value=MockPipeline(),
        )

        pipeline = DiffusersAdapterPipeline(od_config=od_config)
        pipeline.load_weights()

        mock_from_pretrained.assert_called_once()
        _, kwargs = mock_from_pretrained.call_args
        assert kwargs["boundary_ratio"] == 0.875

        problematic_request = _make_request(
            prompt="a prompt from a string",
            sampling_params=OmniDiffusionSamplingParams(
                boundary_ratio=0.875,
            ),
        )
        with pytest.raises(ValueError):
            pipeline.forward(DiffusionRequestBatch(requests=[problematic_request]))

        interpolation_request = _make_request(
            sampling_params=OmniDiffusionSamplingParams(
                enable_frame_interpolation=True,
            ),
        )
        with pytest.raises(ValueError, match="Frame interpolation is not supported with the Diffusers backend"):
            pipeline.forward(DiffusionRequestBatch(requests=[interpolation_request]))

    def test_diffusers_config_preserves_checkpoint_class_for_metadata(self, mocker):
        class MockWanImageToVideoPipeline:
            pass

        od_config = _make_od_config(model_class_name=None)
        mocker.patch(
            "vllm_omni.diffusion.utils.hf_utils.get_diffusion_model_index",
            return_value={"_class_name": "WanImageToVideoPipeline"},
        )
        mocker.patch.object(
            diffusion_data,
            "diffusers",
            SimpleNamespace(WanImageToVideoPipeline=MockWanImageToVideoPipeline),
        )
        mocker.patch.object(OmniDiffusionConfig, "update_multimodal_support")

        od_config.enrich_config()

        assert od_config.model_class_name == "WanImageToVideoPipeline"
        assert od_config.diffusers_pipeline_cls is MockWanImageToVideoPipeline

    def test_diffusers_config_falls_back_to_adapter_without_pipeline_index(self, mocker):
        od_config = _make_od_config(model_class_name=None)
        mocker.patch(
            "vllm_omni.diffusion.utils.hf_utils.get_diffusion_model_index",
            return_value=None,
        )

        od_config.enrich_config()

        assert od_config.model_class_name == "DiffusersAdapterPipeline"


@pytest.mark.core_model
@pytest.mark.cpu
class TestDDUFCompatibility:
    @pytest.fixture
    def bridge(self, tmp_path, monkeypatch):
        from vllm_omni.diffusion.models.diffusers_adapter import dduf

        monkeypatch.setattr(dduf, "diffusers_version", "0.42.0")
        monkeypatch.setattr(dduf.tempfile, "tempdir", str(tmp_path))
        return dduf

    @pytest.fixture
    def archive_path(self, tmp_path):
        from huggingface_hub import export_entries_as_dduf

        path = tmp_path / "model.dduf"
        export_entries_as_dduf(
            path,
            [("model_index.json", b"{}"), ("weights.safetensors", b"tiny fixture")],
        )
        return path

    @pytest.mark.parametrize("version,dduf_file", [("0.40.0", "model.dduf"), ("0.42.0", None), ("0.42.0", "")])
    def test_legacy_and_non_dduf_loading_preserve_arguments(self, bridge, mocker, version, dduf_file):
        bridge.diffusers_version = version
        loader = mocker.Mock(spec=DiffusionPipeline)
        kwargs = {"dduf_file": dduf_file, "revision": "frozen", "local_files_only": True}

        result = bridge.load_diffusers_pipeline(loader, "test/model", kwargs)

        expected = kwargs if version == "0.40.0" else {k: v for k, v in kwargs.items() if k != "dduf_file"}
        loader.from_pretrained.assert_called_once_with("test/model", **expected)
        assert result is loader.from_pretrained.return_value
        assert kwargs["dduf_file"] == dduf_file

    def test_download_streaming_and_directory_lifetime(self, bridge, archive_path, tmp_path, mocker):
        api = mocker.Mock(spec=["hf_hub_download"])
        api.hf_hub_download.return_value = str(archive_path)
        mocker.patch.object(bridge, "hf_api", return_value=api)
        loader = mocker.Mock(spec=DiffusionPipeline)
        loader.from_pretrained.side_effect = lambda *args, **kwargs: DiffusionPipeline()
        copy = mocker.spy(bridge.shutil, "copyfileobj")
        kwargs = {
            "dduf_file": "model.dduf",
            "cache_dir": str(tmp_path / "cache"),
            "revision": "frozen",
            "token": "test-token",
            "local_files_only": False,
            "force_download": False,
            "subfolder": "archives",
            "torch_dtype": torch.float32,
        }

        pipeline = bridge.load_diffusers_pipeline(loader, "test/model", kwargs)

        download_kwargs = api.hf_hub_download.call_args.kwargs
        assert download_kwargs == {
            "repo_id": "test/model",
            "filename": "model.dduf",
            **{
                key: kwargs[key]
                for key in ("cache_dir", "revision", "token", "local_files_only", "force_download", "subfolder")
            },
        }
        directory = Path(loader.from_pretrained.call_args.args[0])
        assert (directory / "weights.safetensors").read_bytes() == b"tiny fixture"
        assert loader.from_pretrained.call_args.kwargs == {
            **{k: v for k, v in kwargs.items() if k != "dduf_file"},
            "local_files_only": True,
        }
        assert kwargs["local_files_only"] is False
        assert copy.call_count == 2
        assert all(0 < call.kwargs["length"] <= 1024 * 1024 for call in copy.call_args_list)
        assert directory.is_dir()
        del pipeline
        gc.collect()
        assert not directory.exists()

    @pytest.mark.parametrize("failure", ["loader", "post_load", "to"])
    def test_failed_load_cleans_materialized_directory(self, bridge, archive_path, tmp_path, mocker, failure):
        load = mocker.patch.object(
            DiffusionPipeline, "from_pretrained", side_effect=lambda *args, **kwargs: DiffusionPipeline()
        )
        if failure == "loader":
            load.side_effect = RuntimeError("fixture load failure")
        elif failure == "post_load":
            mocker.patch(
                "vllm_omni.diffusion.models.diffusers_adapter.pipeline_utils.BasePipelineUtils.apply_post_load_updates",
                side_effect=RuntimeError("fixture load failure"),
            )
        else:
            mocker.patch.object(DiffusionPipeline, "to", side_effect=RuntimeError("fixture load failure"))
        adapter = DiffusersAdapterPipeline(
            od_config=_make_od_config(model=str(tmp_path), diffusers_load_kwargs={"dduf_file": archive_path.name}),
            device=torch.device("cpu"),
        )

        with pytest.raises(RuntimeError, match="fixture load failure"):
            adapter.load_weights()

        assert not list(tmp_path.glob("vllm-omni-dduf-*"))

    def test_failed_reload_preserves_live_pipeline_directory(self, bridge, archive_path, tmp_path, mocker):
        loader = mocker.Mock(spec=DiffusionPipeline)
        loader.from_pretrained.side_effect = lambda *args, **kwargs: DiffusionPipeline()
        pipeline = bridge.load_diffusers_pipeline(loader, str(tmp_path), {"dduf_file": archive_path.name})
        directory = Path(loader.from_pretrained.call_args.args[0])
        adapter = DiffusersAdapterPipeline(
            od_config=_make_od_config(model=str(tmp_path), diffusers_load_kwargs={"dduf_file": "missing.dduf"}),
            device=torch.device("cpu"),
        )
        adapter._pipeline = pipeline

        with pytest.raises(FileNotFoundError):
            adapter.load_weights()

        assert adapter._pipeline is pipeline
        assert directory.is_dir()
        bridge.cleanup_dduf_pipeline(pipeline)
        assert not directory.exists()

    @pytest.mark.parametrize("malformed", ["traversal", "absolute", "duplicate", "symlink", "compressed", "crc"])
    def test_invalid_archive_is_rejected_without_writes(self, bridge, archive_path, tmp_path, mocker, malformed):
        from huggingface_hub.errors import DDUFCorruptedFileError

        model_index = {"..": ["diffusers", "UNet2DModel"]} if malformed == "traversal" else {}
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("model_index.json", json.dumps(model_index))
            if malformed == "traversal":
                archive.writestr("../config.json", "{}")
                archive.writestr("../escape.txt", "escape")
            elif malformed == "absolute":
                archive.writestr(str(tmp_path / "escape.txt"), "escape")
            else:
                info = zipfile.ZipInfo("weights.safetensors")
                if malformed == "symlink":
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                if malformed == "compressed":
                    info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, b"tiny fixture")
                if malformed == "duplicate":
                    with pytest.warns(UserWarning, match="Duplicate name"):
                        archive.writestr("weights.safetensors", b"duplicate")
        if malformed == "crc":
            entry = bridge.read_dduf_file(archive_path)["weights.safetensors"]
            with archive_path.open("r+b") as archive:
                archive.seek(entry.offset)
                archive.write(b"X")
        loader = mocker.Mock(spec=DiffusionPipeline)

        with pytest.raises((ValueError, DDUFCorruptedFileError, zipfile.BadZipFile)):
            bridge.load_diffusers_pipeline(loader, str(tmp_path), {"dduf_file": archive_path.name})

        loader.from_pretrained.assert_not_called()
        assert not (tmp_path / "escape.txt").exists()
        assert not list(tmp_path.glob("vllm-omni-dduf-*"))

    def test_insufficient_disk_space_fails_before_copy(self, bridge, archive_path, tmp_path, mocker):
        usage = bridge.shutil.disk_usage(tmp_path)
        mocker.patch.object(bridge.shutil, "disk_usage", return_value=usage._replace(free=0))
        copy = mocker.spy(bridge.shutil, "copyfileobj")
        loader = mocker.Mock(spec=DiffusionPipeline)

        with pytest.raises(OSError) as error:
            bridge.load_diffusers_pipeline(loader, str(tmp_path), {"dduf_file": archive_path.name})

        assert error.value.errno == errno.ENOSPC
        copy.assert_not_called()
        loader.from_pretrained.assert_not_called()
        assert not list(tmp_path.glob("vllm-omni-dduf-*"))

    @pytest.mark.parametrize("unsupported", ["custom_pipeline", "load_connected_pipeline"])
    def test_unsupported_pipeline_mode_remains_rejected(self, bridge, archive_path, tmp_path, unsupported):
        with pytest.raises(NotImplementedError, match="not supported with DDUF"):
            bridge.load_diffusers_pipeline(
                DiffusionPipeline, str(tmp_path), {"dduf_file": archive_path.name, unsupported: True}
            )
        assert not list(tmp_path.glob("vllm-omni-dduf-*"))

    def test_real_diffusers_loads_dduf_weights_on_cpu(self, tmp_path, monkeypatch):
        from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
        from huggingface_hub import export_folder_as_dduf
        from packaging.version import Version

        from vllm_omni.diffusion.models.diffusers_adapter import dduf

        if Version(dduf.diffusers_version) < Version("0.42.0"):
            pytest.skip("Actual new directory loader requires Diffusers >=0.42.0")
        monkeypatch.setattr(dduf.tempfile, "tempdir", str(tmp_path))
        unet = UNet2DModel(
            sample_size=8,
            in_channels=3,
            out_channels=3,
            layers_per_block=1,
            block_out_channels=(8,),
            down_block_types=("DownBlock2D",),
            up_block_types=("UpBlock2D",),
            norm_num_groups=4,
        )
        original = DDPMPipeline(unet=unet, scheduler=DDPMScheduler(num_train_timesteps=4))
        source = tmp_path / "original"
        original.save_pretrained(source, safe_serialization=True)
        archive_path = tmp_path / "tiny.dduf"
        export_folder_as_dduf(archive_path, source)

        loaded = dduf.load_diffusers_pipeline(
            DiffusionPipeline, str(tmp_path), {"dduf_file": archive_path.name, "local_files_only": True}
        )

        assert isinstance(loaded, DDPMPipeline)
        assert loaded.unet.device == torch.device("cpu")
        assert loaded.scheduler.config.num_train_timesteps == 4
        assert loaded.unet.state_dict().keys() == unet.state_dict().keys()
        for name, tensor in unet.state_dict().items():
            assert torch.equal(loaded.unet.state_dict()[name], tensor), name
        directories = list(tmp_path.glob("vllm-omni-dduf-*"))
        assert len(directories) == 1
        del loaded
        gc.collect()
        assert not directories[0].exists()


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
class TestDiffusersBackendEndToEndExecution:
    @pytest.mark.parametrize(
        "omni_server",
        [
            OmniServerParams(
                model="tiny-random/Qwen-Image",
                server_args=[
                    "--diffusion-load-format",
                    "diffusers",
                    "--diffusers-call-kwargs",
                    '{"height": 512, "width": 0}',  # deliberately weird width to be overridden
                ],
            ),
        ],
        indirect=True,
    )
    def test_t2i_random_weights(
        self,
        omni_server: OmniServer,
        online_client: OnlineOmniClient,
    ):
        messages = dummy_messages_from_mix_data(content_text="a photo of an astronaut riding a horse on mars")

        request_config = {
            "model": omni_server.model,
            "messages": messages,
            "extra_body": {
                "width": 512,
                "num_inference_steps": 2,
                "negative_prompt": "blurry",
                "true_cfg_scale": 4.0,
                "seed": 42,
            },
        }

        response = online_client.send_diffusion_request(request_config)
        images = response[0].images
        assert images is not None
        image: Image.Image = images[0]

        # Request config has incomplete width/height, so internal assertion in `send_diffusion_request` is incomplete.
        assert image.size == (512, 512)
