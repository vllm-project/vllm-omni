# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch
from diffusers import DiffusionPipeline  # pyright: ignore[reportPrivateImportUsage]
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler, dummy_messages_from_mix_data
from vllm_omni.diffusion.data import (
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

    @pytest.mark.parametrize(
        "feature_id",
        ["cfg_parallel", "ulysses", "ring", "teacache", "cache_dit", "enforce_eager", "unsupported_quantization"],
    )
    def test_adapter_guard_unsupported_feature(self, feature_id):
        if feature_id == "cfg_parallel":
            od_config = _make_od_config(
                parallel_config=DiffusionParallelConfig(cfg_parallel_size=2, sequence_parallel_size=1),
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
        openai_client: OpenAIClientHandler,
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

        response = openai_client.send_diffusion_request(request_config)
        image: Image.Image = response[0].images[0]  # pyright: ignore[reportOptionalSubscript]

        # Request config has incomplete width/height, so internal assertion in `send_diffusion_request` is incomplete.
        assert image.size == (512, 512)
