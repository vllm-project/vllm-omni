# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch
from diffusers import DiffusionPipeline  # pyright: ignore[reportPrivateImportUsage]
from diffusers.models.modeling_utils import ModelMixin
from PIL import Image
from torch import nn

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler, dummy_messages_from_mix_data
from vllm_omni.diffusion.data import (
    DiffusionOutput,
    DiffusionParallelConfig,
    OmniDiffusionConfig,
)
from vllm_omni.diffusion.models.diffusers_adapter import DiffusersAdapterPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
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
        "prompts": [prompt_obj],
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


def _make_adapter(**attrs) -> DiffusersAdapterPipeline:
    """Create a dummy adapter for testing HSDP validation."""
    adapter = object.__new__(DiffusersAdapterPipeline)
    nn.Module.__init__(adapter)
    adapter.is_initialized = False
    for k, v in attrs.items():
        setattr(adapter, k, v)
    return adapter


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
        adapter.is_initialized = True

        mocker.patch.object(
            MockPipeline,
            "__call__",
            return_value=MockPipelineOutput(image=stub_image),
        )
        output = adapter.forward(request)

        assert isinstance(output, DiffusionOutput)
        assert isinstance(output.output, MockPipelineOutput)
        assert output.output.image is stub_image

    @pytest.mark.parametrize(
        "feature_id",
        ["cfg_parallel", "ulysses", "ring", "teacache", "cache_dit", "quantization"],
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
        elif feature_id == "quantization":
            od_config = _make_od_config(quantization_config=SimpleNamespace(quant_method="fp8"))
        else:
            raise ValueError(f"Unknown feature ID: {feature_id}")

        with pytest.raises(NotImplementedError):
            DiffusersAdapterPipeline(od_config=od_config)

    def test_adapter_guard_unknown_output_type(self, mocker):
        """Test that the adapter wraps an unknown output type as-is.
        This is useful when `return_dict=True` and the diffusers pipeline returns an OrderedDict subclass."""

        adapter = DiffusersAdapterPipeline(od_config=_make_od_config())
        raw_output = {"unexpected": "dict-output"}

        MockPipeline = type("MockPipeline", (DiffusionPipeline,), {})
        adapter._pipeline = MockPipeline()
        adapter.is_initialized = True

        mocker.patch.object(
            MockPipeline,
            "__call__",
            return_value=raw_output,
        )
        output = adapter.forward(_make_request())

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

        kwargs = adapter._build_call_kwargs(req)

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
            pipeline.forward(problematic_request)


@pytest.mark.core_model
@pytest.mark.cpu
class TestDiffusersAdapterHSDP:
    def test_hsdp_matches_only_direct_modulelist_names(self):
        """Ensure we match only names that are direct child Modulelist-like."""
        condition = DiffusersAdapterPipeline._build_hsdp_condition({"blocks", "layers"})
        stub = nn.Module()
        assert condition("blocks.0", stub) is True
        assert condition("layers.2", stub) is True
        assert condition("other.0", stub) is False
        assert condition("blocks.0.sublayer.1", stub) is False
        assert condition("blocks.name", stub) is False

    def test_set_dit_hsdp_conditions_sets_shard_conditions_on_transformer(self):
        """Ensure set DiT hsdp conditions matches direct child module lists."""
        transformer = nn.Module()
        transformer.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

        adapter = _make_adapter(is_initialized=True, _transformer=transformer, _transformer_2=None)
        adapter._set_dit_hsdp_conditions()

        conditions = transformer._hsdp_shard_conditions
        matched = [name for name, mod in transformer.named_modules() if any(c(name, mod) for c in conditions)]
        assert matched == ["blocks.0", "blocks.1", "blocks.2"]

    def test_hsdp_conditions_with_dual_transformers(self):
        """Ensure set DiT hsdp conditions matches direct child module lists correctly
        if we have 2 DiTs with potentially different architectures."""
        transformer = nn.Module()
        transformer.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        # Name dual transformer block something else, since the conditions are set at the DiT level
        other_transformer = nn.Module()
        other_transformer.t_blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

        adapter = _make_adapter(is_initialized=True, _transformer=transformer, _transformer_2=other_transformer)
        adapter._set_dit_hsdp_conditions()

        # The first DiT will match blocks.<x>
        conditions = transformer._hsdp_shard_conditions
        matched = [name for name, mod in transformer.named_modules() if any(c(name, mod) for c in conditions)]
        assert matched == ["blocks.0", "blocks.1", "blocks.2"]

        # The secondary DiT will match t_blocks.<x>
        other_conditions = other_transformer._hsdp_shard_conditions
        matched = [
            name for name, mod in other_transformer.named_modules() if any(c(name, mod) for c in other_conditions)
        ]
        assert matched == ["t_blocks.0", "t_blocks.1", "t_blocks.2"]

    def test_set_dit_hsdp_conditions_raises_without_transformer(self):
        """Ensure set DiT hsdp conditions needs a DiT."""
        adapter = _make_adapter(is_initialized=True, _transformer=None, _transformer_2=None)
        with pytest.raises(RuntimeError, match="HSDP requires an encapsulated DiT"):
            adapter._set_dit_hsdp_conditions()

    @pytest.mark.parametrize("attr_name", ["pipeline", "transformer"])
    def test_pipeline_property_guards_before_load_weights(self, attr_name):
        """Ensure that attributes that require loading raise if accessed too early."""
        adapter = _make_adapter()
        with pytest.raises(RuntimeError):
            getattr(adapter, attr_name)


class TestDiffusersAttentionInitialization:
    @pytest.mark.parametrize(
        "transformer_attrs",
        (["transformer"], ["transformer", "transformer_2"]),
    )
    def test_attn_backend(self, transformer_attrs, mocker):
        adapter = DiffusersAdapterPipeline(od_config=_make_od_config())

        class MockTransformer(ModelMixin):
            def __init__(self, name):
                self.name = name

            def __eq__(self, other):
                # This is just added for readability to ensure we are passing the right DiT
                return self is other

        class MockPipeline:
            def __init__(self):
                for attr_name in transformer_attrs:
                    fake_transformer = MockTransformer(name=attr_name)
                    setattr(self, attr_name, fake_transformer)

            def __call__(self, *args, **kwargs):
                return None

            def to(self, *args, **kwargs):
                return self

        mock_from_pretrained = mocker.patch(
            "vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter.DiffusionPipeline.from_pretrained",
            return_value=MockPipeline(),
        )
        mock_set_attn = mocker.patch(
            "vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter.DiffusersAdapterPipeline._set_attention_backend",
        )

        adapter.load_weights()
        mock_from_pretrained.assert_called_once()
        assert adapter.is_initialized
        for attr_name in transformer_attrs:
            # Ensure that we tried to set the attention backend for each defined DiT
            fake_dit = getattr(adapter.pipeline, attr_name)
            mock_set_attn.assert_any_call(fake_dit)


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
