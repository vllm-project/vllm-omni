from types import SimpleNamespace

import PIL.Image
import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    WAN22_MAX_SEQUENCE_LENGTH,
    Wan22Pipeline,
)
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v import (
    Wan22I2VPipeline,
)
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_ti2v import (
    Wan22TI2VPipeline,
)
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_vace import (
    Wan22VACEPipeline,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _RejectingTextEncoder:
    dtype = torch.float32

    def __call__(self, *args, **kwargs):
        raise AssertionError("text encoder should not run for prompts that exceed max_sequence_length")


class _FakeTokenBatch:
    def __init__(self, total_sequence_length: int):
        attention_mask = torch.ones((1, total_sequence_length), dtype=torch.long)
        self.input_ids = attention_mask.clone()
        self.attention_mask = attention_mask


class _FakeTokenizer:
    def __init__(self, total_sequence_length: int):
        self.total_sequence_length = total_sequence_length

    def __call__(self, *args, **kwargs):
        return _FakeTokenBatch(self.total_sequence_length)


PIPELINE_CASES = [
    pytest.param(Wan22Pipeline, id="wan22-t2v"),
    pytest.param(Wan22I2VPipeline, id="wan22-i2v"),
    pytest.param(Wan22TI2VPipeline, id="wan22-ti2v"),
    pytest.param(Wan22VACEPipeline, id="wan22-vace"),
]


def _make_pipeline(pipeline_class: type, *, total_sequence_length: int):
    pipeline = object.__new__(pipeline_class)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.text_encoder = _RejectingTextEncoder()
    pipeline.tokenizer = _FakeTokenizer(total_sequence_length)
    pipeline.tokenizer_max_length = WAN22_MAX_SEQUENCE_LENGTH
    return pipeline


@pytest.mark.parametrize("pipeline_class", PIPELINE_CASES)
def test_encode_prompt_rejects_prompt_longer_than_default_max_sequence_length(pipeline_class: type):
    pipeline = _make_pipeline(pipeline_class, total_sequence_length=WAN22_MAX_SEQUENCE_LENGTH + 1)

    with pytest.raises(ValueError, match=r"got 513 tokens, but `max_sequence_length` is 512"):
        pipeline.encode_prompt(prompt="prompt")


@pytest.mark.parametrize("pipeline_class", PIPELINE_CASES)
def test_encode_prompt_rejects_prompt_longer_than_explicit_max_sequence_length(pipeline_class: type):
    pipeline = _make_pipeline(pipeline_class, total_sequence_length=17)

    with pytest.raises(ValueError, match=r"got 17 tokens, but `max_sequence_length` is 16"):
        pipeline.encode_prompt(prompt="prompt", max_sequence_length=16)


def _sampling_params(**overrides):
    defaults = dict(
        height=None,
        width=None,
        num_frames=None,
        num_inference_steps=None,
        generator=None,
        guidance_scale_provided=False,
        guidance_scale_2=None,
        boundary_ratio=None,
        num_outputs_per_prompt=0,
        max_sequence_length=None,
        seed=None,
        extra_args={},
        prompt_embeds=None,
        negative_prompt_embeds=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.parametrize(
    ("pipeline_class", "prompt_value", "forward_kwargs"),
    [
        pytest.param(Wan22Pipeline, "prompt", {}, id="wan22-t2v"),
        pytest.param(
            Wan22I2VPipeline,
            {"prompt": "prompt", "multi_modal_data": {"image": PIL.Image.new("RGB", (64, 64))}},
            {"image": PIL.Image.new("RGB", (64, 64))},
            id="wan22-i2v",
        ),
        pytest.param(
            Wan22TI2VPipeline,
            {"prompt": "prompt", "multi_modal_data": {"image": PIL.Image.new("RGB", (64, 64))}},
            {"image": PIL.Image.new("RGB", (64, 64))},
            id="wan22-ti2v",
        ),
        pytest.param(Wan22VACEPipeline, "prompt", {}, id="wan22-vace"),
    ],
)
def test_forward_defaults_to_wan22_tokenizer_max_length(
    pipeline_class: type,
    prompt_value,
    forward_kwargs,
):
    pipeline = object.__new__(pipeline_class)
    nn.Module.__init__(pipeline)
    pipeline.tokenizer_max_length = WAN22_MAX_SEQUENCE_LENGTH
    pipeline.boundary_ratio = None
    pipeline.vae_scale_factor_temporal = 4
    pipeline.vae_scale_factor_spatial = 8
    pipeline.transformer_config = SimpleNamespace(patch_size=(1, 2, 2))

    captured = {}

    def _fake_check_inputs(*args, **kwargs):
        captured["max_sequence_length"] = kwargs["max_sequence_length"]
        raise RuntimeError("stop after capture")

    pipeline.check_inputs = _fake_check_inputs

    req = SimpleNamespace(
        prompts=[prompt_value],
        sampling_params=_sampling_params(),
    )

    with pytest.raises(RuntimeError, match="stop after capture"):
        pipeline.forward(req, **forward_kwargs)

    assert captured["max_sequence_length"] == WAN22_MAX_SEQUENCE_LENGTH
