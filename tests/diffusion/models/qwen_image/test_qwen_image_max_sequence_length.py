import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import (
    QwenImagePipeline,
)
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit import (
    QwenImageEditPipeline,
)
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit_plus import (
    QwenImageEditPlusPipeline,
)
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_layered import (
    QwenImageLayeredPipeline,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _RejectingTextEncoder:
    dtype = torch.float32

    def __call__(self, *args, **kwargs):
        raise AssertionError("text encoder should not run for prompts that exceed max_sequence_length")


class _FakeModelInputs:
    def __init__(self, total_sequence_length: int):
        attention_mask = torch.ones((1, total_sequence_length), dtype=torch.long)
        self.input_ids = attention_mask.clone()
        self.attention_mask = attention_mask
        self.pixel_values = None
        self.image_grid_thw = None

    def to(self, device):
        return self


class _FakeTokenizer:
    def __init__(self, total_sequence_length: int | list[int]):
        if isinstance(total_sequence_length, list):
            self.total_sequence_lengths = list(total_sequence_length)
        else:
            self.total_sequence_lengths = [total_sequence_length]

    def __call__(self, *args, **kwargs):
        if len(self.total_sequence_lengths) > 1:
            total_sequence_length = self.total_sequence_lengths.pop(0)
        else:
            total_sequence_length = self.total_sequence_lengths[0]
        return _FakeModelInputs(total_sequence_length)


class _FakeProcessor(_FakeTokenizer):
    pass


class _FakeScheduler:
    def __init__(self):
        self.begin_index = None

    def set_begin_index(self, begin_index: int):
        self.begin_index = begin_index


PIPELINE_CASES = [
    pytest.param(QwenImagePipeline, 34, "tokenizer", id="qwen-image"),
    pytest.param(QwenImageLayeredPipeline, 34, "tokenizer", id="qwen-image-layered"),
    pytest.param(QwenImageEditPipeline, 64, "processor", id="qwen-image-edit"),
    pytest.param(QwenImageEditPlusPipeline, 64, "processor", id="qwen-image-edit-plus"),
]


def _make_pipeline(
    pipeline_class: type,
    *,
    total_sequence_length: int,
    drop_idx: int,
    input_kind: str,
):
    pipeline = object.__new__(pipeline_class)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.text_encoder = _RejectingTextEncoder()
    pipeline.tokenizer_max_length = 1024
    pipeline.prompt_template_encode = "{}"
    pipeline.prompt_template_encode_start_idx = drop_idx
    pipeline.tokenizer = _FakeTokenizer([total_sequence_length, 0])
    if input_kind == "processor":
        pipeline.processor = _FakeProcessor(total_sequence_length)
    return pipeline


@pytest.mark.parametrize(("pipeline_class", "drop_idx", "input_kind"), PIPELINE_CASES)
def test_encode_prompt_rejects_prompt_longer_than_default_max_sequence_length(
    pipeline_class: type,
    drop_idx: int,
    input_kind: str,
):
    pipeline = _make_pipeline(
        pipeline_class,
        total_sequence_length=1025,
        drop_idx=drop_idx,
        input_kind=input_kind,
    )

    with pytest.raises(ValueError, match=r"got 1025 tokens, but `max_sequence_length` is 1024"):
        pipeline.encode_prompt(prompt="prompt")


@pytest.mark.parametrize(("pipeline_class", "drop_idx", "input_kind"), PIPELINE_CASES)
def test_encode_prompt_rejects_prompt_longer_than_explicit_max_sequence_length(
    pipeline_class: type,
    drop_idx: int,
    input_kind: str,
):
    pipeline = _make_pipeline(
        pipeline_class,
        total_sequence_length=17,
        drop_idx=drop_idx,
        input_kind=input_kind,
    )

    with pytest.raises(ValueError, match=r"got 17 tokens, but `max_sequence_length` is 16"):
        pipeline.encode_prompt(prompt="prompt", max_sequence_length=16)


def test_prepare_encode_defaults_to_tokenizer_max_length():
    pipeline = object.__new__(QwenImagePipeline)
    nn.Module.__init__(pipeline)
    pipeline.tokenizer_max_length = 1024
    pipeline.vae_scale_factor = 8
    pipeline.default_sample_size = 128
    pipeline.scheduler = _FakeScheduler()
    pipeline._extract_prompts = lambda prompts: (["prompt"], None)

    captured = {}

    def _fake_prepare_generation_context(**kwargs):
        captured["max_sequence_length"] = kwargs["max_sequence_length"]
        embeds = torch.ones((1, 1, 1))
        mask = torch.ones((1, 1), dtype=torch.long)
        return {
            "prompt_embeds": embeds,
            "prompt_embeds_mask": mask,
            "negative_prompt_embeds": None,
            "negative_prompt_embeds_mask": None,
            "latents": embeds,
            "timesteps": torch.tensor([1]),
            "do_true_cfg": False,
            "guidance": None,
            "img_shapes": [[(1, 1, 1)]],
            "txt_seq_lens": [1],
            "negative_txt_seq_lens": None,
        }

    pipeline._prepare_generation_context = _fake_prepare_generation_context
    state = SimpleNamespace(
        prompts=["prompt"],
        sampling=SimpleNamespace(
            height=None,
            width=None,
            num_inference_steps=None,
            sigmas=None,
            guidance_scale_provided=False,
            num_outputs_per_prompt=0,
            generator=None,
            true_cfg_scale=None,
            max_sequence_length=None,
        ),
    )

    pipeline.prepare_encode(state)

    assert captured["max_sequence_length"] == 1024


@pytest.mark.parametrize(
    ("pipeline_class", "drop_idx"),
    [
        pytest.param(QwenImageEditPipeline, 64, id="qwen-image-edit"),
        pytest.param(QwenImageEditPlusPipeline, 64, id="qwen-image-edit-plus"),
    ],
)
def test_edit_pipelines_validate_text_prompt_length_before_image_token_expansion(
    pipeline_class: type,
    drop_idx: int,
):
    pipeline = object.__new__(pipeline_class)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.text_encoder = _RejectingTextEncoder()
    pipeline.tokenizer_max_length = 1024
    pipeline.prompt_template_encode = "{}"
    pipeline.prompt_template_encode_start_idx = drop_idx
    pipeline.tokenizer = _FakeTokenizer([8, 0])
    pipeline.processor = _FakeProcessor(drop_idx + 1500)

    with pytest.raises(AssertionError, match="text encoder should not run"):
        pipeline.encode_prompt(prompt="short prompt")


@pytest.mark.parametrize(
    "pipeline_class",
    [
        pytest.param(QwenImagePipeline, id="qwen-image"),
        pytest.param(QwenImageLayeredPipeline, id="qwen-image-layered"),
    ],
)
def test_qwen_generation_validator_excludes_template_suffix_from_budget(pipeline_class: type):
    pipeline = object.__new__(pipeline_class)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.text_encoder = _RejectingTextEncoder()
    pipeline.tokenizer_max_length = 1024
    pipeline.prompt_template_encode = "{}"
    pipeline.prompt_template_encode_start_idx = 34
    pipeline.tokenizer = _FakeTokenizer([1029, 5])

    with pytest.raises(AssertionError, match="text encoder should not run"):
        pipeline.encode_prompt(prompt="boundary prompt")


@pytest.mark.parametrize(
    "pipeline_class",
    [
        pytest.param(QwenImageEditPipeline, id="qwen-image-edit"),
        pytest.param(QwenImageEditPlusPipeline, id="qwen-image-edit-plus"),
    ],
)
def test_qwen_edit_validator_excludes_image_placeholders_from_budget(pipeline_class: type):
    pipeline = object.__new__(pipeline_class)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.text_encoder = _RejectingTextEncoder()
    pipeline.tokenizer_max_length = 1024
    pipeline.prompt_template_encode = "{}"
    pipeline.prompt_template_encode_start_idx = 64
    pipeline.tokenizer = _FakeTokenizer([30, 20])
    pipeline.processor = _FakeProcessor(1500)

    with pytest.raises(AssertionError, match="text encoder should not run"):
        pipeline.encode_prompt(prompt="short prompt")


@pytest.mark.parametrize(
    "pipeline_class",
    [
        QwenImagePipeline,
        QwenImageLayeredPipeline,
        QwenImageEditPipeline,
        QwenImageEditPlusPipeline,
    ],
)
def test_forward_max_sequence_length_default_is_1024(pipeline_class: type):
    assert inspect.signature(pipeline_class.forward).parameters["max_sequence_length"].default == 1024
