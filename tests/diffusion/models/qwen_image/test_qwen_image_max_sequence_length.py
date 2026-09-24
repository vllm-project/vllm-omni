# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.qwen_image.cfg_parallel import (
    QwenImageCFGParallelMixin,
    canonicalize_qwen_image_attention_mask,
)
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
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_attention_mask_canonicalization_drops_only_all_valid_masks():
    all_valid = torch.ones((1, 4), dtype=torch.bool)
    padded = torch.tensor([[True, True, False, False]])

    assert canonicalize_qwen_image_attention_mask(None) is None
    assert canonicalize_qwen_image_attention_mask(all_valid) is None
    assert canonicalize_qwen_image_attention_mask(padded) is padded


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


@pytest.mark.parametrize(("pipeline_class", "_drop_idx", "_input_kind"), PIPELINE_CASES)
def test_all_qwen_image_pipelines_share_mask_canonicalization_path(
    pipeline_class: type[QwenImageCFGParallelMixin],
    _drop_idx: int,
    _input_kind: str,
):
    assert pipeline_class.diffuse is QwenImageCFGParallelMixin.diffuse


@dataclass
class _StepSampling:
    true_cfg_scale: float = 1.0
    cfg_normalize: bool = False
    image_latent: torch.Tensor | None = None


@dataclass
class _TransformerState:
    do_true_cfg: bool = False


def _make_qwen_step_state(request_id: str, prompt_length: int) -> StepRequestState:
    return StepRequestState(
        request_id=request_id,
        sampling=_StepSampling(),
        prompt_embeds=torch.ones((1, prompt_length, 2)),
        prompt_embeds_mask=torch.ones((1, prompt_length), dtype=torch.bool),
        latents=torch.ones((1, 2, 2)),
        timesteps=torch.tensor([1.0]),
        img_shapes=[[(1, 1, 1)]],
        txt_seq_lens=[prompt_length],
    )


def _capture_qwen_denoise_mask(input_batch: InputBatch) -> torch.Tensor | None:
    pipeline = object.__new__(QwenImagePipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = _TransformerState()
    pipeline._attention_kwargs = {}
    pipeline._interrupt = False

    captured = {}

    def _fake_predict_noise(*args):
        captured["mask"] = args[2]["encoder_hidden_states_mask"]
        return input_batch.latents

    pipeline.predict_noise_maybe_with_cfg = _fake_predict_noise
    pipeline.denoise_step(input_batch)
    return captured["mask"]


def test_step_denoise_keeps_padding_mask_after_variable_length_batching():
    input_batch = InputBatch.make_batch(
        [
            _make_qwen_step_state("short", 2),
            _make_qwen_step_state("long", 4),
        ]
    )

    mask = _capture_qwen_denoise_mask(input_batch)

    assert torch.equal(
        mask,
        torch.tensor(
            [
                [True, True, False, False],
                [True, True, True, True],
            ]
        ),
    )


def test_step_denoise_drops_all_valid_mask_after_batching():
    input_batch = InputBatch.make_batch(
        [
            _make_qwen_step_state("first", 4),
            _make_qwen_step_state("second", 4),
        ]
    )

    assert _capture_qwen_denoise_mask(input_batch) is None
    assert input_batch.prompt_embeds_mask is None


def _make_pipeline(
    pipeline_class: type[QwenImageCFGParallelMixin],
    *,
    total_sequence_length: int,
    drop_idx: int,
    input_kind: str,
):
    pipeline: Any = object.__new__(pipeline_class)
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
        prompt="prompt",
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


def _make_request_batch_prompt_sampling(**overrides):
    values = {
        "height": 32,
        "width": 32,
        "num_inference_steps": 2,
        "sigmas": None,
        "max_sequence_length": None,
        "num_outputs_per_prompt": 0,
        "generator": None,
        "latents": None,
        "true_cfg_scale": None,
        "guidance_scale_provided": False,
        "guidance_scale": 1.0,
        "output_type": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_forward_collates_request_prompt_tensors_for_qwen_image():
    pipeline = object.__new__(QwenImagePipeline)
    nn.Module.__init__(pipeline)
    pipeline.vae_scale_factor = 8
    pipeline.default_sample_size = 128

    class StopAfterPrepareContextError(Exception):
        pass

    captured = {}

    def _fake_prepare_generation_context(**kwargs):
        captured.update(kwargs)
        raise StopAfterPrepareContextError

    pipeline._prepare_generation_context = _fake_prepare_generation_context

    prompt_embeds_a = torch.zeros(2, 3)
    prompt_embeds_b = torch.ones(2, 3)
    prompt_embeds_mask_a = torch.tensor([True, True])
    prompt_embeds_mask_b = torch.tensor([True, False])
    negative_prompt_embeds_a = torch.full((2, 3), 2.0)
    negative_prompt_embeds_b = torch.full((2, 3), 3.0)
    negative_prompt_embeds_mask_a = torch.tensor([False, True])
    negative_prompt_embeds_mask_b = torch.tensor([False, False])

    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="qwen-prompt-a",
                prompt={
                    "prompt": "prompt-a",
                    "negative_prompt": "negative-a",
                    "prompt_embeds": prompt_embeds_a,
                    "prompt_embeds_mask": prompt_embeds_mask_a,
                    "negative_prompt_embeds": negative_prompt_embeds_a,
                    "negative_prompt_embeds_mask": negative_prompt_embeds_mask_a,
                },
                sampling_params=_make_request_batch_prompt_sampling(),
            ),
            SimpleNamespace(
                request_id="qwen-prompt-b",
                prompt={
                    "prompt": "prompt-b",
                    "negative_prompt": "negative-b",
                    "additional_information": {
                        "prompt_embeds": [prompt_embeds_b],
                        "prompt_embeds_mask": [prompt_embeds_mask_b],
                        "negative_prompt_embeds": [negative_prompt_embeds_b],
                        "negative_prompt_embeds_mask": [negative_prompt_embeds_mask_b],
                    },
                },
                sampling_params=_make_request_batch_prompt_sampling(),
            ),
        ]
    )

    with pytest.raises(StopAfterPrepareContextError):
        pipeline.forward(batch)

    assert captured["prompt"] is None
    assert captured["negative_prompt"] is None
    torch.testing.assert_close(
        captured["prompt_embeds"],
        torch.stack([prompt_embeds_a, prompt_embeds_b], dim=0),
    )
    torch.testing.assert_close(
        captured["prompt_embeds_mask"],
        torch.stack([prompt_embeds_mask_a, prompt_embeds_mask_b], dim=0),
    )
    torch.testing.assert_close(
        captured["negative_prompt_embeds"],
        torch.stack([negative_prompt_embeds_a, negative_prompt_embeds_b], dim=0),
    )
    torch.testing.assert_close(
        captured["negative_prompt_embeds_mask"],
        torch.stack([negative_prompt_embeds_mask_a, negative_prompt_embeds_mask_b], dim=0),
    )


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
    pipeline: Any = object.__new__(pipeline_class)
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
    pipeline: Any = object.__new__(pipeline_class)
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
    pipeline: Any = object.__new__(pipeline_class)
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
