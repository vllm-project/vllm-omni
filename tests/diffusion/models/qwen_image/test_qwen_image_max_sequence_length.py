from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput
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
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import DiffusionRequestState

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


def test_encode_defaults_to_tokenizer_max_length():
    pipeline = object.__new__(QwenImagePipeline)
    nn.Module.__init__(pipeline)
    pipeline.tokenizer_max_length = 1024
    pipeline.vae_scale_factor = 8
    pipeline.default_sample_size = 128
    pipeline.check_cfg_parallel_validity = lambda true_cfg_scale, has_neg_prompt: None

    captured = {}

    def _fake_encode_prompt(**kwargs):
        captured["max_sequence_length"] = kwargs["max_sequence_length"]
        embeds = torch.ones((1, 1, 1))
        mask = torch.ones((1, 1), dtype=torch.long)
        return embeds, mask

    pipeline.encode_prompt = _fake_encode_prompt
    state = DiffusionRequestState(
        request_id="qwen-prompt",
        prompt="prompt",
        sampling=SimpleNamespace(
            height=None,
            width=None,
            num_inference_steps=None,
            sigmas=None,
            guidance_scale_provided=False,
            num_outputs_per_prompt=0,
            generator=None,
            latents=None,
            true_cfg_scale=None,
            max_sequence_length=None,
            output_type=None,
        ),
    )

    pipeline.encode(state)

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


@pytest.mark.parametrize("use_additional_information", [False, True])
def test_state_generation_context_uses_request_prompt_tensors(use_additional_information: bool):
    pipeline = object.__new__(QwenImagePipeline)
    nn.Module.__init__(pipeline)
    pipeline.vae_scale_factor = 8
    pipeline.default_sample_size = 128
    pipeline.tokenizer_max_length = 1024

    prompt_embeds_a = torch.zeros(2, 3)
    prompt_embeds_mask_a = torch.tensor([True, True])
    negative_prompt_embeds_a = torch.full((2, 3), 2.0)
    negative_prompt_embeds_mask_a = torch.tensor([False, True])
    prompt_fields = {
        "prompt_embeds": prompt_embeds_a,
        "prompt_embeds_mask": prompt_embeds_mask_a,
        "negative_prompt_embeds": negative_prompt_embeds_a,
        "negative_prompt_embeds_mask": negative_prompt_embeds_mask_a,
    }
    if use_additional_information:
        prompt_fields = {key: [value] for key, value in prompt_fields.items()}
        prompt = {
            "prompt": "prompt-a",
            "negative_prompt": "negative-a",
            "additional_information": prompt_fields,
        }
    else:
        prompt = {
            "prompt": "prompt-a",
            "negative_prompt": "negative-a",
            **prompt_fields,
        }

    state = DiffusionRequestState(
        request_id="qwen-prompt-a",
        prompt=prompt,
        sampling=_make_request_batch_prompt_sampling(),
    )

    context = pipeline._state_generation_context(state)

    assert context["prompt"] is None
    assert context["negative_prompt"] is None
    torch.testing.assert_close(context["prompt_embeds"], prompt_embeds_a.unsqueeze(0))
    torch.testing.assert_close(context["prompt_embeds_mask"], prompt_embeds_mask_a.unsqueeze(0))
    torch.testing.assert_close(context["negative_prompt_embeds"], negative_prompt_embeds_a.unsqueeze(0))
    torch.testing.assert_close(context["negative_prompt_embeds_mask"], negative_prompt_embeds_mask_a.unsqueeze(0))


def test_forward_uses_atoms_and_batches_denoise_for_qwen_image():
    pipeline = object.__new__(QwenImagePipeline)
    nn.Module.__init__(pipeline)
    pipeline._interrupt = False

    def _request(request_id: str, prompt: str):
        return SimpleNamespace(
            request_id=request_id,
            prompt=prompt,
            sampling_params=_make_request_batch_prompt_sampling(),
            kv_sender_info=None,
        )

    batch = DiffusionRequestBatch(
        requests=[
            _request("qwen-prompt-a", "prompt-a"),
            _request("qwen-prompt-b", "prompt-b"),
        ]
    )

    events = []
    build_calls = []

    def _record(stage: str, state: DiffusionRequestState):
        events.append((stage, state.request_id))
        return state

    def _fake_prepare(state):
        _record("prepare", state)
        offset = 0.0 if state.request_id.endswith("a") else 10.0
        state.latents = torch.tensor([[offset, offset + 1]])
        state.timesteps = torch.tensor([1])
        state.step_index = 0
        return state

    def _fake_build_step_batch(states, *, cached_batch=None):
        build_calls.append(([state.request_id for state in states], cached_batch is not None))
        return SimpleNamespace(states=states)

    def _fake_denoise_step(input_batch):
        events.append(("denoise", tuple(state.request_id for state in input_batch.states)))
        return torch.cat([state.latents + 1 for state in input_batch.states], dim=0)

    def _fake_step_scheduler(state, noise_pred):
        events.append(("step", state.request_id))
        state.latents = noise_pred
        state.step_index += 1
        return state

    def _fake_decode(state):
        _record("decode", state)
        state.extra["decoded_output"] = DiffusionOutput(output=state.latents)
        return state

    pipeline.init_state = lambda state: _record("init", state)
    pipeline.check_inputs = lambda state: _record("check", state)
    pipeline.encode = lambda state: _record("encode", state)
    pipeline.prepare = _fake_prepare
    pipeline.build_step_batch = _fake_build_step_batch
    pipeline.denoise_step = _fake_denoise_step
    pipeline.step_scheduler = _fake_step_scheduler
    pipeline.decode = _fake_decode

    outputs = pipeline.forward(batch)

    assert build_calls == [(["qwen-prompt-a", "qwen-prompt-b"], False)]
    assert [event for event in events[:8]] == [
        ("init", "qwen-prompt-a"),
        ("check", "qwen-prompt-a"),
        ("encode", "qwen-prompt-a"),
        ("prepare", "qwen-prompt-a"),
        ("init", "qwen-prompt-b"),
        ("check", "qwen-prompt-b"),
        ("encode", "qwen-prompt-b"),
        ("prepare", "qwen-prompt-b"),
    ]
    assert ("denoise", ("qwen-prompt-a", "qwen-prompt-b")) in events
    assert [output.output.tolist() for output in outputs] == [[[1.0, 2.0]], [[11.0, 12.0]]]


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
