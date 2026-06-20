# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.deepseek_janus import pipeline_janus
from vllm_omni.diffusion.models.deepseek_janus.pipeline_janus import JanusPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeTokenizer:
    def encode(self, prompt: str) -> list[int]:
        assert prompt.endswith("<img>")
        return [1, 2, 3]


class _FakeProcessor:
    sft_format = "deepseek"
    image_start_tag = "<img>"
    pad_id = 0

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def apply_sft_template_for_multi_turn_prompts(self, conversations, sft_format, system_prompt):
        del conversations, sft_format, system_prompt
        return "stub"


class _FakeLanguageModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace()

    def get_input_embeddings(self):
        def _embed(tokens: torch.Tensor) -> torch.Tensor:
            batch, seq = tokens.shape
            return torch.ones((batch, seq, 4), dtype=torch.float32)

        return _embed


class _FakeTransformer:
    def __call__(self, *, inputs_embeds, use_cache, past_key_values, return_dict, cache_position=None):
        del use_cache, past_key_values, return_dict, cache_position
        batch = inputs_embeds.shape[0]
        return SimpleNamespace(last_hidden_state=torch.zeros((batch, 1, 4), dtype=inputs_embeds.dtype))


class _FakeGenVisionModel:
    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def decode_code(self, generated: torch.Tensor, shape: list[int]) -> torch.Tensor:
        del generated
        self.calls.append(shape)
        h = shape[2] * 16
        w = shape[3] * 16
        return torch.zeros((shape[0], 3, h, w), dtype=torch.float32)


class _FakeMMModel:
    def __init__(self, gen_vision_model: _FakeGenVisionModel) -> None:
        self.language_model = _FakeLanguageModel()
        self.gen_vision_model = gen_vision_model

    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def gen_head(self, hidden: torch.Tensor) -> torch.Tensor:
        batch = hidden.shape[0]
        logits = torch.zeros((batch, 6), dtype=hidden.dtype)
        logits[:, 0] = 1.0
        return logits

    def prepare_gen_img_embeds(self, stacked: torch.Tensor) -> torch.Tensor:
        return torch.ones((stacked.shape[0], 4), dtype=torch.float32)


def _build_pipeline() -> tuple[JanusPipeline, _FakeGenVisionModel]:
    pipe = JanusPipeline.__new__(JanusPipeline)
    nn.Module.__init__(pipe)
    pipe.processor = _FakeProcessor()
    pipe.mm_model = _FakeMMModel(_FakeGenVisionModel())
    pipe.transformer = _FakeTransformer()
    pipe.od_config = SimpleNamespace(enforce_eager=True)
    pipe._prefill_chunk_size = 2048
    pipe._cudagraph_wrapper = None
    pipe._stage_durations = {}
    return pipe, pipe.mm_model.gen_vision_model


def test_janus_pipeline_reads_prompt_extra_image_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_janus, "StaticCache", lambda **kwargs: object())
    pipe, gen_vision_model = _build_pipeline()
    req = OmniDiffusionRequest(
        prompts=[{"prompt": "p", "extra": {"img_size": 512, "patch_size": 32}}],
        sampling_params=OmniDiffusionSamplingParams(num_outputs_per_prompt=1),
        request_id="req-1",
    )

    output = pipe.forward(req)

    assert output.error is None
    assert gen_vision_model.calls == [[1, 8, 16, 16]]


def test_janus_pipeline_prefers_sampling_extra_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_janus, "StaticCache", lambda **kwargs: object())
    pipe, gen_vision_model = _build_pipeline()
    req = OmniDiffusionRequest(
        prompts=[{"prompt": "p", "extra": {"img_size": 128, "patch_size": 8}}],
        sampling_params=OmniDiffusionSamplingParams(
            num_outputs_per_prompt=1,
            extra_step_kwargs={"img_size": 384, "patch_size": 16},
        ),
        request_id="req-2",
    )

    output = pipe.forward(req)

    assert output.error is None
    assert gen_vision_model.calls == [[1, 8, 24, 24]]
