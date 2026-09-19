# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU routing coverage for single-stage SenseNova text generation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline
from vllm_omni.diffusion.models.sensenova_vision import single_stage as snv_single_stage
from vllm_omni.diffusion.models.sensenova_vision.pipeline_sensenova_vision import (
    SenseNovaVisionPipeline,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def _pipeline() -> SenseNovaVisionPipeline:
    return object.__new__(SenseNovaVisionPipeline)


def _contexts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], tuple[int, int]]:
    cache = SimpleNamespace(seq_lens=[8])
    context = {"kv_lens": [8], "ropes": [8], "past_key_values": cache}
    return context, dict(context), dict(context), (16, 16)


def test_single_stage_img2text_decodes_from_sensenova_local_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Text output does not fall back to BAGEL's generic image prefill."""
    pipeline = _pipeline()
    contexts = _contexts()
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        pipeline,
        "_prepare_single_stage_contexts",
        lambda prompt, sampling: calls.setdefault("prepared", contexts),
    )
    monkeypatch.setattr(
        pipeline,
        "_decode_single_stage_text",
        lambda context, sampling: calls.setdefault("decoded", "a giraffe beside a fence"),
    )

    output = pipeline._forward_single(
        {"prompt": "describe", "modalities": ["text"]},
        OmniDiffusionSamplingParams(extra_args={"max_think_tokens": 8192}),
    )

    assert calls["prepared"] is contexts
    assert calls["decoded"] == "a giraffe beside a fence"
    assert output.output["payload"] == {"text": "a giraffe beside a fence"}


def test_single_stage_caption_generate_decodes_before_injected_kv_denoising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Thinking image modes retain their locally decoded caption for merging."""
    pipeline = _pipeline()
    contexts = _contexts()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(pipeline, "_prepare_single_stage_contexts", lambda prompt, sampling: contexts)
    monkeypatch.setattr(pipeline, "_decode_single_stage_text", lambda context, sampling: "interleaved caption")
    monkeypatch.setattr(
        pipeline,
        "_update_single_stage_text_context",
        lambda context, text: captured.setdefault("reencoded", (context, text)),
    )

    def fake_base_forward(self, prompt, sampling, *, prepare_only=False):
        captured["sampling"] = sampling
        return DiffusionOutput(output={"payload": {"image": "image"}})

    monkeypatch.setattr(BagelPipeline, "_forward_single", fake_base_forward)
    sampling = OmniDiffusionSamplingParams(extra_args={"think": True, "max_think_tokens": 8192})

    output = pipeline._forward_single(
        {"prompt": "caption", "modalities": ["img2img"]},
        sampling,
    )

    assert output.output["payload"] == {"image": "image"}
    assert captured["sampling"].past_key_values is contexts[0]["past_key_values"]
    assert captured["reencoded"] == (contexts[0], "interleaved caption")
    assert sampling.extra_args["text_output"] == "interleaved caption"


def test_single_stage_prefill_normalizes_transport_markers_and_uses_vit_only_for_understanding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Understanding converts the two-stage prompt into upstream raw terms."""

    class FakeCache:
        def __init__(self, _layers: int) -> None:
            self.seq_lens = [0]

    calls: dict[str, Any] = {"vae": 0, "vit": 0, "texts": []}
    pipeline = _pipeline()
    pipeline.tokenizer = SimpleNamespace()
    pipeline.device = torch.device("cpu")
    pipeline.od_config = SimpleNamespace(dtype=torch.bfloat16)
    pipeline.new_token_ids = {}
    pipeline.bagel = SimpleNamespace(
        config=SimpleNamespace(llm_config=SimpleNamespace(num_hidden_layers=1)),
        max_latent_size=64,
        latent_downsample=8,
        prepare_prompts=lambda curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids: (
            calls["texts"].append(prompts[0]) or {"text": torch.tensor([1])},
            [curr_kvlens[0] + 3],
            [curr_rope[0] + 3],
        ),
        prepare_vit_images=lambda curr_kvlens, curr_rope, images, transforms, new_token_ids: (
            (calls.update({"vit": calls["vit"] + 1}) or {"vit": torch.tensor([1])}),
            [curr_kvlens[0] + 3],
            [curr_rope[0] + 1],
        ),
        forward_cache_update_vit=lambda cache, **kwargs: cache,
        forward_cache_update_text=lambda cache, **kwargs: cache,
        prepare_vae_images=lambda *args, **kwargs: calls.update({"vae": calls["vae"] + 1}),
    )
    pipeline._resize_context_image = lambda image, **kwargs: image
    pipeline._context_vit_transform = lambda image, **kwargs: torch.zeros(3, 14, 14)
    monkeypatch.setattr(snv_single_stage, "NaiveCache", FakeCache)

    prompt = "<|im_start|>user\n<|image_pad|>\nfind birds<|im_end|>\n<|im_start|>assistant\n"
    pipeline._prepare_single_stage_contexts(
        {"prompt": prompt, "multi_modal_data": {"image": Image.new("RGB", (16, 16))}},
        OmniDiffusionSamplingParams(),
    )

    # Positive and image-CFG contexts both receive only the raw user term;
    # Bagel.prepare_prompts owns its BOS/EOS wrapper.
    assert calls["texts"] == ["find birds", "find birds"]
    assert calls["vae"] == 0
    assert calls["vit"] == 1
