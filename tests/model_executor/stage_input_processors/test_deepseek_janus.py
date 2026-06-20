# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.deepseek_janus import stage_input_processors as janus_model_processors
from vllm_omni.model_executor.stage_input_processors.deepseek_janus import ar2generation, ar_tokens_to_vq

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_ar_row(token_ids: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=token_ids)],
        sampled_token_ids=None,
    )


def _make_text_row(text: str) -> SimpleNamespace:
    return SimpleNamespace(outputs=[SimpleNamespace(text=text)])


def test_ar_tokens_to_vq_emits_prompt_extra_with_image_tokens() -> None:
    token_ids = list(range(600))

    result = ar_tokens_to_vq(
        [SimpleNamespace(engine_outputs=[_make_ar_row(token_ids)])],
        [0],
        prompt=[{"prompt": "p", "height": 384, "width": 384, "patch_size": 32}],
    )

    assert len(result) == 1
    assert result[0]["height"] == 384
    assert result[0]["width"] == 384
    assert "extra" in result[0]
    image_tokens = result[0]["extra"]["image_tokens"]
    assert isinstance(image_tokens, torch.Tensor)
    assert image_tokens.dtype == torch.long
    assert image_tokens.shape == (576,)
    assert result[0]["extra"]["img_size"] == 384
    assert result[0]["extra"]["patch_size"] == 32


def test_ar2generation_preserves_prompt_extra() -> None:
    result = ar2generation(
        [SimpleNamespace(engine_outputs=[_make_text_row("refined")])],
        [0],
        prompt=[{"prompt": "base", "extra": {"img_size": 512, "patch_size": 32}}],
    )

    assert result[0]["prompt"] == "base\nrefined"
    assert result[0]["extra"]["img_size"] == 512
    assert result[0]["extra"]["patch_size"] == 32
    assert result[0]["extra"]["ar_generated_text"] == "refined"


def test_compat_module_reexports_model_local_bridge() -> None:
    assert ar_tokens_to_vq is janus_model_processors.ar_tokens_to_vq
