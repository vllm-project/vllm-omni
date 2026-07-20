# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image as qwen_image

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_qwen_image_postprocess_preserves_output_envelope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    model_path = tmp_path / "model"
    vae_path = model_path / "vae"
    vae_path.mkdir(parents=True)
    (vae_path / "config.json").write_text(json.dumps({}), encoding="utf-8")

    processed = object()
    captured: dict[str, object] = {}

    class FakeImageProcessor:
        def __init__(self, *, vae_scale_factor: int) -> None:
            captured["vae_scale_factor"] = vae_scale_factor

        def postprocess(self, images: torch.Tensor) -> object:
            captured["images"] = images
            return processed

    monkeypatch.setattr(qwen_image, "VaeImageProcessor", FakeImageProcessor)
    postprocess = qwen_image.get_qwen_image_post_process_func(SimpleNamespace(model=str(model_path)))

    image = torch.zeros(1, 3, 4, 4)
    output = postprocess(
        {
            "payload": {"image": image},
            "metadata": {"prompt_embeddings": {"present": True}},
        }
    )

    assert captured["vae_scale_factor"] == 16
    assert captured["images"] is image
    assert output == {
        "payload": {"image": processed},
        "metadata": {"prompt_embeddings": {"present": True}},
    }
