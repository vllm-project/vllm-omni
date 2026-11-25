# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image
from vllm import SamplingParams

from vllm_omni.config.omni_config import normalize_and_validate_diffusion_engine_ingress_kwargs

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class _Completion:
    text: str


@dataclass
class _Output:
    outputs: list[_Completion]


@pytest.mark.parametrize(
    ("family", "default_deploy"),
    [
        ("bagel", None),
        ("hunyuan_image3", "hunyuan_image3_ar.yaml"),
        ("mammoth_moda2", "mammoth_moda2_ar.yaml"),
    ],
)
@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("explicit_deploy", [False, True])
def test_x_to_text_reaches_generation_with_valid_engine_options(
    monkeypatch, tmp_path, capsys, family, default_deploy, with_image, explicit_deploy
):
    path = _REPO_ROOT / "examples/offline_inference/x_to_text/x_to_text.py"
    spec = importlib.util.spec_from_file_location("_x_to_text_mode_test_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    engine_kwargs: dict[str, object] = {}
    received_prompts: list[dict[str, object]] = []
    received_params: list[SamplingParams] = []
    closed: list[bool] = []

    class FakeOmni:
        def __init__(self, **kwargs):
            normalize_and_validate_diffusion_engine_ingress_kwargs(kwargs, stage_id=1)
            engine_kwargs.update(kwargs)
            self.default_sampling_params_list = [SamplingParams()]

        def generate(self, prompts, *, sampling_params_list):
            received_prompts.extend(prompts)
            received_params.extend(sampling_params_list)
            return [_Output(outputs=[_Completion(text="Paris.")])]

        def close(self):
            closed.append(True)

    def build_prompt(*, model_family, model, prompt, has_image):
        assert (model_family, model, has_image) == (family, "test-model", with_image)
        return {"prompt": prompt, "modalities": ["text"]}, [7]

    monkeypatch.setattr(module, "Omni", FakeOmni)
    monkeypatch.setattr(module, "get_x_to_text_model_family", lambda model: family)
    monkeypatch.setattr(module, "build_x_to_text_prompt", build_prompt)
    output_path = tmp_path / "answer.txt"
    argv = [
        str(path),
        "--model",
        "test-model",
        "--prompt",
        "What is the capital of France?",
        "--output",
        str(output_path),
    ]
    if with_image:
        image_path = tmp_path / "input.png"
        Image.new("RGB", (3, 2), color="red").save(image_path)
        argv.extend(["--image", str(image_path)])
    if explicit_deploy:
        argv.extend(["--deploy-config", "custom.yaml"])
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    assert capsys.readouterr().out == "Paris.\n"
    assert output_path.read_text(encoding="utf-8") == "Paris.\n"
    assert closed == [True]
    assert len(received_prompts) == 1
    prompt = received_prompts[0]
    assert prompt["prompt"] == "What is the capital of France?"
    assert prompt["modalities"] == ["text"]
    if with_image:
        multi_modal_data = prompt["multi_modal_data"]
        assert isinstance(multi_modal_data, dict)
        image = multi_modal_data["image"]
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        assert image.size == (3, 2)
        assert image.getpixel((0, 0)) == (255, 0, 0)
    else:
        assert "multi_modal_data" not in prompt
    assert len(received_params) == 1
    assert received_params[0].stop_token_ids == [7]
    expected_deploy = str(_REPO_ROOT / "vllm_omni/deploy" / default_deploy) if default_deploy else None
    if explicit_deploy:
        expected_deploy = "custom.yaml"
    assert engine_kwargs.get("deploy_config") == expected_deploy
    assert engine_kwargs["trust_remote_code"] == (family != "bagel")
