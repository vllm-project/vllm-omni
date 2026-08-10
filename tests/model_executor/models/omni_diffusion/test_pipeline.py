# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Omni-Diffusion pipeline topology and deploy defaults."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy
from vllm_omni.model_executor.models.omni_diffusion.pipeline import (
    OMNI_DIFFUSION_TEXT_PIPELINE,
)
from vllm_omni.model_executor.models.omni_diffusion.utils import (
    OMNI_DIFFUSION_END_OF_TEXT_TOKEN_ID,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY_DIR = Path(__file__).resolve().parents[4] / "vllm_omni" / "deploy"
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_offline_example():
    example_path = _REPO_ROOT / "examples" / "offline_inference" / "omni_diffusion" / "end2end.py"
    spec = importlib.util.spec_from_file_location("omni_diffusion_offline_example", example_path)
    assert spec is not None and spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    return example


def _load_online_client():
    client_path = _REPO_ROOT / "examples" / "online_serving" / "omni_diffusion" / "client.py"
    spec = importlib.util.spec_from_file_location("omni_diffusion_online_client", client_path)
    assert spec is not None and spec.loader is not None
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    return client


@pytest.mark.parametrize(
    ("deploy_name", "expected_stage0_tokens", "expected_stage1_tokens"),
    [
        ("omni_diffusion_asr.yaml", 50, 51),
        ("omni_diffusion_vqa.yaml", 64, 65),
        ("omni_diffusion_svqa.yaml", 64, 65),
    ],
)
def test_text_pipeline_uses_task_level_max_tokens(
    deploy_name: str,
    expected_stage0_tokens: int,
    expected_stage1_tokens: int,
) -> None:
    deploy = load_deploy_config(_DEPLOY_DIR / deploy_name)
    stage0_deploy, stage1_deploy = deploy.stages

    stage0_tokens = stage0_deploy.engine_extras["additional_config"]["max_new_tokens"]
    assert stage0_tokens == expected_stage0_tokens
    assert stage1_deploy.default_sampling_params["max_tokens"] == expected_stage1_tokens
    assert expected_stage1_tokens >= stage0_tokens + 1

    stage0, stage1 = merge_pipeline_deploy(OMNI_DIFFUSION_TEXT_PIPELINE, deploy)
    del stage0
    sampling_params = stage1.yaml_extras["default_sampling_params"]
    assert sampling_params["max_tokens"] == expected_stage1_tokens
    assert sampling_params["stop_token_ids"] == [OMNI_DIFFUSION_END_OF_TEXT_TOKEN_ID]


def test_text_pipeline_does_not_force_max_tokens() -> None:
    _, text_adapter_stage = OMNI_DIFFUSION_TEXT_PIPELINE.stages
    assert "max_tokens" not in text_adapter_stage.sampling_constraints


def test_examples_explicitly_trust_remote_model_code() -> None:
    deploy = load_deploy_config(_DEPLOY_DIR / "omni_diffusion_t2i.yaml")
    assert deploy.trust_remote_code is None

    server_script = (_REPO_ROOT / "examples" / "online_serving" / "omni_diffusion" / "run_server.sh").read_text(
        encoding="utf-8"
    )
    assert "--trust-remote-code" in server_script

    offline_example = (_REPO_ROOT / "examples" / "offline_inference" / "omni_diffusion" / "end2end.py").read_text(
        encoding="utf-8"
    )
    syntax_tree = ast.parse(offline_example)
    omni_calls = [
        node
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Omni"
    ]
    assert len(omni_calls) == 1
    trust_keyword = next(
        (keyword for keyword in omni_calls[0].keywords if keyword.arg == "trust_remote_code"),
        None,
    )
    assert trust_keyword is not None
    assert isinstance(trust_keyword.value, ast.Constant)
    assert trust_keyword.value.value is True


@pytest.mark.parametrize(
    ("task", "prompt", "expected_prompt"),
    [
        (
            "t2i",
            "a lighthouse in a storm",
            "Generate an image based on the provided text description.\na lighthouse in a storm",
        ),
        (
            "tts",
            "Hello from Omni-Diffusion.",
            "Convert the text to speech.\nHello from Omni-Diffusion.",
        ),
    ],
)
def test_online_client_adds_task_instruction(task: str, prompt: str, expected_prompt: str) -> None:
    client = _load_online_client()
    args = SimpleNamespace(
        task=task,
        model="lijiang/Omni-Diffusion",
        prompt=prompt,
        image_path=None,
        audio_path=None,
    )

    payload = client.build_payload(args)

    assert payload["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": expected_prompt}],
        }
    ]


def test_offline_image_tasks_read_pipeline_and_completion_outputs() -> None:
    example = _load_offline_example()

    pipeline_image = Image.new("RGB", (2, 2))
    pipeline_result = SimpleNamespace(
        images=[],
        request_output=SimpleNamespace(images=[pipeline_image]),
    )
    assert example._get_result_image(pipeline_result) is pipeline_image

    completion_result = SimpleNamespace(
        images=[],
        request_output=SimpleNamespace(
            images=[],
            outputs=[SimpleNamespace(multimodal_output={"image": torch.zeros((3, 2, 2))})],
        ),
    )
    assert example._get_result_image(completion_result).size == (2, 2)


def test_offline_text_only_generation_uses_model_chat_template(monkeypatch: pytest.MonkeyPatch) -> None:
    example = _load_offline_example()

    class FakeTokenizer:
        def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
            assert messages == [{"role": "user", "content": "describe a beach"}]
            assert add_generation_prompt is True
            assert tokenize is False
            return "<rendered-chat-prompt>"

    monkeypatch.setattr(
        example.AutoTokenizer,
        "from_pretrained",
        lambda model, *, trust_remote_code: FakeTokenizer(),
    )
    assert example._render_chat_prompt("model", "describe a beach") == "<rendered-chat-prompt>"
