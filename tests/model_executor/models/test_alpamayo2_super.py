# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the Alpamayo 2 Super reasoning stage, registration and prompts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import StageExecutionType
from vllm_omni.diffusion.registry import _DIFFUSION_MODELS
from vllm_omni.model_executor.models.alpamayo2_super import Alpamayo2SuperConfig
from vllm_omni.model_executor.models.alpamayo2_super import openpi as alpamayo_openpi
from vllm_omni.model_executor.models.alpamayo2_super.alpamayo2_super import (
    Alpamayo2SuperPolicyForConditionalGeneration,
)
from vllm_omni.model_executor.models.alpamayo2_super.processing import (
    SPECIAL_TOKENS,
    build_trajectory_messages,
    encode_history_tokens,
)
from vllm_omni.model_executor.models.registry import _OMNI_MODELS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

FUTURE_START = 20
FUTURE_END = 21
TEXT_EOS = 2


def test_config_exposes_nested_qwen3_vl_fields():
    config = Alpamayo2SuperConfig(
        vlm_config={"text_config": {"hidden_size": 64}, "vision_config": {"depth": 2}, "image_token_id": 7},
        traj_ids={"future_start": FUTURE_START},
        history_vocab_size=10,
        future_vocab_size=30,
    )
    assert config.text_config.hidden_size == 64
    assert config.vision_config.depth == 2
    assert config.image_token_id == 7
    assert config.traj_vocab_size == 40
    assert config.traj_ids["future_start"] == FUTURE_START


def test_registered_as_reasoning_and_expert_stages():
    pipeline = OMNI_PIPELINES["alpamayo2_super"]
    reasoning, expert = pipeline.stages
    assert reasoning.execution_type == StageExecutionType.LLM_AR
    assert reasoning.omni_kv_config == {"need_send_cache": True}
    assert expert.execution_type == StageExecutionType.DIFFUSION and expert.input_sources == (0,)
    assert expert.omni_kv_config == {"need_recv_cache": True} and expert.final_output
    assert _OMNI_MODELS["Alpamayo2SuperPolicy"][2] == "Alpamayo2SuperPolicyForConditionalGeneration"
    assert _DIFFUSION_MODELS[expert.model_arch][2] == "Alpamayo2SuperPipeline"


def test_history_tokens_quantize_waypoint_deltas():
    # Constant 1 m steps along x; y and z stay fixed.
    history = torch.zeros(16, 3)
    history[:, 0] = torch.arange(-15.0, 1.0)
    tokens = encode_history_tokens(history, num_bins=1000)
    assert tokens.shape == (45,)
    x_bin = round((1.0 + 4.0) / 8.0 * 999)
    y_bin = round(4.0 / 8.0 * 999)
    z_bin = round(10.0 / 20.0 * 999)
    assert tokens.view(15, 3).tolist() == [[x_bin, y_bin, z_bin]] * 15


def test_trajectory_messages_follow_released_prompt():
    config = Alpamayo2SuperConfig(include_camera_ids=True, frame_label="frame_num", tokens_per_history_traj=3)
    messages = build_trajectory_messages([["a0", "a1"], ["b0", "b1"]], [1, 6], config)
    assert [message["role"] for message in messages] == ["system", "user"]
    content = messages[1]["content"]
    assert content[0] == {"type": "text", "text": "Front camera: "}
    assert content[1] == {"type": "text", "text": "frame 0 "}
    assert content[2] == {"type": "image", "image": "a0"}
    assert content[5] == {"type": "text", "text": "Front telephoto camera: "}
    assert content[7] == {"type": "image", "image": "b0"}
    history = SPECIAL_TOKENS["traj_history"] * 3
    assert content[-2]["text"] == f"<|traj_history_start|>{history}<|traj_history_end|>"
    assert content[-1]["text"].endswith("then output the future trajectory.")
    with pytest.raises(ValueError, match="sorted"):
        build_trajectory_messages([["a"], ["b"]], [6, 1], config)


def _make_model() -> Alpamayo2SuperPolicyForConditionalGeneration:
    model = Alpamayo2SuperPolicyForConditionalGeneration.__new__(Alpamayo2SuperPolicyForConditionalGeneration)
    nn.Module.__init__(model)
    model.future_start_id = FUTURE_START
    model.future_end_id = FUTURE_END
    model.traj_vocab_start = 10
    model.traj_vocab_size = 5
    model.text_eos_ids = torch.tensor([TEXT_EOS])
    model._step_triggers = None
    model._recent_steps = {}
    model._step = 0
    return model


def _step(model, input_ids, req_ids, num_computed_tokens, num_scheduled_tokens):
    total = len(input_ids)
    model.prepare_runner_inputs(
        input_ids=torch.tensor(input_ids),
        positions=torch.stack([torch.arange(total), torch.arange(total) + 10, torch.arange(total) + 20]),
        inputs_embeds=None,
        req_ids=req_ids,
        num_computed_tokens=num_computed_tokens,
        num_scheduled_tokens=num_scheduled_tokens,
        input_ids_buffer=torch.tensor(input_ids),
    )


def test_future_start_ends_request_and_publishes_kv_metadata():
    model = _make_model()
    # Request "a" is prefilling, "b" feeds back future_start, "c" is decoding.
    _step(model, [5, 6, FUTURE_START, FUTURE_START + 1], ["a", "b", "c"], [0, 100, 50], [2, 1, 1])
    assert model._step_triggers.tolist() == [False, True, False]
    assert model.get_kv_transfer_metadata("b") == {"prefix_len": 101, "trigger_position": [2, 12, 22]}
    assert model.get_kv_transfer_metadata("b") is None
    # "a" never fed back future_start (for example it stopped at max_tokens).
    assert model.get_kv_transfer_metadata("a") is None


def test_kv_metadata_survives_a_step_scheduled_after_the_trigger():
    # With async scheduling, "b" runs once more (feeding back future_end)
    # before the runner learns it finished.
    model = _make_model()
    _step(model, [FUTURE_START, 7], ["b", "c"], [100, 50], [1, 1])
    _step(model, [FUTURE_END, 8], ["b", "c"], [101, 51], [1, 1])
    assert model.get_kv_transfer_metadata("b") == {"prefix_len": 101, "trigger_position": [0, 10, 20]}


def test_requests_that_leave_the_batch_are_forgotten():
    model = _make_model()
    _step(model, [5], ["aborted"], [10], [1])
    for _ in range(65):
        _step(model, [6], ["c"], [50], [1])
    assert "aborted" not in model._recent_steps and "c" in model._recent_steps


def test_speculative_decoding_is_rejected():
    model = Alpamayo2SuperPolicyForConditionalGeneration.__new__(Alpamayo2SuperPolicyForConditionalGeneration)
    with pytest.raises(ValueError, match="speculative decoding"):
        model.__init__(vllm_config=SimpleNamespace(speculative_config=object()))


def test_compute_logits_masks_trajectory_vocab_and_forces_end(monkeypatch):
    model = _make_model()
    monkeypatch.setattr(Qwen3VLForConditionalGeneration, "compute_logits", lambda self, hidden: torch.zeros(3, 32))
    model._step_triggers = torch.tensor([False, True, False])
    logits = model.compute_logits(torch.zeros(3, 4))
    assert torch.isinf(logits[:, 10:15]).all()
    assert torch.isinf(logits[:, TEXT_EOS]).all()
    assert logits[1].argmax().item() == FUTURE_END
    assert torch.isinf(logits[1, :FUTURE_END]).all()
    # State is consumed: the next step only applies the masks.
    logits = model.compute_logits(torch.zeros(3, 4))
    assert not torch.isinf(logits[1, FUTURE_END + 1])


def _make_prompt_builder() -> alpamayo_openpi.Alpamayo2SuperOpenPIPrompt:
    builder = alpamayo_openpi.Alpamayo2SuperOpenPIPrompt.__new__(alpamayo_openpi.Alpamayo2SuperOpenPIPrompt)
    builder.config = Alpamayo2SuperConfig()
    builder.tokenizer = None
    builder.frames_per_camera = 4
    builder.history_steps = 16
    return builder


def _observation(**overrides):
    obs = {
        "image_frames": np.zeros((2, 4, 3, 8, 8), dtype=np.uint8),
        "camera_indices": np.array([1, 6]),
        "ego_history_xyz": np.zeros((16, 3)),
        "ego_history_rot": np.tile(np.eye(3), (16, 1, 1)),
    }
    obs.update(overrides)
    return obs


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"image_frames": np.zeros((2, 3, 3, 8, 8), dtype=np.uint8)}, "image_frames"),
        ({"image_frames": np.zeros((2, 4, 3, 8, 8), dtype=np.float32)}, "image_frames"),
        ({"camera_indices": np.array([6, 1])}, "camera_indices"),
        ({"camera_indices": np.array([1, 1])}, "camera_indices"),
        ({"camera_indices": np.array([1, 9])}, "camera_indices"),
        ({"ego_history_xyz": np.zeros((3, 16))}, "ego_history"),
    ],
)
def test_openpi_prompt_rejects_invalid_observations(overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_prompt_builder().build_prompt(_observation(**overrides))


def test_openpi_prompt_builds_reasoning_prompt(monkeypatch):
    calls: dict[str, object] = {}

    def fake_prompt(tokenizer, config, *, frames, camera_indices, ego_history_xyz):
        calls.update(frames=frames.shape, cameras=camera_indices, history=ego_history_xyz.shape)
        return [1, 2, 3]

    monkeypatch.setattr(alpamayo_openpi, "build_trajectory_prompt", fake_prompt)
    # A leading batch dimension on the history is accepted.
    prompt = _make_prompt_builder().build_prompt(_observation(ego_history_xyz=np.zeros((1, 16, 3))))
    assert calls == {"frames": (2, 4, 3, 8, 8), "cameras": [1, 6], "history": (16, 3)}
    assert prompt["prompt_token_ids"] == [1, 2, 3]
    assert len(prompt["multi_modal_data"]["image"]) == 8
