# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the DreamX-World-5B-Cam (WanCameraPipeline) runtime path.

Covers the camera pre-process (request mutation, action validation, camera
condition generation), the explicit action-frame allocation, and the
forward-side camera-condition extraction. Uses real ``OmniDiffusionRequest``
objects for the happy paths so request-structure refactors keep being caught.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.wan2_2.camera_pose_utils import (
    ActionToPoseFromID,
    _allocate_action_durations,
    build_camera_condition,
    validate_action_sequence,
)
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_camera import (
    DREAMX_NEGATIVE_PROMPT,
    Wan22CameraPipeline,
    _extract_camera_condition,
    get_wan22_camera_pre_process_func,
)
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

VALID_EXTRA_ARGS = {"action_seq": ["w", "wj"], "action_speed_list": [4, 6]}


def _make_request(
    prompt,
    *,
    extra_args=VALID_EXTRA_ARGS,
    request_id: str = "camera-test-req",
    num_frames: int = 9,
) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt=prompt,
        sampling_params=OmniDiffusionSamplingParams(
            height=704,
            width=1280,
            num_frames=num_frames,
            extra_args=dict(extra_args),
        ),
        request_id=request_id,
    )


@pytest.fixture
def preprocess():
    return get_wan22_camera_pre_process_func(SimpleNamespace())


# ---------------------------------------------------------------------------
# Pre-process: request mutation
# ---------------------------------------------------------------------------


def test_preprocess_mutates_singular_request_prompt(preprocess) -> None:
    # A real OmniDiffusionRequest only has `prompt` (not `prompts`); this is the
    # path that used to crash with AttributeError before the fix.
    request = _make_request({"prompt": "a scene"})
    result = preprocess(request)

    assert result is request
    condition = result.prompt["additional_information"]["camera_condition"]
    assert set(condition) == {"viewmats", "K"}
    assert result.prompt["prompt"] == "a scene"


def test_string_prompt_is_converted_and_retains_text(preprocess) -> None:
    result = preprocess(_make_request("just text"))

    assert not isinstance(result.prompt, str)
    assert result.prompt["prompt"] == "just text"
    assert "camera_condition" in result.prompt["additional_information"]


def test_user_negative_prompt_is_preserved(preprocess) -> None:
    result = preprocess(_make_request({"prompt": "p", "negative_prompt": "user negative"}))

    assert result.prompt["negative_prompt"] == "user negative"


def test_missing_negative_prompt_gets_dreamx_default(preprocess) -> None:
    result = preprocess(_make_request({"prompt": "p"}))

    assert result.prompt["negative_prompt"] == DREAMX_NEGATIVE_PROMPT


# ---------------------------------------------------------------------------
# Pre-process: camera condition contents
# ---------------------------------------------------------------------------


def test_camera_condition_is_cpu_float32(preprocess) -> None:
    condition = preprocess(_make_request({"prompt": "p"})).prompt["additional_information"]["camera_condition"]

    for tensor in condition.values():
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"


@pytest.mark.parametrize(
    ("num_frames", "latent_frames"),
    [(9, 3), (81, 21), (121, 31)],
)
def test_camera_condition_shapes_follow_latent_frames(preprocess, num_frames, latent_frames) -> None:
    condition = preprocess(_make_request({"prompt": "p"}, num_frames=num_frames)).prompt["additional_information"][
        "camera_condition"
    ]

    assert condition["viewmats"].shape == (latent_frames, 4, 4)
    assert condition["K"].shape == (latent_frames, 3, 3)


@pytest.mark.parametrize(("requested", "snapped"), [(10, 9), (120, 121), (121, 121)])
def test_num_frames_snapped_to_1_plus_4k_and_written_back(preprocess, requested, snapped) -> None:
    result = preprocess(_make_request({"prompt": "p"}, num_frames=requested))

    assert result.sampling_params.num_frames == snapped
    viewmats = result.prompt["additional_information"]["camera_condition"]["viewmats"]
    assert viewmats.shape[0] == 1 + (snapped - 1) // 4


# ---------------------------------------------------------------------------
# Pre-process: action validation and dummy warmup
# ---------------------------------------------------------------------------


def test_composite_action_tokens_are_accepted() -> None:
    validate_action_sequence(["wj", "sl"], [1, 2.5])


@pytest.mark.parametrize(
    "extra_args",
    [
        {"action_seq": ["w", "wj"], "action_speed_list": [4]},  # length mismatch
        {"action_seq": [], "action_speed_list": []},  # empty controls
        {"action_seq": [""], "action_speed_list": [1]},  # empty token
        {"action_seq": ["x"], "action_speed_list": [1]},  # unknown token
        {"action_seq": ["wx"], "action_speed_list": [1]},  # unknown char in composite
        {"action_seq": ["w"], "action_speed_list": ["4"]},  # non-numeric speed
        {"action_seq": ["w"], "action_speed_list": [True]},  # boolean speed
        {"action_seq": [42], "action_speed_list": [1]},  # non-string token
    ],
)
def test_malformed_controls_raise_value_error(preprocess, extra_args) -> None:
    with pytest.raises(ValueError):
        preprocess(_make_request({"prompt": "p"}, extra_args=extra_args))


def test_missing_controls_raise_for_real_request(preprocess) -> None:
    with pytest.raises(ValueError, match="action_seq"):
        preprocess(_make_request({"prompt": "p"}, extra_args={}))


def test_dummy_warmup_gets_minimal_camera_condition(preprocess) -> None:
    # The engine dummy run does not route extra_body and uses num_frames=1
    # (io_support.get_dummy_run_num_frames default); it must not raise.
    request = _make_request(
        {"prompt": "dummy run"},
        extra_args={},
        request_id=DUMMY_DIFFUSION_REQUEST_ID,
        num_frames=1,
    )
    result = preprocess(request)

    viewmats = result.prompt["additional_information"]["camera_condition"]["viewmats"]
    assert viewmats.shape == (1, 4, 4)
    assert torch.allclose(viewmats[0], torch.eye(4))


# ---------------------------------------------------------------------------
# Action-frame allocation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("num_frames", "num_actions", "expected"),
    [
        (121, 2, [60, 60]),
        (121, 3, [40, 40, 40]),
        (81, 3, [27, 26, 27]),
        (11, 3, [3, 4, 3]),
        (2, 1, [1]),
    ],
)
def test_allocate_action_durations_table(num_frames, num_actions, expected) -> None:
    assert _allocate_action_durations(num_frames, num_actions) == expected


def test_allocate_action_durations_invariants() -> None:
    for num_frames in range(2, 200):
        for num_actions in range(1, min(num_frames - 1, 8) + 1):
            durations = _allocate_action_durations(num_frames, num_actions)
            assert sum(durations) == num_frames - 1
            assert min(durations) >= 1
            assert max(durations) - min(durations) <= 1


def test_too_many_actions_for_motion_frames_raise() -> None:
    with pytest.raises(ValueError, match="at least"):
        _allocate_action_durations(3, 3)
    with pytest.raises(ValueError, match="at least"):
        build_camera_condition(["w", "a", "l"], [1, 1, 1], 0, 0, 3)


def test_num_frames_1_yields_identity_only_condition() -> None:
    # Exercised on every engine start by the dummy warmup; must never raise.
    condition = build_camera_condition(["w"], [1], 0, 0, 1)

    assert condition["viewmats"].shape == (1, 4, 4)
    assert torch.allclose(condition["viewmats"][0], torch.eye(4))


@pytest.mark.parametrize("num_frames", [5, 9, 81, 121])
def test_pose_count_exactly_matches_num_frames(num_frames) -> None:
    durations = _allocate_action_durations(num_frames, 2)
    poses = ActionToPoseFromID(["w", "wj"], [4, 6], duration=durations)

    assert len(poses) == num_frames  # 1 identity row + num_frames-1 motion rows


def test_action_to_pose_scalar_duration_form_is_unchanged() -> None:
    poses = ActionToPoseFromID(["w", "wj"], [4, 6], duration=33)

    assert len(poses) == 1 + 33 * 2


def test_action_to_pose_rejects_mismatched_duration_list() -> None:
    with pytest.raises(ValueError, match="equal length"):
        ActionToPoseFromID(["w", "wj"], [4, 6], duration=[10])


# ---------------------------------------------------------------------------
# Forward-side camera-condition extraction
# ---------------------------------------------------------------------------


def test_extract_camera_condition_adds_batch_dim(preprocess) -> None:
    request = preprocess(_make_request({"prompt": "p"}))
    condition = _extract_camera_condition(DiffusionRequestBatch(requests=[request]))

    assert condition["viewmats"].shape == (1, 3, 4, 4)
    assert condition["K"].shape == (1, 3, 3, 3)


@pytest.mark.parametrize(
    "prompts",
    [
        [{"prompt": "p"}],  # no additional_information
        [{"prompt": "p", "additional_information": {}}],  # no camera_condition
        ["bare string"],  # cannot carry a condition
        [],  # empty batch
    ],
)
def test_extract_camera_condition_raises_when_missing(prompts) -> None:
    with pytest.raises(ValueError, match="camera_condition"):
        _extract_camera_condition(SimpleNamespace(prompts=prompts))


def test_extract_camera_condition_rejects_batched_requests() -> None:
    with pytest.raises(ValueError, match="batch size 1"):
        _extract_camera_condition(SimpleNamespace(prompts=[{"prompt": "p1"}, {"prompt": "p2"}]))


def test_forward_raises_instead_of_plain_i2v_fallback() -> None:
    # Lightweight instance: the raise must happen before super().forward(), so
    # no transformer/VAE/scheduler is needed.
    pipeline = object.__new__(Wan22CameraPipeline)
    nn.Module.__init__(pipeline)

    with pytest.raises(ValueError, match="camera_condition"):
        pipeline.forward(SimpleNamespace(prompts=[{"prompt": "p"}]))


def test_dummy_warmup_survives_forward_extraction(preprocess) -> None:
    # End-to-end warmup safety: pre-process attaches the minimal condition, so
    # the forward-side raise can never fire during engine startup.
    request = _make_request(
        {"prompt": "dummy run"},
        extra_args={},
        request_id=DUMMY_DIFFUSION_REQUEST_ID,
        num_frames=1,
    )
    condition = _extract_camera_condition(DiffusionRequestBatch(requests=[preprocess(request)]))

    assert condition["viewmats"].shape == (1, 1, 4, 4)
