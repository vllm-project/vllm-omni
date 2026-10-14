# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SANA-WM request normalization."""

import pytest

from vllm_omni.diffusion.models.sana_wm.request import (
    SANA_WM_DEFAULT_HEIGHT,
    SANA_WM_DEFAULT_NUM_FRAMES,
    SANA_WM_DEFAULT_WIDTH,
    normalize_sana_wm_payload,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _prompt(**sana_wm):
    """A valid request, with the action rollout kept in step with num_frames."""
    block = {"num_frames": SANA_WM_DEFAULT_NUM_FRAMES, **sana_wm}
    block.setdefault("action", f"w-{block['num_frames'] - 1}")
    return {
        "prompt": "a quiet city street",
        "multi_modal_data": {"image": object()},
        "sana_wm": block,
    }


def _identity_poses(count):
    return [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] for _ in range(count)]


def _camera_prompt(**camera):
    prompt = _prompt(num_frames=9)
    prompt["sana_wm"].pop("action")
    prompt["sana_wm"]["camera"] = {"poses": _identity_poses(9), **camera}
    return prompt


def test_defaults_are_latent_aligned():
    payload = normalize_sana_wm_payload(_prompt())["additional_information"]["sana_wm"]
    assert payload["num_frames"] == SANA_WM_DEFAULT_NUM_FRAMES
    assert payload["height"] == SANA_WM_DEFAULT_HEIGHT
    assert payload["width"] == SANA_WM_DEFAULT_WIDTH


@pytest.mark.parametrize("num_frames", [9, 33, 161])
def test_accepts_aligned_geometry(num_frames):
    payload = normalize_sana_wm_payload(_prompt(num_frames=num_frames))["additional_information"]["sana_wm"]
    assert payload["num_frames"] == num_frames


@pytest.mark.parametrize("num_frames", [8, 10, 160])
def test_rejects_unaligned_num_frames(num_frames):
    # The VAE compresses 8x temporally; an unaligned count would be floored.
    with pytest.raises(ValueError, match="num_frames"):
        normalize_sana_wm_payload(_prompt(num_frames=num_frames))


@pytest.mark.parametrize(("height", "width"), [(700, 1280), (704, 1270), (703, 1279)])
def test_rejects_unaligned_resolution(height, width):
    with pytest.raises(ValueError, match="divisible by 32"):
        normalize_sana_wm_payload(_prompt(height=height, width=width))


def test_normalization_is_idempotent():
    once = normalize_sana_wm_payload(_prompt(num_frames=33))
    twice = normalize_sana_wm_payload(once)
    assert once["additional_information"]["sana_wm"] == twice["additional_information"]["sana_wm"]


def test_rejects_action_shorter_than_num_frames():
    # Used to shrink the camera tensor silently, then fail deep in the
    # transformer on a camera/latent frame mismatch.
    with pytest.raises(ValueError, match="rolls out 33 frames but num_frames is 161"):
        normalize_sana_wm_payload(_prompt(action="w-32"))


def test_rejects_action_longer_than_num_frames():
    with pytest.raises(ValueError, match="rolls out 161 frames but num_frames is 9"):
        normalize_sana_wm_payload(_prompt(num_frames=9, action="w-160"))


def test_rejects_oversized_action_without_expanding_it():
    # A 12-byte request body: expanding this rollout before checking the frame
    # count allocated ~16 GB. Rejection has to be arithmetic, so this test also
    # fails (by hanging/OOM) if the duration is ever expanded again.
    with pytest.raises(ValueError, match="rolls out 1000000001 frames"):
        normalize_sana_wm_payload(_prompt(action="w-1000000000"))


@pytest.mark.parametrize("action", ["", "   ", "w-0", "w-abc", "w", "x-4"])
def test_rejects_malformed_action(action):
    with pytest.raises(ValueError):
        normalize_sana_wm_payload(_prompt(action=action))


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_rejects_unusable_focal_length(value):
    # compute_raymap divides by fx/fy; a zero or non-finite focal length turns
    # the whole Plucker map into NaN and the response into a NaN video.
    intrinsics = {"fx": value, "fy": 640, "cx": 640, "cy": 352}
    with pytest.raises(ValueError, match="fx"):
        normalize_sana_wm_payload(_prompt(intrinsics=intrinsics))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_rejects_non_finite_principal_point(value):
    intrinsics = {"fx": 640, "fy": 640, "cx": value, "cy": 352}
    with pytest.raises(ValueError, match="must be finite"):
        normalize_sana_wm_payload(_prompt(intrinsics=intrinsics))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_rejects_non_finite_speeds(value):
    # `parsed <= 0` is False for NaN, so the positivity check alone let it pass.
    with pytest.raises(ValueError, match="must be finite"):
        normalize_sana_wm_payload(_prompt(translation_speed=value))


def test_rejects_non_finite_pose_entry():
    prompt = _camera_prompt()
    prompt["sana_wm"]["camera"]["poses"][3][2][1] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        normalize_sana_wm_payload(prompt)


def test_accepts_default_camera_metadata():
    payload = normalize_sana_wm_payload(_camera_prompt())["additional_information"]["sana_wm"]
    assert payload["camera"]["poses"]
    # The unread fields are not carried into the canonical payload.
    assert set(payload["camera"]) == {"poses"}


@pytest.mark.parametrize(
    ("field", "value"),
    [("format", "w2c_4x4"), ("coordinate_system", "opengl")],
)
def test_rejects_unsupported_camera_convention(field, value):
    # Nothing downstream reads these, so accepting one would mean interpreting
    # the poses under the wrong convention.
    with pytest.raises(ValueError, match=field.split("_")[0]):
        normalize_sana_wm_payload(_camera_prompt(**{field: value}))
