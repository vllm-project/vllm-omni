# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import unittest

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.motubrain2.obs_adapter import (
    build_observation,
    camera_clip,
    camera_sources,
    expected_rgb_frames,
)
from vllm_omni.diffusion.models.motubrain2.state_motubrain2 import (
    SessionStore,
    build_rtc,
    validate_rtc_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

VIEWS = [
    {"name": "cam_high", "source": "head_camera", "view_id": 0},
    {"name": "cam_left_wrist", "source": "left_camera", "view_id": 1},
]


def _frame(height=8, width=12):
    return np.zeros((height, width, 3), dtype=np.uint8)


def _obs(**overrides):
    observation = {
        "prompt": "pick up the red block",
        "head_camera": _frame(),
        "left_camera": _frame(),
    }
    observation.update(overrides)
    return observation


class ObservationTests(unittest.TestCase):
    def test_camera_sources_and_frame_count(self):
        self.assertEqual(camera_sources(VIEWS), ["head_camera", "left_camera"])
        self.assertEqual(expected_rgb_frames(1, 4), 1)
        self.assertEqual(expected_rgb_frames(2, 4), 5)

    def test_single_hwc_frame_becomes_uint8_clip(self):
        result = build_observation(_obs(), VIEWS, wire_dim=None)
        clip = result["cameras"]["head_camera"]
        self.assertEqual(tuple(clip.shape), (1, 8, 12, 3))
        self.assertEqual(clip.dtype, torch.uint8)
        self.assertEqual(result["instruction"], "pick up the red block")

    def test_chw_frame_is_transposed(self):
        frame = np.zeros((3, 8, 12), dtype=np.uint8)
        clip = camera_clip(frame, "camera")
        self.assertEqual(tuple(clip.shape), (1, 8, 12, 3))

    def test_nested_and_flat_camera_observations_match(self):
        flat = _obs()
        native = {"instruction": flat.pop("prompt"), "cameras": flat}
        result = build_observation(native, VIEWS)
        for name, clip in build_observation(_obs(), VIEWS)["cameras"].items():
            self.assertTrue(torch.equal(result["cameras"][name], clip))

    def test_frame_list_and_four_dimensional_array(self):
        frames = [_frame(), _frame() + 1]
        clip = camera_clip(frames, "camera")
        self.assertEqual(tuple(clip.shape), (2, 8, 12, 3))
        self.assertEqual(int(clip[1, 0, 0, 0]), 1)
        stacked = np.stack(frames)
        self.assertTrue(torch.equal(camera_clip(stacked, "camera"), clip))

    def test_float_frames_scale_to_the_uint8_range(self):
        frame = np.full((8, 12, 3), 0.5, dtype=np.float32)
        clip = camera_clip(frame, "camera")
        self.assertEqual(clip.dtype, torch.uint8)
        self.assertEqual(int(clip[0, 0, 0, 0]), 128)

    def test_torch_frame_moves_to_the_cpu(self):
        frame = torch.zeros(8, 12, 3, dtype=torch.uint8)
        clip = camera_clip(frame, "camera")
        self.assertEqual(clip.device.type, "cpu")
        self.assertEqual(tuple(clip.shape), (1, 8, 12, 3))

    def test_name_is_used_when_no_source_is_declared(self):
        views = [{"name": "front", "view_id": 0}]
        observation = {"prompt": "move", "front": _frame()}
        result = build_observation(observation, views)
        self.assertIn("front", result["cameras"])

    def test_missing_camera_names_the_expected_source(self):
        observation = _obs()
        observation.pop("left_camera")
        with self.assertRaisesRegex(ValueError, "left_camera"):
            build_observation(observation, VIEWS)

    def test_multi_frame_window_is_enforced(self):
        observation = {
            "prompt": "move",
            "head_camera": [_frame() for _ in range(5)],
            "left_camera": [_frame() for _ in range(5)],
        }
        result = build_observation(observation, VIEWS, observed_frames=2, temporal_stride=4)
        self.assertEqual(tuple(result["cameras"]["head_camera"].shape), (5, 8, 12, 3))
        observation["head_camera"] = [_frame()]
        with self.assertRaisesRegex(ValueError, "must carry 5 RGB frames"):
            build_observation(observation, VIEWS, observed_frames=2, temporal_stride=4)

    def test_state_and_camera_extrinsic_are_validated(self):
        observation = _obs(
            state=np.array([1.0, 2.0], dtype=np.float32),
            camera_extrinsic=np.eye(4, dtype=np.float32),
        )
        result = build_observation(observation, VIEWS, wire_dim=2)
        self.assertEqual(tuple(result["state"].shape), (2,))
        self.assertEqual(tuple(result["camera_extrinsic"].shape), (4, 4))
        with self.assertRaisesRegex(ValueError, "state must have 3"):
            build_observation(observation, VIEWS, wire_dim=3)
        observation["camera_extrinsic"] = np.eye(3, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "4x4"):
            build_observation(observation, VIEWS)

    def test_missing_instruction_is_rejected(self):
        observation = _obs()
        observation.pop("prompt")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            build_observation(observation, VIEWS)

    def test_sparse_history_requires_client_frames(self):
        with self.assertRaisesRegex(ValueError, "history_cameras"):
            build_observation(_obs(), VIEWS, history_frames=5)
        history = {
            "head_camera": [_frame() for _ in range(5)],
            "left_camera": [_frame() for _ in range(5)],
        }
        result = build_observation(_obs(history_cameras=history), VIEWS, history_frames=5)
        self.assertEqual(tuple(result["history_cameras"]["head_camera"].shape), (5, 8, 12, 3))
        with self.assertRaisesRegex(ValueError, "does not use sparse history"):
            build_observation(_obs(history_cameras=history), VIEWS)


class SessionStateTests(unittest.TestCase):
    def test_store_is_lru_bounded_and_resettable(self):
        store = SessionStore(max_sessions=2)
        first = store.get("a")
        store.get("b")
        self.assertIs(store.get("a"), first)
        store.get("c")
        self.assertEqual(len(store), 2)
        self.assertIsNone(store.peek("b"))
        store.reset("a")
        self.assertIsNone(store.peek("a"))

    def test_rtc_config_defaults_and_validation(self):
        self.assertIsNone(validate_rtc_config(None, 24))
        self.assertIsNone(validate_rtc_config({"enabled": False}, 24))
        config = validate_rtc_config({"enabled": True}, 24)
        self.assertEqual(config, {"enabled": True, "execute_prefix": 1, "delay": 23})
        config = validate_rtc_config({"enabled": True, "execute_prefix": 2, "delay": 4}, 24)
        self.assertEqual(config["delay"], 4)
        with self.assertRaisesRegex(ValueError, "execute_prefix"):
            validate_rtc_config({"enabled": True, "execute_prefix": -1}, 24)
        with self.assertRaisesRegex(ValueError, "smaller than"):
            validate_rtc_config({"enabled": True, "execute_prefix": 24}, 24)
        with self.assertRaisesRegex(ValueError, "delay"):
            validate_rtc_config({"enabled": True, "delay": 0}, 24)

    def test_rtc_prefix_uses_the_unexecuted_previous_actions(self):
        pytest.importorskip("motubrain2.inference")
        store = SessionStore()
        session = store.get("robot")
        config = validate_rtc_config({"enabled": True, "execute_prefix": 1, "delay": 2}, 24)
        self.assertIsNone(build_rtc(session, config, 24))
        session.last_action = np.arange(24, dtype=np.float32).reshape(6, 4)
        rtc = build_rtc(session, config, 24)
        self.assertIsNotNone(rtc)
        anchor, future = rtc.prefix(4, 24)
        self.assertIsNone(anchor)
        self.assertTrue(np.array_equal(future.numpy(), session.last_action[1:3]))

    def test_rtc_is_skipped_when_the_prefix_is_exhausted(self):
        session = SessionStore().get("robot")
        config = validate_rtc_config({"enabled": True, "execute_prefix": 3, "delay": 1}, 4)
        session.last_action = np.zeros((3, 2), dtype=np.float32)
        self.assertIsNone(build_rtc(session, config, 4))


if __name__ == "__main__":
    unittest.main()
