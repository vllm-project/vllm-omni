"""Exercise render cadence and compare native physics / camera conventions."""

import os
from pathlib import Path

import numpy as np
import pytest

from vllm_omni_cosmos3_client.rendering.observation import ObservationRenderEnv


def test_real_mujoco_only_draws_at_policy_observations(monkeypatch):
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    pytest.importorskip("mujoco")
    from robosuite.utils.binding_utils import MjSim

    from vllm_omni_cosmos3_client._vendor.libero import benchmark, get_libero_path
    from vllm_omni_cosmos3_client._vendor.libero.envs import OffScreenRenderEnv

    calls = []
    original = MjSim.render

    def counted(self, *args, **kwargs):
        calls.append(kwargs.get("camera_name"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MjSim, "render", counted)
    task_suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = task_suite.get_task(0)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    state = task_suite.get_task_init_states(0)[0]
    env = ObservationRenderEnv(
        OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=64,
            camera_widths=64,
            use_camera_obs=False,
        ),
        ["agentview", "wrist"],
        64,
    )
    try:
        env.seed(0)
        env.reset()
        env.set_init_state(state)
        dummy = [0, 0, 0, 0, 0, 0, -1]
        for _ in range(10):
            env.step(dummy)
        assert calls == []  # includes construction/reset/init and stabilization
        first = env.policy_observation
        assert len(calls) == 2
        assert first["agentview_image"].shape == (64, 64, 3)
        assert env.policy_observation is first  # vector hasattr/getattr doesn't redraw
        for _ in range(16):
            obs, _, _, _ = env.step(dummy)
            assert "agentview_image" not in obs
        assert len(calls) == 2
        physics_state = env.get_sim_state().copy()
        current = env.policy_observation
        np.testing.assert_array_equal(physics_state, env.get_sim_state())
        assert len(calls) == 4
        assert env.rendered_camera_frames == 4
        assert env.rendered_observations == 2
        # A forced observation using the original robosuite camera sensor must
        # match the new on-demand frame pixel for pixel, without another step.
        sensors, _ = env.env._create_camera_sensors(
            "agentview",
            cam_w=64,
            cam_h=64,
            cam_d=False,
            cam_segs=None,
            modality="image",
        )
        np.testing.assert_array_equal(current["agentview_image"], sensors[0]({}))
    finally:
        env.close()
