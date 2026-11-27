"""A closed-loop episode still applies stabilization actions before policy work."""

import json
from types import SimpleNamespace

from vllm_omni_cosmos3_client.closed_loop_eval import _run_episode
from vllm_omni_cosmos3_client.timeline import TIMELINE


def test_dummy_steps_advance_physics_and_have_an_independent_span(monkeypatch, tmp_path):
    timeline = tmp_path / "client_timeline.jsonl"
    monkeypatch.setenv("COSMOS3_CLIENT_TIMELINE", str(timeline))

    class Environment:
        def __init__(self):
            self.actions = []

        def reset(self):
            return None

        def get_observation(self):
            return {}

        def step(self, action):
            self.actions.append(action)
            return {}, 0, False, {}

    env = Environment()
    result = _run_episode(
        env,
        SimpleNamespace(task_id=4, episode=2),
        cameras=["agentview"],
        flip_images=False,
        rotate_180=True,
        action_horizon=16,
        action_dim=10,
        action_space="frame_wise_relative",
        rotation_space="6d",
        gripper_mode="zero_one",
        max_steps=10,
        warmup_steps=10,
        initial_state=None,
        gif_path=None,
        gif_fps=20,
    )
    TIMELINE.flush()
    events = [json.loads(line) for line in timeline.read_text().splitlines()]
    assert len(env.actions) == 10
    assert all(action == [0.0] * 6 + [-1.0] for action in env.actions)
    assert result.steps == len(result.actions) == 10
    assert [event["stage"] for event in events] == ["client.env_reset", "client.dummy_stabilization"]
    dummy = events[-1]
    assert dummy["task_id"] == 4 and dummy["episode"] == 2
    assert dummy["end_s"] > dummy["start_s"]
