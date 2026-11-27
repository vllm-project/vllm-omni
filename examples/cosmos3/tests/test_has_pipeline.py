"""Exercise the production denoising loop with deterministic velocity fields."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.cosmos3.has_schedule import compute_has_local_times
from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
    Cosmos3OmniDiffusersPipeline,
    get_cosmos3_post_process_func,
)
from vllm_omni.entrypoints.cosmos3.action_ready_stream import ActionReadyPublisher


class Harness:
    def __init__(self):
        self.scheduler = object()
        self.calls = []
        self.reset_count = 0
        self.transformer = SimpleNamespace(
            reset_cache=lambda: None,
            reset_pp_step=lambda: None,
        )

    def _scheduler_step_via_pp_driver(self, noise_pred, timestep, latents, *, step_scheduler, generator):
        return step_scheduler.step(noise_pred, timestep, latents, generator=generator, return_dict=False)[0]

    def _sync_pp_send(self):
        pass

    def _pp_parallel_active(self):
        return False

    def _set_denoise_step_metadata(self, *args):
        pass

    def _clear_denoise_step_metadata(self):
        pass

    def _predict_noise_via_pp_driver(self, *, shared_kwargs, **kwargs):
        kwargs.pop("_cache_context", None)
        return self.velocity(**kwargs, **shared_kwargs)

    def velocity(self, **kwargs):
        actions = kwargs["action_latents"]
        times = kwargs.get("action_timesteps", kwargs["timestep"][:, None].expand(-1, actions.shape[1]))
        self.calls.append((times.clone(), actions.clone()))
        return torch.ones_like(kwargs["hidden_states"]), actions + 1 + times[..., None] / 1000

    def _cfg_parallel_active(self):
        return False

    def _new_cosmos3_state(self, session_id):
        return None

    def _kv_reset_und(self, state):
        pass

    def _kv_load_und(self, state, *, is_negative):
        return False

    def _set_mixed_precision_step(self, index, total):
        pass

    def _reset_mixed_precision(self):
        self.reset_count += 1

    def progress_bar(self, iterable):
        return iterable


def denoise(harness, *, stream=None, device="cpu", schedule="has_euler", early_stop=None, metrics=None):
    return Cosmos3OmniDiffusersPipeline.diffuse(
        harness,
        latents=torch.zeros(1, 1, 2, 1, 1, device=device),
        timesteps=torch.linspace(1000, 100, 10, device=device),
        cond_ids=torch.zeros(1, 1, dtype=torch.long, device=device),
        cond_mask=torch.ones(1, 1, device=device),
        uncond_ids=torch.zeros(1, 1, dtype=torch.long, device=device),
        uncond_mask=torch.ones(1, 1, device=device),
        guidance_scale=1,
        shared_kwargs={
            "_action_schedule": schedule,
            "_action_stream": stream,
            "_early_stop_actions": early_stop,
            "_sampler_metrics": metrics,
        },
        action_latents=torch.zeros(1, 16, 64, device=device),
        action_velocity_mask=torch.ones(1, 16, 1, device=device),
        raw_action_dim=10,
    )


def test_paper_schedule_finalizes_first_action_in_one_step():
    rho = torch.linspace(1, 0, 11)[:, None, None]
    times = compute_has_local_times(rho, torch.zeros(1, dtype=torch.long), 16, alpha=0.6, u0=0.9, formula="paper_v3")
    assert torch.equal(times[0], torch.ones(1, 16))
    assert times[1, 0, 0] == 0 and bool((times[1, 0, 1:] > 0).all())
    assert torch.equal(times[-1], torch.zeros(1, 16))
    assert bool((times[1:] <= times[:-1]).all())
    assert bool((times[:, :, 1:] >= times[:, :, :-1]).all())
    torch.testing.assert_close(times[:, 0, -1], rho[:, 0, 0])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_production_loop_streams_after_update_and_matches_nonstream(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    baseline = Harness()
    _, expected = denoise(baseline, device=device)
    harness = Harness()
    events = []
    closed = []

    class Session:
        def post(self, url, *, json, timeout):
            # Capture real publisher events during model execution.
            events.append((len(harness.calls), json))
            return SimpleNamespace(raise_for_status=lambda: None)

        def close(self):
            closed.append(True)

    original_init = ActionReadyPublisher.__init__

    def initialize(self, *args):
        original_init(self, *args)
        self.session.close()
        self.session = Session()

    monkeypatch.setattr(ActionReadyPublisher, "__init__", initialize)
    _, actual = denoise(
        harness, stream={"version": 1, "callback_url": "http://127.0.0.1/internal/action_events/test"}, device=device
    )
    assert len(harness.calls) == 10 and harness.reset_count == 1
    assert events[0][0] == 1 and events[0][1]["step"] == 1
    assert events[0][1]["action_indices"] == [0]
    merged = np.zeros((16, 10), dtype=np.float32)
    seen = set()
    for count, event in events:
        indices = event["action_indices"]
        assert count == event["step"]
        assert not seen.intersection(indices)
        seen.update(indices)
        merged[indices] = event["actions"]
        # Completed actions are unchanged by every subsequent denoising step.
        for _, later_actions in harness.calls[count:]:
            np.testing.assert_array_equal(later_actions[0, indices, :10].cpu(), event["actions"])
    assert seen == set(range(16)) and closed == [True]
    np.testing.assert_array_equal(merged, actual[0, :, :10].cpu())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert bool((actual[..., 10:] == 0).all())


def test_callback_failure_closes_publisher_and_resets_pipeline(monkeypatch):
    harness = Harness()
    closed = []

    def fail(*args):
        raise RuntimeError("callback failed")

    monkeypatch.setattr(ActionReadyPublisher, "after_step", fail)
    monkeypatch.setattr(ActionReadyPublisher, "close", lambda self: (self.session.close(), closed.append(True)))
    with pytest.raises(RuntimeError, match="callback failed"):
        denoise(harness, stream={"version": 1, "callback_url": "http://127.0.0.1/internal/action_events/test"})
    assert closed == [True] and harness.reset_count == 1


def test_server_modules_are_independent_of_framework():
    import sys

    import vllm_omni.entrypoints.cosmos3.policy_server  # noqa: F401

    assert not any(name == "cosmos_framework" or name.startswith("cosmos_framework.") for name in sys.modules)


def test_constant_baseline_shares_time_and_runs_all_steps():
    harness = Harness()
    _, actions = denoise(harness, schedule="constant_euler")
    assert len(harness.calls) == 10
    for index, (times, _) in enumerate(harness.calls):
        torch.testing.assert_close(times, torch.full((1, 16), float(1000 - index * 100)), rtol=0, atol=1e-4)
    torch.testing.assert_close(actions[:, 1:, :10], actions[:, :1, :10].expand(-1, 15, -1), rtol=0, atol=0)


def test_legacy_default_keeps_original_scheduler_path():
    harness = Harness()
    steps = []

    class Scheduler:
        def step(self, prediction, timestep, latents, **kwargs):
            steps.append(timestep.item())
            return (latents - 0.1 * prediction,)

    harness.scheduler = Scheduler()
    _, legacy = denoise(harness, schedule="default")
    _, constant = denoise(Harness(), schedule="constant_euler")
    assert len(steps) == 10 and steps == list(range(1000, 0, -100))
    torch.testing.assert_close(legacy, constant, rtol=1e-5, atol=1e-6)


def test_action_only_envelope_passes_real_postprocessor():
    action = torch.zeros(1, 16, 10)
    metadata = {"common": {"action_only_output": True}, "actions": {"raw_action_dim": 10}}
    postprocess = get_cosmos3_post_process_func(SimpleNamespace())
    result = postprocess({"payload": {"actions": action}, "metadata": metadata})
    assert result["payload"]["actions"] is action and result["payload"]["video"] == []
    assert result["metadata"] == metadata
