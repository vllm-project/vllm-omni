"""Check PI-compatible call boundaries without GPU timing or wall-clock sleeps."""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.entrypoints.cosmos3 import libero_remote_policy
from vllm_omni.entrypoints.cosmos3.libero_remote_policy import LiberoRemotePolicyService, RemotePolicyConfig


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("streaming", [False, True])
def test_policy_wall_includes_preparation_queue_cleanup_and_dump_but_not_request_gaps(
    tmp_path, monkeypatch, batch_size, streaming
):
    clock = SimpleNamespace(now=0.0)

    def advance(seconds):
        clock.now += seconds

    monkeypatch.setattr(
        libero_remote_policy, "time", SimpleNamespace(perf_counter=lambda: clock.now, time_ns=lambda: 1)
    )

    class QueueLock:
        def __enter__(self):
            advance(0.020)

        def __exit__(self, *args):
            pass

    class Transport:
        policy_profile = "libero10_native_nonstream_v1"
        action_time_schedule = "has_euler" if streaming else "constant_euler"
        early_stop_actions = None

        def predict_prepared(self, pixels, *, prompt, frame_size, metrics, action_stream=None):
            advance(0.030)  # The transport call includes job completion and cleanup.
            metrics["worker_inference_ms"] = 7.0
            if action_stream is not None:
                token = action_stream["callback_url"].rsplit("/", 1)[-1]
                for step, indices in [(1, [0]), (10, list(range(1, 16)))]:
                    service.streams.accept(
                        token,
                        {
                            "type": "partial",
                            "step": step,
                            "total_steps": 10,
                            "action_indices": indices,
                            "actions": np.zeros((len(indices), 10)).tolist(),
                            "policy_profile": self.policy_profile,
                            "action_stream_version": 1,
                        },
                    )
            return np.zeros((16, 10), dtype=np.float32)

    service = LiberoRemotePolicyService(
        Transport(),
        RemotePolicyConfig(dump_dir=tmp_path, streaming_enabled=streaming, callback_base_url="http://callback"),
    )
    service._lock = QueueLock()
    pixels = np.zeros((256, 512, 3), dtype=np.uint8)

    def prepare(item):
        advance(0.010)
        return pixels, "test caption", [256, 320, 160, 320], pixels

    def denormalize(actions):
        advance(0.040)
        return actions.tolist()

    monkeypatch.setattr(service, "_prepare", prepare)
    monkeypatch.setattr(service.normalizer, "denormalize", denormalize)
    monkeypatch.setattr(libero_remote_policy.Image.Image, "save", lambda *args: advance(0.005))

    def on_actions_ready(event):
        advance(0.050)  # Publishing an event must remain inside policy wall time.

    for request_id in (1, 2):
        request = {"request_id": request_id, "items": [{} for _ in range(batch_size)]}
        result = service.predict_distributed(
            request, is_batch=batch_size > 1, on_actions_ready=on_actions_ready if streaming else None
        )
        # PI's wrapper encloses the entire policy call, including diagnostics.
        callback_ms = 2 * (40 + 50) if streaming else 0
        expected_ms = 20 + batch_size * (10 + 30 + 40 + 10 + callback_ms)
        assert result["benchmark"]["policy_wall_ms"] == pytest.approx(expected_ms)
        assert result["benchmark"]["timing_schema"] == "pi_libero_v1"
        assert result["policy_timing"]["infer_ms"] == result["server_timing"]["infer_ms"]
        assert result["policy_timing"]["infer_ms"] == pytest.approx(expected_ms)
        assert result["policy_timing"]["backend_ms"] == pytest.approx((30 + callback_ms) * batch_size)
        assert result["timing_ms"]["queue_wait"] == pytest.approx(20)
        assert result["timing_ms"]["full_chunk"] == pytest.approx(expected_ms)
        if streaming:
            assert result["timing_ms"]["ttfa"] == pytest.approx(batch_size * 10 + 20 + 30 + 40)
            assert max(row["ready_ms"] for row in result["performance"]["action_timings"]) < expected_ms
        else:
            assert result["timing_ms"]["ttfa"] == pytest.approx(expected_ms)
            assert all(row["ready_ms"] == pytest.approx(expected_ms) for row in result["performance"]["action_timings"])
        assert all(row["worker_inference_ms"] == 7.0 for row in result["performance"]["worker_metrics"])
        advance(500)  # Environment execution / waiting before the next request.
