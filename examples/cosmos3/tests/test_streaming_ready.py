"""A ready environment must execute without waiting for another sample's job."""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import pytest
from closed_loop_eval import _BatchStreamingActionRequest


def test_first_environment_actions_arrive_before_second_job_finishes():
    release = threading.Event()

    class Client:
        timeout = 2
        action_chunk_size = 2

        def predict_batch_stream(self, observations, callback):
            callback([{"sample_index": 0, "action_indices": [0, 1], "actions": [[1], [2]]}])
            assert release.wait(timeout=2)
            callback([{"sample_index": 1, "action_indices": [0, 1], "actions": [[3], [4]]}])
            return {"actions": [[[1], [2]], [[3], [4]]]}

    request = _BatchStreamingActionRequest(Client(), [[], []], 2)
    try:
        assert request.next_ready_actions([0, 1]) == {0: [1]}
        assert request.next_ready_actions([0, 1]) == {0: [2]}
        assert request._thread.is_alive()
        release.set()
        assert request.next_ready_actions([1]) == {1: [3]}
        assert request.next_ready_actions([1]) == {1: [4]}
        assert len(request.finish()["actions"]) == 2
    finally:
        release.set()


def test_premature_stream_end_is_an_error():
    class Client:
        timeout = 2
        action_chunk_size = 1

        def predict_batch_stream(self, observations, callback):
            return {"actions": []}

    request = _BatchStreamingActionRequest(Client(), [[]], 1)
    with pytest.raises(RuntimeError, match="Batch streaming inference failed"):
        request.next_ready_actions([0])
