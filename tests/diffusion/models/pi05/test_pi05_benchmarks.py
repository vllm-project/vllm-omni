# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio

import numpy as np
import pytest

from benchmarks.diffusion.pi05.benchmark_openpi import make_observation, summarize

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_robot_action_comparison_rejects_matching_nonfinite_outputs():
    from tests.helpers.assertions import assert_robot_action_chunks_close

    with pytest.raises(AssertionError, match="Non-finite"):
        assert_robot_action_chunks_close([float("nan")], [float("nan")], atol=1e-4, rtol=1e-5)
    assert_robot_action_chunks_close([[0.1]], [[0.1]], atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("views", [1, 2, 3])
def test_benchmark_observations_are_replayable(views):
    first, second = make_observation(42, views), make_observation(42, views)
    np.testing.assert_array_equal(first["state"], second["state"])
    assert len(first["images"]) == views
    for key in first["images"]:
        np.testing.assert_array_equal(first["images"][key], second["images"][key])
    assert first["sampling_params"]["num_inference_steps"] == 10


def test_benchmark_summary_does_not_count_errors_as_throughput():
    result = summarize([{"latency_ms": 1, "error": None}, {"latency_ms": 99, "error": "failed"}], 2)
    assert result == {"completed": 1, "failed": 1, "req_per_s": 0.5, "p50_ms": 1, "p95_ms": 1, "p99_ms": 1}


def test_openpi_wave_uses_official_codec():
    codec = pytest.importorskip("openpi_client.msgpack_numpy")
    from benchmarks.diffusion.pi05.benchmark_openpi import run_wave

    class Connection:
        async def send(self, message):
            self.observation = codec.unpackb(message)

        async def recv(self):
            return codec.Packer().pack(np.zeros((50, 32), np.float32))

    connection = Connection()
    rows, _ = asyncio.run(
        run_wave(
            [connection],
            [make_observation(42, 1)],
            stagger_ms=0,
            timeout=2,
            save_actions=True,
        )
    )
    assert rows[0]["error"] is None
    assert np.asarray(rows[0]["actions"]).shape == (50, 32)
    assert connection.observation["seed"] == 42
