# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for the benchmark, independent of GPU availability."""

import os
from types import SimpleNamespace

import pytest

from benchmarks.kernels import benchmark_magi2_bf16_moe as bench

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_shape_sweep_and_cli_validation():
    assert bench.parse_args(["--tokens", "2", "10000"]).tokens == [2, 10000]
    for args in (["--iters", "0"], ["--warmup", "-1"], ["--top-k", "257"], ["--tokens", "0"]):
        with pytest.raises(SystemExit):
            bench.parse_args(args)


@pytest.mark.parametrize("values", [[], [0], [-1], [float("nan")], [float("inf")]])
def test_invalid_timings_are_not_reported(values):
    with pytest.raises(ValueError):
        bench.stats(values)


def test_stats_preserve_all_samples():
    result = bench.stats([1.0, 4.0, 2.0, 3.0])
    assert result["p50_us"] == 2.5
    assert result["p90_us"] == 4.0
    assert result["samples_us"] == [1.0, 4.0, 2.0, 3.0]


def test_switch_restores_environment_on_error(monkeypatch):
    monkeypatch.setenv("MAGI2_DETERMINISTIC", "1")
    with pytest.raises(RuntimeError), bench.env_switch(True):
        assert os.environ["MAGI2_DETERMINISTIC"] == "0"
        raise RuntimeError("compile failure")
    assert os.environ["MAGI2_DETERMINISTIC"] == "1"


def test_routed_reference_sort_is_inside_every_timed_call(monkeypatch):
    data = (SimpleNamespace(device="device"), *(object() for _ in range(5)))
    events = []
    monkeypatch.setattr(bench, "make_inputs", lambda *args: data)
    monkeypatch.setattr(bench, "check_and_capture", lambda *args: ([], []))

    def sort(*args):
        assert args == (data[1], data[2], 256)
        events.append("sort")
        return (1, 2, 3)

    def legacy(*args, **kwargs):
        assert kwargs == {"deterministic": True}
        events.append("reference")

    def candidate(*args):
        assert args == (data[0], data[1], data[2], "packed-w13", data[5], "route-buffers")
        events.append("candidate")

    def measure(torch, calls, args):
        assert events == ["pack", "allocate"]
        events.clear()
        for _ in range(2):
            for call in calls.values():
                call()
        return {name: {"synchronized_wall": {"p50_us": 1}} for name in calls}

    monkeypatch.setattr(bench, "measure", measure)

    def pack_w13(*args):
        events.append("pack")
        return "packed-w13"

    def allocate_route_buffers(*args):
        events.append("allocate")
        return "route-buffers"

    moe = SimpleNamespace(
        global_sort_routes=sort,
        triton_mh_moe_forward=legacy,
        _bf16_fused_moe_forward=candidate,
        _pack_bf16_w13=pack_w13,
        _allocate_bf16_route_buffers=allocate_route_buffers,
    )
    result = bench.benchmark(None, moe, None, 2, bench.parse_args(["--mode", "routed"]))
    assert events == ["sort", "reference", "candidate"] * 2
    assert result["mode"] == "routed"
    assert "env_off" not in str(result["timings"])
