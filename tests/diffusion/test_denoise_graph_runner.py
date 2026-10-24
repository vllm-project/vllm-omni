# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.graph as graph_mod
import vllm_omni.diffusion.profiler.diffusion_pipeline_profiler as profiler_mod
from vllm_omni.diffusion.graph import DenoiseGraphRunner, build_denoise_graph_runner, write_through

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _FakeGraph:
    def __init__(self) -> None:
        self.replays = 0
        self.replay_fn = None

    def replay(self) -> None:
        self.replays += 1
        self.replay_fn()


class _FakeGraphs:
    """Stands in for torch.cuda/xpu graphs on CPU.

    A call made under capture is recorded against the exact tensors it received,
    and its output is blanked, because a real capture records kernels without
    running them. Replay recomputes into that same output tensor.
    """

    def __init__(self) -> None:
        self.capturing: _FakeGraph | None = None
        self.graphs: list[_FakeGraph] = []

    def new_graph(self, device):
        del device
        graph = _FakeGraph()
        self.graphs.append(graph)
        return graph

    @contextmanager
    def context(self, graph, device, pool):
        del device, pool
        self.capturing = graph
        try:
            yield
        finally:
            self.capturing = None

    def wrap(self, fn):
        def body(**kwargs):
            out = fn(**kwargs)
            graph = self.capturing
            if graph is not None:
                graph.replay_fn = lambda: out.copy_(fn(**kwargs))
                out.fill_(float("nan"))
            return out

        return body


@pytest.fixture
def fake_graphs(monkeypatch):
    fakes = _FakeGraphs()
    monkeypatch.setattr(graph_mod, "graphs_supported", lambda device: (True, ""))
    monkeypatch.setattr(graph_mod, "_is_capturing", lambda device: False)
    monkeypatch.setattr(graph_mod, "_synchronize", lambda device: None)
    monkeypatch.setattr(graph_mod, "_graph_pool_handle", lambda device: None)
    monkeypatch.setattr(graph_mod, "_new_graph", fakes.new_graph)
    monkeypatch.setattr(graph_mod, "_graph_context", fakes.context)
    return fakes


def _affine(x: torch.Tensor, mode: str = "gen_image") -> torch.Tensor:
    return x * 2 + (1 if mode == "gen_image" else -1)


def test_replay_reads_fresh_inputs_and_returns_static_output(fake_graphs):
    calls = []

    def fn(**kwargs):
        calls.append(kwargs)
        return _affine(**kwargs)

    runner = DenoiseGraphRunner(fn=fake_graphs.wrap(fn), name="t")
    first = runner(x=torch.tensor([1.0, 2.0]), mode="gen_image")
    torch.testing.assert_close(first, torch.tensor([3.0, 5.0]))
    calls_after_capture = len(calls)

    second = runner(x=torch.tensor([10.0, 20.0]), mode="gen_image")

    torch.testing.assert_close(second, torch.tensor([21.0, 41.0]))
    assert second.data_ptr() == first.data_ptr()
    assert len(calls) == calls_after_capture + 1
    assert runner.stats_snapshot() == {"calls": 2, "hits": 1, "captures": 1, "eager": 0, "cache_size": 1}


def test_nested_tensor_arguments_are_refreshed(fake_graphs):
    runner = DenoiseGraphRunner(fn=fake_graphs.wrap(lambda xs: xs[0] + xs[1]), name="t")
    runner(xs=[torch.ones(2), torch.ones(2)])

    out = runner(xs=[torch.full((2,), 3.0), torch.full((2,), 4.0)])

    torch.testing.assert_close(out, torch.full((2,), 7.0))
    assert runner.stats_snapshot()["hits"] == 1


def test_new_shape_or_flag_captures_separately_with_lru_eviction(fake_graphs):
    runner = DenoiseGraphRunner(fn=fake_graphs.wrap(_affine), name="t", max_graphs=2)

    runner(x=torch.zeros(2), mode="gen_image")
    runner(x=torch.zeros(3), mode="gen_image")
    out = runner(x=torch.zeros(3), mode="other")

    torch.testing.assert_close(out, torch.full((3,), -1.0))
    assert runner.stats_snapshot()["captures"] == 3
    assert runner.stats_snapshot()["cache_size"] == 2

    runner(x=torch.zeros(2), mode="gen_image")
    assert runner.stats_snapshot()["captures"] == 4


def test_mismatched_constants_for_cached_key_run_eager_and_drop_entry(fake_graphs):
    runner = DenoiseGraphRunner(fn=fake_graphs.wrap(_affine), name="t")
    runner(x=torch.zeros(2), mode="gen_image")
    next(iter(runner._cache.values())).constants["mode"] = "stale"

    out = runner(x=torch.ones(2), mode="gen_image")

    torch.testing.assert_close(out, torch.full((2,), 3.0))
    assert runner.stats_snapshot()["eager"] == 1
    assert runner.stats_snapshot()["cache_size"] == 0
    assert runner.enabled


def test_invalidate_drops_graphs_but_keeps_capturing(fake_graphs):
    runner = DenoiseGraphRunner(fn=fake_graphs.wrap(_affine), name="t")
    runner(x=torch.zeros(2))

    runner.invalidate("buffers moved")
    out = runner(x=torch.ones(2))

    torch.testing.assert_close(out, torch.full((2,), 3.0))
    assert runner.enabled
    assert runner.stats_snapshot()["captures"] == 2


def test_capture_failure_disables_runner_and_falls_back_to_eager(fake_graphs, monkeypatch):
    def failing_context(graph, device, pool):
        raise RuntimeError("UR_RESULT_ERROR_UNSUPPORTED_FEATURE")

    monkeypatch.setattr(graph_mod, "_graph_context", failing_context)
    runner = DenoiseGraphRunner(fn=_affine, name="t")

    first = runner(x=torch.ones(2))
    second = runner(x=torch.zeros(2))

    torch.testing.assert_close(first, torch.full((2,), 3.0))
    torch.testing.assert_close(second, torch.full((2,), 1.0))
    assert not runner.enabled
    assert runner.stats_snapshot()["captures"] == 0
    assert runner.stats_snapshot()["eager"] == 2


def test_non_tensor_return_disables_runner(fake_graphs):
    runner = DenoiseGraphRunner(fn=lambda x: {"out": x}, name="t")

    out = runner(x=torch.ones(2))

    assert isinstance(out, dict)
    assert not runner.enabled


def test_unsupported_device_disables_runner(fake_graphs, monkeypatch):
    monkeypatch.setattr(graph_mod, "graphs_supported", lambda device: (False, "no graphs"))
    runner = DenoiseGraphRunner(fn=_affine, name="t")

    torch.testing.assert_close(runner(x=torch.ones(2)), torch.full((2,), 3.0))
    assert not runner.enabled
    assert not fake_graphs.graphs


@pytest.mark.parametrize(
    ("kwargs", "outer_capture"),
    [({"x": torch.ones(2)}, True), ({"x": 1.0}, False)],
    ids=["outer-capture-in-flight", "no-tensor-arguments"],
)
def test_runs_eager_without_disabling(fake_graphs, monkeypatch, kwargs, outer_capture):
    monkeypatch.setattr(graph_mod, "_is_capturing", lambda device: outer_capture)
    runner = DenoiseGraphRunner(fn=_affine, name="t")

    runner(**kwargs)

    assert runner.enabled
    assert runner.stats_snapshot()["eager"] == 1
    assert not fake_graphs.graphs


def test_graphs_supported_rejects_cpu():
    ok, reason = graph_mod.graphs_supported(torch.device("cpu"))
    assert not ok
    assert "cpu" in reason


def test_write_through_reuses_storage_only_when_layout_matches():
    dst = torch.zeros(2, 3)
    same = torch.ones(2, 3)
    assert write_through(dst, same) is dst
    torch.testing.assert_close(dst, same)

    other_shape = torch.ones(3, 3)
    assert write_through(dst, other_shape) is other_shape
    other_dtype = torch.ones(2, 3, dtype=torch.float16)
    assert write_through(dst, other_dtype) is other_dtype
    assert write_through(None, same) is same


@pytest.mark.parametrize(("capturing", "expected_syncs"), [(False, 1), (True, 0)])
def test_profiler_skips_synchronize_during_capture(monkeypatch, capturing, expected_syncs):
    syncs = []
    monkeypatch.setattr(
        profiler_mod,
        "current_omni_platform",
        SimpleNamespace(is_available=lambda: True, synchronize=lambda: syncs.append(1)),
    )
    monkeypatch.setattr(profiler_mod, "_capture_in_flight", lambda: capturing)

    profiler_mod._sync_if_useful()

    assert len(syncs) == expected_syncs


def _od_config(**overrides):
    fields = dict(
        diffusion_use_graph=True,
        cache_backend="none",
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_build_runner_when_requested_and_safe():
    runner = build_denoise_graph_runner(_od_config(), lambda x: x, "T")
    assert isinstance(runner, DenoiseGraphRunner)
    assert runner.name == "T"


@pytest.mark.parametrize(
    ("overrides", "refusal"),
    [
        ({"diffusion_use_graph": False}, None),
        ({"cache_backend": "cache_dit"}, None),
        ({"enable_cpu_offload": True}, None),
        ({"enable_layerwise_offload": True}, None),
        ({"parallel_config": SimpleNamespace(pipeline_parallel_size=2)}, None),
        ({}, "pipeline-specific reason"),
    ],
)
def test_build_runner_stays_eager_when_unsafe(overrides, refusal):
    assert build_denoise_graph_runner(_od_config(**overrides), lambda x: x, "T", refusal=refusal) is None


def test_is_capturing_is_false_for_cpu():
    assert graph_mod.is_capturing(torch.device("cpu")) is False


@pytest.mark.parametrize(
    ("use_graph", "has_runner", "warns"),
    [(True, False, True), (True, True, False), (False, False, False)],
)
def test_warn_if_denoise_graph_unsupported(monkeypatch, use_graph, has_runner, warns):
    warnings = []
    monkeypatch.setattr(graph_mod.logger, "warning", lambda *args, **kwargs: warnings.append(args))
    pipeline = SimpleNamespace(_denoise_graph_runner=None) if has_runner else SimpleNamespace()

    graph_mod.warn_if_denoise_graph_unsupported(SimpleNamespace(diffusion_use_graph=use_graph), pipeline)

    assert bool(warnings) is warns
