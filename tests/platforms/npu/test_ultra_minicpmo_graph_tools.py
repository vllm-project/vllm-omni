# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.platforms.npu.graph_tools import (
    CapturedDeviceGraph,
    NPUExactGraphRunner,
)
from vllm_omni.platforms.npu.models import minicpmo_4_5_code2wav as code2wav_patch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_captured_graph_replay_clones_persistent_outputs():
    static_input = torch.zeros(1)
    static_output = torch.zeros(1)

    class _Graph:
        def replay(self):
            static_output.copy_(static_input * 2)

    graph = CapturedDeviceGraph(
        graph=_Graph(),
        static_inputs=(static_input,),
        static_outputs=(static_output,),
    )

    first = graph.replay((torch.tensor([2.0]),))[0]
    second = graph.replay((torch.tensor([3.0]),))[0]

    torch.testing.assert_close(first, torch.tensor([4.0]))
    torch.testing.assert_close(second, torch.tensor([6.0]))
    assert first.data_ptr() != second.data_ptr()


def test_capture_uses_vllm_global_graph_pool(monkeypatch):
    runner = NPUExactGraphRunner()
    global_pool = object()
    seen_pools = []
    synchronizations = 0

    class _Graph:
        def replay(self):
            pass

    @contextmanager
    def graph_context(graph, *, pool):
        del graph
        seen_pools.append(pool)
        yield

    def synchronize():
        nonlocal synchronizations
        synchronizations += 1

    fake_npu = SimpleNamespace(
        NPUGraph=_Graph,
        graph=graph_context,
        synchronize=synchronize,
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    import vllm.platforms

    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        SimpleNamespace(get_global_graph_pool=lambda: global_pool),
    )

    captured = runner.capture((torch.tensor([2.0]),), lambda value: (value + 1,))

    assert seen_pools == [global_pool]
    assert synchronizations == 2
    torch.testing.assert_close(captured.static_outputs[0], torch.tensor([3.0]))


def test_exact_signature_dispatch_captures_then_replays(monkeypatch):
    runner = NPUExactGraphRunner(max_graphs=2)
    replay_count = 0

    class _FunctionalGraph:
        def __init__(self, compute):
            self.compute = compute

        def replay(self, inputs):
            nonlocal replay_count
            replay_count += 1
            return tuple(output.detach().clone() for output in self.compute(*inputs))

    monkeypatch.setattr(runner, "_eligible", lambda inputs: True)
    monkeypatch.setattr(runner, "capture", lambda inputs, compute: _FunctionalGraph(compute))

    def compute(value):
        return (value * 2,)

    first = runner.run("unit", (torch.tensor([2.0]),), (False,), compute)
    second = runner.run("unit", (torch.tensor([3.0]),), (False,), compute)

    torch.testing.assert_close(first[0], torch.tensor([4.0]))
    torch.testing.assert_close(second[0], torch.tensor([6.0]))
    assert runner.stats == {"captures": 1, "failed": 0, "hits": 1}
    assert replay_count == 1


def test_capture_failure_is_fatal_for_stage_process(monkeypatch):
    runner = NPUExactGraphRunner(
        component_name="test component",
        disable_config_hint="disable the test graph",
    )
    computes = 0

    monkeypatch.setattr(runner, "_eligible", lambda inputs: True)
    monkeypatch.setattr(
        runner,
        "capture",
        lambda inputs, compute: (_ for _ in ()).throw(RuntimeError("unsupported op")),
    )

    def compute(value):
        nonlocal computes
        computes += 1
        return (value + 1,)

    with pytest.raises(RuntimeError, match="restart the stage process"):
        runner.run("unit", (torch.tensor([1.0]),), (False,), compute)
    with pytest.raises(RuntimeError, match="cannot continue"):
        runner.run("other", (torch.tensor([1.0, 2.0]),), (True,), compute)

    assert computes == 1
    assert runner.stats == {"captures": 0, "failed": 1, "hits": 0}


def test_graph_limit_keeps_unseen_signatures_on_eager(monkeypatch):
    runner = NPUExactGraphRunner(max_graphs=1)
    captures = 0

    class _FunctionalGraph:
        def __init__(self, compute):
            self.compute = compute

        def replay(self, inputs):
            return self.compute(*inputs)

    def capture(inputs, compute):
        del inputs
        nonlocal captures
        captures += 1
        return _FunctionalGraph(compute)

    monkeypatch.setattr(runner, "_eligible", lambda inputs: True)
    monkeypatch.setattr(runner, "capture", capture)

    runner.run("unit", (torch.tensor([1.0]),), (), lambda value: (value + 1,))
    unseen = runner.run(
        "unit",
        (torch.tensor([1.0, 2.0]),),
        (),
        lambda value: (value + 1,),
    )

    assert captures == 1
    assert runner.stats == {"captures": 1, "failed": 0, "hits": 0}
    torch.testing.assert_close(unseen[0], torch.tensor([2.0, 3.0]))


@pytest.mark.parametrize(
    ("additional_config", "environment", "expected"),
    [
        ({"code2wav_enable_npu_graph": False}, "1", False),
        ({"code2wav_enable_npu_graph": True}, "0", True),
        ({}, "0", False),
        ({}, None, True),
    ],
)
def test_graph_settings_prioritize_stage_config(monkeypatch, additional_config, environment, expected):
    if environment is None:
        monkeypatch.delenv(code2wav_patch._ENABLE_ENV, raising=False)
    else:
        monkeypatch.setenv(code2wav_patch._ENABLE_ENV, environment)
    model = SimpleNamespace(vllm_config=SimpleNamespace(additional_config=additional_config))

    enabled, max_graphs = code2wav_patch._graph_settings(model)

    assert enabled is expected
    assert max_graphs == 32


def test_code2wav_runtime_disables_internal_format_and_jit(monkeypatch):
    monkeypatch.delenv("ASCEND_LAUNCH_BLOCKING", raising=False)
    compile_modes = []
    fake_npu = SimpleNamespace(
        config=SimpleNamespace(allow_internal_format=True),
        set_compile_mode=lambda **kwargs: compile_modes.append(kwargs),
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    code2wav_patch.prepare_code2wav_graph_runtime()

    assert fake_npu.config.allow_internal_format is False
    assert compile_modes == [{"jit_compile": False}]


def test_code2wav_preflight_failure_falls_back_to_eager(monkeypatch):
    class _Backend:
        speech_window = SimpleNamespace(device=SimpleNamespace(type="npu"))
        flow = SimpleNamespace(training=False)

    model = SimpleNamespace(
        backend=None,
        vllm_config=SimpleNamespace(additional_config={}),
    )

    monkeypatch.delenv(code2wav_patch._ENABLE_ENV, raising=False)
    monkeypatch.setattr(code2wav_patch.NPUExactGraphRunner, "is_supported", staticmethod(lambda: True))
    monkeypatch.setattr(
        code2wav_patch,
        "prepare_code2wav_graph_runtime",
        lambda: (_ for _ in ()).throw(RuntimeError("preflight unavailable")),
    )
    monkeypatch.setattr(
        code2wav_patch,
        "_original_build_backend",
        lambda instance: setattr(instance, "backend", _Backend()),
    )

    code2wav_patch._patched_build_backend(model)

    assert model.backend not in code2wav_patch._backend_graph_runners


@pytest.mark.parametrize("with_cache", [False, True])
def test_estimator_graph_dispatch_keeps_cached_and_uncached_buckets(monkeypatch, with_cache):
    calls = []

    class _Backend:
        pass

    class _Runner:
        def run(self, operation, inputs, constants, compute):
            calls.append((operation, inputs, constants))
            return compute(*inputs)

    backend = _Backend()
    code2wav_patch._backend_graph_runners[backend] = _Runner()

    def graphable(
        instance,
        estimator,
        *,
        x,
        mu,
        time_embedding,
        speakers,
        cond,
        cnn_cache,
        att_cache,
    ):
        del instance, estimator, time_embedding, speakers, cond, att_cache
        marker = x.new_tensor(0 if cnn_cache is None else 1)
        return x + mu, marker, marker.clone()

    monkeypatch.setattr(code2wav_patch, "_original_estimator_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(code2wav_patch, "_graphable_estimator_step", graphable)
    value = torch.tensor([2.0])
    cache = torch.tensor([1.0]) if with_cache else None
    estimator = SimpleNamespace(t_embedder=lambda time: time + 1)

    outputs = code2wav_patch._patched_estimator_step(
        backend,
        estimator,
        x=value,
        mu=value,
        time=value,
        speakers=value,
        cond=value,
        cnn_cache=cache,
        att_cache=cache,
    )

    assert calls[0][0] == "cfm_estimator"
    assert calls[0][2] == (with_cache, "float32")
    assert len(calls[0][1]) == (7 if with_cache else 5)
    torch.testing.assert_close(outputs[0], torch.tensor([4.0]))


def test_estimator_graph_dispatch_separates_fp16_and_fallback_epochs(monkeypatch):
    constants = []

    class _Backend:
        _npu_flow_float16_requested = True
        _npu_autocast_available = True

    class _Runner:
        def run(self, operation, inputs, graph_constants, compute):
            del operation
            constants.append(graph_constants)
            return compute(*inputs)

    backend = _Backend()
    code2wav_patch._backend_graph_runners[backend] = _Runner()

    def graphable(
        instance,
        estimator,
        *,
        x,
        mu,
        time_embedding,
        speakers,
        cond,
        cnn_cache,
        att_cache,
    ):
        del instance, estimator, time_embedding, speakers, cond, cnn_cache, att_cache
        marker = x.new_zeros(1)
        return x + mu, marker, marker.clone()

    monkeypatch.setattr(code2wav_patch, "_original_estimator_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(code2wav_patch, "_graphable_estimator_step", graphable)
    value = torch.tensor([2.0])
    estimator = SimpleNamespace(t_embedder=lambda time: time + 1)

    def dispatch():
        return code2wav_patch._patched_estimator_step(
            backend,
            estimator,
            x=value,
            mu=value,
            time=value,
            speakers=value,
            cond=value,
            cnn_cache=None,
            att_cache=None,
        )

    dispatch()
    backend._npu_autocast_available = False
    dispatch()

    assert constants == [(False, "float16"), (False, "float32")]
