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

    captured = runner.capture(
        (torch.tensor([2.0]),),
        lambda value: (value + 1,),
    )

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
    monkeypatch.setattr(
        runner,
        "capture",
        lambda inputs, compute: _FunctionalGraph(compute),
    )

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
    captures = 0
    computes = 0

    monkeypatch.setattr(runner, "_eligible", lambda inputs: True)

    def fail_capture(inputs, compute):
        nonlocal captures
        captures += 1
        raise RuntimeError("unsupported op")

    monkeypatch.setattr(runner, "capture", fail_capture)

    def compute(value):
        nonlocal computes
        computes += 1
        return (value + 1,)

    with pytest.raises(RuntimeError, match="restart the stage process"):
        runner.run("unit", (torch.tensor([1.0]),), (False,), compute)
    with pytest.raises(RuntimeError, match="cannot continue"):
        runner.run(
            "different-shape",
            (torch.tensor([1.0, 2.0]),),
            (True,),
            compute,
        )

    assert captures == 1
    assert computes == 1
    assert runner.stats == {"captures": 0, "failed": 1, "hits": 0}


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


def test_code2wav_runtime_rejects_launch_blocking(monkeypatch):
    monkeypatch.setenv("ASCEND_LAUNCH_BLOCKING", "1")

    with pytest.raises(RuntimeError, match="ASCEND_LAUNCH_BLOCKING=1"):
        code2wav_patch.prepare_code2wav_graph_runtime()


@pytest.mark.parametrize(
    ("enabled", "max_graphs", "expected_prepared"),
    [
        (True, 7, 1),
        (False, 7, 0),
        (True, 0, 0),
    ],
)
def test_code2wav_patch_reads_stage_additional_config(
    monkeypatch,
    enabled,
    max_graphs,
    expected_prepared,
):
    prepared = 0

    class _Backend:
        speech_window = torch.zeros(1)
        flow = SimpleNamespace(training=False)

    model = SimpleNamespace(
        backend=None,
        vllm_config=SimpleNamespace(
            additional_config={
                "code2wav_enable_npu_graph": enabled,
                "code2wav_max_npu_graphs": max_graphs,
            }
        ),
    )

    def build_backend(instance):
        instance.backend = _Backend()

    def prepare_runtime():
        nonlocal prepared
        prepared += 1

    monkeypatch.setattr(code2wav_patch, "_original_build_backend", build_backend)
    monkeypatch.setattr(code2wav_patch, "prepare_code2wav_graph_runtime", prepare_runtime)
    code2wav_patch._patched_build_backend(model)

    assert prepared == expected_prepared
    if expected_prepared:
        graph_runner = code2wav_patch._backend_graph_runners[model.backend]
        assert isinstance(graph_runner, NPUExactGraphRunner)
        assert graph_runner.max_graphs == max_graphs
    else:
        assert model.backend not in code2wav_patch._backend_graph_runners


@pytest.mark.parametrize("with_cache", [False, True])
def test_code2wav_estimator_graph_dispatch_is_platform_owned(monkeypatch, with_cache):
    calls = []

    class _Backend:
        _trt_stepper = None
        _cfm_graph_wrapper = None

    class _Runner:
        def run(self, operation, inputs, constants, compute):
            calls.append((operation, inputs, constants))
            return compute(*inputs)

    backend = _Backend()
    runner = _Runner()
    code2wav_patch._backend_graph_runners[backend] = runner

    def graphable_estimator_step(
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
        del instance, estimator, time_embedding, speakers, cond
        cache_marker = x.new_tensor(0 if cnn_cache is None else 1)
        return x + mu, cache_marker, cache_marker.clone()

    monkeypatch.setattr(code2wav_patch, "_original_estimator_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(code2wav_patch, "_graphable_estimator_step", graphable_estimator_step)
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
    assert calls[0][2] == (with_cache,)
    assert len(calls[0][1]) == (7 if with_cache else 5)
    torch.testing.assert_close(calls[0][1][2], torch.tensor([[3.0]]))
    torch.testing.assert_close(outputs[0], torch.tensor([4.0]))
    torch.testing.assert_close(outputs[1], torch.tensor(1.0 if with_cache else 0.0))


def test_code2wav_platform_wraps_flow_execution_context(monkeypatch):
    entered = []

    class _Backend:
        pass

    class _Context:
        def __init__(self, device, require_math):
            self.device = device
            self.require_math = require_math

        def __enter__(self):
            entered.append((self.device, self.require_math))

        def __exit__(self, exc_type, exc, traceback):
            return False

    backend = _Backend()
    code2wav_patch._backend_graph_runners[backend] = NPUExactGraphRunner()
    monkeypatch.setattr(
        code2wav_patch,
        "_flow_execution_context",
        lambda device, *, require_math: _Context(device, require_math),
    )
    monkeypatch.setattr(
        code2wav_patch,
        "_original_setup_batch",
        lambda instance, features, batch_size: ("setup", batch_size),
    )
    monkeypatch.setattr(
        code2wav_patch,
        "_original_decode_batch",
        lambda instance, tokens, features, states, *, last_chunk, flush_encoder: (
            "decode",
            last_chunk,
            flush_encoder,
        ),
    )
    features = SimpleNamespace(speech_tokens=torch.zeros(1))

    assert code2wav_patch._patched_setup_batch(backend, features, 3) == ("setup", 3)
    assert code2wav_patch._patched_decode_batch(
        backend,
        torch.zeros(1),
        features,
        [],
        last_chunk=True,
    ) == ("decode", True, False)
    assert entered == [(torch.device("cpu"), True), (torch.device("cpu"), True)]
