# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager, nullcontext
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


def test_run_with_info_reports_capture_replay_and_bounded_fallback(monkeypatch):
    runner = NPUExactGraphRunner(max_graphs=1)

    class _FunctionalGraph:
        tensor_workspace_bytes = 64

        def __init__(self, compute):
            self.compute = compute

        def replay(self, inputs):
            return self.compute(*inputs)

    monkeypatch.setattr(runner, "_eligible", lambda inputs: True)
    monkeypatch.setattr(
        runner,
        "capture",
        lambda inputs, compute: _FunctionalGraph(compute),
    )

    first, capture = runner.run_with_info(
        "unit",
        (torch.tensor([1.0]),),
        ("bucket",),
        lambda value: (value + 1,),
    )
    second, replay = runner.run_with_info(
        "unit",
        (torch.tensor([2.0]),),
        ("bucket",),
        lambda value: (value + 1,),
    )
    fallback, miss = runner.run_with_info(
        "unit",
        (torch.tensor([3.0, 4.0]),),
        ("other",),
        lambda value: (value + 100,),
        fallback_compute=lambda value: (value - 1,),
    )

    torch.testing.assert_close(first[0], torch.tensor([2.0]))
    torch.testing.assert_close(second[0], torch.tensor([3.0]))
    torch.testing.assert_close(fallback[0], torch.tensor([2.0, 3.0]))
    assert (capture.mode, capture.reason, capture.workspace_bytes) == ("capture", "signature_miss", 64)
    assert (replay.mode, replay.reason, replay.workspace_bytes) == ("replay", None, 64)
    assert (miss.mode, miss.reason, miss.workspace_bytes) == ("fallback", "graph_capacity", 0)
    assert runner.telemetry == {
        "captures": 1,
        "failed": 0,
        "hits": 1,
        "misses": 2,
        "fallbacks": 1,
        "workspace_bytes": 64,
    }


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


@pytest.mark.parametrize(
    ("additional_config", "environment", "expected"),
    [
        ({"code2wav_enable_cfm_loop_npu_graph": False}, "1", False),
        ({"code2wav_enable_cfm_loop_npu_graph": True}, "0", True),
        ({}, "1", True),
        ({}, None, False),
    ],
)
def test_cfm_loop_graph_is_opt_in_and_config_wins(
    monkeypatch,
    additional_config,
    environment,
    expected,
):
    if environment is None:
        monkeypatch.delenv(code2wav_patch._LOOP_ENABLE_ENV, raising=False)
    else:
        monkeypatch.setenv(code2wav_patch._LOOP_ENABLE_ENV, environment)
    monkeypatch.delenv(code2wav_patch._LOOP_MAX_GRAPHS_ENV, raising=False)
    model = SimpleNamespace(vllm_config=SimpleNamespace(additional_config=additional_config))

    enabled, max_graphs = code2wav_patch._cfm_loop_graph_settings(model)

    assert enabled is expected
    assert max_graphs == 8


@pytest.mark.parametrize("value", ["invalid", "", "2", -1, 2])
def test_cfm_loop_graph_rejects_invalid_boolean(monkeypatch, value):
    monkeypatch.setenv(code2wav_patch._LOOP_ENABLE_ENV, str(value))
    model = SimpleNamespace(vllm_config=SimpleNamespace(additional_config={}))

    with pytest.raises(ValueError, match="explicit boolean"):
        code2wav_patch._cfm_loop_graph_settings(model)


@pytest.mark.parametrize("value", ["invalid", "-1", "9"])
def test_cfm_loop_graph_rejects_invalid_capacity(monkeypatch, value):
    monkeypatch.setenv(code2wav_patch._LOOP_ENABLE_ENV, "1")
    monkeypatch.setenv(code2wav_patch._LOOP_MAX_GRAPHS_ENV, value)
    model = SimpleNamespace(vllm_config=SimpleNamespace(additional_config={}))

    with pytest.raises(ValueError, match=r"integer in \[0, 8\]"):
        code2wav_patch._cfm_loop_graph_settings(model)


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


@pytest.mark.parametrize("mode", ["capture", "replay"])
def test_cfm_loop_graph_dispatch_uses_exact_bucket_and_single_clone_boundary(monkeypatch, mode):
    calls = []
    timeline = []

    class _Backend:
        _npu_flow_float16_requested = False
        _npu_autocast_available = None
        _cfm_loop_bucket = "first"
        n_timesteps = 6
        flow = SimpleNamespace(decoder=SimpleNamespace(estimator=object()))

        def _emit_timeline(self, event, **kwargs):
            timeline.append((event, kwargs))

    class _Runner:
        telemetry = {
            "captures": 1,
            "failed": 0,
            "hits": int(mode == "replay"),
            "misses": 1,
            "fallbacks": 0,
            "workspace_bytes": 128,
        }

        def run_with_info(self, operation, inputs, constants, compute, *, fallback_compute):
            del fallback_compute
            calls.append((operation, constants, len(inputs)))
            return compute(*inputs), SimpleNamespace(
                mode=mode,
                reason="signature_miss" if mode == "capture" else None,
                workspace_bytes=128,
            )

    backend = _Backend()
    code2wav_patch._backend_loop_graph_runners[backend] = _Runner()
    monkeypatch.setattr(code2wav_patch, "_original_decode_cfm", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        code2wav_patch,
        "_cfm_loop_constants",
        lambda *args, **kwargs: SimpleNamespace(num_bytes=16),
    )
    monkeypatch.setattr(
        code2wav_patch,
        "_graphable_cfm_loop",
        lambda instance, estimator, loop_constants, **kwargs: (
            kwargs["mu"] + 1,
            kwargs["mu"].new_zeros(1),
            kwargs["mu"].new_zeros(1),
        ),
    )
    value = torch.tensor([[[2.0]]])

    outputs = code2wav_patch._patched_decode_cfm(
        backend,
        value,
        torch.tensor([[1.0]]),
        value,
        cnn_cache=None,
        att_cache=None,
    )

    torch.testing.assert_close(outputs[0], torch.tensor([[[3.0]]]))
    assert calls == [("cfm_loop:first", (False, 6, "float32"), 3)]
    assert timeline[0][0] == f"cfm_loop_graph_{mode}"
    assert timeline[0][1]["details"]["bucket"] == "first"


def test_cfm_loop_capacity_falls_back_to_estimator_graph(monkeypatch):
    original_calls = []

    class _Backend:
        _npu_flow_float16_requested = False
        _npu_autocast_available = None
        n_timesteps = 10
        flow = SimpleNamespace(decoder=SimpleNamespace(estimator=object()))

        def _emit_timeline(self, *args, **kwargs):
            pass

    class _Runner:
        telemetry = {
            "captures": 8,
            "failed": 0,
            "hits": 0,
            "misses": 9,
            "fallbacks": 1,
            "workspace_bytes": 128,
        }

        def run_with_info(self, operation, inputs, constants, compute, *, fallback_compute):
            del operation, constants, compute
            return fallback_compute(*inputs), SimpleNamespace(
                mode="fallback",
                reason="graph_capacity",
                workspace_bytes=0,
            )

    def original(instance, mu, speakers, cond, *, cnn_cache, att_cache):
        del instance, speakers, cond, cnn_cache, att_cache
        original_calls.append(True)
        return mu - 1, mu.new_zeros(1), mu.new_zeros(1)

    backend = _Backend()
    code2wav_patch._backend_loop_graph_runners[backend] = _Runner()
    monkeypatch.setattr(code2wav_patch, "_original_decode_cfm", original)
    value = torch.tensor([[[2.0]]])

    outputs = code2wav_patch._patched_decode_cfm(
        backend,
        value,
        torch.tensor([[1.0]]),
        value,
        cnn_cache=None,
        att_cache=None,
    )

    assert original_calls == [True]
    torch.testing.assert_close(outputs[0], torch.tensor([[[1.0]]]))


@pytest.mark.parametrize(
    ("speech_width", "last_chunk", "expected"),
    [(0, False, "first"), (2, False, "steady"), (0, True, "tail")],
)
def test_decode_wrapper_labels_first_steady_and_tail_buckets(
    monkeypatch,
    speech_width,
    last_chunk,
    expected,
):
    seen = []

    class _Backend:
        pass

    def original(instance, *args, **kwargs):
        seen.append(instance._cfm_loop_bucket)
        return [], []

    backend = _Backend()
    state = SimpleNamespace(hift_cache={"speech": torch.zeros(1, speech_width)})
    monkeypatch.setattr(code2wav_patch, "_original_decode_batch", original)
    monkeypatch.setattr(code2wav_patch, "_flow_execution_context", lambda *args, **kwargs: nullcontext())

    code2wav_patch._patched_decode_batch(
        backend,
        torch.zeros(1, 1),
        object(),
        [state],
        last_chunk=last_chunk,
    )

    assert seen == [expected]
    assert not hasattr(backend, "_cfm_loop_bucket")
