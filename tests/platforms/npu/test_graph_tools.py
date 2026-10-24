# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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
        (None, 7, 1),  # No NPU override: honor the shared CFM graph config.
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
        _extra_config=lambda: {},
        _cfm_graph_config={"enabled": True, "max_graphs": 32, "enable_whole_euler": True},
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
    assert model._cfm_graph_config == {
        "enabled": bool(expected_prepared),
        "max_graphs": max_graphs,
        "enable_whole_euler": True,
    }
    if expected_prepared:
        graph_runner = code2wav_patch._backend_graph_runners[model.backend]
        assert isinstance(graph_runner, NPUExactGraphRunner)
        assert graph_runner.max_graphs == max_graphs
    else:
        assert model.backend not in code2wav_patch._backend_graph_runners


@pytest.mark.parametrize("value", [True, "true", "false", "0", "off"])
def test_code2wav_patch_rejects_bfloat16_attention_cache(monkeypatch, value):
    built = False

    model = SimpleNamespace(
        backend=None,
        _extra_config=lambda: {"code2wav_bfloat16_attention_cache": value},
    )

    def build_backend(_instance):
        nonlocal built
        built = True

    monkeypatch.setattr(code2wav_patch, "_original_build_backend", build_backend)

    with pytest.raises(ValueError, match="currently supported only on CUDA"):
        code2wav_patch._patched_build_backend(model)

    assert not built


@pytest.mark.parametrize("value", [False, None, 0, ""])
def test_code2wav_patch_allows_disabled_bfloat16_attention_cache(monkeypatch, value):
    built = False

    model = SimpleNamespace(
        backend=None,
        _extra_config=lambda: {"code2wav_bfloat16_attention_cache": value},
    )

    def build_backend(instance):
        nonlocal built
        built = True
        instance.backend = object()

    monkeypatch.setattr(code2wav_patch, "_original_build_backend", build_backend)
    code2wav_patch._patched_build_backend(model)

    assert built


@pytest.mark.parametrize("with_cache", [False, True])
def test_code2wav_estimator_graph_dispatch_is_platform_owned(monkeypatch, with_cache):
    calls = []

    class _Backend:
        _trt_stepper = None
        _cfm_graph_wrapper = None

        def _time_embedding(self, estimator, time):
            return estimator.t_embedder(time)

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
    assert calls[0][2] == (with_cache, None)
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


@pytest.mark.parametrize("with_cache", [False, True])
def test_code2wav_pad_mask_replays_as_a_graph_input(monkeypatch, with_cache):
    """A bucketing pad mask is shape-stable, so it replays inside the graph."""
    calls = []
    seen = []

    class _Backend:
        _trt_stepper = None
        _cfm_graph_wrapper = None

        def _time_embedding(self, estimator, time):
            return estimator.t_embedder(time)

    class _Runner:
        def run(self, operation, inputs, constants, compute):
            calls.append((operation, inputs, constants))
            return compute(*inputs)

    backend = _Backend()
    code2wav_patch._backend_graph_runners[backend] = _Runner()

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
        attn_mask=None,
    ):
        del instance, estimator, time_embedding, speakers, cond, cnn_cache, att_cache
        seen.append(attn_mask)
        return x + mu, x.clone(), x.clone()

    monkeypatch.setattr(code2wav_patch, "_original_estimator_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(code2wav_patch, "_graphable_estimator_step", graphable_estimator_step)
    value = torch.tensor([2.0])
    mask = torch.tensor([1.0, 0.0])
    estimator = SimpleNamespace(t_embedder=lambda time: time + 1)

    code2wav_patch._patched_estimator_step(
        backend,
        estimator,
        x=value,
        mu=value,
        time=value,
        speakers=value,
        cond=value,
        attn_mask=mask,
        cnn_cache=value if with_cache else None,
        att_cache=value if with_cache else None,
    )

    assert calls[0][0] == "cfm_estimator"
    assert calls[0][2] == (with_cache, None)
    assert any(item is mask for item in calls[0][1])
    assert seen[0] is mask


@pytest.mark.parametrize("with_cache", [False, True])
def test_estimator_graph_distinguishes_padding_and_reuses_embedding(monkeypatch, with_cache):
    class Backend:
        _trt_stepper = None
        _cfm_graph_wrapper = None

        def _time_embedding(self, estimator, time):
            return time + 7

    backend = Backend()
    runner = NPUExactGraphRunner(max_graphs=4)
    code2wav_patch._backend_graph_runners[backend] = runner
    monkeypatch.setattr(runner, "_eligible", lambda inputs: True)

    class Captured:
        def __init__(self, compute):
            self.compute = compute

        def replay(self, inputs):
            return self.compute(*inputs)

    monkeypatch.setattr(runner, "capture", lambda inputs, compute: Captured(compute))
    monkeypatch.setattr(code2wav_patch, "_original_estimator_step", lambda *args, **kwargs: None)

    def body(backend, estimator, *, x, mu, time_embedding, speakers, cond, cnn_cache, att_cache, valid_frames=None):
        output = speakers.unsqueeze(-1).expand_as(x).clone()
        if valid_frames is not None:
            output[..., valid_frames:] = 0
        return output, time_embedding, x

    monkeypatch.setattr(code2wav_patch, "_graphable_estimator_step", body)
    x = torch.ones(1, 1, 6)
    cache = x if with_cache else None
    kwargs = dict(x=x, mu=x, time=torch.ones(1), speakers=torch.ones(1, 1), cond=x, cnn_cache=cache, att_cache=cache)
    for frames in (4, 2, 4):
        output, embedding, _ = code2wav_patch._patched_estimator_step(backend, object(), **kwargs, valid_frames=frames)
        expected = x.clone()
        expected[..., frames:] = 0
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(embedding, torch.tensor([[8.0]]))
    supplied = torch.tensor([[19.0]])
    _, embedding, _ = code2wav_patch._patched_estimator_step(
        backend, object(), **kwargs, valid_frames=4, time_embedding=supplied
    )
    torch.testing.assert_close(embedding, supplied)
    assert runner.stats == {"captures": 2, "failed": 0, "hits": 2}


def _install_fake_npu(monkeypatch, *, with_pool_handle: bool = True):
    """Install a fake torch.npu whose graph() records the capture pool."""
    seen_pools = []
    handles = iter(range(1000, 2000))
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

    npu_members = {
        "NPUGraph": _Graph,
        "graph": graph_context,
        "synchronize": synchronize,
        "is_current_stream_capturing": lambda: False,
    }
    if with_pool_handle:
        npu_members["graph_pool_handle"] = lambda: next(handles)
    fake_npu = SimpleNamespace(**npu_members)
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    import vllm.platforms

    global_pool = object()
    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        SimpleNamespace(get_global_graph_pool=lambda: global_pool),
    )
    return seen_pools, global_pool


def test_dedicated_pool_runners_use_distinct_handles(monkeypatch):
    seen_pools, global_pool = _install_fake_npu(monkeypatch)

    runner_a = NPUExactGraphRunner(use_shared_pool=False, component_name="A")
    runner_b = NPUExactGraphRunner(use_shared_pool=False, component_name="B")

    for runner in (runner_a, runner_b):
        runner.capture((torch.tensor([1.0]),), lambda value: (value + 1,))
        # Second capture on the same runner must reuse its own handle.
        runner.capture((torch.tensor([2.0, 3.0]),), lambda value: (value + 1,))

    # Two captures per runner: one dedicated handle per runner, reused.
    assert len(seen_pools) == 4
    handles_a = [id(p) for p in seen_pools[:2]]
    handles_b = [id(p) for p in seen_pools[2:]]
    assert handles_a[0] == handles_a[1], "runner A must reuse its dedicated handle"
    assert handles_b[0] == handles_b[1], "runner B must reuse its dedicated handle"
    assert handles_a[0] != handles_b[0], "dedicated handles must be distinct per runner"
    assert all(p is not global_pool for p in seen_pools), "dedicated runners must not touch the global pool"


def test_dedicated_pool_missing_handle_warns_and_falls_back(monkeypatch):
    seen_pools, global_pool = _install_fake_npu(monkeypatch, with_pool_handle=False)

    import vllm_omni.platforms.npu.graph_tools as graph_tools_module

    warnings = []
    monkeypatch.setattr(
        graph_tools_module.logger,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )

    runner = NPUExactGraphRunner(use_shared_pool=False, component_name="fallback-check")
    runner.capture((torch.tensor([1.0]),), lambda value: (value + 1,))

    assert seen_pools == [global_pool]
    assert warnings, "the shared-pool fallback must be announced, not silent"
    assert "graph_pool_handle" in str(warnings[0][0])


def test_vae_npu_graph_runner_is_per_instance(monkeypatch):
    """Two AudioVAE instances must never share (or cross-replay) a runner.

    Regression test for the process-global runner singleton: identical input
    signatures across instances must still bind each decode callable to its
    own instance's runner, so a second model can never replay the first
    model's captured weights.
    """
    seen_pools, _ = _install_fake_npu(monkeypatch)

    import vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker as talker_module
    from vllm_omni.platforms import current_omni_platform
    from vllm_omni.platforms.npu import graph_tools as graph_tools_module

    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)
    monkeypatch.setenv("VLLM_OMNI_DISABLE_VAE_NPU_GRAPH", "0")

    class _FakeRunner:
        def __init__(self, *, max_graphs, component_name, disable_config_hint, use_shared_pool):
            self.use_shared_pool = use_shared_pool
            self.bound_compute = None

        def is_supported(self):
            return True

        def run(self, operation, inputs, constants, compute):
            self.bound_compute = compute
            return compute(*inputs)

    constructed = []
    monkeypatch.setattr(
        graph_tools_module,
        "NPUExactGraphRunner",
        lambda **kwargs: (constructed.append(kwargs) or _FakeRunner(**kwargs)),
    )

    class _FakeVae:
        def __init__(self, tag):
            self.tag = tag

        def decode(self, z, sr_cond=None):
            return torch.full_like(z, self.tag)

    vae_a = _FakeVae(1.0)
    vae_b = _FakeVae(2.0)

    runner_a = talker_module._vae_npu_graph_runner_for(vae_a)
    runner_b = talker_module._vae_npu_graph_runner_for(vae_b)

    assert runner_a is not runner_b, "each AudioVAE instance must own its runner"
    assert runner_a is not False and runner_b is not False
    assert len(constructed) == 2, "one runner per instance, not one shared singleton"
    assert all(kwargs["use_shared_pool"] is False for kwargs in constructed)

    out_a = runner_a.run("decode", (torch.zeros(1),), (), lambda z, sr=None: (vae_a.decode(z),))[0]
    out_b = runner_b.run("decode", (torch.zeros(1),), (), lambda z, sr=None: (vae_b.decode(z),))[0]
    torch.testing.assert_close(out_a, torch.tensor([1.0]))
    torch.testing.assert_close(out_b, torch.tensor([2.0]))

    # The same instance returns the same runner on repeated calls (memoized).
    assert talker_module._vae_npu_graph_runner_for(vae_a) is runner_a
