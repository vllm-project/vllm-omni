# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The talker MTP graphs are part of the CUDA graph memory the KV budget reserves.

The runner captures them after the dispatcher's graphs. Upstream's profiling
pass sees them only where ``_dummy_run`` runs the MTP inside the FULL decoder
graphs, which v1 does for models with a talker. Both runners measure the rest
while profiling and report them again with the real capture, which the worker
compares with the estimate. Runners are built with ``object.__new__``; device
memory is faked. Pure CPU.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch
import vllm.config
import vllm.v1.worker.gpu.model_runner as upstream_v2
import vllm.v1.worker.gpu_model_runner as upstream_v1
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor

import vllm_omni.worker.gpu_ar_model_runner as ar_runner_module
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner as OmniV1Runner
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner as OmniV2Runner
from vllm_omni.worker_v2.staged_writes import StagedWrites

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

GIB = 1024**3


@pytest.fixture
def device_memory(monkeypatch):
    """Free device memory, set by each capture step."""
    state = SimpleNamespace(free=40 * GIB)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda *args, **kwargs: (state.free, 80 * GIB))
    return state


def _make_v1_runner(monkeypatch, device_memory, *, mtp_size, fail=False):
    runner = object.__new__(GPUARModelRunner)
    wrapper = object.__new__(CUDAGraphWrapper)
    wrapper.graph_pool = "global-pool"
    wrapper.concrete_cudagraph_entries = {}
    runner.talker_mtp = wrapper
    runner.has_talker_mtp = True
    runner.model = SimpleNamespace()
    runner.vllm_config = SimpleNamespace()
    runner.cudagraph_dispatcher = SimpleNamespace(
        cudagraph_keys={"FULL": {"profiling-key"}},
        keys_initialized=True,
    )
    events: list = []

    def init_minimal_kv_cache():
        events.append("init-kv")

    def capture_talker_mtp(skip_full=False):
        events.append(("capture", wrapper.graph_pool, skip_full))
        wrapper.concrete_cudagraph_entries["bs1"] = object()
        compilation_counter.num_cudagraph_captured += 1
        device_memory.free -= mtp_size
        if fail:
            raise RuntimeError("capture failed")

    def cleanup_profiling_kv_cache():
        events.append("cleanup-kv")

    runner._init_minimal_kv_cache_for_profiling = init_minimal_kv_cache
    runner._capture_talker_mtp_graphs = capture_talker_mtp
    runner._cleanup_profiling_kv_cache = cleanup_profiling_kv_cache
    monkeypatch.setattr(vllm.config, "set_current_vllm_config", lambda config: _NullContext())
    monkeypatch.setattr(
        ar_runner_module, "current_omni_platform", SimpleNamespace(graph_pool_handle=lambda: "profiling-pool")
    )
    return runner, wrapper, events


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_v1_profile_adds_the_talker_mtp_graphs_and_restores_the_runner(monkeypatch, device_memory):
    runner, wrapper, events = _make_v1_runner(monkeypatch, device_memory, mtp_size=GIB // 2)
    monkeypatch.setattr(upstream_v1.GPUModelRunner, "profile_cudagraph_memory", lambda self: 3 * GIB)
    captured_before = compilation_counter.num_cudagraph_captured

    assert runner.profile_cudagraph_memory() == 3 * GIB + GIB // 2

    # Captured into a throwaway pool while a profiling KV cache existed.
    assert events == ["init-kv", ("capture", "profiling-pool", False), "cleanup-kv"]
    assert wrapper.graph_pool == "global-pool"
    assert wrapper.concrete_cudagraph_entries == {}
    assert runner.cudagraph_dispatcher.cudagraph_keys == {"FULL": set()}
    assert runner.cudagraph_dispatcher.keys_initialized is False
    assert compilation_counter.num_cudagraph_captured == captured_before


def test_v1_profile_restores_the_runner_when_capture_fails(monkeypatch, device_memory):
    runner, wrapper, events = _make_v1_runner(monkeypatch, device_memory, mtp_size=GIB, fail=True)

    with pytest.raises(RuntimeError, match="capture failed"):
        runner._profile_talker_mtp_graph_memory()

    assert events[-1] == "cleanup-kv"
    assert wrapper.graph_pool == "global-pool"
    assert wrapper.concrete_cudagraph_entries == {}
    assert runner.cudagraph_dispatcher.keys_initialized is False


@pytest.mark.parametrize("has_talker_mtp, graph_wrapped", [(False, True), (True, False)])
def test_v1_profile_skips_runners_without_talker_mtp_graphs(monkeypatch, device_memory, has_talker_mtp, graph_wrapped):
    runner, _, events = _make_v1_runner(monkeypatch, device_memory, mtp_size=GIB)
    runner.has_talker_mtp = has_talker_mtp
    if not graph_wrapped:
        runner.talker_mtp = object()

    assert runner._profile_talker_mtp_graph_memory() == 0
    assert events == []


def test_v1_profile_leaves_the_full_sizes_to_upstream_for_talker_models(monkeypatch, device_memory):
    """With a talker, _dummy_run runs the MTP inside each FULL decoder capture,
    so upstream's estimate already holds those sizes."""
    runner, _, events = _make_v1_runner(monkeypatch, device_memory, mtp_size=GIB // 2)
    runner.model = SimpleNamespace(talker=object())

    assert runner._profile_talker_mtp_graph_memory() == GIB // 2
    assert events[1] == ("capture", "profiling-pool", True)


def test_v1_capture_reports_the_talker_mtp_graphs(monkeypatch, device_memory):
    runner, _, events = _make_v1_runner(monkeypatch, device_memory, mtp_size=GIB // 4)
    monkeypatch.setattr(upstream_v1.GPUModelRunner, "capture_model", lambda self: 2 * GIB)

    assert runner.capture_model() == 2 * GIB + GIB // 4
    assert events == [("capture", "global-pool", False)]


def test_v1_capture_without_talker_mtp_graphs_measures_nothing(monkeypatch, device_memory):
    runner, _, events = _make_v1_runner(monkeypatch, device_memory, mtp_size=GIB)
    runner.has_talker_mtp = False
    monkeypatch.setattr(upstream_v1.GPUModelRunner, "capture_model", lambda self: 2 * GIB)
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda *args, **kwargs: pytest.fail("nothing to measure"))

    assert runner.capture_model() == 2 * GIB
    assert events == []


class _RecordingWrapper(CUDAGraphWrapper):
    def __call__(self, ids, emb, hid, ts):
        self.calls.append(ids.shape[0])


def _make_capture_runner(monkeypatch, *, model=None, entries=()):
    """Sizes 1 and 2 dispatch to FULL graphs here, size 4 does not."""
    import vllm.compilation.monitor as monitor
    import vllm.distributed.parallel_state as parallel_state

    runner = object.__new__(GPUARModelRunner)
    wrapper = object.__new__(_RecordingWrapper)
    wrapper.calls = []
    wrapper.concrete_cudagraph_entries = {BatchDescriptor(num_tokens=n): object() for n in entries}
    runner.talker_mtp = wrapper
    runner.has_talker_mtp = True
    runner.model = model if model is not None else SimpleNamespace()
    runner.device = "cpu"
    runner.vllm_config = SimpleNamespace()
    runner.compilation_config = SimpleNamespace(cudagraph_capture_sizes=[1, 2, 4], cudagraph_num_of_warmups=1)
    for name in ("talker_mtp_input_ids", "talker_mtp_inputs_embeds", "last_talker_hidden", "text_step"):
        setattr(runner, name, SimpleNamespace(gpu=torch.zeros(8)))

    def dispatch(*, num_tokens, **kwargs):
        mode = CUDAGraphMode.FULL if num_tokens <= 2 else CUDAGraphMode.PIECEWISE
        return mode, BatchDescriptor(num_tokens=num_tokens), False, None, None

    runner._determine_batch_execution_and_padding = dispatch
    monkeypatch.setattr(parallel_state, "graph_capture", lambda device: _NullContext())
    monkeypatch.setattr(monitor, "set_cudagraph_capturing_enabled", lambda enabled: None)
    monkeypatch.setattr(ar_runner_module, "set_forward_context", lambda *args, **kwargs: _NullContext())
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *args, **kwargs: None)
    return runner, wrapper


@pytest.mark.parametrize("skip_full, expected", [(False, [4, 4, 2, 2, 1, 1]), (True, [4, 4])])
def test_capture_skips_only_the_sizes_that_dispatch_to_full(monkeypatch, skip_full, expected):
    runner, wrapper = _make_capture_runner(monkeypatch)

    runner._capture_talker_mtp_graphs(skip_full=skip_full)

    # One warmup and one capture per size, largest first.
    assert wrapper.calls == expected


def _capture_log(monkeypatch, caplog, **kwargs):
    runner, wrapper = _make_capture_runner(monkeypatch, **kwargs)
    with caplog.at_level("INFO", logger=ar_runner_module.logger.name):
        runner._capture_talker_mtp_graphs()
    return wrapper.calls, [(record.levelname, record.getMessage()) for record in caplog.records]


def test_capture_leaves_the_graphs_captured_with_the_decoder_alone(monkeypatch, caplog):
    """With a talker, _dummy_run captures the MTP inside the FULL decoder
    graphs; the real capture neither runs nor counts those sizes again."""
    calls, records = _capture_log(monkeypatch, caplog, model=SimpleNamespace(talker=object()), entries=(1, 2))

    assert calls == [4, 4]
    assert ("INFO", "Captured talker_mtp graphs for 1 sizes (2 already captured)") in records
    assert not [message for level, message in records if level == "WARNING"]


def test_capture_warns_about_a_full_size_the_decoder_capture_missed(monkeypatch, caplog):
    """Profiling skipped size 2 because it dispatches to FULL, so a new graph
    there is outside the budget."""
    calls, records = _capture_log(monkeypatch, caplog, model=SimpleNamespace(talker=object()), entries=(1,))

    assert calls == [4, 4, 2, 2]
    assert ("INFO", "Captured talker_mtp graphs for 2 sizes (1 already captured)") in records
    warnings = [message for level, message in records if level == "WARNING"]
    assert len(warnings) == 1 and "FULL sizes [2]" in warnings[0]


def test_capture_without_a_talker_expects_no_graphs_from_the_decoder(monkeypatch, caplog):
    """Without a talker, _dummy_run never runs the MTP and profiling measured
    every size, so the FULL sizes are captured here without a warning."""
    calls, records = _capture_log(monkeypatch, caplog)

    assert calls == [4, 4, 2, 2, 1, 1]
    assert ("INFO", "Captured talker_mtp graphs for 3 sizes (0 already captured)") in records
    assert not [message for level, message in records if level == "WARNING"]


@pytest.mark.parametrize(
    ("has_talker", "has_talker_mtp", "expected"),
    [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
)
def test_profiling_and_dummy_run_share_one_talker_mtp_predicate(has_talker, has_talker_mtp, expected):
    """skip_full is right only while it matches the condition under which
    _dummy_run runs talker_mtp; both read the base runner's helper."""
    assert "_talker_mtp_runs_in_dummy_run" not in GPUARModelRunner.__dict__
    assert "self._talker_mtp_runs_in_dummy_run()" in inspect.getsource(OmniV1Runner._dummy_run)
    runner = object.__new__(GPUARModelRunner)
    runner.model = SimpleNamespace(talker=object()) if has_talker else SimpleNamespace()
    runner.has_talker_mtp = has_talker_mtp
    assert runner._talker_mtp_runs_in_dummy_run() is expected


def _make_v2_runner(monkeypatch, device_memory, *, decoder_size, mtp_size):
    runner = object.__new__(OmniV2Runner)
    runner._staged_writes = StagedWrites()
    runner._exclude_full_graph = False
    runner._model_returns_tuple = False
    events: list = []

    def capture_decoder(self, *, profile_only=False):
        events.append(("decoder", profile_only))
        device_memory.free -= decoder_size
        return decoder_size

    def capture_mtp(dispatch, *, restore_capturing):
        events.append(("mtp", restore_capturing))
        device_memory.free -= mtp_size

    monkeypatch.setattr(upstream_v2.GPUModelRunner, "capture_model", capture_decoder)
    runner.model_state = SimpleNamespace(capture_mtp_graphs=capture_mtp, _is_mtp_graph_runner=lambda: True)
    runner.model = SimpleNamespace(capture_first_frame_graphs=lambda: events.append("first-frame"))
    return runner, events


def test_v2_profile_includes_the_mtp_graphs(monkeypatch, device_memory):
    """Upstream's profiling pass discards every CUDAGraphWrapper capture, so the
    MTP wrapper is measured there; model-owned graphs are not captured."""
    runner, events = _make_v2_runner(monkeypatch, device_memory, decoder_size=2 * GIB, mtp_size=GIB)

    assert runner.capture_model(profile_only=True) == 3 * GIB
    assert events == [("decoder", True), ("mtp", True)]


def test_v2_capture_reports_the_mtp_graphs(monkeypatch, device_memory):
    runner, events = _make_v2_runner(monkeypatch, device_memory, decoder_size=2 * GIB, mtp_size=GIB)

    assert runner.capture_model() == 3 * GIB
    assert events == [("decoder", False), ("mtp", False), "first-frame"]


@pytest.mark.parametrize("profile_only", [True, False])
def test_v2_an_eager_mtp_runner_is_neither_captured_nor_measured(monkeypatch, device_memory, profile_only):
    runner, events = _make_v2_runner(monkeypatch, device_memory, decoder_size=2 * GIB, mtp_size=GIB)
    runner.model_state._is_mtp_graph_runner = lambda: False
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda *args, **kwargs: pytest.fail("nothing to measure"))

    assert runner.capture_model(profile_only=profile_only) == 2 * GIB
    assert not [event for event in events if event[0] == "mtp"]


@pytest.mark.parametrize(
    "enabled_before, restore_capturing, enabled_after",
    [(True, True, True), (False, True, False), (True, False, False)],
)
def test_v2_mtp_capture_restores_the_capturing_flag_while_profiling(
    monkeypatch, enabled_before, restore_capturing, enabled_after
):
    """MRv2 leaves CUDA graph capturing enabled for its own capture. The MTP
    capture also runs inside the profiling pass, before that real capture, so
    there it restores the flag; after the real capture it disables it as before."""
    import vllm.compilation.monitor as monitor
    import vllm.distributed.parallel_state as parallel_state
    import vllm.forward_context as forward_context

    from vllm_omni.worker_v2.model_states.mtp_model_state import MTPState
    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

    state = object.__new__(OmniModelState)
    state._embed_dim = 0
    state.device = torch.device("cpu")
    state.model = SimpleNamespace()
    state.model.has_talker_mtp = False
    state._mtp_state = MTPState(state)
    state._mtp_state._raw_mtp = None
    calls = []
    state._mtp_state._is_mtp_graph_runner = lambda: True
    state.model = SimpleNamespace(mtp_accepts_req_infos=False)
    state.vllm_config = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_num_of_warmups=1))
    state.device = "cpu"
    for name in ("_mtp_input_ids", "_mtp_input_embeds", "_mtp_hidden", "_mtp_text_step"):
        setattr(state._mtp_state, name, torch.zeros(4))
    state._mtp_state._mtp_sample_uniforms = None
    state._mtp_state._get_mtp_capture_sizes = lambda: [2]
    state._mtp_state._get_mtp_base_sampling_kwargs = lambda: {}
    state._mtp_state._call_mtp_runner = lambda *args, **kwargs: calls.append(monitor.cudagraph_capturing_enabled)
    monkeypatch.setattr(parallel_state, "graph_capture", lambda device: _NullContext())
    monkeypatch.setattr(forward_context, "set_forward_context", lambda *args, **kwargs: _NullContext())
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(monitor, "cudagraph_capturing_enabled", enabled_before)

    state._mtp_state.capture_mtp_graphs(
        lambda num_reqs: SimpleNamespace(num_tokens=num_reqs), restore_capturing=restore_capturing
    )

    assert calls == [True, True]
    assert monitor.cudagraph_capturing_enabled is enabled_after
