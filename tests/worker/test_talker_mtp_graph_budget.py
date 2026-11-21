# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""V2 Talker MTP graphs contribute to the worker's CUDA graph budget.

Pure CPU fakes isolate memory accounting and forward-context capture order.
"""

from types import SimpleNamespace

import pytest
import torch
import vllm.v1.worker.gpu.model_runner as upstream_v2
from vllm.config import CUDAGraphMode

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


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _make_v2_runner(monkeypatch, device_memory, *, decoder_size, mtp_size):
    runner = object.__new__(OmniV2Runner)
    runner._staged_writes = StagedWrites()
    runner._exclude_full_graph = False
    runner._model_has_structured_output = False
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
    "ambient_mode, restore_capturing",
    [(CUDAGraphMode.NONE, True), (CUDAGraphMode.PIECEWISE, True), (CUDAGraphMode.PIECEWISE, False)],
)
def test_v2_mtp_capture_uses_forward_context_modes(monkeypatch, ambient_mode, restore_capturing):
    """Native V2 warms eagerly, captures FULL, and restores the caller's context."""
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
    state.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_num_of_warmups=1, fast_moe_cold_start=False, static_forward_context={}
        ),
        parallel_config=SimpleNamespace(data_parallel_size=1, use_sequence_parallel_moe=False, is_moe_model=False),
    )
    state.device = "cpu"
    for name in ("_mtp_input_ids", "_mtp_input_embeds", "_mtp_hidden", "_mtp_text_step"):
        setattr(state._mtp_state, name, torch.zeros(4))
    state._mtp_state._mtp_sample_uniforms = None
    state._mtp_state._get_mtp_capture_sizes = lambda: [2]
    state._mtp_state._get_mtp_base_sampling_kwargs = lambda: {}
    state._mtp_state._call_mtp_runner = lambda *args, **kwargs: calls.append(
        forward_context.get_forward_context().cudagraph_runtime_mode
    )
    monkeypatch.setattr(parallel_state, "graph_capture", lambda device: _NullContext())
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *args, **kwargs: None)
    ambient = forward_context.ForwardContext(
        no_compile_layers={}, attn_metadata=None, slot_mapping={}, cudagraph_runtime_mode=ambient_mode
    )
    with forward_context.override_forward_context(ambient):
        state._mtp_state.capture_mtp_graphs(
            lambda num_reqs: SimpleNamespace(num_tokens=num_reqs), restore_capturing=restore_capturing
        )
        assert forward_context.get_forward_context() is ambient

    assert calls == [CUDAGraphMode.NONE, CUDAGraphMode.FULL]
