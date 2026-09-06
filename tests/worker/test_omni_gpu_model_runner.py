# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.cudagraph_dispatcher import CUDAGraphMode
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_model_runner import (
    OmniGPUModelRunner,
    _filter_mrope_kwargs_for_model,
)
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _runner_for_talker_graph_init(
    *,
    talker_mtp_graph_safe: bool | None,
    has_separate_talker: bool = True,
    model_stage: str = "talker",
) -> OmniGPUModelRunner:
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(
        talker=object() if has_separate_talker else None,
        talker_mtp=object(),
        model_stage=model_stage,
    )
    if talker_mtp_graph_safe is not None:
        runner.model.talker_mtp_graph_safe = talker_mtp_graph_safe
    runner.model_config = SimpleNamespace(hf_text_config=SimpleNamespace(hidden_size=4))
    runner.compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL,
        max_cudagraph_capture_size=1,
    )
    runner.vllm_config = object()
    runner.max_num_reqs = 1
    runner.dtype = torch.float32
    runner._make_buffer = lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs)
    return runner


def test_talker_mtp_skips_graph_when_model_declares_unsafe(monkeypatch):
    runner = _runner_for_talker_graph_init(talker_mtp_graph_safe=False)
    talker_mtp = runner.model.talker_mtp
    monkeypatch.setattr(
        "vllm_omni.worker.gpu_model_runner.current_omni_platform.get_graph_wrapper_cls",
        lambda: pytest.fail("graph wrapper must not be selected"),
    )

    OmniGPUModelRunner._init_talker_mtp(runner)

    assert runner.talker_mtp is talker_mtp


@pytest.mark.parametrize("model_stage", ["thinker", "code2wav"])
def test_non_talker_stage_does_not_use_talker_mtp_graph(monkeypatch, model_stage: str):
    runner = _runner_for_talker_graph_init(
        talker_mtp_graph_safe=None,
        has_separate_talker=False,
        model_stage=model_stage,
    )
    talker_mtp = runner.model.talker_mtp
    monkeypatch.setattr(
        "vllm_omni.worker.gpu_model_runner.current_omni_platform.get_graph_wrapper_cls",
        lambda: pytest.fail("graph wrapper must not be selected"),
    )

    OmniGPUModelRunner._init_talker_mtp(runner)

    assert runner.talker_mtp is talker_mtp


@pytest.mark.parametrize("talker_mtp_graph_safe", [None, True])
def test_talker_mtp_uses_graph_for_legacy_or_explicit_safe_model(monkeypatch, talker_mtp_graph_safe):
    runner = _runner_for_talker_graph_init(talker_mtp_graph_safe=talker_mtp_graph_safe)
    wrapped = object()
    monkeypatch.setattr(
        "vllm_omni.worker.gpu_model_runner.current_omni_platform.get_graph_wrapper_cls",
        lambda: lambda *args, **kwargs: wrapped,
    )

    OmniGPUModelRunner._init_talker_mtp(runner)

    assert runner.talker_mtp is wrapped


def test_shutdown_clears_model_state_before_named_kv_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class Model:
        def clear_runtime_state(self) -> None:
            events.append("model-clear")

    class Branch:
        name = "negative"

        def close(self) -> None:
            events.append("branch-close")

    runner = object.__new__(OmniGPUModelRunner)
    runner.model = Model()
    runner.named_kv_branches = {"negative": Branch()}
    monkeypatch.setattr(
        GPUModelRunner,
        "shutdown",
        lambda _runner: events.append("upstream-shutdown"),
    )

    OmniGPUModelRunner.shutdown(runner)

    assert events == [
        "model-clear",
        "branch-close",
        "upstream-shutdown",
    ]
    assert runner.named_kv_branches == {}


def test_shutdown_continues_after_model_and_branch_cleanup_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class Model:
        def clear_runtime_state(self) -> None:
            events.append("model-clear")
            raise RuntimeError("injected model cleanup failure")

    class Branch:
        name = "negative"

        def close(self) -> None:
            events.append("branch-close")
            raise RuntimeError("injected branch cleanup failure")

    runner = object.__new__(OmniGPUModelRunner)
    runner.model = Model()
    runner.named_kv_branches = {"negative": Branch()}
    monkeypatch.setattr(
        GPUModelRunner,
        "shutdown",
        lambda _runner: events.append("upstream-shutdown"),
    )

    OmniGPUModelRunner.shutdown(runner)

    assert events == ["model-clear", "branch-close", "upstream-shutdown"]
    assert runner.named_kv_branches == {}


class DummyBuffer:
    """A minimal buffer wrapper that exposes the `.gpu` attribute."""

    def __init__(self, t: torch.Tensor):
        self.gpu = t


class DummyInputBatch:
    """A minimal input batch that only provides `req_ids`."""

    def __init__(self, req_ids):
        self.req_ids = req_ids
        self.req_id_to_index = {r: i for i, r in enumerate(req_ids)}


class DummyReqState:
    """A minimal request state container."""

    mm_features: list[object]
    additional_information_cpu: dict[str, object]


def test_model_forward_passes_request_ids_to_decode_metadata(monkeypatch):
    received = {}
    model = SimpleNamespace(
        supports_omni_decode_step_metadata=True,
        update_decode_step_metadata=lambda **kwargs: received.update(kwargs),
    )
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = model
    runner.input_batch = DummyInputBatch(["request-a", "request-b"])
    runner._build_model_kwargs_extra = lambda: {}
    monkeypatch.setattr(GPUModelRunner, "_model_forward", lambda *_args, **_kwargs: torch.zeros(1))

    OmniGPUModelRunner._model_forward(runner, input_ids=torch.ones(2, dtype=torch.long))

    assert received["req_ids"] == ["request-a", "request-b"]


class MiMoAudioForConditionalGeneration(torch.nn.Module):
    """Dummy model whose class name must exactly match the production check."""

    def __init__(self):
        super().__init__()

    # No real forward needed for these tests.


class DummyTalkerMTP(torch.nn.Module):
    """A fake talker_mtp module for deterministic CPU testing."""

    def forward(
        self,
        req_input_ids,
        req_embeds,
        last_talker_hidden,
        text_step,
        do_sample=None,
        temperature=None,
        top_k=None,
        top_p=None,
    ):
        # Deterministic behavior:
        # - output embeds = input embeds + 1
        # - output codes = [[0], [1], ...]
        bsz = req_embeds.shape[0]
        new_embeds = req_embeds + 1.0
        codes = torch.arange(bsz, dtype=torch.int64).view(bsz, 1)
        return new_embeds, codes


class CaptureTalkerMTP(torch.nn.Module):
    """A fake talker_mtp module that records sampling kwargs."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self,
        req_input_ids,
        req_embeds,
        last_talker_hidden,
        text_step,
        do_sample=None,
        temperature=None,
        top_k=None,
        top_p=None,
        generator=None,
        generators=None,
    ):
        self.calls.append(
            {
                "batch_size": int(req_embeds.shape[0]),
                "do_sample": do_sample,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "generator": generator,
                "generators": generators,
            }
        )
        codes = torch.zeros((req_embeds.shape[0], 1), dtype=torch.int64)
        return req_embeds, codes


class StrictMRoPEModel:
    def get_mrope_input_positions(self, input_tokens, mm_features):
        raise NotImplementedError


class FlexibleMRoPEModel:
    def get_mrope_input_positions(self, input_tokens, mm_features=None, **kwargs):
        raise NotImplementedError


@contextmanager
def _noop_forward_context(*args, **kwargs):
    """A no-op context manager to replace vLLM forward context in CPU tests."""
    yield


def test_scheduled_input_token_ids_are_read_from_existing_cpu_batch() -> None:
    runner = object.__new__(OmniGPUModelRunner)
    runner.input_batch = SimpleNamespace(
        token_ids_cpu=np.array(
            [
                [10, 11, 12, 13, 14],
                [20, 21, 22, 23, 24],
            ],
            dtype=np.int64,
        )
    )

    assert runner._scheduled_input_token_ids_cpu(
        req_id="request-1",
        req_index=1,
        start_token_index=1,
        num_scheduled_tokens=3,
    ) == (21, 22, 23)
    assert (
        runner._scheduled_input_token_ids_cpu(
            req_id="request-0",
            req_index=0,
            start_token_index=5,
            num_scheduled_tokens=0,
        )
        == ()
    )


def test_scheduled_input_token_ids_reject_incomplete_cpu_span() -> None:
    runner = object.__new__(OmniGPUModelRunner)
    runner.input_batch = SimpleNamespace(token_ids_cpu=np.array([[1, 2]], dtype=np.int64))

    with pytest.raises(RuntimeError, match="expected 2 CPU input token IDs, got 1"):
        runner._scheduled_input_token_ids_cpu(
            req_id="request",
            req_index=0,
            start_token_index=1,
            num_scheduled_tokens=2,
        )


class _ReadyEvent:
    def __init__(self) -> None:
        self.synchronize_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1


def _remember_async_feedback(
    runner: OmniGPUModelRunner,
    *,
    sampled_rows: list[list[int]],
    req_id_to_index: dict[str, int],
) -> _ReadyEvent:
    event = _ReadyEvent()
    runner.use_async_scheduling = True
    runner._omni_requires_input_token_ids_cpu = True
    runner._remember_async_sampled_token_feedback(
        sampled_token_ids_cpu=torch.tensor(sampled_rows, dtype=torch.int64),
        ready_event=event,
        req_id_to_index=req_id_to_index,
    )
    return event


def test_async_sampled_token_feedback_resolves_reordered_batch_once() -> None:
    runner = object.__new__(OmniGPUModelRunner)
    runner.input_batch = SimpleNamespace(
        token_ids_cpu=np.array(
            [
                [-1],
                [123],
                [-1],
                [-1],
                [-1],
            ],
            dtype=np.int64,
        )
    )
    event = _remember_async_feedback(
        runner,
        sampled_rows=[[151654], [151653], [151652], [151643]],
        req_id_to_index={
            "request-a": 0,
            "request-b": 1,
            "request-c": 2,
            "request-d": 3,
        },
    )

    # A prompt token in the mixed batch must not wait on sampled-token D2H.
    assert runner._scheduled_input_token_ids_cpu(
        req_id="request-new",
        req_index=1,
        start_token_index=0,
        num_scheduled_tokens=1,
    ) == (123,)
    assert event.synchronize_calls == 0

    current_order = ["request-c", "request-a", "request-d", "request-b"]
    expected = [151652, 151654, 151643, 151653]
    for req_index, (req_id, token_id) in enumerate(zip(current_order, expected, strict=True)):
        batch_index = req_index if req_index == 0 else req_index + 1
        assert runner._scheduled_input_token_ids_cpu(
            req_id=req_id,
            req_index=batch_index,
            start_token_index=0,
            num_scheduled_tokens=1,
        ) == (token_id,)

    assert event.synchronize_calls == 1


def test_async_sampled_token_feedback_is_capability_gated() -> None:
    runner = object.__new__(OmniGPUModelRunner)
    runner.use_async_scheduling = True
    runner._omni_requires_input_token_ids_cpu = False
    event = _ReadyEvent()

    runner._remember_async_sampled_token_feedback(
        sampled_token_ids_cpu=torch.tensor([[151654]], dtype=torch.int64),
        ready_event=event,
        req_id_to_index={"request": 0},
    )

    assert runner._omni_async_sampled_token_feedback is None
    assert event.synchronize_calls == 0


@pytest.mark.parametrize(
    ("token_ids_cpu", "sampled_rows", "req_id_to_index", "error"),
    [
        ([-1], None, None, "missing async sampled-token feedback"),
        ([-1], [[151654]], {"other": 0}, "cannot map async sampled-token feedback"),
        ([-1], [[-1]], {"request": 0}, "exactly one valid async sampled token"),
        ([-1], [[151654, 151653]], {"request": 0}, "exactly one valid async sampled token"),
        ([-1, -1], [[151654]], {"request": 0}, "multiple async sampled-token sentinels"),
    ],
)
def test_async_sampled_token_feedback_rejects_unresolvable_sentinel(
    token_ids_cpu: list[int],
    sampled_rows: list[list[int]] | None,
    req_id_to_index: dict[str, int] | None,
    error: str,
) -> None:
    runner = object.__new__(OmniGPUModelRunner)
    runner.input_batch = SimpleNamespace(token_ids_cpu=np.array([token_ids_cpu], dtype=np.int64))
    if sampled_rows is not None and req_id_to_index is not None:
        _remember_async_feedback(
            runner,
            sampled_rows=sampled_rows,
            req_id_to_index=req_id_to_index,
        )

    with pytest.raises(RuntimeError, match=error):
        runner._scheduled_input_token_ids_cpu(
            req_id="request",
            req_index=0,
            start_token_index=0,
            num_scheduled_tokens=len(token_ids_cpu),
        )


def test_async_sampled_token_feedback_zero_span_does_not_wait() -> None:
    runner = object.__new__(OmniGPUModelRunner)
    runner.input_batch = SimpleNamespace(token_ids_cpu=np.array([[1]], dtype=np.int64))
    event = _remember_async_feedback(
        runner,
        sampled_rows=[[151654]],
        req_id_to_index={"request": 0},
    )

    assert (
        runner._scheduled_input_token_ids_cpu(
            req_id="request",
            req_index=0,
            start_token_index=1,
            num_scheduled_tokens=0,
        )
        == ()
    )
    assert event.synchronize_calls == 0


def test_filter_mrope_kwargs_for_strict_model_signature():
    kwargs = {
        "mm_features": ["audio"],
        "hf_config": object(),
        "image_grid_thw": [],
    }

    assert _filter_mrope_kwargs_for_model(StrictMRoPEModel(), kwargs) == {
        "mm_features": ["audio"],
    }


def test_filter_mrope_kwargs_preserves_flexible_model_kwargs():
    kwargs = {
        "mm_features": ["video"],
        "hf_config": object(),
        "video_grid_thw": [[1, 2, 3]],
    }

    assert _filter_mrope_kwargs_for_model(FlexibleMRoPEModel(), kwargs) is kwargs


def _make_runner(req_ids=("r1", "r2"), hidden_size=4):
    # Create an instance without calling OmniGPUModelRunner.__init__
    runner = object.__new__(OmniGPUModelRunner)

    # Minimal attributes used by OmniGPUModelRunner._talker_mtp_forward
    runner.input_batch = DummyInputBatch(list(req_ids))
    runner.requests = {rid: DummyReqState() for rid in req_ids}
    runner.model_intermediate_buffer = {}

    # query_start_loc.cpu[req_index] is used to locate the token position
    # in the flattened `inputs_embeds`.
    runner.query_start_loc = type("QSL", (), {})()
    # Map: r1 -> offset 0, r2 -> offset 3
    runner.query_start_loc.cpu = torch.tensor([0, 3], dtype=torch.int32)

    bsz = len(req_ids)
    runner.talker_mtp_input_ids = DummyBuffer(torch.zeros((bsz,), dtype=torch.int64))
    runner.talker_mtp_inputs_embeds = DummyBuffer(torch.zeros((bsz, hidden_size), dtype=torch.float32))
    runner.last_talker_hidden = DummyBuffer(torch.zeros((bsz, hidden_size), dtype=torch.float32))
    runner.text_step = DummyBuffer(torch.zeros((bsz, hidden_size), dtype=torch.float32))

    runner.talker_mtp = DummyTalkerMTP()
    runner.model = SimpleNamespace(talker_mtp_output_key=("codes", "audio"))
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace())

    # Provide a minimal implementation that returns the expected 4-tuple.
    def _determine_batch_execution_and_padding(**kwargs):
        return None, object(), None, None, None

    runner._determine_batch_execution_and_padding = _determine_batch_execution_and_padding

    # Use the real merge method from OmniGPUModelRunner.
    return runner


def _make_runner_for_mimo(req_id="r_mimo"):
    """Create a minimal runner with MiMoAudio-like model and request state."""
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = MiMoAudioForConditionalGeneration()

    # Minimal vllm_config / model_config used by helper.
    class _DummyModelConfig:
        async_chunk = False

    class _DummyVllmConfig:
        model_config = _DummyModelConfig()

    runner.vllm_config = _DummyVllmConfig()

    # Attach a single request state with mm_features and additional_information_cpu.
    req_state = DummyReqState()
    req_state.mm_features = ["mm_feature_obj"]
    req_state.additional_information_cpu = {"some_key": "some_value"}

    runner.requests = {req_id: req_state}

    return runner


def test_talker_mtp_forward_cpu_updates_inputs_and_info(monkeypatch):
    # `_talker_mtp_forward` dispatches through the active platform.
    import vllm_omni.worker.gpu_model_runner as mod  # Must be the same module that defines OmniGPUModelRunner

    monkeypatch.setattr(mod.current_omni_platform, "set_forward_context", _noop_forward_context)

    runner = _make_runner(req_ids=("r1", "r2"), hidden_size=4)

    def fake_determine(self, num_tokens, num_reqs, num_scheduled_tokens_np, max_num_scheduled_tokens, use_cascade_attn):
        batch_desc = SimpleNamespace(num_tokens=int(num_tokens))
        return (False, batch_desc, None, None, None)

    monkeypatch.setattr(runner, "_determine_batch_execution_and_padding", fake_determine.__get__(runner, type(runner)))

    # Initialize per-request embeds (batch-major inside talker_mtp_inputs_embeds)
    runner.talker_mtp_inputs_embeds.gpu[0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    runner.talker_mtp_inputs_embeds.gpu[1] = torch.tensor([10.0, 20.0, 30.0, 40.0])

    # Flattened `inputs_embeds`: offsets 0 and 3 will be overwritten
    inputs_embeds = torch.zeros((6, 4), dtype=torch.float32)

    # Call the original implementation from OmniGPUModelRunner (no re-implementation)
    OmniGPUModelRunner._talker_mtp_forward(runner, ["r1", "r2"], inputs_embeds)

    # Validate embeds were written back (+1)
    assert torch.allclose(inputs_embeds[0], torch.tensor([2.0, 3.0, 4.0, 5.0]))
    assert torch.allclose(inputs_embeds[3], torch.tensor([11.0, 21.0, 31.0, 41.0]))

    # Validate per-request additional_information_cpu was updated
    info_r1 = runner.requests["r1"].additional_information_cpu
    info_r2 = runner.requests["r2"].additional_information_cpu
    assert int(info_r1["codes"]["audio"][0, 0]) == 0
    assert int(info_r2["codes"]["audio"][0, 0]) == 1


def test_talker_mtp_forward_cpu_empty_batch_noop(monkeypatch):
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod.current_omni_platform, "set_forward_context", _noop_forward_context)

    runner = _make_runner(req_ids=("r1",), hidden_size=4)

    inputs_embeds = torch.randn((2, 4))
    before = inputs_embeds.clone()

    OmniGPUModelRunner._talker_mtp_forward(runner, [], inputs_embeds)

    # Ensure no changes were made
    assert torch.allclose(inputs_embeds, before)


def test_talker_mtp_forward_ignores_default_sampling_seed_without_request_marker(monkeypatch):
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod.current_omni_platform, "set_forward_context", _noop_forward_context)

    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner.requests["r1"].sampling_params = SimpleNamespace(seed=42)
    runner.talker_mtp = CaptureTalkerMTP()
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(subtalker_sampling_params={}))

    def fake_determine(self, num_tokens, num_reqs, num_scheduled_tokens_np, max_num_scheduled_tokens, use_cascade_attn):
        batch_desc = SimpleNamespace(num_tokens=int(num_tokens))
        return (False, batch_desc, None, None, None)

    monkeypatch.setattr(runner, "_determine_batch_execution_and_padding", fake_determine.__get__(runner, type(runner)))

    inputs_embeds = torch.zeros((2, 4), dtype=torch.float32)
    OmniGPUModelRunner._talker_mtp_forward(runner, ["r1"], inputs_embeds)

    assert runner.talker_mtp.calls[0]["generator"] is None


def test_talker_mtp_forward_passes_qwen3_tts_subtalker_sampling_params_to_talker(monkeypatch):
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod.current_omni_platform, "set_forward_context", _noop_forward_context)

    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner.requests["r1"].sampling_params = SimpleNamespace(
        seed=42,
        extra_args={"tts_local_seed": 42},
    )
    runner.talker_mtp = CaptureTalkerMTP()
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            subtalker_sampling_params={
                "do_sample": False,
                "temperature": 0.2,
                "top_k": 9,
                "top_p": 0.55,
            }
        )
    )

    def fake_determine(self, num_tokens, num_reqs, num_scheduled_tokens_np, max_num_scheduled_tokens, use_cascade_attn):
        batch_desc = SimpleNamespace(num_tokens=int(num_tokens))
        return (False, batch_desc, None, None, None)

    monkeypatch.setattr(runner, "_determine_batch_execution_and_padding", fake_determine.__get__(runner, type(runner)))

    inputs_embeds = torch.zeros((2, 4), dtype=torch.float32)
    OmniGPUModelRunner._talker_mtp_forward(runner, ["r1"], inputs_embeds)

    assert runner.talker_mtp.calls == [
        {
            "batch_size": 1,
            "do_sample": False,
            "temperature": 0.2,
            "top_k": 9,
            "top_p": 0.55,
            "generator": runner.talker_mtp.calls[0]["generator"],
            "generators": None,
        }
    ]
    assert runner.talker_mtp.calls[0]["generator"] is not None


def test_talker_mtp_forward_keeps_explicit_seeded_requests_scalar(monkeypatch):
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod.current_omni_platform, "set_forward_context", _noop_forward_context)

    runner = _make_runner(req_ids=("r1", "r2"), hidden_size=4)
    runner.requests["r1"].sampling_params = SimpleNamespace(
        seed=11,
        extra_args={"tts_local_seed": 11},
    )
    runner.requests["r2"].sampling_params = SimpleNamespace(
        seed=22,
        extra_args={"tts_local_seed": 22},
    )
    runner.talker_mtp = CaptureTalkerMTP()
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(subtalker_sampling_params={}))

    def fake_determine(self, num_tokens, num_reqs, num_scheduled_tokens_np, max_num_scheduled_tokens, use_cascade_attn):
        batch_desc = SimpleNamespace(num_tokens=int(num_tokens))
        return (False, batch_desc, None, None, None)

    monkeypatch.setattr(runner, "_determine_batch_execution_and_padding", fake_determine.__get__(runner, type(runner)))

    runner.talker_mtp_input_ids.gpu[:] = torch.tensor([101, 202], dtype=torch.int64)
    runner.talker_mtp_inputs_embeds.gpu[0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    runner.talker_mtp_inputs_embeds.gpu[1] = torch.tensor([10.0, 20.0, 30.0, 40.0])
    saved_input_ids = runner.talker_mtp_input_ids.gpu.clone()
    saved_embeds = runner.talker_mtp_inputs_embeds.gpu.clone()

    inputs_embeds = torch.zeros((6, 4), dtype=torch.float32)
    OmniGPUModelRunner._talker_mtp_forward(runner, ["r1", "r2"], inputs_embeds)

    assert [call["batch_size"] for call in runner.talker_mtp.calls] == [1, 1]
    assert all(call["generator"] is not None for call in runner.talker_mtp.calls)
    assert runner.talker_mtp.calls[0]["generator"] is not runner.talker_mtp.calls[1]["generator"]
    assert torch.equal(runner.talker_mtp_input_ids.gpu, saved_input_ids)
    assert torch.equal(runner.talker_mtp_inputs_embeds.gpu, saved_embeds)


def test_talker_mtp_forward_batches_seeded_requests_for_opted_in_models(monkeypatch):
    """Models with talker_mtp_accepts_per_row_generators get one batched call (#4883)."""
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod.current_omni_platform, "set_forward_context", _noop_forward_context)

    runner = _make_runner(req_ids=("r1", "r2"), hidden_size=4)
    runner.requests["r1"].sampling_params = SimpleNamespace(
        seed=11,
        extra_args={"tts_local_seed": 11},
    )
    runner.requests["r2"].sampling_params = SimpleNamespace(
        seed=22,
        extra_args={"tts_local_seed": 22},
    )
    runner.talker_mtp = CaptureTalkerMTP()
    runner.model = SimpleNamespace(
        talker_mtp_output_key=("codes", "audio"),
        talker_mtp_accepts_per_row_generators=True,
    )
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(subtalker_sampling_params={}))

    def fake_determine(self, num_tokens, num_reqs, num_scheduled_tokens_np, max_num_scheduled_tokens, use_cascade_attn):
        batch_desc = SimpleNamespace(num_tokens=int(num_tokens))
        return (False, batch_desc, None, None, None)

    monkeypatch.setattr(runner, "_determine_batch_execution_and_padding", fake_determine.__get__(runner, type(runner)))

    inputs_embeds = torch.zeros((6, 4), dtype=torch.float32)
    OmniGPUModelRunner._talker_mtp_forward(runner, ["r1", "r2"], inputs_embeds)

    # One batched call with distinct per-row generators, not two scalar calls.
    assert [call["batch_size"] for call in runner.talker_mtp.calls] == [2]
    row_generators = runner.talker_mtp.calls[0]["generators"]
    assert runner.talker_mtp.calls[0]["generator"] is None
    assert len(row_generators) == 2
    assert all(generator is not None for generator in row_generators)
    assert row_generators[0] is not row_generators[1]

    # The per-request generator stream persists across steps...
    OmniGPUModelRunner._talker_mtp_forward(runner, ["r1", "r2"], inputs_embeds)
    assert runner.talker_mtp.calls[1]["generators"][0] is row_generators[0]
    assert runner.talker_mtp.calls[1]["generators"][1] is row_generators[1]

    # ...and is evicted once its request finishes.
    del runner.requests["r2"]
    OmniGPUModelRunner._talker_mtp_forward(runner, ["r1"], inputs_embeds)
    assert set(runner._talker_mtp_generators) == {"r1"}
    assert runner.talker_mtp.calls[2]["generator"] is row_generators[0]


def test_update_intermediate_buffer_writes_to_buffer_and_setattr(monkeypatch):
    """Validate that _update_intermediate_buffer writes to model_intermediate_buffer
    (forward path) and mirrors to additional_information_cpu setattr (backward compat)."""
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod.current_omni_platform, "set_forward_context", _noop_forward_context)

    runner = _make_runner(req_ids=("r1",), hidden_size=4)

    update = {"my_tensor": torch.tensor([1.0, 2.0]), "my_list": [3, 4]}
    OmniGPUModelRunner._update_intermediate_buffer(runner, "r1", update)

    # Forward: buffer is populated
    assert "r1" in runner.model_intermediate_buffer
    buf = runner.model_intermediate_buffer["r1"]
    assert torch.allclose(buf["my_tensor"], torch.tensor([1.0, 2.0]))
    assert buf["my_list"] == [3, 4]

    # Backward compat: setattr is also populated
    info_cpu = runner.requests["r1"].additional_information_cpu
    assert torch.allclose(info_cpu["my_tensor"], torch.tensor([1.0, 2.0]))
    assert info_cpu["my_list"] == [3, 4]


def test_update_intermediate_buffer_accumulates():
    """Validate that successive merges accumulate keys in the buffer."""
    runner = _make_runner(req_ids=("r1",), hidden_size=4)

    OmniGPUModelRunner._update_intermediate_buffer(runner, "r1", {"a": torch.tensor([1.0])})
    OmniGPUModelRunner._update_intermediate_buffer(runner, "r1", {"b": torch.tensor([2.0])})

    buf = runner.model_intermediate_buffer["r1"]
    assert "a" in buf and "b" in buf
    assert torch.allclose(buf["a"], torch.tensor([1.0]))
    assert torch.allclose(buf["b"], torch.tensor([2.0]))


def test_update_additional_information_deserializes_new_request_payload():
    from vllm_omni.engine.serialization import serialize_additional_information

    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    conditioning = {
        "tts_token_ids": torch.tensor([1, 2]),
        "tts_hidden_states": torch.ones(2, 4),
    }
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="r1",
                additional_information=serialize_additional_information(conditioning),
            )
        ],
        scheduled_cached_reqs=SimpleNamespace(),
    )

    OmniGPUModelRunner._update_additional_information(runner, scheduler_output)

    assert torch.equal(runner.model_intermediate_buffer["r1"]["tts_token_ids"], conditioning["tts_token_ids"])
    assert torch.equal(
        runner.model_intermediate_buffer["r1"]["tts_hidden_states"],
        conditioning["tts_hidden_states"],
    )


def test_streaming_new_request_marker_replaces_terminal_chunk_snapshot():
    from vllm_omni.engine.serialization import serialize_additional_information

    runner = _make_runner(req_ids=("r1", "r2"), hidden_size=4)
    runner.model.replace_runtime_additional_information = True
    terminal = {
        "codes": {"audio": torch.tensor([1, 2])},
        "meta": {"cache_epoch": 0, "chunk_seq": 2, "last_chunk": True},
    }
    peer = {
        "codes": {"audio": torch.tensor([9])},
        "meta": {"cache_epoch": 3, "chunk_seq": 1, "last_chunk": False},
    }
    runner.model_intermediate_buffer.update(r1=terminal, r2=peer)
    marker = {
        "meta": {
            "finished": False,
            "is_segment_finished": True,
            "request_finished": False,
            "replace_runtime_additional_information": True,
        }
    }
    new_req = SimpleNamespace(
        req_id="r1",
        model_intermediate_buffer=marker,
        additional_information=serialize_additional_information(terminal),
    )

    OmniGPUModelRunner._update_streaming_input_additional_info(runner, new_req, "r1")
    OmniGPUModelRunner._update_additional_information(
        runner,
        SimpleNamespace(
            scheduled_new_reqs=[new_req],
            scheduled_cached_reqs=SimpleNamespace(),
        ),
    )

    info = runner.model_intermediate_buffer["r1"]
    assert "codes" not in info
    assert info["meta"] == {
        **marker["meta"],
        "num_processed_tokens": 0,
        "resumable": True,
    }
    assert runner.requests["r1"].additional_information_cpu == info
    assert runner.model_intermediate_buffer["r2"] == peer


def test_cached_empty_marker_replaces_terminal_chunk_snapshot():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner.model.replace_runtime_additional_information = True
    runner.model_intermediate_buffer["r1"] = {
        "codes": {"audio": torch.tensor([1, 2])},
        "meta": {"cache_epoch": 0, "chunk_seq": 2, "last_chunk": True},
    }
    marker = {
        "meta": {
            "is_segment_finished": torch.tensor(True, dtype=torch.bool),
            "replace_runtime_additional_information": True,
        }
    }

    OmniGPUModelRunner._update_additional_information(
        runner,
        SimpleNamespace(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(additional_information={"r1": marker}),
        ),
    )

    assert runner.model_intermediate_buffer["r1"] == marker
    assert runner.requests["r1"].additional_information_cpu == marker


def test_update_intermediate_buffer_skips_empty_update():
    """Validate that an empty update dict is a no-op."""
    runner = _make_runner(req_ids=("r1",), hidden_size=4)

    OmniGPUModelRunner._update_intermediate_buffer(runner, "r1", {})

    assert "r1" not in runner.model_intermediate_buffer


def test_update_intermediate_buffer_skips_unknown_req_id():
    """Validate that merge is a no-op when req_id is not in self.requests."""
    runner = _make_runner(req_ids=("r1",), hidden_size=4)

    OmniGPUModelRunner._update_intermediate_buffer(runner, "unknown_req", {"key": torch.tensor([1.0])})

    assert "unknown_req" not in runner.model_intermediate_buffer


def test_streaming_input_update_merges_model_intermediate_buffer():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner.model_intermediate_buffer["r1"] = {
        "duplex": {
            "session_id": "sid",
            "seq": 1,
        }
    }
    runner.requests["r1"].additional_information_cpu = runner.model_intermediate_buffer["r1"]
    new_req_data = SimpleNamespace(
        model_intermediate_buffer={
            "duplex": {
                "session_id": "sid",
                "seq": 2,
                "payload": {"type": "audio"},
            }
        },
        additional_information=None,
    )

    OmniGPUModelRunner._update_streaming_input_additional_info(runner, new_req_data, "r1")

    info = runner.model_intermediate_buffer["r1"]
    assert info["duplex"]["session_id"] == "sid"
    assert info["duplex"]["seq"] == 2
    assert info["duplex"]["payload"] == {"type": "audio"}
    assert runner.requests["r1"].additional_information_cpu is info


def _make_full_payload_accumulation_runner(
    model_arch="Qwen3OmniMoeForConditionalGeneration",
    model_stage="talker",
    async_chunk=False,
    final_output=False,
    custom_process_next_stage_input_func="module.full_payload",
):
    runner = object.__new__(OmniConnectorModelRunnerMixin)
    runner.model_config = SimpleNamespace(
        model_arch=model_arch,
        model_stage=model_stage,
        async_chunk=async_chunk,
        final_output=final_output,
        custom_process_next_stage_input_func=custom_process_next_stage_input_func,
    )
    runner._custom_process_func = object()
    runner._pending_full_payload_send = {}
    runner._stage_id = 1
    runner._omni_connector = object()
    return runner


def test_accumulate_full_payload_output_preserves_aligned_all_zero_qwen3_omni_codec_rows():
    runner = _make_full_payload_accumulation_runner()
    request = SimpleNamespace(output_token_ids=[0, 1])
    codes = torch.zeros((2, 3), dtype=torch.long)

    OmniConnectorModelRunnerMixin.accumulate_full_payload_output(runner, "r1", {"codes.audio": codes}, request)

    stored, _ = OmniConnectorModelRunnerMixin._materialize_full_payload_entry(runner._pending_full_payload_send["r1"])
    assert torch.equal(stored["codes.audio"], codes)


def test_accumulate_full_payload_output_keeps_misaligned_all_zero_qwen3_omni_codec_rows():
    # After removing the sender-side zero filter, the accumulator keeps every
    # codec row including misaligned all-zero rows. The downstream consumer
    # (_extract_qwen3_full_payload_codec_rows) is the authoritative crop and
    # filters by output_token_ids.
    runner = _make_full_payload_accumulation_runner()
    request = SimpleNamespace(output_token_ids=[0, 1])
    codes = torch.zeros((1, 3), dtype=torch.long)

    OmniConnectorModelRunnerMixin.accumulate_full_payload_output(runner, "r1", {"codes.audio": codes}, request)

    stored, _ = OmniConnectorModelRunnerMixin._materialize_full_payload_entry(runner._pending_full_payload_send["r1"])
    assert "codes.audio" in stored
    assert torch.equal(stored["codes.audio"], codes)


def test_accumulate_full_payload_output_preserves_incremental_aligned_all_zero_qwen3_omni_codec_rows():
    runner = _make_full_payload_accumulation_runner()
    request = SimpleNamespace(output_token_ids=[0, 1])
    runner._pending_full_payload_send["r1"] = (
        {"codes.audio": torch.ones((1, 3), dtype=torch.long)},
        request,
    )
    codes = torch.zeros((1, 3), dtype=torch.long)

    OmniConnectorModelRunnerMixin.accumulate_full_payload_output(runner, "r1", {"codes.audio": codes}, request)

    stored, _ = OmniConnectorModelRunnerMixin._materialize_full_payload_entry(runner._pending_full_payload_send["r1"])
    assert stored["codes.audio"].shape == (2, 3)
    assert torch.equal(stored["codes.audio"][1], torch.zeros(3, dtype=torch.long))


def test_accumulate_full_payload_output_keeps_all_zero_qwen3_omni_prefill_placeholder():
    # Prefill placeholder rows (output_token_ids empty) are no longer dropped
    # at the sender. The consumer-side crop trims them off using
    # output_token_ids, so the end-to-end semantics are unchanged.
    runner = _make_full_payload_accumulation_runner()
    request = SimpleNamespace(output_token_ids=[])
    codes = torch.zeros((2, 3), dtype=torch.long)

    OmniConnectorModelRunnerMixin.accumulate_full_payload_output(runner, "r1", {"codes.audio": codes}, request)

    stored, _ = OmniConnectorModelRunnerMixin._materialize_full_payload_entry(runner._pending_full_payload_send["r1"])
    assert "codes.audio" in stored
    assert torch.equal(stored["codes.audio"], codes)


def test_full_payload_output_accumulation_hook_matrix():
    assert _make_full_payload_accumulation_runner(model_stage="thinker")._should_accumulate_full_payload_output()
    assert _make_full_payload_accumulation_runner(model_stage="talker")._should_accumulate_full_payload_output()
    assert not _make_full_payload_accumulation_runner(
        model_stage="code2wav", final_output=True
    )._should_accumulate_full_payload_output()
    assert not _make_full_payload_accumulation_runner(
        model_stage="token2audio",
        custom_process_next_stage_input_func=None,
    )._should_accumulate_full_payload_output()
    assert not _make_full_payload_accumulation_runner(
        model_stage="talker", async_chunk=True
    )._should_accumulate_full_payload_output()
    for model_arch in (
        "Qwen3TTSForConditionalGeneration",
        "Qwen2_5OmniForConditionalGeneration",
    ):
        runner = _make_full_payload_accumulation_runner(model_arch=model_arch)
        runner._custom_process_func = None
        assert not runner._should_accumulate_full_payload_output()


def _make_request_end_payload_runner(*, enabled=True, prefix_cache=None):
    runner = object.__new__(GPUARModelRunner)
    runner.model = SimpleNamespace(omni_payload_at_request_end=enabled)
    runner.omni_prefix_cache = prefix_cache
    runner.model_config = SimpleNamespace(
        model_arch="IndexTTS25TalkerForConditionalGeneration",
        model_stage="indextts2_5_talker",
        async_chunk=False,
        final_output=False,
        custom_process_next_stage_input_func="module.full_payload",
    )
    runner._custom_process_func = object()
    runner._pending_full_payload_send = {}
    runner._stage_id = 0
    runner._omni_connector = object()
    return runner


def test_request_end_payload_d2h_gate_requires_opt_in_and_no_prefix_cache():
    assert _make_request_end_payload_runner()._should_defer_full_payload_d2h()
    assert not _make_request_end_payload_runner(enabled=False)._should_defer_full_payload_d2h()
    assert not _make_request_end_payload_runner(prefix_cache=object())._should_defer_full_payload_d2h()


def test_request_end_payload_suppresses_per_step_multimodal_outputs():
    runner = _make_request_end_payload_runner()

    def unexpected_build(_payload):
        raise AssertionError("request-end payloads must stay inside the GPU accumulator")

    runner._build_multimodal_outputs = unexpected_build
    pooler_inter = [{"codes.mel": torch.tensor([[7]])}]

    inter_stage, client = runner._build_omni_step_outputs(
        pooler_inter,
        pooler_inter,
        defer_full_payload_d2h=True,
    )

    assert inter_stage is None
    assert client is None


def test_sync_local_stage_payloads_retains_payload_until_request_is_active():
    runner = object.__new__(OmniGPUModelRunner)
    payload = {"codes": {"audio": [1, 2, 3]}}
    runner._local_stage_payload_cache = {"late": payload}
    runner._full_payload_pending_broadcast_req_ids = set()
    runner.requests = {}
    runner.model_intermediate_buffer = {}

    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    assert runner._local_stage_payload_cache == {"late": payload}
    assert runner.model_intermediate_buffer == {}

    runner.requests = {"late": DummyReqState()}
    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    assert runner._local_stage_payload_cache == {}
    assert runner.model_intermediate_buffer["late"] == payload
    assert runner.requests["late"].additional_information_cpu == payload


def test_maybe_attach_mimo_audio_req_infos_enriches_dict():
    runner = _make_runner_for_mimo()
    req_id = "r_mimo"
    req_state = runner.requests[req_id]

    # Existing req_infos should be copied and enriched, not mutated in place.
    original_req_infos = {"existing": 1}
    enriched = OmniGPUModelRunner._maybe_attach_mimo_audio_req_infos(runner, req_state, original_req_infos, req_id)

    assert enriched is not original_req_infos
    assert enriched["existing"] == 1
    # mm_features should be filled from req_state when missing
    assert enriched["mm_features"] == req_state.mm_features
    # req_id should always be attached
    assert enriched["req_id"] == req_id


def test_maybe_attach_mimo_audio_req_infos_no_req_state_returns_input():
    runner = _make_runner_for_mimo()
    req_id = "missing"
    req_state = None
    req_infos = {"k": "v"}

    result = OmniGPUModelRunner._maybe_attach_mimo_audio_req_infos(runner, req_state, req_infos, req_id)

    # When no req_state, helper should be a no-op.
    assert result is req_infos
