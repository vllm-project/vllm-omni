# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.utils.torch_utils import weak_ref_tensors
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.cudagraph_dispatcher import CUDAGraphMode
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_model_runner import (
    OmniGPUModelRunner,
    _filter_mrope_kwargs_for_model,
)
from vllm_omni.worker.lmcache_model_runner_mixin import LMCacheHiddenStateMixin
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_model_forward_preserves_omni_payload_after_graph_weak_ref(monkeypatch):
    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    payload = {
        "latent": hidden,
        "latent_input_ids": torch.tensor([[21], [22]]),
        "latent_positions": torch.tensor([[10], [11]]),
    }
    original = OmniOutput(text_hidden_states=hidden, multimodal_outputs=payload)
    # CUDAGraphWrapper weak-references its outputs both at capture and replay.
    # vLLM converts NamedTuple outputs into plain tuples along this path.
    # Only the CUDA storage-alias primitive is replaced for this CPU test;
    # the upstream container conversion is exercised unchanged.
    monkeypatch.setattr("vllm.utils.torch_utils.weak_ref_tensor", lambda value: value)
    replay = weak_ref_tensors(original)
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(have_multimodal_outputs=True)
    runner._build_model_kwargs_extra = lambda: {}
    monkeypatch.setattr(GPUModelRunner, "_model_forward", lambda *_args, **_kwargs: replay)

    output = runner._model_forward()
    output_hidden, output_payload = runner.extract_multimodal_outputs(output)

    assert isinstance(output, OmniOutput)
    torch.testing.assert_close(output_hidden, hidden)
    assert output_payload is payload


def test_model_forward_keeps_auxiliary_hidden_tuple(monkeypatch):
    hidden = torch.ones(2, 4)
    auxiliary = (hidden, hidden.clone(), None, None)
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(have_multimodal_outputs=True)
    runner._build_model_kwargs_extra = lambda: {}
    monkeypatch.setattr(GPUModelRunner, "_model_forward", lambda *_args, **_kwargs: auxiliary)

    assert runner._model_forward() is auxiliary


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

    mm_features: list[str]
    additional_information_cpu: dict


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


@pytest.mark.parametrize("replace_payload", [False, True])
def test_resumed_streaming_payload_converts_to_typed_model_input(replace_payload):
    from vllm_omni.core.sched.output import OmniNewRequestData
    from vllm_omni.data_entry_keys import to_struct

    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner.model.replace_runtime_additional_information = replace_payload
    payload = {
        "codes": {"audio": torch.tensor([1, 2])},
        "meta": {"req_id": ["r1"], "stream_finished": torch.tensor(False)},
    }
    new_req = OmniNewRequestData(
        req_id="r1",
        prompt_token_ids=[1, 2],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=0,
        lora_request=None,
        model_intermediate_buffer=payload,
    )

    OmniGPUModelRunner._update_streaming_input_additional_info(runner, new_req, "r1")
    # CosyVoice and other typed-payload consumers perform this conversion
    # after the runner adds its resume metadata.
    result = to_struct(runner.model_intermediate_buffer["r1"])
    assert result.meta.resumable is True
    assert result.meta.num_processed_tokens == 0
    assert result.meta.req_id == ["r1"]
    torch.testing.assert_close(result.codes.audio, payload["codes"]["audio"])


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
    # A stage may publish a user-visible output and still feed another stage
    # (Qwen3-Omni thinker publishes text while sending audio state to talker).
    assert _make_full_payload_accumulation_runner(
        model_stage="thinker", final_output=True
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
    runner.omni_prefix_cache = prefix_cache
    runner.model = SimpleNamespace(omni_payload_at_request_end=enabled)
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


def _make_phase_runner(monkeypatch, rows, *, batched_decode):
    """Exercise real preprocessing and MTP routing with CPU model buffers.

    Each row is (request id, prompt length, computed tokens, scheduled tokens).
    Only model computation and the device forward context are replaced.
    """
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))
    monkeypatch.setattr(mod.current_omni_platform, "set_forward_context", _noop_forward_context)
    runner = _make_runner(req_ids=tuple(row[0] for row in rows))
    total = sum(row[3] for row in rows)
    runner.supports_mm_inputs = False
    runner.enable_prompt_embeds = False
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner.has_talker_mtp = True
    runner.model_config = SimpleNamespace(is_encoder_decoder=False)
    runner.vllm_config.model_config.async_chunk = True
    runner._init_model_kwargs = dict
    runner.input_ids = DummyBuffer(torch.arange(total, dtype=torch.int64))
    runner.inputs_embeds = DummyBuffer(torch.zeros(total, 4))
    runner.positions = torch.arange(total)
    runner.input_batch.num_computed_tokens_cpu = [row[2] for row in rows]
    offsets = [0]
    for rid, prompt_len, _, scheduled in rows:
        runner.requests[rid].prompt_token_ids = list(range(prompt_len))
        offsets.append(offsets[-1] + scheduled)
        # Runner-owned metadata must override stale model/intermediate state.
        runner.model_intermediate_buffer[rid] = {
            "_omni_is_prefill": "stale",
            "_omni_prompt_len": -1,
            "_omni_num_computed_tokens": -1,
        }
    runner.query_start_loc.cpu = torch.tensor(offsets, dtype=torch.int32)
    calls = []
    batch_calls = []

    def preprocess(input_ids, input_embeds, **info):
        calls.append(("normal", dict(info)))
        embeds = input_ids.float().view(-1, 1).expand(-1, 4).clone()
        updates = {}
        if not info["_omni_is_prefill"]:
            updates["mtp_inputs"] = (torch.zeros_like(embeds), torch.zeros_like(embeds))
        return input_ids, embeds, updates

    def preprocess_decode_batch(input_ids, req_infos):
        batch_calls.append([info["request_id"] for info in req_infos])
        calls.extend(("batch", dict(info)) for info in req_infos)
        embeds = input_ids.float().view(-1, 1).expand(-1, 4).clone()
        return input_ids, embeds, torch.zeros_like(embeds), torch.zeros_like(embeds), [{} for _ in req_infos]

    runner.model = SimpleNamespace(
        has_preprocess=True,
        preprocess=preprocess,
        talker_mtp_output_key=("codes", "audio"),
    )
    if batched_decode:
        runner.model.preprocess_decode_batch = preprocess_decode_batch
    output = SchedulerOutput(
        total_num_scheduled_tokens=total,
        num_scheduled_tokens={row[0]: row[3] for row in rows},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    return runner, output, calls, batch_calls


@pytest.mark.parametrize("batched_decode", [False, True], ids=["normal", "batch"])
@pytest.mark.parametrize("order", ["interleaved", "reversed", "adjacent"])
def test_preprocess_phase_contract_mixed_batch(monkeypatch, batched_decode, order):
    rows = [
        ("decode-before", 5, 5, 1),
        ("tail", 5, 4, 1),
        ("decode-after", 5, 6, 1),
        ("single-prompt", 1, 0, 1),
        ("cached-prefill", 8, 4, 3),
        ("multi-token-decode", 5, 5, 2),
    ]
    if order == "reversed":
        rows.reverse()
    elif order == "adjacent":
        rows = [rows[1], rows[0], rows[2], *rows[3:]]
    runner, scheduled, calls, batch_calls = _make_phase_runner(monkeypatch, rows, batched_decode=batched_decode)
    _, embeds, *_ = runner._preprocess(scheduled, scheduled.total_num_scheduled_tokens)

    if batched_decode:
        decode_order = [rid for rid, _, _, _ in rows if rid.startswith("decode-")]
        expected_batches = [decode_order] if order == "adjacent" else [[rid] for rid in decode_order]
        assert batch_calls == expected_batches
    else:
        assert batch_calls == []
    assert len(calls) == len(rows)
    offset = 0
    code_index = 0
    for (route, info), (rid, prompt_len, computed, span) in zip(calls, rows, strict=True):
        assert info["request_id"] == rid
        assert type(info["_omni_prompt_len"]) is int
        assert type(info["_omni_num_computed_tokens"]) is int
        assert type(info["_omni_is_prefill"]) is bool
        assert info["_omni_prompt_len"] == prompt_len
        assert info["_omni_num_computed_tokens"] == computed
        assert info["_omni_is_prefill"] == (computed < prompt_len)
        mtp_eligible = rid in {"decode-before", "decode-after"}
        assert route == ("batch" if batched_decode and mtp_eligible else "normal")
        expected = torch.arange(offset, offset + span).float().view(-1, 1).expand(-1, 4)
        torch.testing.assert_close(embeds[offset : offset + span], expected + int(mtp_eligible))
        assert ("codes" in runner.model_intermediate_buffer[rid]) == mtp_eligible
        if mtp_eligible:
            torch.testing.assert_close(
                runner.model_intermediate_buffer[rid]["codes"]["audio"],
                torch.tensor([[code_index]], dtype=torch.int64),
                rtol=0,
                atol=0,
            )
            code_index += 1
        offset += span


@pytest.mark.parametrize("batched_decode", [False, True], ids=["normal", "batch"])
def test_preprocess_one_token_chunked_prefill_tail_then_decode(monkeypatch, batched_decode):
    # Replay a five-token prompt split at a four-token scheduling budget.
    runner, scheduled, calls, _ = _make_phase_runner(monkeypatch, [("r", 5, 0, 4)], batched_decode=batched_decode)
    runner._preprocess(scheduled, 4)
    assert calls[-1][1]["_omni_is_prefill"] is True
    assert "codes" not in runner.model_intermediate_buffer["r"]

    for computed, is_prefill in [(4, True), (5, False)]:
        runner.input_batch.num_computed_tokens_cpu = [computed]
        scheduled.total_num_scheduled_tokens = 1
        scheduled.num_scheduled_tokens = {"r": 1}
        runner.query_start_loc.cpu = torch.tensor([0, 1], dtype=torch.int32)
        runner.input_ids.gpu[0] = 42
        _, embeds, *_ = runner._preprocess(scheduled, 1)
        route, info = calls[-1]
        assert info["_omni_is_prefill"] is is_prefill
        assert info["_omni_num_computed_tokens"] == computed
        assert info["_omni_prompt_len"] == 5
        assert route == ("batch" if batched_decode and not is_prefill else "normal")
        torch.testing.assert_close(embeds, torch.full((1, 4), 42.0 if is_prefill else 43.0))
        assert ("codes" in runner.model_intermediate_buffer["r"]) == (not is_prefill)


class DecodeOnlyPreprocessModel:
    has_preprocess = True

    def __init__(self, *, rewrite_ids=False, invalid_output=None):
        self.rewrite_ids = rewrite_ids
        self.invalid_output = invalid_output
        self.calls = []

    @staticmethod
    def embed_input_ids(input_ids, **kwargs):
        return input_ids.float().view(-1, 1).expand(-1, 4).clone()

    def preprocess(self, input_ids, input_embeds, **info):
        self.calls.append(("scalar", [info["request_id"]], [info["_omni_is_prefill"]]))
        return input_ids, self.embed_input_ids(input_ids), {}

    def preprocess_decode_batch(self, *, input_ids, req_infos):
        self.calls.append(
            ("batch", [info["request_id"] for info in req_infos], [info["_omni_is_prefill"] for info in req_infos])
        )
        ids = input_ids + 100 if self.rewrite_ids else input_ids
        embeds = self.embed_input_ids(ids) + 10
        updates = [{"decode_marker": info["request_id"]} for info in req_infos]
        if self.invalid_output == "embeddings":
            embeds = embeds[:-1]
        elif self.invalid_output == "ids":
            ids = ids[:-1]
        elif self.invalid_output == "updates":
            updates = updates[:-1]
        return ids, embeds, updates


def _without_mtp_buffers(runner):
    runner.has_talker_mtp = False
    for name in ("talker_mtp", "talker_mtp_input_ids", "talker_mtp_inputs_embeds", "last_talker_hidden", "text_step"):
        delattr(runner, name)


@pytest.mark.parametrize("rewrite_ids", [False, True], ids=["original-ids", "replaced-ids"])
@pytest.mark.parametrize("order", ["interleaved", "reversed", "adjacent"])
def test_decode_batch_without_mtp_preserves_mixed_rows_and_buffers(monkeypatch, rewrite_ids, order):
    rows = [
        ("cfg-uncond", 5, 5, 1),
        ("prefill-tail", 5, 4, 1),
        ("cfg-cond", 5, 6, 1),
        ("single-prompt", 1, 0, 1),
        ("multi-token-decode", 5, 5, 2),
    ]
    if order == "reversed":
        rows.reverse()
    elif order == "adjacent":
        rows = [rows[1], rows[0], rows[2], *rows[3:]]
    runner, scheduled, _, _ = _make_phase_runner(monkeypatch, rows, batched_decode=False)
    _without_mtp_buffers(runner)
    model = DecodeOnlyPreprocessModel(rewrite_ids=rewrite_ids)
    runner.model = model
    total = scheduled.total_num_scheduled_tokens
    runner.inputs_embeds = DummyBuffer(torch.full((total + 2, 4), -99.0))
    ids_buffer = runner.input_ids.gpu
    embeds_buffer = runner.inputs_embeds.gpu
    output_ids, output_embeds, *_ = runner._preprocess(scheduled, total)
    assert output_ids.data_ptr() == ids_buffer.data_ptr()
    assert output_embeds.data_ptr() == embeds_buffer.data_ptr()
    assert torch.equal(embeds_buffer[total:], torch.full((2, 4), -99.0))
    seen = [
        (route, rid, prefill) for route, ids, phases in model.calls for rid, prefill in zip(ids, phases, strict=True)
    ]
    offset = 0
    for (route, rid, prefill), (expected_id, prompt, computed, span) in zip(seen, rows, strict=True):
        eligible = span == 1 and computed >= prompt
        assert rid == expected_id
        assert prefill == (computed < prompt)
        assert route == ("batch" if eligible else "scalar")
        expected_ids = torch.arange(offset, offset + span) + (100 if eligible and rewrite_ids else 0)
        assert torch.equal(output_ids[offset : offset + span], expected_ids)
        expected_embeds = expected_ids.float().view(-1, 1).expand(-1, 4) + (10 if eligible else 0)
        torch.testing.assert_close(output_embeds[offset : offset + span], expected_embeds)
        info = runner.model_intermediate_buffer[rid]
        assert info.get("decode_marker") == (rid if eligible else None)
        offset += span


def test_decode_batch_without_mtp_accepts_wrapper_input_ids_none(monkeypatch):
    rows = [("first", 5, 5, 1), ("second", 5, 6, 1)]
    runner, scheduled, _, _ = _make_phase_runner(monkeypatch, rows, batched_decode=False)
    _without_mtp_buffers(runner)
    model = DecodeOnlyPreprocessModel()
    runner.model = model
    runner.supports_mm_inputs = True
    runner.maybe_get_ec_connector_output = _noop_forward_context
    runner.encoder_cache = {}
    runner._execute_mm_encoder = lambda output: None
    runner._gather_mm_embeddings = lambda output: ([], None)
    runner._extract_mm_kwargs = lambda output: {}
    runner._prepare_mm_inputs = lambda size: (None, runner.inputs_embeds.gpu[:size])
    ids, embeds, *_ = runner._preprocess(scheduled, 2)
    assert ids.data_ptr() == runner.input_ids.gpu.data_ptr()
    assert model.calls == [("batch", ["first", "second"], [False, False])]
    torch.testing.assert_close(embeds, torch.tensor([[10.0] * 4, [11.0] * 4]))


@pytest.mark.parametrize(
    "offsets",
    [
        pytest.param([0, 2, 3], id="gap"),
        pytest.param([1, 0, 2], id="reversed"),
        pytest.param([0, 0, 1], id="duplicate"),
        pytest.param([-1, 0, 1], id="negative-contiguous-start"),
        pytest.param([1, 2, 3], id="contiguous-out-of-bounds"),
    ],
)
def test_decode_batch_without_mtp_rejects_invalid_offsets_before_model_hook(monkeypatch, offsets):
    rows = [("first", 5, 5, 1), ("second", 5, 6, 1)]
    runner, scheduled, _, _ = _make_phase_runner(monkeypatch, rows, batched_decode=False)
    _without_mtp_buffers(runner)
    model = DecodeOnlyPreprocessModel(rewrite_ids=True)
    runner.model = model
    scalar_hook = Mock(wraps=model.preprocess)
    batch_hook = Mock(wraps=model.preprocess_decode_batch)
    monkeypatch.setattr(model, "preprocess", scalar_hook)
    monkeypatch.setattr(model, "preprocess_decode_batch", batch_hook)
    runner.query_start_loc.cpu = torch.tensor(offsets, dtype=torch.int32)
    runner.inputs_embeds = DummyBuffer(torch.full((4, 4), -99.0))
    ids_before = runner.input_ids.gpu.clone()
    embeds_before = runner.inputs_embeds.gpu.clone()

    with pytest.raises(
        RuntimeError,
        match="Non-MTP batched decode preprocessing requires contiguous in-bounds token offsets",
    ):
        runner._preprocess(scheduled, scheduled.total_num_scheduled_tokens)

    # These checks remain active when the runner is imported under python -O.
    scalar_hook.assert_not_called()
    batch_hook.assert_not_called()
    torch.testing.assert_close(runner.input_ids.gpu, ids_before, rtol=0, atol=0)
    torch.testing.assert_close(runner.inputs_embeds.gpu, embeds_before, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["embeddings", "ids", "updates"])
def test_decode_batch_without_mtp_rejects_incomplete_outputs(monkeypatch, invalid):
    rows = [("first", 5, 5, 1), ("second", 5, 6, 1)]
    runner, scheduled, _, _ = _make_phase_runner(monkeypatch, rows, batched_decode=False)
    _without_mtp_buffers(runner)
    runner.model = DecodeOnlyPreprocessModel(invalid_output=invalid)
    before = runner.inputs_embeds.gpu.clone()
    with pytest.raises(ValueError, match="decode preprocessing"):
        runner._preprocess(scheduled, 2)
    assert torch.equal(runner.inputs_embeds.gpu, before)


@pytest.mark.parametrize("supports_boundaries", [False, True])
def test_multimodal_embeddings_receive_opt_in_host_request_boundaries(supports_boundaries):
    runner = object.__new__(OmniGPUModelRunner)
    received = {}

    def embed(ids, **kwargs):
        received.update(kwargs)
        return ids

    runner.model = SimpleNamespace(supports_embed_input_ids_query_start_loc=supports_boundaries, embed_input_ids=embed)
    runner.input_batch = SimpleNamespace(num_reqs=3)
    runner.query_start_loc = SimpleNamespace(cpu=torch.tensor([0, 6, 7, 11, 999]))
    runner.input_ids = SimpleNamespace(gpu=torch.arange(16))
    mm = [torch.ones(2, 4), torch.ones(1, 4)]
    mask = torch.tensor([True] * 4 + [False] * 3 + [True] * 3 + [False])

    result = runner._embed_multimodal_input_ids(11, mm, mask)

    torch.testing.assert_close(result, torch.arange(11))
    assert received["multimodal_embeddings"] is mm
    assert received["is_multimodal"] is mask
    if supports_boundaries:
        assert received["query_start_loc"] == [0, 6, 7, 11]
    else:
        assert "query_start_loc" not in received


# ---------------------------------------------------------------------------
# LMCache HS chunk-boundary buffering tests
# ---------------------------------------------------------------------------


class _FakeHSStore:
    """Records every store_hidden_states call so tests can assert behavior."""

    def __init__(self):
        self.calls = []

    def store_hidden_states(self, token_ids, hidden_states, *, layer_idx=0, token_offset=0):
        self.calls.append(
            SimpleNamespace(
                token_ids=list(token_ids),
                hidden_states=hidden_states.clone(),
                layer_idx=layer_idx,
                token_offset=token_offset,
            )
        )


def _make_lmcache_runner(chunk_size=4, hidden_size=2, req_id="r1", token_capacity=64):
    """Build a runner stub wired with the LMCache HS path but no real engine."""
    runner = object.__new__(LMCacheHiddenStateMixin)
    runner._has_lmcache = True
    runner._lmcache_hs_mm_keys = ()
    runner._hs_pending_buffer = {}
    runner._hs_saved_boundary = {}
    runner._hs_buffer_origin = {}
    runner._hs_mm_features = {}

    hs_store = _FakeHSStore()
    engine = SimpleNamespace(
        hidden_state_store=hs_store,
        config=SimpleNamespace(chunk_size=chunk_size),
    )
    adapter = SimpleNamespace(lmcache_engine=engine)
    runner._get_lmcache_adapter = lambda: adapter

    runner.input_batch = SimpleNamespace(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        num_computed_tokens_cpu=torch.tensor([0]),
        token_ids_cpu=torch.arange(token_capacity).reshape(1, token_capacity),
    )
    runner.query_start_loc = SimpleNamespace(cpu=torch.tensor([0]))
    return runner, hs_store


def _drive_step(runner, sched, num_computed, hs_rows, hidden_size=2):
    """Simulate one forward: update num_computed_tokens_cpu and feed HS."""
    runner.input_batch.num_computed_tokens_cpu = torch.tensor([num_computed])
    hidden_states = torch.arange(hs_rows * hidden_size, dtype=torch.float32).reshape(hs_rows, hidden_size) + (
        num_computed * 100.0
    )
    sched_out = SimpleNamespace(num_scheduled_tokens={"r1": sched})
    LMCacheHiddenStateMixin._maybe_store_hs_to_lmcache(
        runner,
        hidden_states,
        None,
        num_tokens_unpadded=hs_rows,
        scheduler_output=sched_out,
    )


def test_hs_lmcache_prefill_stores_full_chunks_only():
    runner, hs_store = _make_lmcache_runner(chunk_size=4, hidden_size=2)

    # Prefill 10 tokens, chunk_size 4 → new_boundary = 8 (chunks [0:4],[4:8]).
    _drive_step(runner, sched=10, num_computed=0, hs_rows=10)

    assert len(hs_store.calls) == 1
    call = hs_store.calls[0]
    assert call.token_offset == 0
    assert call.token_ids == list(range(8))
    assert call.hidden_states.shape == (8, 2)
    assert runner._hs_saved_boundary["r1"] == 8
    # The trailing 2 rows stay in the buffer for the next boundary crossing.
    assert sum(t.shape[0] for t in runner._hs_pending_buffer["r1"]["hidden"]) == 2


def _drive_positioned_step(runner, sched, num_computed, hidden_size=2):
    """One forward whose every HS row is stamped with its absolute position."""
    runner.input_batch.num_computed_tokens_cpu = torch.tensor([num_computed])
    positions = torch.arange(num_computed, num_computed + sched, dtype=torch.float32)
    hidden_states = positions.unsqueeze(1).repeat(1, hidden_size)
    LMCacheHiddenStateMixin._maybe_store_hs_to_lmcache(
        runner,
        hidden_states,
        None,
        num_tokens_unpadded=sched,
        scheduler_output=SimpleNamespace(num_scheduled_tokens={"r1": sched}),
    )


def _stored_positions(call):
    return [int(v) for v in call.hidden_states[:, 0].tolist()]


def test_hs_store_keeps_absolute_positions_after_an_unaligned_hit():
    """A full-prompt hit leaves num_computed one short of the chunk boundary.

    The buffer then starts mid-chunk. Storing its first rows at the rounded-down
    boundary would file them under earlier token positions than they hold.
    """
    runner, hs_store = _make_lmcache_runner(chunk_size=4, hidden_size=2)

    # LMCache covered 8 tokens; vLLM must recompute the last one, so the
    # request arrives with num_computed=7.
    for step, num_computed in enumerate(range(7, 12)):
        _drive_positioned_step(runner, sched=1, num_computed=num_computed)

    assert len(hs_store.calls) == 1
    call = hs_store.calls[0]
    assert call.token_offset == 8
    assert _stored_positions(call) == [8, 9, 10, 11]
    assert call.token_ids == list(range(12))
    assert runner._hs_saved_boundary["r1"] == 12


def test_hs_store_positions_stay_aligned_across_two_chunks():
    """The second flush must continue where the first stopped, not re-skip."""
    runner, hs_store = _make_lmcache_runner(chunk_size=4, hidden_size=2)

    for num_computed in range(7, 20):
        _drive_positioned_step(runner, sched=1, num_computed=num_computed)

    assert [c.token_offset for c in hs_store.calls] == [8, 12, 16]
    assert [_stored_positions(c) for c in hs_store.calls] == [
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
    ]


def test_hs_store_from_a_cold_request_starts_at_zero():
    """Rounding up must not shift a request that had no hit at all."""
    runner, hs_store = _make_lmcache_runner(chunk_size=4, hidden_size=2)

    _drive_positioned_step(runner, sched=10, num_computed=0)

    assert len(hs_store.calls) == 1
    assert hs_store.calls[0].token_offset == 0
    assert _stored_positions(hs_store.calls[0]) == list(range(8))


def test_hs_lmcache_decode_buffers_until_boundary():
    runner, hs_store = _make_lmcache_runner(chunk_size=4, hidden_size=2)

    # Step 1: prefill exactly 2 full chunks (8 tokens, no remainder).
    _drive_step(runner, sched=8, num_computed=0, hs_rows=8)
    assert len(hs_store.calls) == 1
    assert runner._hs_saved_boundary["r1"] == 8
    assert sum(t.shape[0] for t in runner._hs_pending_buffer["r1"]["hidden"]) == 0
    hs_store.calls.clear()

    # Decode tokens 9, 10, 11 — chunk [8:12] not yet full, must not flush.
    for nc in (8, 9, 10):
        _drive_step(runner, sched=1, num_computed=nc, hs_rows=1)
    assert hs_store.calls == []
    assert runner._hs_saved_boundary["r1"] == 8
    assert sum(t.shape[0] for t in runner._hs_pending_buffer["r1"]["hidden"]) == 3

    # Decode token 12 completes the chunk [8:12]; one flush expected.
    _drive_step(runner, sched=1, num_computed=11, hs_rows=1)
    assert len(hs_store.calls) == 1
    call = hs_store.calls[0]
    assert call.token_offset == 8
    assert call.token_ids == list(range(12))
    assert call.hidden_states.shape == (4, 2)
    assert runner._hs_saved_boundary["r1"] == 12
    assert sum(t.shape[0] for t in runner._hs_pending_buffer["r1"]["hidden"]) == 0


def test_hs_lmcache_drop_pending_state():
    runner, _ = _make_lmcache_runner()
    runner._hs_pending_buffer["r1"] = {"hidden": [torch.zeros((2, 2))]}
    runner._hs_saved_boundary["r1"] = 4

    runner._drop_hs_pending_state("r1")

    assert "r1" not in runner._hs_pending_buffer
    assert "r1" not in runner._hs_saved_boundary
    # Dropping an unknown req_id must be a no-op (idempotent).
    runner._drop_hs_pending_state("unknown")


class _RaisingHSStore(_FakeHSStore):
    def store_hidden_states(self, token_ids, hidden_states, *, layer_idx=0, token_offset=0):
        raise RuntimeError("store failed")


def test_hs_lmcache_store_failure_keeps_boundary_and_buffer():
    """#6b: a failed store must not advance the boundary or trim the buffer."""
    runner, _ = _make_lmcache_runner(chunk_size=4, hidden_size=2)
    runner._get_lmcache_adapter().lmcache_engine.hidden_state_store = _RaisingHSStore()

    _drive_step(runner, sched=8, num_computed=0, hs_rows=8)

    assert runner._hs_saved_boundary.get("r1", 0) == 0
    assert sum(t.shape[0] for t in runner._hs_pending_buffer["r1"]["hidden"]) == 8


def test_hs_lmcache_regression_resets_state():
    """#3: num_computed regressing below the saved boundary (preempt) resets state."""
    runner, hs_store = _make_lmcache_runner(chunk_size=4, hidden_size=2)
    _drive_step(runner, sched=8, num_computed=0, hs_rows=8)
    assert runner._hs_saved_boundary["r1"] == 8
    hs_store.calls.clear()

    # Request comes back with fewer computed tokens than we had flushed.
    _drive_positioned_step(runner, sched=4, num_computed=2)

    # Without the reset, new_boundary (4) <= stale 8 would skip flushing forever.
    # The buffer now starts at token 2, so the first storable chunk is [4:8) and
    # nothing is written until it fills -- storing these four rows at offset 0
    # would file tokens 2..5 as 0..3.
    assert runner._hs_saved_boundary["r1"] == 4
    assert hs_store.calls == []

    _drive_positioned_step(runner, sched=2, num_computed=6)

    assert len(hs_store.calls) == 1
    assert hs_store.calls[0].token_offset == 4
    assert _stored_positions(hs_store.calls[0]) == [4, 5, 6, 7]


class _FakeRetrieveStore:
    """retrieve_hidden_states returns a configurable per-layer prefix (or None)."""

    def __init__(self, rows_by_layer):
        self._rows_by_layer = rows_by_layer

    def retrieve_hidden_states(self, token_ids, *, layer_idx=0):
        rows = self._rows_by_layer.get(layer_idx)
        if rows is None:
            return None
        return torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2)


def _make_restore_runner(rows_by_layer, num_computed=8, chunk_size=4, mm_keys=()):
    runner = object.__new__(LMCacheHiddenStateMixin)
    runner._has_lmcache = True
    runner._lmcache_hs_mm_keys = mm_keys
    runner._hs_mm_features = {}
    engine = SimpleNamespace(
        hidden_state_store=_FakeRetrieveStore(rows_by_layer),
        config=SimpleNamespace(chunk_size=chunk_size),
    )
    runner._get_lmcache_adapter = lambda: SimpleNamespace(lmcache_engine=engine)
    runner.input_batch = SimpleNamespace(
        req_id_to_index={"r1": 0},
        num_prompt_tokens=torch.tensor([64]),
        token_ids_cpu=torch.arange(64).reshape(1, 64),
    )
    sched_out = SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(req_id="r1", num_computed_tokens=num_computed)])
    return runner, sched_out


def test_restore_incomplete_hs_sets_no_payload():
    """#2: a missing required layer must not produce a partial payload."""
    pytest.importorskip("lmcache", reason="the restore path keys through lmcache helpers")

    runner, sched_out = _make_restore_runner(rows_by_layer={-1: None})  # "hidden" (idx -1) misses

    LMCacheHiddenStateMixin._maybe_restore_hs_from_lmcache(runner, sched_out)

    assert runner._restored_mm.get("r1") is None


def test_restore_full_hs_sets_payload():
    """#2: all required layers present -> payload is set."""
    pytest.importorskip("lmcache", reason="the restore path keys through lmcache helpers")

    runner, sched_out = _make_restore_runner(rows_by_layer={-1: 8})  # "hidden" (idx -1) full-length

    LMCacheHiddenStateMixin._maybe_restore_hs_from_lmcache(runner, sched_out)

    assert "hidden" in runner._restored_mm["r1"]
    assert runner._restored_mm["r1"]["hidden"].shape[0] == 8


def test_restore_remaps_mm_layers_to_flattened_payload_keys():
    """Qwen3-Omni-style captures must land under the flattened payload keys.

    Only models exposing talker_config.accept_hidden_layer populate
    _lmcache_hs_mm_keys, so this path never runs on Qwen2.5-Omni.
    """
    pytest.importorskip("lmcache", reason="the restore path keys through lmcache helpers")

    runner, sched_out = _make_restore_runner(
        rows_by_layer={-1: 8, 0: 8, 24: 8},  # "hidden" (-1) plus captures 0 and 24
        mm_keys=("0", "24"),
    )

    LMCacheHiddenStateMixin._maybe_restore_hs_from_lmcache(runner, sched_out)

    assert set(runner._restored_mm["r1"]) == {"hidden", "hidden_states.layer_0", "hidden_states.layer_24"}


def test_pooler_payload_casts_restored_prefix_to_batch_dtype(monkeypatch):
    """#5: LMCache returns CPU float32; prepending onto bf16 activations must not raise."""
    import vllm_omni.worker.gpu_ar_model_runner as ar_mod

    monkeypatch.setattr(
        ar_mod, "build_omni_mm_payload", lambda **kwargs: {"mm0": torch.ones(2, 2, dtype=torch.bfloat16)}
    )

    runner = object.__new__(GPUARModelRunner)
    restored_mm = {
        "r1": {
            "hidden": torch.zeros(3, 2, dtype=torch.float32),
            "mm0": torch.zeros(3, 2, dtype=torch.float32),
        }
    }

    payload = GPUARModelRunner._build_omni_pooler_payload(
        runner,
        restored_mm=restored_mm,
        rid="r1",
        idx=0,
        start=0,
        end=2,
        hidden_states_cpu=None,
        req_hidden_states_cpu={"r1": torch.ones(2, 2, dtype=torch.bfloat16)},
        combined_hidden_states=None,
        combined_multimodal_outputs=None,
        mm_cpu=None,
        audio_sparse_output=False,
        sparse_mm_index={},
        hidden_seq_len=2,
        scheduled_seq_len=2,
    )

    # Both the "hidden" tap and the mm layer are prepended in the batch's dtype.
    assert payload["hidden"].dtype == torch.bfloat16
    assert payload["hidden"].shape[0] == 5
    assert payload["mm0"].dtype == torch.bfloat16
    assert payload["mm0"].shape[0] == 5


def test_hs_lmcache_store_slices_per_request_in_multi_request_batch():
    """#11: each request must buffer only its own rows of the flattened batch HS."""
    runner = object.__new__(LMCacheHiddenStateMixin)
    runner._has_lmcache = True
    runner._lmcache_hs_mm_keys = ()
    runner._hs_pending_buffer = {}
    runner._hs_saved_boundary = {}
    runner._hs_buffer_origin = {}
    runner._hs_mm_features = {}

    hs_store = _FakeHSStore()
    engine = SimpleNamespace(hidden_state_store=hs_store, config=SimpleNamespace(chunk_size=4))
    runner._get_lmcache_adapter = lambda: SimpleNamespace(lmcache_engine=engine)

    # r0 occupies rows 0..3, r1 rows 4..7 of the flattened batch.
    runner.input_batch = SimpleNamespace(
        req_ids=["r0", "r1"],
        req_id_to_index={"r0": 0, "r1": 1},
        num_computed_tokens_cpu=torch.tensor([0, 0]),
        token_ids_cpu=torch.arange(128).reshape(2, 64),
    )
    runner.query_start_loc = SimpleNamespace(cpu=torch.tensor([0, 4]))

    # Row i is filled with value i so provenance is checkable.
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(8, 1).repeat(1, 2)
    sched_out = SimpleNamespace(num_scheduled_tokens={"r0": 4, "r1": 4})

    LMCacheHiddenStateMixin._maybe_store_hs_to_lmcache(
        runner, hidden_states, None, num_tokens_unpadded=8, scheduler_output=sched_out
    )

    assert len(hs_store.calls) == 2
    by_offset = {tuple(c.token_ids[:1]): c.hidden_states for c in hs_store.calls}
    # r0 stored rows 0..3, r1 stored rows 4..7 -- not the batch's first 4 twice.
    assert torch.equal(hs_store.calls[0].hidden_states[:, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(hs_store.calls[1].hidden_states[:, 0], torch.tensor([4.0, 5.0, 6.0, 7.0]))
    assert len(by_offset) == 2


# ---------------------------------------------------------------------------
# LMCache HS restore: one consumer, and a short store is a failure
# ---------------------------------------------------------------------------


class _StubHSStore:
    """Returns a fixed-length prefix for every layer."""

    def __init__(self, rows: int, hidden_size: int = 2):
        self.rows = rows
        self.hidden_size = hidden_size

    def retrieve_hidden_states(self, token_ids, *, layer_idx=0):
        if self.rows <= 0:
            return None
        return torch.ones(self.rows, self.hidden_size)


def _make_stub_restore_runner(*, stored_rows, num_computed=8, prompt_tokens=16):
    runner = object.__new__(LMCacheHiddenStateMixin)
    runner._has_lmcache = True
    runner._lmcache_hs_mm_keys = ()
    runner._hs_mm_features = {}
    engine = SimpleNamespace(
        hidden_state_store=_StubHSStore(stored_rows),
        config=SimpleNamespace(chunk_size=4),
    )
    runner._get_lmcache_adapter = lambda: SimpleNamespace(lmcache_engine=engine)
    runner.input_batch = SimpleNamespace(
        req_id_to_index={"r1": 0},
        num_prompt_tokens=[prompt_tokens],
        token_ids_cpu=torch.arange(64).reshape(1, 64),
    )
    sched_out = SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(req_id="r1", num_computed_tokens=num_computed)])
    return runner, sched_out


def test_hs_restore_stashes_the_prefix_for_the_pooler_payload():
    pytest.importorskip("lmcache", reason="the restore path keys through lmcache helpers")

    runner, sched_out = _make_stub_restore_runner(stored_rows=8)

    LMCacheHiddenStateMixin._maybe_restore_hs_from_lmcache(runner, sched_out)

    assert set(runner._restored_mm) == {"r1"}
    assert runner._restored_mm["r1"]["hidden"].shape[0] == 8


def test_hs_restore_skips_everything_when_the_store_is_short():
    pytest.importorskip("lmcache", reason="the restore path keys through lmcache helpers")

    runner, sched_out = _make_stub_restore_runner(stored_rows=5)

    LMCacheHiddenStateMixin._maybe_restore_hs_from_lmcache(runner, sched_out)

    assert runner._restored_mm == {}


def test_hs_store_short_write_does_not_advance_the_boundary():
    """A full pool stops the store early and returns normally, not by raising."""
    runner, hs_store = _make_lmcache_runner(chunk_size=4, hidden_size=2)
    hs_store.store_hidden_states = lambda *a, **k: 0

    _drive_step(runner, sched=8, num_computed=0, hs_rows=8)

    assert runner._hs_saved_boundary.get("r1", 0) == 0
    assert sum(t.shape[0] for t in runner._hs_pending_buffer["r1"]["hidden"]) == 8


def test_hs_store_after_a_restore_still_flushes():
    """A restored prefix was never buffered, so the boundary must start past it."""
    runner, hs_store = _make_lmcache_runner(chunk_size=4, hidden_size=2)

    # Request arrives with 4 tokens already restored and 4 scheduled.
    _drive_step(runner, sched=4, num_computed=4, hs_rows=4)

    assert runner._hs_saved_boundary["r1"] == 8
    assert len(hs_store.calls) == 1
    call = hs_store.calls[0]
    assert call.token_offset == 4
    assert call.hidden_states.shape == (4, 2)


def test_keyed_token_ids_hash_multimodal_spans():
    """Hidden states share the KV chunk keys, which have mm spans hashed.

    Asserts that the span is rewritten and that the rewrite is stable, not what
    the values are. The keys only have to match whatever LMCache derives for the
    KV they ride with, and we get that by calling the same helper it does --
    LMCache has changed both the helper's name and its per-position output
    before, and pinning either breaks this test without telling us anything
    about our own contract.
    """
    pytest.importorskip("lmcache", reason="lmcache not installed")

    runner = object.__new__(LMCacheHiddenStateMixin)
    runner.input_batch = SimpleNamespace(token_ids_cpu=torch.arange(16).reshape(1, 16))
    placeholder = SimpleNamespace(offset=4, length=3)
    runner._hs_mm_features = {"r1": (["ab" * 16], [placeholder])}

    keyed = runner._keyed_token_ids(0, "r1", 10)
    again = runner._keyed_token_ids(0, "r1", 10)

    # Only the span is touched, and the same request keys the same way twice.
    assert keyed[:4] == [0, 1, 2, 3]
    assert keyed[7:] == [7, 8, 9]
    assert keyed[4:7] != [4, 5, 6]
    assert keyed == again


def test_keyed_token_ids_pass_through_without_multimodal_spans():
    runner = object.__new__(LMCacheHiddenStateMixin)
    runner.input_batch = SimpleNamespace(token_ids_cpu=torch.arange(16).reshape(1, 16))
    runner._hs_mm_features = {}

    assert runner._keyed_token_ids(0, "r1", 5) == [0, 1, 2, 3, 4]


def test_take_restored_mm_removes_only_the_requested_ids():
    """The payload builder can run late, so it consumes a snapshot, not live state."""
    runner = object.__new__(LMCacheHiddenStateMixin)
    runner._restored_mm = {
        "r1": {"hidden": torch.ones(2, 2)},
        "r2": {"hidden": torch.ones(3, 2)},
    }

    taken = runner._take_restored_mm(["r1"])

    assert set(taken) == {"r1"}
    assert set(runner._restored_mm) == {"r2"}


def test_take_restored_mm_is_empty_without_lmcache():
    runner = object.__new__(LMCacheHiddenStateMixin)
    assert runner._take_restored_mm(["r1"]) == {}
