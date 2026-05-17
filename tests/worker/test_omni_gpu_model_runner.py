from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class DummyBuffer:
    """A minimal buffer wrapper that exposes the `.gpu` attribute."""

    def __init__(self, t: torch.Tensor):
        self.gpu = t


class DummyInputBatch:
    """A minimal input batch that only provides `req_ids`."""

    def __init__(self, req_ids):
        self.req_ids = req_ids


class DummyReqState:
    """A minimal request state container."""

    pass


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
    ):
        self.calls.append(
            {
                "do_sample": do_sample,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "generator": generator,
            }
        )
        codes = torch.zeros((req_embeds.shape[0], 1), dtype=torch.int64)
        return req_embeds, codes


@contextmanager
def _noop_forward_context(*args, **kwargs):
    """A no-op context manager to replace vLLM forward context in CPU tests."""
    yield


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
    # Patch the module-level `set_forward_context` symbol used inside
    # OmniGPUModelRunner._talker_mtp_forward.
    import vllm_omni.worker.gpu_model_runner as mod  # Must be the same module that defines OmniGPUModelRunner

    monkeypatch.setattr(mod, "set_forward_context", _noop_forward_context)

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

    monkeypatch.setattr(mod, "set_forward_context", _noop_forward_context)

    runner = _make_runner(req_ids=("r1",), hidden_size=4)

    inputs_embeds = torch.randn((2, 4))
    before = inputs_embeds.clone()

    OmniGPUModelRunner._talker_mtp_forward(runner, [], inputs_embeds)

    # Ensure no changes were made
    assert torch.allclose(inputs_embeds, before)


def test_talker_mtp_forward_ignores_default_sampling_seed_without_request_marker(monkeypatch):
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod, "set_forward_context", _noop_forward_context)

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

    monkeypatch.setattr(mod, "set_forward_context", _noop_forward_context)

    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner.requests["r1"].sampling_params = SimpleNamespace(
        seed=42,
        extra_args={"qwen3_tts_request_seed": 42},
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
            "do_sample": False,
            "temperature": 0.2,
            "top_k": 9,
            "top_p": 0.55,
            "generator": runner.talker_mtp.calls[0]["generator"],
        }
    ]
    assert runner.talker_mtp.calls[0]["generator"] is not None


def test_update_intermediate_buffer_writes_to_buffer_and_setattr(monkeypatch):
    """Validate that _update_intermediate_buffer writes to model_intermediate_buffer
    (forward path) and mirrors to additional_information_cpu setattr (backward compat)."""
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod, "set_forward_context", _noop_forward_context)

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


def _make_full_payload_accumulation_runner(
    model_arch="Qwen3OmniMoeForConditionalGeneration",
    model_stage="talker",
    async_chunk=False,
):
    runner = object.__new__(OmniConnectorModelRunnerMixin)
    runner.model_config = SimpleNamespace(
        model_arch=model_arch,
        model_stage=model_stage,
        async_chunk=async_chunk,
    )
    runner._custom_process_func = object()
    runner._pending_full_payload_send = {}
    runner._stage_id = 1
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
    assert not _make_full_payload_accumulation_runner(model_stage="code2wav")._should_accumulate_full_payload_output()
    assert not _make_full_payload_accumulation_runner(
        model_stage="talker", async_chunk=True
    )._should_accumulate_full_payload_output()
    assert not _make_full_payload_accumulation_runner(
        model_arch="Qwen3TTSForConditionalGeneration"
    )._should_accumulate_full_payload_output()
    assert not _make_full_payload_accumulation_runner(
        model_arch="Qwen2_5OmniForConditionalGeneration"
    )._should_accumulate_full_payload_output()


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
