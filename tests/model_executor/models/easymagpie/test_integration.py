# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig

from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters import detect_tts_model_type, resolve_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.model_executor.models.easymagpie.scheduler import EasyMagpieCodecTransferAdapter
from vllm_omni.model_executor.models.registry import OmniModelRegistry

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("codec", [False, True])
def test_optimized_precision_matches_reference_stage_policy(monkeypatch, codec):
    from vllm_omni.model_executor.models.easymagpie import runner

    monkeypatch.setattr(runner.os, "environ", {})
    # Restore process-wide backend policy after this CPU-only startup check.
    settings = (
        (torch.backends.cuda.matmul, "allow_tf32"),
        (torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"),
        (torch.backends.cuda.matmul, "allow_fp16_accumulation"),
        (torch.backends.cudnn, "allow_tf32"),
        (torch.backends.cudnn, "deterministic"),
        (torch.backends.cudnn, "benchmark"),
    )
    previous = [(owner, name, getattr(owner, name)) for owner, name in settings]
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(easymagpie_optimized=True)))
    try:
        runner.configure_inference(config, codec=codec)
        assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction is not codec
        assert not torch.backends.cuda.matmul.allow_fp16_accumulation
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
    finally:
        for owner, name, value in previous:
            setattr(owner, name, value)


def test_native_registration():
    assert "EasyMagpieTTSForConditionalGeneration" in OmniModelRegistry.get_supported_archs()
    assert "EasyMagpieCodecForConditionalGeneration" in OmniModelRegistry.get_supported_archs()
    assert AutoConfig.for_model("easymagpie_codec").num_stacked_codebooks == 16
    assert detect_tts_model_type("easymagpie", None) == "easymagpie"
    assert resolve_adapter("easymagpie") is not None


@pytest.mark.parametrize("profile", ["easymagpie.yaml", "easymagpie_a4500.yaml", "easymagpie_h100_c256.yaml"])
def test_deploy_resolves_native_stages(profile):
    root = Path(__file__).resolve().parents[4]
    pipeline = resolve_pipeline_config("easymagpie")
    deploy = load_deploy_config(root / "vllm_omni" / "deploy" / profile)
    legacy = merge_pipeline_deploy(pipeline, deploy)
    config = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy)
    talker, codec = config.stage_configs
    assert talker.model_config.use_v2_model_runner is False
    assert codec.model_config.use_v2_model_runner is False
    assert talker.model_config.default_sampling_params["stop_token_ids"] == [1]
    assert legacy[1].yaml_engine_args["retains_state_across_chunks"] is True
    assert pipeline.stages[1].model_subdir == "codec_native"
    if "c256" in profile:
        assert talker.scheduler_config.max_num_seqs == codec.scheduler_config.max_num_seqs == 256
        assert legacy[1].yaml_engine_args["dtype"] == "float16"


@pytest.mark.asyncio
async def test_speech_prompt_includes_causal_text_prefix():
    adapter = resolve_adapter("easymagpie")(SpeechServingContext(server=SimpleNamespace()))
    adapter._prompt_len = lambda speaker, context: 23
    adapter._text_stream_metadata = lambda: (99, 6)
    adapter._model_tokenizer = lambda: SimpleNamespace(encode=lambda text, add_special_tokens: [7, 8])
    request = OpenAICreateSpeechRequest(input="Hello world", voice="Aria", extra_params={"context_text": "[EN]"})
    prepared = await adapter.build(request, [], False)
    info = prepared.prompt["additional_information"]
    assert info["text_tokens"] == [7, 8, 99]
    assert info["prefill_text_tokens"] == [7, 8, 99]
    assert info["speaker_id"] == "Aria"
    assert len(prepared.prompt["prompt_token_ids"]) == 29


@pytest.mark.parametrize("voice", ["../speaker", "/tmp/speaker", ".", "..", " "])
def test_voice_must_be_a_bundled_name(voice):
    adapter = resolve_adapter("easymagpie")(SpeechServingContext(server=SimpleNamespace()))
    assert (
        adapter.validate(OpenAICreateSpeechRequest(input="Hello", voice=voice))
        == "voice must be a bundled speaker name"
    )


def test_request_generation_limit_only_changes_talker():
    from vllm import SamplingParams

    adapter = resolve_adapter("easymagpie")(SpeechServingContext(server=SimpleNamespace()))
    defaults = [SamplingParams(max_tokens=2048), SamplingParams(max_tokens=65536)]
    request = OpenAICreateSpeechRequest(input="Hello", voice="Aria", max_new_tokens=8)
    actual = adapter.apply_sampling_overrides(defaults, request)
    assert actual[0].max_tokens == 8
    assert actual[1].max_tokens == defaults[1].max_tokens == 65536
    assert defaults[0].max_tokens == 2048


def test_codec_chunks_preserve_state_position(monkeypatch, mocker):
    adapter = object.__new__(EasyMagpieCodecTransferAdapter)
    request = SimpleNamespace(
        num_computed_tokens=8,
        prompt_token_ids=[0] * 8,
        _all_token_ids=[0] * 8,
        update_block_hashes=mocker.Mock(),
    )

    def replace_prompt(self, request, result, **kwargs):
        request.num_computed_tokens = 0
        request.prompt_token_ids = [0]
        return True

    monkeypatch.setattr(OmniChunkTransferAdapter, "_commit_received_chunk", replace_prompt)
    assert adapter._commit_received_chunk(request, ({"codes": {"audio": torch.zeros(6, 16)}}, 0))
    assert request.num_computed_tokens == 8
    assert request.num_prompt_tokens == 14
    assert len(request._all_token_ids) == 14
    request.num_computed_tokens = 14
    assert adapter._commit_received_chunk(request, ({"codes": {"audio": torch.empty(0)}}, 0))
    assert request.num_prompt_tokens == request.num_computed_tokens == 14


def test_codec_does_not_poll_while_output_is_pending(monkeypatch, mocker):
    adapter = object.__new__(EasyMagpieCodecTransferAdapter)
    adapter.codec_lock = threading.RLock()
    poll = mocker.Mock(return_value=True)
    monkeypatch.setattr(OmniChunkTransferAdapter, "_poll_single_request", poll)
    entry = SimpleNamespace(request=SimpleNamespace(num_in_flight_tokens=8))
    assert adapter._poll_single_request(entry) is False
    poll.assert_not_called()
    entry.request.num_in_flight_tokens = 0
    assert adapter._poll_single_request(entry) is True
    poll.assert_called_once_with(entry)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_codec_binds_padded_state_pages(dtype):
    from vllm_omni.model_executor.models.easymagpie.codec.packed import CODEC_STATE_ELEMENTS, CodecStateLayer

    layer = object.__new__(CodecStateLayer)
    torch.nn.Module.__init__(layer)
    layer.dtype = dtype
    state_bytes = CODEC_STATE_ELEMENTS * torch.empty((), dtype=dtype).element_size()
    raw = torch.zeros((4, 1, 1, state_bytes + 128), dtype=torch.int8)
    layer.bind_kv_cache(raw)
    (state,) = layer.kv_cache
    assert state.shape == (4, CODEC_STATE_ELEMENTS)
    assert state.dtype == dtype
    state[2].fill_(1)
    assert torch.equal(raw[2, 0, 0, :state_bytes].view(dtype), torch.ones(CODEC_STATE_ELEMENTS, dtype=dtype))
    assert not raw[:, :, :, state_bytes:].count_nonzero()


def test_codec_warmup_metadata_uses_native_mamba_contract():
    from vllm_omni.model_executor.models.easymagpie.codec.cuda_graph import uniform_codec_metadata

    metadata = uniform_codec_metadata(4, 6, torch.device("cpu"))
    assert metadata.num_prefills == 4
    assert metadata.num_prefill_tokens == 24
    assert metadata.query_start_loc_p.tolist() == [0, 6, 12, 18, 24]
    assert not metadata.has_initial_states_p.any()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_moe_preserves_routed_scale_and_shared_expert(dtype, monkeypatch):
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.model_executor.models.nemotron_h import NemotronHMoEDecoderLayer

    from vllm_omni.model_executor.models.easymagpie.easymagpie import EasyMagpieMoEDecoderLayer

    module = SimpleNamespace(routed_scaling_factor=2.5)
    shared, routed = MoERunner._maybe_apply_routed_scale_to_output(
        module, torch.ones(4, dtype=dtype), torch.full((4,), 3, dtype=dtype)
    )
    monkeypatch.setattr(NemotronHMoEDecoderLayer, "forward", lambda self, x, residual: (shared + routed, residual))
    layer = object.__new__(EasyMagpieMoEDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.mixer = SimpleNamespace(routed_scaling_factor=2.5, shared_experts=object())
    actual, _ = layer.forward(None, None)
    torch.testing.assert_close(actual, torch.full((4,), 8.5, dtype=dtype))


def test_sender_snapshot_freezes_only_consumed_fields():
    from vllm.v1.request import RequestStatus

    from vllm_omni.model_executor.models.easymagpie.scheduler import EasyMagpieTalkerTransferAdapter

    request = SimpleNamespace(
        request_id="internal",
        external_req_id="external",
        status=RequestStatus.RUNNING,
        additional_information={"text_prefill_num": 5, "reset_codec_on_segment": False, "text_tokens": [1, 2]},
    )
    snapshot = EasyMagpieTalkerTransferAdapter._snapshot_processor_request(request)
    request.status = RequestStatus.FINISHED_STOPPED
    request.additional_information["text_prefill_num"] = 0
    assert snapshot.status == RequestStatus.RUNNING
    assert snapshot.additional_information == {"text_prefill_num": 5, "reset_codec_on_segment": False}


def test_sender_cleanup_releases_cancelled_codec_history(monkeypatch):
    from vllm_omni.model_executor.models.easymagpie.scheduler import EasyMagpieTalkerTransferAdapter

    adapter = object.__new__(EasyMagpieTalkerTransferAdapter)
    states = (
        "_emp_emitted_frames",
        "_emp_emitted_chunks",
        "_emp_seen_frames",
        "_emp_request_speech_delay",
        "_emp_frame_buffer_base",
        "_emp_frame_buffer",
    )
    for name in states:
        setattr(adapter, name, {"cancelled": [1], "live": [2]})
    monkeypatch.setattr(OmniChunkTransferAdapter, "_clear_sender_state_locked", lambda self, request_id: None)
    adapter._clear_sender_state_locked("cancelled")
    assert all(getattr(adapter, name) == {"live": [2]} for name in states)


@pytest.mark.parametrize("failed", [False, True])
def test_prefill_phase_is_bounded_and_restores_scheduler_eligibility(monkeypatch, failed):
    from vllm.v1.request import RequestStatus

    from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
    from vllm_omni.model_executor.models.easymagpie.scheduler import EasyMagpieARAsyncScheduler

    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    request = SimpleNamespace(
        status=RequestStatus.RUNNING,
        next_decode_eligible_step=0,
        num_tokens_with_spec=6,
        num_output_placeholders=0,
        num_computed_tokens=5,
        is_prefill_chunk=False,
    )
    scheduler._separate_prefills = True
    scheduler._decode_steps_remaining = 0
    scheduler._throttle_prefills = False
    scheduler._deferred_decode_steps = []
    scheduler._prefill_step = False
    scheduler.parallel_config = SimpleNamespace(pipeline_parallel_size=1)
    scheduler.num_spec_tokens = 0
    scheduler.current_step = 0
    scheduler.running, scheduler.waiting = [request], [object()]
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.max_num_running_reqs = 32
    phases = []
    monkeypatch.setattr(OmniARAsyncScheduler, "_should_defer_waiting_admission", lambda self: False)

    def schedule(self, throttle_prefills=False):
        defer = self._should_defer_waiting_admission()
        phases.append("decode" if defer else "prefill")
        assert (request.next_decode_eligible_step > self.current_step + 1) == (not defer)
        if failed:
            raise RuntimeError("schedule failed")
        self.current_step += 1
        return SimpleNamespace(num_scheduled_tokens={"request": 1})

    monkeypatch.setattr(OmniARAsyncScheduler, "schedule", schedule)
    if failed:
        with pytest.raises(RuntimeError, match="schedule failed"):
            scheduler.schedule()
    else:
        for _ in range(6):
            scheduler.schedule()
        assert phases == ["prefill", "decode", "decode", "prefill", "decode", "decode"]
    assert request.next_decode_eligible_step == 0
    assert not scheduler._deferred_decode_steps


def test_chunk_ramp_emits_each_frame_once():
    from collections import defaultdict

    from vllm_omni.model_executor.stage_input_processors.easymagpie import talker2code2wav_async_chunk

    manager = SimpleNamespace(
        config=SimpleNamespace(hf_config=SimpleNamespace(streaming_speech_delay=5)),
        connector=SimpleNamespace(config={"extra": {"codec_chunk_frames": 8, "codec_startup_chunk_frames": [2, 6]}}),
        code_prompt_token_ids=defaultdict(list),
    )
    request = SimpleNamespace(
        external_req_id="one",
        additional_information={"text_prefill_num": 5},
        is_finished=lambda: False,
    )
    payloads = []
    for frame in range(1, 20):
        payload = talker2code2wav_async_chunk(
            manager,
            {"codes": {"audio": torch.full((1, 16), frame)}},
            request,
            is_finished=False,
        )
        if payload is not None:
            payloads.append(payload.codes.audio)
    request.is_finished = lambda: True
    final = talker2code2wav_async_chunk(manager, None, request, is_finished=True)
    assert [len(chunk) for chunk in payloads] == [2, 6, 8]
    assert final.meta.finished.item()
    payloads.append(final.codes.audio)
    actual = torch.cat(payloads)
    actual = actual[actual[:, 0] >= 0]
    assert actual[:, 0].tolist() == list(range(1, 20))
    assert not manager._emp_frame_buffer
