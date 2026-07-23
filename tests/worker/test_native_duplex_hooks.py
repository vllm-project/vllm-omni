from __future__ import annotations

import inspect
import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _minicpmo_duplex_policy_case(
    state: SimpleNamespace,
    payload: dict[str, object],
):
    from vllm_omni.experimental.fullduplex.model_executor import DuplexSamplingRow
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    session_key = ("sid-policy", 1)
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model._minicpmo45_native_duplex_token_ids_cache = {
        "listen_token_id": 7,
        "tts_bos_token_id": 8,
        "turn_eos_token_id": 9,
    }
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={session_key: state})
    row = DuplexSamplingRow(
        row_idx=0,
        request_id="req-policy",
        session_id=session_key[0],
        incarnation=session_key[1],
        seq=3,
        payload=payload,
        max_tokens=20,
    )
    return model, row


def test_minicpmo_model_hook_owns_duplex_sampling_rows_and_force_listen():
    from vllm_omni.experimental.fullduplex.model_executor import DuplexSamplingRow
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    listen_id = 7
    state = SimpleNamespace(
        current_turn_ended=True,
        last_terminator_token=None,
        pending_terminator_token=None,
    )
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model._minicpmo45_native_duplex_token_ids_cache = {
        "listen_token_id": listen_id,
        "tts_bos_token_id": 8,
        "turn_eos_token_id": 9,
    }
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={("sid-hook", 2): state})
    logits = torch.zeros((1, 16), dtype=torch.float32)
    row = DuplexSamplingRow(
        row_idx=0,
        request_id="req-hook",
        session_id="sid-hook",
        incarnation=2,
        seq=3,
        payload={"force_listen": True, "is_speech": True},
        max_tokens=20,
    )

    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))

    assert model._minicpmo45_active_duplex_rows == [0]
    assert model._minicpmo45_duplex_row_sessions == {0: ("sid-hook", 2)}
    assert model._minicpmo45_duplex_row_payloads == {0: row.payload}
    assert model._minicpmo45_duplex_row_max_tokens == {0: 20}
    assert logits[0, listen_id].item() == 0.0
    assert torch.isneginf(logits[0, :listen_id]).all()
    assert torch.isneginf(logits[0, listen_id + 1 :]).all()


def test_minicpmo_model_hook_turn_end_latch_forces_silence_to_listen():
    state = SimpleNamespace(
        current_turn_ended=True,
        last_terminator_token=9,
        pending_terminator_token=None,
    )
    model, row = _minicpmo_duplex_policy_case(state, {"is_speech": False})
    logits = torch.zeros((1, 16), dtype=torch.float32)
    logits[0, 10] = 20.0

    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))

    assert logits[0, 7].item() == 0.0
    assert torch.isneginf(logits[0, :7]).all()
    assert torch.isneginf(logits[0, 8:]).all()


def test_minicpmo_model_hook_pending_speech_after_turn_eos_allows_silence_sampling():
    state = SimpleNamespace(
        current_turn_ended=True,
        last_terminator_token=9,
        pending_terminator_token=None,
        pending_speech_context=True,
    )
    model, row = _minicpmo_duplex_policy_case(state, {"is_speech": False})
    logits = torch.zeros((1, 16), dtype=torch.float32)
    logits[0, 7] = 30.0
    logits[0, 10] = 20.0
    original_logits = logits.clone()

    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))

    assert torch.equal(logits, original_logits)
    assert state.pending_speech_context is True


def test_minicpmo_model_hook_speech_row_does_not_create_pending_context():
    state = SimpleNamespace(
        current_turn_ended=True,
        last_terminator_token=9,
        pending_terminator_token=None,
        pending_speech_context=False,
    )
    model, row = _minicpmo_duplex_policy_case(state, {"is_speech": True})
    logits = torch.zeros((1, 16), dtype=torch.float32)
    logits[0, 7] = 30.0
    logits[0, 8] = -2.0
    logits[0, 10] = 20.0
    original_logits = logits.clone()

    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))

    assert state.current_turn_ended is True
    assert state.last_terminator_token is None
    assert state.pending_speech_context is False
    assert torch.equal(logits, original_logits)


def test_minicpmo_model_hook_old_response_output_does_not_clear_pending_speech_context():
    state = SimpleNamespace(
        current_turn_ended=False,
        last_terminator_token=None,
        pending_terminator_token=None,
        pending_speech_context=True,
    )
    model, row = _minicpmo_duplex_policy_case(state, {"is_speech": True})
    model._minicpmo45_duplex_row_sessions = {0: (row.session_id, row.incarnation)}
    model._minicpmo45_duplex_row_payloads = {0: row.payload}

    model._record_minicpmo45_duplex_terminator(
        0,
        10,
        {"listen_token_id": 7, "chunk_eos_token_id": -1, "chunk_tts_eos_token_id": -1, "turn_eos_token_id": 9},
    )

    assert state.pending_speech_context is True
    assert state.current_turn_ended is False


def test_minicpmo_model_hook_new_response_output_clears_pending_speech_context():
    state = SimpleNamespace(
        current_turn_ended=True,
        last_terminator_token=7,
        pending_terminator_token=7,
        pending_speech_context=True,
    )
    model, row = _minicpmo_duplex_policy_case(state, {"is_speech": False})
    model._minicpmo45_duplex_row_sessions = {0: (row.session_id, row.incarnation)}
    model._minicpmo45_duplex_row_payloads = {0: row.payload}

    model._record_minicpmo45_duplex_terminator(
        0,
        8,
        {"listen_token_id": 7, "chunk_eos_token_id": -1, "chunk_tts_eos_token_id": -1, "turn_eos_token_id": 9},
    )

    assert state.pending_speech_context is False
    assert state.current_turn_ended is False


def test_minicpmo_model_hook_empty_speak_envelope_preserves_pending_speech_context():
    state = SimpleNamespace(
        current_turn_ended=True,
        last_terminator_token=9,
        pending_terminator_token=None,
        pending_speech_context=True,
        pending_speech_response_open=False,
    )
    model, row = _minicpmo_duplex_policy_case(state, {"is_speech": False})
    model._minicpmo45_duplex_row_sessions = {0: (row.session_id, row.incarnation)}
    model._minicpmo45_duplex_row_payloads = {0: row.payload}
    token_ids = {
        "listen_token_id": 7,
        "tts_bos_token_id": 8,
        "chunk_eos_token_id": -1,
        "chunk_tts_eos_token_id": -1,
        "turn_eos_token_id": 9,
    }

    model._record_minicpmo45_duplex_terminator(0, 8, token_ids)
    model._record_minicpmo45_duplex_terminator(0, 9, token_ids)

    assert state.current_turn_ended is True
    assert state.pending_speech_context is True
    assert state.pending_speech_response_open is False

    logits = torch.zeros((1, 16), dtype=torch.float32)
    logits[0, 7] = 30.0
    logits[0, 10] = 20.0
    original_logits = logits.clone()

    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))

    assert torch.equal(logits, original_logits)


def test_minicpmo_model_hook_second_new_response_step_is_not_forced_to_listen():
    state = SimpleNamespace(
        current_turn_ended=True,
        last_terminator_token=9,
        pending_terminator_token=None,
        pending_speech_context=True,
    )
    model, row = _minicpmo_duplex_policy_case(state, {"is_speech": False})
    logits = torch.zeros((1, 16), dtype=torch.float32)
    logits[0, 7] = 30.0
    logits[0, 10] = 20.0
    original_logits = logits.clone()

    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))
    model._minicpmo45_duplex_row_sessions = {0: (row.session_id, row.incarnation)}
    model._minicpmo45_duplex_row_payloads = {0: row.payload}
    model._record_minicpmo45_duplex_terminator(
        0,
        8,
        {"listen_token_id": 7, "chunk_eos_token_id": -1, "chunk_tts_eos_token_id": -1, "turn_eos_token_id": 9},
    )
    next_logits = torch.zeros((1, 16), dtype=torch.float32)
    next_logits[0, 7] = 30.0
    next_logits[0, 10] = 20.0
    next_original_logits = next_logits.clone()

    model.prepare_duplex_sampling(next_logits, SimpleNamespace(), (row,))

    assert torch.equal(logits, original_logits)
    assert torch.equal(next_logits, next_original_logits)
    assert state.pending_speech_context is False
    assert state.current_turn_ended is False


def test_minicpmo_model_hook_same_speech_row_second_step_does_not_rearm_pending_context():
    state = SimpleNamespace(
        current_turn_ended=True,
        last_terminator_token=9,
        pending_terminator_token=None,
        pending_speech_context=True,
    )
    model, row = _minicpmo_duplex_policy_case(state, {"is_speech": True})
    logits = torch.zeros((1, 16), dtype=torch.float32)
    logits[0, 7] = 30.0
    logits[0, 10] = 20.0

    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))
    model._minicpmo45_duplex_row_sessions = {0: (row.session_id, row.incarnation)}
    model._minicpmo45_duplex_row_payloads = {0: row.payload}
    model._record_minicpmo45_duplex_terminator(
        0,
        10,
        {"listen_token_id": 7, "chunk_eos_token_id": -1, "chunk_tts_eos_token_id": -1, "turn_eos_token_id": 9},
    )
    next_logits = torch.zeros((1, 16), dtype=torch.float32)
    next_logits[0, 7] = 30.0
    next_logits[0, 8] = -2.0
    next_logits[0, 10] = 20.0

    model.prepare_duplex_sampling(next_logits, SimpleNamespace(), (row,))

    assert state.pending_speech_context is False
    assert state.current_turn_ended is False
    assert torch.isneginf(next_logits[0, 7])
    assert next_logits[0, 8].item() == 30.0


def test_minicpmo_model_hook_mid_turn_speech_redirects_listen_to_tts_bos():
    state = SimpleNamespace(
        current_turn_ended=False,
        last_terminator_token=None,
        pending_terminator_token=None,
    )
    model, row = _minicpmo_duplex_policy_case(state, {"is_speech": True})
    logits = torch.zeros((1, 16), dtype=torch.float32)
    logits[0, 7] = 30.0
    logits[0, 8] = -2.0
    logits[0, 10] = 20.0

    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))

    assert torch.isneginf(logits[0, 7])
    assert logits[0, 8].item() == 30.0
    assert logits[0, 10].item() == 20.0


def test_minicpmo_model_hook_ignores_serving_new_user_turn_marker():
    state = SimpleNamespace(
        current_turn_ended=False,
        last_terminator_token=8,
        pending_terminator_token=8,
    )
    model, row = _minicpmo_duplex_policy_case(
        state,
        {"is_speech": True, "new_user_turn": True, "force_speak": True},
    )
    logits = torch.zeros((1, 16), dtype=torch.float32)
    logits[0, 7] = 5.0
    logits[0, 10] = 20.0
    model.prepare_duplex_sampling(logits, SimpleNamespace(), (row,))

    assert state.current_turn_ended is False
    assert state.last_terminator_token is None
    assert torch.isneginf(logits[0, 7])
    assert logits[0, 8].item() == 5.0


def test_generic_ar_runner_builds_typed_duplex_sampling_rows():
    from vllm_omni.experimental.fullduplex.model_executor import DuplexSamplingHelper
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    runner = GPUARModelRunner.__new__(GPUARModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["req-duplex", "req-plain"])
    runner.model_intermediate_buffer = {
        "req-duplex": {
            "duplex": {
                "data_plane": True,
                "session_id": "sid-runner-hook",
                "incarnation": 4,
                "seq": 4,
                "payload": {"is_speech": True},
            }
        }
    }
    runner.requests = {
        "req-duplex": SimpleNamespace(
            sampling_params=SimpleNamespace(max_tokens=32),
        )
    }
    helper = DuplexSamplingHelper()
    helper.active_request_ids = {"req-duplex"}

    rows = helper.rows(runner)

    assert len(rows) == 1
    assert rows[0].row_idx == 0
    assert rows[0].request_id == "req-duplex"
    assert rows[0].session_id == "sid-runner-hook"
    assert rows[0].incarnation == 4
    assert rows[0].seq == 4
    assert rows[0].payload == {"is_speech": True}
    assert rows[0].max_tokens == 32


def test_generic_ar_runner_skips_duplex_rows_without_model_hook():
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    runner = GPUARModelRunner.__new__(GPUARModelRunner)
    runner.input_batch = SimpleNamespace(
        sampling_metadata=object(),
        update_async_output_token_ids=lambda: None,
    )
    runner.model = SimpleNamespace(sample=lambda *_args, **_kwargs: None, prefer_model_sampler=False)
    runner.sampler = lambda **_kwargs: "standard-sampler"
    runner._duplex_sampling_helper = SimpleNamespace(
        active_request_ids={"req-duplex"},
        rows=lambda *_args: pytest.fail("duplex rows built without a model hook"),
    )

    assert runner._sample(torch.zeros((1, 4)), spec_decode_metadata=None) == "standard-sampler"


def test_plain_model_hook_resolution_does_not_allocate_duplex_tracking():
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    runner = GPUARModelRunner.__new__(GPUARModelRunner)
    runner.model = SimpleNamespace()
    runner._duplex_sampling_hook = None
    runner._duplex_sampling_hook_resolved = False

    assert runner._resolve_duplex_sampling_hook() is None
    assert not hasattr(runner, "_duplex_sampling_helper")


def test_minicpmo_non_duplex_sample_skips_duplex_row_scan():
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    calls = []
    runner = GPUARModelRunner.__new__(GPUARModelRunner)
    runner.input_batch = SimpleNamespace(
        sampling_metadata=SimpleNamespace(output_token_ids=[]),
        update_async_output_token_ids=lambda: None,
    )
    runner.model = SimpleNamespace(
        sample=lambda *_args, **_kwargs: "model-sampler",
        prefer_model_sampler=True,
        skips_model_sampler_output_token_history=True,
        prepare_duplex_sampling=lambda *_args: calls.append("prepare"),
    )
    runner.sampler = SimpleNamespace()
    runner._duplex_sampling_helper = SimpleNamespace(
        active_request_ids=set(),
        hook_active=False,
        rows=lambda *_args: pytest.fail("non-duplex MiniCPM scanned request rows"),
    )

    assert runner._sample(torch.zeros((1, 4)), spec_decode_metadata=None) == "model-sampler"
    assert calls == []


def test_minicpmo_duplex_sample_clears_stale_rows_once_without_scanning():
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    calls = []
    runner = GPUARModelRunner.__new__(GPUARModelRunner)
    runner.input_batch = SimpleNamespace(
        sampling_metadata=SimpleNamespace(output_token_ids=[]),
        update_async_output_token_ids=lambda: None,
    )
    runner.model = SimpleNamespace(
        sample=lambda *_args, **_kwargs: "model-sampler",
        prefer_model_sampler=True,
        skips_model_sampler_output_token_history=True,
        prepare_duplex_sampling=lambda _logits, _metadata, rows: calls.append(rows),
    )
    runner.sampler = SimpleNamespace()
    runner._duplex_sampling_helper = SimpleNamespace(
        active_request_ids=set(),
        hook_active=True,
        rows=lambda *_args: pytest.fail("duplex cleanup scanned request rows"),
    )

    assert runner._sample(torch.zeros((1, 4)), spec_decode_metadata=None) == "model-sampler"
    assert runner._sample(torch.zeros((1, 4)), spec_decode_metadata=None) == "model-sampler"
    assert calls == [()]


def test_generic_ar_runner_has_no_minicpmo_sampler_state_or_typeerror_probe():
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    source = inspect.getsource(GPUARModelRunner)

    assert "_minicpmo45_duplex_row" not in source
    assert "_minicpmo45_native_duplex_token_ids" not in source
    assert 'if "duplex_rows" not in str(exc)' not in source


def test_minicpmo_model_cleans_incarnation_state_when_request_finishes():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    request_id = "duplex-sid-cleanup-i3-e0-stage0"
    session_key = ("sid-cleanup", 3)
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model = SimpleNamespace()
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={session_key: object()})
    model._minicpmo45_duplex_request_sessions = {request_id: session_key}
    model._minicpmo45_force_listen_applied_segments = {
        (request_id, 1),
        ("duplex-sid-other-e0-stage0", 2),
    }

    model.on_requests_finished({request_id})

    assert model._minicpmo45_duplex_data_plane_helper.sessions == {}
    assert model._minicpmo45_duplex_request_sessions == {}
    assert model._minicpmo45_force_listen_applied_segments == {
        ("duplex-sid-other-e0-stage0", 2),
    }


def test_minicpmo_stage0_routes_duplex_metadata_per_batched_request():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )
    from vllm_omni.utils.mm_outputs import to_payload_element

    class _Thinker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("device_anchor", torch.zeros(1))

        def forward(self, *, input_ids, **kwargs):
            del kwargs
            return torch.arange(input_ids.numel() * 4, dtype=torch.float32).reshape(input_ids.numel(), 4)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage = "llm"
    model.thinker = _Thinker()
    output = model.forward(
        input_ids=torch.tensor([1, 2, 3, 4]),
        positions=torch.arange(4),
        runtime_additional_information=[
            {
                "global_request_id": ["req-a"],
                "duplex": {
                    "duplex_prompt_token_ids": [101, 102],
                    "special_token_ids": {"listen_token_id": 701},
                },
            },
            {
                "global_request_id": ["req-b"],
                "duplex": {
                    "duplex_prompt_token_ids": [201, 202, 203],
                    "special_token_ids": {"listen_token_id": 702},
                },
            },
        ],
    )

    prompt_rows = output.multimodal_outputs["duplex_prompt_token_ids"]
    listen_rows = output.multimodal_outputs["meta"]["listen_token_id"]
    assert to_payload_element(prompt_rows, 0, 0, 2) == [101, 102]
    assert to_payload_element(prompt_rows, 1, 2, 4) == [201, 202, 203]
    assert int(to_payload_element(listen_rows, 0, 0, 2).reshape(-1)[0]) == 701
    assert int(to_payload_element(listen_rows, 1, 2, 4).reshape(-1)[0]) == 702


def test_minicpmo_stage0_rejects_invalid_resolved_ref_audio():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
    )

    with pytest.raises(ValueError, match="invalid native duplex ref_audio_data"):
        MiniCPMO45Stage0DuplexRuntime._decode_ref_audio_from_session_config(
            {
                "ref_audio_data": "a",
                "ref_audio_format": "pcm_f32le",
            }
        )


def test_minicpmo_tts_native_duplex_exports_segment_text_not_accumulated_condition_text():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Talker:
        _ar_last_chunk_flags = [False]
        _ar_last_emitted_text = "你好，你有什莫想聊的吗？"

        def __call__(self, **kwargs):
            del kwargs
            return None, torch.zeros(8, dtype=torch.float32)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "tts"
    model.config = SimpleNamespace(hidden_size=4)
    model.talker = _Talker()

    output = model.forward(
        input_ids=torch.zeros(1, dtype=torch.long),
        positions=torch.zeros(1, dtype=torch.long),
        runtime_additional_information=[
            {
                "native_duplex": True,
                "llm_output_text": ["你好，你有什莫想聊的吗？你好，你有什莫想聊的吗？"],
                "meta": {
                    "native_duplex_segment_text": "你好，你有什莫想聊的吗？",
                },
            }
        ],
    )

    text_bytes = output.multimodal_outputs["meta.llm_output_text_utf8"].detach().cpu().tolist()
    assert bytes(text_bytes).decode("utf-8") == "你好，你有什莫想聊的吗？"
    assert int(output.multimodal_outputs["meta.audio_text_total_chars"].item()) == len("你好，你有什莫想聊的吗？")


def test_minicpmo_tts_native_duplex_exports_model_turn_end_metadata():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Talker:
        _ar_last_chunk_flags = [True]
        _ar_turn_end_flags = [True]

        def __call__(self, **kwargs):
            del kwargs
            return None, torch.zeros(0, dtype=torch.float32)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "tts"
    model.config = SimpleNamespace(hidden_size=4)
    model.talker = _Talker()

    output = model.forward(
        input_ids=torch.zeros(1, dtype=torch.long),
        positions=torch.zeros(1, dtype=torch.long),
        runtime_additional_information=[
            {
                "native_duplex": True,
                "meta": {"native_duplex_segment_text": ""},
            }
        ],
    )

    assert int(output.multimodal_outputs["meta.turn_end"].item()) == 1


def test_minicpmo_tts_routes_batched_requests_and_terminal_flags_independently():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Talker:
        def __init__(self):
            self.calls = []
            self._ar_last_chunk_flags = [True]
            self._ar_turn_end_flags = [False]
            self._ar_last_emitted_text = ""

        def __call__(self, **kwargs):
            info = kwargs["additional_information"]
            request_id = info["global_request_id"][0]
            self.calls.append(request_id)
            is_last = request_id == "req-b"
            self._ar_last_chunk_flags = [is_last]
            self._ar_turn_end_flags = [is_last]
            self._ar_last_emitted_text = request_id
            value = 1.0 if request_id == "req-a" else 2.0
            return None, torch.full((2,), value, dtype=torch.float32)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "tts"
    model.config = SimpleNamespace(hidden_size=4)
    model.talker = _Talker()

    output = model.forward(
        input_ids=torch.zeros(3, dtype=torch.long),
        positions=torch.zeros(3, dtype=torch.long),
        runtime_additional_information=[
            {
                "global_request_id": ["req-a"],
                "native_duplex": True,
                "duplex": {"epoch": 3, "turn_id": 7},
                "meta": {"native_duplex_segment_text": "first"},
            },
            {
                "global_request_id": ["req-idle"],
                "native_duplex": True,
                "duplex": {"epoch": 3, "turn_id": 9},
                "meta": {"native_duplex_segment_text": "idle"},
            },
            {
                "global_request_id": ["req-b"],
                "native_duplex": True,
                "duplex": {"epoch": 4, "turn_id": 8},
                "meta": {"native_duplex_segment_text": "second"},
            },
        ],
        request_token_spans=[(0, 2), (2, 2), (2, 3)],
    )

    assert model.talker.calls == ["req-a", "req-b"]
    assert model.talker._ar_last_chunk_flags == [False, False, True]
    assert model.talker._ar_turn_end_flags == [False, False, True]
    assert output.multimodal_outputs["meta.req_id"] == ["req-a", "req-b"]
    assert output.multimodal_outputs["meta.sparse_audio"] == ["1"]
    assert [waveform.tolist() for waveform in output.multimodal_outputs["model_outputs"]] == [
        [1.0, 1.0],
        [2.0, 2.0],
    ]
    assert [int(value.item()) for value in output.multimodal_outputs["meta.tts_is_last_chunk"]] == [0, 1]
    assert [int(value.item()) for value in output.multimodal_outputs["meta.turn_end"]] == [0, 1]
    assert [int(value.item()) for value in output.multimodal_outputs["meta.duplex_epoch"]] == [3, 4]
    assert [int(value.item()) for value in output.multimodal_outputs["meta.duplex_turn_id"]] == [7, 8]


@pytest.mark.parametrize(
    ("runtime_info", "request_token_spans", "expected_rows", "expected_ids"),
    [
        (
            [
                {
                    "global_request_id": ["req-a"],
                    "native_duplex": True,
                    "duplex": {"epoch": 3, "turn_id": 7},
                    "meta": {"native_duplex_segment_text": "first"},
                }
            ],
            [(0, 3)],
            [3],
            ["req-a"],
        ),
        (
            [
                {
                    "global_request_id": ["req-a"],
                    "native_duplex": True,
                    "duplex": {"epoch": 3, "turn_id": 7},
                    "meta": {"native_duplex_segment_text": "first"},
                },
                {
                    "global_request_id": ["req-b"],
                    "native_duplex": True,
                    "duplex": {"epoch": 4, "turn_id": 8},
                    "meta": {"native_duplex_segment_text": "second"},
                },
            ],
            [(0, 2), (2, 3)],
            [2, 1],
            ["req-a", "req-b"],
        ),
    ],
    ids=["single-request", "batched-requests"],
)
def test_minicpmo_tts_ignores_cuda_graph_padding_after_request_spans(
    runtime_info,
    request_token_spans,
    expected_rows,
    expected_ids,
):
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Talker:
        def __init__(self):
            self.input_rows = []
            self._ar_last_chunk_flags = [True]
            self._ar_turn_end_flags = [False]
            self._ar_last_emitted_text = ""

        def __call__(self, **kwargs):
            self.input_rows.append(int(kwargs["input_ids"].shape[0]))
            return None, torch.ones(2, dtype=torch.float32)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "tts"
    model.config = SimpleNamespace(hidden_size=4)
    model.talker = _Talker()

    output = model.forward(
        input_ids=torch.zeros(4, dtype=torch.long),
        positions=torch.zeros(4, dtype=torch.long),
        runtime_additional_information=runtime_info,
        request_token_spans=request_token_spans,
    )

    assert model.talker.input_rows == expected_rows
    assert output.text_hidden_states.shape == (4, 4)
    assert output.multimodal_outputs["meta.req_id"] == expected_ids


def test_minicpmo_stage0_special_token_ids_are_tokenizer_derived():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
    )

    class _Tokenizer:
        unk_token_id = 0
        ids = {
            "<unit>": 101,
            "</unit>": 102,
            "<|listen|>": 103,
            "<|speak|>": 104,
            "<|tts_bos|>": 105,
            "<|tts_eos|>": 106,
            "<|tts_pad|>": 107,
            "<|chunk_eos|>": 108,
            "<|chunk_tts_eos|>": 109,
            "<|turn_eos|>": 110,
        }

        def convert_tokens_to_ids(self, token):
            return self.ids.get(token, self.unk_token_id)

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [self.ids[text]] if text in self.ids else [201, self.ids["<|tts_bos|>"]]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.tokenizer = _Tokenizer()
    runtime._init_token_ids()

    runtime._require_special_token_ids()
    assert runtime.tts_bos_token_id == 105
    assert runtime.stage_padding_token_id() == 102
    assert runtime._special_token_ids()["chunk_tts_eos_token_id"] == 109


def test_minicpmo_stage0_rejects_unknown_special_token_fallbacks():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
    )

    class _Tokenizer:
        unk_token_id = 0
        ids = {
            "<unit>": 101,
            "</unit>": 102,
            "<|listen|>": 103,
            "<|speak|>": 104,
            "<|tts_eos|>": 106,
            "<|tts_pad|>": 107,
            "<|chunk_eos|>": 108,
            "<|chunk_tts_eos|>": 109,
            "<|turn_eos|>": 110,
        }

        def convert_tokens_to_ids(self, token):
            return self.ids.get(token, self.unk_token_id)

        def encode(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return [self.unk_token_id]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.tokenizer = _Tokenizer()
    runtime._init_token_ids()

    with pytest.raises(ValueError, match=r"<\|tts_bos\|>"):
        runtime._require_special_token_ids()


def test_minicpmo_stage0_data_plane_prefill_matches_official_unit_format():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, _data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
    runtime.thinker = runtime.stage_model
    runtime.tokenizer = SimpleNamespace(
        unk_token_id=0,
        convert_tokens_to_ids=lambda token: {
            "<unit>": 1,
            "</unit>": 2,
            "<|listen|>": 3,
            "<|speak|>": 4,
            "<|tts_bos|>": 5,
            "<|tts_eos|>": 6,
            "<|tts_pad|>": 7,
            "<|chunk_eos|>": 8,
            "<|chunk_tts_eos|>": 9,
            "<|turn_eos|>": 10,
            "<|audio|>": 11,
        }.get(token, 0),
        encode=lambda text, add_special_tokens=False: [201, 5],
    )
    runtime.processor = SimpleNamespace(get_streaming_chunk_size=lambda: 4)
    runtime.device = "cpu"
    runtime._init_token_ids()
    state = _MiniCPMO45Stage0SessionState(session_id="sid-data-plane-prefill")

    # Official duplex format: each unit is <unit> + audio embeddings with no
    # per-chunk assistant header or <|tts_bos|> boundary. Decoding starts right
    # after the audio so the first sampled token is the listen/speak decision.
    result = runtime._stage_prefill_embeddings_only(state, np.zeros(4, dtype=np.float32), seq=1)

    assert result["success"] is True
    assert result["input_token_ids"] == [1, 11]
    assert result["prompt_suffix_len"] == 0

    # Subsequent units must close the previous unit with </unit> first,
    # mirroring the official finalize_unit() feed.
    result = runtime._stage_prefill_embeddings_only(state, np.zeros(4, dtype=np.float32), seq=2)

    assert result["success"] is True
    assert result["input_token_ids"] == [2, 1, 11]
    assert result["prompt_suffix_len"] == 0


def test_minicpmo_stage0_speech_append_sets_pending_context_once():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, _data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
    runtime.thinker = runtime.stage_model
    runtime.tokenizer = SimpleNamespace(
        unk_token_id=0,
        convert_tokens_to_ids=lambda token: {
            "<unit>": 1,
            "</unit>": 2,
            "<|listen|>": 3,
            "<|speak|>": 4,
            "<|tts_bos|>": 5,
            "<|tts_eos|>": 6,
            "<|tts_pad|>": 7,
            "<|chunk_eos|>": 8,
            "<|chunk_tts_eos|>": 9,
            "<|turn_eos|>": 10,
            "<|audio|>": 11,
        }.get(token, 0),
        encode=lambda text, add_special_tokens=False: [],
    )
    runtime.processor = SimpleNamespace(get_streaming_chunk_size=lambda: 4)
    runtime.device = "cpu"
    runtime._init_token_ids()
    state = _MiniCPMO45Stage0SessionState(session_id="sid-speech-append")

    result = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        epoch=0,
        seq=1,
        is_speech=True,
    )

    assert result["success"] is True
    assert state.pending_speech_context is True
    assert state.pending_speech_append_identity == (0, 1)

    state.pending_speech_context = False
    cached = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        epoch=0,
        seq=1,
        is_speech=True,
    )

    assert cached["success"] is True
    assert state.pending_speech_context is False
    assert state.pending_speech_append_identity == (0, 1)

    next_seq = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        epoch=0,
        seq=2,
        is_speech=True,
    )

    assert next_seq["success"] is True
    assert state.pending_speech_context is True
    assert state.pending_speech_append_identity == (0, 2)

    state.pending_speech_context = False
    next_epoch = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        epoch=1,
        seq=1,
        is_speech=True,
    )

    assert next_epoch["success"] is True
    assert state.pending_speech_context is True
    assert state.pending_speech_append_identity == (1, 1)

    state.pending_speech_context = False
    silence = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        epoch=1,
        seq=2,
        is_speech=False,
    )

    assert silence["success"] is True
    assert state.pending_speech_context is False
    assert state.pending_speech_append_identity == (1, 1)


def test_minicpmo_stage0_failed_append_does_not_set_pending_speech_context():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, _data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
    runtime.thinker = runtime.stage_model
    runtime.tokenizer = SimpleNamespace(
        unk_token_id=0,
        convert_tokens_to_ids=lambda token: {
            "<unit>": 1,
            "</unit>": 2,
            "<|listen|>": 3,
            "<|speak|>": 4,
            "<|tts_bos|>": 5,
            "<|tts_eos|>": 6,
            "<|tts_pad|>": 7,
            "<|chunk_eos|>": 8,
            "<|chunk_tts_eos|>": 9,
            "<|turn_eos|>": 10,
            "<|audio|>": 11,
        }.get(token, 0),
        encode=lambda text, add_special_tokens=False: [],
    )
    runtime.processor = SimpleNamespace(get_streaming_chunk_size=lambda: 4)
    runtime.device = "cpu"
    runtime._init_token_ids()
    state = _MiniCPMO45Stage0SessionState(session_id="sid-failed-speech")

    result = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(0, dtype=np.float32),
        epoch=0,
        seq=1,
        is_speech=True,
    )

    assert result["success"] is False
    assert state.pending_speech_context is False
    assert state.pending_speech_append_identity is None


def test_minicpmo_stage0_streaming_processor_is_isolated_per_session():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _Mel:
        def __init__(self):
            self.counter = 0

    class _Processor:
        def __init__(self):
            self._streaming_mel_processor = _Mel()

        def set_streaming_mode(self, **_kwargs):
            return None

        def reset_streaming(self):
            self._streaming_mel_processor.counter = 0

        def process_audio_streaming(self, _audio, **_kwargs):
            self._streaming_mel_processor.counter += 1
            return self._streaming_mel_processor.counter

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.processor = _Processor()
    runtime.stage_model = SimpleNamespace()
    runtime.thinker = runtime.stage_model
    a = _MiniCPMO45Stage0SessionState(session_id="a")
    b = _MiniCPMO45Stage0SessionState(session_id="b")
    processor_a = runtime._configure_streaming_processor(a)
    processor_b = runtime._configure_streaming_processor(b)

    observed = [
        runtime._process_streaming_audio([], 0, processor=processor_a),
        runtime._process_streaming_audio([], 0, processor=processor_b),
        runtime._process_streaming_audio([], 1, processor=processor_a),
        runtime._process_streaming_audio([], 1, processor=processor_b),
    ]

    assert observed == [1, 1, 2, 2]
    assert processor_a is not processor_b
    assert runtime.processor._streaming_mel_processor.counter == 0


def test_minicpmo_stage0_data_plane_next_append_reinjects_previous_listen():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
    runtime.thinker = runtime.stage_model
    runtime.tokenizer = SimpleNamespace(
        unk_token_id=0,
        convert_tokens_to_ids=lambda token: {
            "<unit>": 1,
            "</unit>": 2,
            "<|listen|>": 3,
            "<|speak|>": 4,
            "<|tts_bos|>": 5,
            "<|tts_eos|>": 6,
            "<|tts_pad|>": 7,
            "<|chunk_eos|>": 8,
            "<|chunk_tts_eos|>": 9,
            "<|turn_eos|>": 10,
            "<|audio|>": 11,
        }.get(token, 0),
        encode=lambda text, add_special_tokens=False: [],
    )
    runtime.processor = SimpleNamespace(get_streaming_chunk_size=lambda: 4)
    runtime.device = "cpu"
    runtime._init_token_ids()
    state = _MiniCPMO45Stage0SessionState(
        session_id="sid-new-speech-prefill",
        audio_chunk_idx=1,
        pending_terminator_token=3,
        last_terminator_token=3,
        current_turn_ended=True,
    )

    result = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        seq=2,
    )

    assert result["success"] is True
    assert result["input_token_ids"] == [3, 2, 1, 11]
    assert state.pending_terminator_token is None
    assert state.last_terminator_token == 3
    assert state.current_turn_ended is True


def test_minicpmo_stage0_data_plane_turn_eos_closes_previous_unit():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
    runtime.thinker = runtime.stage_model
    runtime.tokenizer = SimpleNamespace(
        unk_token_id=0,
        convert_tokens_to_ids=lambda token: {
            "<unit>": 1,
            "</unit>": 2,
            "<|listen|>": 3,
            "<|speak|>": 4,
            "<|tts_bos|>": 5,
            "<|tts_eos|>": 6,
            "<|tts_pad|>": 7,
            "<|chunk_eos|>": 8,
            "<|chunk_tts_eos|>": 9,
            "<|turn_eos|>": 10,
            "<|audio|>": 11,
        }.get(token, 0),
        encode=lambda text, add_special_tokens=False: [],
    )
    runtime.processor = SimpleNamespace(get_streaming_chunk_size=lambda: 4)
    runtime.device = "cpu"
    runtime._init_token_ids()
    state = _MiniCPMO45Stage0SessionState(
        session_id="sid-new-user-turn-prefill",
        audio_chunk_idx=1,
        pending_terminator_token=10,
        last_terminator_token=10,
        current_turn_ended=True,
    )

    result = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        seq=2,
    )

    assert result["success"] is True
    assert result["input_token_ids"] == [10, 2, 1, 11]
    assert state.pending_terminator_token is None
    assert state.last_terminator_token == 10
    assert state.current_turn_ended is True


def test_minicpmo_stage0_data_plane_model_owned_turn_boundary_preserves_audio_cache():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    stale_cache = object()

    class _StageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 2)
            self.audio_past_key_values = stale_cache

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, data):
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
    runtime.thinker = runtime.stage_model
    runtime.tokenizer = SimpleNamespace(
        unk_token_id=0,
        convert_tokens_to_ids=lambda token: {
            "<unit>": 1,
            "</unit>": 2,
            "<|listen|>": 3,
            "<|speak|>": 4,
            "<|tts_bos|>": 5,
            "<|tts_eos|>": 6,
            "<|tts_pad|>": 7,
            "<|chunk_eos|>": 8,
            "<|chunk_tts_eos|>": 9,
            "<|turn_eos|>": 10,
            "<|audio|>": 11,
        }.get(token, 0),
        encode=lambda text, add_special_tokens=False: [],
    )
    runtime.processor = SimpleNamespace(get_streaming_chunk_size=lambda: 4)
    runtime.device = "cpu"
    runtime._init_token_ids()
    state = _MiniCPMO45Stage0SessionState(
        session_id="sid-new-user-turn-audio-cache",
        audio_chunk_idx=1,
        audio_past_key_values=stale_cache,
        pending_terminator_token=8,
        last_terminator_token=8,
        current_turn_ended=True,
    )

    result = runtime._stage_prefill_embeddings_only(
        state,
        np.zeros(4, dtype=np.float32),
        seq=2,
    )

    assert result["success"] is True
    assert state.audio_past_key_values is stale_cache
    assert runtime.thinker.audio_past_key_values is stale_cache


def test_minicpmo_stage0_data_plane_final_first_chunk_does_not_add_silence_unit():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _StageModel(torch.nn.Module):
        first_chunk_ms = 10
        sample_rate = 1000

        def __init__(self):
            super().__init__()
            self.seen_audio = None
            self.embed = torch.nn.Embedding(256, 2)

        def get_input_embeddings(self):
            return self.embed

        def get_audio_hidden_states(self, data):
            self.seen_audio = np.asarray(data["audio_features"], dtype=np.float32)
            return [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]

    class _MelProcessor:
        sample_rate = 1000

        def get_config(self):
            return {"effective_first_chunk_ms": 10}

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel()
    runtime.thinker = runtime.stage_model
    runtime.tokenizer = SimpleNamespace(
        unk_token_id=0,
        convert_tokens_to_ids=lambda token: {
            "<unit>": 1,
            "</unit>": 2,
            "<|listen|>": 3,
            "<|speak|>": 4,
            "<|tts_bos|>": 5,
            "<|tts_eos|>": 6,
            "<|tts_pad|>": 7,
            "<|chunk_eos|>": 8,
            "<|chunk_tts_eos|>": 9,
            "<|turn_eos|>": 10,
            "<|audio|>": 11,
        }.get(token, 0),
        encode=lambda text, add_special_tokens=False: [],
    )
    runtime.processor = SimpleNamespace(
        _streaming_mel_processor=_MelProcessor(),
        get_streaming_chunk_size=lambda: 10,
    )
    runtime.device = "cpu"
    runtime._init_token_ids()
    state = _MiniCPMO45Stage0SessionState(session_id="sid-first-chunk-padding")

    result = runtime._stage_prefill_embeddings_only(
        state,
        np.arange(8, dtype=np.float32),
        seq=1,
        final=True,
    )

    assert result["success"] is True
    assert result["input_token_ids"] == [1, 11]
    assert runtime.stage_model.seen_audio is not None
    np.testing.assert_allclose(
        runtime.stage_model.seen_audio.reshape(-1),
        np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32),
    )


def test_minicpmo_stage0_runtime_uses_loaded_vllm_embed_tokens_when_get_input_embeddings_is_broken():
    import torch

    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
    )

    class _Embed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(128, 2))
            self.calls = []

        def forward(self, input_ids):
            ids = input_ids.reshape(-1).tolist()
            self.calls.append(ids)
            return torch.tensor([[float(i), 0.0] for i in ids], dtype=torch.float32)

    class _Thinker:
        def __init__(self):
            self.llm = SimpleNamespace(model=SimpleNamespace(embed_tokens=_Embed()))

        def get_input_embeddings(self, input_ids, multimodal_embeddings=None):
            raise AttributeError("'Qwen3ForCausalLM' object has no attribute 'get_input_embeddings'")

    thinker = _Thinker()
    stage_model = SimpleNamespace(model_stage="llm", thinker=thinker, processor=None)
    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = stage_model
    runtime.thinker = thinker
    runtime.device = "cpu"

    embeds = runtime._embed_token(11)

    assert embeds.shape == (1, 2)
    assert thinker.llm.model.embed_tokens.calls == [[11]]


def test_minicpmo_remote_config_patch_handles_nested_and_dict_configs():
    from vllm_omni.experimental.fullduplex.minicpmo45.compat import (
        patch_minicpmo_remote_config,
    )

    nested = SimpleNamespace(base_model_tp_plan=None)
    config = SimpleNamespace(
        base_model_tp_plan=None,
        text_config=nested,
        tts_config={},
    )

    patch_minicpmo_remote_config(config)

    assert config.base_model_tp_plan == {}
    assert nested.base_model_tp_plan == {}
    assert config.tts_config["top_p"] == 0.8
    assert config.tts_config["top_k"] == 100
    assert config.tts_config["temperature"] == 0.8
    assert config.tts_config["repetition_penalty"] == 1.05


def test_minicpmo_stage0_short_audio_buffers_without_context_mutation():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    class _Processor:
        def get_streaming_chunk_size(self):
            return 16000

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = SimpleNamespace()
    runtime.thinker = SimpleNamespace()
    runtime.processor = _Processor()
    runtime._require_special_token_ids = lambda: None
    state = _MiniCPMO45Stage0SessionState(session_id="sid")

    result = runtime._stage_prefill_embeddings_only(state, np.zeros(1600, dtype=np.float32))

    assert result["success"] is False
    assert result["reason"]
    assert len(state.audio_buffer) >= 1600
    assert state.context_embeds == []


def test_minicpmo_stage0_native_sampler_penalizes_repeated_text_token():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    vocab_size = 151723
    repeated = 198
    alternative = 1234
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, repeated] = 20.0
    logits[0, alternative] = 19.5
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[repeated] * 8],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[alternative]]


def test_minicpmo_stage0_native_sampler_penalizes_text_from_prior_chunk():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        _MiniCPMO45Stage0SessionState,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    session_key = ("sid-cross-chunk-repetition", 0)
    state = _MiniCPMO45Stage0SessionState(session_id=session_key[0])
    state.generated_text_tokens = [198] * 8
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={session_key: state})
    model._minicpmo45_duplex_row_sessions = {0: session_key}

    vocab_size = 151723
    repeated = 198
    alternative = 1234
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, repeated] = 20.0
    logits[0, alternative] = 19.5
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[alternative]]


def test_minicpmo_stage0_native_sampler_matches_official_negative_logit_penalty():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        _MiniCPMO45Stage0SessionState,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    session_key = ("sid-negative-logit-repetition", 0)
    state = _MiniCPMO45Stage0SessionState(session_id=session_key[0])
    repeated = 198
    alternative = 1234
    state.generated_text_tokens = [repeated]
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={session_key: state})
    model._minicpmo45_duplex_row_sessions = {0: session_key}

    logits = torch.full((1, 151723), -100.0)
    logits[0, repeated] = -1.0
    logits[0, alternative] = -0.97
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[repeated]]


def test_minicpmo_stage0_records_only_bounded_text_repetition_history():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        _MiniCPMO45Stage0SessionState,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    session_key = ("sid-text-history", 0)
    state = _MiniCPMO45Stage0SessionState(session_id=session_key[0])
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={session_key: state})
    model._minicpmo45_duplex_row_sessions = {0: session_key}
    token_ids = {
        "unit_token_id": 1,
        "unit_end_token_id": 2,
        "listen_token_id": 3,
        "speak_token_id": 4,
        "tts_bos_token_id": 5,
        "tts_eos_token_id": 6,
        "tts_pad_token_id": 7,
        "chunk_eos_token_id": 8,
        "chunk_tts_eos_token_id": 9,
        "turn_eos_token_id": 10,
    }

    for sampled in range(1000, 1520):
        model._record_minicpmo45_duplex_terminator(0, sampled, token_ids)

    expected_history = list(range(1008, 1520))
    assert state.generated_text_tokens == expected_history

    for sampled in token_ids.values():
        model._record_minicpmo45_duplex_terminator(0, sampled, token_ids)

    assert state.generated_text_tokens == expected_history


def test_minicpmo_stage0_native_sampler_does_not_override_model_at_punctuation():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        text = {
            200: "我",
            201: "喜",
            202: "欢",
            203: "。",
        }

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return "".join(self.text.get(int(token_id), "") for token_id in ids)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model.min_new_speak_tokens_before_chunk_boundary = 4
    model.max_new_speak_tokens_per_chunk = 64
    vocab_size = 151723
    alternative = 1234
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, alternative] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706, 200, 201, 202, 203]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[alternative]]


def test_minicpmo_stage0_native_sampler_does_not_cut_before_natural_boundary_minimum():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

        def decode(self, ids, skip_special_tokens=True):
            del ids, skip_special_tokens
            return "。"

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model.min_new_speak_tokens_before_chunk_boundary = 4
    model.max_new_speak_tokens_per_chunk = 64
    vocab_size = 151723
    alternative = 1234
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, alternative] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706, 200, 201, 202]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[alternative]]


def test_minicpmo_stage0_native_sampler_does_not_rewrite_model_punctuation():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        text = {
            108386: "你",
            104256: "好",
            3837: "，",
            100644: "今",
            99172: "天",
            100281: "想",
            27442: "聊",
            99217: "什",
            1773: "。",
            99218: "么",
        }

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return {"。": [1773]}.get(text, [])

        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return "".join(self.text.get(int(token_id), "") for token_id in ids)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model.min_new_speak_tokens_before_chunk_boundary = 4
    model.max_new_speak_tokens_per_chunk = 64
    vocab_size = 151723
    period = 1773
    continuation = 99218
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, period] = 30.0
    logits[0, continuation] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706, 108386, 104256, 3837, 100644, 99172, 100281, 27442, 99217]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[period]]


def test_minicpmo_stage0_native_sampler_preserves_model_chunk_eos_decision():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    logits = torch.full((1, 151723), -100.0)
    logits[0, 151718] = 30.0
    logits[0, 1234] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=True,
        all_random=False,
        temperature=torch.tensor([0.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706, 200, 201]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151718]]


def test_minicpmo_stage0_native_sampler_preserves_early_model_turn_eos_decision():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model.min_new_speak_tokens_before_chunk_boundary = 8
    logits = torch.full((1, 151723), -100.0)
    logits[0, 151717] = 30.0
    logits[0, 1234] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=True,
        all_random=False,
        temperature=torch.tensor([0.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706, 200, 201]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151717]]


def test_minicpmo_stage0_native_sampler_char_cap_does_not_override_special_token():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = [151706, 151717]

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return "已经超过二十八个字符的当前语音分段文本"

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    logits = torch.full((1, 151723), -100.0)
    logits[0, 151717] = 30.0
    sampling_metadata = SimpleNamespace(
        all_greedy=True,
        all_random=False,
        temperature=torch.tensor([0.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706, 200, 201]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151717]]


def test_minicpmo_stage0_native_sampler_preserves_model_chunk_eos():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model.min_new_speak_tokens_before_chunk_boundary = 8
    model.max_new_speak_tokens_per_chunk = 64
    vocab_size = 151723
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, 151718] = 30.0
    logits[0, 1234] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706, 200, 201, 202]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151718]]


def test_minicpmo_stage0_native_sampler_keeps_hard_chunk_cap():
    from vllm_omni.experimental.fullduplex.minicpmo45.policy import (
        MiniCPMO45DuplexPolicy,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

        def decode(self, ids, skip_special_tokens=True):
            del ids, skip_special_tokens
            return "没有自然边界"

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    vocab_size = 151723
    alternative = 1234
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, alternative] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[200] * (MiniCPMO45DuplexPolicy.DEFAULT_MAX_NEW_SPEAK_TOKENS_PER_CHUNK - 1)],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert MiniCPMO45DuplexPolicy.DEFAULT_MAX_NEW_SPEAK_TOKENS_PER_CHUNK == 20
    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151718]]


def test_minicpmo_stage0_native_sampler_cuts_before_request_length_cap():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

        def decode(self, ids, skip_special_tokens=True):
            del ids, skip_special_tokens
            return "没有自然边界"

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model.max_new_speak_tokens_per_chunk = 64
    model._minicpmo45_duplex_row_max_tokens = {0: 20}
    vocab_size = 151723
    alternative = 1234
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, alternative] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706] + [200] * 18],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151718]]


def test_minicpmo_stage0_native_sampler_cuts_on_decoded_text_length():
    from vllm_omni.experimental.fullduplex.minicpmo45.policy import (
        MiniCPMO45DuplexPolicy,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = [151706, 151718, 151717, 151705]

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

        def decode(self, ids, skip_special_tokens=True):
            token_text = {
                200: "一二三四五六七八九十",
                201: "十一十二十三十四十五",
                202: "十六十七十八十九",
            }
            special = set(self.all_special_ids) if skip_special_tokens else set()
            return "".join(token_text.get(int(token_id), "") for token_id in ids if int(token_id) not in special)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    vocab_size = 151723
    candidate = 202
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, candidate] = 20.0
    sampling_metadata = SimpleNamespace(
        all_greedy=True,
        all_random=False,
        temperature=torch.tensor([0.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[151706, 200, 201]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert MiniCPMO45DuplexPolicy.DEFAULT_MAX_SPEAK_CHARS_PER_CHUNK == 28
    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151718]]


def test_minicpmo_stage0_native_sampler_ignores_pending_placeholders():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    vocab_size = 151723
    newline = 198
    alternative = 1234
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, newline] = 20.0
    logits[0, alternative] = 19.5
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[-1, -1, -1]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[newline]]


def test_minicpmo_stage0_native_sampler_converts_mid_turn_listen_to_tts_bos():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    state = SimpleNamespace(current_turn_ended=False)
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model._minicpmo45_duplex_row_sessions = {0: ("sid-native", 0)}
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={("sid-native", 0): state})
    vocab_size = 151723
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, 151705] = 30.0
    sampling_metadata = SimpleNamespace(
        all_greedy=True,
        all_random=False,
        temperature=torch.tensor([0.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151703]]
    assert state.current_turn_ended is False


def test_minicpmo_stage0_native_sampler_forced_listen_yields_floor():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    class _Tokenizer:
        eos_token_id = 151705
        unk_token_id = -1
        bad_token_ids = []
        all_special_ids = []

        def convert_tokens_to_ids(self, token):
            return {
                "<unit>": 151683,
                "</unit>": 151684,
                "<|listen|>": 151705,
                "<|speak|>": 151706,
                "<|tts_bos|>": 151703,
                "<|tts_eos|>": 151704,
                "<|tts_pad|>": 151722,
                "<|chunk_eos|>": 151718,
                "<|chunk_tts_eos|>": 151721,
                "<|turn_eos|>": 151717,
            }.get(token, -1)

    state = SimpleNamespace(current_turn_ended=False)
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.thinker = SimpleNamespace(get_tokenizer=lambda: _Tokenizer())
    model._minicpmo45_duplex_row_sessions = {0: ("sid-native", 0)}
    model._minicpmo45_duplex_row_payloads = {0: {"force_listen": True}}
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={("sid-native", 0): state})
    vocab_size = 151723
    logits = torch.full((1, vocab_size), -100.0)
    logits[0, 151705] = 30.0
    sampling_metadata = SimpleNamespace(
        all_greedy=True,
        all_random=False,
        temperature=torch.tensor([0.0]),
        top_k=torch.tensor([1]),
        top_p=torch.tensor([1.0]),
        generators={},
        prompt_token_ids=torch.tensor([[151683] * 16]),
        output_token_ids=[[]],
    )

    sampled = model.sample(logits, sampling_metadata)

    assert sampled is not None
    assert sampled.sampled_token_ids.tolist() == [[151705]]
    assert state.current_turn_ended is True


def test_minicpmo_stage0_native_sampler_uses_runner_duplex_rows():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    metadata = SimpleNamespace(
        prompt_token_ids=torch.tensor([[1, 2, 3]]),
    )

    rows = model._minicpmo45_native_duplex_prompt_rows(
        metadata,
        unit_id=151683,
        batch_size=1,
        duplex_rows=[0],
    )

    assert rows == [0]


def test_minicpmo_stage0_session_context_includes_resolved_ref_audio():
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.unit_token_id = 151683
    runtime.processor = SimpleNamespace()
    runtime.stage_model = SimpleNamespace()
    runtime.thinker = SimpleNamespace()
    runtime.device = "cpu"
    token_map = {
        "<|im_start|>system\nUse speech.\n<|audio_start|>": [1, 2, 3],
        "<|audio_end|><|im_end|>": [4, 5],
    }
    runtime._stage_runtime_ready = lambda: True
    runtime._require_special_token_ids = lambda: None
    runtime._decode_ref_audio_from_session_config = lambda _config: np.array([0.1, -0.1], dtype=np.float32)
    runtime._encode_text = lambda text: token_map[text]
    runtime._embed_token = lambda token_id: torch.full((1, 2), float(token_id))
    runtime._stage_ref_audio_embeddings = lambda ref_audio, state=None: torch.tensor([[10.0, 11.0], [12.0, 13.0]])

    state = _MiniCPMO45Stage0SessionState(session_id="sid-ref")
    runtime._prepare_session_context(state, {"instructions": "Use speech.", "extra_body": {"ref_audio_data": "x"}})

    assert state.context_token_ids == [1, 2, 3, 151683, 151683, 4, 5]
    assert len(state.context_embeds) == 6


def test_minicpmo_tts_ref_audio_lru_evicts_matching_vocoder_base_cache(tmp_path):
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        MiniCPMO45OmniTTSForConditionalGeneration,
        MiniCPMO45TTSRuntimeConfig,
    )

    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    model._runtime_config = MiniCPMO45TTSRuntimeConfig(ref_audio_file_cache_size=1)
    model._t2w_base_caches = {}
    model._token2wav_state_lock = threading.RLock()
    first_path = model._write_ref_audio_prompt_wav(np.array([0.1, -0.1], dtype=np.float32), 16_000)
    assert first_path is not None
    model._t2w_base_caches[first_path] = (torch.tensor([1]), {"cache": torch.tensor([2])})

    second_path = model._write_ref_audio_prompt_wav(np.array([0.2, -0.2], dtype=np.float32), 16_000)

    assert second_path is not None
    assert second_path != first_path
    assert first_path not in model._t2w_base_caches
    assert not os.path.exists(first_path)


def test_minicpmo_tts_prompt_wav_path_does_not_default_to_model_asset(tmp_path):
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        MiniCPMO45OmniTTSForConditionalGeneration,
    )

    model_dir = tmp_path / "MiniCPM-o-4_5"
    assets_dir = model_dir / "assets"
    assets_dir.mkdir(parents=True)
    (assets_dir / "HT_ref_audio.wav").write_bytes(b"placeholder")

    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    model.vllm_config = SimpleNamespace(model_config=SimpleNamespace(model=str(model_dir)))
    model._write_ref_audio_prompt_wav = lambda ref_audio, ref_audio_sr: None

    assert model._resolve_prompt_wav_path(None, None) == (None, None)


def test_minicpmo_tts_stream_turn_id_prefers_duplex_identity():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        MiniCPMO45OmniTTSForConditionalGeneration,
    )

    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)

    assert (
        model._stream_turn_id(
            {
                "duplex": {"turn_id": 0, "model_turn_id": 4},
                "meta": {"turn_id": 3},
            }
        )
        == 4
    )
    assert (
        model._stream_turn_id(
            {
                "duplex": {"turn_id": 0},
                "meta": {"turn_id": 3},
            }
        )
        == 0
    )
    assert model._stream_turn_id({"meta": {"turn_id": 3}, "turn_id": 2}) == 3
    assert model._stream_turn_id({"turn_id": 2}) == 2


def test_minicpmo_tts_no_ref_audio_initializes_empty_vocoder_cache():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        MiniCPMO45OmniTTSForConditionalGeneration,
        _TalkerTurnState,
    )

    class AudioTokenizer:
        def set_stream_cache(self, prompt_wav_path):
            raise AssertionError(f"unexpected reference-audio cache load: {prompt_wav_path!r}")

    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    model._t2w_base_caches = {}
    model._token2wav_state_lock = threading.RLock()
    model.audio_tokenizer = AudioTokenizer()
    state = _TalkerTurnState(None, None)

    model._begin_turn_vocoder_cache(None, state=state)

    assert state.stream_cache is None
    assert state.hift_cache_dict == {}
    assert state.vocoder_initialized is True
    assert model._t2w_base_caches[""] == (None, {})


def test_minicpmo_tts_native_duplex_merges_and_left_pads_nonfinal_unit_audio():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        _native_duplex_unit_waveform,
    )

    waveform = _native_duplex_unit_waveform(
        [torch.tensor([1.0, 2.0]), torch.tensor([3.0])],
        turn_end=False,
        target_samples=6,
    )

    assert waveform is not None
    assert waveform.tolist() == [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]


def test_minicpmo_tts_native_duplex_does_not_pad_final_unit_audio():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        _native_duplex_unit_waveform,
    )

    waveform = _native_duplex_unit_waveform(
        [torch.tensor([1.0, 2.0]), torch.tensor([3.0])],
        turn_end=True,
        target_samples=6,
    )

    assert waveform is not None
    assert waveform.tolist() == [1.0, 2.0, 3.0]


def test_minicpmo_tts_turn_cleanup_removes_deleted_ref_audio_vocoder_cache(tmp_path):
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        MiniCPMO45OmniTTSForConditionalGeneration,
        _TalkerTurnState,
    )

    prompt_path = tmp_path / "session-ref.wav"
    prompt_path.write_bytes(b"wav")
    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    model._talker_turn_states = {
        "request": _TalkerTurnState(str(prompt_path), str(prompt_path)),
    }
    model._talker_consumed_tokens = {"request": 1}
    model._talker_request_keys = {}
    model._t2w_base_caches = {
        str(prompt_path): (torch.tensor([1]), {"cache": torch.tensor([2])}),
    }
    model._token2wav_state_lock = threading.RLock()
    model.audio_tokenizer = None

    assert model._close_turn_state("request") is True
    assert str(prompt_path) not in model._t2w_base_caches
    assert not prompt_path.exists()
