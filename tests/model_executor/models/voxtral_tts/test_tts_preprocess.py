# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import functools
from typing import Any

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@functools.lru_cache(maxsize=1)
def _voxtral_tts_model_cls():
    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules

    bootstrap_vllm_layer_custom_op_modules()
    import vllm.model_executor.models.utils  # noqa: F401

    from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts import (
        VoxtralTTSForConditionalGeneration,
    )

    return VoxtralTTSForConditionalGeneration


class FakeAudioGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_multimodal_calls = []

    def embed_multimodal(self, **kwargs):
        self.embed_multimodal_calls.append(kwargs)
        audio_tokens = kwargs["audio_tokens"]
        if audio_tokens.ndim == 3:
            count = audio_tokens.shape[1]
            values = audio_tokens[0, :, 0].to(torch.float32)
            return [values.unsqueeze(-1).expand(count, 4)]
        return [torch.full((1, 4), 7.0)]

    def compute_mm_logits(self, hidden_states, cfg_alpha):
        self.cfg_alpha = cfg_alpha
        batch_size = hidden_states.shape[0] if hidden_states.ndim > 1 else 1
        fake_eos = torch.arange(batch_size, device=hidden_states.device, dtype=hidden_states.dtype)
        frames = [torch.full((1, 1, 4), float(index + 1), device=hidden_states.device) for index in range(batch_size)]
        return fake_eos, {"codes": {"audio": frames}}


def _make_voxtral_tts_model():
    model_cls = _voxtral_tts_model_cls()
    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.model_stage = "audio_generation"
    model.model = FakeAudioGeneration()
    model._audio_token_id = 42
    model._cudagraph_acoustic_transformer = None
    return model


def test_tts_preprocess_consumes_nested_codes_audio_feedback():
    model = _make_voxtral_tts_model()
    input_ids = torch.tensor([model._audio_token_id])
    input_embeds = torch.zeros((1, 4))
    audio_tokens = torch.tensor([[1, 2, 3, 4]])

    _, output_embeds, _ = model.tts_preprocess(
        input_ids=input_ids,
        input_embeds=input_embeds,
        codes={"audio": audio_tokens},
    )

    assert len(model.model.embed_multimodal_calls) == 1
    torch.testing.assert_close(model.model.embed_multimodal_calls[0]["audio_tokens"], audio_tokens)
    torch.testing.assert_close(output_embeds, torch.full((1, 4), 7.0))


def test_tts_preprocess_replays_each_generated_audio_frame_by_token_position():
    model = _make_voxtral_tts_model()
    input_ids = torch.tensor([5, model._audio_token_id, model._audio_token_id])
    input_embeds = torch.zeros((3, 4))
    history = [torch.tensor([10, 11, 12, 13]), torch.tensor([20, 21, 22, 23])]

    _, output_embeds, _ = model.tts_preprocess(
        input_ids=input_ids,
        input_embeds=input_embeds,
        codes={"audio": history[-1].reshape(1, 1, -1), "audio_history": history},
        _omni_prompt_len=1,
        _omni_num_computed_tokens=0,
    )

    assert len(model.model.embed_multimodal_calls) == 1
    torch.testing.assert_close(
        model.model.embed_multimodal_calls[0]["audio_tokens"],
        torch.tensor([[[10, 11, 12, 13], [20, 21, 22, 23]]]),
    )
    torch.testing.assert_close(output_embeds[0], torch.zeros(4))
    torch.testing.assert_close(output_embeds[1], torch.full((4,), 10.0))
    torch.testing.assert_close(output_embeds[2], torch.full((4,), 20.0))


def test_tts_preprocess_replay_chunk_uses_absolute_computed_token_offset():
    model = _make_voxtral_tts_model()
    input_ids = torch.tensor([model._audio_token_id])
    input_embeds = torch.zeros((1, 4))
    history = [torch.tensor([10, 11, 12, 13]), torch.tensor([20, 21, 22, 23])]

    _, output_embeds, _ = model.tts_preprocess(
        input_ids=input_ids,
        input_embeds=input_embeds,
        codes={"audio": history[-1].reshape(1, 1, -1), "audio_history": history},
        _omni_prompt_len=3,
        _omni_num_computed_tokens=4,
    )

    torch.testing.assert_close(
        model.model.embed_multimodal_calls[0]["audio_tokens"],
        torch.tensor([[[20, 21, 22, 23]]]),
    )
    torch.testing.assert_close(output_embeds, torch.full((1, 4), 20.0))


def test_tts_preprocess_steady_state_decode_avoids_cuda_sync_queries(monkeypatch):
    model = _make_voxtral_tts_model()
    input_ids = torch.tensor([model._audio_token_id])
    input_embeds = torch.zeros((1, 4))
    audio_tokens = torch.tensor([[[5, 6, 7, 8]]])

    def unexpected_sync_query(*args, **kwargs):
        raise AssertionError("steady-state decode must not inspect CUDA mask values on the CPU")

    monkeypatch.setattr(torch.Tensor, "any", unexpected_sync_query)
    monkeypatch.setattr(torch.Tensor, "tolist", unexpected_sync_query)

    _, output_embeds, _ = model.tts_preprocess(
        input_ids=input_ids,
        input_embeds=input_embeds,
        codes={"audio": audio_tokens, "audio_history": [torch.tensor([5, 6, 7, 8])]},
        _omni_prompt_len=4,
        _omni_num_computed_tokens=4,
    )

    torch.testing.assert_close(output_embeds, torch.full((1, 4), 5.0))


def test_tts_preprocess_uses_latest_feedback_without_mrv2_positions():
    model = _make_voxtral_tts_model()
    input_ids = torch.tensor([model._audio_token_id])
    input_embeds = torch.zeros((1, 4))
    audio_tokens = torch.tensor([[5, 6, 7, 8]])

    _, output_embeds, _ = model.tts_preprocess(
        input_ids=input_ids,
        input_embeds=input_embeds,
        codes={"audio": audio_tokens, "audio_history": [torch.tensor([1, 2, 3, 4])]},
    )

    torch.testing.assert_close(model.model.embed_multimodal_calls[0]["audio_tokens"], audio_tokens)
    torch.testing.assert_close(output_embeds, torch.full((1, 4), 7.0))


def test_tts_preprocess_keeps_legacy_top_level_audio_feedback():
    model = _make_voxtral_tts_model()
    input_ids = torch.tensor([model._audio_token_id])
    input_embeds = torch.zeros((1, 4))
    audio_tokens = torch.tensor([[5, 6, 7, 8]])

    _, output_embeds, _ = model.tts_preprocess(
        input_ids=input_ids,
        input_embeds=input_embeds,
        audio=audio_tokens,
    )

    assert len(model.model.embed_multimodal_calls) == 1
    torch.testing.assert_close(model.model.embed_multimodal_calls[0]["audio_tokens"], audio_tokens)
    torch.testing.assert_close(output_embeds, torch.full((1, 4), 7.0))


def test_v1_output_hook_persists_feedback_from_preprocess_state_delta():
    from types import SimpleNamespace

    from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

    model = _make_voxtral_tts_model()
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = model
    runner.model_intermediate_buffer = {}
    runner.requests = {"request-a": SimpleNamespace()}

    _, _, update = model.tts_preprocess(torch.tensor([5]), torch.zeros((1, 4)))
    assert update == {"codes": {"audio_history": []}}
    runner._update_intermediate_buffer("request-a", update)
    request_info = runner.model_intermediate_buffer["request-a"]

    model.make_omni_output(
        torch.zeros((1, 4)),
        logits_index=torch.tensor([0]),
        runtime_additional_information=[request_info],
    )

    torch.testing.assert_close(
        runner.model_intermediate_buffer["request-a"]["codes"]["audio_history"][0],
        torch.ones((1, 1, 4)),
    )
    _, output_embeds, _ = model.tts_preprocess(
        torch.tensor([model._audio_token_id]),
        torch.zeros((1, 4)),
        _omni_prompt_len=4,
        _omni_num_computed_tokens=4,
        **runner.model_intermediate_buffer["request-a"],
    )
    torch.testing.assert_close(output_embeds, torch.ones((1, 4)))


def test_make_omni_output_keeps_feedback_request_local():
    model = _make_voxtral_tts_model()
    model._cudagraph_acoustic_transformer = None
    request_infos: list[dict[str, Any]] = [{"request_id": "req-a"}, {"request_id": "req-b"}]
    hidden_states = torch.zeros((2, 4))

    output = model.make_omni_output(
        hidden_states,
        logits_index=torch.tensor([0, 1]),
        runtime_additional_information=request_infos,
    )

    torch.testing.assert_close(request_infos[0]["codes"]["audio"], torch.full((1, 1, 4), 1.0))
    torch.testing.assert_close(request_infos[1]["codes"]["audio"], torch.full((1, 1, 4), 2.0))
    torch.testing.assert_close(output.text_hidden_states[:, 0], torch.tensor([0.0, 1.0]))
    assert not hasattr(model, "post_process_idx")


def test_make_omni_output_mrv2_uses_request_slots_and_sampling_cfg():
    model = _make_voxtral_tts_model()
    request_infos = [
        {"request_id": "req-a", "sampling_params": type("Sampling", (), {"extra_args": {"cfg_alpha": 1.1}})()},
        {"request_id": "req-b", "sampling_params": type("Sampling", (), {"extra_args": {"cfg_alpha": 1.7}})()},
    ]
    input_batch = type(
        "InputBatch",
        (),
        {
            "num_reqs": 2,
            "logits_indices": torch.tensor([2, 0]),
            "num_computed_prefill_tokens_np": [20, 20],
            "num_scheduled_tokens": [1, 1],
            "prefill_len_np": [20, 20],
        },
    )()
    hidden_states = torch.zeros((3, 4))

    output = model.make_omni_output_mrv2(
        hidden_states,
        input_batch=input_batch,
        req_states=None,
        model_intermediate_buffer=request_infos,
    )

    torch.testing.assert_close(model.model.cfg_alpha, torch.tensor([1.1, 1.7]))
    torch.testing.assert_close(request_infos[0]["codes"]["audio"], torch.full((1, 1, 4), 1.0))
    torch.testing.assert_close(request_infos[1]["codes"]["audio"], torch.full((1, 1, 4), 2.0))
    torch.testing.assert_close(request_infos[0]["codes"]["audio_history"][0], torch.full((1, 1, 4), 1.0))
    torch.testing.assert_close(request_infos[1]["codes"]["audio_history"][0], torch.full((1, 1, 4), 2.0))
    torch.testing.assert_close(output.text_hidden_states[[2, 0], 0], torch.tensor([0.0, 1.0]))


def test_make_omni_output_mrv2_filters_intermediate_prefill_frames_by_request():
    model = _make_voxtral_tts_model()
    request_infos: list[dict[str, Any]] = [
        {"request_id": "req-prefill"},
        {"request_id": "req-decode"},
    ]
    input_batch = type(
        "InputBatch",
        (),
        {
            "num_reqs": 2,
            "logits_indices": torch.tensor([2, 0]),
            "num_computed_prefill_tokens_np": [5, 20],
            "num_scheduled_tokens": [5, 1],
            "prefill_len_np": [20, 20],
        },
    )()

    output = model.make_omni_output_mrv2(
        torch.zeros((3, 4)),
        input_batch=input_batch,
        req_states=None,
        model_intermediate_buffer=request_infos,
    )

    assert "codes" not in request_infos[0]
    torch.testing.assert_close(request_infos[1]["codes"]["audio_history"][0], torch.full((1, 1, 4), 2.0))
    audio_frames = output.multimodal_outputs["codes"]["audio"]
    assert audio_frames[0] is None
    torch.testing.assert_close(audio_frames[1], torch.full((1, 1, 4), 2.0))


def test_voxtral_pipeline_declares_both_mrv2_stages_native():
    from vllm_omni.model_executor.models.voxtral_tts.pipeline import VOXTRAL_TTS_PIPELINE

    assert len(VOXTRAL_TTS_PIPELINE.stages) == 2
    assert all(stage.supports_native_mrv2_data_plane for stage in VOXTRAL_TTS_PIPELINE.stages)
