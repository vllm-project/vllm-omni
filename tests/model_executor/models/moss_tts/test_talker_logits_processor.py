# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for MOSS-TTS talker logits processing."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

pytest.importorskip("vllm")
pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


class _RecordingLogitsProcessor:
    def __init__(self) -> None:
        self.args: tuple[object, ...] | None = None
        self.kwargs: dict[str, object] | None = None

    def __call__(self, *args: object, **kwargs: object) -> torch.Tensor:
        self.args = args
        self.kwargs = kwargs
        return torch.zeros((1, 4))


class _EmbeddingBackbone(nn.Module):
    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 4)


def test_moss_tts_delay_compute_logits_does_not_forward_sampling_metadata() -> None:
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSDelayTalkerForGeneration,
    )

    model = MossTTSDelayTalkerForGeneration.__new__(MossTTSDelayTalkerForGeneration)
    nn.Module.__init__(model)
    model.text_lm_head = object()
    model.logits_processor = _RecordingLogitsProcessor()
    model._batch_state = None
    hidden_states = torch.randn(1, 4)
    sampling_metadata = object()

    logits = model.compute_logits(hidden_states, sampling_metadata=sampling_metadata)

    assert logits.shape == (1, 4)
    assert model.logits_processor.args == (model.text_lm_head, hidden_states)
    assert model.logits_processor.kwargs == {}


def test_moss_tts_local_keeps_fixed_shape_rows_and_routes_stop_logits() -> None:
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSLocalTalkerForGeneration,
    )

    model = MossTTSLocalTalkerForGeneration.__new__(MossTTSLocalTalkerForGeneration)
    nn.Module.__init__(model)
    model.audio_pad_token_id = 1024
    model.audio_assistant_slot_token_id = 2
    model.im_end_token_id = 3
    model.text_vocab_size = 8
    model.n_vq = 4
    model._batch_state = None
    model._batch_state_spans = None
    model._batch_should_continue = None

    valid = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    stopped = torch.full((1, 4), model.audio_pad_token_id, dtype=torch.long)
    infos = [
        {"audio_state": {}, "audio_codes": {"current": valid}},
        {"audio_state": {}, "audio_codes": {"current": stopped}},
    ]
    spans = [(0, 3), (3, 4)]

    output = model.make_omni_output(
        torch.randn(4, 8),
        model_intermediate_buffer=infos,
        request_token_spans=spans,
    )

    audio = output.multimodal_outputs["codes"]["audio"]
    assert len(audio) == 2
    torch.testing.assert_close(audio[0], valid)
    # Keeping this row avoids CUDA boolean-index shape discovery; the
    # talker2codec raw async-chunk processor drops all-pad rows on CPU.
    torch.testing.assert_close(audio[1], stopped)
    assert model._batch_should_continue.tolist() == [True, False]

    logits = model.compute_logits(torch.randn(4, 8))

    assert logits is not None
    assert logits.argmax(dim=-1).tolist() == [2, 2, 2, 3]


def test_moss_tts_local_single_token_prefill_does_not_advance_audio() -> None:
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSLocalTalkerForGeneration,
    )

    model = MossTTSLocalTalkerForGeneration.__new__(MossTTSLocalTalkerForGeneration)
    nn.Module.__init__(model)
    model.model = _EmbeddingBackbone()
    model.n_vq = 4

    input_ids = torch.tensor([7], dtype=torch.long)
    _, embeds, update = model.preprocess(
        input_ids,
        None,
        _omni_is_prefill=True,
        audio_state={"is_stopping": True},
        ref_offset=5,
    )

    torch.testing.assert_close(embeds, torch.full((1, 4), 7.0))
    assert update == {
        "audio_state": {"is_stopping": False},
        "ref_offset": 6,
    }
    assert "mtp_inputs" not in update
