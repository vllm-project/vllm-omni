# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts_nano.modeling_moss_tts_nano import (
    _configure_attention_implementation,
    _validate_max_num_seqs,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_non_cuda_attention_uses_sdpa():
    class FakeModel:
        attention_implementation = None

        def _set_attention_implementation(self, implementation):
            self.attention_implementation = implementation

    model = FakeModel()
    _configure_attention_implementation(model, torch.device("cpu"))

    assert model.attention_implementation == "sdpa"


def test_moss_tts_nano_rejects_concurrent_active_sequences():
    _validate_max_num_seqs(1)

    with pytest.raises(ValueError, match="requires max_num_seqs=1"):
        _validate_max_num_seqs(2)


def test_moss_tts_nano_mrv2_dummy_run_with_empty_info_dict_skips_generator():
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.moss_tts_nano.modeling_moss_tts_nano import (
        MossTTSNanoForGeneration,
    )

    model = MossTTSNanoForGeneration.__new__(MossTTSNanoForGeneration)
    model.config = SimpleNamespace(audio_tokenizer_sample_rate=48000, hidden_size=64, vocab_size=32)
    model._device = torch.device("cpu")
    model._stream_gens = {}
    model._ar_last_chunk_flags = []

    # OmniModelState.prepare_inputs on a dummy run passes [{}]
    out = model.forward(
        input_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        runtime_additional_information=[{}],
    )
    assert model._stream_gens == {}
    assert model._ar_last_chunk_flags == [True]
    assert out.multimodal_outputs["model_outputs"][0].numel() == 0
    logits = model.compute_logits(out)
    assert logits.shape == (3, 32)
    assert torch.all(logits.argmax(dim=-1) == 2)

    # None and explicit _is_dummy with text also short-circuit cleanly
    for dummy_payload in (None, [{"_is_dummy": True, "text": "warmup"}]):
        out_dummy = model.forward(
            input_ids=torch.tensor([1], dtype=torch.long),
            runtime_additional_information=dummy_payload,
        )
        assert model._stream_gens == {}
        assert model._ar_last_chunk_flags == [True]
        assert out_dummy.multimodal_outputs["model_outputs"][0].numel() == 0

    # Mixed batch: empty row short-circuits while real row opens a stream generator
    chunk_tensor = torch.ones(160, dtype=torch.float32)

    def _fake_stream_gen(_info):
        yield chunk_tensor, False
        yield chunk_tensor, True

    model._create_stream_gen = _fake_stream_gen
    out_mixed = model.forward(
        input_ids=torch.tensor([1, 2], dtype=torch.long),
        runtime_additional_information=[{}, {"text": "hello", "global_request_id": "req-live"}],
    )
    assert "req-live" in model._stream_gens
    assert model._ar_last_chunk_flags == [True, False]
    assert out_mixed.multimodal_outputs["model_outputs"][0].numel() == 0
    assert torch.equal(out_mixed.multimodal_outputs["model_outputs"][1], chunk_tensor)
