# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_realtime_talker_preserves_state_across_chunked_prefill() -> None:
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSRealtimeTalkerForGeneration,
    )

    talker = MossTTSRealtimeTalkerForGeneration.__new__(MossTTSRealtimeTalkerForGeneration)
    talker.n_vq = 2
    talker.audio_pad_token = 1024
    talker.text_vocab_size = 16
    talker.text_pad_id = 2
    talker._batch_state = None
    talker._batch_state_spans = None

    embedded_audio: list[torch.Tensor | None] = []

    def build_input_embeds(input_ids, audio_codes):
        embedded_audio.append(audio_codes.clone() if audio_codes is not None else None)
        return torch.zeros((input_ids.shape[0], 1))

    generated: list[None] = []

    def generate_frame(*args, **kwargs):
        generated.append(None)
        return torch.ones((1, talker.n_vq), dtype=torch.long)

    talker._build_input_embeds = build_input_embeds
    talker.local_transformer = SimpleNamespace(generate_frame=generate_frame)
    talker.local_lm_heads = []

    ref_codes = torch.arange(10, dtype=torch.long).reshape(5, 2)
    info: dict[str, Any] = {
        "codes": {"ref": ref_codes},
        "ids": {"all": [7, 8]},
        "max_new_frames": [3],
    }
    chunks = [(0, 2, False), (2, 2, False), (4, 1, True)]

    state: dict[str, Any] | None = None
    for computed, length, eligible in chunks:
        _, _, update = talker.preprocess(
            torch.ones(length, dtype=torch.long),
            None,
            _omni_is_prefill=True,
            _omni_num_computed_tokens=computed,
            _omni_prompt_len=5,
            **info,
        )
        info.update(update)
        if state is None:
            state = info["audio_state"]
        else:
            assert info["audio_state"] is state

        output = talker.make_omni_output(
            torch.ones((length, 1)),
            model_intermediate_buffer=[info],
            request_token_spans=[(0, length)],
            request_sample_eligible=[eligible],
        )
        if not eligible:
            assert output.multimodal_outputs == {}
        logits = talker.compute_logits(torch.ones((1, 1)))
        assert logits.argmax().item() == (7 if eligible else talker.text_pad_id)

    assert state is not None
    assert info["ref_offset"] == 5
    assert state["step"] == 1
    assert state["text_cursor"] == 1
    assert state["remaining_text"] == [7, 8]
    assert len(generated) == 1
    assert output.multimodal_outputs["codes"]["audio"][0].shape == (1, 2)
    for actual, expected in zip(embedded_audio, ref_codes.split((2, 2, 1)), strict=True):
        assert actual is not None
        torch.testing.assert_close(actual, expected)


def test_mrv2_postprocess_model_output_wires_sample_eligibility_to_realtime_talker() -> None:
    import numpy as np

    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSRealtimeTalkerForGeneration,
    )
    from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer
    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

    talker = MossTTSRealtimeTalkerForGeneration.__new__(MossTTSRealtimeTalkerForGeneration)
    talker.n_vq = 2
    talker.audio_pad_token = 1024
    talker.text_vocab_size = 16
    talker.text_pad_id = 2
    talker.audio_eos_id = 15
    talker._batch_state = None
    talker._batch_state_spans = None
    talker.local_transformer = SimpleNamespace(
        generate_frame=lambda *args, **kwargs: torch.tensor([[3, 4]], dtype=torch.long)
    )
    talker.local_lm_heads = []

    state = OmniModelState.__new__(OmniModelState)
    state.model = talker
    state.have_multimodal_outputs = True
    state.intermediate_buffer = OmniIntermediateBuffer(max_num_reqs=4)
    # Row 0 (slot 2) is mid-prefill (computed 2 + scheduled 2 < prefill 5 -> ineligible)
    # Row 1 (slot 0) is finishing prefill (computed 4 + scheduled 1 >= prefill 5 -> eligible)
    state.intermediate_buffer.buffers[2] = {
        "req_id": "req_mid",
        "audio_state": {"is_stopping": False, "step": 0, "text_cursor": 0, "remaining_text": [9]},
    }
    state.intermediate_buffer.buffers[0] = {
        "req_id": "req_done",
        "audio_state": {"is_stopping": False, "step": 0, "text_cursor": 0, "remaining_text": [10]},
    }

    input_batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping_np=np.array([2, 0], dtype=np.int32),
        query_start_loc_np=np.array([0, 2, 3], dtype=np.int32),
        num_scheduled_tokens=np.array([2, 1], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([2, 4], dtype=np.int32),
        prefill_len_np=np.array([5, 5], dtype=np.int32),
    )

    hidden = torch.ones((3, 4), dtype=torch.float32)
    text_hidden, mm_outputs = state.postprocess_model_output(hidden, input_batch, SimpleNamespace())
    assert text_hidden.shape == (3, 4)
    assert "codes" in mm_outputs
    audio_codes = mm_outputs["codes"]["audio"]
    assert len(audio_codes) == 2
    assert audio_codes[0].numel() == 0  # mid-prefill request skipped
    assert torch.equal(audio_codes[1], torch.tensor([[3, 4]], dtype=torch.long))
    assert state.intermediate_buffer.buffers[2]["audio_state"]["step"] == 0
    assert state.intermediate_buffer.buffers[0]["audio_state"]["step"] == 1

    # Negative check: models not declaring requires_request_sample_eligibility do not receive the kwarg
    received_kwargs: dict[str, Any] = {}
    plain_model = SimpleNamespace(make_omni_output=lambda out, **kw: (received_kwargs.update(kw), out)[1])
    state.model = plain_model
    state.postprocess_model_output(hidden, input_batch, SimpleNamespace())
    assert "request_sample_eligible" not in received_kwargs


def test_moss_realtime_codec_mrv2_dummy_run_skips_empty_runtime_info() -> None:
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import (
        MossTTSCodecDecoder,
    )

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(n_vq=16, model_type="moss_tts_realtime"),
            async_chunk=True,
            use_v2_model_runner=True,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
    )
    decoder = MossTTSCodecDecoder(vllm_config=config)
    decoder._codec = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]))
    # 65536 is divisible by n_vq=16; dummy profiling on MRv2 passes [{}]
    dummy_ids = torch.zeros(65536, dtype=torch.long)
    out = decoder.forward(
        input_ids=dummy_ids,
        runtime_additional_information=[{}],
        seq_token_counts=[65536],
    )
    assert len(out.multimodal_outputs["model_outputs"]) == 1
    assert out.multimodal_outputs["model_outputs"][0].numel() == 0
    assert decoder._stream_req_slots == {}


def test_moss_realtime_local_transformer_incremental_kv_matches_reprefill(monkeypatch) -> None:
    import torch.nn as nn
    from vllm.config import VllmConfig, set_current_vllm_config

    from vllm_omni.model_executor.models.moss_tts.configuration_moss_tts import (
        MossTTSLocalTransformerConfig,
    )
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local import (
        MossTTSRealtimeLocalTransformer,
    )

    cfg = MossTTSLocalTransformerConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        audio_vocab_size=32,
        rvq=4,
    )
    with set_current_vllm_config(VllmConfig()):
        local_tf = MossTTSRealtimeLocalTransformer(cfg).eval()
        lm_heads = nn.ModuleList([nn.Linear(64, 32, bias=False) for _ in range(4)])

    hidden = torch.randn((2, 64), dtype=torch.float32)
    hist = [[1, 2], [3, 4], [5], []]

    torch.manual_seed(99)
    cached_codes = local_tf.generate_frame(
        hidden,
        lm_heads,
        temperature=0.8,
        top_k=10,
        top_p=0.6,
        do_sample=True,
        repetition_penalty=1.1,
        history_per_codebook=hist,
    )

    monkeypatch.delattr(type(local_tf.model), "forward_cached")
    torch.manual_seed(99)
    reprefill_codes = local_tf.generate_frame(
        hidden,
        lm_heads,
        temperature=0.8,
        top_k=10,
        top_p=0.6,
        do_sample=True,
        repetition_penalty=1.1,
        history_per_codebook=hist,
    )
    assert torch.equal(cached_codes, reprefill_codes)
