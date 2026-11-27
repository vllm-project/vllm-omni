# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config.compilation import CompilationConfig, CUDAGraphMode
from vllm.config.vllm import set_current_vllm_config
from vllm.sampling_params import SamplingParams

from vllm_omni.model_executor.models.audio8_tts import audio8_tts_slow_ar
from vllm_omni.model_executor.models.audio8_tts.configuration_audio8_tts import Audio8TTSConfig
from vllm_omni.worker_v2.model_states import init_omni_model_state

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PAD = 1
_EOS = 2
_SEMANTIC_BEGIN = 100
_DIM = 2


def test_mrv2_emits_audio8_frames_from_current_sample_and_hidden(monkeypatch):
    hf_config = Audio8TTSConfig(
        dim=_DIM,
        vocab_size=128,
        num_codebooks=2,
        codebook_size=16,
        semantic_begin_id=_SEMANTIC_BEGIN,
        semantic_end_id=_SEMANTIC_BEGIN + 15,
        pad_token_id=_PAD,
        eos_token_id=_EOS,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="audio8",
            hf_config=hf_config,
            dtype=torch.bfloat16,
            head_dtype=None,
            enable_prompt_embeds=False,
            uses_mrope=False,
            max_model_len=32,
            get_inputs_embeds_size=lambda: _DIM,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=8),
        compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        parallel_config=SimpleNamespace(data_parallel_size=1, use_sequence_parallel_moe=False),
    )
    backbone = SimpleNamespace(
        layers=[],
        embed_input_ids=lambda ids: ids.to(torch.bfloat16).unsqueeze(-1).expand(*ids.shape, _DIM),
        make_empty_intermediate_tensors=None,
    )
    fast_ar_calls = []

    def fast_ar(slow_ar_hidden, semantic_token_id, generator, **_):
        fast_ar_calls.append((semantic_token_id.item(), slow_ar_hidden.tolist(), generator))
        semantic_code = (semantic_token_id - _SEMANTIC_BEGIN).clamp_min(0)
        return torch.stack((semantic_code, slow_ar_hidden[:, 0].long()), dim=1)

    monkeypatch.setattr(audio8_tts_slow_ar, "Qwen2Model", lambda **_: backbone)
    monkeypatch.setattr(audio8_tts_slow_ar, "Audio8TTSFastAR", lambda **_: fast_ar)
    monkeypatch.setattr(audio8_tts_slow_ar, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False))
    with set_current_vllm_config(config):
        model = audio8_tts_slow_ar.Audio8TTSSlowARForConditionalGeneration(vllm_config=config)
    with torch.no_grad():
        model.codebook_embeddings.weight.copy_(torch.arange(32).unsqueeze(-1).expand(-1, _DIM))
    state = init_omni_model_state(config, model, None, torch.device("cpu"))
    for slot, (name, prompt) in enumerate((("short", [10, 11]), ("long", [20, 21, 22]))):
        state.intermediate_buffer.buffers[slot] = {
            "req_id": name,
            "embed": {"prefill": torch.tensor(prompt, dtype=torch.bfloat16).unsqueeze(-1).expand(-1, _DIM)},
            "sampling_params": SamplingParams(extra_args={"tts_local_seed": 17 + slot}),
        }
    req_states = SimpleNamespace(prompt_len=[2, 3], num_computed_tokens=[0, 0])
    req_ids = ["short", "long"]

    def step(slots, counts, input_ids, sampled, hidden):
        starts = np.cumsum([0, *counts], dtype=np.int32)
        batch = SimpleNamespace(
            num_reqs=len(slots),
            idx_mapping_np=np.array(slots),
            req_ids=[req_ids[slot] for slot in slots],
            num_scheduled_tokens=np.array(counts),
            query_start_loc_np=starts,
            query_start_loc=torch.from_numpy(starts),
            logits_indices=torch.from_numpy(starts[1:] - 1),
        )
        input_ids = torch.tensor(input_ids)
        inputs = {"input_ids": input_ids, "inputs_embeds": model.embed_input_ids(input_ids).contiguous()}
        state.run_preprocess(batch, inputs, req_states)
        hidden = torch.tensor(hidden, dtype=torch.bfloat16).unsqueeze(-1).expand(-1, _DIM).contiguous()
        text_hidden, output = state.postprocess_model_output(hidden, batch, req_states)
        torch.testing.assert_close(text_hidden, hidden)
        state.post_sample(batch, text_hidden, torch.tensor(sampled).unsqueeze(-1), output)
        return inputs, output

    inputs, output = step([0, 1], [2, 2], [10, 11, 20, 21], [103, 104], [5, 2, 6, 7])
    assert inputs["input_ids"].tolist() == [_PAD] * 4
    assert output["meta"]["codec_frame_valid"].tolist() == [False, True, False, False]
    assert output["codes"]["audio"].tolist() == [[0, 0], [3, 2], [0, 0], [0, 0]]

    # Long finishes prefill in one token while short decodes in a reordered batch.
    req_states.num_computed_tokens = [2, 2]
    inputs, output = step([1, 0], [1, 1], [22, 103], [104, _EOS], [8, 9])
    assert inputs["input_ids"].tolist() == [_PAD, 103]
    # Short's next input is semantic embedding 103 plus both codebook embeddings (3 + 18).
    assert inputs["inputs_embeds"].tolist() == [[22, 22], [124, 124]]
    assert output["meta"]["codec_frame_valid"].tolist() == [True, False]
    assert output["codes"]["audio"][0].tolist() == [4, 8]
    assert [(token, hidden) for token, hidden, _ in fast_ar_calls] == [
        (103, [[2, 2]]),
        (104, [[8, 8]]),
        (_EOS, [[9, 9]]),
    ]
    assert [rng.initial_seed() for _, _, rng in fast_ar_calls] == [17, 18, 17]
    assert fast_ar_calls[0][2] is fast_ar_calls[2][2]
