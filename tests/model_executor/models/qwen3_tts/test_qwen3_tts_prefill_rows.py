# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
    Qwen3TTSConfig,
    Qwen3TTSTalkerConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    PRECOMPUTED_TEXT_IDS_KEY,
    Qwen3TTSPromptEmbedsBuilder,
)
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
    Qwen3TTSTalkerForConditionalGeneration,
)

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

HIDDEN = 48
PAD_ID = 14


def _talker() -> Qwen3TTSTalkerForConditionalGeneration:
    torch.manual_seed(0)
    dev = torch.device("cuda")
    config = Qwen3TTSConfig(tts_bos_token_id=50, tts_eos_token_id=51, tts_pad_token_id=52)
    talker_config = Qwen3TTSTalkerConfig(
        codec_nothink_id=10,
        codec_think_id=11,
        codec_think_bos_id=12,
        codec_think_eos_id=13,
        codec_pad_id=PAD_ID,
        codec_bos_id=15,
        codec_language_id={"english": 16},
        spk_id={"vivian": 17, "serena": 18},
        num_code_groups=4,
    )
    text_embedding = torch.nn.Embedding(200, 32).to(dev, torch.bfloat16)
    projection = torch.nn.Sequential(torch.nn.Linear(32, 40), torch.nn.SiLU(), torch.nn.Linear(40, HIDDEN))
    projection = projection.to(dev, torch.bfloat16)
    pad = torch.randn(1, HIDDEN, device=dev).to(torch.bfloat16)
    builder = Qwen3TTSPromptEmbedsBuilder(
        config=config,
        talker_config=talker_config,
        model_path="",
        text_embedding=text_embedding,
        text_projection=projection,
        codec_embed=torch.nn.Embedding(64, HIDDEN).to(dev, torch.bfloat16),
        residual_code_embeddings=lambda: [],
        speaker_encoder=torch.nn.Identity(),
        tts_pad_embed=pad,
        encode_ref_audio_batch=lambda *args, **kwargs: [],
    )
    # The prompt paths under test take their ids precomputed; only the full path asks for a tokenizer.
    builder._text_tokenizer = lambda *args, **kwargs: {"input_ids": torch.tensor([[2, 3, 4, 5]])}
    with torch.inference_mode():
        builder.build_projected_text_table(chunk_rows=64)

    model = Qwen3TTSTalkerForConditionalGeneration.__new__(Qwen3TTSTalkerForConditionalGeneration)
    model.talker_config = talker_config
    model._embedding_dtype = torch.bfloat16
    model._tts_pad_embed = pad
    model._prompt_builder = builder
    model._use_v2_model_runner = True
    model._silence_ban_frames = 0
    model._ref_codes_pending = False
    return model


def _request(req_id: str, ids: list[int], speaker: str = "Vivian", **extra: Any) -> dict[str, Any]:
    info = {
        "req_id": req_id,
        "text": ["hello"],
        "task_type": ["CustomVoice"],
        "language": ["English"],
        "speaker": [speaker],
        "instruct": [""],
        "non_streaming_mode": [True],
        PRECOMPUTED_TEXT_IDS_KEY: [ids],
        "_omni_num_computed_tokens": 0,
    }
    info.update(extra)
    return info


def _assert_same(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected)
        for key in expected:
            _assert_same(actual[key], expected[key])
    elif isinstance(expected, torch.Tensor):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
    else:
        assert actual == expected


@torch.inference_mode()
def test_batched_prefill_rows_match_per_request_preprocess():
    model = _talker()
    dev = torch.device("cuda")
    # The first request of a speaker takes the full path and caches its prompt pieces.
    warm = _request("warm", list(range(100, 112)))
    model.preprocess(torch.zeros(30, dtype=torch.int32, device=dev), torch.zeros(30, HIDDEN, device=dev), **warm)

    stored_ids = list(range(120, 140))
    # (start, span, info) rows of one step.
    full = _request("full", list(range(1, 15)))
    head = _request("head", stored_ids)
    longer = _request("longer", list(range(60, 71)))
    uncached = _request("uncached", list(range(80, 95)), speaker="Serena")
    instruct = _request("instruct", list(range(1, 15)), instruct=["Speak calmly"])
    # Prompt lengths: 9 constant prefix rows + text + eos + tail.
    entries = [(0, 17, full), (17, 5, head), (22, 40, longer), (62, 20, uncached), (82, 18, instruct)]

    def run(batched: bool, rows: list[tuple[int, int, dict[str, Any]]], total: int):
        ids = torch.arange(total, dtype=torch.int32, device=dev)
        embeds = torch.randn(total, HIDDEN, device=dev).to(torch.bfloat16)
        embeds_before = embeds.clone()
        infos = [dict(info) for _start, _span, info in rows]
        if batched:
            results = model.preprocess_prefill_rows_mrv2(
                entries=[(start, span, info) for (start, span, _), info in zip(rows, infos)],
                input_ids=ids,
                input_embeds=embeds,
            )
        else:
            # Serving projects the new requests' texts in one batch first (same table rows).
            model._prompt_builder.preprocess_infos_batch(req_infos=infos, device=dev)
            results = []
            for (start, span, _), info in zip(rows, infos):
                new_ids, new_emb, update = model.preprocess(
                    ids[start : start + span], embeds[start : start + span], **info
                )
                embeds[start : start + span] = new_emb
                ids[start : start + span] = new_ids
                results.append(update)
        return ids, embeds, embeds_before, results

    total = 100
    b_ids, b_embeds, b_before, b_results = run(True, entries, total)
    assert [r is None for r in b_results] == [False, False, False, True, True]
    handled = [(0, 17), (17, 22), (22, 62)]
    r_ids, r_embeds, _, r_results = run(False, entries[:3], total)
    for lo, hi in handled:
        assert torch.equal(b_embeds[lo:hi], r_embeds[lo:hi])
        assert torch.equal(b_ids[lo:hi], torch.full((hi - lo,), PAD_ID, dtype=torch.int32, device=dev))
    # Rows of requests the batch left alone are untouched.
    assert torch.equal(b_embeds[62:], b_before[62:])
    for actual, expected in zip(b_results[:3], r_results, strict=True):
        _assert_same(actual, expected)
    stored_prompt = b_results[1]["embed"]["prefill"]
    assert stored_prompt.shape[0] == 9 + (len(stored_ids) - 8) + 2

    # The next step continues "head" from its stored prompt, past its end (padding).
    head_next = dict(head)
    head_next["embed"] = {"prefill": stored_prompt}
    head_next["meta"] = dict(b_results[1]["meta"])
    head_next["hidden_states"] = dict(b_results[1]["hidden_states"])
    head_next["_omni_num_computed_tokens"] = 5
    rows = [(3, 25, head_next)]
    b_ids, b_embeds, _, b_results = run(True, rows, 40)
    r_ids, r_embeds, _, r_results = run(False, rows, 40)
    assert b_results[0] is not None
    assert torch.equal(b_embeds[3:28], r_embeds[3:28])
    assert torch.equal(b_ids[3:28], r_ids[3:28])
    _assert_same(b_results[0], r_results[0])
