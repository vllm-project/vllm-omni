# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Batched decode preparation preserves the single-request state machine."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.easymagpie.easymagpie import EasyMagpieTTSForConditionalGeneration
from vllm_omni.model_executor.models.easymagpie.runner import EasyMagpieGPUARModelRunner
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("phoneme", [False, True])
@pytest.mark.parametrize("gapped", [False, True])
def test_batched_prefill_matches_scalar_for_variable_contexts(phoneme, gapped):
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.arch = SimpleNamespace(text_prefill_num=4, phoneme_stacking_factor=2, condition_on_user_speech=False)
    model._batched_decode_inputs_enabled = True
    model.has_phoneme = phoneme
    model.phonemes_delay = 3
    model.phoneme_bos_id = 10
    model.embedding_dim = 3
    model.text_embedding = torch.nn.Embedding.from_pretrained(torch.arange(60).view(20, 3).float())
    model._embed_phoneme = lambda ids: torch.full((len(ids), 3), 7.0)
    model._maybe_set_lt_sampling_params = lambda info: None
    model._prefill_cache = {
        (0, "Aria", context, "cpu"): torch.arange(length * 3).view(length, 3).float()
        for context, length in (("short", 2), ("long", 5))
    }
    contexts = ["short", "long", "short"] if gapped else ["short", "short", "long"]
    prefixes = [[1, 2, 3, 4], [5, 6], []]
    infos, offsets, lengths = {}, [], []
    end = 2
    for index, (context, prefix) in enumerate(zip(contexts, prefixes, strict=True)):
        length = model._prefill_cache[(0, "Aria", context, "cpu")].shape[0] + 4
        offsets.append(end)
        lengths.append(length)
        end += length + (2 if gapped else 0)
        infos[str(index)] = dict(
            speaker_id="Aria",
            context_text=context,
            prefill_text_tokens=prefix,
            text_prefill_num=4,
            text_tokens=prefix + [11],
        )
    ids = torch.full((end + 3,), 99, dtype=torch.long)
    embeds = torch.full((end + 3, 3), -1.0)
    expected_ids, expected_embeds = ids.clone(), embeds.clone()
    for info, start, count in zip(infos.values(), offsets, lengths, strict=True):
        scalar_info = dict(info, _omni_num_computed_tokens=0, _omni_prompt_len=count)
        scalar_ids, scalar_embeds, _ = model._preprocess_prefill(
            ids[start : start + count], None, count, torch.device("cpu"), scalar_info
        )
        expected_ids[start : start + count] = scalar_ids
        expected_embeds[start : start + count] = scalar_embeds
    handled = model.preprocess_batch(
        req_ids=list(infos),
        model_intermediate_buffer=infos,
        device=torch.device("cpu"),
        input_ids=ids,
        inputs_embeds=embeds,
        token_offsets=offsets,
        num_scheduled_tokens=lengths,
        num_computed_tokens=[0] * 3,
        prompt_lens=lengths,
    )
    assert set(handled) == set(infos)
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(embeds, expected_embeds, rtol=0, atol=0)
    assert all(info["decode_offset"] == 4 for info in infos.values())


@pytest.mark.parametrize("computed,count,cached", [(2, 4, True), (0, 4, True), (0, 6, False)])
def test_batched_prefill_leaves_partial_or_cold_prompts_for_scalar_path(computed, count, cached):
    model = SimpleNamespace(
        _batched_decode_inputs_enabled=True,
        arch=SimpleNamespace(text_prefill_num=4),
        _prefill_cache={(0, "Aria", "[EN]", "cpu"): torch.ones(2, 3)} if cached else {},
    )
    ids, embeds = torch.full((count,), 99), torch.full((count, 3), -1.0)
    assert not EasyMagpieTTSForConditionalGeneration.preprocess_batch(
        model,
        req_ids=["one"],
        device=torch.device("cpu"),
        model_intermediate_buffer={"one": {"speaker_id": "Aria", "text_prefill_num": 4, "text_tokens": [1]}},
        input_ids=ids,
        inputs_embeds=embeds,
        token_offsets=[0],
        num_scheduled_tokens=[count],
        num_computed_tokens=[computed],
        prompt_lens=[6],
    )
    assert torch.all(ids == 99)
    assert torch.all(embeds == -1)


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("stacking", [0, 1, 2])
def test_batched_preparation_matches_individual_requests(device, stacking):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    reference = _model(device, stacking)
    batched = _model(device, stacking)
    device = reference._dec_audio_codes.device
    ids = torch.zeros(32, device=device, dtype=torch.long)
    infos = [
        {"decode_offset": 0},
        {"decode_offset": 3, "phoneme_ended": True},
        {"decode_offset": 5},
        {"decode_offset": 6, "phoneme_ended": torch.tensor(False, device=device)},
        {"decode_offset": 7, "phoneme_ended": torch.tensor(True, device=device)},
        {"decode_offset": 8},
        # Partial feedback retains the original single-row behavior.
        {"decode_offset": 9, "last_audio_codes": torch.tensor([12], device=device)},
    ]
    for index, info in enumerate(infos):
        info.update(text_tokens=[10, 11, 12], text_token=[13, 14, 15, 16, 17, 18, 19], text_token_start=3)
        info.setdefault("last_audio_codes", torch.arange(4, device=device) + index * 7)
        if stacking and index >= 3:
            tokens = torch.full((1, stacking), 9, device=device)
            if index == 3:
                tokens[0, -1] = reference.phoneme_eos_id
            info["last_phoneme_token"] = tokens
    retained_ended = []
    for epoch, offsets in enumerate(([11, 2, 8, 1, 15, 4, 19], [4, 15, 1, 8, 2, 11, 19])):
        rows = list(zip(offsets, infos, strict=True))
        batched.prepare_decode_batch(rows, device)
        for start, info in rows:
            expected = reference._preprocess_decode(ids[start : start + 1], start, device, deepcopy(info))
            actual = batched._preprocess_decode(ids[start : start + 1], start, device, deepcopy(info))
            torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
            _assert_updates(actual[2], expected[2])
            if stacking:
                retained_ended.append((actual[2]["phoneme_ended"], expected[2]["phoneme_ended"].clone()))
            info.update(actual[2])
        assert not batched._prepared_decode_updates
        _assert_buffers(reference, batched, stacking)
        # Completed slots are admitted again with fresh BOS/startup metadata.
        if epoch == 0:
            infos[0] = {"decode_offset": 3, "text_tokens": [21, 22, 23, 24]}
    for actual, expected in retained_ended:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu"])
def test_failed_or_cancelled_preparation_does_not_leak_metadata(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = _model(device, 2)
    device = model._dec_audio_codes.device
    ids = torch.zeros(32, device=device, dtype=torch.long)
    model.prepare_decode_batch([(1, {"decode_offset": 5}), (2, {"decode_offset": 6})], device)
    model.clear_decode_batch()
    _, _, fresh = model._preprocess_decode(ids[1:2], 1, device, {"decode_offset": 3})
    assert fresh["decode_offset"] == 4
    assert not fresh["phoneme_ended"]
    assert model._dec_audio_valid[1] == 0
    assert torch.equal(model._dec_phoneme_tokens[1], torch.tensor([10, 10], device=device))


def test_runner_prepares_only_completed_prompt_single_token_rows_after_reordering():
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    observed = []
    runner.model = SimpleNamespace(prepare_decode_batch=lambda rows, device: observed.append((rows, device)))
    runner.query_start_loc = SimpleNamespace(cpu=np.array([0, 4, 5, 6, 7]))
    runner._omni_num_scheduled_tokens_np = np.array([4, 1, 1, 1])
    keys = ("c", "a", "b", "one-token-prefill")
    runner.model_intermediate_buffer = {key: {"decode_offset": i} for i, key in enumerate(keys)}
    runner.requests = {key: SimpleNamespace(prompt_token_ids=[0] * 4) for key in keys}
    runner.input_batch = SimpleNamespace(num_computed_tokens_cpu=np.array([0, 4, 4, 3]))
    runner._maybe_run_batch_preprocess(
        list(keys), torch.device("cpu"), None, None, runner._omni_num_scheduled_tokens_np
    )
    assert observed == [([(4, {"decode_offset": 1}), (5, {"decode_offset": 2})], torch.device("cpu"))]


@pytest.mark.parametrize("resident", [False, True])
def test_user_audio_decode_keeps_speaking_end_token(resident):
    from vllm_omni.model_executor.models.easymagpie.device_decode_state import DeviceDecodeState

    reference, batched = _model("cpu", 0), _model("cpu", 0)
    for model in (reference, batched):
        model.arch.use_user_speaking_end_token = True
        model.arch.audio_user_speaking_end_id = 1030
    if resident:
        batched._device_decode_state = DeviceDecodeState(batched, 3, 32, 32)
    device = torch.device("cpu")
    ids = torch.zeros(32, dtype=torch.long)
    infos = [{"decode_offset": 5, "user_audio_prefilled": True}, {"decode_offset": 5}, {"decode_offset": 5}]
    rows = list(enumerate(infos))
    keys = {"request_keys": [(str(index), index) for index in range(3)]} if resident else {}
    batched.prepare_decode_batch(rows, device, **keys)
    assert 0 not in batched._prepared_decode_updates
    assert set(batched._prepared_decode_updates) == {1, 2}
    for start, info in rows:
        reference._preprocess_decode(ids[start : start + 1], start, device, dict(info))
        batched._preprocess_decode(ids[start : start + 1], start, device, dict(info))
    if resident:
        batched._device_decode_state.stage(batched, torch.arange(3))
    _assert_buffers(reference, batched, 0)
    assert batched._dec_audio_codes[0].tolist() == [1030] * 4
    assert batched._dec_audio_codes[1].tolist() == [1025] * 4


@pytest.mark.parametrize("failed", [False, True])
def test_runner_clears_prepared_rows_on_success_and_failure(monkeypatch, failed):
    runner = object.__new__(EasyMagpieGPUARModelRunner)
    cleared = []
    runner.model = SimpleNamespace(clear_decode_batch=lambda: cleared.append(True))

    def preprocess(*args):
        if failed:
            raise RuntimeError("preprocessing failed")
        return "prepared"

    monkeypatch.setattr(GPUARModelRunner, "_preprocess", preprocess)
    if failed:
        with pytest.raises(RuntimeError, match="preprocessing failed"):
            runner._preprocess(None, 2)
    else:
        assert runner._preprocess(None, 2) == "prepared"
    assert cleared == [True]


def _model(device, stacking):
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.arch = SimpleNamespace(phoneme_stacking_factor=stacking, audio_bos_id=1025)
    model.has_phoneme = stacking > 0
    model.phonemes_delay = 3
    model.phoneme_bos_id = 10
    model.phoneme_eos_id = 11
    model.speech_delay = 5
    model.embedding_dim = 3
    model.num_codebooks = 4
    model._combined_embeddings = torch.zeros(32, 3, device=device)
    model._dec_text_tokens = torch.arange(32, dtype=torch.long, device=device)
    model._dec_text_mask = torch.zeros(32, dtype=torch.long, device=device)
    model._dec_audio_codes = torch.arange(128, dtype=torch.long, device=device).view(32, 4)
    model._dec_audio_valid = torch.zeros(32, dtype=torch.long, device=device)
    if stacking:
        model._dec_phoneme_tokens = torch.arange(32 * stacking, dtype=torch.long, device=device).view(32, stacking)
        model._dec_phoneme_valid = torch.zeros(32, dtype=torch.long, device=device)
    return model


def _assert_updates(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], torch.Tensor):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        else:
            assert actual[key] == expected[key]


def _assert_buffers(reference, actual, stacking):
    names = ["_dec_text_tokens", "_dec_text_mask", "_dec_audio_codes", "_dec_audio_valid"]
    if stacking:
        names += ["_dec_phoneme_tokens", "_dec_phoneme_valid"]
    for name in names:
        torch.testing.assert_close(getattr(actual, name), getattr(reference, name), rtol=0, atol=0)
