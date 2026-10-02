# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The MiniCPM-o 4.5 Talker's per-step path must not block the host on the GPU."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from tests.model_executor.models.minicpmo_4_5.test_talker_host_sync import _EOS, _infos, _make_talker, _states, _step
from vllm_omni.utils.device_copy import index_to_device

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


def test_talker_step_does_not_synchronize_cuda(mocker) -> None:
    talker = _make_talker("cuda")
    states = _states()
    talker._request_audio_states = copy.deepcopy(states)
    infos = _infos(states)
    input_ids = torch.tensor([3, _EOS, _EOS, 6], dtype=torch.int32, device="cuda")
    hidden = torch.randn(4, 4, device="cuda")
    penalties = torch.tensor([1.05, 1.2, 1.05, 1.0], device="cuda")
    # Warm the pinned host allocator and CUDA context outside the check.
    index_to_device([1], "cuda")
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        with torch.inference_mode():
            *_, sampled = _step(talker, infos, hidden, mocker, batched=True, input_ids=input_ids, penalties=penalties)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert sampled.reshape(-1).tolist()[1:] == [_EOS, _EOS, _EOS]
    assert talker._request_audio_states["req-live"]["recent_codes"] == [1, 2, 3]
    assert talker._request_audio_states["req-eos"]["finished"] is True


def test_talker_condition_upload_does_not_synchronize_cuda() -> None:
    talker = _make_talker("cuda")
    talker.emb_text = nn.Embedding(16, 4).cuda()
    talker.projector_semantic = nn.Linear(6, 4).cuda()
    talker._normalize = True
    talker._text_eos_id = 14
    talker._tts_bos_id = 15
    token_ids = torch.tensor([2, 3, 4])
    hidden_states = torch.randn(3, 6)
    expected = talker._build_condition_embeddings(token_ids.cuda(), hidden_states.cuda())
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        with torch.inference_mode():
            condition = talker._build_condition_embeddings(token_ids, hidden_states)
            boundary = talker._build_condition_embeddings(token_ids[:0], hidden_states[:0])
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert torch.equal(condition, expected)
    assert boundary.shape == (2, 4)


@torch.inference_mode()
def test_v1_batched_decode_and_async_snapshot_match_scalar(mocker):
    from vllm_omni.worker.gpu_ar_model_runner import _snapshot_tensor_payload_to_cpu_async

    ids = torch.tensor([3, _EOS, _EOS, 6], dtype=torch.int32, device="cuda")
    hidden = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
    penalties = torch.tensor([1.05, 1.2, 1.05, 1.0], device="cuda")
    results, states = [], []
    for batched in (False, True):
        talker = _make_talker("cuda")
        talker._request_audio_states = copy.deepcopy(_states())
        results.append(
            _step(talker, _infos(_states()), hidden, mocker, batched=batched, input_ids=ids, penalties=penalties)
        )
        states.append(copy.deepcopy(talker._request_audio_states))
    for index in (0, 2, 3, 4):
        torch.testing.assert_close(results[0][index], results[1][index], rtol=0, atol=0)
    assert states[0] == states[1]
    reference, candidate = [result[1].multimodal_outputs for result in results]
    for group in ("codes", "meta"):
        for key in reference[group]:
            for left, right in zip(reference[group][key], candidate[group][key], strict=True):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
    snapshot = _snapshot_tensor_payload_to_cpu_async(
        {"hidden_states": hidden, "multimodal_outputs": candidate},
        copy_stream=torch.cuda.Stream(),
        pin_memory=True,
    )
    expected_hidden = hidden.cpu().clone()
    snapshot.wait()
    torch.testing.assert_close(snapshot.payload["hidden_states"], expected_hidden, rtol=0, atol=0)
    for group in ("codes", "meta"):
        for key in reference[group]:
            for left, right in zip(
                reference[group][key], snapshot.payload["multimodal_outputs"][group][key], strict=True
            ):
                torch.testing.assert_close(left.cpu(), right, rtol=0, atol=0)
    hidden.fill_(-1)
    torch.testing.assert_close(snapshot.payload["hidden_states"], expected_hidden, rtol=0, atol=0)
