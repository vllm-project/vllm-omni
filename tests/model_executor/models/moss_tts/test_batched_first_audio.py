# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.first_audio_state import MossEarlyFirstAudioState
from vllm_omni.model_executor.models.output_templates import RequestBatchTensor


def state():
    owner = SimpleNamespace(model=SimpleNamespace(audio_pad_token_id=16, audio_assistant_slot_token_id=7))
    result = MossEarlyFirstAudioState(owner, None)
    owner._first_audio_sender = object()
    return result


def entry(rid="a", computed=2, length=2, cap=5):
    return (
        0,
        0,
        0,
        length,
        dict(
            req_id=rid,
            _omni_prompt_len=4,
            _omni_num_computed_tokens=computed,
            sampling_params=SimpleNamespace(max_tokens=cap),
        ),
        True,
    )


@pytest.mark.parametrize(
    "computed,length,cap,eligible", [(0, 2, 5, False), (2, 2, 1, False), (2, 2, 5, True), (3, 1, 5, True)]
)
def test_final_prefill_waits_for_normal_mtp(computed, length, cap, eligible):
    s = state()
    s.record_prefills(object(), [entry(computed=computed, length=length, cap=cap)])
    assert ("a" in s.waiting) == eligible
    # Prefill itself does not invoke sampling or create any first output.
    original: dict = {}
    assert s.after_sample(None, None, None, None, original, None) is original


def test_mixed_batch_owns_codes_and_delivers_only_first_once(mocker):
    s = state()
    s.record_prefills(object(), [entry("b")])
    p = mocker.patch.object(s, "_publish", return_value=["b"])
    codes = torch.tensor([[1, 2], [3, 4], [5, 6]])
    s.after_mtp(["a", "b", "c"], codes, torch.tensor([7, 7, 7]))
    codes.fill_(99)
    assert p.call_args.args[0] == ["b"]
    assert p.call_args.args[1].tolist() == [[3, 4]]
    assert p.call_args.args[2].tolist() == [True]
    batch = SimpleNamespace(req_ids=["c", "b", "a"])
    original = {"codes": {"audio": RequestBatchTensor(codes)}}
    result = s.after_sample(batch, torch.zeros(3, 4), None, None, original, None)
    assert result["meta"]["first_audio"].tensor.tolist() == [False, True, False]
    assert "meta" not in original
    assert s.after_sample(batch, None, None, None, original, None) is original
    s.after_mtp(["a", "b", "c"], codes, torch.tensor([7, 7, 7]))
    assert p.call_count == 1
    s.remove("b")
    assert not s.waiting and not s.seen and not s.delivered and not s.updates


@pytest.mark.parametrize("codes,token,valid", [([16, 16], 7, False), ([1, 2], 9, False), ([1, 2], 7, True)])
def test_validity(codes, token, valid, mocker):
    s = state()
    s.record_prefills(None, [entry()])
    p = mocker.patch.object(s, "_publish", return_value=["a"])
    s.after_mtp(["a"], torch.tensor([codes]), torch.tensor([token]))
    assert p.call_args.args[2].tolist() == [valid]
    assert bool(s.delivered["a"]) == valid


def test_rejected_sink_keeps_normal_codec_path(mocker):
    s = state()
    s.record_prefills(None, [entry()])
    mocker.patch.object(s, "_publish", return_value=[])
    s.after_mtp(["a"], torch.tensor([[1, 2]]), torch.tensor([7]))
    assert not s.delivered and not s.updates
    result: dict = {}
    assert s.after_sample(None, None, None, None, result, None) is result


def test_cancel_before_first_mtp(mocker):
    s = state()
    s.record_prefills(None, [entry()])
    s.remove("a")
    p = mocker.patch.object(s, "_publish")
    s.after_mtp(["a"], torch.tensor([[1, 2]]), torch.tensor([7]))
    p.assert_not_called()


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
