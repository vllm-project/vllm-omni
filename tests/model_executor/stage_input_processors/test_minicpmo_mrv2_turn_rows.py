# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRv2 ordinary-turn Thinker rows: runner trimming -> accumulation -> llm2tts.

Each step's hidden rows and ledger are fixtures standing in for a target
forward. The row trimming (OmniAsyncOutput), accumulation (MultimodalPayload)
and Talker handoff (llm2tts) are the real code. The step plan is chosen by the
test, not by the scheduler.
"""

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from tests.model_executor.stage_input_processors.test_minicpmo_4_5_omni import _output
from tests.worker_v2.test_omni_ar_model_runner import _async_output
from vllm_omni.model_executor.models.minicpmo_4_5.handoff import unpack_handoff_tensor
from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import llm2tts
from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.outputs.output_modality import OutputModality

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PROMPT = [101, 102]
_EOS = 151645


def _value(step: int, token: int, position: int) -> float:
    # The step index separates a rejected row from a later row at the same
    # token and position.
    return float(step * 100000 + position * 100 + token)


def _step_payload(mocker, step: int, inputs: list[tuple[int, int]], num_rejected: int) -> dict:
    n = len(inputs)
    hidden = torch.stack([torch.full((4,), _value(step, t, p)) for t, p in inputs])
    batch = mocker.Mock(spec=InputBatch)
    batch.query_start_loc_np = np.array([0, n])
    batch.num_scheduled_tokens = np.array([n])
    batch.num_reqs = 1
    batch.num_tokens_after_padding = n
    output = _async_output(
        req_ids=["req-1"],
        sampler_output=SamplerOutput(
            sampled_token_ids=torch.ones(1, n, dtype=torch.long),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=torch.tensor([n - num_rejected]),
            num_rejected=torch.tensor([num_rejected]),
        ),
        num_sampled_tokens=torch.tensor([n - num_rejected]),
        num_rejected_tokens=torch.tensor([num_rejected]),
        text_hidden=hidden,
        multimodal_outputs={
            "latent": hidden,
            "latent_input_ids": torch.tensor([t for t, _ in inputs]).reshape(-1, 1),
            "latent_positions": torch.tensor([p for _, p in inputs]).reshape(-1, 1),
        },
        input_batch=batch,
        async_chunk=False,
    ).get_output()
    return output.inter_stage_outputs[0]


def _handoff(mocker, monkeypatch, steps, output_ids, *, repeat=1):
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    accumulated = MultimodalPayload()
    for index, (inputs, num_rejected) in enumerate(steps):
        payload = _step_payload(mocker, index, inputs, num_rejected)
        keep = {k: payload[k] for k in ("latent", "latent_input_ids", "latent_positions")}
        for _ in range(repeat):
            accumulated = accumulated.merged_with(MultimodalPayload.from_raw(keep, "latent"))
    accumulated.consolidate_tensors(OutputModality.from_string("latent"))
    mm = accumulated.to_dict()
    source = _output(
        prompt_ids=_PROMPT,
        output_ids=output_ids,
        latent=mm.pop("latent"),
        multimodal_output=mm,
    )
    info = llm2tts([source], prompt=[{}])[0]["model_intermediate_buffer"]
    hidden = torch.as_tensor(unpack_handoff_tensor(info["hidden_states"]["tts"]))
    return info["ids"]["tts"], hidden[:, 0].tolist()


_PREFILL = ([(101, 0), (102, 1)], 0)


@pytest.mark.parametrize("repeat", [1, 2], ids=["single-channel", "both-channels"])
def test_length_stop_without_terminal_step_drops_the_final_reply_token(mocker, monkeypatch, repeat):
    """Current MRv2 behaviour: the last sampled token never becomes a target input."""
    steps = [_PREFILL, ([(11, 2)], 0), ([(12, 3)], 0)]
    ids, rows = _handoff(mocker, monkeypatch, steps, [11, 12, 13], repeat=repeat)
    assert ids == [11, 12]
    assert rows == [_value(1, 11, 2), _value(2, 12, 3)]


def test_correction_token_at_the_length_limit_has_no_row(mocker, monkeypatch):
    # Step 1 verifies drafts 12, 99; 99 is rejected and 13 is the correction.
    steps = [_PREFILL, ([(11, 2), (12, 3), (99, 4)], 1)]
    ids, rows = _handoff(mocker, monkeypatch, steps, [11, 12, 13])
    assert ids == [11, 12]
    assert rows == [_value(1, 11, 2), _value(1, 12, 3)]


def test_length_limit_inside_accepted_drafts_keeps_every_row(mocker, monkeypatch):
    # All drafts accepted; the length limit trims the bonus token, so the
    # final committed token 13 was itself a target input.
    steps = [_PREFILL, ([(11, 2), (12, 3), (13, 4)], 0)]
    ids, rows = _handoff(mocker, monkeypatch, steps, [11, 12, 13])
    assert ids == [11, 12, 13]
    assert rows == [_value(1, 11, 2), _value(1, 12, 3), _value(1, 13, 4)]


def test_eos_stop_needs_no_row_for_the_eos_token(mocker, monkeypatch):
    steps = [_PREFILL, ([(11, 2)], 0), ([(12, 3)], 0)]
    ids, rows = _handoff(mocker, monkeypatch, steps, [11, 12, _EOS])
    assert ids == [11, 12]
    assert rows == [_value(1, 11, 2), _value(2, 12, 3)]


@pytest.mark.parametrize("repeat", [1, 2], ids=["single-channel", "both-channels"])
def test_draft_free_terminal_step_supplies_the_final_own_row(mocker, monkeypatch, repeat):
    """The consumer needs no change once a target forward consumes the final token."""
    steps = [_PREFILL, ([(11, 2)], 0), ([(12, 3)], 0), ([(13, 4)], 0)]
    ids, rows = _handoff(mocker, monkeypatch, steps, [11, 12, 13], repeat=repeat)
    assert ids == [11, 12, 13]
    assert rows == [_value(1, 11, 2), _value(2, 12, 3), _value(3, 13, 4)]


def test_rejected_row_with_the_final_id_and_position_is_not_reused(mocker, monkeypatch):
    # Draft 13 at position 4 is rejected (sampling) and the correction is also
    # 13. Only the later draft-free step's row conditions the Talker.
    steps = [_PREFILL, ([(11, 2), (12, 3), (13, 4)], 1), ([(13, 4)], 0)]
    ids, rows = _handoff(mocker, monkeypatch, steps, [11, 12, 13])
    assert ids == [11, 12, 13]
    assert rows == [_value(1, 11, 2), _value(1, 12, 3), _value(2, 13, 4)]
