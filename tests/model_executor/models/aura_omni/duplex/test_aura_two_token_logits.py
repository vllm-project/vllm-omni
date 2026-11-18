# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU checks for AURA_v2 first-token silent/respond logits force."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate

from vllm_omni.model_executor.models.aura_omni.control_tokens import AURA_V2
from vllm_omni.model_executor.models.aura_omni.two_token_logits import (
    AuraTwoTokenLogitsProcessor,
    AuraTwoTokenLogitsProcessorV1,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _v1_proc() -> AuraTwoTokenLogitsProcessorV1:
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(tokenizer="/nonexistent/AURA_v2", model="/nonexistent/AURA_v2")
    )
    return AuraTwoTokenLogitsProcessorV1(vllm_config, device=torch.device("cpu"), is_pin_memory=False)


def test_resolve_v2_control_ids() -> None:
    proc = _v1_proc()
    assert proc.silent == AURA_V2.silent_id
    assert proc.respond == AURA_V2.respond_id


def test_first_step_masks_to_silent_and_respond() -> None:
    proc = _v1_proc()
    silent, respond = proc.silent, proc.respond
    output: list[int] = []
    params = SamplingParams(
        temperature=0.0,
        min_tokens=0,
        stop_token_ids=[silent, AURA_V2.im_end_id],
        extra_args={"aura_mode": "streaming"},
    )
    proc.update_state(BatchUpdate(batch_size=1, removed=[], added=[(0, params, None, output)], moved=[]))
    vocab = max(silent, respond) + 8
    logits = torch.zeros((1, vocab), dtype=torch.float32)
    logits[0, silent] = 1.5
    logits[0, respond] = 2.0
    logits[0, 3] = 9.0
    out = proc.apply(logits)
    assert out[0, 3] == float("-inf")
    assert out[0, silent] == 1.5
    assert out[0, respond] == 2.0


def test_body_masks_both_control_ids() -> None:
    proc = _v1_proc()
    silent, respond = proc.silent, proc.respond
    output = [respond, 100]
    params = SamplingParams(
        temperature=0.0,
        min_tokens=0,
        stop_token_ids=[silent],
        extra_args={"aura_mode": "streaming"},
    )
    proc.update_state(BatchUpdate(batch_size=1, removed=[], added=[(0, params, None, output)], moved=[]))
    vocab = max(silent, respond) + 8
    logits = torch.zeros((1, vocab), dtype=torch.float32)
    logits[0, silent] = 3.0
    logits[0, respond] = 4.0
    logits[0, 5] = 1.0
    out = proc.apply(logits)
    assert out[0, silent] == float("-inf")
    assert out[0, respond] == float("-inf")
    assert out[0, 5] == 1.0


def test_streaming_accepts_sampling_temperature() -> None:
    AuraTwoTokenLogitsProcessor.validate_params(
        SamplingParams(temperature=0.2, min_tokens=0, extra_args={"aura_mode": "streaming"})
    )


def test_offline_qa_is_a_no_op() -> None:
    proc = _v1_proc()
    params = SamplingParams(temperature=0.0, extra_args={"aura_mode": "offline_qa"})
    proc.update_state(BatchUpdate(batch_size=1, removed=[], added=[(0, params, None, [])], moved=[]))
    assert proc.entries == {}


def _mrv2_proc(*, max_num_reqs: int = 2, max_model_len: int = 16) -> AuraTwoTokenLogitsProcessor:
    silent, respond = AURA_V2.silent_id, AURA_V2.respond_id
    assert respond is not None
    vocab = max(silent, respond) + 8
    prompt_len = SimpleNamespace(np=torch.zeros(max_num_reqs, dtype=torch.int32).numpy())
    total_len = SimpleNamespace(gpu=torch.zeros(max_num_reqs, dtype=torch.int32))
    all_token_ids = SimpleNamespace(gpu=torch.zeros((max_num_reqs, max_model_len), dtype=torch.int32))
    req_states = SimpleNamespace(
        max_num_reqs=max_num_reqs,
        vocab_size=vocab,
        prompt_len=prompt_len,
        total_len=total_len,
        all_token_ids=all_token_ids,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(tokenizer="/nonexistent/AURA_v2", model="/nonexistent/AURA_v2")
    )
    return AuraTwoTokenLogitsProcessor(vllm_config, req_states)  # type: ignore[arg-type]


def test_mrv2_first_step_masks_to_silent_and_respond() -> None:
    proc = _mrv2_proc()
    silent, respond = proc.silent, proc.respond
    params = SamplingParams(
        temperature=0.0,
        min_tokens=0,
        stop_token_ids=[silent, AURA_V2.im_end_id],
        extra_args={"aura_mode": "streaming"},
    )
    assert proc.add_request(0, params) is True
    proc.req_states.prompt_len.np[0] = 4
    proc.req_states.total_len.gpu[0] = 4
    vocab = max(silent, respond) + 8
    logits = torch.zeros((1, vocab), dtype=torch.float32)
    logits[0, silent] = 1.5
    logits[0, respond] = 2.0
    logits[0, 3] = 9.0
    ctx = SimpleNamespace(expanded_idx_mapping=torch.tensor([0], dtype=torch.int32))
    out = proc.apply(logits, ctx)  # type: ignore[arg-type]
    assert out[0, 3] == float("-inf")
    assert out[0, silent] == 1.5
    assert out[0, respond] == 2.0


def test_mrv2_body_masks_both_control_ids() -> None:
    proc = _mrv2_proc()
    silent, respond = proc.silent, proc.respond
    params = SamplingParams(
        temperature=0.0,
        min_tokens=0,
        stop_token_ids=[silent],
        extra_args={"aura_mode": "streaming"},
    )
    assert proc.add_request(0, params) is True
    proc.req_states.prompt_len.np[0] = 4
    proc.req_states.total_len.gpu[0] = 5
    proc.req_states.all_token_ids.gpu[0, 4] = respond
    vocab = max(silent, respond) + 8
    logits = torch.zeros((1, vocab), dtype=torch.float32)
    logits[0, silent] = 3.0
    logits[0, respond] = 4.0
    logits[0, 5] = 1.0
    ctx = SimpleNamespace(expanded_idx_mapping=torch.tensor([0], dtype=torch.int32))
    out = proc.apply(logits, ctx)  # type: ignore[arg-type]
    assert out[0, silent] == float("-inf")
    assert out[0, respond] == float("-inf")
    assert out[0, 5] == 1.0
