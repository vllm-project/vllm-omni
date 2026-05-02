# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for OmniRequestState multimodal DELTA drain and consolidation guard."""

from unittest.mock import MagicMock

import pytest
import torch
from vllm.outputs import PoolingRequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import FinishReason

from vllm_omni.engine.output_modality import OutputModalityNames
from vllm_omni.engine.output_processor import OmniRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Audio is explicitly listed as a drainable modality
AUDIO = OutputModalityNames.AUDIO

# Latent is explicitly not drainable, but the choice doesn't matter here as
# long as isn't listed as drainable. I.e., could also be arbitrary keys for
# the purposes of these tests
LATENT = OutputModalityNames.LATENT

# NOTE: detokenizer and logprobs aren't really used here, but we mock them since
# some of the utils called in vLLM superclass assert require them to be None.
_DETOK = MagicMock(
    output_token_ids=[0],
    get_next_output_text=MagicMock(return_value=""),
    num_output_tokens=MagicMock(return_value=1),
)
_LOGPROBS = MagicMock(logprobs=None, cumulative_logprob=None, prompt_logprobs=None)

_DEFAULT_STATE_KWARGS = dict(
    request_id="r",
    external_req_id="r",
    parent_req=None,
    request_index=0,
    lora_request=None,
    prompt=None,
    prompt_token_ids=[0],
    prompt_embeds=None,
    logprobs_processor=_LOGPROBS,
    detokenizer=_DETOK,
    max_tokens_param=None,
    arrival_time=0.0,
    queue=None,
    log_stats=False,
    stream_interval=1,
)


def _make_state(output_kind: RequestOutputKind):
    return OmniRequestState(**_DEFAULT_STATE_KWARGS, output_kind=output_kind)


def test_init_empty_dict():
    """Ensure mm_accumulated is initially empty."""
    assert _make_state(RequestOutputKind.CUMULATIVE).mm_accumulated == {}
    assert _make_state(RequestOutputKind.DELTA).mm_accumulated == {}


def test_delta_drains_output_modality_per_step():
    """DELTA drains the mm_type key (output modality) but preserves hidden-state keys."""
    s = _make_state(RequestOutputKind.DELTA)
    audio1, audio2, hs1, hs2 = [torch.ones(num_elem) for num_elem in range(1, 5)]

    # Add audio and hidden state tensors
    s.add_multimodal_tensor(audio1, mm_type=AUDIO)  # should be drained
    s.add_multimodal_tensor(hs1, mm_type=LATENT)  # shouldn't be drained

    out1 = s._new_completion_output([1], None, None)
    out1_audio = out1.multimodal_output[AUDIO]
    out1_hidden = out1.multimodal_output[LATENT]
    assert isinstance(out1_audio, torch.Tensor)
    assert torch.equal(out1.multimodal_output[AUDIO], audio1)
    assert isinstance(out1_hidden, torch.Tensor)
    assert torch.equal(out1.multimodal_output[LATENT], hs1)

    # After emission, hidden states should remain, but audio is drained
    assert set(s.mm_accumulated.keys()) == {LATENT}

    s.add_multimodal_tensor(audio2, AUDIO)
    s.add_multimodal_tensor(hs2, mm_type=LATENT)
    out2 = s._new_completion_output([2], None, None)
    out2_audio = out2.multimodal_output[AUDIO]
    out2_hidden = out2.multimodal_output[LATENT]
    assert isinstance(out2_audio, torch.Tensor)
    assert torch.equal(out2_audio, audio2)
    # Since hidden isn't drained, it's grown to a list
    assert isinstance(out2_hidden, list) and len(out2_hidden) == 2
    assert torch.equal(hs1, out2_hidden[0])
    assert torch.equal(hs2, out2_hidden[1])


def test_cumulative_emits_consolidated_audio_each_step():
    """Ensure cumulative accumulates and consolidates modality keys every step."""
    s = _make_state(RequestOutputKind.CUMULATIVE)
    # NOTE: audio is usually emitted as (1, size) chunks; we need to be sure
    # to not change the tensor dimension when we consolidate
    audio1 = torch.ones(1, 500)
    s.add_multimodal_tensor(audio1, mm_type=AUDIO)
    req_out = s.make_request_output([1], None, None, None)
    assert req_out is not None
    cons_audio = req_out.outputs[0].multimodal_output[AUDIO]
    # Single chunk keeps original shape [1, 500]
    assert isinstance(cons_audio, torch.Tensor) and cons_audio.shape == audio1.shape

    audio2 = torch.ones(1, 300)
    s.add_multimodal_tensor(audio2, mm_type=AUDIO)
    req_out = s.make_request_output([2], None, None, None)
    assert req_out is not None
    cons_audio = req_out.outputs[0].multimodal_output[AUDIO]
    # After consolidation, audio chunks are concatenated on last axis,
    # preserving the [1, N] channel dimension
    total_audio_len = audio1.shape[-1] + audio2.shape[-1]
    assert isinstance(audio2, torch.Tensor) and cons_audio.shape == (1, total_audio_len)

    assert "audio" in s.mm_accumulated


def test_finish_consolidates_hidden_states():
    """Ensure consolidation merges hidden-state tensor lists on finish."""
    s = _make_state(RequestOutputKind.CUMULATIVE)
    s.add_multimodal_tensor(torch.ones(5, 4), mm_type=LATENT)
    s.add_multimodal_tensor(torch.ones(3, 4), mm_type=LATENT)

    result = s.make_request_output([1], None, FinishReason.STOP, None)
    assert result is not None and not isinstance(result, PoolingRequestOutput)

    hs = result.outputs[0].multimodal_output[LATENT]
    assert isinstance(hs, torch.Tensor) and hs.shape[0] == 8


def test_finish_consolidation_for_hs_delta():
    """Ensure finish doesn't drop the accumulated hidden states."""
    s = _make_state(RequestOutputKind.DELTA)
    # hidden state accumulation (nothing drained)
    s.add_multimodal_tensor({"foo": torch.ones(5, 4)}, mm_type=LATENT)
    result = s.make_request_output([0], None, FinishReason.STOP, None)
    assert result is not None and not isinstance(result, PoolingRequestOutput)
    hs = result.outputs[0].multimodal_output["foo"]
    assert isinstance(hs, torch.Tensor) and hs.shape[0] == 5

    # Since we don't drain the hidden states, if we add 3 elements, we should get 8
    s.add_multimodal_tensor({"foo": torch.ones(3, 4)}, mm_type=LATENT)
    result = s.make_request_output([0], None, FinishReason.STOP, None)
    assert result is not None and not isinstance(result, PoolingRequestOutput)
    hs = result.outputs[0].multimodal_output["foo"]
    assert isinstance(hs, torch.Tensor) and hs.shape[0] == 8
    assert "foo" in s.mm_accumulated


def test_finish_consolidation_drains_mm_delta():
    """Ensure making the request output drains modality deltas (e.g., audio)."""
    s = _make_state(RequestOutputKind.DELTA)
    # multimodal data accumulation (drained)
    s.add_multimodal_tensor({AUDIO: torch.ones(5, 4)}, mm_type=AUDIO)
    result = s.make_request_output([0], None, FinishReason.STOP, None)
    assert result is not None and not isinstance(result, PoolingRequestOutput)
    hs = result.outputs[0].multimodal_output[AUDIO]
    assert isinstance(hs, torch.Tensor) and hs.shape[0] == 5

    # Since we did drain the hidden states, we no longer get the 5 back
    s.add_multimodal_tensor({AUDIO: torch.ones(3, 4)}, mm_type=AUDIO)
    result = s.make_request_output([0], None, FinishReason.STOP, None)
    assert result is not None and not isinstance(result, PoolingRequestOutput)
    hs = result.outputs[0].multimodal_output[AUDIO]
    assert isinstance(hs, torch.Tensor) and hs.shape[0] == 3
    assert AUDIO not in s.mm_accumulated  # drained


@pytest.mark.parametrize("mm_type", [AUDIO, "hidden"])
def test_final_only_consolidates_drainable_keys(mm_type):
    """FINAL_ONLY never drains per-step, so modality keys and hidden state
    keys both accumulate and are consolidated on finish."""
    s = _make_state(RequestOutputKind.FINAL_ONLY)

    # NOTE: Currently there is brittlness in the tensor stacking, so we just
    # test a 1D tensor here. The intention is just to ensure audio /hidden
    # behave the same.
    s.add_multimodal_tensor(torch.ones(500), mm_type=mm_type)
    # Non-finish step returns None without calling _new_completion_output
    assert s.make_request_output([1], None, None, None) is None
    assert mm_type in s.mm_accumulated

    s.add_multimodal_tensor(torch.ones(300), mm_type=mm_type)
    result = s.make_request_output([2], None, FinishReason.STOP, None)
    assert result is not None and not isinstance(result, PoolingRequestOutput)

    audio = result.outputs[0].multimodal_output[mm_type]
    assert isinstance(audio, torch.Tensor)
    assert audio.shape == (800,)


def test_cumulative_token_ids_always_set():
    """cumulative_token_ids is set for all output kinds."""
    for kind in (RequestOutputKind.DELTA, RequestOutputKind.CUMULATIVE, RequestOutputKind.FINAL_ONLY):
        s = _make_state(kind)
        out = s._new_completion_output([42], None, None)
        assert hasattr(out, "cumulative_token_ids")
        # The mock detokenizer has output_token_ids=[0]
        assert list(out.cumulative_token_ids) == [0]


def test_cumulative_token_ids_is_a_copy():
    """cumulative_token_ids must be a snapshot, not a live reference."""
    s = _make_state(RequestOutputKind.DELTA)
    out = s._new_completion_output([42], None, None)
    assert out.cumulative_token_ids is not _DETOK.output_token_ids
