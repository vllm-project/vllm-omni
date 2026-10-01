# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model Runner V2 path of MiniCPM-o 4.5's Talker: device-side codec output,
EOS control and the 16-frame codec penalty must match the V1 host path."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    _OFFLINE_CODEC_MAX_NEW_TOKENS,
    MiniCPMO45OmniTTSForConditionalGeneration,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_EOS = 6561


def _talker(max_reqs: int = 8, max_position_embeddings: int = 4096):
    talker = object.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    talker._codec_eos_id = _EOS
    talker._tts_config = SimpleNamespace(max_position_embeddings=max_position_embeddings)
    talker._mrv2_empty_speech = torch.zeros(max_reqs, dtype=torch.bool)
    talker._mrv2_forced_eos = None
    talker._mrv2_decode_rows_logged = False
    return talker


def _batch(rows: list[dict], *, pad_to: int | None = None):
    """rows: one dict per request with slot, prompt_len, span (token ids) and prefill flag."""
    ids = [token for row in rows for token in row["span"]]
    num_tokens = len(ids)
    padded = pad_to or num_tokens
    starts = np.cumsum([0] + [len(row["span"]) for row in rows])
    seq_lens = [row["computed"] + len(row["span"]) for row in rows]
    return SimpleNamespace(
        num_reqs=len(rows),
        has_prefill=any(row["prefill"] for row in rows),
        is_prefilling_np=np.array([row["prefill"] for row in rows]),
        idx_mapping_np=np.array([row["slot"] for row in rows]),
        idx_mapping=torch.tensor([row["slot"] for row in rows]),
        input_ids=torch.tensor(ids + [0] * (padded - num_tokens), dtype=torch.int32),
        logits_indices=torch.tensor(starts[1:] - 1),
        seq_lens=torch.tensor(seq_lens + [0] * 2, dtype=torch.int32),
    ), padded


def _req_states(prompt_lens: dict[int, int], max_reqs: int = 8):
    prompt_len = torch.zeros(max_reqs, dtype=torch.int32)
    for slot, value in prompt_lens.items():
        prompt_len[slot] = value
    return SimpleNamespace(prompt_len=SimpleNamespace(gpu=prompt_len))


def test_mrv2_output_emits_decode_input_codes_with_validity_and_forced_eos() -> None:
    talker = _talker()
    rows = [
        # Final prefill chunk: no code yet (V1 emits an empty delta).
        dict(slot=2, prompt_len=5, computed=0, span=[0, 0, 0, 0, 0], prefill=True),
        # Decode with a codec id: emitted this step.
        dict(slot=0, prompt_len=4, computed=6, span=[17], prefill=False),
        # Decode whose input is codec EOS: request ended, force EOS again.
        dict(slot=1, prompt_len=4, computed=8, span=[_EOS], prefill=False),
    ]
    batch, padded = _batch(rows, pad_to=8)
    buffers = [{"audio_state": {"finished": False}}, {}, {}]
    hidden = torch.zeros((padded, 4))
    out = talker.make_omni_output_mrv2(
        hidden,
        input_batch=batch,
        req_states=_req_states({cast(int, row["slot"]): cast(int, row["prompt_len"]) for row in rows}),
        model_intermediate_buffer=buffers,
    )
    assert isinstance(out, OmniOutput)
    codes = out.multimodal_outputs["codes"]["audio"]
    valid = out.multimodal_outputs["meta"]["codec_frame_valid"]
    assert codes.shape == (padded, 1) and valid.shape == (padded,)
    assert codes[:, 0].tolist() == batch.input_ids.tolist()
    assert valid.tolist() == [False] * 5 + [True, False] + [False]
    assert talker.take_mrv2_forced_eos(batch, None, 3).tolist() == [False, False, True]
    # Consumed once; a warmup sampler call sees nothing.
    assert talker.take_mrv2_forced_eos(batch, None, 3) is None


def test_mrv2_output_empty_condition_and_length_cap() -> None:
    talker = _talker()
    prompt_lens = {3: 6, 4: 4096 - 100}
    # Empty Thinker condition: preprocess marks the request finished at prefill.
    batch, _ = _batch([dict(slot=3, prompt_len=6, computed=0, span=[0] * 6, prefill=True)])
    talker.make_omni_output_mrv2(
        torch.zeros((6, 4)),
        input_batch=batch,
        req_states=_req_states(prompt_lens),
        model_intermediate_buffer=[{"audio_state": {"finished": True}}],
    )
    assert talker.take_mrv2_forced_eos(batch, None, 1).tolist() == [True]
    # Its later decode rows stay forced and emit nothing, whatever the input id.
    batch, _ = _batch([dict(slot=3, prompt_len=6, computed=6, span=[42], prefill=False)])
    out = talker.make_omni_output_mrv2(
        torch.zeros((1, 4)), input_batch=batch, req_states=_req_states(prompt_lens), model_intermediate_buffer=[{}]
    )
    assert out.multimodal_outputs["meta"]["codec_frame_valid"].tolist() == [False]
    assert talker.take_mrv2_forced_eos(batch, None, 1).tolist() == [True]

    # Offline cap: min(2048, context - prompt) samples, the last one EOS.
    limit = min(_OFFLINE_CODEC_MAX_NEW_TOKENS, 100) - 1
    for step, forced in ((limit - 1, False), (limit, True)):
        batch, _ = _batch(
            [dict(slot=4, prompt_len=prompt_lens[4], computed=prompt_lens[4] + step - 1, span=[5], prefill=False)]
        )
        out = talker.make_omni_output_mrv2(
            torch.zeros((1, 4)), input_batch=batch, req_states=_req_states(prompt_lens), model_intermediate_buffer=[{}]
        )
        assert out.multimodal_outputs["meta"]["codec_frame_valid"].tolist() == [True]
        assert talker.take_mrv2_forced_eos(batch, None, 1).tolist() == [forced]


@pytest.mark.parametrize("forced", [None, [False, True]])
def test_sampler_adapter_keeps_upstream_counts_and_only_forces_codec_eos(mocker, forced):
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import MiniCPMO45TalkerSampler

    output = SimpleNamespace(sampled_token_ids=torch.tensor([[2], [3]]), num_sampled=torch.tensor([1, 0]))
    base = mocker.Mock(return_value=output)
    base.req_states = object()
    mask = None if forced is None else torch.tensor(forced)
    talker = SimpleNamespace(_codec_eos_id=7, take_mrv2_forced_eos=mocker.Mock(return_value=mask))
    sampler = MiniCPMO45TalkerSampler(base, talker)
    logits, batch = torch.zeros(2, 8), object()
    assert sampler(logits, batch) is output
    assert output.sampled_token_ids.tolist() == [[2], [3 if forced is None else 7]]
    assert output.num_sampled.tolist() == [1, 0]
    base.assert_called_once_with(logits, batch)
    talker.take_mrv2_forced_eos.assert_called_once_with(batch, base.req_states, 2)
