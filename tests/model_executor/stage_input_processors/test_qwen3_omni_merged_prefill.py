# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Thinker -> Talker chunk 0 sends one merged prefill tensor.

The Talker reads the accept-layer hidden state only at audio, image and video
placeholders in user turns and the word embedding at every other row. The merged payload
must give the Talker bitwise the same prefill inputs as sending both tensors.
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm_omni.model_executor.stage_input_processors.qwen3_omni as q3
from vllm_omni.data_entry_keys import to_dict
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ACCEPT = 24
THINKER_HIDDEN = 16
TALKER_HIDDEN = 8
IM_START, IM_END, NL = 151644, 151645, 198
SYSTEM, USER, ASSISTANT = 8948, 872, 77091
AUDIO, IMAGE, VIDEO = 151675, 151655, 151656
FIRST_TOKEN = 1000

# A duplex-style prompt: system, an earlier audio turn and its answer, then
# the current camera turn.
PROMPT = [
    *(IM_START, SYSTEM, NL, 11, 12, IM_END, NL),
    *(IM_START, USER, NL, *[AUDIO] * 5, 13, IM_END, NL),
    *(IM_START, ASSISTANT, NL, 14, 15, 16, IM_END, NL),
    *(IM_START, USER, NL, *[IMAGE] * 4, *[VIDEO] * 2, 17, 18, IM_END, NL),
    *(IM_START, ASSISTANT, NL),
]
PLACEHOLDER = torch.tensor([token in (AUDIO, IMAGE, VIDEO) for token in PROMPT])
# continue_final_message: the reply goes on from an assistant prefix, and the
# Talker text-projects that whole turn, placeholder included.
ASSISTANT_PREFIX_PROMPT = [*PROMPT, IMAGE, 19]


class _Manager:
    def __init__(self) -> None:
        self.put_req_chunk: defaultdict[str, int] = defaultdict(int)
        self.request_payload: dict[str, dict] = {}
        self._pending_streaming_prefills: dict[str, dict] = {}
        self.config = SimpleNamespace(
            hf_config=SimpleNamespace(
                im_start_token_id=IM_START,
                user_token_id=USER,
                talker_config=SimpleNamespace(accept_hidden_layer=ACCEPT),
                thinker_config=SimpleNamespace(audio_token_id=AUDIO, image_token_id=IMAGE, video_token_id=VIDEO),
            )
        )


def _request(output_token_ids, *, resumable=False, prompt=PROMPT):
    return SimpleNamespace(
        external_req_id="r",
        prompt_token_ids=list(prompt),
        output_token_ids=list(output_token_ids),
        all_token_ids=[*prompt, *output_token_ids],
        output_token_count=len(output_token_ids),
        last_output_token_id=output_token_ids[-1] if output_token_ids else None,
        resumable=resumable,
        additional_information=None,
        sampling_params=SimpleNamespace(all_stop_token_ids={IM_END}, max_tokens=64),
    )


def _thinker_rows(prompt=PROMPT):
    generator = torch.Generator().manual_seed(0)
    embed = torch.randn(len(prompt), THINKER_HIDDEN, generator=generator).to(torch.bfloat16)
    hidden = torch.randn(len(prompt), THINKER_HIDDEN, generator=generator).to(torch.bfloat16)
    sampled = torch.randn(1, THINKER_HIDDEN, generator=generator).to(torch.bfloat16)
    # The Thinker publishes each TTS embedding as [1, 1, H].
    tts = {
        name: torch.randn(1, 1, THINKER_HIDDEN, generator=generator).to(torch.bfloat16)
        for name in ("bos", "eos", "pad")
    }
    return embed, hidden, sampled, tts


def _chunk0(embed, hidden, sampled, tts, prefill_chunks, prompt=PROMPT):
    manager = _Manager()
    start = 0
    for i, size in enumerate(prefill_chunks):
        last = i == len(prefill_chunks) - 1
        output = {
            "hidden_states": {"layers": {0: embed[start : start + size], ACCEPT: hidden[start : start + size]}},
            "embed": {
                "tts_bos": tts["bos"],
                "tts_eos": tts["eos"],
                "tts_pad": tts["pad"],
                "sampled": sampled if last else torch.empty(0, dtype=torch.bfloat16),
            },
        }
        payload = q3.thinker2talker_async_chunk(manager, output, _request([FIRST_TOKEN] if last else [], prompt=prompt))
        start += size
        if last:
            return payload
        assert payload is None


@pytest.mark.parametrize("prefill_chunks", [(len(PROMPT),), (9, len(PROMPT) - 9)])
def test_chunk0_merges_placeholder_hidden_rows_into_prefill(prefill_chunks):
    embed, hidden, sampled, tts = _thinker_rows()
    payload = _chunk0(embed, hidden, sampled, tts, prefill_chunks)

    assert payload.hidden_states is None
    assert payload.meta.prefill_holds_accept_hidden is True
    merged = payload.embed.prefill
    assert merged.shape == (len(PROMPT) + 1, THINKER_HIDDEN)
    assert torch.equal(merged[: len(PROMPT)][PLACEHOLDER], hidden[PLACEHOLDER])
    assert torch.equal(merged[: len(PROMPT)][~PLACEHOLDER], embed[~PLACEHOLDER])
    assert torch.equal(merged[len(PROMPT) :], sampled)


def test_resumable_chunk0_keeps_both_tensors():
    # Later segment chunks of a resumable session still carry
    # hidden_states.output, so its chunk 0 must not switch layout.
    embed, hidden, sampled, tts = _thinker_rows()
    manager = _Manager()
    output = {
        "hidden_states": {"layers": {0: embed, ACCEPT: hidden}},
        "embed": {"tts_bos": tts["bos"], "tts_eos": tts["eos"], "tts_pad": tts["pad"], "sampled": sampled},
    }
    assert q3.thinker2talker_async_chunk(manager, output, _request([FIRST_TOKEN], resumable=True)) is None
    step = {"hidden_states": {"layers": {0: sampled, ACCEPT: sampled}}, "embed": {}}
    payload = q3.thinker2talker_async_chunk(manager, step, _request([FIRST_TOKEN, 1001], resumable=True))

    assert payload.meta.prefill_holds_accept_hidden is None
    assert torch.equal(payload.embed.prefill, torch.cat((embed, sampled)))
    assert torch.equal(payload.hidden_states.output, torch.cat((hidden, sampled)))


class _Talker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.text_projection = nn.Linear(THINKER_HIDDEN, TALKER_HIDDEN).to(torch.bfloat16)
        self.hidden_projection = nn.Linear(THINKER_HIDDEN, TALKER_HIDDEN).to(torch.bfloat16)
        self.codec_embedding = nn.Embedding(4300, TALKER_HIDDEN).to(torch.bfloat16)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.codec_embedding(input_ids)


def _model() -> Qwen3OmniMoeForConditionalGeneration:
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    nn.Module.__init__(model)
    model.talker = _Talker()
    model.config = SimpleNamespace(
        im_start_token_id=IM_START,
        system_token_id=SYSTEM,
        user_token_id=USER,
        assistant_token_id=ASSISTANT,
        tts_pad_token_id=151671,
        talker_config=SimpleNamespace(
            codec_nothink_id=4203,
            codec_think_bos_id=4204,
            codec_think_eos_id=4205,
            codec_pad_id=4196,
            codec_bos_id=4197,
            text_config=SimpleNamespace(hidden_size=TALKER_HIDDEN),
        ),
    )
    model.thinker_config = SimpleNamespace(audio_token_id=AUDIO, image_token_id=IMAGE, video_token_id=VIDEO)
    model.tts_text_spk_token_ids = {"ethan": 2302}
    model.default_tts_text_spk_type = "ethan"
    return model


def _two_tensor_payload(embed, hidden, sampled, tts, prompt):
    return {
        "embed": {
            "prefill": torch.cat((embed, sampled)),
            "tts_bos": tts["bos"],
            "tts_eos": tts["eos"],
            "tts_pad": tts["pad"],
        },
        "hidden_states": {"output": hidden},
        "ids": {"all": [*prompt, FIRST_TOKEN], "prompt": list(prompt)},
        "meta": {},
    }


def test_talker_prefill_inputs_match_two_tensor_payload():
    embed, hidden, sampled, tts = _thinker_rows()
    merged = to_dict(_chunk0(embed, hidden, sampled, tts, (len(PROMPT),)))
    both = _two_tensor_payload(embed, hidden, sampled, tts, PROMPT)
    assert merged["ids"] == both["ids"]

    model = _model()
    span = torch.zeros(len(PROMPT) + 1, TALKER_HIDDEN, dtype=torch.bfloat16)
    with torch.no_grad():
        ids_a, embeds_a, update_a = model.talker_preprocess_prefill(None, span, both)
        ids_b, embeds_b, update_b = model.talker_preprocess_prefill(None, span, merged)

    # Both user turns (11 + 13 rows) and the 9-row assistant bootstrap.
    assert embeds_a.shape[0] == 11 + 13 + 9
    assert torch.equal(ids_a, ids_b)
    assert torch.equal(embeds_a, embeds_b)
    assert torch.equal(update_a["hidden_states"]["trailing_text"], update_b["hidden_states"]["trailing_text"])
    assert torch.equal(update_a["embed"]["tts_pad_projected"], update_b["embed"]["tts_pad_projected"])

    # The marker decides the layout, not a hidden_states.output that a later
    # chunk may have left next to the merged prefill.
    stray = {**merged, "hidden_states": {"output": hidden[:1]}}
    with torch.no_grad():
        ids_c, embeds_c, _ = model.talker_preprocess_prefill(None, span, stray)
    assert torch.equal(ids_a, ids_c)
    assert torch.equal(embeds_a, embeds_c)


def test_assistant_prefix_placeholder_stays_an_embedding():
    prompt = ASSISTANT_PREFIX_PROMPT
    embed, hidden, sampled, tts = _thinker_rows(prompt)
    merged = to_dict(_chunk0(embed, hidden, sampled, tts, (len(prompt),), prompt=prompt))
    assert torch.equal(merged["embed"]["prefill"][len(PROMPT)], embed[len(PROMPT)])

    model = _model()
    span = torch.zeros(len(prompt) + 1, TALKER_HIDDEN, dtype=torch.bfloat16)
    with torch.no_grad():
        ids_a, embeds_a, update_a = model.talker_preprocess_prefill(
            None, span, _two_tensor_payload(embed, hidden, sampled, tts, prompt)
        )
        ids_b, embeds_b, update_b = model.talker_preprocess_prefill(None, span, merged)

    assert torch.equal(ids_a, ids_b)
    assert torch.equal(embeds_a, embeds_b)
    assert torch.equal(update_a["hidden_states"]["trailing_text"], update_b["hidden_states"]["trailing_text"])


def test_talker_prefill_without_marker_needs_hidden_states():
    embed, hidden, sampled, tts = _thinker_rows()
    merged = to_dict(_chunk0(embed, hidden, sampled, tts, (len(PROMPT),)))
    merged["meta"].pop("prefill_holds_accept_hidden")
    span = torch.zeros(len(PROMPT) + 1, TALKER_HIDDEN, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="hidden_states.output"), torch.no_grad():
        _model().talker_preprocess_prefill(None, span, merged)
