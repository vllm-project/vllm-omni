# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Unit tests for Qwen2.5-Omni talker span assembly from the runner's positions.

A chunked-prefill chunk, a speculative-decoding verify step and a preemption
replay all schedule spans whose length does not identify their phase. Each
position must receive the row for its absolute position: a prompt row of the
reconstructed Talker prompt, or ``thinker_reply[k] + codec_embedding`` for
generation offset ``k``. Replaying the same positions must rebuild the same rows.
"""

import functools
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HIDDEN = 4


@functools.lru_cache(maxsize=1)
def _qwen2_5_omni_mod():
    """Import lazily so collection does not pull model_executor too early."""
    import vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni as m

    return m


def _fake_embed_input_ids(ids: torch.Tensor) -> torch.Tensor:
    # Deterministic, id-distinguishing stand-in for the codec embedding table.
    return ids.reshape(-1, 1).to(torch.float32).expand(-1, HIDDEN).contiguous()


def _make_minimal_model():
    cls = _qwen2_5_omni_mod().Qwen2_5OmniForConditionalGeneration
    # ``__new__`` skips ``__init__`` (no weights, no config); set only what
    # the decode path touches.
    model = cls.__new__(cls)
    model.talker = SimpleNamespace(embed_input_ids=_fake_embed_input_ids)
    return model


def _thinker_reply(num_rows: int) -> torch.Tensor:
    # Row k filled with 100 * (k + 1) so any offset error is visible.
    return (torch.arange(1, num_rows + 1, dtype=torch.float32).reshape(-1, 1) * 100.0).expand(-1, HIDDEN).contiguous()


def test_spec_verify_span_gets_one_thinker_row_per_position():
    # A verify step with 2 draft tokens schedules a 3-position decode span at
    # generation offset 2 (prompt_len 4, num_computed 6). Every position must
    # get thinker_reply[offset + i] + codec_embedding(token_i).
    model = _make_minimal_model()
    input_ids = torch.tensor([11, 22, 33], dtype=torch.long)
    q = _thinker_reply(8)

    out_ids, out_embeds, update = model.talker_preprocess(
        input_ids,
        None,
        embed={"thinker_reply": q},
        _omni_is_prefill=False,
        _omni_prompt_len=4,
        _omni_num_computed_tokens=6,
    )

    assert torch.equal(out_ids, input_ids)
    expected = _fake_embed_input_ids(input_ids) + q[2:5]
    assert torch.equal(out_embeds, expected)
    # Stateless: the reply buffer must not be consumed or rewritten.
    assert update == {}
    assert torch.equal(q, _thinker_reply(8))


def test_decode_span_is_replayable():
    # Preempt/resume replays already-generated tokens as one multi-token span.
    # Because the reply buffer is never consumed, calling preprocess twice with
    # the same metadata must produce identical embeddings (the stock consuming
    # path would have shifted by one queue row per call).
    model = _make_minimal_model()
    input_ids = torch.tensor([7, 8], dtype=torch.long)
    payload = dict(
        embed={"thinker_reply": _thinker_reply(6)},
        _omni_is_prefill=False,
        _omni_prompt_len=3,
        _omni_num_computed_tokens=4,
    )

    _, first, _ = model.talker_preprocess(input_ids, None, **payload)
    _, second, _ = model.talker_preprocess(input_ids, None, **payload)

    assert torch.equal(first, second)


@pytest.mark.parametrize(
    ("offset", "expected_rows"),
    [(4, [4, 5, 5]), (5, [5, 5, 5]), (6, [5, 5, 5]), (9, [5, 5, 5])],
)
def test_positions_past_thinker_reply_repeat_final_pad(offset, expected_rows):
    # Prefill appends text EOS and PAD as the final two reply rows. The HF
    # consumer retains the last PAD row for every remaining audio position.
    model = _make_minimal_model()
    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    q = _thinker_reply(6)

    _, out_embeds, _ = model.talker_preprocess(
        input_ids,
        None,
        embed={"thinker_reply": q},
        _omni_is_prefill=False,
        _omni_prompt_len=2,
        _omni_num_computed_tokens=2 + offset,
    )

    base = _fake_embed_input_ids(input_ids)
    assert torch.equal(out_embeds, base + q[expected_rows])
    assert torch.equal(q, _thinker_reply(6))


THINKER_PROMPT_LEN = 6
TALKER_PROMPT_LEN = THINKER_PROMPT_LEN + 2  # thinker prompt + speaker/pad row + BOS row
CODEC_PAD, CODEC_START, CODEC_MASK = 8292, 8293, 8296


class _FakeTalker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=HIDDEN)
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def embed_input_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return _fake_embed_input_ids(ids)


def _make_prefill_model():
    """Minimal model on which the real ``_thinker_to_talker_prefill`` runs."""
    cls = _qwen2_5_omni_mod().Qwen2_5OmniForConditionalGeneration
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.talker = _FakeTalker()
    model.model = model.talker
    model.thinker_config = SimpleNamespace()
    model.thinker_embedding = torch.nn.Embedding.from_pretrained(_fake_embed_input_ids(torch.arange(256)))
    model.talker_config = SimpleNamespace(
        tts_codec_pad_token_id=CODEC_PAD,
        tts_codec_start_token_id=CODEC_START,
        tts_codec_mask_token_id=CODEC_MASK,
    )
    model.voice_type = "Chelsie"
    model.embed_text_bos_token = torch.full((1, HIDDEN), 7.0)
    model.embed_text_spk_tokens = {}
    model.embed_text_eos_token = torch.full((1, HIDDEN), 8.0)
    model.embed_text_pad_token = torch.full((1, HIDDEN), 9.0)
    model.embed_codec_pad_token = torch.full((1, HIDDEN), 0.5)
    model.embed_codec_bos_token = torch.full((1, HIDDEN), 0.25)
    return model


def _thinker_payload(num_output_rows: int = 5) -> dict:
    # Hidden rows and token embeddings are both position-distinguishing:
    # prompt hidden row i is 1000 * (i + 1), output hidden row k is 10 * (k + 1),
    # and each text-token embedding adds its token ID.
    prompt = (torch.arange(1, THINKER_PROMPT_LEN + 1, dtype=torch.float32).reshape(-1, 1) * 1000.0).expand(-1, HIDDEN)
    output = (torch.arange(1, num_output_rows + 1, dtype=torch.float32).reshape(-1, 1) * 10.0).expand(-1, HIDDEN)
    return dict(
        embed={"prefill": prompt.contiguous()},
        hidden_states={"output": output.contiguous()},
        ids={"prompt": list(range(100, 100 + THINKER_PROMPT_LEN)), "output": list(range(200, 200 + num_output_rows))},
    )


def _full_talker_prompt(model, payload):
    """The complete prompt as reconstructed for an unchunked prefill."""
    ids, embeds, _ = model.talker_preprocess(
        torch.zeros(TALKER_PROMPT_LEN, dtype=torch.long),
        None,
        **payload,
        _omni_is_prefill=True,
        _omni_prompt_len=TALKER_PROMPT_LEN,
        _omni_num_computed_tokens=0,
    )
    assert ids.shape[0] == embeds.shape[0] == TALKER_PROMPT_LEN
    return ids, embeds


@pytest.mark.parametrize(("num_computed", "span_len"), [(7, 1), (2, 3), (0, 4)])
def test_chunked_prefill_span_gets_rows_at_its_offset(num_computed, span_len):
    # The runner copies the returned rows into the positions it scheduled,
    # starting at num_computed_tokens. A chunk at a non-zero offset (e.g. a
    # 1-token tail at offset 7) must receive those prompt rows, not row 0.
    model = _make_prefill_model()
    payload = _thinker_payload()
    full_ids, full_embeds = _full_talker_prompt(model, payload)

    out_ids, out_embeds, update = model.talker_preprocess(
        torch.zeros(span_len, dtype=torch.long),
        None,
        **payload,
        _omni_is_prefill=True,
        _omni_prompt_len=TALKER_PROMPT_LEN,
        _omni_num_computed_tokens=num_computed,
    )

    end = num_computed + span_len
    assert torch.equal(out_ids, full_ids[num_computed:end])
    assert torch.equal(out_embeds, full_embeds[num_computed:end])
    reply_conditioning = payload["hidden_states"]["output"] + _fake_embed_input_ids(
        torch.tensor(payload["ids"]["output"])
    )
    expected_reply = torch.cat([reply_conditioning[1:], model.embed_text_eos_token, model.embed_text_pad_token])
    assert torch.equal(update["embed"]["thinker_reply"], expected_reply)


def test_prefill_tail_row_is_first_thinker_output_plus_bos():
    # The last prompt row conditions on thinker output row 0 plus its text-token
    # embedding and codec BOS; a 1-token tail at offset 7 must receive that row.
    model = _make_prefill_model()
    payload = _thinker_payload()

    out_ids, out_embeds, _ = model.talker_preprocess(
        torch.tensor([5], dtype=torch.long),
        None,
        **payload,
        _omni_is_prefill=True,
        _omni_prompt_len=TALKER_PROMPT_LEN,
        _omni_num_computed_tokens=TALKER_PROMPT_LEN - 1,
    )

    assert out_ids.tolist() == [CODEC_START]
    expected = (
        payload["hidden_states"]["output"][:1]
        + _fake_embed_input_ids(torch.tensor(payload["ids"]["output"][:1]))
        + model.embed_codec_bos_token
    )
    assert torch.equal(out_embeds, expected)


def _original_decode_embeds(model, payload, generated_ids):
    """Embeddings the decode steps produced before preemption, one step at a time."""
    _, _, update = model.talker_preprocess(
        torch.zeros(TALKER_PROMPT_LEN, dtype=torch.long),
        None,
        **payload,
        _omni_is_prefill=True,
        _omni_prompt_len=TALKER_PROMPT_LEN,
        _omni_num_computed_tokens=0,
    )
    rows = []
    for k, token in enumerate(generated_ids):
        _, embeds, _ = model.talker_preprocess(
            torch.tensor([token], dtype=torch.long),
            None,
            embed={**payload["embed"], **update["embed"]},
            _omni_is_prefill=False,
            _omni_prompt_len=TALKER_PROMPT_LEN,
            _omni_num_computed_tokens=TALKER_PROMPT_LEN + k,
        )
        rows.append(embeds)
    return torch.cat(rows)


@pytest.mark.parametrize("num_computed", [0, 5])
def test_replay_span_crossing_prompt_boundary_conditions_generated_suffix(num_computed):
    # After preemption the scheduler resets num_computed_tokens to 0 but keeps
    # the generated tokens, and may schedule the prompt plus those tokens in
    # one span (or, with chunking, a span starting inside the prompt). The
    # prompt rows come from the reconstructed prompt at the offset; the
    # generated suffix must get the same thinker-conditioned embeddings as the
    # original decode steps, which a prompt-only result would leave unset.
    model = _make_prefill_model()
    payload = _thinker_payload()
    full_ids, full_embeds = _full_talker_prompt(model, payload)
    generated = list(range(301, 309))  # Includes EOS and several PAD-conditioned positions.
    span_ids = torch.cat(
        [
            torch.zeros(TALKER_PROMPT_LEN - num_computed, dtype=torch.long),
            torch.tensor(generated, dtype=torch.long),
        ]
    )

    out_ids, out_embeds, _ = model.talker_preprocess(
        span_ids,
        None,
        **payload,
        _omni_is_prefill=True,
        _omni_prompt_len=TALKER_PROMPT_LEN,
        _omni_num_computed_tokens=num_computed,
    )

    num_prompt = TALKER_PROMPT_LEN - num_computed
    assert out_ids.shape[0] == out_embeds.shape[0] == span_ids.shape[0]
    assert torch.equal(out_ids[:num_prompt], full_ids[num_computed:])
    assert torch.equal(out_embeds[:num_prompt], full_embeds[num_computed:])
    assert out_ids[num_prompt:].tolist() == generated
    assert torch.equal(out_embeds[num_prompt:], _original_decode_embeds(model, payload, generated))
    # Check the reference tail rule independently of our sequential decoder.
    num_reply = payload["hidden_states"]["output"].shape[0] - 1
    terminal_ids = torch.tensor(generated[num_reply:])
    terminal_conditioning = torch.cat(
        [model.embed_text_eos_token, model.embed_text_pad_token.expand(len(terminal_ids) - 1, -1)]
    )
    assert torch.equal(
        out_embeds[num_prompt + num_reply :], _fake_embed_input_ids(terminal_ids) + terminal_conditioning
    )


def test_repeated_replay_span_is_identical():
    # A request can be preempted and replayed more than once; each replay of
    # the same span must rebuild identical embeddings.
    model = _make_prefill_model()
    payload = _thinker_payload()
    span_ids = torch.cat([torch.zeros(TALKER_PROMPT_LEN, dtype=torch.long), torch.arange(301, 309)])
    meta = dict(_omni_is_prefill=True, _omni_prompt_len=TALKER_PROMPT_LEN, _omni_num_computed_tokens=0)

    _, first, _ = model.talker_preprocess(span_ids, None, **payload, **meta)
    _, second, _ = model.talker_preprocess(span_ids, None, **payload, **meta)

    assert torch.equal(first, second)


def test_missing_runner_metadata_is_rejected():
    # Every in-tree runner supplies the span positions (#3662); without them
    # the span cannot be assembled correctly, so preprocessing fails loudly.
    model = _make_prefill_model()

    with pytest.raises(ValueError, match="runner metadata"):
        model.talker_preprocess(torch.zeros(TALKER_PROMPT_LEN, dtype=torch.long), None, **_thinker_payload())


def test_prompt_length_mismatch_is_rejected():
    # If the reconstructed Talker prompt and the scheduled prompt disagree in
    # length, slicing by position would silently misalign every row.
    model = _make_prefill_model()

    with pytest.raises(ValueError, match="Reconstructed Talker prompt"):
        model.talker_preprocess(
            torch.zeros(4, dtype=torch.long),
            None,
            **_thinker_payload(),
            _omni_is_prefill=True,
            _omni_prompt_len=TALKER_PROMPT_LEN + 1,
            _omni_num_computed_tokens=0,
        )
