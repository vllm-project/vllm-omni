# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved
"""High-level unit tests for ``Qwen3TTSTalkerForConditionalGenerationNv``.

The full model has heavy dependencies (``Qwen3Model``, ``ParallelLMHead``,
``VocabParallelEmbedding``, etc.) that require a distributed init, so these
tests construct the instance via ``object.__new__`` and inject only the
attributes that ``forward`` / ``compute_logits`` / ``make_omni_output`` /
``postprocess`` actually read.

The interesting behavior under test is the per-step dispatch in
:meth:`Qwen3TTSTalkerForConditionalGenerationNv.forward`:

* **Decode-only batch** — ``_get_decode_idxs`` returns ``(None, 0)`` and the
  code predictor runs on every token.
* **Mixed prefill + decode batch** — only decode positions are routed
  through the code predictor; prefill positions keep the prefill embedding
  produced by ``preprocess``.
* **All-prefill batch** — code predictor is skipped entirely.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    Qwen3TTSPromptEmbedsBuilder,
)
from vllm_omni.model_executor.models.qwen3_tts_nv import qwen3_tts_talker_nv as nv
from vllm_omni.model_executor.models.qwen3_tts_nv.qwen3_tts_talker_nv import (
    Qwen3TTSTalkerForConditionalGenerationNv,
    _dict_to_namespace,
    _get_talker_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


HIDDEN = 8
NUM_CODE_GROUPS = 4
VOCAB_SIZE = 16
MAX_NUM_TOKENS = 16


# ──────────────────────────────────────────────────────────────────────
# Helpers: build a minimal ``Qwen3TTSTalkerForConditionalGenerationNv``
# without running the real ``__init__`` (avoids distributed init).
# ──────────────────────────────────────────────────────────────────────


class _FakeCodePredictor(nn.Module):
    """Stand-in for ``self.code_predictor`` exposing the surface used by forward."""

    def __init__(self) -> None:
        super().__init__()
        self.num_code_groups = NUM_CODE_GROUPS
        # Per-group embedding tables for groups 1..N-1.
        self._group_embeddings = nn.ModuleList([nn.Embedding(VOCAB_SIZE, HIDDEN) for _ in range(NUM_CODE_GROUPS - 1)])
        # Group-0 codec embedding.
        self.codec_embedding = nn.Embedding(VOCAB_SIZE, HIDDEN)
        self.generate_calls: list[dict[str, torch.Tensor]] = []

    def get_group_embeddings(self) -> nn.ModuleList:
        return self._group_embeddings

    def generate_groups_1_15(
        self,
        prev_hidden: torch.Tensor,
        group0_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Record inputs and return a deterministic [seq_len, N-1] code tensor."""
        self.generate_calls.append(
            {
                "prev_hidden": prev_hidden.detach().clone(),
                "group0_tokens": group0_tokens.detach().clone(),
            }
        )
        seq_len = group0_tokens.shape[0]
        # Deterministic codes derived from group0 so we can assert later.
        codes = group0_tokens.view(-1, 1).expand(seq_len, NUM_CODE_GROUPS - 1) % VOCAB_SIZE
        return codes.contiguous()


class _FakeBackbone(nn.Module):
    """Stand-in for ``self.model``: returns the input embeds directly."""

    def __init__(self) -> None:
        super().__init__()
        self.last_call: dict[str, torch.Tensor] | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        self.last_call = {
            "input_ids": input_ids.detach().clone(),
            "inputs_embeds": inputs_embeds.detach().clone(),
        }
        # Returning the embeds preserves whatever forward built into the buffer.
        return inputs_embeds.clone()


def _make_fake_attn_metadata(query_lens: list[int], device: torch.device = torch.device("cpu")) -> SimpleNamespace:
    """Build a fake attn_metadata mimicking the runner's contract."""
    start_loc = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(query_lens), 0).tolist()),
        dtype=torch.long,
        device=device,
    )
    return SimpleNamespace(
        max_query_len=int(max(query_lens)),
        query_start_loc=start_loc,
    )


def _make_fake_forward_context(attn_metadata) -> SimpleNamespace:
    return SimpleNamespace(
        attn_metadata=attn_metadata,
        batch_descriptor=None,
    )


def _make_talker_instance() -> Qwen3TTSTalkerForConditionalGenerationNv:
    """Construct a Talker without running ``__init__`` and inject the
    attributes that the methods under test read."""
    model = object.__new__(Qwen3TTSTalkerForConditionalGenerationNv)
    nn.Module.__init__(model)

    # Persistent scratch buffers.
    model._combined_embeddings = torch.zeros(MAX_NUM_TOKENS, HIDDEN)
    model._out_codes = torch.zeros(MAX_NUM_TOKENS, NUM_CODE_GROUPS, dtype=torch.long)
    model._prev_hidden_buffer = torch.zeros(MAX_NUM_TOKENS, HIDDEN)
    # tts_pad text embedding (a fixed constant, populated from weights at
    # load time; here we set a recognisable bias so we can verify it lands
    # in the assembled decode embedding).
    model._tts_pad_text_embed = torch.full((1, HIDDEN), 0.5)

    # Submodules.
    model.code_predictor = _FakeCodePredictor()
    model.model = _FakeBackbone()

    # vllm_config: only ``compilation_config.cudagraph_mode`` is read by
    # ``_get_decode_idxs``. We set NONE so no padding kicks in; tests for
    # padding cover that branch separately.
    from vllm.config import CUDAGraphMode

    model.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE,
            cudagraph_capture_sizes=[],
        )
    )

    # ``compute_logits`` reads these.
    model.codec_head = nn.Linear(HIDDEN, VOCAB_SIZE, bias=False)
    model.suppress_mask = nn.Parameter(torch.zeros(VOCAB_SIZE, dtype=torch.bool), requires_grad=False)
    model.logits_processor = lambda head, hs: head(hs)

    return model


# ──────────────────────────────────────────────────────────────────────
# Forward dispatch: decode-only vs mixed vs all-prefill
# ──────────────────────────────────────────────────────────────────────


def test_get_decode_idxs_returns_none_when_no_attn_metadata(monkeypatch):
    """Profile / dummy run: code predictor must run on every position."""
    model = _make_talker_instance()
    monkeypatch.setattr(nv, "get_forward_context", lambda: _make_fake_forward_context(None))

    decode_idx, num_req = model._get_decode_idxs()

    assert decode_idx is None
    assert num_req == 0


def test_get_decode_idxs_returns_none_for_decode_only_batch(monkeypatch):
    """Decode-only batch (``max_query_len == 1``): apply everywhere."""
    model = _make_talker_instance()
    attn_md = _make_fake_attn_metadata([1, 1, 1, 1])
    monkeypatch.setattr(nv, "get_forward_context", lambda: _make_fake_forward_context(attn_md))

    decode_idx, num_req = model._get_decode_idxs()

    assert decode_idx is None
    assert num_req == 0


def test_get_decode_idxs_picks_decode_indices_in_mixed_batch(monkeypatch):
    """Mixed batch: decode tokens are at positions 0 (req#0=1 tok) and 5
    (req#2=1 tok). Req#1 is prefill (4 tokens at positions 1..4)."""
    model = _make_talker_instance()
    attn_md = _make_fake_attn_metadata([1, 4, 1])
    monkeypatch.setattr(nv, "get_forward_context", lambda: _make_fake_forward_context(attn_md))

    decode_idx, num_req = model._get_decode_idxs()

    assert num_req == 2
    assert decode_idx.tolist() == [0, 5]


def test_get_decode_idxs_returns_empty_for_all_prefill_batch(monkeypatch):
    """All-prefill batch (no req with query_len == 1)."""
    model = _make_talker_instance()
    attn_md = _make_fake_attn_metadata([3, 4])
    monkeypatch.setattr(nv, "get_forward_context", lambda: _make_fake_forward_context(attn_md))

    decode_idx, num_req = model._get_decode_idxs()

    assert num_req == 0
    assert decode_idx.numel() == 0


def test_forward_decode_only_runs_code_predictor_everywhere(monkeypatch):
    """When ``decode_idx`` is None, the code predictor must be called once
    on the full batch and the assembled decode embedding (codec_emb +
    tts_pad + sum(group_embs)) must be written to every position.
    """
    model = _make_talker_instance()
    monkeypatch.setattr(nv, "get_forward_context", lambda: _make_fake_forward_context(None))

    num_tokens = 3
    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    positions = torch.arange(num_tokens, dtype=torch.long)
    inputs_embeds = torch.zeros(num_tokens, HIDDEN)
    prev_hidden_slot = torch.randn(num_tokens, HIDDEN)
    model._prev_hidden_buffer[:num_tokens].copy_(prev_hidden_slot)

    out = model.forward(
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=None,
        inputs_embeds=inputs_embeds,
    )

    # Code predictor was called on the full batch.
    assert len(model.code_predictor.generate_calls) == 1
    call = model.code_predictor.generate_calls[0]
    torch.testing.assert_close(call["group0_tokens"], input_ids)
    torch.testing.assert_close(call["prev_hidden"], prev_hidden_slot)

    # Output codes: column 0 == input_ids, columns 1..N-1 == fake codes.
    expected_codes_1_15 = input_ids.view(-1, 1).expand(num_tokens, NUM_CODE_GROUPS - 1) % VOCAB_SIZE
    torch.testing.assert_close(model._out_codes[:num_tokens, 0], input_ids)
    torch.testing.assert_close(model._out_codes[:num_tokens, 1:], expected_codes_1_15)

    # Backbone was fed an embedding equal to the analytical decode assembly.
    cp = model.code_predictor
    expected_emb = cp.codec_embedding(input_ids) + model._tts_pad_text_embed
    for i, emb in enumerate(cp.get_group_embeddings()):
        expected_emb = expected_emb + emb(expected_codes_1_15[:, i])

    assert model.model.last_call is not None
    torch.testing.assert_close(model.model.last_call["inputs_embeds"], expected_emb)
    # And forward returned that hidden states (FakeBackbone is identity).
    torch.testing.assert_close(out, expected_emb)


def test_forward_mixed_batch_only_runs_code_predictor_on_decode(monkeypatch):
    """Mixed batch: code predictor runs on decode positions only and the
    prefill positions in ``inputs_embeds`` flow through unchanged."""
    model = _make_talker_instance()

    # 3 reqs: decode (1 tok), prefill (3 tok), decode (1 tok). Decode positions
    # in the flat batch are 0 and 4.
    query_lens = [1, 3, 1]
    num_tokens = sum(query_lens)
    attn_md = _make_fake_attn_metadata(query_lens)
    monkeypatch.setattr(nv, "get_forward_context", lambda: _make_fake_forward_context(attn_md))

    input_ids = torch.tensor([7, 0, 0, 0, 5], dtype=torch.long)
    positions = torch.arange(num_tokens, dtype=torch.long)
    # Distinct prefill marker so we can assert it survives at prefill positions.
    prefill_marker = torch.full((HIDDEN,), -3.0)
    inputs_embeds = torch.zeros(num_tokens, HIDDEN)
    inputs_embeds[1:4] = prefill_marker
    # prev_hidden values for the decode slots.
    prev_hidden = torch.zeros(num_tokens, HIDDEN)
    prev_hidden[0] = torch.full((HIDDEN,), 0.7)
    prev_hidden[4] = torch.full((HIDDEN,), 0.9)
    model._prev_hidden_buffer[:num_tokens].copy_(prev_hidden)

    model.forward(
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=None,
        inputs_embeds=inputs_embeds,
    )

    # Code predictor called exactly once on the decode slice [0, 4].
    assert len(model.code_predictor.generate_calls) == 1
    call = model.code_predictor.generate_calls[0]
    torch.testing.assert_close(call["group0_tokens"], torch.tensor([7, 5], dtype=torch.long))
    torch.testing.assert_close(call["prev_hidden"], prev_hidden[[0, 4]])

    # ``_out_codes`` only has groups 1..N-1 written at the decode rows.
    decode_rows = model._out_codes[[0, 4], 1:]
    expected_decode_codes = torch.tensor([[7, 7, 7], [5, 5, 5]], dtype=torch.long)
    torch.testing.assert_close(decode_rows, expected_decode_codes)

    # Prefill rows for groups 1..N-1 must remain untouched (zero).
    torch.testing.assert_close(
        model._out_codes[1:4, 1:],
        torch.zeros((3, NUM_CODE_GROUPS - 1), dtype=torch.long),
    )

    # input_ids fully written into column 0.
    torch.testing.assert_close(model._out_codes[:num_tokens, 0], input_ids)

    # Backbone embeddings: prefill rows preserved, decode rows replaced by
    # the assembled decode embedding.
    fed = model.model.last_call["inputs_embeds"]
    torch.testing.assert_close(fed[1], prefill_marker)
    torch.testing.assert_close(fed[2], prefill_marker)
    torch.testing.assert_close(fed[3], prefill_marker)

    cp = model.code_predictor
    decode_ids = torch.tensor([7, 5], dtype=torch.long)
    expected_decode_emb = cp.codec_embedding(decode_ids) + model._tts_pad_text_embed
    for i, emb in enumerate(cp.get_group_embeddings()):
        expected_decode_emb = expected_decode_emb + emb(expected_decode_codes[:, i])
    torch.testing.assert_close(fed[[0, 4]], expected_decode_emb)


def test_forward_all_prefill_skips_code_predictor(monkeypatch):
    """All-prefill batch: code predictor must not be called at all."""
    model = _make_talker_instance()
    query_lens = [3, 4]
    num_tokens = sum(query_lens)
    attn_md = _make_fake_attn_metadata(query_lens)
    monkeypatch.setattr(nv, "get_forward_context", lambda: _make_fake_forward_context(attn_md))

    input_ids = torch.zeros(num_tokens, dtype=torch.long)
    inputs_embeds = torch.randn(num_tokens, HIDDEN)
    expected_passthrough = inputs_embeds.clone()

    model.forward(
        input_ids=input_ids,
        positions=torch.arange(num_tokens, dtype=torch.long),
        intermediate_tensors=None,
        inputs_embeds=inputs_embeds,
    )

    # No code predictor call.
    assert model.code_predictor.generate_calls == []
    # Backbone was fed exactly the prefill embeddings.
    torch.testing.assert_close(model.model.last_call["inputs_embeds"], expected_passthrough)
    # Groups 1..N-1 must remain zero (they're never produced for prefill).
    torch.testing.assert_close(
        model._out_codes[:num_tokens, 1:],
        torch.zeros((num_tokens, NUM_CODE_GROUPS - 1), dtype=torch.long),
    )


# ──────────────────────────────────────────────────────────────────────
# make_omni_output / postprocess / compute_logits
# ──────────────────────────────────────────────────────────────────────


def test_make_omni_output_wraps_hidden_and_codes():
    model = _make_talker_instance()

    num_tokens = 5
    hidden = torch.randn(num_tokens, HIDDEN)
    # Pre-populate _out_codes with a recognisable pattern.
    model._out_codes[:num_tokens] = torch.arange(num_tokens * NUM_CODE_GROUPS, dtype=torch.long).view(
        num_tokens, NUM_CODE_GROUPS
    )

    out = model.make_omni_output(hidden)

    assert isinstance(out, OmniOutput)
    torch.testing.assert_close(out.text_hidden_states, hidden)
    assert out.multimodal_outputs is not None
    audio_codes = out.multimodal_outputs["audio_codes"]
    assert audio_codes.shape == (num_tokens, NUM_CODE_GROUPS)
    torch.testing.assert_close(audio_codes, model._out_codes[:num_tokens])


def test_make_omni_output_passes_through_existing_omni_output():
    model = _make_talker_instance()
    existing = OmniOutput(
        text_hidden_states=torch.zeros(1, HIDDEN),
        multimodal_outputs={"audio_codes": torch.zeros(1, NUM_CODE_GROUPS)},
    )
    assert model.make_omni_output(existing) is existing


def test_postprocess_returns_last_hidden():
    model = _make_talker_instance()
    hidden = torch.arange(3 * HIDDEN, dtype=torch.float32).view(3, HIDDEN)
    out = model.postprocess(hidden)
    assert "last_talker_hidden" in out
    torch.testing.assert_close(out["last_talker_hidden"], hidden[-1])


def test_postprocess_empty_hidden_returns_empty_dict():
    model = _make_talker_instance()
    assert model.postprocess(torch.empty(0, HIDDEN)) == {}


def test_compute_logits_applies_suppress_mask():
    model = _make_talker_instance()
    # Set deterministic codec_head weights so we know what logits to expect.
    with torch.no_grad():
        model.codec_head.weight.copy_(torch.eye(VOCAB_SIZE, HIDDEN)[:VOCAB_SIZE, :HIDDEN])
        # Suppress two tokens.
        mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
        mask[3] = True
        mask[7] = True
        model.suppress_mask.data.copy_(mask)

    hidden = torch.randn(2, HIDDEN)
    logits = model.compute_logits(hidden)

    assert logits.shape == (2, VOCAB_SIZE)
    assert torch.isinf(logits[:, 3]).all() and (logits[:, 3] < 0).all()
    assert torch.isinf(logits[:, 7]).all() and (logits[:, 7] < 0).all()
    # Other entries are finite.
    finite_cols = [i for i in range(VOCAB_SIZE) if i not in (3, 7)]
    assert torch.isfinite(logits[:, finite_cols]).all()


def test_compute_logits_unwraps_omni_output():
    model = _make_talker_instance()
    hidden = torch.randn(1, HIDDEN)
    wrapped = OmniOutput(text_hidden_states=hidden)
    direct = model.compute_logits(hidden)
    via_omni = model.compute_logits(wrapped)
    torch.testing.assert_close(via_omni, direct)


# ──────────────────────────────────────────────────────────────────────
# Static helpers
# ──────────────────────────────────────────────────────────────────────


def test_first_str_handles_lists_scalars_and_none():
    f = Qwen3TTSTalkerForConditionalGenerationNv._first_str
    assert f(["hello", "ignored"]) == "hello"
    assert f([]) == ""
    assert f("plain") == "plain"
    assert f(None) == ""
    assert f(42) == "42"


def test_build_assistant_text_layout():
    text = Qwen3TTSTalkerForConditionalGenerationNv._build_assistant_text("hi")
    assert text == "<|im_start|>assistant\nhi<|im_end|>\n<|im_start|>assistant\n"


def _make_tokenizer(token_count: int):
    """Return a fake tokenizer that returns ``token_count`` ints regardless of input."""
    return lambda s: [0] * token_count


# The NV talker no longer wraps ``estimate_prompt_len_from_additional_information``;
# production callers (``runtime.py``, ``serving_speech.py``) and ``preprocess`` now
# invoke :class:`Qwen3TTSPromptEmbedsBuilder` directly. The tests below pin the
# CustomVoice math against the shared builder to guard the prefill-length contract
# the NV talker relies on.


def test_estimate_prompt_len_no_language_id_uses_prefill_3():
    """No language_id -> prefill_len=3, total = 3 + assistant_len - 1."""
    assistant_len = 12
    out = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        {"text": "hi", "speaker": "alice", "language": "Auto"},
        task_type="CustomVoice",
        tokenize_prompt=_make_tokenizer(assistant_len),
        codec_language_id={"english": 1},
        spk_is_dialect=None,
    )
    assert out == 3 + assistant_len - 1


def test_estimate_prompt_len_with_language_id_uses_prefill_4():
    """Resolved language_id -> prefill_len=4."""
    assistant_len = 12
    out = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        {"text": "hi", "speaker": "alice", "language": "English"},
        task_type="CustomVoice",
        tokenize_prompt=_make_tokenizer(assistant_len),
        codec_language_id={"english": 1},
        spk_is_dialect=None,
    )
    assert out == 4 + assistant_len - 1


def test_estimate_prompt_len_dialect_fallback_promotes_to_4():
    """Auto language + speaker registered as a dialect resolves a language_id."""
    assistant_len = 12
    out = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        {"text": "hi", "speaker": "shanghainese_voice", "language": "Auto"},
        task_type="CustomVoice",
        tokenize_prompt=_make_tokenizer(assistant_len),
        codec_language_id={"shanghainese": 7},
        spk_is_dialect={"shanghainese_voice": "shanghainese"},
    )
    assert out == 4 + assistant_len - 1


def test_estimate_prompt_len_unwraps_list_values():
    assistant_len = 12
    out = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        {
            "text": ["hi"],
            "speaker": ["alice"],
            "language": ["Auto"],
        },
        task_type="CustomVoice",
        tokenize_prompt=_make_tokenizer(assistant_len),
        codec_language_id={"english": 1},
        spk_is_dialect=None,
    )
    assert out == 3 + assistant_len - 1


def test_estimate_prompt_len_short_assistant_raises():
    with pytest.raises(ValueError, match="assistant prompt length"):
        Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
            {"text": "hi", "speaker": "alice"},
            task_type="CustomVoice",
            tokenize_prompt=_make_tokenizer(5),
            codec_language_id=None,
            spk_is_dialect=None,
        )


# ──────────────────────────────────────────────────────────────────────
# Internal helpers _dict_to_namespace / _get_talker_config
# ──────────────────────────────────────────────────────────────────────


def test_dict_to_namespace_recurses_but_keeps_rope_scaling_as_dict():
    src = {
        "hidden_size": 32,
        "rope_scaling": {"rope_type": "yarn", "factor": 4.0},
        "nested": {"a": 1},
    }
    ns = _dict_to_namespace(src)
    assert ns.hidden_size == 32
    # rope_scaling is preserved as a dict (downstream expects dict-like).
    assert isinstance(ns.rope_scaling, dict)
    assert ns.rope_scaling == {"rope_type": "yarn", "factor": 4.0}
    # Other nested dicts get converted.
    assert ns.nested.a == 1


def test_get_talker_config_with_full_config_returns_talker_field():
    talker_dict = {"hidden_size": 32, "vocab_size": 16}
    full = SimpleNamespace(talker_config=talker_dict)
    out = _get_talker_config(full)
    assert out.hidden_size == 32
    assert out.vocab_size == 16


def test_get_talker_config_with_already_talker_config_returns_unchanged():
    talker_cfg = SimpleNamespace(hidden_size=32)
    out = _get_talker_config(talker_cfg)
    assert out is talker_cfg
