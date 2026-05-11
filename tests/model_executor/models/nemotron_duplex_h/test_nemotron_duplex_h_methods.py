# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimalistic CPU unit tests for ``NemotronDuplexHForCausalLM``.

The full constructor requires building a NemotronH backbone (heavy and
GPU-only on practical configs), so these tests use the
``object.__new__`` shortcut and stub the embed_tokens / embed_asr_tokens
modules with plain :class:`torch.nn.Embedding` instances.

The class layer we exercise is its three vLLM-Omni hooks:

* ``preprocess`` (prefill short-circuit + decode embedding sum).
* ``postprocess`` (storage-offset-driven last-row pickup).
* ``forward`` (thin pass-through of ``self.model(...)``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.nemotron_duplex_h.nemotron_duplex_h import (
    NemotronDuplexHForCausalLM,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Helpers — build a ``NemotronDuplexHForCausalLM`` shell suitable for the
# hooks under test.
# ---------------------------------------------------------------------------


def _make_model_shell(
    *,
    vocab_size: int = 16,
    hidden_size: int = 4,
    dtype: torch.dtype = torch.float32,
) -> NemotronDuplexHForCausalLM:
    """Bypass ``__init__`` so we don't need to construct a full NemotronH backbone."""
    model = object.__new__(NemotronDuplexHForCausalLM)
    nn.Module.__init__(model)

    embed_tokens = nn.Embedding(vocab_size, hidden_size).to(dtype)
    embed_asr_tokens = nn.Embedding(vocab_size, hidden_size).to(dtype)
    # ``self.model`` only needs ``.embed_tokens`` for ``preprocess``.
    model.model = SimpleNamespace(embed_tokens=embed_tokens)
    model.embed_asr_tokens = embed_asr_tokens
    return model


def _stub_forward_pass_through(model: NemotronDuplexHForCausalLM) -> None:
    """Replace ``model.model`` with a callable that returns ``inputs_embeds``."""

    def _passthrough(input_ids, positions, intermediate_tensors, inputs_embeds):
        # Mirror the streaming-path contract: backbone consumes
        # pre-computed embeddings; ``input_ids`` is just informational.
        return inputs_embeds

    backbone = SimpleNamespace(
        embed_tokens=model.model.embed_tokens,
        __call__=_passthrough,
    )

    # SimpleNamespace isn't callable; wrap into an object with ``__call__``.
    class _Backbone:
        def __init__(self, embed_tokens):
            self.embed_tokens = embed_tokens

        def __call__(self, input_ids, positions, intermediate_tensors, inputs_embeds):
            return _passthrough(input_ids, positions, intermediate_tensors, inputs_embeds)

    model.model = _Backbone(backbone.embed_tokens)


# ---------------------------------------------------------------------------
# preprocess — prefill short-circuit
# ---------------------------------------------------------------------------


class TestPreprocessPrefill:
    def test_returns_combined_tensor_and_clears_buffer(self) -> None:
        model = _make_model_shell(hidden_size=4)
        n = 5
        prefill = torch.randn(n, 4)
        input_ids = torch.zeros(n, dtype=torch.long)

        ids_out, embeds_out, info_update = model.preprocess(
            input_ids,
            None,
            prefill_combined_embeddings=prefill,
        )

        # The hook hands the pre-computed tensor straight to the backbone.
        assert ids_out is input_ids
        assert embeds_out.shape == (n, 4)
        assert torch.allclose(embeds_out, prefill)
        # And clears the buffer so the next call falls through to the
        # decode branch (this is the contract documented at the top of
        # nemotron_duplex_h.preprocess).
        assert info_update == {"prefill_combined_embeddings": None}

    def test_dtype_is_cast_to_embed_tokens_dtype(self) -> None:
        model = _make_model_shell(hidden_size=4, dtype=torch.float16)
        input_ids = torch.zeros(2, dtype=torch.long)
        prefill = torch.randn(2, 4, dtype=torch.float32)

        _, embeds_out, _ = model.preprocess(
            input_ids,
            None,
            prefill_combined_embeddings=prefill,
        )
        assert embeds_out.dtype == torch.float16

    def test_rejects_mismatched_length(self) -> None:
        model = _make_model_shell(hidden_size=4)
        input_ids = torch.zeros(3, dtype=torch.long)
        prefill = torch.randn(5, 4)  # wrong length
        with pytest.raises(AssertionError, match="does not match scheduled token count"):
            model.preprocess(input_ids, None, prefill_combined_embeddings=prefill)

    def test_rejects_non_2d_prefill(self) -> None:
        model = _make_model_shell(hidden_size=4)
        input_ids = torch.zeros(2, dtype=torch.long)
        prefill = torch.randn(2, 4, 1)  # 3D
        with pytest.raises(AssertionError, match="must be 2D"):
            model.preprocess(input_ids, None, prefill_combined_embeddings=prefill)


# ---------------------------------------------------------------------------
# preprocess — decode branch (sum of text + asr + acoustic embeddings)
# ---------------------------------------------------------------------------


class TestPreprocessDecode:
    def test_decode_sums_text_asr_acoustic(self) -> None:
        model = _make_model_shell(vocab_size=16, hidden_size=4)
        # Make the embedding tables non-zero & known.
        with torch.no_grad():
            model.model.embed_tokens.weight.zero_()
            model.embed_asr_tokens.weight.zero_()
            model.model.embed_tokens.weight[3] = torch.tensor([1.0, 2.0, 3.0, 4.0])
            model.embed_asr_tokens.weight[7] = torch.tensor([10.0, 20.0, 30.0, 40.0])

        input_ids = torch.tensor([3], dtype=torch.long)
        input_asr_ids = torch.tensor([7], dtype=torch.long)
        acoustic = torch.tensor([[100.0, 200.0, 300.0, 400.0]])

        ids_out, embeds_out, info_update = model.preprocess(
            input_ids,
            None,
            input_asr_ids=input_asr_ids,
            acoustic_embedding=acoustic,
        )

        expected = torch.tensor([[111.0, 222.0, 333.0, 444.0]])
        assert torch.allclose(embeds_out, expected)
        assert ids_out is input_ids
        # Decode path returns an empty info_update (autoregressive ASR
        # feedback is owned by ``postprocess``).
        assert info_update == {}

    def test_decode_rejects_missing_asr_ids(self) -> None:
        model = _make_model_shell(vocab_size=8, hidden_size=4)
        input_ids = torch.tensor([1, 2], dtype=torch.long)
        acoustic = torch.zeros(2, 4)
        with pytest.raises(AssertionError, match="input_asr_ids is not a tensor"):
            model.preprocess(
                input_ids,
                None,
                acoustic_embedding=acoustic,
            )

    def test_decode_rejects_mismatched_asr_length(self) -> None:
        model = _make_model_shell(vocab_size=8, hidden_size=4)
        input_ids = torch.tensor([1, 2], dtype=torch.long)
        # Wrong length: 3 ASR ids for 2 input ids.
        input_asr_ids = torch.tensor([0, 0, 0], dtype=torch.long)
        acoustic = torch.zeros(2, 4)
        with pytest.raises(AssertionError, match="does not match scheduled token count"):
            model.preprocess(
                input_ids,
                None,
                input_asr_ids=input_asr_ids,
                acoustic_embedding=acoustic,
            )

    def test_decode_rejects_missing_acoustic(self) -> None:
        model = _make_model_shell(vocab_size=8, hidden_size=4)
        input_ids = torch.tensor([1], dtype=torch.long)
        input_asr_ids = torch.tensor([0], dtype=torch.long)
        with pytest.raises(AssertionError, match="acoustic_embedding is required"):
            model.preprocess(
                input_ids,
                None,
                input_asr_ids=input_asr_ids,
            )

    def test_decode_rejects_mismatched_acoustic_length(self) -> None:
        model = _make_model_shell(vocab_size=8, hidden_size=4)
        input_ids = torch.tensor([1, 2], dtype=torch.long)
        input_asr_ids = torch.tensor([0, 0], dtype=torch.long)
        # Wrong length: 3 acoustic rows for 2 input ids.
        acoustic = torch.zeros(3, 4)
        with pytest.raises(AssertionError, match="acoustic_embedding length"):
            model.preprocess(
                input_ids,
                None,
                input_asr_ids=input_asr_ids,
                acoustic_embedding=acoustic,
            )


# ---------------------------------------------------------------------------
# postprocess — picks the last asr id of this request's slice.
# ---------------------------------------------------------------------------


class TestPostprocess:
    def test_postprocess_picks_last_asr_token_of_slice(self) -> None:
        model = _make_model_shell()

        # Simulate a flat-batch hidden_states of 5 rows belonging to
        # request whose slice spans rows [2:5]. Pick row 4 (= 2 + 3 - 1).
        flat_hidden = torch.arange(5 * 3, dtype=torch.float32).view(5, 3)
        request_slice = flat_hidden[2:5]
        asr_tokens = torch.tensor([10, 20, 30, 40, 50], dtype=torch.long)

        out = model.postprocess(
            request_slice,
            multimodal_outputs={"asr_tokens": asr_tokens},
        )
        assert set(out.keys()) == {"input_asr_ids"}
        assert out["input_asr_ids"].tolist() == [50]
        assert out["input_asr_ids"].dtype == torch.long

    def test_postprocess_first_request_slice(self) -> None:
        model = _make_model_shell()
        flat_hidden = torch.arange(4 * 3, dtype=torch.float32).view(4, 3)
        request_slice = flat_hidden[0:2]
        asr_tokens = torch.tensor([100, 200, 300, 400], dtype=torch.long)

        out = model.postprocess(
            request_slice,
            multimodal_outputs={"asr_tokens": asr_tokens},
        )
        assert out["input_asr_ids"].tolist() == [200]

    def test_postprocess_requires_multimodal_outputs(self) -> None:
        model = _make_model_shell()
        hidden = torch.zeros(2, 3)
        with pytest.raises(AssertionError):
            model.postprocess(hidden, multimodal_outputs=None)

    def test_postprocess_requires_asr_tokens_tensor(self) -> None:
        model = _make_model_shell()
        hidden = torch.zeros(2, 3)
        with pytest.raises(AssertionError):
            model.postprocess(hidden, multimodal_outputs={"foo": "bar"})


# ---------------------------------------------------------------------------
# Class-level metadata — these flags drive runner dispatch & buffer plumbing,
# so a tiny regression check is worth having.
# ---------------------------------------------------------------------------


def test_class_level_omni_hooks_enabled() -> None:
    assert NemotronDuplexHForCausalLM.has_preprocess is True
    assert NemotronDuplexHForCausalLM.has_postprocess is True
    assert NemotronDuplexHForCausalLM.have_multimodal_outputs is True
    assert "input_asr_ids" in NemotronDuplexHForCausalLM.gpu_resident_buffer_keys
