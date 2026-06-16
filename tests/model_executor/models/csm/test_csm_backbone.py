# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU correctness tests for the CSM-1B Stage-0 backbone control logic.

These lock the three places the 2-stage redesign is subtle, all reachable on
CPU by building the model via ``object.__new__`` + stubbing the vLLM-native
backbone / depth wrapper:

  * ``compute_logits`` EOS row-mapping (the "unknown #2" fix): EOS is forced
    POSITIONALLY on ``_eos_flags_by_row`` rows, never by dict insertion order.
  * ``forward`` EOS / GATE-B frame-cap emit policy: a natural all-zero EOS frame
    is DROPPED (empty latent) while a cap-forced stop KEEPS its real audio frame;
    both latch the row so the scheduler stops.
  * I5 per-request state: ``preprocess`` decode re-injects the cached Sigma-embed,
    and ``on_requests_finished`` frees every per-request dict.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.csm.csm_backbone import (
    _CODEBOOK_EOS_ID,
    CsmBackboneForConditionalGeneration,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_VOCAB = 8
_HIDDEN = 8


class _FakeLogitsProcessor:
    """Deterministic per-row logits so EOS overrides are checkable. Row i gets a
    strictly-increasing ramp offset by i, so its natural argmax is the last id
    (never id 0), making a forced-EOS (argmax -> 0) unambiguous."""

    def __call__(self, head, hidden):
        n = int(hidden.shape[0])
        return torch.arange(n * _VOCAB, dtype=torch.float32).reshape(n, _VOCAB)


def _make_backbone() -> CsmBackboneForConditionalGeneration:
    m = object.__new__(CsmBackboneForConditionalGeneration)
    nn.Module.__init__(m)
    m.num_codebooks = 32
    m.hidden_size = _HIDDEN
    m._backbone_dtype = torch.float32
    m._dtype = torch.float32
    m._eos_flags_by_row = []
    m._eos_by_req = {}
    m._cached_sigma_by_req = {}
    m._sampling_by_req = {}
    m._max_frames_by_req = {}
    m._frames_emitted_by_req = {}
    m.backbone = SimpleNamespace(logits_processor=_FakeLogitsProcessor(), cb0_head=object())
    return m


# --------------------------------------------------------------------------
# compute_logits: positional EOS row mapping
# --------------------------------------------------------------------------


def test_compute_logits_forces_eos_on_flagged_rows_only():
    m = _make_backbone()
    m._eos_flags_by_row = [False, True, False]
    logits = m.compute_logits(torch.randn(3, _HIDDEN))

    assert logits.shape == (3, _VOCAB)
    # Flagged row 1 -> all -inf except the EOS id, set to a huge positive.
    assert int(logits[1].argmax()) == _CODEBOOK_EOS_ID
    assert logits[1, _CODEBOOK_EOS_ID].item() == pytest.approx(1.0e6)
    others = torch.cat([logits[1, :_CODEBOOK_EOS_ID], logits[1, _CODEBOOK_EOS_ID + 1 :]])
    assert torch.isneginf(others).all()
    # Unflagged rows are untouched (finite ramp, argmax != EOS id).
    assert torch.isfinite(logits[0]).all() and int(logits[0].argmax()) != _CODEBOOK_EOS_ID
    assert torch.isfinite(logits[2]).all() and int(logits[2].argmax()) != _CODEBOOK_EOS_ID


def test_compute_logits_none_hidden_returns_none():
    assert _make_backbone().compute_logits(None) is None


def test_compute_logits_unsqueezes_1d_hidden():
    out = _make_backbone().compute_logits(torch.randn(_HIDDEN))
    assert out.shape == (1, _VOCAB)


def test_compute_logits_accepts_omni_output():
    m = _make_backbone()
    m._eos_flags_by_row = [False, False]
    oo = OmniOutput(text_hidden_states=torch.randn(2, _HIDDEN), multimodal_outputs=None)
    assert m.compute_logits(oo).shape == (2, _VOCAB)


def test_compute_logits_tolerates_flags_shorter_than_batch():
    # Stale/short flag list must not crash or touch unflagged rows.
    m = _make_backbone()
    m._eos_flags_by_row = [True]  # only row 0
    logits = m.compute_logits(torch.randn(3, _HIDDEN))
    assert int(logits[0].argmax()) == _CODEBOOK_EOS_ID
    assert torch.isfinite(logits[1]).all()
    assert torch.isfinite(logits[2]).all()


# --------------------------------------------------------------------------
# forward: EOS / GATE-B frame-cap emit policy
# --------------------------------------------------------------------------


def _drive_forward(monkeypatch, frame, *, cap, emitted=0):
    m = _make_backbone()
    m._sampling_by_req = {"r0": (0.0, 0)}  # greedy -> deterministic cb0
    m._max_frames_by_req = {"r0": cap}
    m._frames_emitted_by_req = {"r0": emitted}
    m.backbone.forward = lambda **kw: torch.randn(1, _HIDDEN)
    m.depth = SimpleNamespace(run=lambda **kw: frame)
    m._compose_frame_embed = lambda fc: torch.zeros(1, _HIDDEN)
    monkeypatch.setattr(
        "vllm.forward_context.get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    out = m.forward(
        input_ids=torch.tensor([5], dtype=torch.long),
        positions=torch.tensor([0]),
        inputs_embeds=torch.zeros(1, _HIDDEN),
        runtime_additional_information=[{"request_id": "r0"}],
    )
    return m, out


def test_forward_natural_eos_drops_frame_and_flags_row(monkeypatch):
    frame = torch.zeros(1, 32, dtype=torch.long)  # cb0..cb30 == 0 -> natural EOS
    m, out = _drive_forward(monkeypatch, frame, cap=100)
    codes = out.multimodal_outputs["codes"]["audio"]
    assert codes[0].shape == (0, 32)  # all-zero EOS frame is dropped
    assert m._eos_flags_by_row == [True]
    assert m._eos_by_req["r0"] is True


def test_forward_eos_ignores_codebook31(monkeypatch):
    # EOS is cb0..cb30 all-zero; a nonzero cb31 must NOT defeat the EOS check.
    frame = torch.zeros(1, 32, dtype=torch.long)
    frame[0, 31] = 9
    m, out = _drive_forward(monkeypatch, frame, cap=100)
    assert m._eos_flags_by_row == [True]
    assert out.multimodal_outputs["codes"]["audio"][0].shape == (0, 32)


def test_forward_cap_forced_keeps_real_frame_and_flags_row(monkeypatch):
    frame = torch.ones(1, 32, dtype=torch.long)  # real audio, no natural EOS
    m, out = _drive_forward(monkeypatch, frame, cap=1, emitted=0)
    codes = out.multimodal_outputs["codes"]["audio"]
    assert codes[0].shape == (1, 32)  # cap-forced stop KEEPS the final frame
    assert torch.equal(codes[0], frame)
    assert m._eos_flags_by_row == [True]  # but still stops the scheduler
    assert m._frames_emitted_by_req["r0"] == 1


def test_forward_normal_frame_is_kept_and_not_flagged(monkeypatch):
    frame = torch.ones(1, 32, dtype=torch.long)
    m, out = _drive_forward(monkeypatch, frame, cap=100, emitted=0)
    codes = out.multimodal_outputs["codes"]["audio"]
    assert codes[0].shape == (1, 32)
    assert m._eos_flags_by_row == [False]
    assert m._eos_by_req["r0"] is False
    # Sigma cached for the next step's feedback (I5), frame counter advanced.
    assert "r0" in m._cached_sigma_by_req
    assert m._frames_emitted_by_req["r0"] == 1


def test_forward_returns_omni_output_with_codes_latent(monkeypatch):
    m, out = _drive_forward(monkeypatch, torch.ones(1, 32, dtype=torch.long), cap=100)
    assert isinstance(out, OmniOutput)
    assert "codes" in out.multimodal_outputs
    assert "audio" in out.multimodal_outputs["codes"]


# --------------------------------------------------------------------------
# I5 per-request state: preprocess Sigma re-inject + cleanup
# --------------------------------------------------------------------------


def test_preprocess_decode_adds_cached_sigma_to_base_embed():
    m = _make_backbone()
    m.config = SimpleNamespace(vocab_size=2051)
    m._compose_frame_embed = lambda fc: torch.full((1, _HIDDEN), 2.0)
    m._cached_sigma_by_req = {"r0": torch.full((1, _HIDDEN), 5.0)}
    ids, embeds, upd = m.preprocess(
        input_ids=torch.tensor([7], dtype=torch.long),
        input_embeds=None,
        request_id="r0",
        _omni_is_prefill=False,
    )
    assert torch.equal(ids, torch.tensor([7]))
    # base (2.0) + cached Sigma (5.0) == 7.0 elementwise.
    torch.testing.assert_close(embeds, torch.full((1, _HIDDEN), 7.0))
    assert upd == {}


def test_preprocess_decode_without_cache_uses_base_embed_only():
    m = _make_backbone()
    m.config = SimpleNamespace(vocab_size=2051)
    m._compose_frame_embed = lambda fc: torch.full((1, _HIDDEN), 2.0)
    m._cached_sigma_by_req = {}
    _, embeds, _ = m.preprocess(
        input_ids=torch.tensor([7], dtype=torch.long),
        input_embeds=None,
        request_id="r0",
        _omni_is_prefill=False,
    )
    torch.testing.assert_close(embeds, torch.full((1, _HIDDEN), 2.0))


def test_preprocess_prefill_returns_text_prompt_span():
    m = _make_backbone()
    m.config = SimpleNamespace(vocab_size=2051)
    prompt = torch.arange(3 * _HIDDEN, dtype=torch.float32).reshape(3, _HIDDEN)
    m._embed_text_prompt = lambda info, device: prompt
    ids, embeds, upd = m.preprocess(
        input_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        input_embeds=None,
        request_id="rp",
        _omni_is_prefill=True,
        _omni_num_computed_tokens=0,
        _omni_prompt_len=3,
    )
    assert embeds.shape == (3, _HIDDEN)
    torch.testing.assert_close(embeds, prompt)
    assert upd == {}


def test_on_requests_finished_frees_all_per_request_state():
    m = _make_backbone()
    m._cached_sigma_by_req = {"r": torch.zeros(1)}
    m._eos_by_req = {"r": True}
    m._sampling_by_req = {"r": (0.9, 50)}
    m._max_frames_by_req = {"r": 64}
    m._frames_emitted_by_req = {"r": 5}
    m.on_requests_finished(["r"])
    assert m._cached_sigma_by_req == {}
    assert m._eos_by_req == {}
    assert m._sampling_by_req == {}
    assert m._max_frames_by_req == {}
    assert m._frames_emitted_by_req == {}
