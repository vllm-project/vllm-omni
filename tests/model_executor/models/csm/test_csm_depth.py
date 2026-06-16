# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU correctness tests for the CSM-1B depth decoder (Stage-0 inner AR loop).

Two surfaces, both CPU-only:
  * ``sample_logits`` numerics -- the b3 fix (fp32 + nan_to_num before any
    softmax/argmax) and the greedy / top-k contract.
  * ``CsmDepthDecoder.run`` loop wiring -- the 31-step seeding contract (step 0
    gets the padded ``(B, 2)`` ids + ``backbone_last_hidden_state``; steps 1..30
    get ``(B, 1)`` and no hidden), cb0 passthrough, output shape/dtype, and
    determinism. Bit-parity against the real HF module (with checkpoint weights)
    lives in ``test_csm_gpu_parity.py``; here we lock the loop logic with a
    deterministic fake so it runs without weights or a GPU.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.csm.csm_depth import CsmDepthDecoder, sample_logits

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# --------------------------------------------------------------------------
# sample_logits
# --------------------------------------------------------------------------


def test_sample_logits_greedy_is_argmax():
    logits = torch.tensor([[0.1, 5.0, 0.2], [9.0, 0.0, 1.0]])
    out = sample_logits(logits, temperature=0.0, top_k=0)
    assert out.tolist() == [1, 0]
    assert out.dtype == torch.long
    assert out.shape == (2,)


def test_sample_logits_none_or_negative_temperature_is_greedy():
    logits = torch.tensor([[0.1, 0.2, 9.0]])
    assert sample_logits(logits, temperature=None, top_k=0).tolist() == [2]
    assert sample_logits(logits, temperature=-1.0, top_k=0).tolist() == [2]


def test_sample_logits_sanitizes_nonfinite_without_assert():
    # The bf16 backbone can emit +/-inf or NaN; nan_to_num must clamp them so
    # multinomial never hits its device-side "probability tensor" assert.
    logits = torch.tensor([[float("nan"), float("inf"), float("-inf"), 0.0]])
    out = sample_logits(logits, temperature=0.9, top_k=0)
    # posinf -> 1e4 dominates the softmax; others underflow to ~0.
    assert out.tolist() == [1]
    assert torch.isfinite(out.float()).all()


def test_sample_logits_topk1_collapses_to_the_top_token():
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.5]])
    # top_k=1 masks all but the max to -inf, so the sample is deterministic.
    assert sample_logits(logits, temperature=1.0, top_k=1).tolist() == [2]


def test_sample_logits_topk_ge_vocab_is_a_noop_mask():
    # top_k >= vocab must not mask anything (the `top_k < shape[-1]` guard).
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    assert sample_logits(logits, temperature=1.0, top_k=999).tolist() == [0]


# --------------------------------------------------------------------------
# CsmDepthDecoder.run -- 31-step loop wiring
# --------------------------------------------------------------------------


class _FakeDepthOut:
    def __init__(self, logits, past):
        self.logits = logits
        self.past_key_values = past


class _FakeDepthModule:
    """Deterministic stand-in for ``CsmDepthDecoderForCausalLM``.

    Records each call so the seeding contract is assertable, and emits logits
    whose argmax is a fixed function of the step index, so greedy decode is
    reproducible. ``config`` is consumed only by the (patched) ``DynamicCache``.
    """

    def __init__(self, vocab: int = 2051):
        self.vocab = vocab
        self.calls: list[dict] = []
        self.config = SimpleNamespace()

    def __call__(self, *, input_ids, past_key_values, use_cache, logits_to_keep, backbone_last_hidden_state=None):
        self.calls.append(
            {
                "input_shape": tuple(input_ids.shape),
                "has_hidden": backbone_last_hidden_state is not None,
                "use_cache": use_cache,
                "logits_to_keep": logits_to_keep,
            }
        )
        bsz = int(input_ids.shape[0])
        step = len(self.calls) - 1  # 0-based depth step
        logits = torch.full((bsz, 1, self.vocab), -10.0)
        logits[:, -1, (step + 1) % self.vocab] = 100.0  # argmax -> step+1
        return _FakeDepthOut(logits, past_key_values)


@pytest.fixture
def _patch_cache(monkeypatch):
    # Isolate the loop from transformers' real DynamicCache construction.
    monkeypatch.setattr("transformers.cache_utils.DynamicCache", lambda config=None: object())


def _make_depth(fake: _FakeDepthModule, num_codebooks: int = 32) -> CsmDepthDecoder:
    depth = CsmDepthDecoder(num_codebooks=num_codebooks, hidden_size=2048, aux_dtype=torch.float32)
    depth.set_module(fake)
    return depth


def test_run_returns_32_long_codes_with_cb0_passthrough(_patch_cache):
    fake = _FakeDepthModule()
    depth = _make_depth(fake)
    cb0 = torch.tensor([7], dtype=torch.long)
    codes = depth.run(
        cb0=cb0,
        backbone_last_hidden_state=torch.randn(1, 2048),
        temperature=0.0,  # greedy -> deterministic
        top_k=0,
    )
    assert codes.shape == (1, 32)
    assert codes.dtype == torch.long
    # cb0 is the backbone's token, copied verbatim into slot 0.
    assert int(codes[0, 0]) == 7
    # Greedy argmax of the fake = step+1, so cb1..cb31 == 1..31.
    assert codes[0, 1:].tolist() == list(range(1, 32))


def test_run_seeds_step0_with_padded_ids_and_hidden_state(_patch_cache):
    fake = _FakeDepthModule()
    depth = _make_depth(fake)
    depth.run(
        cb0=torch.tensor([3], dtype=torch.long),
        backbone_last_hidden_state=torch.randn(1, 2048),
        temperature=0.0,
        top_k=0,
    )
    # 31 inner steps for 32 codebooks.
    assert len(fake.calls) == 31
    # Step 0: input padded to (B, 2) AND the backbone hidden state is supplied.
    assert fake.calls[0]["input_shape"] == (1, 2)
    assert fake.calls[0]["has_hidden"] is True
    # Steps 1..30: single-token (B, 1) input, NO hidden state re-supplied.
    for c in fake.calls[1:]:
        assert c["input_shape"] == (1, 1)
        assert c["has_hidden"] is False
        assert c["logits_to_keep"] == 1
        assert c["use_cache"] is True


def test_run_handles_batch_dim(_patch_cache):
    fake = _FakeDepthModule()
    depth = _make_depth(fake)
    cb0 = torch.tensor([5, 9], dtype=torch.long)
    codes = depth.run(
        cb0=cb0,
        backbone_last_hidden_state=torch.randn(2, 2048),
        temperature=0.0,
        top_k=0,
    )
    assert codes.shape == (2, 32)
    assert codes[:, 0].tolist() == [5, 9]
    assert fake.calls[0]["input_shape"] == (2, 2)


def test_run_is_deterministic_under_greedy(_patch_cache):
    cb0 = torch.tensor([11], dtype=torch.long)
    hs = torch.randn(1, 2048)
    a = _make_depth(_FakeDepthModule()).run(cb0=cb0, backbone_last_hidden_state=hs, temperature=0.0, top_k=0)
    b = _make_depth(_FakeDepthModule()).run(cb0=cb0, backbone_last_hidden_state=hs, temperature=0.0, top_k=0)
    assert torch.equal(a, b)
