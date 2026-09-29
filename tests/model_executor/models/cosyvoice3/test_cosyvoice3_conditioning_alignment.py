# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for cross-request voice mixing under concurrency (#8235).

Scenario: the scheduler batch has ``N`` requests in a step, but only ``M`` of
them (M < N) carried mm conditioning kwargs this step. The collated inputs are
split into per-request lists of length ``M`` by ``_split_prompt_conditioning``,
while ``to_payload_element`` routes payloads by the request's index ``idx`` in
the full scheduled batch ``[0, N)``. A shorter list falls back to ``element[0]``
and hands one request's reference voice to the rest -- the aliasing bug. The fix
(``_align_conditioning_to_batch``) pads the lists to ``N`` with ``None`` so out
of range indices take the skip path instead of aliasing request 0.
"""

from __future__ import annotations

import functools

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@functools.lru_cache(maxsize=1)
def _conditioning_symbols():
    """Defer the CosyVoice3 import (pulls onnxruntime/vllm) until first use."""
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model
    from vllm_omni.utils.mm_outputs import to_payload_element

    return CosyVoice3Model, to_payload_element


def _collated_batch(n_reqs: int, t: int, f: int, d: int):
    """Build distinct per-request conditioning so aliasing is detectable.

    Request ``i``'s tensors are filled with ``i + 1``, so any ``element[0]``
    fallback for ``idx > 0`` is observable by value.
    """
    speech_token = torch.cat([torch.full((1, t), float(i + 1)) for i in range(n_reqs)], dim=0)
    speech_feat = torch.cat([torch.full((1, 2 * t, f), float(i + 1)) for i in range(n_reqs)], dim=0)
    embedding = torch.cat([torch.full((1, d), float(i + 1)) for i in range(n_reqs)], dim=0)
    speech_token_len = torch.cat([torch.full((1, 1), t) for _ in range(n_reqs)], dim=0)
    return speech_token, speech_feat, embedding, speech_token_len


def test_split_returns_only_mm_carrying_requests():
    """The bug premise: collated inputs hold M < N rows, so lists are length M."""
    CosyVoice3Model, _ = _conditioning_symbols()

    speech_token, speech_feat, embedding, speech_token_len = _collated_batch(n_reqs=2, t=4, f=80, d=256)
    st_list, sf_list, emb_list, stl_list = CosyVoice3Model._split_prompt_conditioning(
        speech_token, speech_feat, embedding, speech_token_len
    )
    assert len(st_list) == len(sf_list) == len(emb_list) == len(stl_list) == 2


def test_un_aligned_short_list_aliases_request_zero():
    """Root-cause check: idx >= len(element) silently falls back to element[0].

    With M=2 conditioning in a batch of N=3, request 2 would receive request
    0's reference voice -- the #8235 symptom, reproduced here on CPU.
    """
    CosyVoice3Model, to_payload_element = _conditioning_symbols()

    speech_token, speech_feat, embedding, speech_token_len = _collated_batch(n_reqs=2, t=4, f=80, d=256)
    _, _, emb_list, _ = CosyVoice3Model._split_prompt_conditioning(
        speech_token, speech_feat, embedding, speech_token_len
    )

    out = to_payload_element(element=emb_list, idx=2, start=0, end=1)
    assert isinstance(out, torch.Tensor)
    # Request 2 would silently receive request 0's embedding (all 1s).
    assert bool((out[0] == 1.0).all())


def test_align_pads_to_full_batch():
    CosyVoice3Model, _ = _conditioning_symbols()

    speech_token, speech_feat, embedding, speech_token_len = _collated_batch(n_reqs=2, t=4, f=80, d=256)
    lists = CosyVoice3Model._split_prompt_conditioning(speech_token, speech_feat, embedding, speech_token_len)

    aligned = CosyVoice3Model._align_conditioning_to_batch(*lists, batch_size=3)
    assert all(len(lst) == 3 for lst in aligned)
    # Padding slots are None; real slots keep their tensors.
    assert aligned[0][2] is None and aligned[1][2] is None
    assert aligned[2][2] is None and aligned[3][2] is None
    assert isinstance(aligned[0][0], torch.Tensor)
    assert isinstance(aligned[2][1], torch.Tensor)


def test_aligned_payload_routes_each_request_its_own_voice():
    """After alignment, idx maps to the right request and out-of-range is None."""
    CosyVoice3Model, to_payload_element = _conditioning_symbols()

    speech_token, speech_feat, embedding, speech_token_len = _collated_batch(n_reqs=2, t=4, f=80, d=256)
    lists = CosyVoice3Model._split_prompt_conditioning(speech_token, speech_feat, embedding, speech_token_len)
    st_list, sf_list, emb_list, stl_list = CosyVoice3Model._align_conditioning_to_batch(*lists, batch_size=3)

    # Conditioning-carrying requests keep their own voice.
    for idx, want in ((0, 1.0), (1, 2.0)):
        out = to_payload_element(element=emb_list, idx=idx, start=0, end=1)
        assert isinstance(out, torch.Tensor)
        assert bool((out[0] == want).all())
        out_st = to_payload_element(element=st_list, idx=idx, start=0, end=1)
        assert isinstance(out_st, torch.Tensor)
        assert bool((out_st[0] == want).all())
        out_sf = to_payload_element(element=sf_list, idx=idx, start=0, end=1)
        assert isinstance(out_sf, torch.Tensor)
        assert out_sf.shape == (1, 8, 80)
        out_stl = to_payload_element(element=stl_list, idx=idx, start=0, end=1)
        assert isinstance(out_stl, torch.Tensor)

    # The extra scheduled request has no conditioning: skip (None), not aliasing.
    assert to_payload_element(element=emb_list, idx=2, start=0, end=1) is None
    assert to_payload_element(element=st_list, idx=2, start=0, end=1) is None
    assert to_payload_element(element=sf_list, idx=2, start=0, end=1) is None
    assert to_payload_element(element=stl_list, idx=2, start=0, end=1) is None


def test_align_is_noop_when_already_batch_aligned():
    CosyVoice3Model, _ = _conditioning_symbols()

    speech_token, speech_feat, embedding, speech_token_len = _collated_batch(n_reqs=3, t=4, f=80, d=256)
    lists = CosyVoice3Model._split_prompt_conditioning(speech_token, speech_feat, embedding, speech_token_len)
    aligned = CosyVoice3Model._align_conditioning_to_batch(*lists, batch_size=3)
    assert all(len(lst) == 3 for lst in aligned)
    assert aligned[2][2] is not None  # third request has its own conditioning


def test_align_is_robust_to_per_request_list_input():
    """_split_prompt_conditioning also accepts per-request lists; alignment
    must still pad to the full batch when those lists are shorter."""
    CosyVoice3Model, _ = _conditioning_symbols()

    st_list = [torch.full((1, 4), 1.0), torch.full((1, 4), 2.0)]
    sf_list = [torch.full((1, 8, 80), 1.0), torch.full((1, 8, 80), 2.0)]
    emb_list = [torch.full((1, 256), 1.0), torch.full((1, 256), 2.0)]
    stl_list = [torch.full((1, 1), 4), torch.full((1, 1), 4)]

    aligned = CosyVoice3Model._align_conditioning_to_batch(st_list, sf_list, emb_list, stl_list, batch_size=3)
    assert all(len(lst) == 3 for lst in aligned)
    assert aligned[0][2] is None
