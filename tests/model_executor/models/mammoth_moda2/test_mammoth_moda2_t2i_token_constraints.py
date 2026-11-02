# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for MammothModa2 t2i AR-grid logit constraints.

CUDA-graph FULL replay skips Python ``forward``, so generated_len must be
refreshed via ``update_decode_step_metadata`` / ``compute_logits`` kwargs
rather than a cache written only inside ``forward`` (#7318).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.mammoth_moda2 import mammoth_moda2 as m

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_AR_WIDTH = 4
_EOL = 6
_VISUAL_START = 8
_VISUAL_END = 12
_BASE_VOCAB = 8


def _t2i_info(generated_len: int) -> dict:
    return {
        "generated_len": generated_len,
        "omni_task": ["t2i"],
        "ar_width": [_AR_WIDTH],
        "eol_token_id": [_EOL],
        "visual_token_start_id": [_VISUAL_START],
        "visual_token_end_id": [_VISUAL_END],
    }


def _host() -> m._MammothModa2T2ITokenConstraints:
    host = m._MammothModa2T2ITokenConstraints()
    host.language_model = SimpleNamespace(base_vocab_size=_BASE_VOCAB)
    host._last_runtime_additional_information = None
    return host


def test_update_decode_step_metadata_refreshes_cache_without_forward() -> None:
    host = _host()
    host.update_decode_step_metadata(runtime_additional_information=[_t2i_info(3)])
    assert host._last_runtime_additional_information[0]["generated_len"] == 3


def test_intra_row_allows_only_visual_tokens() -> None:
    host = _host()
    host.update_decode_step_metadata(runtime_additional_information=[_t2i_info(0)])
    logits = torch.zeros(1, 16)
    logits[0] = torch.arange(16, dtype=torch.float32)
    out = host._apply_t2i_token_constraints(logits.clone())

    allowed = out[0, _VISUAL_START : _VISUAL_END + 1]
    assert torch.isfinite(allowed).all()
    assert torch.isneginf(out[0, :_VISUAL_START]).all()
    assert torch.isneginf(out[0, _VISUAL_END + 1 :]).all()
    assert torch.isneginf(out[0, _EOL])


def test_row_boundary_forces_eol() -> None:
    host = _host()
    # column_id = generated_len % (ar_width + 1) == ar_width
    host.update_decode_step_metadata(runtime_additional_information=[_t2i_info(_AR_WIDTH)])
    logits = torch.ones(1, 16)
    out = host._apply_t2i_token_constraints(logits.clone())

    assert out[0, _EOL] == 1.0
    mask = torch.ones(16, dtype=torch.bool)
    mask[_EOL] = False
    assert torch.isneginf(out[0][mask]).all()


def test_compute_logits_kwargs_refresh_generated_len_across_steps() -> None:
    host = _host()
    logits = torch.ones(1, 16)

    host.update_decode_step_metadata(runtime_additional_information=[_t2i_info(0)])
    intra = host._apply_t2i_token_constraints(logits.clone())
    assert torch.isfinite(intra[0, _VISUAL_START])
    assert torch.isneginf(intra[0, _EOL])

    host.update_decode_step_metadata(runtime_additional_information=[_t2i_info(_AR_WIDTH)])
    boundary = host._apply_t2i_token_constraints(logits.clone())
    assert boundary[0, _EOL] == 1.0
    assert torch.isneginf(boundary[0, _VISUAL_START])


def test_t2i_missing_generated_len_leaves_row_unconstrained() -> None:
    host = _host()
    info = _t2i_info(0)
    del info["generated_len"]
    host.update_decode_step_metadata(runtime_additional_information=[info])
    logits = torch.arange(16, dtype=torch.float32).unsqueeze(0)
    torch.testing.assert_close(host._apply_t2i_token_constraints(logits.clone()), logits)


def test_missing_runtime_info_leaves_logits_unchanged() -> None:
    host = _host()
    logits = torch.arange(16, dtype=torch.float32).unsqueeze(0)
    torch.testing.assert_close(host._apply_t2i_token_constraints(logits.clone()), logits)


def test_non_t2i_masks_gen_vocab() -> None:
    host = _host()
    host.update_decode_step_metadata(runtime_additional_information=[{"omni_task": ["chat"]}])
    logits = torch.ones(1, 16)
    out = host._apply_t2i_token_constraints(logits.clone())
    assert torch.isfinite(out[0, :_BASE_VOCAB]).all()
    assert torch.isneginf(out[0, _BASE_VOCAB:]).all()


def test_top_level_wrapper_delegates_decode_metadata() -> None:
    inner = _host()
    wrapper = m.MammothModa2ForConditionalGeneration.__new__(m.MammothModa2ForConditionalGeneration)
    wrapper.model = inner
    wrapper.update_decode_step_metadata(runtime_additional_information=[_t2i_info(7)])
    assert inner._last_runtime_additional_information[0]["generated_len"] == 7
