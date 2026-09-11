# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""``build_hidream_o1_attention_mask``: text rows causal, image/timestep rows fully attended.

Before the capability-flag fix, ``supports_piecewise_spans`` was unconditionally ``True``, so
on XPU and NPU this function was never called at all -- the model skipped the mask and ran
dense. The fix makes it load-bearing on those platforms for the first time, which is what
these tests are here to pin.

A wrong mask is silent: the run exits 0 and produces a plausible-looking image. So the
assertions cover the fully-unmasked row *count* and not only the values written, since a
partially-applied mask writes correct values into the rows it does reach while leaving the
others causal.

The CPU cases pin the semantics on every platform. The device case builds the mask on the
accelerator itself, because a mask this large is assembled by device kernels whose behavior
the CPU cases cannot stand in for.
"""

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.hidream_o1_image.pipeline_hidream_o1_image import (
    build_hidream_o1_attention_mask,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

# The production shape for a 2048x2048 generation: 128 text tokens then 4103 image tokens.
TEXT_LEN = 128
PROD_SEQ_LEN = 4231


def _token_types(seq_len: int, text_len: int, device: str = "cpu") -> torch.Tensor:
    """1 for full-attention (image/timestep) rows, 0 for causal text rows."""
    token_types = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    token_types[:, text_len:] = 1
    return token_types


def _reference_mask(token_types: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """An independent ``[S, S]`` reference, built by a different route than the code under test.

    ``torch.where`` rather than the production ``attention_mask[token_types.bool()] = 0``, so that
    the reference and the implementation cannot share a bug in the same indexing path.
    """
    seq_len = token_types.shape[1]
    causal = torch.triu(
        torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=token_types.device),
        diagonal=1,
    )
    return torch.where(token_types[0].bool().unsqueeze(-1), torch.zeros_like(causal), causal)


def _assert_mask_is_correct(mask: torch.Tensor, token_types: torch.Tensor, dtype: torch.dtype) -> None:
    """Every full row fully unmasked, every causal row masked exactly above the diagonal.

    Deliberately free of boolean row indexing (``rows[is_full]``) so that the check itself does
    not depend on the same indexing path the mask is built with.
    """
    rows = mask[0, 0]  # [S, S]
    torch.testing.assert_close(rows, _reference_mask(token_types, dtype), rtol=0, atol=0)

    # A count, not just an elementwise compare: a partially-applied mask writes correct values
    # into the rows it reaches and silently leaves the rest causal.
    fully_unmasked = int((rows == 0).all(dim=-1).sum())
    assert fully_unmasked == int(token_types[0].sum())


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(("seq_len", "text_len"), [(129, 128), (1000, 128), (PROD_SEQ_LEN, TEXT_LEN)])
def test_mask_rows_are_causal_or_full(dtype, seq_len, text_len):
    token_types = _token_types(seq_len, text_len)
    mask = build_hidream_o1_attention_mask(token_types, dtype=dtype)

    assert mask.shape == (1, 1, seq_len, seq_len)
    assert mask.dtype == dtype
    _assert_mask_is_correct(mask, token_types, dtype)


@pytest.mark.cpu
def test_all_causal_and_all_full_edge_cases():
    dtype = torch.bfloat16
    seq_len = 16

    all_causal = build_hidream_o1_attention_mask(torch.zeros(1, seq_len, dtype=torch.long), dtype=dtype)
    expected = torch.triu(torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype), diagonal=1)
    torch.testing.assert_close(all_causal[0, 0], expected, rtol=0, atol=0)

    all_full = build_hidream_o1_attention_mask(torch.ones(1, seq_len, dtype=torch.long), dtype=dtype)
    assert bool((all_full == 0).all())


@pytest.mark.cpu
@pytest.mark.parametrize("shape", [(PROD_SEQ_LEN,), (PROD_SEQ_LEN, 1)])
def test_accepts_1d_and_column_token_types(shape):
    """The pipeline passes a column vector on some paths; both must give the same mask."""
    flat = _token_types(PROD_SEQ_LEN, TEXT_LEN)[0]
    token_types = flat.reshape(shape)

    mask = build_hidream_o1_attention_mask(token_types, dtype=torch.bfloat16)

    assert mask.shape == (1, 1, PROD_SEQ_LEN, PROD_SEQ_LEN)
    torch.testing.assert_close(
        mask, build_hidream_o1_attention_mask(flat.unsqueeze(0), dtype=torch.bfloat16), rtol=0, atol=0
    )


@pytest.mark.cpu
def test_rejects_higher_rank_token_types():
    with pytest.raises(ValueError, match="one or two dimensions"):
        build_hidream_o1_attention_mask(torch.zeros(1, 1, 8, dtype=torch.long), dtype=torch.bfloat16)


@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"})
@pytest.mark.parametrize("seq_len", [1000, PROD_SEQ_LEN])
def test_mask_rows_are_correct_on_device(seq_len):
    """Builds the mask on the accelerator itself, where the boolean-index write actually runs.

    The CPU cases cannot stand in for this: ``attention_mask[token_types.bool()] = 0`` routes
    through ``torch.nonzero``, and an indexing-kernel defect there is correct on CPU and CUDA
    while miscounting rows (or asserting) only on the device -- which is exactly how an earlier
    XPU build behaved. So the invariant has to be checked where the kernel runs.
    """
    from vllm_omni.platforms import current_omni_platform

    device = current_omni_platform.device_type
    # ``hardware_test`` only tags the test for CI resource selection; it adds no skipif. With
    # no accelerator present ``device_type`` is "cpu", and this would pass without having
    # exercised the device at all -- a false pass on the one case that matters.
    if device == "cpu":
        pytest.skip("no accelerator visible; the CPU cases above already cover this")

    dtype = torch.bfloat16
    token_types = _token_types(seq_len, TEXT_LEN, device=device)

    mask = build_hidream_o1_attention_mask(token_types, dtype=dtype)

    assert mask.device.type == device
    _assert_mask_is_correct(mask, token_types, dtype)
