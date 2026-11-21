# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
import types

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
    QwenImage21Rope,
    QwenImage21SequencePrepare,
    _apply_qwen_image21_rotary_emb_native,
    _npu_rope_tables,
    _qwen_image21_rope_text_mask,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_mixed_prompt_padding_keeps_each_image_frame_position():
    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    prepare = QwenImage21SequencePrepare(nn.Identity(), rope)
    layout = [(1, 2, 2)]
    image_slots = torch.tensor([[False] * 6 + [True]] * 2)
    text_mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1] * 6])

    _, _, _, target_freqs = prepare(
        torch.zeros(2, 4, 64),
        torch.zeros(2, 6, 64),
        layout,
        image_slots,
        False,
        text_mask,
    )

    short = rope(layout, torch.tensor([False] * 4 + [True] * 4), torch.device("cpu"))[-4:]
    long = rope(layout, torch.tensor([False] * 6 + [True] * 4), torch.device("cpu"))[-4:]
    assert target_freqs.shape == (2, 4, 64)
    assert torch.equal(target_freqs[0], short)
    assert torch.equal(target_freqs[1], long)


def test_mixed_prompt_padding_with_reference_image_keeps_target_frame_position():
    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    prepare = QwenImage21SequencePrepare(nn.Identity(), rope)
    layout = [(1, 2, 2), (1, 2, 2)]
    image_slots = torch.tensor([[False] * 2 + [True] + [False] * 6 + [True]] * 2)
    text_mask = torch.tensor([[0] + [1] * 8, [1] * 9])

    _, _, _, target_freqs = prepare(
        torch.zeros(2, 8, 64),
        torch.zeros(2, 9, 64),
        layout,
        image_slots,
        False,
        text_mask,
    )

    short_mask = torch.tensor([False, True] + [False] * 6 + [True])
    short = rope(layout, prepare.expand_image_pad_mask(short_mask), torch.device("cpu"))[-4:]
    long = rope(layout, prepare.expand_image_pad_mask(image_slots[0]), torch.device("cpu"))[-4:]
    assert torch.equal(target_freqs[0], short)
    assert torch.equal(target_freqs[1], long)


def test_singleton_decode_after_mixed_prefill_keeps_target_frame_position():
    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    prepare = QwenImage21SequencePrepare(nn.Identity(), rope)
    layout = [(1, 2, 2)]
    image_slots = torch.tensor([[False] * 6 + [True]] * 2)
    text_mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1] * 6])

    _, _, _, prefill_target = prepare(
        torch.zeros(2, 4, 64), torch.zeros(2, 6, 64), layout, image_slots, False, text_mask
    )
    short_request_mask = text_mask[:1]
    short_mask = _qwen_image21_rope_text_mask(short_request_mask, torch.device("cuda"), sp_active=False)
    assert short_mask is short_request_mask
    prepare._request_geometry = {}
    prefix, _, _, decode_target = prepare(torch.zeros(1, 4, 64), None, layout, image_slots[:1], True, short_mask)

    assert prefix.shape[1] == 0
    assert decode_target.shape == (1, 4, 64)
    assert torch.equal(decode_target[0], prefill_target[0])
    assert torch.equal(prepare._request_geometry["rotary_emb"][0, -4:], prefill_target[0])


@pytest.mark.parametrize("device,sp_active", [("cpu", False), ("cuda", True), ("npu", True)])
def test_unsupported_mixed_padding_is_rejected(device, sp_active):
    text_mask = torch.tensor([[0, 1, 1]])
    with pytest.raises(ValueError, match="mixed-length text padding requires CUDA or NPU without sequence parallelism"):
        _qwen_image21_rope_text_mask(text_mask, torch.device(device), sp_active)


@pytest.mark.parametrize("device", ["cuda", "npu"])
def test_per_request_mask_is_accepted_on_cuda_and_npu(device):
    text_mask = torch.tensor([[0, 1, 1]])
    assert _qwen_image21_rope_text_mask(text_mask, torch.device(device), False) is text_mask


def test_batched_rotary_matches_individual_requests():
    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    image_pad_mask = torch.tensor([False] * 6 + [True] * 4)
    text_valid = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1], [1] * 10])
    freqs = rope([(1, 2, 2)], image_pad_mask, torch.device("cpu"), text_valid)
    x = torch.randn(2, 10, 1, 128, dtype=torch.float32)

    actual = _apply_qwen_image21_rotary_emb_native(x, freqs)
    expected = torch.cat([_apply_qwen_image21_rotary_emb_native(x[i : i + 1], freqs[i]) for i in range(2)])
    assert torch.equal(actual, expected)


def test_npu_rope_tables_keep_the_batch_axis_of_a_per_request_table():
    """The fused NPU kernel takes a (B', S, 1, W) table.

    A per-request table carries one row of angles per batch element, so its
    batch axis must survive the normalization: flattening it into the sequence
    axis would give every request the angles of another one.
    """
    freqs = torch.polar(torch.ones(3, 5, 64), torch.randn(3, 5, 64))
    cos, sin = _npu_rope_tables(freqs)

    assert cos.shape == (3, 5, 1, 128)
    assert sin.shape == (3, 5, 1, 128)
    # Each batch row keeps its own angles.
    for batch in range(3):
        torch.testing.assert_close(
            cos[batch, :, 0], torch.real(freqs[batch]).float().repeat_interleave(2, dim=-1), atol=0, rtol=0
        )
        torch.testing.assert_close(
            sin[batch, :, 0], torch.imag(freqs[batch]).float().repeat_interleave(2, dim=-1), atol=0, rtol=0
        )


def test_npu_rope_tables_lift_a_shared_table_to_the_batch_axis():
    """A shared (S, W) table broadcasts against every batch row."""
    freqs = torch.polar(torch.ones(5, 64), torch.randn(5, 64))
    cos, sin = _npu_rope_tables(freqs)

    assert cos.shape == (1, 5, 1, 128)
    assert sin.shape == (1, 5, 1, 128)
    torch.testing.assert_close(cos[0, :, 0], torch.real(freqs).float().repeat_interleave(2, dim=-1), atol=0, rtol=0)


def test_npu_rope_tables_reject_an_unsupported_table_rank():
    with pytest.raises(ValueError, match="RoPE table must be 2-D or 3-D"):
        _npu_rope_tables(torch.zeros(2, 3, 4, 64))


def test_npu_interleaved_rotation_matches_the_reference_per_request_rotation(monkeypatch):
    """The NPU fused rotation must reproduce the per-request reference.

    Regression guard for the NPU nightly failure where a mixed-length batch
    was rejected before it reached the model: the NPU path has to rotate every
    batch row with its own angles, exactly like the CUDA reference.
    """
    calls: list[tuple[torch.Tensor, ...]] = []

    def npu_rotary_mul(x, cos, sin, rotary_mode=None):
        calls.append((x, cos, sin))
        assert rotary_mode == "interleave"
        # Mirror the kernel's own input guard.
        assert x.dim() == 4, f"npu_rotary_mul requires 4D x, got {x.dim()}"
        # The interleaved kernel takes a table whose batch axis is 1.
        assert cos.shape[0] == 1, f"the interleaved kernel needs a shared table, got {tuple(cos.shape)}"
        # In interleave mode the table carries the full rotary width and the
        # kernel rotates the GPT-J pairs (x[..., 0::2], x[..., 1::2]).
        x1, x2 = x.float()[..., 0::2], x.float()[..., 1::2]
        cos1 = cos.float()[..., 0::2]
        sin1 = sin.float()[..., 0::2]
        out = torch.stack((x1 * cos1 - x2 * sin1, x1 * sin1 + x2 * cos1), dim=-1)
        return out.flatten(-2).to(x.dtype)

    monkeypatch.setitem(sys.modules, "torch_npu", types.SimpleNamespace(npu_rotary_mul=npu_rotary_mul))

    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
        _apply_qwen_image21_rotary_emb_npu,
    )

    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    image_pad_mask = torch.tensor([False] * 6 + [True] * 4)
    text_valid = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1], [1] * 10])
    freqs = rope([(1, 2, 2)], image_pad_mask, torch.device("cpu"), text_valid)
    x = torch.randn(2, 10, 3, 128, dtype=torch.float32)

    actual = _apply_qwen_image21_rotary_emb_npu(x, freqs)

    # The interleaved kernel cannot take a per-batch table, so every request
    # is rotated with its own angles.
    assert len(calls) == 2
    assert [tuple(call[1].shape) for call in calls] == [(1, 10, 1, 128), (1, 10, 1, 128)]
    expected = torch.cat([_apply_qwen_image21_rotary_emb_native(x[i : i + 1], freqs[i]) for i in range(2)])
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_npu_interleaved_rotation_uses_one_call_for_a_shared_table(monkeypatch):
    """A batch without padding keeps the single-call fused path."""
    calls: list[torch.Tensor] = []

    def npu_rotary_mul(x, cos, sin, rotary_mode=None):
        calls.append(x)
        return x

    monkeypatch.setitem(sys.modules, "torch_npu", types.SimpleNamespace(npu_rotary_mul=npu_rotary_mul))

    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
        _apply_qwen_image21_rotary_emb_npu,
    )

    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    image_pad_mask = torch.tensor([False] * 6 + [True] * 4)
    freqs = rope([(1, 2, 2)], image_pad_mask, torch.device("cpu"))
    x = torch.randn(2, 10, 3, 128, dtype=torch.float32)

    _apply_qwen_image21_rotary_emb_npu(x, freqs)

    assert len(calls) == 1


def test_npu_interleaved_rotation_rejects_a_mismatched_table(monkeypatch):
    """A per-request table that does not cover the batch is a bug, not a crash."""
    monkeypatch.setitem(sys.modules, "torch_npu", types.SimpleNamespace(npu_rotary_mul=lambda *a, **k: a[0]))

    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
        _apply_qwen_image21_rotary_emb_npu,
    )

    freqs = torch.polar(torch.ones(3, 10, 64), torch.randn(3, 10, 64))
    x = torch.randn(2, 10, 3, 128, dtype=torch.float32)

    with pytest.raises(ValueError, match="request rows"):
        _apply_qwen_image21_rotary_emb_npu(x, freqs)
