# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression coverage for MiniCPM-o 4.5 SigLIP FlashAttention."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from pytest_mock import MockerFixture
from transformers.utils import is_flash_attn_2_available

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni_llm as minicpmo_model
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    SiglipAttention,
    SiglipEncoder,
    SiglipFlashAttention2,
    SiglipVisionConfig,
)

_VISION_NUM_HEADS = 16
_VISION_HEAD_DIM = 72
_VISION_HIDDEN_SIZE = _VISION_NUM_HEADS * _VISION_HEAD_DIM
_DTYPE = torch.bfloat16


def _attention_config(*, hidden_size: int, num_heads: int) -> SiglipVisionConfig:
    return SiglipVisionConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        attention_dropout=0.0,
    )


def _encoder_config(*, num_layers: int = 3) -> SiglipVisionConfig:
    config = SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        attention_dropout=0.0,
    )
    config._attn_implementation = "flash_attention_2"
    return config


def _capture_output(
    outputs: dict[str, torch.Tensor], name: str
) -> Callable[[torch.nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
    def hook(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        outputs[name] = output

    return hook


@pytest.mark.core_model
@pytest.mark.cpu
def test_qkv_bshd_layout_reuses_projection_storage(mocker: MockerFixture) -> None:
    batch_size = 2
    seq_len = 7
    hidden_size = 32
    num_heads = 4
    head_dim = hidden_size // num_heads
    attention = SiglipFlashAttention2(_attention_config(hidden_size=hidden_size, num_heads=num_heads)).eval()

    projected: dict[str, torch.Tensor] = {}
    attention.q_proj.register_forward_hook(_capture_output(projected, "query"))
    attention.k_proj.register_forward_hook(_capture_output(projected, "key"))
    attention.v_proj.register_forward_hook(_capture_output(projected, "value"))

    flash_inputs: dict[str, torch.Tensor] = {}

    def fake_flash_attention(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        _attention_mask: torch.Tensor | None,
        _query_length: int,
        *,
        dropout: float = 0.0,
        softmax_scale: float | None = None,
        unpadding_metadata: minicpmo_model.FlashAttentionUnpaddingMetadata | None = None,
    ) -> torch.Tensor:
        del dropout, softmax_scale, unpadding_metadata
        assert _attention_mask is None
        assert _query_length == seq_len
        flash_inputs.update(
            query=query_states,
            key=key_states,
            value=value_states,
        )
        return query_states

    flash_forward = mocker.patch.object(
        attention,
        "_flash_attention_forward",
        side_effect=fake_flash_attention,
    )

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    with torch.inference_mode():
        output, _ = attention(hidden_states)

    assert output.shape == hidden_states.shape
    flash_forward.assert_called_once()

    expected_shape = (batch_size, seq_len, num_heads, head_dim)
    expected_stride = (
        seq_len * num_heads * head_dim,
        num_heads * head_dim,
        head_dim,
        1,
    )
    for name in ("query", "key", "value"):
        projected_tensor = projected[name]
        flash_tensor = flash_inputs[name]
        assert flash_tensor.shape == expected_shape
        assert flash_tensor.stride() == expected_stride
        assert flash_tensor.is_contiguous()
        assert flash_tensor.storage_offset() == projected_tensor.storage_offset()
        assert flash_tensor.untyped_storage().data_ptr() == projected_tensor.untyped_storage().data_ptr()


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize(
    ("attention_mask", "expected_calls"),
    [
        pytest.param(torch.tensor([[True, True, False], [True, False, False]]), 1, id="padded"),
        pytest.param(None, 0, id="dense"),
    ],
)
def test_encoder_computes_unpadding_metadata_once_per_forward(
    mocker: MockerFixture,
    attention_mask: torch.Tensor | None,
    expected_calls: int,
) -> None:
    encoder = SiglipEncoder(_encoder_config()).eval()
    metadata_spy = mocker.spy(minicpmo_model, "_get_unpad_data")
    received_metadata: list[minicpmo_model.FlashAttentionUnpaddingMetadata | None] = []

    def fake_flash_attention(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        _attention_mask: torch.Tensor | None,
        _query_length: int,
        *,
        dropout: float = 0.0,
        softmax_scale: float | None = None,
        unpadding_metadata: minicpmo_model.FlashAttentionUnpaddingMetadata | None = None,
    ) -> torch.Tensor:
        del key_states, value_states, _attention_mask, _query_length, dropout, softmax_scale
        received_metadata.append(unpadding_metadata)
        return query_states

    for layer in encoder.layers:
        mocker.patch.object(layer.self_attn, "_flash_attention_forward", side_effect=fake_flash_attention)

    hidden_states = torch.randn(2, 3, 32)
    with torch.inference_mode():
        encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )

    assert metadata_spy.call_count == expected_calls
    assert len(received_metadata) == len(encoder.layers)
    if attention_mask is None:
        assert all(metadata is None for metadata in received_metadata)
    else:
        assert received_metadata[0] is not None
        assert all(metadata is received_metadata[0] for metadata in received_metadata)


@pytest.mark.core_model
@pytest.mark.cpu
def test_encoder_recomputes_unpadding_metadata_for_new_mask(mocker: MockerFixture) -> None:
    encoder = SiglipEncoder(_encoder_config(num_layers=1)).eval()
    metadata_spy = mocker.spy(minicpmo_model, "_get_unpad_data")
    received_metadata: list[minicpmo_model.FlashAttentionUnpaddingMetadata | None] = []

    def fake_flash_attention(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        _attention_mask: torch.Tensor | None,
        _query_length: int,
        *,
        dropout: float = 0.0,
        softmax_scale: float | None = None,
        unpadding_metadata: minicpmo_model.FlashAttentionUnpaddingMetadata | None = None,
    ) -> torch.Tensor:
        del key_states, value_states, _attention_mask, _query_length, dropout, softmax_scale
        received_metadata.append(unpadding_metadata)
        return query_states

    mocker.patch.object(
        encoder.layers[0].self_attn,
        "_flash_attention_forward",
        side_effect=fake_flash_attention,
    )
    hidden_states = torch.randn(2, 4, 32)
    masks = (
        torch.tensor([[True, True, True, False], [True, True, False, False]]),
        torch.tensor([[True, True, False, False], [True, False, False, False]]),
    )

    with torch.inference_mode():
        for attention_mask in masks:
            encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )

    assert metadata_spy.call_count == len(masks)
    first_metadata, second_metadata = received_metadata
    assert first_metadata is not None
    assert second_metadata is not None
    assert first_metadata is not second_metadata
    assert not torch.equal(first_metadata[0], second_metadata[0])


def _require_flash_attention() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FlashAttention execution")
    if not is_flash_attn_2_available():
        pytest.skip("Transformers reports that FlashAttention 2 is unavailable")


def _make_cuda_attention_pair() -> tuple[SiglipAttention, SiglipFlashAttention2]:
    config = _attention_config(
        hidden_size=_VISION_HIDDEN_SIZE,
        num_heads=_VISION_NUM_HEADS,
    )
    torch.manual_seed(0)
    eager_attention = SiglipAttention(config).eval().to(device="cuda", dtype=_DTYPE)
    flash_attention = SiglipFlashAttention2(config).eval().to(device="cuda", dtype=_DTYPE)
    flash_attention.load_state_dict(eager_attention.state_dict())
    return eager_attention, flash_attention


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"})
def test_encoder_metadata_reuse_matches_per_layer_recomputation() -> None:
    _require_flash_attention()
    encoder = SiglipEncoder(_encoder_config()).eval().to(device="cuda", dtype=_DTYPE)
    torch.manual_seed(3)
    hidden_states = torch.randn(2, 64, 32, device="cuda", dtype=_DTYPE)
    attention_mask = torch.arange(64, device="cuda").unsqueeze(0) < torch.tensor(
        (64, 43),
        device="cuda",
    ).unsqueeze(1)

    with torch.inference_mode():
        expected = hidden_states
        for encoder_layer in encoder.layers:
            expected = encoder_layer(
                expected,
                attention_mask,
                output_attentions=False,
                unpadding_metadata=None,
            )[0]
        actual = encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )[0]

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"})
def test_flash_attention_unpadded_matches_eager_reference() -> None:
    _require_flash_attention()
    eager_attention, flash_attention = _make_cuda_attention_pair()
    torch.manual_seed(1)
    hidden_states = torch.randn(
        2,
        64,
        _VISION_HIDDEN_SIZE,
        device="cuda",
        dtype=_DTYPE,
    )

    with torch.inference_mode():
        expected, _ = eager_attention(hidden_states)
        actual, _ = flash_attention(hidden_states)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize(
    "valid_lengths",
    [
        pytest.param((64, 43), id="moderate_padding"),
        pytest.param((64, 17), id="high_padding"),
    ],
)
def test_flash_attention_padded_matches_per_sample_unpadded(
    valid_lengths: tuple[int, int],
) -> None:
    _require_flash_attention()
    _, flash_attention = _make_cuda_attention_pair()
    batch_size = len(valid_lengths)
    seq_len = max(valid_lengths)
    torch.manual_seed(2)
    hidden_states = torch.randn(
        batch_size,
        seq_len,
        _VISION_HIDDEN_SIZE,
        device="cuda",
        dtype=_DTYPE,
    )
    attention_mask = torch.arange(seq_len, device="cuda").unsqueeze(0) < torch.tensor(
        valid_lengths,
        device="cuda",
    ).unsqueeze(1)

    with torch.inference_mode():
        padded_output, _ = flash_attention(
            hidden_states,
            attention_mask=attention_mask,
        )
        unpadded_outputs = [
            flash_attention(hidden_states[index : index + 1, :valid_length])[0]
            for index, valid_length in enumerate(valid_lengths)
        ]

    for index, valid_length in enumerate(valid_lengths):
        torch.testing.assert_close(
            padded_output[index : index + 1, :valid_length],
            unpadded_outputs[index],
            atol=2e-2,
            rtol=2e-2,
        )
