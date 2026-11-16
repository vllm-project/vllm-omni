# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.indextts2.indextts2_s2mel_decoder import (
    IndexTTS2S2MelDecoder,
)
from vllm_omni.model_executor.models.indextts2.s2mel.modules.wavenet import WN

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_wavenet() -> WN:
    torch.manual_seed(0)
    return WN(
        hidden_channels=2,
        kernel_size=5,
        dilation_rate=1,
        n_layers=8,
        gin_channels=2,
        p_dropout=0.0,
        causal=False,
    ).eval()


def _sequence_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    positions = torch.arange(max_length, device=lengths.device)
    return (positions.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1)


def _singleton_outputs(
    wavenet: WN,
    inputs: torch.Tensor,
    lengths: list[int],
    conditioning: torch.Tensor,
) -> list[torch.Tensor]:
    # This is the independent oracle: every request ends at the physical tensor
    # boundary and therefore uses SConv1d's original reflect-padding behavior.
    wavenet._assume_full_mask = True
    return [
        wavenet(
            inputs[row : row + 1, :, :length],
            torch.ones(1, 1, length, dtype=torch.bool),
            g=conditioning[row : row + 1],
        )
        for row, length in enumerate(lengths)
    ]


def test_ragged_wavenet_matches_original_singleton_reflect_boundaries():
    wavenet = _make_wavenet()
    lengths = [6, 5, 4, 3]
    inputs = torch.randn(len(lengths), 2, max(lengths))
    conditioning = torch.randn(len(lengths), 2, 1)
    expected = _singleton_outputs(wavenet, inputs, lengths, conditioning)

    mask = _sequence_mask(torch.tensor(lengths), inputs.shape[-1])
    wavenet._assume_full_mask = False
    # Keep deliberately non-zero invalid tails. The first WaveNet layer sees
    # transformer output before the recurrent mask has cleared those values.
    actual = wavenet(inputs, mask, g=conditioning)

    for row, (length, singleton) in enumerate(zip(lengths, expected)):
        torch.testing.assert_close(
            actual[row : row + 1, :, :length],
            singleton,
            rtol=1e-5,
            atol=1e-6,
        )


def test_wavenet_exposes_minimum_safe_ragged_length():
    # kernel=5, dilation=1 has two right-reflect ghost positions, so a
    # logical row must contain at least three frames before it can use the
    # vectorized ragged path. Shorter rows stay on original singleton SConv1d.
    assert _make_wavenet().ragged_min_length == 3


def test_cfm_grouping_isolates_rows_too_short_for_logical_reflect():
    groups = IndexTTS2S2MelDecoder._cfm_group_indices(
        sorted_indices=[0, 1, 2, 3, 4],
        total_lengths=[0, 1, 2, 3, 6],
        max_group_size=2,
        min_ragged_length=3,
    )

    assert groups == [[0], [1], [2], [3, 4]]


def test_ragged_wavenet_preserves_cfg_cond_uncond_row_order():
    wavenet = _make_wavenet()
    base_lengths = [3, 6]
    conditional = torch.randn(2, 2, 6)
    unconditional = torch.randn(2, 2, 6)
    cfg_inputs = torch.cat([conditional, unconditional], dim=0)
    cfg_lengths = base_lengths + base_lengths
    cfg_conditioning = torch.randn(4, 2, 1)
    expected = _singleton_outputs(
        wavenet,
        cfg_inputs,
        cfg_lengths,
        cfg_conditioning,
    )

    mask = _sequence_mask(torch.tensor(cfg_lengths), cfg_inputs.shape[-1])
    wavenet._assume_full_mask = False
    actual = wavenet(cfg_inputs, mask, g=cfg_conditioning)

    for row, (length, singleton) in enumerate(zip(cfg_lengths, expected)):
        torch.testing.assert_close(
            actual[row : row + 1, :, :length],
            singleton,
            rtol=1e-5,
            atol=1e-6,
        )


def test_full_mask_wavenet_path_is_bit_exact_with_original_path():
    wavenet = _make_wavenet()
    inputs = torch.randn(3, 2, 6)
    mask = torch.ones(3, 1, 6, dtype=torch.bool)
    conditioning = torch.randn(3, 2, 1)

    if hasattr(wavenet, "_assume_full_mask"):
        del wavenet._assume_full_mask
    expected = wavenet(inputs, mask, g=conditioning)

    wavenet._assume_full_mask = True
    actual = wavenet(inputs, mask, g=conditioning)

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_ragged_wavenet_is_one_dynamic_fullgraph():
    wavenet = _make_wavenet()
    lengths = [3, 6]
    inputs = torch.randn(2, 2, 6)
    conditioning = torch.randn(2, 2, 1)
    mask = _sequence_mask(torch.tensor(lengths), inputs.shape[-1])
    expected = _singleton_outputs(wavenet, inputs, lengths, conditioning)
    wavenet._assume_full_mask = False

    compiled = torch.compile(
        wavenet,
        backend="eager",
        fullgraph=True,
        dynamic=True,
    )
    actual = compiled(inputs, mask, g=conditioning)

    for row, (length, singleton) in enumerate(zip(lengths, expected)):
        torch.testing.assert_close(
            actual[row : row + 1, :, :length],
            singleton,
            rtol=1e-5,
            atol=1e-6,
        )


def test_decoder_full_mask_switch_also_controls_wavenet_boundary_path():
    decoder = object.__new__(IndexTTS2S2MelDecoder)
    attention_layers = [SimpleNamespace(attention=SimpleNamespace()) for _ in range(2)]
    estimator = SimpleNamespace(
        transformer=SimpleNamespace(layers=attention_layers),
        wavenet=SimpleNamespace(),
    )

    decoder._set_dit_full_mask_fast_path(estimator, enabled=False)

    assert all(layer.attention._assume_full_mask is False for layer in attention_layers)
    assert estimator.wavenet._assume_full_mask is False

    decoder._set_dit_full_mask_fast_path(estimator, enabled=True)

    assert all(layer.attention._assume_full_mask is True for layer in attention_layers)
    assert estimator.wavenet._assume_full_mask is True


def test_decoder_full_mask_switch_unwraps_compiled_estimator():
    decoder = object.__new__(IndexTTS2S2MelDecoder)
    attention_layers = [SimpleNamespace(attention=SimpleNamespace()) for _ in range(2)]
    eager_estimator = SimpleNamespace(
        transformer=SimpleNamespace(layers=attention_layers),
        wavenet=SimpleNamespace(),
    )
    compiled_estimator = SimpleNamespace(_orig_mod=eager_estimator)

    decoder._set_dit_full_mask_fast_path(compiled_estimator, enabled=False)

    assert all(layer.attention._assume_full_mask is False for layer in attention_layers)
    assert eager_estimator.wavenet._assume_full_mask is False
