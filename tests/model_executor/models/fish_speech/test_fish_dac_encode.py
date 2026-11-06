# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.fish_speech.dac_modules.codec import DAC
from vllm_omni.model_executor.models.fish_speech.dac_modules.rvq import (
    DownsampleResidualVectorQuantize,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def quantizer():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = DownsampleResidualVectorQuantize(
            input_dim=8,
            n_codebooks=3,
            codebook_dim=4,
            codebook_size=16,
            semantic_codebook_size=32,
            downsample_factor=(2, 2),
            pre_module=nn.Conv1d(8, 8, 1),
            post_module=nn.Conv1d(8, 8, 1),
        )
    return model.eval()


@pytest.fixture
def codec(quantizer):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        model = DAC(
            encoder_dim=4,
            encoder_rates=[2],
            latent_dim=8,
            decoder_dim=8,
            decoder_rates=[2],
            quantizer=quantizer,
            encoder_transformer_layers=[0],
            decoder_transformer_layers=[0],
        )
    return model.eval()


@pytest.mark.parametrize("batch_size,length", [(1, 8), (2, 9), (2, 17)])
@pytest.mark.parametrize("n_quantizers", [None, 1, 3])
@torch.inference_mode()
def test_quantizer_encode_matches_forward_codes(quantizer, batch_size, length, n_quantizers):
    z = torch.randn(batch_size, 8, length, generator=torch.Generator().manual_seed(2))

    expected = quantizer(z, n_quantizers=n_quantizers)
    codes = quantizer.encode(z, n_quantizers=n_quantizers)

    assert torch.equal(codes, expected.codes)
    residual_codebooks = 3 if n_quantizers is None else n_quantizers
    assert codes.shape == (batch_size, residual_codebooks + 1, (length + 3) // 4)
    assert codes.dtype == torch.long
    assert expected.z.shape == z.shape


@torch.inference_mode()
def test_quantizer_encode_skips_reconstruction(quantizer, mocker):
    z = torch.randn(2, 8, 8, generator=torch.Generator().manual_seed(3))
    forward = mocker.spy(quantizer, "forward")
    post_module = mocker.spy(quantizer.post_module, "forward")
    upsample = mocker.spy(quantizer.upsample, "forward")

    codes = quantizer.encode(z)

    forward.assert_not_called()
    post_module.assert_not_called()
    upsample.assert_not_called()

    result = quantizer(z)
    post_module.assert_called_once()
    upsample.assert_called_once()
    assert torch.equal(codes, result.codes)

    decoded = quantizer.decode(codes)
    assert post_module.call_count == 2
    assert upsample.call_count == 2
    torch.testing.assert_close(decoded, result.z, rtol=1e-4, atol=1e-6)


def test_quantizer_forward_preserves_outputs_and_gradients(quantizer):
    z = torch.randn(2, 8, 9, generator=torch.Generator().manual_seed(5), requires_grad=True)

    result = quantizer(z, n_quantizers=1)

    assert result.z.shape == z.shape
    assert result.codes.shape == (2, 2, 3)
    assert result.codes.dtype == torch.long
    assert result.latents.shape == (2, 8, 3)
    assert result.semantic_distill_z is None
    for loss in (result.commitment_loss, result.codebook_loss):
        assert loss.ndim == 0
        assert loss.requires_grad
        assert torch.isfinite(loss)

    loss = result.z.square().mean() + result.latents.square().mean() + result.commitment_loss + result.codebook_loss
    loss.backward()

    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert torch.count_nonzero(z.grad) > 0
    for module in (
        quantizer.downsample,
        quantizer.pre_module,
        quantizer.semantic_quantizer,
        quantizer.quantizer.quantizers[0],
        quantizer.post_module,
        quantizer.upsample,
    ):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)
    for module in quantizer.quantizer.quantizers[1:]:
        assert all(parameter.grad is None for parameter in module.parameters())


@pytest.mark.parametrize(
    "audio_shape,audio_lengths,expected_lengths",
    [
        ((1, 16), None, [2]),
        ((2, 17), None, [3]),
        ((2, 1, 17), [17, 8], [3, 1]),
        ((2, 1, 16), [9, 16], [2, 2]),
    ],
)
@pytest.mark.parametrize("n_quantizers", [None, 1, 3])
@torch.inference_mode()
def test_dac_encode_preserves_padding_and_lengths(
    codec, mocker, audio_shape, audio_lengths, expected_lengths, n_quantizers
):
    audio = torch.randn(audio_shape, generator=torch.Generator().manual_seed(4))
    batch_size, length = audio.shape[0], audio.shape[-1]
    padded_length = ((length + codec.frame_length - 1) // codec.frame_length) * codec.frame_length
    padded_audio = torch.zeros(batch_size, 1, padded_length)
    padded_audio[..., :length] = audio.reshape(batch_size, 1, length)
    lengths = None if audio_lengths is None else torch.tensor(audio_lengths)
    semantic_len = torch.tensor(expected_lengths)

    # The existing full quantizer path is the reference for the codes.
    expected_codes = codec.quantizer(codec.encoder(padded_audio), n_quantizers=n_quantizers).codes
    encoder = mocker.spy(codec.encoder, "forward")
    encode = mocker.spy(codec.quantizer, "encode")
    forward = mocker.spy(codec.quantizer, "forward")

    codes, code_lengths = codec.encode(audio, lengths, n_quantizers=n_quantizers, semantic_len=semantic_len)

    encoder.assert_called_once()
    assert torch.equal(encoder.call_args.args[0], padded_audio)
    encode.assert_called_once()
    assert encode.call_args.kwargs["semantic_len"] is semantic_len
    forward.assert_not_called()
    assert torch.equal(codes, expected_codes)
    assert codes.shape[-1] == padded_length // codec.frame_length
    assert torch.equal(code_lengths, torch.tensor(expected_lengths, dtype=torch.long))
    assert code_lengths.device == audio.device
