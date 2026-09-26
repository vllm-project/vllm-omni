# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Checkpoint-free coverage using real streaming layers and the HF Mimi RVQ."""

import pytest
import torch
from pytest_mock import MockerFixture
from torch import nn
from transformers import MimiConfig
from transformers.models.mimi.modeling_mimi import MimiSplitResidualVectorQuantizer

from vllm_omni.model_executor.models.personaplex import personaplex_mimi as mimi

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _codec(batch_size: int) -> mimi.PersonaPlexMimiCodec:
    # Use a small real front end rather than loading pretrained codec weights.
    # Keep all 32 codebooks, with nonzero centroids, to exercise prefix parity.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        codec = mimi.PersonaPlexMimiCodec.__new__(mimi.PersonaPlexMimiCodec)
        nn.Module.__init__(codec)
        codec.device = torch.device("cpu")
        codec.dtype = torch.float32
        codec.model = nn.Module()
        config = MimiConfig(
            hidden_size=8,
            vector_quantization_hidden_dimension=8,
            codebook_dim=8,
            codebook_size=16,
            num_quantizers=32,
            num_semantic_quantizers=1,
        )
        codec.model.quantizer = MimiSplitResidualVectorQuantizer(config).eval()
        for rvq in (
            codec.model.quantizer.semantic_residual_vector_quantizer,
            codec.model.quantizer.acoustic_residual_vector_quantizer,
        ):
            for layer in rvq.layers:
                layer.codebook.embed_sum.normal_()
        codec._enc_stages = [("conv", mimi._StreamConv1d(nn.Conv1d(1, 8, mimi.FRAME_SIZE + 2, stride=mimi.FRAME_SIZE)))]
        codec._dec_stages = []
        codec._downsample = mimi._StreamConv1d(nn.Conv1d(8, 8, 3), pad_mode="replicate")
        codec._upsample = mimi._StreamConvTr1d(nn.ConvTranspose1d(8, 8, 3))
        for name in ("encoder_transformer", "decoder_transformer"):
            transformer = mimi._MimiStreamingTransformer(num_layers=1, dim=8, num_heads=2, context=3)
            transformer.layers = nn.ModuleList([mimi._MimiTransformerLayer(dim=8, num_heads=2, ffn=16)])
            for parameter in transformer.parameters():
                nn.init.uniform_(parameter, -0.2, 0.2)
            setattr(codec, name, transformer)
        codec.streaming_init(batch_size)
        return codec.eval()


@pytest.mark.parametrize("batch_size", [1, 2, 9])
def test_encode_frame_computes_only_consumed_codebooks(batch_size: int, mocker: MockerFixture) -> None:
    codec = _codec(batch_size)
    quantizer = codec.model.quantizer
    layers = [
        *quantizer.semantic_residual_vector_quantizer.layers,
        *quantizer.acoustic_residual_vector_quantizer.layers,
    ]
    layer_spies = [mocker.spy(layer, "encode") for layer in layers]
    quantizer_spy = mocker.spy(quantizer, "encode")
    pcm = torch.randn(batch_size, mimi.FRAME_SIZE, generator=torch.Generator().manual_seed(23))

    actual = codec.encode_frame(pcm)
    counts = [spy.call_count for spy in layer_spies]
    latents = quantizer_spy.call_args.args[0]
    reference = quantizer.encode(latents, num_quantizers=32)[: mimi.CODEBOOKS, :, 0].transpose(0, 1).contiguous()

    assert actual.shape == (batch_size, mimi.CODEBOOKS)
    assert actual.dtype == torch.long
    assert actual.is_contiguous()
    assert torch.equal(actual, reference)
    assert actual.unique().numel() > 1  # Do not pass through all-zero codebooks.
    assert counts == [1] * mimi.CODEBOOKS + [0] * (32 - mimi.CODEBOOKS)
    assert quantizer.max_num_quantizers == 32  # Encoding must not mutate decoder capacity.


@pytest.mark.parametrize("batch_size", [1, 2, 9])
def test_encode_prefix_survives_ring_wrap_and_slot_reset(batch_size: int, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = _codec(batch_size)
    reference = _codec(batch_size)
    encode_all = reference.model.quantizer.encode

    def full_encode(embeddings: torch.Tensor, num_quantizers: int | None = None) -> torch.Tensor:
        # Execute the pre-optimization 32-codebook path on the same HF quantizer.
        return encode_all(embeddings, num_quantizers=32)

    monkeypatch.setattr(reference.model.quantizer, "encode", full_encode)
    generator = torch.Generator().manual_seed(29)
    for frame in range(9):
        if frame == 4:
            candidate.reset_slot(batch_size - 1)
            reference.reset_slot(batch_size - 1)
        elif frame == 7:
            candidate.reset_streaming()
            reference.reset_streaming()
        pcm = torch.randn(batch_size, mimi.FRAME_SIZE, generator=generator)
        assert torch.equal(candidate.encode_frame(pcm), reference.encode_frame(pcm))
        assert torch.equal(candidate.encoder_transformer._offset, reference.encoder_transformer._offset)
        for actual, expected in zip(candidate.encoder_transformer._kv, reference.encoder_transformer._kv):
            assert torch.equal(actual.end_offset, expected.end_offset)
            assert torch.equal(actual.start_offset, expected.start_offset)
        for actual, expected in zip(candidate._conv_states(), reference._conv_states()):
            actual_buffer = actual.prev if isinstance(actual, mimi._StreamConv1d) else actual.partial
            expected_buffer = expected.prev if isinstance(expected, mimi._StreamConv1d) else expected.partial
            assert torch.equal(actual_buffer, expected_buffer)
