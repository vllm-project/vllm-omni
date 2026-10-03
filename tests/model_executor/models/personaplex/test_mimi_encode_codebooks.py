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
    # Small real front end; keep all 32 codebooks with nonzero centroids.
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


def test_encode_frame_computes_only_consumed_codebooks(mocker: MockerFixture) -> None:
    batch_size = 2
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
