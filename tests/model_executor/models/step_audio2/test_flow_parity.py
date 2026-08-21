# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.models.step_audio2.cosyvoice2.cfm import CausalConditionalCFM
from vllm_omni.model_executor.models.step_audio2.cosyvoice2.dit import DiT
from vllm_omni.model_executor.models.step_audio2.cosyvoice2.flow import CausalMaskedDiffWithXvec
from vllm_omni.model_executor.models.step_audio2.cosyvoice2.upsample_encoder import UpsampleConformerEncoderV2
from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import _load_flow_state_dict_strict

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

external_dit = pytest.importorskip("cosyvoice2.flow.decoder_dit")
external_cfm = pytest.importorskip("cosyvoice2.flow.flow_matching")
external_flow = pytest.importorskip("cosyvoice2.flow.flow")
external_encoder = pytest.importorskip("cosyvoice2.transformer.upsample_encoder_v2")


@pytest.fixture(autouse=True)
def _qkv_parallel_test_environment(init_fake_tp_group, mocker):
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    mocker.patch(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        return_value=default_unquantized_gemm,
    )


def _make_models() -> tuple[torch.nn.Module, torch.nn.Module]:
    kwargs = {
        "in_channels": 320,
        "out_channels": 80,
        "mlp_ratio": 2.0,
        "depth": 2,
        "num_heads": 2,
        "head_dim": 32,
        "hidden_size": 64,
    }
    torch.manual_seed(0)
    reference = external_dit.DiT(**kwargs).eval()
    vendored = DiT(**kwargs).eval()

    # This is the same strict compatibility guarantee used when loading
    # Step-Audio2's flow.pt checkpoint at runtime.
    _load_flow_state_dict_strict(vendored, reference.state_dict())
    return reference, vendored


def _make_flow(flow_cls, cfm_cls, dit_cls, encoder_cls) -> torch.nn.Module:
    encoder = encoder_cls(
        input_size=32,
        output_size=32,
        pre_lookahead_len=1,
        num_blocks=1,
        num_up_blocks=1,
        up_stride=2,
        attention_heads=2,
        linear_units=64,
        dropout_rate=0.0,
        positional_dropout_rate=0.0,
        attention_dropout_rate=0.0,
    )
    estimator = dit_cls(
        in_channels=320,
        out_channels=80,
        mlp_ratio=2.0,
        depth=2,
        num_heads=2,
        head_dim=32,
        hidden_size=64,
    )
    decoder = cfm_cls(estimator=estimator)

    # The production buffers reserve space for 600 seconds of audio. Keep the
    # same cache path while sizing them for this two-chunk CPU parity test.
    estimator.cnn_cache_buffer = torch.zeros(2, 2, 128, 2)
    estimator.att_cache_buffer = torch.zeros(2, 2, 2, 32, 64)
    decoder.rand_noise = torch.randn(1, 80, 32)
    decoder.cnn_cache_buffer = torch.zeros(2, 2, 2, 128, 2)
    decoder.att_cache_buffer = torch.zeros(2, 2, 2, 2, 32, 64)
    return flow_cls(
        input_size=32,
        output_size=80,
        spk_embed_dim=16,
        vocab_size=64,
        encoder=encoder,
        decoder=decoder,
    ).eval()


def _make_flows() -> tuple[torch.nn.Module, torch.nn.Module]:
    torch.manual_seed(0)
    reference = _make_flow(
        external_flow.CausalMaskedDiffWithXvec,
        external_cfm.CausalConditionalCFM,
        external_dit.DiT,
        external_encoder.UpsampleConformerEncoderV2,
    )
    vendored = _make_flow(
        CausalMaskedDiffWithXvec,
        CausalConditionalCFM,
        DiT,
        UpsampleConformerEncoderV2,
    )
    _load_flow_state_dict_strict(vendored, reference.state_dict())
    # rand_noise is intentionally non-persistent and therefore is not copied
    # by load_state_dict, but it is part of Flow inference semantics.
    vendored.decoder.rand_noise.copy_(reference.decoder.rand_noise)
    return reference, vendored


def _chunk_inputs(model: torch.nn.Module, seq_len: int = 7):
    batch = 2
    depth = len(model.blocks)
    heads = model.blocks[0].attn.num_heads
    head_dim = model.blocks[0].attn.head_dim
    hidden_size = model.blocks[0].conv.in_channels

    x = torch.randn(batch, model.in_channels, seq_len)
    time = model.t_embedder(torch.rand(batch)).unsqueeze(1)
    cnn_out = torch.empty(depth, batch, hidden_size * 2, 2)
    att_out = torch.empty(depth, batch, heads, seq_len, head_dim * 2)
    return x, time, None, [None] * depth, [None] * depth, cnn_out, att_out


def test_vendored_dit_forward_matches_external() -> None:
    reference, vendored = _make_models()
    batch, seq_len = 2, 7
    inputs = {
        "x": torch.randn(batch, 80, seq_len),
        "mask": torch.ones(batch, 1, seq_len),
        "mu": torch.randn(batch, 80, seq_len),
        "t": torch.rand(batch),
        "spks": torch.randn(batch, 80),
        "cond": torch.randn(batch, 80, seq_len),
    }

    with torch.inference_mode():
        expected = reference(**inputs)
        actual = vendored(**inputs)

    torch.testing.assert_close(actual, expected)


def test_vendored_dit_uncached_chunk_matches_external() -> None:
    reference, vendored = _make_models()
    inputs = _chunk_inputs(reference)
    reference_buffers = (inputs[-2].clone(), inputs[-1].clone())
    vendored_buffers = (inputs[-2].clone(), inputs[-1].clone())

    with torch.inference_mode():
        expected = reference.blocks_forward_chunk(*inputs[:5], *reference_buffers)
        actual = vendored.blocks_forward_chunk(*inputs[:5], *vendored_buffers)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(vendored_buffers[0], reference_buffers[0])
    torch.testing.assert_close(vendored_buffers[1], reference_buffers[1])


def test_vendored_dit_cached_chunk_matches_external() -> None:
    reference, vendored = _make_models()
    first_inputs = _chunk_inputs(reference)
    reference_first_buffers = (first_inputs[-2].clone(), first_inputs[-1].clone())
    vendored_first_buffers = (first_inputs[-2].clone(), first_inputs[-1].clone())

    with torch.inference_mode():
        reference.blocks_forward_chunk(*first_inputs[:5], *reference_first_buffers)
        vendored.blocks_forward_chunk(*first_inputs[:5], *vendored_first_buffers)

    seq_len = 5
    batch = first_inputs[0].shape[0]
    depth = len(reference.blocks)
    heads = reference.blocks[0].attn.num_heads
    head_dim = reference.blocks[0].attn.head_dim
    hidden_size = reference.blocks[0].conv.in_channels
    x = torch.randn(batch, reference.in_channels, seq_len)
    time = reference.t_embedder(torch.rand(batch)).unsqueeze(1)
    reference_buffers = (
        torch.empty(depth, batch, hidden_size * 2, 2),
        torch.empty(depth, batch, heads, first_inputs[0].shape[-1] + seq_len, head_dim * 2),
    )
    vendored_buffers = tuple(buffer.clone() for buffer in reference_buffers)

    with torch.inference_mode():
        expected = reference.blocks_forward_chunk(
            x,
            time,
            None,
            reference_first_buffers[0],
            reference_first_buffers[1],
            *reference_buffers,
        )
        actual = vendored.blocks_forward_chunk(
            x,
            time,
            None,
            vendored_first_buffers[0],
            vendored_first_buffers[1],
            *vendored_buffers,
        )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(vendored_buffers[0], reference_buffers[0])
    torch.testing.assert_close(vendored_buffers[1], reference_buffers[1])


def test_vendored_streaming_flow_matches_external() -> None:
    reference, vendored = _make_flows()
    speaker = torch.randn(1, 16)
    prompt_token = torch.randint(0, 64, (1, 5))
    prompt_mel = torch.randn(1, 8, 80)
    next_token = torch.randint(0, 64, (1, 4))

    with torch.inference_mode():
        reference_cache = reference.setup_cache(prompt_token, prompt_mel, speaker, n_timesteps=2)
        vendored_cache = vendored.setup_cache(prompt_token, prompt_mel, speaker, n_timesteps=2)

        expected, reference_cache = reference.inference_chunk(next_token, speaker, reference_cache, n_timesteps=2)
        actual, vendored_cache = vendored.inference_chunk(next_token, speaker, vendored_cache, n_timesteps=2)

    torch.testing.assert_close(actual, expected)
    assert vendored_cache.keys() == reference_cache.keys()
    for name in reference_cache:
        torch.testing.assert_close(vendored_cache[name], reference_cache[name])
