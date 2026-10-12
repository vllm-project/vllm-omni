# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU shape tests for the vendored S3Gen inference modules."""

import pytest
import torch

from vllm_omni.model_executor.models.chatterbox.s3gen_core.configs import CFM_PARAMS
from vllm_omni.model_executor.models.chatterbox.s3gen_core.decoder import ConditionalDecoder
from vllm_omni.model_executor.models.chatterbox.s3gen_core.flow import CausalMaskedDiffWithXvec
from vllm_omni.model_executor.models.chatterbox.s3gen_core.flow_matching import CausalConditionalCFM
from vllm_omni.model_executor.models.chatterbox.s3gen_core.mask import (
    add_optional_chunk_mask,
    make_pad_mask,
    subsequent_chunk_mask,
)
from vllm_omni.model_executor.models.chatterbox.s3gen_core.upsample_encoder import UpsampleConformerEncoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(("chunk_size", "left_chunks"), [(0, -1), (3, -1), (3, 0)])
def test_chunk_mask_repairs_empty_rows_without_scalar_reads(chunk_size, left_chunks):
    tokens = torch.zeros(3, 7, 4)
    padding = ~make_pad_mask(torch.tensor([7, 3, 0]), 7).unsqueeze(1)
    expected = padding
    if chunk_size:
        expected = padding & subsequent_chunk_mask(7, chunk_size, left_chunks)
    expected = expected.clone()
    expected[expected.sum(dim=-1) == 0] = True
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = add_optional_chunk_mask(tokens, padding, False, False, 0, chunk_size, left_chunks)
    assert torch.equal(actual, expected)
    assert "aten::item" not in {event.key for event in profile.key_averages()}


def tiny_flow() -> CausalMaskedDiffWithXvec:
    # The encoder hardcodes 512 channels in its pre-lookahead and upsample layers, so it cannot be narrower.
    encoder = UpsampleConformerEncoder(
        output_size=512,
        attention_heads=2,
        linear_units=64,
        num_blocks=1,
        dropout_rate=0.0,
        positional_dropout_rate=0.0,
        attention_dropout_rate=0.0,
        normalize_before=True,
        input_layer="linear",
        pos_enc_layer_type="rel_pos_espnet",
        selfattention_layer_type="rel_selfattn",
        input_size=512,
        use_cnn_module=False,
        macaron_style=False,
    )
    estimator = ConditionalDecoder(
        in_channels=320,
        out_channels=80,
        causal=True,
        channels=[32],
        dropout=0.0,
        attention_head_dim=16,
        n_blocks=1,
        num_mid_blocks=1,
        num_heads=2,
        act_fn="gelu",
        meanflow=True,
    )
    decoder = CausalConditionalCFM(spk_emb_dim=80, cfm_params=CFM_PARAMS, estimator=estimator)
    return CausalMaskedDiffWithXvec(input_size=512, spk_embed_dim=192, encoder=encoder, decoder=decoder).eval()


def test_flow_meanflow_two_steps_shape_001() -> None:
    flow = tiny_flow()
    prompt_len, gen_len = 4, 10
    kwargs = dict(
        token=torch.randint(0, 6561, (1, gen_len)),
        token_len=torch.tensor([gen_len]),
        prompt_token=torch.randint(0, 6561, (1, prompt_len)),
        prompt_token_len=torch.tensor([prompt_len]),
        prompt_feat=torch.randn(1, 2 * prompt_len, 80),
        prompt_feat_len=None,
        embedding=torch.randn(1, 192),
        n_timesteps=2,
        meanflow=True,
    )
    final, _ = flow.inference(finalize=True, **kwargs)
    assert final.shape == (1, 80, 2 * gen_len)
